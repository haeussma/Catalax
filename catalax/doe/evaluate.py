"""Expected posterior shrinkage of a design under the priors or a belief."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.distributions import constraints

from catalax.dataset.dataset import Dataset
from catalax.doe.information import DesignArrays, _prepare_information_function
from catalax.doe.noise import NoiseModel
from catalax.model.simconfig import SimulationConfig

if TYPE_CHECKING:
    from catalax.model.model import Model

_N_PRIOR_SAMPLES = 20_000
_N_INNER_SAMPLES = 65_536
_CHOLESKY_JITTER = 1e-6
_PROPOSAL_SCALE = 1.25
_DEFENSIVE_FRACTION = 0.05
"""Scale ``c`` and prior fraction ``alpha`` of ``_proposal_sd``'s proposal.

Both set the efficiency only, never the limit: the estimator is consistent
for any ``c > 0`` and ``alpha >= 0``. Measured on competitive product
inhibition (3 parameters, LogUniform priors spanning 40-100x, the 2- and
6-arm designs of progress-doe's ``catalax_competitive_pi_arms.py``) and on
the bi-substrate model (4 parameters, Uniform priors, a 4-arm optimum and
16 random arms), 256 draws, M = 65536, 8 replicate keys. Per cell: the
smallest ESS over draws (median over keys, worst workload), and in brackets
the largest ``|ratio - 1| / SE`` over parameters and workloads, where ratio is
the mean sd over draws against a reference at M = 4194304 (c = 1.75,
alpha = 0.15, independent keys):

====== ============= ============= =============
c      alpha = 0.05  alpha = 0.1   alpha = 0.25
====== ============= ============= =============
1.0    379 (2.1)     365 (2.1)     307 (1.6)
1.25   463 (1.9)     441 (1.8)     365 (2.2)
1.5    387 (3.0)     371 (3.8)     319 (2.9)
2.0    207 (1.6)     199 (1.6)     167 (1.4)
====== ============= ============= =============

At c = 1.25, alpha = 0.05 the median ESS was 44226, 49706, 26624 and 24708
(CPI 2 and 6 arms, MAT 4 and 16) and every ratio within 1e-4 of 1; the
mean absolute error of a single draw's sd was at most 0.36%. Prior sampling
at the same M had a median ESS of 76, 11, 230 and 361, minimum 2-11, its
mean sd 0.24-1.6% low (3.8-5.7 SE) and single draws 2.6-12% off; at
M = 4194304 it was still 0.04-0.18% low, with a minimum ESS of 170 at 6
arms.
"""

type EfficiencyFunction = Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]
"""``(y0s, constants) -> (efficiency[n_draws, n_free], valid[n_draws])``."""


@dataclass
class DesignReport:
    """How much a design is expected to shrink each parameter's uncertainty.

    All numbers are in log space, from the reweighted posterior sd of
    ``evaluate_design``. Efficiency ``e_j = 1 - sd_post_j / sd_prior_j``: 0
    means the design adds nothing to the prior, 1 means the parameter is pinned
    exactly. It can come out slightly below 0 for a design that teaches almost
    nothing: the posterior sd is an average over possible datasets, and some
    push the posterior against a bound of a Uniform prior. With a ``belief``,
    "prior" is that belief: efficiency is the shrinkage beyond what earlier
    rounds already taught, not beyond the model's priors.

    ``print(report)`` gives one line per parameter, e.g.
    ``kcat  ±21% (×/÷1.23)  prior ±38%  efficiency 0.44``.

    Attributes:
        efficiency: Mean efficiency per free parameter over valid draws;
            ``1 - relative_sd[j] / prior_relative_sd[j]`` up to rounding.
        relative_sd: Expected posterior sd of ``log(theta_j)`` after the
            experiment, mean over valid draws. 0.21 reads as about ±21% at one
            sd, or ×/÷ ``exp(0.21)`` = 1.23.
        prior_relative_sd: Sd of ``log(theta_j)`` under the priors or the
            belief.
        maximin: Mean over valid draws of the worst parameter's efficiency,
            ``mean_i min_j e_ij``.
        n_valid: Number of draws whose simulations all solved.
        ess_min: Smallest effective sample size of the importance weights
            over valid draws.
        ess_median: Median effective sample size over valid draws. The
            weighted sd of a draw has a relative Monte Carlo error of about
            ``1 / sqrt(2 * ESS)``: 7% at 100, 22% at 10. With a small ESS
            the sd is also biased low, i.e. optimistic. Under priors the
            samples follow each draw's posterior, so the ESS does not fall
            with design quality: it is at most about 0.72 of the 65536
            samples at 4 parameters, and in the worst case about that of
            prior sampling with 3277 samples (see ``_proposal_sd``). With a
            ``belief`` the samples are the belief's, so a design far more
            informative than the belief lowers it; judge it here.
        n_draws: Number of draws from the priors or the belief.
        parameter_order: Names of the free parameters.
    """

    efficiency: dict[str, float]
    relative_sd: dict[str, float]
    prior_relative_sd: dict[str, float]
    maximin: float
    n_valid: int
    ess_min: float
    ess_median: float
    n_draws: int
    parameter_order: list[str]

    def __str__(self) -> str:
        width = max(len(name) for name in self.parameter_order)
        lines = [
            f"{name:<{width}}  ±{self.relative_sd[name]:.0%} "
            f"(×/÷{math.exp(self.relative_sd[name]):.2f})  "
            f"prior ±{self.prior_relative_sd[name]:.0%}  "
            f"efficiency {self.efficiency[name]:.2f}"
            for name in self.parameter_order
        ]
        lines.append(
            f"maximin {self.maximin:.2f} over {self.n_valid}/{self.n_draws} "
            f"draws; ESS min {self.ess_min:.0f}, median {self.ess_median:.0f}"
        )
        return "\n".join(lines)


def evaluate_design(
    model: Model,
    design: Dataset,
    noise: NoiseModel | float,
    *,
    key: jax.Array,
    n_draws: int = 256,
    config: SimulationConfig | None = None,
    belief: Mapping[str, ArrayLike] | None = None,
) -> DesignReport:
    """Scores a design by its expected per-parameter posterior shrinkage.

    For each draw ``theta_i`` from the free parameters' priors, the data are
    summarised by the Gaussian likelihood of ``log(theta)`` with precision
    ``F(theta_i)``, centred where a noisy dataset would put it. That likelihood
    is combined with the *exact* prior by importance sampling from a proposal
    centred on the draw's Laplace posterior, and the posterior sd is the
    weighted sd of those samples (see ``_proposal_sd``). Averaged over the
    draws, this is the expected posterior sd after the experiment. The
    proposal follows the posterior, so the ESS does not fall as designs
    improve; weighting prior samples instead, it did, and the sd came out
    optimistic.

    This report is not the search criterion. ``optimize_design`` searches on
    the Laplace posterior ``inv(F + inv(Sigma_0))``, which replaces a Uniform
    prior with a Gaussian of the same log variance. That Gaussian is wrong in
    both directions: it adds curvature the box does not have, and it ignores
    the box's edges, which cut a ridge short near a corner. Laplace sds are
    then off either way, by 3x in the worst cell. It still ranks designs
    correctly and is cheap and smooth. Checked against NUTS on the
    bi-substrate model (Uniform priors, 4 designs x 7 truths):

    ============================== ============ ==========
    predicted / NUTS               Laplace      reweighted
    ============================== ============ ==========
    posterior sd ratio, 10-90%     0.81-1.55    0.84-1.12
    bias in worst-parameter e      +0.036       +0.030
    rank of designs, Spearman      0.97         0.94
    ============================== ============ ==========

    A search on the reweighted criterion landed in the same basin, and the
    two designs tied under NUTS (-0.003 +- 0.007 worst-parameter e).

    With a ``belief`` (posterior samples from an earlier round), the draws are a
    bootstrap resample of those samples and the weights run over all of them,
    so correlations the earlier data left (a ridge between two parameters) are
    kept. The efficiency is then relative to the belief. A belief has samples
    but no density, so its samples are the candidates themselves (see
    ``_reweighted_sd``); they are far fewer than a proposal's, so check
    ``ess_min``. Longer chains raise it.

    Args:
        model: Model whose free parameters (``constant=False``) all carry a prior
            with strictly positive support, e.g. ``Uniform(low > 0)``.
        design: Dataset of initial-condition-only measurements, one per arm.
        noise: Observation noise model; a bare float means ``Homoskedastic``.
        key: PRNG key for the prior draws. The draws are the ones
            ``optimize_design``'s Laplace score uses for the same key.
        n_draws: Number of prior draws to average over.
        config: Solver configuration, see ``fisher_information``.
        belief: ``name -> samples`` in natural units, e.g.
            ``run_mcmc(...).get_samples()``: one 1-D array per free parameter,
            all the same length. Other keys (``sigma``, constant parameters)
            are ignored. None scores against the model's priors.

    Returns:
        A ``DesignReport``. Its numbers are NaN if no draw solved.

    Raises:
        ValueError: If ``belief`` is None and a free parameter has no prior or
            its prior allows values that are not strictly positive, or if
            ``belief`` misses a free parameter or holds samples that are not
            1-D, positive and finite, or too few of them.
    """
    information, parameter_order, (y0s, constants, times) = (
        _prepare_information_function(model, design, noise, config=config)
    )
    thetas, _, _ = _draws(model, parameter_order, key, n_draws, belief)
    # The search draws thetas from draw_key and uses cov_key only for the prior
    # variance, so the report keeps its thetas and takes the rest from cov_key.
    _, cov_key = jax.random.split(key)
    inner_key, z_key = jax.random.split(jax.random.fold_in(cov_key, 1))
    if belief is None:
        priors = _extract_priors(model, parameter_order)
        *keys, eps_key = jax.random.split(inner_key, len(priors) + 1)
        inner = jnp.log(
            jnp.stack(
                [p.sample(k, (_N_INNER_SAMPLES,)) for p, k in zip(priors, keys)],
                axis=-1,
            )
        )
    else:
        inner = jnp.log(_belief_samples(belief, parameter_order))
    zs = jax.random.normal(z_key, thetas.shape)

    def score(
        thetas: jax.Array, zs: jax.Array, inner: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        F, valid = jax.vmap(lambda t: information(t, y0s, constants, times))(thetas)

        # lax.map, not vmap: vmapped weights are (n_draws, n_inner, n_free),
        # about 0.5 GB at 256 draws.
        if belief is None:
            n_prior = round(_DEFENSIVE_FRACTION * _N_INNER_SAMPLES)
            moments = inner.mean(0), jnp.diag(1.0 / inner.var(0))
            log_prior = _log_prior_density(priors)

            def _proposal(args: tuple[jax.Array, ...]) -> tuple:
                log_theta, z, f, i = args
                # One block per draw, so the inner errors of the draws are
                # independent and average out.
                eps = jax.random.normal(
                    jax.random.fold_in(eps_key, i),
                    (_N_INNER_SAMPLES - n_prior, inner.shape[1]),
                )
                return _proposal_sd(
                    log_theta, z, f, eps, inner[:n_prior], *moments, log_prior
                )

            sd, ess = jax.lax.map(
                _proposal, (jnp.log(thetas), zs, F, jnp.arange(len(thetas)))
            )
            return sd, ess, valid

        def _one(args: tuple[jax.Array, jax.Array, jax.Array]) -> tuple:
            return _reweighted_sd(*args, inner)

        sd, ess = jax.lax.map(_one, (jnp.log(thetas), zs, F))
        return sd, ess, valid

    sd, ess, valid = jax.jit(score)(thetas, zs, inner)

    prior_sd = inner.std(0)
    ratio = sd / prior_sd
    n_valid = int(valid.sum())
    valid_ess = ess[valid]
    # One reduction for both, so maximin <= min(efficiency) holds to the bit when
    # the same parameter is the worst in every draw.
    columns = jnp.column_stack([ratio, ratio.max(-1)])
    *mean, worst = jnp.where(valid[:, None], columns, 0.0).sum(0) / n_valid
    return DesignReport(
        efficiency={name: float(1.0 - r) for name, r in zip(parameter_order, mean)},
        relative_sd={
            name: float(r * s) for name, r, s in zip(parameter_order, mean, prior_sd)
        },
        prior_relative_sd={
            name: float(s) for name, s in zip(parameter_order, prior_sd)
        },
        maximin=float(1.0 - worst),
        n_valid=n_valid,
        ess_min=float(valid_ess.min()) if n_valid else math.nan,
        ess_median=float(jnp.median(valid_ess)) if n_valid else math.nan,
        n_draws=n_draws,
        parameter_order=parameter_order,
    )


def _prepare_efficiency(
    model: Model,
    design: Dataset,
    noise: NoiseModel | float,
    *,
    key: jax.Array,
    n_draws: int,
    config: SimulationConfig | None,
    belief: Mapping[str, ArrayLike] | None = None,
) -> tuple[EfficiencyFunction, list[str], DesignArrays]:
    """Builds ``(y0s, constants) -> (e[n_draws, n_free], valid[n_draws])``.

    The draws are taken once, here, so every call scores against the same
    draws; the sampling times are fixed to the design's. The function is
    traceable and differentiable with respect to ``y0s`` and ``constants``.

    Args:
        model: Model whose free parameters all carry a positive-support prior.
        design: Design dataset; fixes the sampling times and the array shapes.
        noise: Observation noise model.
        key: PRNG key for the draws.
        n_draws: Number of draws.
        config: Solver configuration, or None for the defaults.
        belief: Posterior samples to draw from instead of the priors, see
            ``evaluate_design``.

    Returns:
        The efficiency function, the free parameter order and the design's
        ``(y0s, constants, times)``.
    """
    information, parameter_order, arrays = _prepare_information_function(
        model, design, noise, config=config
    )
    thetas, prior_var, prior_precision = _draws(
        model, parameter_order, key, n_draws, belief
    )
    times = arrays[2]

    def efficiency(
        thetas: jax.Array, y0s: jax.Array, constants: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        def _one(theta: jax.Array) -> tuple[jax.Array, jax.Array]:
            F, valid = information(theta, y0s, constants, times)
            posterior = jnp.linalg.inv(F + prior_precision)
            return 1.0 - jnp.sqrt(jnp.diag(posterior) / prior_var), valid

        return jax.vmap(_one)(thetas)

    return jax.tree_util.Partial(efficiency, thetas), parameter_order, arrays


def _reweighted_sd(
    log_theta: jax.Array,
    z: jax.Array,
    F: jax.Array,
    inner: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Posterior sd of ``log(theta)`` from importance weights over belief samples.

    Used for a ``belief``, which has samples but no density to build a proposal
    from. Under a prior, ``_proposal_sd`` replaced it; the measurements below
    are of this estimator over prior samples, the M = 4096 / 65536 tables
    included, and still describe it.

    The weight of inner sample ``u_k`` is the Gaussian likelihood
    ``exp(-1/2 ||L.T (u_k - log_theta) - z||^2)``, ``L L.T = F + 1e-6 I``: data
    whose estimate lands at ``log_theta + inv(L.T) z``. ``z`` is a fixed
    standard normal per draw, so averaging over draws averages over where noisy
    data would land, which is what "expected posterior sd" means. Setting
    ``z = 0`` instead (the posterior at noise-free data) overstated the NUTS sd
    by 5-13% at 16 arms.

    ``_N_INNER_SAMPLES = 65536``. Predicted / fixed-noise NUTS posterior sd on
    the bi-substrate model (Uniform priors, truth at the prior midpoints, 3
    noise seeds; kcat, ki_u, km_a, km_b; median and minimum ESS over 256 draws):

    ====== ====== ======================= ============
    arms   M      sd ratio                ESS med/min
    ====== ====== ======================= ============
    4      4096   0.87 0.89 0.92 0.87     48 / 16
    4      65536  0.91 0.92 0.96 0.92     867 / 293
    16     4096   0.92 0.94 0.97 0.98     21 / 6
    16     65536  1.01 1.02 1.03 1.00     409 / 115
    ====== ====== ======================= ============

    A weighted sd from few effective samples is biased low, so M = 4096 is
    optimistic. At 65536 the estimate has converged: on the 4-arm optimum
    (64 draws) it is within 0.1% of M = 4194304 for every
    parameter.

    There is no automatic switch to another estimator when few samples carry
    weight: any threshold for it would be arbitrary and would decide which
    formula produces the number. The effective sample size of every draw is
    returned instead (``DesignReport.ess_min``/``ess_median``) for the user to
    judge. Measured against the M = 4194304 reference on the 4-arm optimum (64
    draws), the sd was within 0.1% with the design's noise, 0.88-0.92 with the
    noise divided by 4 (ESS < 10 in every draw), and 0.29-0.50 with it divided
    by 16. A low ESS therefore means an optimistic number; Laplace is no safe
    substitute there either (1.10-1.15 at /4, and 2.3-2.5x the fixed-noise
    NUTS sd for kcat and ki_u at one truth where 79% of draws had ESS < 100).
    Over 4 designs x 7 truths, mean ``|log(predicted / NUTS sd)|`` was 0.196 for
    Laplace and 0.090 for reweighting.

    Cost: ``evaluate_design`` with 256 draws and M = 65536 on the bi-substrate
    model (4 states, 4 and 16 arms x 6 times, CPU, float64) took 1.0-1.2 s per
    call including the compile, the same as the Laplace score of the same draws
    (1.1-1.4 s). The sensitivity solves dominate.

    Args:
        log_theta: ``log(theta_i)``, shape ``(n_free,)``.
        z: Standard normal offset of the data, shape ``(n_free,)``.
        F: Fisher information at ``theta_i``.
        inner: Log prior (or belief) samples, shape ``(n_inner, n_free)``.

    Returns:
        The posterior sd per parameter and the effective sample size
        ``1 / sum(w**2)`` of the normalised weights.
    """
    L = jnp.linalg.cholesky(F + _CHOLESKY_JITTER * jnp.eye(F.shape[0]))
    residual = (inner - log_theta) @ L - z
    w = jax.nn.softmax(-0.5 * jnp.sum(residual**2, axis=-1))
    mean = w @ inner
    sd = jnp.sqrt(w @ (inner - mean) ** 2)
    return sd, 1.0 / jnp.sum(w**2)


def _proposal_sd(
    log_theta: jax.Array,
    z: jax.Array,
    F: jax.Array,
    eps: jax.Array,
    prior_inner: jax.Array,
    mu0: jax.Array,
    prec0: jax.Array,
    log_prior: Callable[[jax.Array], jax.Array],
) -> tuple[jax.Array, jax.Array]:
    """Posterior sd of ``log(theta)`` under a prior, by importance sampling.

    The prior-path twin of ``_reweighted_sd``: the same target, the exact
    prior times the Gaussian likelihood ``exp(-1/2 ||L.T (u - log_theta) -
    z||^2)`` with ``L L.T = F_J = F + 1e-6 I``, but the samples come from a
    proposal that follows the posterior instead of from the prior. Prior
    samples land in a posterior far narrower than the prior at a rate of about
    ``prod(sd_post / sd_prior)``, so their ESS fell as designs improved (median
    11, minimum 2 at 6 arms on competitive product inhibition), and a small ESS
    biases the sd low.

    The proposal is a defensive mixture (Hesterberg 1995) of ``n_g`` samples
    from ``N(m, c^2 inv(P))`` and ``n_p`` prior samples. ``P = F_J + prec0``
    and ``m = inv(P) (F_J u_hat + prec0 mu0)``, with ``u_hat = log_theta +
    inv(L.T) z`` where the data put the estimate, make the Gaussian part the
    Laplace posterior under the log prior's moments, widened by
    ``c = _PROPOSAL_SCALE``. Its density
    is weighted by the actual counts ``n_g / M`` and ``n_p / M``, which makes
    this a deterministic-mixture estimator whose normaliser estimate is
    unbiased (Owen & Zhou 2000); only the usual O(1/ESS) bias of
    self-normalising remains. In ``u`` the target is a Gaussian cut off at the
    prior's support (Uniform, LogUniform) or a Gaussian (LogNormal), so it has
    no second mode to miss, and the Gaussian part is the exact posterior for
    LogNormal at ``c = 1``. The ESS is at most about ``(1 - alpha) M
    (sqrt(2 c^2 - 1) / c^2)^n_free``, 0.72 M at 4 parameters. The prior part,
    ``alpha = _DEFENSIVE_FRACTION``, bounds the weights by ``M / n_p``, so the
    worst case (a box corner, a near-singular ``F``) is about prior sampling
    with ``n_p`` samples, not worse. Samples outside the prior's support get
    weight 0. Measurements for ``c`` and ``alpha`` are with the constants.

    Cost: ``evaluate_design`` with 1024 draws on the bi-substrate model (4
    arms x 6 times, CPU, float64, M4 Pro) took 3.2-4.1 s per call including
    the compile, against 1.4-2.0 s with prior sampling. About 1 s of the
    difference is drawing the proposal's normals.

    Args:
        log_theta: ``log(theta_i)``, shape ``(n_free,)``.
        z: Standard normal offset of the data, shape ``(n_free,)``.
        F: Fisher information at ``theta_i``.
        eps: Standard normals for the Gaussian part, shape ``(n_g, n_free)``.
        prior_inner: Log prior samples for the prior part, ``(n_p, n_free)``.
        mu0: Mean of the log prior, shape ``(n_free,)``.
        prec0: Precision matrix of the log prior.
        log_prior: Log prior density in ``u``, see ``_log_prior_density``.

    Returns:
        The posterior sd per parameter and the effective sample size
        ``1 / sum(w**2)`` of the normalised weights.
    """
    n_free = F.shape[0]
    n_gauss, n_prior = eps.shape[0], prior_inner.shape[0]
    F_J = F + _CHOLESKY_JITTER * jnp.eye(n_free)
    L = jnp.linalg.cholesky(F_J)
    L_P = jnp.linalg.cholesky(F_J + prec0)
    # F_J u_hat = F_J log_theta + L z; u_hat itself is ~1e3 in directions that
    # only the jitter informs, so it is never formed.
    m = jax.scipy.linalg.cho_solve((L_P, True), F_J @ log_theta + L @ z + prec0 @ mu0)
    c = _PROPOSAL_SCALE
    gauss = m + c * jax.scipy.linalg.solve_triangular(L_P.T, eps.T).T
    u = jnp.concatenate([gauss, prior_inner])

    log_gauss = (
        -0.5 * jnp.sum(((u - m) @ L_P) ** 2, axis=-1) / c**2
        + jnp.sum(jnp.log(jnp.diag(L_P)))
        - n_free * (jnp.log(c) + 0.5 * jnp.log(2 * jnp.pi))
    )
    lp = log_prior(u)
    n = n_gauss + n_prior
    log_q = jnp.logaddexp(jnp.log(n_gauss / n) + log_gauss, jnp.log(n_prior / n) + lp)
    residual = (u - log_theta) @ L - z
    w = jax.nn.softmax(lp - 0.5 * jnp.sum(residual**2, axis=-1) - log_q)
    mean = w @ u
    sd = jnp.sqrt(w @ (u - mean) ** 2)
    return sd, 1.0 / jnp.sum(w**2)


def _log_prior_density(
    priors: list[dist.Distribution],
) -> Callable[[jax.Array], jax.Array]:
    """``u -> log p(u)`` for ``u = log(theta)`` of shape ``(..., n_free)``.

    The priors are independent; ``+ u`` is the Jacobian into log space. Outside
    a prior's support the density is ``-inf``: numpyro's own ``log_prob`` is
    finite there for ``Uniform`` and ``LogUniform``, and NaN at 0 for
    ``LogNormal``.
    """

    def log_prior(u: jax.Array) -> jax.Array:
        theta = jnp.exp(u)
        return sum(
            jnp.where(
                prior.support(theta[..., j]),
                prior.log_prob(theta[..., j]) + u[..., j],
                -jnp.inf,
            )
            for j, prior in enumerate(priors)
        )

    return log_prior


def _draws(
    model: Model,
    parameter_order: list[str],
    key: jax.Array,
    n_draws: int,
    belief: Mapping[str, ArrayLike] | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Draws ``theta[n_draws, n_free]`` and the Laplace prior in log space.

    Returns:
        The draws in natural units, the prior variance of each log parameter
        and the prior precision matrix ``inv(Sigma_0)``.
    """
    draw_key, cov_key = jax.random.split(key)
    if belief is None:
        priors = _extract_priors(model, parameter_order)
        thetas = jnp.stack(
            [
                prior.sample(k, (n_draws,))
                for prior, k in zip(priors, jax.random.split(draw_key, len(priors)))
            ],
            axis=-1,
        )
        prior_var = _prior_log_variance(priors, cov_key)
        prior_precision = jnp.diag(1.0 / prior_var)
    else:
        samples = _belief_samples(belief, parameter_order)
        # A bootstrap resample, not the first n_draws rows: those would be one
        # autocorrelated stretch of the chain.
        rows = jax.random.choice(draw_key, len(samples), (n_draws,), replace=True)
        thetas = samples[rows]
        # Full covariance: posteriors are correlated (log kcat/ki_u at -0.98 on MAT).
        prior_cov = jnp.atleast_2d(jnp.cov(jnp.log(samples), rowvar=False))
        prior_var = jnp.diag(prior_cov)
        prior_precision = jnp.linalg.inv(prior_cov)
    return thetas, prior_var, prior_precision


def _extract_priors(model: Model, names: list[str]) -> list[dist.Distribution]:
    """Numpyro distributions of the named parameters' priors.

    Raises:
        ValueError: If a prior is missing or its support is not strictly positive.
    """
    priors = []
    for name in names:
        prior = model.parameters[name].prior
        if prior is None:
            raise ValueError(
                f"Parameter '{name}' has no prior. Set one, e.g. "
                f"model.parameters['{name}'].prior = "
                "catalax.mcmc.priors.Uniform(low=..., high=...), or mark it "
                "constant=True to exclude it from the design."
            )
        distribution = prior._distribution_fun()
        support = distribution.support
        lower = getattr(support, "lower_bound", None)
        if support is not constraints.positive and not (
            lower is not None and lower > 0
        ):
            raise ValueError(
                f"The prior of parameter '{name}' ({type(prior).__name__}) allows "
                "values <= 0, but design is scored in log parameters. Use a prior "
                "with strictly positive support, e.g. Uniform with low > 0, "
                "LogNormal or LogUniform."
            )
        priors.append(distribution)
    return priors


def _belief_samples(belief: Mapping[str, ArrayLike], names: list[str]) -> jax.Array:
    """Stacks the named parameters' samples into ``(n_samples, n_free)``.

    Raises:
        ValueError: Naming the parameter and the fix.
    """
    missing = [name for name in names if name not in belief]
    if missing:
        raise ValueError(
            f"belief has no samples for the free parameters {missing}. Pass "
            "run_mcmc(...).get_samples(), or mark a parameter constant=True to "
            "exclude it from the design."
        )
    columns = []
    for name in names:
        column = np.asarray(belief[name], dtype=float)
        if column.ndim != 1:
            raise ValueError(
                f"belief['{name}'] has shape {column.shape}; expected one sample "
                "per entry. Flatten the chains, e.g. with .reshape(-1)."
            )
        if not np.all(np.isfinite(column) & (column > 0)):
            raise ValueError(
                f"belief['{name}'] holds values that are not finite and > 0, but "
                "design is scored in log parameters. Drop those samples, or check "
                "that the prior has strictly positive support."
            )
        columns.append(column)
    lengths = {name: len(column) for name, column in zip(names, columns)}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"belief arrays have different lengths {lengths}. Pass samples from "
            "one run, row i of every array being the same draw."
        )
    n_samples = len(columns[0])
    if n_samples <= len(names):
        raise ValueError(
            f"belief has {n_samples} samples for {len(names)} free parameters; "
            "the covariance needs more samples than parameters. Draw more."
        )
    return jnp.asarray(np.stack(columns, axis=-1))


def _prior_log_variance(priors: list[dist.Distribution], key: jax.Array) -> jax.Array:
    """Prior variance of each log parameter.

    Priors are independent, so the log-space prior covariance is diagonal.
    Exact for LogNormal; otherwise the variance of ``log`` of prior samples.
    """
    variances = []
    for prior, k in zip(priors, jax.random.split(key, len(priors))):
        if isinstance(prior, dist.LogNormal):
            variances.append(jnp.asarray(prior.scale, dtype=float) ** 2)
        else:
            variances.append(jnp.var(jnp.log(prior.sample(k, (_N_PRIOR_SAMPLES,)))))
    return jnp.stack(variances)
