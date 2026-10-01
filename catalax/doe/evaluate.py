"""Expected posterior shrinkage of a design under the priors or a belief."""

from __future__ import annotations

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

type EfficiencyFunction = Callable[[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]
"""``(y0s, constants) -> (efficiency[n_draws, n_free], valid[n_draws])``."""


@dataclass
class DesignReport:
    """How much a design is expected to shrink each parameter's uncertainty.

    Efficiency ``e_j = 1 - sd_post_j / sd_prior_j`` in log space: 0 means the
    design adds nothing to the prior, 1 means the parameter is pinned exactly.
    With a ``belief``, "prior" is that belief: efficiency is the shrinkage
    beyond what earlier rounds already taught, not beyond the model's priors.

    Attributes:
        efficiency: Mean efficiency per free parameter over valid draws.
        maximin: Mean over valid draws of the worst parameter's efficiency,
            ``mean_i min_j e_ij``.
        n_valid: Number of draws whose simulations all solved.
        n_draws: Number of draws from the priors or the belief.
        parameter_order: Names of the free parameters.
    """

    efficiency: dict[str, float]
    maximin: float
    n_valid: int
    n_draws: int
    parameter_order: list[str]


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

    For each draw ``theta_i`` from the free parameters' priors, the Laplace
    posterior in log space is ``Sigma_post = inv(F(theta_i) + inv(Sigma_0))``,
    where ``Sigma_0`` is the prior covariance of ``log(theta)``: exact for
    ``LogNormal`` priors, estimated from prior samples otherwise.

    With a ``belief`` (posterior samples from an earlier round), the draws are a
    bootstrap resample of those samples and ``Sigma_0`` is their full log-space
    covariance, so correlations the earlier data left (a ridge between two
    parameters) are kept. The efficiency is then relative to the belief.

    Args:
        model: Model whose free parameters (``constant=False``) all carry a prior
            with strictly positive support, e.g. ``Uniform(low > 0)``.
        design: Dataset of initial-condition-only measurements, one per arm.
        noise: Observation noise model; a bare float means ``Homoskedastic``.
        key: PRNG key for the prior draws.
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
    efficiency_fn, parameter_order, (y0s, constants, _) = _prepare_efficiency(
        model, design, noise, key=key, n_draws=n_draws, config=config, belief=belief
    )
    # Prior draws enter the jit as an argument (they are the Partial's leaf) and
    # the design as a constant, exactly as before the helper was factored out:
    # swapping the two moves the scores in the 15th digit.
    efficiency, valid = jax.jit(lambda fn: fn(y0s, constants))(efficiency_fn)

    n_valid = int(valid.sum())
    # One reduction for both, so maximin <= min(efficiency) holds to the bit when
    # the same parameter is the worst in every draw.
    columns = jnp.column_stack([efficiency, efficiency.min(-1)])
    *mean, maximin = jnp.where(valid[:, None], columns, 0.0).sum(0) / n_valid
    return DesignReport(
        efficiency={name: float(e) for name, e in zip(parameter_order, mean)},
        maximin=float(maximin),
        n_valid=n_valid,
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
