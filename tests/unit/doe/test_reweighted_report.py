"""The reweighted report of evaluate_design against exact references."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate, stats

import catalax as ctx
import catalax.doe as cdoe
import catalax.mcmc as cmm
from catalax.doe.evaluate import (
    _DEFENSIVE_FRACTION,
    _N_INNER_SAMPLES,
    _draws,
    _log_prior_density,
    _prepare_efficiency,
    _proposal_sd,
    _reweighted_sd,
)
from catalax.doe.information import _prepare_information_function

pytestmark = pytest.mark.usefixtures("x64")

TIMES = [10.0, 40.0, 80.0, 150.0, 300.0]
NOISE = cdoe.Proportional(cv=0.05, floor=0.5)
N_REPLICATES = 16


def _mm_model(prior: str = "uniform") -> ctx.Model:
    """Michaelis-Menten with ``e`` as an unobservable zero-rate state."""
    model = ctx.Model(name="Michaelis-Menten")
    model.add_state("s, p, e")
    model.add_ode("e", "0", observable=False)
    model.add_ode("s", "-kcat * e * s / (k_m + s)", observable=True)
    model.add_ode("p", "kcat * e * s / (k_m + s)", observable=False)
    for name, (low, high) in {"k_m": (50.0, 200.0), "kcat": (5.0, 20.0)}.items():
        model.parameters[name].value = math.sqrt(low * high)
        if prior == "uniform":
            model.parameters[name].prior = cmm.priors.Uniform(low=low, high=high)
        else:
            model.parameters[name].prior = cmm.priors.LogNormal(
                mu=math.sqrt(low * high), sigma=0.5
            )
    return model


def _mm_design(model: ctx.Model) -> ctx.Dataset:
    design = ctx.Dataset.from_model(model)
    for s0 in (50.0, 400.0):
        design.add_initial(time=TIMES, s=s0, p=0.0, e=0.1)
    return design


def _sd_list(report: cdoe.DesignReport) -> list[float]:
    return [report.relative_sd[name] for name in report.parameter_order]


def test_matches_quadrature_when_the_information_is_constant():
    # dP/dt = k from P = 0: dP/dlog(k) = k t and sd(P) = cv k t (the floor is
    # 1e-6), so every time point adds 1/cv^2 and F = n / cv^2 for every k. The
    # posterior of u = log k given an estimate u_hat ~ N(u, 1/F) is then a
    # normal N(u_hat + 1/F, 1/F) (prior density e^u) truncated to the log box.
    low, high, cv, times = 1.0, 10.0, 0.5, [1.0, 2.0]
    F = len(times) / cv**2
    model = ctx.Model(name="constant rate")
    model.add_state("P")
    model.add_ode("P", "k", observable=True)
    model.parameters["k"].value = 3.0
    model.parameters["k"].prior = cmm.priors.Uniform(low=low, high=high)
    design = ctx.Dataset.from_model(model)
    design.add_initial(time=times, P=0.0)
    noise = cdoe.Proportional(cv=cv, floor=1e-6)
    np.testing.assert_allclose(
        cdoe.fisher_information(model, design, noise).matrix, [[F]], rtol=1e-9
    )

    lo, hi, s = math.log(low), math.log(high), 1 / math.sqrt(F)

    def posterior_sd(u_hat: float) -> float:
        m = u_hat + 1 / F
        return stats.truncnorm.std((lo - m) / s, (hi - m) / s, loc=m, scale=s)

    expected, _ = integrate.dblquad(
        lambda z, u: (
            math.exp(u) / (high - low) * stats.norm.pdf(z) * posterior_sd(u + z * s)
        ),
        lo,
        hi,
        -8.0,
        8.0,
    )

    reports = [
        cdoe.evaluate_design(model, design, noise, key=k, n_draws=256)
        for k in jax.random.split(jax.random.PRNGKey(0), N_REPLICATES)
    ]
    assert all(r.n_valid == 256 for r in reports)
    sds = np.array([r.relative_sd["k"] for r in reports])
    se = sds.std(ddof=1) / math.sqrt(N_REPLICATES)
    laplace = 1 / math.sqrt(F + 1 / reports[0].prior_relative_sd["k"] ** 2)
    print(
        f"\nquadrature {expected:.5f}, reweighted {sds.mean():.5f} +- {se:.5f} "
        f"({(sds.mean() - expected) / se:+.2f} SE), Laplace {laplace:.5f}"
    )
    assert abs(sds.mean() - expected) < 3 * se
    # The oracle has teeth: Laplace is many SE away.
    assert abs(laplace - expected) > 10 * se


def test_lognormal_prior_gives_the_laplace_sd():
    # Gaussian prior x Gaussian likelihood in log space: the posterior sd is
    # Laplace's for every dataset, so only inner Monte Carlo error separates
    # them. Same key, same thetas, so the two are compared draw for draw.
    model = _mm_model("lognormal")
    design = _mm_design(model)
    order = model.get_parameter_order()
    gaps = []
    for key in jax.random.split(jax.random.PRNGKey(0), N_REPLICATES):
        report = cdoe.evaluate_design(model, design, NOISE, key=key, n_draws=32)
        fn, _, (y0s, constants, _) = _prepare_efficiency(
            model, design, NOISE, key=key, n_draws=32, config=None
        )
        e, valid = fn(y0s, constants)
        assert bool(valid.all())
        _, prior_var, _ = _draws(model, order, key, 32, None)
        laplace = np.asarray(((1.0 - e) * jnp.sqrt(prior_var)).mean(0))
        gaps.append(np.array(_sd_list(report)) - laplace)
    gaps = np.array(gaps)
    se = gaps.std(0, ddof=1) / math.sqrt(N_REPLICATES)
    print(f"\nreweighted - Laplace {gaps.mean(0)} +- {se} ({order})")
    assert np.all(np.abs(gaps.mean(0)) < 3 * se)


def test_report_is_consistent_and_deterministic():
    model = _mm_model()
    design = _mm_design(model)
    key = jax.random.PRNGKey(1)
    report = cdoe.evaluate_design(model, design, NOISE, key=key, n_draws=32)

    for name in report.parameter_order:
        ratio = report.relative_sd[name] / report.prior_relative_sd[name]
        assert report.efficiency[name] == pytest.approx(1.0 - ratio, abs=1e-12)
    assert report.maximin <= min(report.efficiency.values())
    assert report == cdoe.evaluate_design(model, design, NOISE, key=key, n_draws=32)
    lines = str(report).splitlines()
    assert [line.split()[0] for line in lines] == ["k_m", "kcat", "maximin"]


def test_near_noiseless_design_keeps_its_ess():
    # Prior samples used to be the candidates, so a design that pins the
    # parameters far below their prior width left almost none with weight (ESS
    # about 1, the sd biased low). The proposal follows each draw's posterior
    # instead. Under a Uniform prior that posterior is N(., inv(F_J)) cut off
    # at the box, so away from the edges its sd is sqrt(diag(inv(F_J))) exactly.
    model = _mm_model()
    design = _mm_design(model)
    noise = cdoe.Homoskedastic(1e-4)
    order = model.get_parameter_order()
    information, _, (y0s, constants, times) = _prepare_information_function(
        model, design, noise, config=None
    )
    bounds = jnp.log(jnp.array([[50.0, 5.0], [200.0, 20.0]]))  # k_m, kcat
    gaps, ess = [], []
    for key in jax.random.split(jax.random.PRNGKey(2), N_REPLICATES):
        report = cdoe.evaluate_design(model, design, noise, key=key, n_draws=16)
        assert report.n_valid == 16
        thetas, _, _ = _draws(model, order, key, 16, None)
        F, _ = jax.vmap(lambda t: information(t, y0s, constants, times))(thetas)
        cov = jnp.linalg.inv(F + 1e-6 * jnp.eye(2))
        exact = jnp.sqrt(jnp.diagonal(cov, axis1=1, axis2=2))
        # The data put the posterior a few sd from the draw; 10 sd from an edge
        # leaves the truncation nothing to cut.
        edge = jnp.minimum(jnp.log(thetas) - bounds[0], bounds[1] - jnp.log(thetas))
        assert bool(jnp.all(edge > 10 * exact))
        gaps.append(np.array(_sd_list(report)) / np.asarray(exact.mean(0)) - 1.0)
        ess.append(report.ess_median)
    gaps = np.array(gaps)
    se = gaps.std(0, ddof=1) / math.sqrt(N_REPLICATES)
    print(
        f"\nnoiseless: sd / exact - 1 = {gaps.mean(0)} +- {se} ({order}); "
        f"ESS median {min(ess):.0f}-{max(ess):.0f} of {_N_INNER_SAMPLES}"
    )
    assert np.all(np.abs(gaps.mean(0)) < 3 * se)
    assert min(ess) > 0.1 * _N_INNER_SAMPLES
    assert "ESS min" in str(report)


def test_matches_quadrature_with_three_parameters():
    # Three independent constant rates, F = 800 I: the posterior factorises,
    # so each parameter's expected sd is the 1-D quadrature of
    # test_matches_quadrature_when_the_information_is_constant. Prior samples
    # land in the posterior at a rate of about prod(sd_post / sd_prior), so
    # the prior-sampling estimator collapses here; one parameter cannot show
    # it (at cv 0.02 in 1-D it was only -0.98 SE off).
    low, high, cv, times = 1.0, 10.0, 0.05, [1.0, 2.0]
    F = len(times) / cv**2
    model = ctx.Model(name="three constant rates")
    model.add_state("P1, P2, P3")
    for j in (1, 2, 3):
        model.add_ode(f"P{j}", f"k{j}", observable=True)
        model.parameters[f"k{j}"].value = 3.0
        model.parameters[f"k{j}"].prior = cmm.priors.Uniform(low=low, high=high)
    design = ctx.Dataset.from_model(model)
    design.add_initial(time=times, P1=0.0, P2=0.0, P3=0.0)
    noise = cdoe.Proportional(cv=cv, floor=1e-6)
    np.testing.assert_allclose(
        cdoe.fisher_information(model, design, noise).matrix,
        F * np.eye(3),
        rtol=1e-9,
        atol=1e-9,
    )

    lo, hi, s = math.log(low), math.log(high), 1 / math.sqrt(F)

    def posterior_sd(u_hat: float) -> float:
        m = u_hat + 1 / F
        return stats.truncnorm.std((lo - m) / s, (hi - m) / s, loc=m, scale=s)

    expected, _ = integrate.dblquad(
        lambda z, u: (
            math.exp(u) / (high - low) * stats.norm.pdf(z) * posterior_sd(u + z * s)
        ),
        lo,
        hi,
        -8.0,
        8.0,
    )

    order = model.get_parameter_order()
    prior = model.parameters["k1"].prior._distribution_fun()
    new, old, ess = [], [], []
    for key in jax.random.split(jax.random.PRNGKey(0), N_REPLICATES):
        report = cdoe.evaluate_design(model, design, noise, key=key, n_draws=256)
        assert report.n_valid == 256
        new.append(_sd_list(report))
        ess.append(report.ess_median)
        # The prior-sampling estimator (the belief path's) on the same draws.
        thetas, _, _ = _draws(model, order, key, 256, None)
        inner_key, z_key = jax.random.split(jax.random.fold_in(key, 1))
        inner = jnp.log(prior.sample(inner_key, (_N_INNER_SAMPLES, 3)))
        zs = jax.random.normal(z_key, thetas.shape)
        sd, _ = jax.lax.map(
            lambda a, inner=inner: _reweighted_sd(*a, F * jnp.eye(3), inner),
            (jnp.log(thetas), zs),
        )
        old.append(sd.mean(0))
    new, old = np.array(new), np.array(old)
    se_new = new.std(0, ddof=1) / math.sqrt(N_REPLICATES)
    se_old = old.std(0, ddof=1) / math.sqrt(N_REPLICATES)
    z_new = (new.mean(0) - expected) / se_new
    z_old = (old.mean(0) - expected) / se_old
    # The three parameters are exchangeable, so their mean has the most teeth.
    pooled = old.mean(1)
    se_pooled = pooled.std(ddof=1) / math.sqrt(N_REPLICATES)
    z_pooled = (pooled.mean() - expected) / se_pooled
    print(
        f"\nquadrature {expected:.5f}; proposal {new.mean(0)} ({z_new} SE, ESS "
        f"median {min(ess):.0f}-{max(ess):.0f}); prior sampling {old.mean(0)} "
        f"({z_old} SE, pooled {z_pooled:+.2f} SE)"
    )
    assert np.all(np.abs(z_new) < 3)
    # The oracle has teeth: prior sampling is biased low, i.e. optimistic.
    assert z_pooled < -3


def test_samples_outside_the_box_get_no_weight():
    # Uniform(1, 10) at its upper edge, with data that put the estimate 3 sd
    # above the box: the proposal's mean lies outside the box, and so do most
    # of its samples. With zero weight there, the posterior is N(u_hat + 1/F,
    # 1/F) cut off at log(10), whose sd is a third of the uncut 1/sqrt(F).
    F, z, theta = 800.0, 3.0, 9.9
    prior = cmm.priors.Uniform(low=1.0, high=10.0)._distribution_fun()
    s = 1 / math.sqrt(F)
    m = math.log(theta) + z * s + 1 / F
    expected = stats.truncnorm.std(-np.inf, (math.log(10.0) - m) / s, loc=m, scale=s)

    n_prior = round(_DEFENSIVE_FRACTION * _N_INNER_SAMPLES)
    sds = []
    for key in jax.random.split(jax.random.PRNGKey(0), N_REPLICATES):
        prior_key, eps_key = jax.random.split(key)
        inner = jnp.log(prior.sample(prior_key, (_N_INNER_SAMPLES, 1)))
        sd, ess = _proposal_sd(
            jnp.log(jnp.array([theta])),
            jnp.array([z]),
            jnp.array([[F]]),
            jax.random.normal(eps_key, (_N_INNER_SAMPLES - n_prior, 1)),
            inner[:n_prior],
            inner.mean(0),
            jnp.diag(1.0 / inner.var(0)),
            _log_prior_density([prior]),
        )
        assert bool(jnp.isfinite(sd).all() & jnp.isfinite(ess))
        sds.append(float(sd[0]))
    se = np.std(sds, ddof=1) / math.sqrt(N_REPLICATES)
    print(
        f"\ncut-off sd {expected:.5f}, proposal {np.mean(sds):.5f} +- {se:.5f}, "
        f"uncut {s:.5f}"
    )
    assert abs(np.mean(sds) - expected) < 3 * se


def test_belief_path_weights_the_belief_samples():
    model = _mm_model("lognormal")
    cov = 0.25 * jnp.array([[1.0, -0.9], [-0.9, 1.0]])
    logs = jax.random.multivariate_normal(
        jax.random.PRNGKey(0), jnp.zeros(2), cov, (4_000,)
    )
    belief = {"k_m": 100.0 * jnp.exp(logs[:, 0]), "kcat": 10.0 * jnp.exp(logs[:, 1])}
    report = cdoe.evaluate_design(
        model, _mm_design(model), NOISE, key=jax.random.PRNGKey(1), belief=belief
    )
    assert report.n_valid == report.n_draws
    np.testing.assert_allclose(
        [report.prior_relative_sd[name] for name in ("k_m", "kcat")],
        logs.std(0),
        rtol=1e-12,
    )
    assert all(0.0 < sd < 0.5 for sd in _sd_list(report))
