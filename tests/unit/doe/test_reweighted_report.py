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
from catalax.doe.evaluate import _draws, _prepare_efficiency

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


def test_near_noiseless_design_shows_a_collapsed_ess():
    # Nothing switches estimators behind the user's back: a design that pins the
    # parameters far below their prior width leaves few prior samples with
    # weight, and the report says so instead of quietly substituting Laplace.
    model = _mm_model()
    design = _mm_design(model)
    good = cdoe.evaluate_design(
        model, design, NOISE, key=jax.random.PRNGKey(2), n_draws=16
    )
    noiseless = cdoe.evaluate_design(
        model, design, cdoe.Homoskedastic(1e-4), key=jax.random.PRNGKey(2), n_draws=16
    )
    print(
        f"\nESS median: design noise {good.ess_median:.0f}, "
        f"noiseless {noiseless.ess_median:.1f}"
    )
    assert noiseless.n_valid == 16
    assert noiseless.ess_median < 0.01 * good.ess_median
    assert "ESS min" in str(noiseless)


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
