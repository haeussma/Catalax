"""Design, simulate, fit, redesign against the posterior, refit: MAT, two rounds."""

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

import catalax as ctx
import catalax.doe as cdoe
import catalax.mcmc as cmm
from catalax.doe.evaluate import _extract_priors, _prior_log_variance

FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "doe_mat_design_oracle.json"
)
NOISE = cdoe.Proportional(cv=0.05, floor=5.0)
SEARCH = {"n_arms": 4, "n_restarts": 4, "n_steps": 150, "n_draws": 16}
MCMC = cmm.MCMCConfig(
    num_warmup=500,
    num_samples=1000,
    likelihood=dist.Normal,
    noise_cv=0.1,
    verbose=0,
    max_steps=16384,
)

pytestmark = [pytest.mark.usefixtures("x64"), pytest.mark.expensive]


def _load_meta() -> dict:
    if not FIXTURE.exists():
        pytest.skip(f"oracle fixture {FIXTURE} is missing")
    return json.loads(FIXTURE.read_text())["meta"]


def _mat_model(meta: dict) -> ctx.Model:
    """MAT with the enzyme as an unobservable zero-rate state, truth at midpoints."""
    rate = meta["rate_law"]
    model = ctx.Model(name="MAT")
    model.add_state("A, B, E, P")
    model.add_ode("A", f"-({rate})", observable=False)
    model.add_ode("B", f"-({rate})", observable=False)
    model.add_ode("E", "0", observable=False)
    model.add_ode("P", rate, observable=True)
    for name, prior in meta["priors"].items():
        model.parameters[name].prior = cmm.priors.Uniform(
            low=prior["low"], high=prior["high"]
        )
        model.parameters[name].value = (prior["low"] * prior["high"]) ** 0.5
    return model


def _measure(
    model: ctx.Model, design: ctx.Dataset, times: list, key: jax.Array
) -> ctx.Dataset:
    """Simulates the design at the truth and adds the design's own noise."""
    config = ctx.SimulationConfig(t1=max(times))
    data = model.simulate(design, config, saveat=design.to_time_matrix())
    for meas in data.measurements:
        key, k = jax.random.split(key)
        y = meas.data["P"]
        meas.data["P"] = y + NOISE.scale(y) * jax.random.normal(k, y.shape)
    return data


def _log_posterior(
    results: cmm.HMCResults, order: list[str]
) -> tuple[jax.Array, jax.Array]:
    samples = results.get_samples()
    logs = jnp.stack([jnp.log(samples[name]) for name in order], axis=-1)
    return logs.mean(0), logs.std(0)


def _prior_var(model: ctx.Model, order: list[str]) -> jax.Array:
    """Prior variance of each log parameter, as the design scores it."""
    return _prior_log_variance(_extract_priors(model, order), jax.random.PRNGKey(0))


def _laplace_sd(model: ctx.Model, design: ctx.Dataset) -> jax.Array:
    """Laplace posterior sd in log space at the truth."""
    info = cdoe.fisher_information(model, design, NOISE)
    prior_var = _prior_var(model, info.parameter_order)
    posterior = jnp.linalg.inv(info.matrix + jnp.diag(1.0 / prior_var))
    return jnp.sqrt(jnp.diag(posterior))


def test_two_rounds_cover_the_truth_and_shrink_the_worst_parameter():
    meta = _load_meta()
    model = _mat_model(meta)
    order = model.get_parameter_order()
    spec = {name: tuple(bounds) for name, bounds in meta["spec"].items()}
    times = meta["times"]
    log_truth = jnp.log(jnp.array([model.parameters[n].value for n in order]))
    prior_sd = jnp.sqrt(_prior_var(model, order))
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    start = time.perf_counter()

    round_1 = cdoe.optimize_design(
        model, spec, NOISE, times=times, key=keys[0], **SEARCH
    )
    data = _measure(model, round_1.dataset, times, keys[1])
    fit_1 = cmm.run_mcmc(model, data, 5.0, MCMC)
    mean_1, sd_1 = _log_posterior(fit_1, order)
    z = (mean_1 - log_truth) / sd_1

    round_2 = cdoe.optimize_design(
        model,
        spec,
        NOISE,
        times=times,
        key=keys[2],
        belief=fit_1.get_samples(),
        **SEARCH,
    )
    # Refit everything under the original priors, never the last posterior as a
    # prior: the posterior samples are finite and the prior would be a stand-in.
    for meas in _measure(model, round_2.dataset, times, keys[3]).measurements:
        data.add_measurement(meas)
    fit_2 = cmm.run_mcmc(model, data, 5.0, MCMC)
    _, sd_2 = _log_posterior(fit_2, order)

    merged = ctx.Dataset.from_model(model)
    for meas in [*round_1.dataset.measurements, *round_2.dataset.measurements]:
        merged.add_measurement(meas)
    ratio_1 = _laplace_sd(model, round_1.dataset) / sd_1
    ratio_2 = _laplace_sd(model, merged) / sd_2

    print(
        f"\nclose the loop {time.perf_counter() - start:.0f} s; order {order}\n"
        f"round 1 predicted efficiency {round_1.report.efficiency}\n"
        f"round 2 predicted efficiency (vs round-1 posterior) "
        f"{round_2.report.efficiency}\n"
        f"z(truth) round 1 {np.round(z, 2)}\n"
        f"posterior log sd / prior log sd: round 1 {np.round(sd_1 / prior_sd, 3)}, "
        f"round 1+2 {np.round(sd_2 / prior_sd, 3)}\n"
        f"Laplace/NUTS sd at the truth (NUTS infers the noise): "
        f"round 1 {np.round(ratio_1, 2)}, round 1+2 {np.round(ratio_2, 2)}"
    )
    assert np.all(np.abs(z) < 3), dict(zip(order, np.asarray(z)))
    assert float((sd_2 / prior_sd).max()) < float((sd_1 / prior_sd).max())
