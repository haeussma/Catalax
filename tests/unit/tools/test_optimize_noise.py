"""ctx.optimize(noise=...): weighted least squares with absolute standard deviations."""

import jax.numpy as jnp
import numpy as np
import pytest

import catalax as ctx

NOISE = jnp.array([0.0, 0.3, -0.2, 0.9, -0.4, 0.1])


def linear_problem() -> tuple[ctx.Model, ctx.Dataset]:
    """P' = k, P(0) = 0, sampled at t = 0, 2, ..., 10 with fixed noise; true k = 0.5.

    Same problem as tests/unit/tools/test_optimize.py (test directories are not packages,
    so it is repeated here rather than imported).
    """
    model = ctx.Model(name="linear")
    model.add_state("P")
    model.add_ode("P", "k")
    model.parameters["k"].value = 0.5
    parameter = model.parameters["k"]
    parameter.initial_value, parameter.lower_bound, parameter.upper_bound = (
        0.1,
        1e-3,
        10.0,
    )
    dataset = ctx.Dataset.from_model(model)
    dataset.add_initial(P=0.0)
    dataset = model.simulate(
        dataset=dataset, config=ctx.SimulationConfig(t1=10, nsteps=6)
    )
    dataset.measurements[0].data["P"] = dataset.measurements[0].data["P"] + NOISE
    return model, dataset


def _arrays(model, dataset):
    data, times, _ = dataset.to_jax_arrays(model.get_observable_state_order())
    return data, times


def _sigma(model, dataset):
    _, times = _arrays(model, dataset)
    return (0.1 + 0.05 * times)[..., None]  # grows with t, shape of the data


def _closed_form(model, dataset, sigma):
    """Weighted least squares for P = k t: estimate and absolute standard error."""
    data, times = _arrays(model, dataset)
    t, y, s = (np.asarray(a).ravel() for a in (times, data[..., 0], sigma))
    w = 1 / s**2
    return float(np.sum(w * t * y) / np.sum(w * t * t)), float(
        1 / np.sqrt(np.sum(w * t * t))
    )


def test_weighted_estimate_and_absolute_stderr():
    model, dataset = linear_problem()
    sigma = _sigma(model, dataset)
    result, _ = ctx.optimize(model, dataset, noise=sigma)
    k, se = _closed_form(model, dataset, sigma)
    assert result.params["k"].value == pytest.approx(k, rel=1e-4)
    assert result.params["k"].stderr == pytest.approx(se, rel=1e-2)


def test_stderr_follows_the_given_sigma():
    """Absolute SDs: ten times the noise means ten times the standard error, same estimate."""
    model, dataset = linear_problem()
    sigma = _sigma(model, dataset)
    small, _ = ctx.optimize(model, dataset, noise=sigma)
    large, _ = ctx.optimize(model, dataset, noise=10 * sigma)
    assert large.params["k"].value == pytest.approx(small.params["k"].value, rel=1e-4)
    assert large.params["k"].stderr == pytest.approx(
        10 * small.params["k"].stderr, rel=1e-2
    )


def test_noise_model_is_evaluated_on_the_data():
    model, dataset = linear_problem()
    noise = ctx.noise.Proportional(cv=0.1, floor=0.2)
    data, _ = _arrays(model, dataset)
    by_model, _ = ctx.optimize(model, dataset, noise=noise)
    by_array, _ = ctx.optimize(model, dataset, noise=noise.scale(data))
    assert by_model.params["k"].value == pytest.approx(
        by_array.params["k"].value, rel=1e-6
    )


def test_missing_points_are_left_out():
    model, dataset = linear_problem()
    dataset.measurements[0].data["P"] = (
        dataset.measurements[0].data["P"].at[3].set(jnp.nan)
    )
    sigma = _sigma(model, dataset)
    result, _ = ctx.optimize(model, dataset, noise=sigma)
    assert result.ndata == 5
    assert np.isfinite(result.params["k"].value)


def test_noise_and_objective_fun_conflict():
    model, dataset = linear_problem()
    with pytest.raises(ValueError, match="noise"):
        ctx.optimize(model, dataset, noise=0.1, objective_fun=lambda y, p: (y - p) ** 2)


def test_non_positive_sd_at_a_measured_point_raises():
    model, dataset = linear_problem()
    sigma = _sigma(model, dataset).at[0, 2, 0].set(0.0)
    with pytest.raises(ValueError, match="positive"):
        ctx.optimize(model, dataset, noise=sigma)
