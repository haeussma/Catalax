"""Tests for ctx.optimize: a least-squares fit on the observed states of a model."""

import jax.numpy as jnp
import numpy as np
import pytest

import catalax as ctx

NOISE = jnp.array([0.0, 0.3, -0.2, 0.9, -0.4, 0.1])


def _bound(model: ctx.Model, name: str, initial: float) -> None:
    parameter = model.parameters[name]
    parameter.initial_value = initial
    parameter.lower_bound = 1e-3
    parameter.upper_bound = 10.0


def linear_problem() -> tuple[ctx.Model, ctx.Dataset]:
    """P' = k, P(0) = 0, sampled at t = 0, 2, ..., 10 with fixed noise; true k = 0.5."""
    model = ctx.Model(name="linear")
    model.add_state("P")
    model.add_ode("P", "k")
    model.parameters["k"].value = 0.5
    _bound(model, "k", initial=0.1)

    dataset = ctx.Dataset.from_model(model)
    dataset.add_initial(P=0.0)
    dataset = model.simulate(
        dataset=dataset, config=ctx.SimulationConfig(t1=10, nsteps=6)
    )
    measurement = dataset.measurements[0]
    measurement.data["P"] = measurement.data["P"] + NOISE
    return model, dataset


def least_squares_slope(dataset: ctx.Dataset) -> float:
    """Closed-form least-squares k for P = k·t over the measured (non-NaN) points."""
    measurement = dataset.measurements[0]
    t, y = np.asarray(measurement.time), np.asarray(measurement.data["P"])
    keep = np.isfinite(y)
    return float(t[keep] @ y[keep] / (t[keep] @ t[keep]))


def least_squares_stderr(dataset: ctx.Dataset) -> float:
    """Closed-form standard error of k, with n - 1 degrees of freedom over measured points."""
    measurement = dataset.measurements[0]
    t, y = np.asarray(measurement.time), np.asarray(measurement.data["P"])
    keep = np.isfinite(y)
    t, y = t[keep], y[keep]
    k = t @ y / (t @ t)
    return float(np.sqrt(np.sum((y - k * t) ** 2) / (len(t) - 1) / (t @ t)))


def test_default_fit_is_ordinary_least_squares():
    model, dataset = linear_problem()
    result, fitted = ctx.optimize(model, dataset)
    assert result.params["k"].value == pytest.approx(
        least_squares_slope(dataset), rel=1e-4
    )
    assert fitted.parameters["k"].value == pytest.approx(result.params["k"].value)


def test_standard_error_matches_least_squares():
    """Fails in float32 if finite differences use scipy's default step (k stays near 0.1)."""
    model, dataset = linear_problem()
    result, _ = ctx.optimize(model, dataset)
    assert result.ndata == 6
    assert result.params["k"].stderr == pytest.approx(
        least_squares_stderr(dataset), rel=1e-2
    )


def test_squared_error_objective_matches_default():
    """A per-point loss is minimised as a sum: (y - p)^2 must give the least-squares k."""
    model, dataset = linear_problem()
    result, _ = ctx.optimize(
        model, dataset, objective_fun=lambda y_true, y_pred: (y_true - y_pred) ** 2
    )
    assert result.params["k"].value == pytest.approx(
        least_squares_slope(dataset), rel=1e-4
    )


def test_reducing_objective_is_rejected():
    """A loss that averages over points cannot be split into per-point residuals."""
    model, dataset = linear_problem()
    with pytest.raises(ValueError, match="one loss per data point"):
        ctx.optimize(model, dataset, objective_fun=ctx.mean_absolute_error)


def test_hidden_state_needs_no_data():
    """S -> P with only P measured: S has an initial condition but no data array."""
    model = ctx.Model(name="hidden substrate")
    model.add_state("S, P")
    model.add_ode("S", "-k * S", observable=False)
    model.add_ode("P", "k * S")
    model.parameters["k"].value = 0.3
    _bound(model, "k", initial=0.1)

    dataset = ctx.Dataset.from_model(model)
    dataset.add_initial(S=2.0, P=0.0)
    dataset = model.simulate(dataset=dataset, config=ctx.SimulationConfig(t1=10, nsteps=6))
    for measurement in dataset.measurements:
        measurement.data.pop("S")

    result, _ = ctx.optimize(model, dataset)
    assert result.params["k"].value == pytest.approx(0.3, rel=1e-3)


def test_nan_points_are_left_out():
    """A NaN data point is skipped: the fit equals least squares over the other points."""
    model, dataset = linear_problem()
    measurement = dataset.measurements[0]
    measurement.data["P"] = measurement.data["P"].at[3].set(jnp.nan)

    result, _ = ctx.optimize(model, dataset)

    assert result.ndata == 5  # dropped, not counted as a zero residual
    assert result.params["k"].value == pytest.approx(least_squares_slope(dataset), rel=1e-4)
    assert result.params["k"].stderr == pytest.approx(least_squares_stderr(dataset), rel=1e-2)
