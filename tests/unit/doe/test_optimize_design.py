"""optimize_design on a one-substrate Michaelis-Menten model, 16 prior draws."""

import jax
import numpy as np
import pytest

import catalax as ctx
import catalax.doe as cdoe
import catalax.mcmc as cmm

pytestmark = pytest.mark.usefixtures("x64")

TIMES = [10.0, 40.0, 80.0, 150.0, 300.0]
NOISE = cdoe.Proportional(cv=0.05, floor=0.5)
SEARCH = {
    "n_arms": 2,
    "times": TIMES,
    "n_restarts": 4,
    "n_steps": 50,
    "n_draws": 16,
    "n_rank_draws": 16,
}
SPEC = {"s": (10.0, 1000.0), "p": (0.0, 0.0), "e": (0.1, 0.1)}


def _mm_model(enzyme: str = "state") -> ctx.Model:
    """Michaelis-Menten with ``e`` as an unobservable zero-rate state or a constant."""
    model = ctx.Model(name="Michaelis-Menten")
    if enzyme == "state":
        model.add_state("s, p, e")
        model.add_ode("e", "0", observable=False)
    else:
        model.add_state("s, p")
        model.add_constant("e")
    model.add_ode("s", "-kcat * e * s / (k_m + s)", observable=True)
    model.add_ode("p", "kcat * e * s / (k_m + s)", observable=False)
    model.parameters["kcat"].value = 10.0
    model.parameters["kcat"].prior = cmm.priors.Uniform(low=5.0, high=20.0)
    model.parameters["k_m"].value = 100.0
    model.parameters["k_m"].prior = cmm.priors.Uniform(low=50.0, high=200.0)
    return model


def _initial_conditions(dataset: ctx.Dataset, name: str) -> np.ndarray:
    return np.array([m.initial_conditions[name] for m in dataset.measurements])


def test_fixed_columns_are_exact_and_free_columns_in_bounds():
    # A fixed state (p) and a free constant (e) take the two scatter paths.
    model = _mm_model("constant")
    spec = {"s": (10.0, 1000.0), "p": (5.0, 5.0), "e": (0.05, 0.5)}
    result = cdoe.optimize_design(
        model, spec, NOISE, key=jax.random.PRNGKey(0), **SEARCH
    )
    np.testing.assert_array_equal(_initial_conditions(result.dataset, "p"), 5.0)
    for name in ("s", "e"):
        values = _initial_conditions(result.dataset, name)
        assert np.all((values > spec[name][0]) & (values < spec[name][1])), name
    assert len(result.restart_scores) == SEARCH["n_restarts"]
    assert np.all(np.isfinite(result.restart_scores))


def test_optimum_beats_the_midpoint_design():
    model = _mm_model()
    result = cdoe.optimize_design(
        model, SPEC, NOISE, key=jax.random.PRNGKey(1), **SEARCH
    )
    midpoint = ctx.Dataset.from_model(model)
    for _ in range(SEARCH["n_arms"]):
        midpoint.add_initial(time=TIMES, s=505.0, p=0.0, e=0.1)

    key = jax.random.PRNGKey(2)
    optimum = cdoe.evaluate_design(model, result.dataset, NOISE, key=key, n_draws=16)
    baseline = cdoe.evaluate_design(model, midpoint, NOISE, key=key, n_draws=16)
    print(f"\noptimum {optimum.maximin:.4f}, midpoint {baseline.maximin:.4f}")
    assert optimum.maximin >= baseline.maximin


def test_same_key_gives_the_same_design():
    model = _mm_model()
    first, second = (
        cdoe.optimize_design(model, SPEC, NOISE, key=jax.random.PRNGKey(3), **SEARCH)
        for _ in range(2)
    )
    np.testing.assert_array_equal(
        _initial_conditions(first.dataset, "s"),
        _initial_conditions(second.dataset, "s"),
    )
    assert first.restart_scores == second.restart_scores


def test_dataset_round_trips_through_evaluate_design():
    model = _mm_model()
    result = cdoe.optimize_design(
        model, SPEC, NOISE, key=jax.random.PRNGKey(4), **SEARCH
    )
    assert len(result.dataset.measurements) == SEARCH["n_arms"]
    for measurement in result.dataset.measurements:
        np.testing.assert_array_equal(measurement.time, TIMES)

    # The report is evaluate_design on the report key, which is the last split.
    report_key = jax.random.split(jax.random.PRNGKey(4), 4)[3]
    report = cdoe.evaluate_design(
        model, result.dataset, NOISE, key=report_key, n_draws=16
    )
    assert report == result.report


@pytest.mark.parametrize(
    ("spec", "kwargs", "match"),
    [
        ({"s": (10.0, 1000.0), "p": (0.0, 0.0)}, {}, r"\['e'\]"),
        ({**SPEC, "x": (0.0, 1.0)}, {}, r"\['x'\]"),
        ({**SPEC, "s": (1000.0, 10.0)}, {}, r"spec\['s'\]"),
        ({**SPEC, "s": (10.0, 10.0)}, {}, "evaluate_design"),
        (SPEC, {"n_restarts": 0}, "n_restarts"),
    ],
    ids=["missing", "unknown", "low>high", "no-free", "n_restarts"],
)
def test_invalid_input_names_the_culprit(spec, kwargs, match):
    with pytest.raises(ValueError, match=match):
        cdoe.optimize_design(
            _mm_model(),
            spec,
            NOISE,
            key=jax.random.PRNGKey(0),
            **{**SEARCH, **kwargs},
        )


def test_every_draw_failing_raises():
    config = ctx.SimulationConfig(t1=max(TIMES), max_steps=2, throw=False)
    with pytest.raises(RuntimeError, match="max_steps"):
        cdoe.optimize_design(
            _mm_model(),
            SPEC,
            NOISE,
            key=jax.random.PRNGKey(0),
            config=config,
            **SEARCH,
        )
