"""The rank test finds confounded parameters and the arms a model needs."""

import jax
import numpy as np
import pytest

import catalax as ctx
import catalax.identifiability as cid
import catalax.mcmc as cmm

pytestmark = pytest.mark.usefixtures("x64")

RATE = "kcat * E * A * B / ((km_a + A) * (km_b + B))"
TIMES = [1.0, 5.0, 20.0, 60.0]
KEY = jax.random.PRNGKey(0)


def _model(enzyme_known: bool = True) -> ctx.Model:
    model = ctx.Model(name="Two-substrate Michaelis-Menten")
    model.add_state("A, B, P")
    if enzyme_known:
        model.add_constant("E")
    model.add_ode("A", f"-({RATE})", observable=False)
    model.add_ode("B", f"-({RATE})", observable=False)
    model.add_ode("P", RATE, observable=True)
    priors = {"kcat": (2.0, 50.0), "km_a": (0.1, 5.0), "km_b": (0.1, 5.0)}
    if not enzyme_known:
        priors["E"] = (0.01, 1.0)
    for name, (low, high) in priors.items():
        model.parameters[name].prior = cmm.priors.Uniform(low=low, high=high)
    return model


def _design(model: ctx.Model, arms: list[dict], times=TIMES) -> ctx.Dataset:
    design = ctx.Dataset.from_model(model)
    for arm in arms:
        design.add_initial(time=times, **arm)
    return design


def test_identifiable_design_writes_back():
    model = _model()
    design = _design(model, [{"A": 1.0, "B": 2.0, "P": 0.0, "E": 0.1}])

    report = cid.check(model, design=design, key=KEY)

    assert report.identifiable and report.rank == 3
    assert report.null_directions.shape == (0, 3)
    assert all(
        model.parameters[name].identifiability.result == "identifiable"
        for name in report.parameter_order
    )


def test_unknown_enzyme_confounds_kcat():
    model = _model(enzyme_known=False)
    arms = [{"A": a, "B": b, "P": 0.0} for a, b in [(0.5, 5.0), (5.0, 0.5)]]

    report = cid.check(model, design=_design(model, arms), key=KEY)

    assert not report.identifiable and report.rank == 3
    flat = dict(zip(report.parameter_order, report.null_directions[0]))
    # Output depends on kcat * E only: up in one, down in the other.
    np.testing.assert_allclose(abs(flat["kcat"]), 2**-0.5, atol=1e-4)
    np.testing.assert_allclose(flat["kcat"], -flat["E"], atol=1e-4)
    assert model.parameters["E"].identifiability.result == "not identifiable"
    assert model.parameters["km_a"].identifiability.result == "identifiable"


def test_one_sampling_time_cannot_resolve_three_parameters():
    model = _model()
    design = _design(model, [{"A": 1.0, "B": 2.0, "P": 0.0, "E": 0.1}], [60.0])

    report = cid.check(model, design=design, key=KEY)

    assert report.rank == 1 and not report.identifiable


def test_spec_finds_fewest_arms():
    model = _model(enzyme_known=False)
    spec = {"A": (0.1, 10.0), "B": (0.1, 10.0), "P": (0.0, 0.0)}

    report = cid.check(model, spec=spec, times=TIMES, max_arms=2, key=KEY)

    # kcat * E stays confounded however many arms, so the search runs to the end.
    assert not report.identifiable and report.n_arms == 2


@pytest.mark.parametrize("kwargs", [{}, {"design": "x", "spec": {}}, {"spec": {}}])
def test_rejects_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        cid.check(_model(), key=KEY, **kwargs)
