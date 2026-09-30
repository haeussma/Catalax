"""catalax.doe against the closed-form MAT model (Julia-validated, progress-doe)."""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import catalax as ctx
import catalax.doe as cdoe

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "doe_mat_oracle.json"

pytestmark = pytest.mark.usefixtures("x64")


def _load_oracle() -> dict:
    if not FIXTURE.exists():
        pytest.skip(f"oracle fixture {FIXTURE} is missing")
    return json.loads(FIXTURE.read_text())


def _mat_model(meta: dict, theta: dict[str, float], enzyme: str) -> ctx.Model:
    """MAT with ``E`` as a constant, or as an unobservable state with zero rate."""
    rate = meta["rate_law"]
    model = ctx.Model(name="MAT")
    if enzyme == "constant":
        model.add_state("A, B, P")
        model.add_constant("E")
    else:
        model.add_state("A, B, E, P")
        model.add_ode("E", "0", observable=False)
    model.add_ode("A", f"-({rate})", observable=False)
    model.add_ode("B", f"-({rate})", observable=False)
    model.add_ode("P", rate, observable=True)
    for name, value in theta.items():
        model.parameters[name].value = value
    return model


def _noise(spec: dict) -> cdoe.NoiseModel:
    if spec["type"] == "homoskedastic":
        return cdoe.Homoskedastic(spec["sigma"])
    return cdoe.Proportional(cv=spec["cv"], floor=spec["floor"])


def _design(model: ctx.Model, meta: dict, case: dict) -> ctx.Dataset:
    design = ctx.Dataset.from_model(model)
    for row in case["initial_conditions"]:
        design.add_initial(time=case["times"], **dict(zip(meta["ic_columns"], row)))
    return design


@pytest.mark.parametrize("enzyme", ["constant", "state"])
def test_oracle_matches_closed_form(enzyme: str):
    oracle = _load_oracle()
    meta = oracle["meta"]
    curve_errors, fisher_errors, direction_errors = [], [], []

    for case in oracle["cases"]:
        model = _mat_model(meta, case["theta"], enzyme)
        design = _design(model, meta, case)
        assert model.get_parameter_order() == meta["param_order"]
        assert model.get_observable_state_order() == [meta["observable"]]

        predicted = model.simulate(
            design,
            ctx.SimulationConfig(t1=max(case["times"]), rtol=1e-8, atol=1e-8),
            saveat=jnp.asarray(case["times"]),
        )
        curves = np.stack([np.asarray(m.data["P"]) for m in predicted.measurements])
        ref_curves = np.asarray(case["curves"])
        curve_errors.append(
            np.max(np.abs(curves - ref_curves) / np.maximum(np.abs(ref_curves), 1e-12))
        )

        info = cdoe.fisher_information(model, design, _noise(case["noise"]))
        assert info.valid
        assert info.parameter_order == meta["param_order"]
        F = np.asarray(info.matrix)
        F_ref = np.asarray(case["fisher"])
        fisher_errors.append(np.linalg.norm(F - F_ref) / np.linalg.norm(F_ref))

        # Frobenius is dominated by the best-informed direction; this is not.
        L_inv = np.linalg.inv(np.linalg.cholesky(F_ref))
        direction_errors.append(np.linalg.norm(L_inv @ (F - F_ref) @ L_inv.T, ord=2))

    print(
        f"\n[E as {enzyme}] max curve rel err {max(curve_errors):.3e}, "
        f"max Fisher Frobenius rel err {max(fisher_errors):.3e}, "
        f"max whitened spectral err {max(direction_errors):.3e}"
    )
    assert max(curve_errors) <= 1e-6
    assert max(fisher_errors) <= 1e-6
    assert max(direction_errors) <= 1e-5
