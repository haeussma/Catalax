"""optimize_design against the closed-form MAT optimum from progress-doe."""

import json
import time
from pathlib import Path

import jax
import pytest

import catalax as ctx
import catalax.doe as cdoe
import catalax.mcmc as cmm

FIXTURE = (
    Path(__file__).resolve().parents[2] / "fixtures" / "doe_mat_design_oracle.json"
)

pytestmark = [pytest.mark.usefixtures("x64"), pytest.mark.expensive]


def _load_oracle() -> dict:
    if not FIXTURE.exists():
        pytest.skip(f"oracle fixture {FIXTURE} is missing")
    return json.loads(FIXTURE.read_text())


def _mat_model(meta: dict) -> ctx.Model:
    """MAT with the enzyme as an unobservable zero-rate state."""
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
    return model


def test_search_matches_the_closed_form_optimum():
    oracle = _load_oracle()
    meta = oracle["meta"]
    model = _mat_model(meta)
    noise = cdoe.Proportional(cv=meta["noise"]["cv"], floor=meta["noise"]["floor"])
    assert model.get_parameter_order() == meta["param_order"]
    assert model.get_state_order() == meta["ic_columns"]

    start = time.perf_counter()
    result = cdoe.optimize_design(
        model,
        {name: tuple(bounds) for name, bounds in meta["spec"].items()},
        noise,
        n_arms=meta["n_arms"],
        times=meta["times"],
        key=jax.random.PRNGKey(0),
        n_restarts=8,
        n_steps=300,
        n_draws=16,
    )
    elapsed = time.perf_counter() - start

    reference = ctx.Dataset.from_model(model)
    for row in oracle["design"]:
        reference.add_initial(time=meta["times"], **dict(zip(meta["ic_columns"], row)))

    key = jax.random.PRNGKey(1)
    ours = cdoe.evaluate_design(model, result.dataset, noise, key=key, n_draws=256)
    theirs = cdoe.evaluate_design(model, reference, noise, key=key, n_draws=256)

    arms = [
        {name: round(m.initial_conditions[name], 4) for name in meta["ic_columns"]}
        for m in result.dataset.measurements
    ]
    print(
        f"\nsearch {elapsed:.0f} s; restarts "
        f"{[round(s, 4) for s in result.restart_scores]}\n"
        f"catalax design {arms}\n"
        f"catalax scores: catalax design {ours.maximin:.4f}, progress-doe design "
        f"{theirs.maximin:.4f}, gap {ours.maximin - theirs.maximin:+.4f}"
    )
    assert ours.n_valid == theirs.n_valid == 256
    assert ours.maximin >= theirs.maximin - 0.01
