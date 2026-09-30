"""Forward sensitivities (integrated as ODE states) against reverse mode."""

import jax.numpy as jnp
import numpy as np
import pytest

import catalax as ctx
from catalax.model.inaxes import InAxes
from catalax.tools.simulation import Simulation

pytestmark = pytest.mark.usefixtures("x64")

TIMES = [1.0, 5.0, 10.0, 30.0, 60.0]


def _ode_model() -> ctx.Model:
    model = ctx.Model(name="MM with product decay")
    model.add_state("s, p")
    model.add_constant("e")
    model.add_ode("s", "-kcat * e * s / (k_m + s)", observable=True)
    model.add_ode("p", "kcat * e * s / (k_m + s) - k_d * p", observable=False)
    return model


def _reaction_model() -> ctx.Model:
    model = ctx.Model(name="MM with product decay, as reactions")
    model.add_state("s, p, q")
    model.add_constant("e")
    model.add_reaction("s -> p", symbol="r1", equation="kcat * e * s / (k_m + s)")
    model.add_reaction("p -> q", symbol="r2", equation="k_d * p")
    return model


def _simulator(model: ctx.Model, method: str | None) -> tuple:
    config = ctx.SimulationConfig(t1=max(TIMES), rtol=1e-10, atol=1e-10)
    if method is None:
        simulation = Simulation(sim_input=model.sim_input, config=config)
    else:
        simulation = Simulation(
            sim_input=model.sim_input,
            config=config,
            sensitivity=InAxes.PARAMETERS,
            sensitivity_method=method,
        )
    return simulation._prepare_func(in_axes=(0, None, 0, 0))[0]


@pytest.mark.parametrize("build", [_ode_model, _reaction_model])
def test_forward_matches_reverse(build):
    model = build()
    assert model.get_parameter_order() == ["k_d", "k_m", "kcat"]
    n_states = len(model.get_state_order())
    y0s = jnp.array([[50.0, 0.0, 0.0], [400.0, 10.0, 0.0]])[:, :n_states]
    theta = jnp.array([0.05, 100.0, 10.0])
    constants = jnp.array([[0.1], [0.3]])
    times = jnp.tile(jnp.array(TIMES), (2, 1))

    ys, S = _simulator(model, "forward")(y0s, theta, constants, times)
    ys_ref = _simulator(model, None)(y0s, theta, constants, times)
    S_ref = _simulator(model, "reverse")(y0s, theta, constants, times)

    assert S.shape == (2, len(TIMES), n_states, 3)
    np.testing.assert_allclose(ys, ys_ref, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(
        S, S_ref, rtol=1e-6, atol=1e-6 * float(jnp.abs(S_ref).max())
    )


@pytest.mark.parametrize(
    "sensitivity", [None, InAxes.Y0, InAxes.CONSTANTS, InAxes.TIME]
)
def test_forward_supports_only_parameters(sensitivity):
    simulation = Simulation(
        sim_input=_ode_model().sim_input,
        config=ctx.SimulationConfig(t1=1.0),
        sensitivity=sensitivity,
        sensitivity_method="forward",
    )
    with pytest.raises(NotImplementedError, match="InAxes.PARAMETERS"):
        simulation._prepare_func()
