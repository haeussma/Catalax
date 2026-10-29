"""Sensitivities of the observable states to the log of the free parameters.

Taken in log space, so they are dimensionless and do not depend on the units a
parameter is reported in. ``catalax.doe`` whitens them into Fisher information;
``catalax.identifiability`` takes their rank.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from catalax.dataset.dataset import Dataset
from catalax.model.inaxes import InAxes
from catalax.model.simconfig import SimulationConfig
from catalax.tools.simulation import Simulation

if TYPE_CHECKING:
    from catalax.model.model import Model

type SensitivityFunction = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array],
]
"""``(theta_free, y0s, constants, times) -> (y, S_log, finite)``."""
type DesignArrays = tuple[jax.Array, jax.Array, jax.Array]
"""``(y0s, constants, times)``, one row per arm, as read from a design dataset."""


def _prepare_sensitivity_function(
    model: Model,
    design: Dataset,
    *,
    config: SimulationConfig | None,
) -> tuple[SensitivityFunction, list[str], DesignArrays]:
    """Builds ``(theta_free, y0s, constants, times) -> (y, S_log, finite)``.

    ``y`` is ``(arms, times, observed)``, ``S_log = dy/dlog(theta_free)`` is
    ``(arms, times, observed, free)`` and ``finite`` flags the points that solved.
    Points that did not solve are zero in ``y`` and ``S_log``. The function is
    traceable, ``vmap``-able over ``theta_free`` and differentiable with respect
    to the design arrays. The design is validated here, outside traced code.

    Args:
        model: Model to simulate. Only parameters with ``constant=False`` are
            differentiated; constant ones enter at their ``value``.
        design: Dataset of initial-condition-only measurements, one per
            experimental arm, built with ``Dataset.add_initial(time=..., **ics)``.
        config: Solver configuration, or None for Tsit5 with tight tolerances
            (rtol = atol = 1e-8, max_steps = 4096, throw = False).

    Returns:
        The sensitivity function, taking free parameter values in natural units
        and the design arrays; the free parameter order; and the design's
        ``(y0s, constants, times)``.

    Raises:
        ValueError: If the model has no free parameters or observable states, a
            constant parameter has no positive value, or the design is incomplete.
    """
    if not jax.config.jax_enable_x64:
        warnings.warn(
            "catalax is computing sensitivities in float32; information matrices "
            "built from them are often too ill-conditioned for it. Call "
            "catalax.enable_x64() first.",
            stacklevel=4,
        )

    model = model._replace_assignments()
    parameter_order = model.get_parameter_order()
    free = [name for name in parameter_order if not model.parameters[name].constant]
    if not free:
        raise ValueError(
            "The model has no free parameters. Set `constant=False` on the "
            "parameters the design should inform."
        )
    fixed = [name for name in parameter_order if name not in free]
    base = (
        jnp.zeros(len(parameter_order))
        .at[jnp.array([parameter_order.index(name) for name in fixed], dtype=int)]
        .set(_extract_values(model, fixed))
    )
    free_index = jnp.array([parameter_order.index(name) for name in free])

    observed = model.get_observable_state_order(as_indices=True)
    if not observed:
        raise ValueError("The model has no observable states; nothing is measured.")

    y0s, constants, times = _extract_design_arrays(model, design)
    if config is None:
        config = SimulationConfig(
            t1=float(times.max()), rtol=1e-8, atol=1e-8, max_steps=4096, throw=False
        )

    simulate, _ = Simulation(
        sim_input=model.sim_input,
        config=config,
        sensitivity=InAxes.PARAMETERS,
        sensitivity_method="forward",
    )._prepare_func(in_axes=(0, None, 0, 0))

    def sensitivities(
        theta_free: jax.Array,
        y0s: jax.Array,
        constants: jax.Array,
        times: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        theta = base.at[free_index].set(theta_free)
        y, S = simulate(y0s, theta, constants, times)
        y, S = y[..., observed], S[..., observed, :]
        S_log = S[..., free_index] * theta_free  # chain rule: dy/dlog(t) = t dy/dt

        finite = jnp.isfinite(y) & jnp.all(jnp.isfinite(S_log), axis=-1)
        y = jnp.where(finite, y, 0.0)
        S_log = jnp.where(finite[..., None], S_log, 0.0)
        return y, S_log, finite

    return sensitivities, free, (y0s, constants, times)


def _extract_values(model: Model, names: list[str]) -> jax.Array:
    """Positive parameter values, in the order of ``names``.

    Raises:
        ValueError: If a value is missing or not strictly positive.
    """
    values = [model.parameters[name].value for name in names]
    for name, value in zip(names, values):
        if value is None:
            raise ValueError(
                f"Parameter '{name}' has no value. Set "
                f"model.parameters['{name}'].value before scoring a design."
            )
        if value <= 0:
            raise ValueError(
                f"Parameter '{name}' has value {value}, but sensitivities are taken "
                "in log space and need strictly positive parameters."
            )
    return jnp.array(values, dtype=float)


def _extract_design_arrays(
    model: Model, design: Dataset
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Reads ``(y0s, constants, times)`` from an initial-condition-only dataset.

    Raises:
        ValueError: If the design has no arms, an arm lacks initial conditions or
            sampling times, or the arms have different numbers of times.
    """
    if not design.measurements:
        raise ValueError(
            "The design has no measurements. Add one arm per experiment with "
            "design.add_initial(time=..., **initial_conditions)."
        )

    state_order = model.get_state_order()
    constant_order = model.get_constants_order()
    for meas in design.measurements:
        missing = [
            name
            for name in state_order + constant_order
            if name not in meas.initial_conditions
        ]
        if missing:
            raise ValueError(
                f"Measurement '{meas.id}' has no initial condition for {missing}. "
                "Pass every state and constant to design.add_initial(...)."
            )

    times = design.to_time_matrix().astype(float)
    if not bool(jnp.all(times[:, 0] >= 0)) or not bool(jnp.all(jnp.diff(times) >= 0)):
        raise ValueError(
            "Sampling times must be non-negative and increasing; simulations "
            "start at t = 0."
        )

    return (
        design.to_y0_matrix(state_order),
        design.to_y0_matrix(constant_order),
        times,
    )
