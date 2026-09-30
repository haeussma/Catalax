"""Fisher information of an experimental design.

Sensitivities are taken with respect to the **log** of the free parameters, so
the information is dimensionless and a design's score does not depend on the
units a parameter is reported in.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from catalax.dataset.dataset import Dataset
from catalax.doe.noise import NoiseModel, as_noise_model
from catalax.model.inaxes import InAxes
from catalax.model.simconfig import SimulationConfig
from catalax.tools.simulation import Simulation

if TYPE_CHECKING:
    from catalax.model.model import Model

type InformationFunction = Callable[[jax.Array], tuple[jax.Array, jax.Array]]


@dataclass
class Information:
    """Fisher information of a design at one parameter vector.

    Attributes:
        matrix: Symmetric ``(n_free, n_free)`` information in log parameters.
            Always finite; points whose solve failed contribute nothing.
        parameter_order: Names of the free parameters, the row/column order.
        valid: False if any observed point failed to solve.
    """

    matrix: jax.Array
    parameter_order: list[str]
    valid: bool


def fisher_information(
    model: Model,
    design: Dataset,
    noise: NoiseModel | float,
    *,
    config: SimulationConfig | None = None,
) -> Information:
    """Computes the Fisher information of a design at the model's parameter values.

    ``F = sum W.T @ W`` over measurements, sampling times and observable states,
    with ``W = noise.whiten(dy/dlog(theta), y)``. Only parameters with
    ``constant=False`` are rows/columns; constant ones enter at their ``value``.

    Args:
        model: Model whose parameter values the information is linearised around.
        design: Dataset of initial-condition-only measurements, one per
            experimental arm, built with ``Dataset.add_initial(time=..., **ics)``.
            ``time`` holds the sampling times.
        noise: Observation noise model; a bare float means ``Homoskedastic``.
        config: Solver configuration. Defaults to tight tolerances
            (rtol = atol = 1e-8, max_steps = 4096, throw = False).

    Returns:
        The information matrix, its parameter order and a validity flag.

    Raises:
        ValueError: If a parameter value is missing or not positive, or the
            design is incomplete.
    """
    information, parameter_order = _prepare_information_function(
        model, design, noise, config=config
    )
    theta = _extract_values(model, parameter_order)
    matrix, valid = information(theta)
    return Information(
        matrix=matrix, parameter_order=parameter_order, valid=bool(valid)
    )


def _prepare_information_function(
    model: Model,
    design: Dataset,
    noise: NoiseModel | float,
    *,
    config: SimulationConfig | None,
) -> tuple[InformationFunction, list[str]]:
    """Builds ``theta_free -> (F, valid)``, traceable and ``vmap``-able.

    Args:
        model: Model to simulate.
        design: Design dataset.
        noise: Observation noise model.
        config: Solver configuration, or None for the defaults.

    Returns:
        The information function, taking free parameter values in natural units,
        and the free parameter order.
    """
    if not jax.config.jax_enable_x64:
        warnings.warn(
            "catalax.doe is running in float32; Fisher information matrices are "
            "often too ill-conditioned for it. Call catalax.enable_x64() first.",
            stacklevel=3,
        )

    model = model._replace_assignments()
    noise = as_noise_model(noise)
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

    in_axes = (0, None, 0, 0)
    simulate, _ = Simulation(sim_input=model.sim_input, config=config)._prepare_func(
        in_axes=in_axes
    )
    sensitivities, _ = Simulation(
        sim_input=model.sim_input, config=config, sensitivity=InAxes.PARAMETERS
    )._prepare_func(in_axes=in_axes)

    def information(theta_free: jax.Array) -> tuple[jax.Array, jax.Array]:
        theta = base.at[free_index].set(theta_free)
        y = simulate(y0s, theta, constants, times)[..., observed]
        S = sensitivities(y0s, theta, constants, times)[..., observed, :]
        S_log = S[..., free_index] * theta_free  # chain rule: dy/dlog(t) = t dy/dt

        finite = jnp.isfinite(y) & jnp.all(jnp.isfinite(S_log), axis=-1)
        y = jnp.where(finite, y, 0.0)
        S_log = jnp.where(finite[..., None], S_log, 0.0)

        W = noise.whiten(S_log, y).reshape(-1, len(free))
        F = W.T @ W
        return 0.5 * (F + F.T), jnp.all(finite)

    return information, free


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
