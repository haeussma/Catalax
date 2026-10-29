"""Fisher information of an experimental design.

Sensitivities are taken with respect to the **log** of the free parameters, so
the information is dimensionless and a design's score does not depend on the
units a parameter is reported in.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from catalax.dataset.dataset import Dataset
from catalax.noise import NoiseModel, as_noise_model
from catalax.model.simconfig import SimulationConfig
from catalax.tools.sensitivity import (
    DesignArrays,
    _extract_values,
    _prepare_sensitivity_function,
)

if TYPE_CHECKING:
    from catalax.model.model import Model

type InformationFunction = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]
]
"""``(theta_free, y0s, constants, times) -> (F, valid)``."""


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
        config: Solver configuration. Defaults to Tsit5 with tight tolerances
            (rtol = atol = 1e-8, max_steps = 4096, throw = False). Sensitivities
            are integrated alongside the states, so a stiff solver runs Newton on
            the whole augmented system: for one MAT arm (standalone diffrax probe),
            a design gradient took Kvaerno5 310 steps and 34 ms, Tsit5 75 steps
            and 2 ms.

    Returns:
        The information matrix, its parameter order and a validity flag.

    Raises:
        ValueError: If a parameter value is missing or not positive, or the
            design is incomplete.
    """
    information, parameter_order, arrays = _prepare_information_function(
        model, design, noise, config=config
    )
    theta = _extract_values(model, parameter_order)
    matrix, valid = information(theta, *arrays)
    return Information(
        matrix=matrix, parameter_order=parameter_order, valid=bool(valid)
    )


def _prepare_information_function(
    model: Model,
    design: Dataset,
    noise: NoiseModel | float,
    *,
    config: SimulationConfig | None,
) -> tuple[InformationFunction, list[str], DesignArrays]:
    """Builds ``(theta_free, y0s, constants, times) -> (F, valid)``.

    The function is traceable, ``vmap``-able over ``theta_free`` and
    differentiable with respect to the design arrays. The design is validated
    here, outside traced code.

    Args:
        model: Model to simulate.
        design: Design dataset.
        noise: Observation noise model.
        config: Solver configuration, or None for the defaults.

    Returns:
        The information function, taking free parameter values in natural units
        and the design arrays; the free parameter order; and the design's
        ``(y0s, constants, times)``.
    """
    noise = as_noise_model(noise, fixed=True)
    sensitivities, free, arrays = _prepare_sensitivity_function(
        model, design, config=config
    )

    def information(
        theta_free: jax.Array,
        y0s: jax.Array,
        constants: jax.Array,
        times: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        y, S_log, finite = sensitivities(theta_free, y0s, constants, times)
        W = noise.whiten(S_log, y).reshape(-1, len(free))
        F = W.T @ W
        return 0.5 * (F + F.T), jnp.all(finite)

    return information, free, arrays
