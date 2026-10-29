"""Local structural identifiability from the rank of the log sensitivities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from catalax.dataset.dataset import Dataset
from catalax.model.parameter import Identifiability
from catalax.model.simconfig import SimulationConfig
from catalax.tools.design_space import _latin_hypercube, _to_dataset, _validate_spec
from catalax.tools.priors import _extract_priors
from catalax.tools.sensitivity import _prepare_sensitivity_function

if TYPE_CHECKING:
    from catalax.model.model import Model

_METHOD = "local rank of log sensitivities"
_NULL_LOADING = 0.1
"""A parameter loading a flat direction by more than this is not identifiable."""


@dataclass
class IdentifiabilityReport:
    """Whether noise-free data from a design pin down every free parameter.

    Attributes:
        identifiable: True if the sensitivities have full rank.
        rank: Numerical rank of the sensitivities, at most ``len(parameter_order)``.
        parameter_order: Names of the free parameters.
        condition_number: Largest over smallest singular value of the scaled
            sensitivities. Large but finite means identifiable in principle and
            hard to resolve in practice.
        null_directions: ``(len(parameter_order) - rank, len(parameter_order))``
            unit vectors in ``log(theta)`` along which noise-free data do not
            change. ``+0.71 kcat -0.71 km`` means only ``kcat / km`` is determined.
        n_arms: Arms of the design that was tested; with ``spec``, the smallest
            number that reached full rank, or ``max_arms`` if none did.
        n_valid: Prior draws whose simulation solved.
        n_draws: Prior draws tried.
    """

    identifiable: bool
    rank: int
    parameter_order: list[str]
    condition_number: float
    null_directions: np.ndarray
    n_arms: int
    n_valid: int
    n_draws: int

    def __str__(self) -> str:
        n_free = len(self.parameter_order)
        arms = f"{self.n_arms} arm{'s' if self.n_arms > 1 else ''}"
        verdict = "identifiable" if self.identifiable else "NOT identifiable"
        lines = [
            f"{verdict} with {arms}: rank {self.rank}/{n_free}, condition number "
            f"{self.condition_number:.1e} ({self.n_valid}/{self.n_draws} draws solved)"
        ]
        for direction in self.null_directions:
            terms = [
                f"{v:+.2f} {name}"
                for name, v in zip(self.parameter_order, direction)
                if abs(v) > _NULL_LOADING
            ]
            lines.append("  flat direction (log): " + "  ".join(terms))
        return "\n".join(lines)


def check(
    model: Model,
    *,
    key: jax.Array,
    design: Dataset | None = None,
    spec: Mapping[str, tuple[float, float]] | None = None,
    times: Sequence[float] | jax.Array | None = None,
    max_arms: int = 5,
    n_draws: int = 8,
    rtol: float = 1e-6,
    config: SimulationConfig | None = None,
) -> IdentifiabilityReport:
    """Tests whether noise-free data pin down every free parameter.

    Stacks the sensitivities ``dy/dlog(theta)`` of every observable state at every
    sampling time of every arm into one matrix, each observable divided by its
    largest value so that units cancel, and counts its singular values above
    ``rtol`` times the largest. Full rank at one parameter point means full rank
    at almost every point, so a few prior draws suffice and the most
    informative one decides. Noise plays no part: it rescales rows and cannot
    change the rank.

    Give either ``design``, to test that experiment, or ``spec`` and ``times``,
    to find the fewest arms with random initial conditions inside the box that
    reach full rank. Random arms are the best case; a hand-made design can fall
    short of it, e.g. when it never varies an inhibitor.

    Writes the result to ``model.parameters[name].identifiability`` for every
    free parameter: a parameter that loads a flat direction is "not
    identifiable", every other one "identifiable".

    Args:
        model: Model whose free parameters (``constant=False``) carry priors with
            strictly positive support.
        key: PRNG key for the prior draws and, with ``spec``, the arms.
        design: Initial-condition-only dataset, one measurement per arm.
        spec: ``name -> (low, high)`` for every state and constant, as in
            ``catalax.doe.optimize_design``; ``low == high`` fixes the value.
        times: Sampling times of every arm, required with ``spec``.
        max_arms: Most arms to try with ``spec``.
        n_draws: Prior draws to test at.
        rtol: Singular values below ``rtol`` times the largest count as zero. The
            sensitivities are integrated to about 1e-8, so smaller values report
            solver error as information.
        config: Solver configuration, see ``catalax.doe.fisher_information``.

    Returns:
        The report.

    Raises:
        ValueError: If not exactly one of ``design`` and ``spec`` is given,
            ``spec`` comes without ``times``, or the spec, the priors or the
            design are invalid.
        RuntimeError: If no prior draw solved.
    """
    if (design is None) == (spec is None):
        raise ValueError("Give either `design` or `spec`, not both or neither.")
    draw_key, arm_key = jax.random.split(key)

    if design is not None:
        report = _rank(model, design, draw_key, n_draws, rtol, config)
    else:
        if times is None:
            raise ValueError("`spec` needs `times`, the sampling times of every arm.")
        names = model.get_state_order() + model.get_constants_order()
        _validate_spec(spec, names)
        free = [name for name in names if spec[name][0] < spec[name][1]]
        low = np.array([spec[name][0] for name in free])
        high = np.array([spec[name][1] for name in free])
        fixed = {name: spec[name][0] for name in names}
        times = np.asarray(times, dtype=float).tolist()
        for n_arms, k in enumerate(jax.random.split(arm_key, max_arms), start=1):
            unit = np.asarray(_latin_hypercube(k, n_arms, len(free)))
            arms = [fixed | dict(zip(free, low + (high - low) * u)) for u in unit]
            design = _to_dataset(model, arms, times)
            report = _rank(model, design, draw_key, n_draws, rtol, config)
            if report.identifiable:
                break

    _write_back(model, report)
    return report


def _rank(
    model: Model,
    design: Dataset,
    key: jax.Array,
    n_draws: int,
    rtol: float,
    config: SimulationConfig | None,
) -> IdentifiabilityReport:
    """The rank test on one design, at the most informative of ``n_draws``."""
    sensitivities, free, arrays = _prepare_sensitivity_function(
        model, design, config=config
    )
    priors = _extract_priors(model, free)
    thetas = jnp.stack(
        [
            p.sample(k, (n_draws,))
            for p, k in zip(priors, jax.random.split(key, len(free)))
        ],
        axis=-1,
    )

    def singular(theta: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        y, S_log, finite = sensitivities(theta, *arrays)
        scale = jnp.max(jnp.abs(y), axis=(0, 1))
        scale = jnp.where(scale > 0, scale, 1.0)  # an observable that stays at 0
        J = (S_log / scale[:, None]).reshape(-1, len(free))
        _, s, Vt = jnp.linalg.svd(J, full_matrices=False)
        return s, Vt, jnp.all(finite)

    s, Vt, valid = jax.vmap(singular)(thetas)
    s, Vt, valid = np.asarray(s), np.asarray(Vt), np.asarray(valid)
    if not valid.any():
        raise RuntimeError(
            f"None of the {n_draws} prior draws solved. Narrow the priors, raise "
            "`config.max_steps`, or use a stiff solver "
            "(e.g. `config.solver = diffrax.Kvaerno5`)."
        )

    ratio = np.where(valid, s[:, -1] / np.maximum(s[:, 0], np.finfo(float).tiny), -1)
    best = int(np.argmax(ratio))
    rank = int(np.sum(s[best] > rtol * s[best, 0]))
    return IdentifiabilityReport(
        identifiable=rank == len(free),
        rank=rank,
        parameter_order=free,
        condition_number=float(1 / ratio[best]) if ratio[best] > 0 else np.inf,
        null_directions=Vt[best, rank:],
        n_arms=len(design.measurements),
        n_valid=int(valid.sum()),
        n_draws=n_draws,
    )


def _write_back(model: Model, report: IdentifiabilityReport) -> None:
    """Stores the verdict on each free parameter of ``model``."""
    flat = np.abs(report.null_directions).max(axis=0, initial=0.0) > _NULL_LOADING
    for name, is_flat in zip(report.parameter_order, flat):
        model.parameters[name].identifiability = Identifiability(
            result="not identifiable" if is_flat else "identifiable",
            method=_METHOD,
            package="catalax",
        )
