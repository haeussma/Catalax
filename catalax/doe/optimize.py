"""Search the initial conditions of a design for the best worst parameter."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.typing import ArrayLike

from catalax.dataset.dataset import Dataset
from catalax.doe.evaluate import DesignReport, _prepare_efficiency, evaluate_design
from catalax.doe.noise import NoiseModel
from catalax.model.simconfig import SimulationConfig

if TYPE_CHECKING:
    from catalax.model.model import Model

# A Latin-hypercube start on a stratum edge would map to z = +-inf.
_START_MARGIN = 1e-6


@dataclass
class DesignResult:
    """The best design found by ``optimize_design``.

    Attributes:
        dataset: One ``add_initial`` arm per experiment, fixed columns at their
            value, ready to run in the lab or pass to ``evaluate_design``.
        report: ``evaluate_design`` of ``dataset`` on fresh prior draws. The
            winner was selected on the ranking draws, which would bias its own
            score upwards there.
        restart_scores: Hard maximin of each restart's best point on the ranking
            draws, NaN if a ranking draw failed to solve. A tight spread means
            the restarts agree; a wide one means the landscape has several basins
            and the restarts were load-bearing.
    """

    dataset: Dataset
    report: DesignReport
    restart_scores: list[float]


def optimize_design(
    model: Model,
    spec: Mapping[str, tuple[float, float]],
    noise: NoiseModel | float,
    *,
    n_arms: int,
    times: Sequence[float] | jax.Array,
    key: jax.Array,
    n_restarts: int = 8,
    n_steps: int = 300,
    n_draws: int = 64,
    n_rank_draws: int = 256,
    learning_rate: float = 0.1,
    temperature: float = 0.01,
    config: SimulationConfig | None = None,
    belief: Mapping[str, ArrayLike] | None = None,
) -> DesignResult:
    """Finds the initial conditions that maximise the expected maximin efficiency.

    Runs ``n_restarts`` Adam ascents from Latin-hypercube starts on the smoothed
    maximin ``mean_i softmin_j e_ij`` over ``n_draws`` fixed prior draws, keeps
    each restart's best point along its trajectory, and ranks the restarts on
    the hard maximin ``mean_i min_j e_ij`` over ``n_rank_draws`` separate
    draws. Every restart and every step scores against the same draws, so the
    objective is a fixed surface and restarts compare draw for draw.

    Bounds are enforced by ``low + (high - low) * sigmoid(z)``, never a clip:
    optima often sit on a bound (an enzyme or stock ceiling), where a clip has
    exactly zero gradient. The sampling times are fixed; optimising them
    specialises the design toward the prior median and costs the tail.

    Args:
        model: Model whose free parameters (``constant=False``) all carry a
            prior with strictly positive support.
        spec: ``name -> (low, high)`` for every state and constant of the model,
            with no defaults. ``low == high`` fixes that initial condition at
            ``low`` in every arm; ``low < high`` lets each arm choose it inside
            the box.
        noise: Observation noise model; a bare float means ``Homoskedastic``.
        n_arms: Number of experiments in the design.
        times: Sampling times, shared by every arm.
        key: PRNG key for the starts, the ascent and ranking draws, and the
            report's draws.
        n_restarts: Number of ascents. The maximin landscape has several
            basins: in progress-doe 8 of 24 starts landed in the lower one, 14%
            short. 1 runs, but is not useful.
        n_steps: Adam steps per restart.
        n_draws: Prior draws the ascent averages over. Cost is linear in draws
            and in restarts. One ``value_and_grad`` on the bi-substrate model
            (4 states, 4 arms x 9 times, Tsit5, CPU, float64):

            ===== ======== ====================== =======================
            draws per step 8 restarts x 300 steps 24 restarts x 600 steps
            ===== ======== ====================== =======================
            16    27 ms    ~1 min                 6.4 min
            64    105 ms   ~4 min                 25 min
            256   499 ms   20 min                 2 h
            ===== ======== ====================== =======================

            The first compile takes a few seconds.
        n_rank_draws: Prior draws the ranking and the report average over. The
            ranking runs once per restart, so it can afford more draws than the
            ascent.
        learning_rate: Adam step size in the unconstrained coordinate ``z``,
            where the box spans roughly ``[-6, 6]``.
        temperature: Softmin temperature of the ascent, in efficiency units. It
            biases the smoothed value below the hard one by at most
            ``temperature * log(n_free)``; 0.01 costs under 0.001.
        config: Solver configuration, see ``fisher_information``.
        belief: Posterior samples of an earlier round to design against
            instead of the priors, e.g. ``run_mcmc(...).get_samples()``; see
            ``evaluate_design``. Every score is then relative to the belief.

    Returns:
        A ``DesignResult``.

    Raises:
        ValueError: If ``spec`` misses or adds a column, has ``low > high``, or
            fixes every column, or a count is below 1.
        RuntimeError: If every restart failed to solve on some ranking draw.
            Averaging over the draws that solved would let a failing design
            score higher, so such restarts are dropped rather than ranked.
    """
    names = model.get_state_order() + model.get_constants_order()
    _validate(spec, names, n_arms, n_restarts, n_steps, n_draws, n_rank_draws)
    free = [name for name in names if spec[name][0] < spec[name][1]]

    times = np.asarray(times, dtype=float).tolist()
    midpoint = {name: (low + high) / 2 for name, (low, high) in spec.items()}
    template = _to_dataset(model, [midpoint] * n_arms, times)
    start_key, ascent_key, rank_key, report_key = jax.random.split(key, 4)
    ascent_efficiency, _, (y0s, constants, _) = _prepare_efficiency(
        model,
        template,
        noise,
        key=ascent_key,
        n_draws=n_draws,
        config=config,
        belief=belief,
    )
    rank_efficiency, _, _ = _prepare_efficiency(
        model,
        template,
        noise,
        key=rank_key,
        n_draws=n_rank_draws,
        config=config,
        belief=belief,
    )

    # Free columns are set on the side-by-side [states | constants] template.
    n_states = len(model.get_state_order())
    template_ics = jnp.concatenate([y0s, constants], axis=1)
    columns = np.array([names.index(name) for name in free], dtype=int)
    low = jnp.array([spec[name][0] for name in free], dtype=float)
    high = jnp.array([spec[name][1] for name in free], dtype=float)

    def to_box(z: jax.Array) -> jax.Array:
        return low + (high - low) * jax.nn.sigmoid(z)

    def to_design(z: jax.Array) -> tuple[jax.Array, jax.Array]:
        ics = template_ics.at[:, columns].set(to_box(z))
        return ics[:, :n_states], ics[:, n_states:]

    def smoothed(z: jax.Array) -> jax.Array:
        e, valid = ascent_efficiency(*to_design(z))
        soft_min = -temperature * jax.nn.logsumexp(-e / temperature, axis=-1)
        # max(n_valid, 1): with every draw failed this is 0 with zero gradient,
        # not NaN.
        return jnp.where(valid, soft_min, 0.0).sum() / jnp.maximum(valid.sum(), 1)

    def hard(z: jax.Array) -> tuple[jax.Array, jax.Array]:
        e, valid = rank_efficiency(*to_design(z))
        return jnp.where(valid, e.min(-1), 0.0).sum() / valid.sum(), valid.all()

    optimizer = optax.adam(learning_rate)
    value_and_grad = jax.value_and_grad(smoothed)

    # ponytail: one scan, no progress bar or chunking. Add one if people wait
    # more than a few minutes, following progress-doe's `_ascent._STEP_CHUNK`.
    def ascend(z0: jax.Array) -> jax.Array:
        def step(carry: tuple, _: None) -> tuple[tuple, None]:
            z, state, best_value, best_z = carry
            value, grad = value_and_grad(z)
            # Keep the best point seen: Adam can step off the ridge at the end.
            improved = value > best_value
            best_value = jnp.where(improved, value, best_value)
            best_z = jnp.where(improved, z, best_z)
            updates, state = optimizer.update(-grad, state, z)
            return (optax.apply_updates(z, updates), state, best_value, best_z), None

        init = (z0, optimizer.init(z0), jnp.array(-jnp.inf, dtype=z0.dtype), z0)
        (_, _, _, best_z), _ = jax.lax.scan(step, init, length=n_steps)
        return best_z

    unit = _latin_hypercube(start_key, n_restarts * n_arms, len(free))
    unit = jnp.clip(unit, _START_MARGIN, 1.0 - _START_MARGIN)
    starts = jnp.log(unit) - jnp.log1p(-unit)
    best_z = jax.jit(jax.vmap(ascend))(starts.reshape(n_restarts, n_arms, len(free)))
    scores, clean = jax.jit(jax.vmap(hard))(best_z)

    restart_scores = [float(s) if c else math.nan for s, c in zip(scores, clean)]
    if not bool(clean.any()):
        raise RuntimeError(
            f"Every restart failed to solve on some of the {n_rank_draws} ranking "
            "draws, so no design can be scored fairly. Narrow the bounds in "
            "`spec`, raise `config.max_steps`, or use a stiff solver "
            "(e.g. `config.solver = diffrax.Kvaerno5`)."
        )
    winner = to_box(best_z[int(jnp.argmax(jnp.where(clean, scores, -jnp.inf)))])

    dataset = _to_dataset(
        model, [midpoint | dict(zip(free, row)) for row in winner], times
    )
    report = evaluate_design(
        model,
        dataset,
        noise,
        key=report_key,
        n_draws=n_rank_draws,
        config=config,
        belief=belief,
    )
    return DesignResult(dataset=dataset, report=report, restart_scores=restart_scores)


def _validate(
    spec: Mapping[str, tuple[float, float]],
    names: list[str],
    n_arms: int,
    n_restarts: int,
    n_steps: int,
    n_draws: int,
    n_rank_draws: int,
) -> None:
    """Checks the spec and the counts before anything is traced.

    Raises:
        ValueError: Naming the column or argument and the fix.
    """
    missing = [name for name in names if name not in spec]
    if missing:
        raise ValueError(
            f"spec has no entry for {missing}. Give every state and constant a "
            "(low, high) range; low == high fixes it, e.g. spec['p'] = (0, 0)."
        )
    unknown = [name for name in spec if name not in names]
    if unknown:
        raise ValueError(
            f"spec names {unknown}, which are not states or constants of the "
            f"model ({names}). Remove them or fix the spelling."
        )
    for name in names:
        low, high = spec[name]
        if low > high:
            raise ValueError(
                f"spec['{name}'] = ({low}, {high}) has low > high. Swap them."
            )
    counts = {
        "n_arms": n_arms,
        "n_restarts": n_restarts,
        "n_steps": n_steps,
        "n_draws": n_draws,
        "n_rank_draws": n_rank_draws,
    }
    for argument, value in counts.items():
        if value < 1:
            raise ValueError(f"{argument} must be >= 1, got {value}.")
    if all(spec[name][0] == spec[name][1] for name in names):
        raise ValueError(
            "Every column of spec is fixed (low == high), so there is nothing to "
            "optimise. Widen a range, or score the design with evaluate_design."
        )


def _to_dataset(
    model: Model, arms: Sequence[Mapping[str, float]], times: list[float]
) -> Dataset:
    """One ``add_initial`` measurement per arm, each sampled at ``times``."""
    dataset = Dataset.from_model(model)
    for arm in arms:
        dataset.add_initial(time=times, **{k: float(v) for k, v in arm.items()})
    return dataset


def _latin_hypercube(key: jax.Array, n_points: int, n_dims: int) -> jax.Array:
    """A Latin-hypercube sample of the unit cube, shape ``(n_points, n_dims)``.

    Each dimension is cut into ``n_points`` strata and each stratum gets exactly
    one point. Points are drawn independently, so no restart starts with two
    identical arms: those get identical gradients forever and stay stuck at
    about 0.25 (measured in progress-doe).
    """
    perm_key, jitter_key = jax.random.split(key)
    strata = jnp.stack(
        [
            jax.random.permutation(k, n_points)
            for k in jax.random.split(perm_key, n_dims)
        ],
        axis=-1,
    )
    jitter = jax.random.uniform(jitter_key, (n_points, n_dims))
    return (strata + jitter) / n_points


__all__ = ["DesignResult", "optimize_design"]
