"""Boxes of initial conditions (``spec``) and the designs drawn from them."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from catalax.dataset.dataset import Dataset

if TYPE_CHECKING:
    from catalax.model.model import Model


def _validate_spec(spec: Mapping[str, tuple[float, float]], names: list[str]) -> None:
    """Checks that ``spec`` gives every name in ``names`` a ``(low, high)`` range.

    Raises:
        ValueError: Naming the column and the fix.
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
