"""Dataset.augment: independent noise for every measurement, round and species."""

import jax.numpy as jnp

import catalax as ctx


def _dataset() -> ctx.Dataset:
    """Two measurements with identical data in two species, so any shared noise shows."""
    model = ctx.Model(name="t")
    model.add_state("A, B")
    dataset = ctx.Dataset.from_model(model)
    for _ in range(2):
        dataset.add_measurement(
            ctx.Measurement(
                initial_conditions={"A": 1.0, "B": 1.0},
                time=[0.0, 1.0, 2.0, 3.0],
                data={"A": [1.0, 1.0, 1.0, 1.0], "B": [1.0, 1.0, 1.0, 1.0]},
            )
        )
    return dataset


def _noise(dataset: ctx.Dataset) -> list[jnp.ndarray]:
    return [m.data[s] - 1.0 for m in dataset.measurements for s in ("A", "B")]


def test_every_measurement_round_and_species_gets_its_own_noise():
    noise = _noise(_dataset().augment(n_augmentations=2, sigma=0.1, append=False))
    assert len(noise) == 8
    for i in range(len(noise)):
        for j in range(i + 1, len(noise)):
            assert not jnp.allclose(noise[i], noise[j])


def test_the_same_seed_reproduces_the_noise():
    first = _noise(_dataset().augment(n_augmentations=1, sigma=0.1, seed=3, append=False))
    second = _noise(_dataset().augment(n_augmentations=1, sigma=0.1, seed=3, append=False))
    assert all(jnp.array_equal(a, b) for a, b in zip(first, second))
