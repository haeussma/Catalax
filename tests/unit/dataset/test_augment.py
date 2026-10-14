"""Dataset.augment: independent noise for every measurement, round and species."""

import jax.numpy as jnp
import pytest

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


def test_noise_model_gives_the_same_noise_as_sigma():
    """Homoskedastic(s) is additive sigma=s; Proportional is multiplicative plus its floor."""
    additive = _noise(_dataset().augment(n_augmentations=1, sigma=0.1, append=False))
    homoskedastic = _noise(_dataset().augment(n_augmentations=1, noise=ctx.noise.Homoskedastic(0.1), append=False))
    assert all(jnp.allclose(a, b) for a, b in zip(additive, homoskedastic))

    # data are 1.0, so the proportional sd is sqrt(floor**2 + cv**2)
    multiplicative = _noise(_dataset().augment(n_augmentations=1, sigma=0.1, multiplicative=True, append=False))
    proportional = _noise(
        _dataset().augment(n_augmentations=1, noise=ctx.noise.Proportional(cv=0.1, floor=0.1), append=False)
    )
    assert all(jnp.allclose(p, m * jnp.sqrt(2.0)) for p, m in zip(proportional, multiplicative))


def test_noise_and_sigma_together_or_priors_raise():
    with pytest.raises(ValueError, match="either sigma"):
        _dataset().augment(n_augmentations=1, sigma=0.1, noise=ctx.noise.Homoskedastic(0.1))
    with pytest.raises(ValueError, match="multiplicative"):
        _dataset().augment(n_augmentations=1, multiplicative=True, noise=ctx.noise.Homoskedastic(0.1))
    with pytest.raises(ValueError, match="priors"):
        _dataset().augment(
            n_augmentations=1, noise=ctx.noise.Proportional(cv=ctx.mcmc.priors.LogNormal(mu=0.1, sigma=1.0), floor=0.01)
        )
