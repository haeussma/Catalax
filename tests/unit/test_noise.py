"""catalax.noise: the shared observation-noise models."""

import jax.numpy as jnp
import pytest

import catalax as ctx
import catalax.doe as cdoe
import catalax.mcmc as cmm


def test_doe_reexports_the_shared_classes():
    assert cdoe.Proportional is ctx.noise.Proportional
    assert cdoe.Homoskedastic is ctx.noise.Homoskedastic
    assert cdoe.noise.NoiseModel is ctx.noise.NoiseModel


def test_homoskedastic_accepts_a_per_point_array():
    sigma = jnp.array([[0.1, 0.2, 0.3]])
    assert jnp.array_equal(ctx.noise.Homoskedastic(sigma).scale(jnp.zeros((1, 3))), sigma)


def test_proportional_scale():
    noise = ctx.noise.Proportional(cv=0.1, floor=0.3)
    assert float(noise.scale(jnp.array(4.0))) == pytest.approx(0.5)


def test_priors_are_kept_and_listed():
    noise = ctx.noise.Proportional(cv=cmm.priors.LogNormal(mu=0.035, sigma=0.4), floor=0.006)
    assert list(ctx.noise.prior_fields(noise)) == ["cv"]
    assert float(noise.floor) == pytest.approx(0.006)


def test_numeric_floor_must_still_be_positive():
    with pytest.raises(ValueError, match="strictly positive"):
        ctx.noise.Proportional(cv=0.05, floor=0.0)


def test_design_refuses_noise_with_priors():
    noise = ctx.noise.Proportional(cv=cmm.priors.LogNormal(mu=0.035, sigma=0.4), floor=0.006)
    with pytest.raises(ValueError, match="fixed"):
        ctx.noise.as_noise_model(noise, fixed=True)
