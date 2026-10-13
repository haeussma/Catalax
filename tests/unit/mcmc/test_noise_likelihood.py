"""HMC with a given noise model: the likelihood scale is the noise SD in the family's units."""

import math

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest

import catalax as ctx
import catalax.mcmc as cmm
from catalax.mcmc.mcmc import SD_TO_SCALE, BayesianModel, Modes, _prepare_mcmc_data

pytestmark = pytest.mark.usefixtures("x64")


def _problem():
    model = ctx.Model(name="t")
    model.add_state("A, B, C")
    model.add_ode("A", "0", observable=False)
    model.add_ode("B", "-k * A * B", observable=True)
    model.add_ode("C", "k * A * B", observable=True)
    model.parameters["k"].value = 0.1
    model.parameters["k"].prior = cmm.priors.Uniform(low=0.01, high=1.0)
    design = ctx.Dataset.from_model(model)
    design.add_initial(time=[0.0, 1.0, 2.0], A=1.0, B=10.0, C=0.0)
    data = model.simulate(design, ctx.SimulationConfig(t1=2.0), saveat=design.to_time_matrix())
    config = cmm.MCMCConfig(num_warmup=1, num_samples=1, verbose=0)
    prep = _prepare_mcmc_data(data, model, None, config.to_simulation_config())
    return model, prep, config


def _trace(noise, likelihood=dist.Normal):
    model, prep, config = _problem()
    bayesian_model = BayesianModel(
        model=model,
        yerrs=None,
        likelihood=likelihood,
        sim_func=prep.sim_func,
        shapes=prep.shapes,
        mode=Modes.MECHANISTIC,
        config=config.to_simulation_config(),
        noise=ctx.noise.as_noise_model(noise),
        noise_factor=SD_TO_SCALE[likelihood],
    )
    trace = numpyro.handlers.trace(numpyro.handlers.seed(bayesian_model, 0)).get_trace(
        y0s=prep.y0s, constants=prep.constants, times=prep.times, mask=prep.mask, data=prep.data
    )
    fn = trace["y"]["fn"]
    return trace, getattr(fn, "base_dist", fn), prep


def test_fixed_array_is_the_normal_scale_and_no_sigma_is_sampled():
    _, prep, _ = _problem()
    sigma = jnp.full(prep.data.shape, 0.2).at[0, 1, 0].set(0.5)
    trace, fn, _ = _trace(sigma)
    assert "sigma" not in trace
    assert jnp.allclose(fn.scale, sigma)


def test_soft_laplace_scale_is_converted_from_the_sd():
    _, prep, _ = _problem()
    sigma = jnp.full(prep.data.shape, 0.2)
    _, fn, _ = _trace(sigma, likelihood=dist.SoftLaplace)
    assert jnp.allclose(fn.scale, 0.2 * 2 / math.pi)
    # the converted scale has the requested SD
    assert float(dist.SoftLaplace(0.0, fn.scale[0, 0, 0]).variance) ** 0.5 == pytest.approx(0.2)


def test_proportional_is_evaluated_on_the_prediction():
    trace, fn, _ = _trace(ctx.noise.Proportional(cv=0.1, floor=0.05))
    prediction = fn.loc
    assert jnp.allclose(fn.scale, jnp.sqrt(0.05**2 + (0.1 * prediction) ** 2))


def test_unknown_family_and_conflicts_raise():
    model, _, _ = _problem()
    data = ctx.Dataset.from_model(model)
    data.add_initial(time=[0.0, 1.0], A=1.0, B=10.0, C=0.0)
    data = model.simulate(data, ctx.SimulationConfig(t1=1.0), saveat=data.to_time_matrix())
    with pytest.raises(ValueError, match="likelihood"):
        cmm.HMC(num_warmup=1, num_samples=1, likelihood=dist.Cauchy, verbose=0).run(model, data, noise=0.1)
    with pytest.raises(ValueError, match="noise_cv"):
        cmm.HMC(num_warmup=1, num_samples=1, noise_cv=0.05, verbose=0).run(model, data, noise=0.1)
    with pytest.raises(ValueError, match="yerrs"):
        cmm.HMC(num_warmup=1, num_samples=1, verbose=0).run(model, data)


def test_noise_priors_are_sampled_and_used():
    noise = ctx.noise.Proportional(
        cv=cmm.priors.LogNormal(mu=0.035, sigma=0.4), floor=cmm.priors.LogNormal(mu=0.006, sigma=0.6)
    )
    trace, fn, _ = _trace(noise)
    cv, floor = trace["cv"]["value"], trace["floor"]["value"]
    assert isinstance(trace["cv"]["fn"], dist.LogNormal)
    assert jnp.allclose(fn.scale, jnp.sqrt(floor**2 + (cv * fn.loc) ** 2))


def _clashing_noise():
    """A noise model whose prior field has the same name as the model parameter k."""
    import equinox as eqx

    class K(eqx.Module):
        k: object

        def scale(self, prediction):
            return jnp.broadcast_to(self.k, prediction.shape)

        def whiten(self, sensitivities, prediction):
            return sensitivities / self.scale(prediction)[..., None]

    return K(k=cmm.priors.LogNormal(mu=0.1, sigma=0.5))


def test_noise_prior_name_clash_with_a_parameter_raises():
    model, prep, config = _problem()
    with pytest.raises(ValueError, match="clash"):
        BayesianModel(
            model=model, yerrs=None, likelihood=dist.Normal, sim_func=prep.sim_func,
            shapes=prep.shapes, mode=Modes.MECHANISTIC, config=config.to_simulation_config(),
            noise=_clashing_noise(),
        )


def test_noise_cv_is_deprecated():
    with pytest.warns(DeprecationWarning, match="Proportional"):
        cmm.HMC(num_warmup=1, num_samples=1, noise_cv=0.05, verbose=0)
