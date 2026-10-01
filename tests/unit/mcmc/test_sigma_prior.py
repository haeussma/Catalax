"""The sigma prior's scale is the mean of yerrs over the observed columns."""

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import pytest

import catalax as ctx
import catalax.mcmc as cmm
from catalax.mcmc.mcmc import BayesianModel, Modes, _prepare_mcmc_data

pytestmark = pytest.mark.usefixtures("x64")


def test_sigma_scale_with_an_unobservable_state_first():
    # A sorts before the observables B and C, so observable order (B, C) and
    # modeled positions (1, 2) differ: indexing yerrs with the latter read column 1
    # twice (JAX clamps the out-of-range 2) and gave 100 instead of 50.5.
    model = ctx.Model(name="t")
    model.add_state("A, B, C")
    model.add_ode("A", "0", observable=False)
    model.add_ode("B", "-k * A * B", observable=True)
    model.add_ode("C", "k * A * B", observable=True)
    model.parameters["k"].value = 0.1
    model.parameters["k"].prior = cmm.priors.Uniform(low=0.01, high=1.0)

    design = ctx.Dataset.from_model(model)
    design.add_initial(time=[0.0, 1.0, 2.0], A=1.0, B=10.0, C=0.0)
    data = model.simulate(
        design, ctx.SimulationConfig(t1=2.0), saveat=design.to_time_matrix()
    )

    config = cmm.MCMCConfig(num_warmup=1, num_samples=1, verbose=0)
    prep = _prepare_mcmc_data(data, model, None, config.to_simulation_config())
    yerrs = jnp.broadcast_to(jnp.array([1.0, 100.0]), prep.data.shape)
    bayesian_model = BayesianModel(
        model=model,
        yerrs=yerrs,
        likelihood=dist.Normal,
        sim_func=prep.sim_func,
        shapes=prep.shapes,
        mode=Modes.MECHANISTIC,
        config=config.to_simulation_config(),
    )
    trace = numpyro.handlers.trace(numpyro.handlers.seed(bayesian_model, 0)).get_trace(
        y0s=prep.y0s,
        constants=prep.constants,
        times=prep.times,
        mask=prep.mask,
        data=prep.data,
    )

    assert float(trace["sigma"]["fn"].scale) == pytest.approx(50.5)
