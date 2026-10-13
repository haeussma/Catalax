"""Same model, data and fixed noise: the posterior equals the weighted least-squares result."""

import jax.numpy as jnp
import numpyro.distributions as dist
import numpy as np
import pytest

import catalax as ctx
import catalax.mcmc as cmm

NOISE = jnp.array([0.0, 0.3, -0.2, 0.9, -0.4, 0.1])


def linear_problem() -> tuple[ctx.Model, ctx.Dataset]:
    """P' = k, P(0) = 0, sampled at t = 0, 2, ..., 10 with fixed noise; true k = 0.5.

    Same problem as tests/unit/tools/test_optimize.py (test directories are not packages,
    so it is repeated here rather than imported).
    """
    model = ctx.Model(name="linear")
    model.add_state("P")
    model.add_ode("P", "k")
    model.parameters["k"].value = 0.5
    parameter = model.parameters["k"]
    parameter.initial_value, parameter.lower_bound, parameter.upper_bound = 0.1, 1e-3, 10.0
    dataset = ctx.Dataset.from_model(model)
    dataset.add_initial(P=0.0)
    dataset = model.simulate(dataset=dataset, config=ctx.SimulationConfig(t1=10, nsteps=6))
    dataset.measurements[0].data["P"] = dataset.measurements[0].data["P"] + NOISE
    return model, dataset


@pytest.mark.expensive
def test_posterior_matches_weighted_least_squares(x64):
    model, dataset = linear_problem()
    data, times, _ = dataset.to_jax_arrays(model.get_observable_state_order())
    sigma = (0.1 + 0.05 * times)[..., None]

    result, _ = ctx.optimize(model, dataset, noise=sigma)

    model.parameters["k"].prior = cmm.priors.Uniform(low=0.01, high=2.0)
    hmc = cmm.HMC(
        num_warmup=500, num_samples=4000, num_chains=1, chain_method="sequential",
        likelihood=dist.Normal, verbose=0,
    )
    samples = hmc.run(model, dataset, noise=sigma).get_samples()["k"]

    assert float(np.mean(samples)) == pytest.approx(result.params["k"].value, rel=1e-2)
    assert float(np.std(samples)) == pytest.approx(result.params["k"].stderr, rel=0.1)
