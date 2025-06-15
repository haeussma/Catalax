from typing import Callable, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jax import Array
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS

from catalax.dataset.dataset import Dataset
from catalax.model.parameter import Parameter


def run_mcmc(
    model: "Model",
    dataset: Dataset,
    yerrs: Union[float, Array],
    num_warmup: int,
    num_samples: int,
    neuralode: Optional["NeuralBase"] = None,
    dense_mass: bool = True,
    thinning: int = 1,
    max_tree_depth: int = 10,
    dt0: float = 0.1,
    chain_method: str = "sequential",
    num_chains: int = 1,
    seed: int = 420,
    verbose: int = 1,
    max_steps: int = 64**4,
    mask: Optional[Array] = None,
):
    """Runs an MCMC simulation to infer the posterior distribution of parameters.

    This function is using the NumPyro package to execute a Markov Chain Monte Carlo simulation
    using the No-U-Turn Sampler (NUTS) algorithm. Priors for the parameters are automatically extracted
    from the model such that the user only needs to specify the initial conditions, the data, and the
    standard deviation of the data. The simulation function of the model is used to simulate the model
    for each set of parameters sampled from the prior distribution. The simulated data is then compared
    to the observed data and the posterior distribution of parameters is inferred.

    Args:
        model (Model): The model to fit.
        dataset (Dataset): The dataset to fit.
        yerrs (Array, float): The standard deviation of the observed data.
        num_warmup (int): Number of warmup steps.
        num_samples (int): Number of samples.
        neuralode (Optional[NeuralBase]): Neural ODE model to use for rate prediction.
        dense_mass (bool, optional): Whether to use a dense mass matrix or not. Defaults to True.
        dt0 (float, optional): Resolution of the simulation. Defaults to 0.1.
        chain_method (str, optional): Choose from 'vectorized', 'parallel' or 'sequential'. Defaults to "sequential".
        num_chains (int, optional): Number of chains. Defaults to 1.
        seed (int, optional): Random number seed to reproduce results. Defaults to 420.
        verbose (int, optional): Whether to show progress and summary. Defaults to 1.
        max_steps (int, optional): Maximum number of steps for the solver. Defaults to 64**4.
        mask (Optional[Array], optional): Boolean mask array indicating which data points to use. Defaults to None.
    """

    # Check if all paramaters have priors
    assert all(param.prior is not None for param in model.parameters.values()), (
        f"Parameters {', '.join([param.name for param in model.parameters.values() if param.prior is None])} do not have priors. Please specify priors for all parameters."
    )

    # Extract data, times, and initial conditions from the dataset for observable species
    data, times, y0s = dataset.to_jax_arrays(
        model.get_observable_species_order(),
        inits_to_array=True,
    )

    # Create initial conditions for all species, including non-observable species
    all_species = model.get_species_order()
    full_y0s = []

    for meas in dataset.measurements:
        # Create array with all initial conditions in the correct order
        y0 = jnp.array([meas.initial_conditions[species] for species in all_species])
        full_y0s.append(y0)

    y0s = jnp.stack(full_y0s)

    # Determine dimensions
    in_axes = dataset.get_vmap_dims(
        data=data,
        time=times,
        y0s=y0s,
    )

    # Compile the model to obtain the simulation function
    model._setup_system(
        in_axes=in_axes,
        dt0=dt0,
        max_steps=max_steps,
    )

    # Get all priors
    priors = [
        (model.parameters[param].name, model.parameters[param].prior._distribution_fun)
        for param in model.get_parameter_order()
    ]

    if neuralode is not None:
        rate_fun = model._setup_rate_function(in_axes=(0, 0, None))
        sim_func = lambda y0s, theta, times: rate_fun(times, y0s, theta)
        times = times.ravel()
        y0s = data.reshape(data.shape[0] * data.shape[1], -1)
        data = _predict_rates_using_neural_ode(
            neuralode=neuralode,
            y0s=y0s,
            data=data,
            time=times,
        )
    else:
        sim_func = model._sim_func

    # Handle NaN values and validate mask
    if mask is not None:
        if verbose:
            print(f"Mask shape: {mask.shape}")
            print(f"Data shape: {data.shape}")
            print(f"Mask dtype: {mask.dtype}")
            print(f"Number of masked points: {(~mask).sum()}")
            print(f"Number of valid points: {mask.sum()}")
            print(f"Number of NaN values in data: {jnp.isnan(data).sum()}")

        # Check if mask shape matches data shape
        if mask.shape != data.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match data shape {data.shape}"
            )

        # Check if mask is boolean
        if mask.dtype != bool:
            raise ValueError(f"Mask must be boolean, got {mask.dtype}")

        # Check if there are any valid points left
        if mask.sum() == 0:
            raise ValueError(
                "Mask excludes all data points. At least some data points must be valid."
            )

        # Replace NaN values with zeros where mask is False (masked out)
        # This prevents NaN from causing issues during model initialization
        data = jnp.where(mask, data, 0.0)

        if verbose:
            print(
                f"After NaN replacement - NaN values in data: {jnp.isnan(data).sum()}"
            )

    # If no mask is provided but data contains NaN, create a mask automatically
    elif jnp.isnan(data).any():
        if verbose:
            print(
                f"No mask provided but data contains {jnp.isnan(data).sum()} NaN values"
            )
            print("Creating automatic mask to exclude NaN values")

        # Create mask that excludes NaN values
        mask = ~jnp.isnan(data)
        # Replace NaN values with zeros
        data = jnp.where(mask, data, 0.0)

        if verbose:
            print(f"Automatic mask created - Number of valid points: {mask.sum()}")

    # Setup the bayes model
    bayes_model = _setup_model(
        yerrs=yerrs,
        priors=priors,  # type: ignore
        sim_func=sim_func,  # type: ignore
        model=model,
        mask=mask,
    )

    mcmc = MCMC(
        NUTS(bayes_model, dense_mass=dense_mass, max_tree_depth=max_tree_depth),
        num_warmup=num_warmup,
        num_samples=num_samples,
        progress_bar=bool(verbose),
        chain_method=chain_method,
        num_chains=num_chains,
        jit_model_args=True,
        thinning=thinning,
    )

    if verbose:
        print("\n🚀 Running MCMC\n")

    mcmc.run(
        PRNGKey(seed),
        data=data,
        y0s=y0s,
        times=times,
    )

    # Print a nice summary
    if verbose:
        print("\n\n🎉 Finished")
        mcmc.print_summary()

    return mcmc, bayes_model


def _setup_model(
    yerrs: Union[float, Array],
    sim_func: Callable,
    priors: List[Tuple[str, dist.Distribution]],
    model: "Model",
    mask: Optional[Array] = None,
):
    """Function to setup the model for the MCMC simulation.

    This is done, to not have to pass the priors and the simulation function to the MCMC.
    If a mask is provided, masked data points will not be used to update the prior during inference.
    The mask should be a boolean array with the same shape as the data, where True indicates
    data points to use and False indicates points to mask out.

    Args:
        yerrs (Union[float, Array]): The standard deviation of the observed data.
        sim_func (Callable): The simulation function of the model.
        priors (List[Tuple[str, dist.Distribution]]): List of parameter priors.
        model (Model): The model to fit.
        mask (Optional[Array]): Boolean mask array indicating which data points to use.
    """

    # Set up the observables to extract from the simulation
    observables = jnp.array(
        [
            i
            for i, species in enumerate(model.get_species_order())
            if model.odes[species].observable
        ]
    )

    def _bayes_model(y0s: Array, times: Array, data: Optional[Array] = None):
        """Generalized bayesian model to infer the posterior distribution of parameters.

        This function is used to sample from the posterior distribution of parameters by
        sampling from the prior distribution and comparing the simulated data with the
        observations. Theta is the vector of parameters, sigma is the standard deviation of
        the noise, and states is the simulated data.

        Args:
            data (Array): The data against which the model is fitted.
            y0s (Array): The initial conditions of the model.
            times (Array): The times at which the data is sampled.
        """

        theta = jnp.array(
            [numpyro.sample(name, distribution) for name, distribution in priors]
        )

        states = sim_func(y0s, theta, times)

        sigma = numpyro.sample("sigma", dist.HalfNormal(yerrs))  # type: ignore

        # If mask is provided, use numpyro.handlers.mask to mask out data points
        if mask is not None:
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample(
                    "y", dist.Normal(states[..., observables], sigma), obs=data
                )  # type: ignore
        else:
            numpyro.sample("y", dist.Normal(states[..., observables], sigma), obs=data)  # type: ignore

    return _bayes_model


def _predict_rates_using_neural_ode(
    neuralode: "NeuralBase",
    y0s: Array,
    data: Array,
    time: Array,
) -> Array:
    """
    Predicts the rates of a given system using a Neural ODE model.

    Args:
        neuralode: A NeuralBase object representing the Neural ODE model.
        y0s: An array of initial conditions for the system.
        data: An array of data points for the system.
        time: An array of time points for the system.

    Returns:
        An array of predicted rates for the system.
    """
    dataset_size, length_size, _ = data.shape
    ins = data.reshape(dataset_size * length_size, -1)

    return jax.vmap(neuralode.func, in_axes=(0, 0, None))(time.ravel(), ins, 0.0)


def _print_priors(parameters: List[Parameter]):
    """
    Prints the prior distributions for each parameter in the list of parameters.

    Args:
        parameters: A list of Parameter objects.

    Returns:
        None
    """
    fun = lambda name, value: f"├── \033[1m{name}\033[0m: {value}"
    statements = [
        "🔸 Priors",
        *[fun(param.name, param.prior._print_str) for param in parameters],
    ]

    print("\n".join(statements))
