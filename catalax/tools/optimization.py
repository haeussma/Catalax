from copy import deepcopy
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from lmfit import Parameters, minimize
from lmfit.minimizer import MinimizerResult

from catalax.dataset.dataset import Dataset
from catalax.model import Model, SimulationConfig


def optimize(
    model: Model,
    dataset: Dataset,
    global_upper_bound: Optional[float] = 1e5,
    global_lower_bound: Optional[float] = 1e-6,
    dt0: float = 0.01,
    objective_fun: Optional[Callable[[jax.Array, jax.Array], jax.Array]] = None,
    max_steps: int = 64**4,
    method: str = "least_squares",
) -> Tuple[MinimizerResult, Model]:
    """Fit the parameters of a model to a dataset with lmfit.

    Only the model's observable states are compared with data (states whose ODE was
    added with ``observable=False`` need initial conditions, not data), and NaN entries
    in the data (unsampled points, padding of shorter measurements) are left out.

    By default this is ordinary least squares: lmfit receives the residual vector
    ``simulation - data`` and minimises its sum of squares. A per-point loss can be
    given instead as ``objective_fun(y_true, y_pred)`` returning a non-negative array;
    the fit then minimises the sum of that loss over the measured points.

    The standard errors in the result are meaningful only for the default least-squares
    objective; with a custom ``objective_fun`` they come from the Jacobian of ``sqrt(loss)``.

    The returned model is a modified COPY of the input model carrying the fitted
    parameter values; the input model is not changed.

    Args:
        model (Model): The model to fit. Every parameter needs an initial value, and
            bounds unless the global bounds apply.
        dataset (Dataset): Data for every observable state of the model, plus initial
            conditions for all states and constants.
        global_upper_bound (Optional[float], optional): Upper bound for parameters
            without their own. Defaults to 1e5.
        global_lower_bound (Optional[float], optional): Lower bound for parameters
            without their own. Defaults to 1e-6.
        dt0 (float, optional): Integration step width. Defaults to 0.01.
        objective_fun (Optional[Callable], optional): Per-point loss
            ``(y_true, y_pred) -> array``. Defaults to None (least squares).
        max_steps (int, optional): Maximum number of integration steps. Defaults to 64**4.
        method (str, optional): lmfit method. Defaults to "least_squares". Gradient-based
            scalar methods (e.g. ``lbfgsb``, ``bfgs``) need ``ctx.enable_x64()``: their
            finite-difference steps cannot be resolved in float32.

    Returns:
        Tuple[MinimizerResult, Model]: lmfit result and the fitted copy of the model.
    """

    params = _initialize_params(model, global_upper_bound, global_lower_bound)

    data, times, _ = dataset.to_jax_arrays(model.get_observable_state_order())
    observed = jnp.array(model.get_observable_state_order(as_indices=True))
    measured = ~jnp.isnan(data)

    fit_kws = {}
    # The finite-difference step must exceed the accumulated rounding error of the ODE
    # simulation, or the Jacobian (and so the reported standard errors, and the point where
    # the fit stops) is noise: in float32 scipy's default 1.5e-8 cannot be resolved, and the
    # naive sqrt(eps) (3.5e-4) is still too small for nonlinear ODEs. The cube root of eps is
    # the standard step for central differences, whose truncation error is O(h^2).
    step = float(jnp.finfo(data.dtype).eps) ** (1 / 3)
    if method == "least_squares":
        fit_kws["diff_step"] = step
        fit_kws["jac"] = "3-point"
    elif method == "leastsq":
        # MINPACK forward-differences with a step of sqrt(epsfcn) * |x|.
        fit_kws["epsfcn"] = step**2

    config = dataset.to_config()
    config.dt0 = dt0
    config.max_steps = max_steps
    config.nsteps = None

    minimize_args = (
        model,
        dataset,
        config,
        data,
        times,
        observed,
        measured,
        objective_fun,
    )

    _ipython_print(params)

    result = minimize(
        _residual,
        params,
        args=minimize_args,
        method=method,
        **fit_kws,
    )

    _ipython_flush()
    _ipython_print(result)

    # Update the model with the inferred parameters
    new_model = deepcopy(model)

    for name, parameter in result.params.items():  # type: ignore
        new_model.parameters[name].value = parameter.value

    return (
        result,
        new_model,
    )


def _ipython_print(obj):
    """Print Jupyter things when available"""
    try:
        from IPython.display import display

        display(obj)
    except ImportError:
        pass


def _ipython_flush():
    """Flush Jupyter things when available"""
    try:
        from IPython.display import clear_output

        clear_output()
    except ImportError:
        pass


def _initialize_params(
    model: Model,
    global_upper_bound: Optional[float],
    global_lower_bound: Optional[float],
):
    """Initialize the Parameter object for the optimization."""

    params = Parameters()

    for param in model.parameters.values():
        if param.lower_bound is None:
            assert global_lower_bound is not None, (
                f"Neither a global lower bound nor lower bound for parameter '{param.name}' is given. \
                Please specify either a global lower bound or a lower bound for the parameter."
            )

            param.lower_bound = global_lower_bound

        if param.upper_bound is None:
            assert global_upper_bound is not None, (
                f"Neither a global upper bound nor upper bound for parameter '{param.name}' is given. \
                Please specify either a global upper bound or a upper bound for the parameter."
            )

            param.upper_bound = global_upper_bound

        if param.initial_value is None and param.value is None:
            raise ValueError(
                f"Neither 'value' nor 'initial_value' given for parameter '{param.name}'. \
                    Please add an initial value for the optimization to work."
            )
        elif param.initial_value is None and param.value is not None:
            param.initial_value = param.value

        params.add(
            param.name,
            value=param.initial_value,
            min=param.lower_bound,
            max=param.upper_bound,
            vary=not param.constant,
        )

    return params


def _residual(
    params: Parameters,
    model: Model,
    dataset: Dataset,
    config: SimulationConfig,
    data: jax.Array,
    times: jax.Array,
    observed: jax.Array,
    measured: jax.Array,
    objective_fun: Optional[Callable[[jax.Array, jax.Array], jax.Array]],
) -> np.ndarray:
    """Residual vector over the measured points of the observable states.

    Args:
        params (Parameters): Current parameter values.
        model (Model): Model to simulate.
        dataset (Dataset): Initial conditions and constants.
        config (SimulationConfig): Simulation configuration.
        data (jax.Array): Observed data, shape (measurements, times, observable states).
        times (jax.Array): Time points, shape (measurements, times).
        observed (jax.Array): Columns of the simulation that are observable states.
        measured (jax.Array): Boolean mask of non-NaN data; fixed across calls, so the
            residual length is constant and lmfit counts only real points.
        objective_fun (Optional[Callable]): Per-point loss, or None for least squares.

    Returns:
        np.ndarray: 1-D residual; lmfit minimises the sum of its squares.
    """

    param_dict = {**params.valuesdict()}  # type: ignore
    parameters = jnp.array([param_dict[param] for param in model.get_parameter_order()])

    _, states = model.simulate(
        dataset=dataset,
        config=config,
        saveat=times,
        parameters=parameters,
        return_array=True,
    )
    predicted = states[:, :, observed]

    if objective_fun is None:
        residual = predicted - data
    else:
        loss = objective_fun(data, predicted)
        if jnp.shape(loss) != data.shape:
            raise ValueError(
                f"objective_fun must return one loss per data point (shape {data.shape}), "
                f"got shape {jnp.shape(loss)}; losses that average over points cannot be used."
            )
        # lmfit squares and sums the residual, so sqrt(loss) makes that sum the total loss.
        residual = jnp.sqrt(loss)

    return np.asarray(residual[measured])
