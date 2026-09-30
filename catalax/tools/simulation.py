from typing import Any, Callable, List, Literal, Optional, Sequence, Tuple

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from diffrax import (
    ConstantStepSize,
    PIDController,
    SaveAt,
    diffeqsolve,
)
from jaxtyping import PyTree
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from catalax.model.equation import Equation
from catalax.model.inaxes import InAxes
from catalax.model.ode import ODE
from catalax.model.reaction import Reaction
from catalax.model.simconfig import SimulationConfig

from .stack import BaseStack, MixedStack, ODEStack, ReactionStack

NON_ADAPTIVE_SOLVERS = [
    dfx.Euler,
    dfx.Heun,
]

EQUATION_TYPES = Literal["ode", "reaction"]


class SimulationInput(BaseModel):
    """
    Input configuration for simulation setup.

    This class encapsulates all the necessary components for setting up a simulation,
    including the differential equations, reactions, and variable definitions.

    Attributes:
        odes: List of ODE objects defining differential equations
        reactions: List of reaction objects defining reaction kinetics
        states: List of state variable names in the system
        parameters: List of parameter names
        constants: List of constant names
    """

    odes: List[ODE] = Field(default_factory=list)
    reactions: List[Reaction] = Field(default_factory=list)
    states: List[str]
    parameters: List[str]
    constants: List[str]

    @property
    def type(self) -> Literal["ode", "reaction", "mixed"]:
        """
        Get the type of the simulation input.
        """
        if len(self.reactions) == 0 and len(self.odes) == 0:
            return "mixed"
        elif len(self.reactions) > 0 and len(self.odes) == 0:
            return "reaction"
        elif len(self.reactions) == 0 and len(self.odes) > 0:
            return "ode"
        else:
            return "mixed"

    @property
    def stack(self) -> BaseStack | MixedStack:
        """
        Get the stack of the simulation input.
        """
        """
        Build the appropriate stack type based on the simulation input configuration.

        This method analyzes the simulation input to determine whether to create an ODE stack,
        reaction stack, or mixed stack, then instantiates the appropriate stack type.

        Args:
            sim_input: Simulation input configuration containing ODEs and reactions

        Returns:
            Stack object (ODEStack, ReactionStack, or MixedStack) configured for the system

        Raises:
            ValueError: If an invalid stack type is determined
        """
        if self.type == "ode":
            return ODEStack(sim_input=self)
        elif self.type == "reaction":
            return ReactionStack(sim_input=self)
        elif self.type == "mixed":
            return MixedStack(sim_input=self)
        else:
            raise ValueError(f"Invalid stack type: {self.type}")

    def get_equations(self, type: EQUATION_TYPES) -> Sequence[Equation]:
        """
        Get equations of a specific type from the simulation input.

        Args:
            type: Type of equations to retrieve ("ode" or "reaction")

        Returns:
            Sequence of equations of the specified type
        """
        if type == "ode":
            return self.odes
        if type == "reaction":
            return self.reactions


class Simulation(BaseModel):
    """
    A simulation class for solving systems of ODEs using diffrax.

    This class handles the setup and execution of ODE simulations with support for
    various solver configurations, sensitivity analysis, and batch processing through
    vectorization. It provides a high-level interface for numerical integration of
    differential equation systems.

    The simulation supports three types of systems:
    - Pure ODE systems (direct differential equations)
    - Pure reaction systems (reaction kinetics with stoichiometry)
    - Mixed systems (combination of ODEs and reactions)

    Attributes:
        sim_input: Configuration containing ODEs, reactions, and variable definitions
        config: Simulation configuration including solver settings and tolerances
        sensitivity: Optional sensitivity analysis configuration for computing Jacobians
        sensitivity_method: How sensitivities are computed. "reverse" differentiates
            through the solver with ``jax.jacobian`` and returns only the Jacobian.
            "forward" integrates ``S = dy/dparameters`` alongside the states
            (``dS/dt = J_y S + df/dparameters``, ``S(0) = 0``) and returns
            ``(ys, S)``; the result stays differentiable with respect to the
            initial conditions, constants and time points. Only
            ``InAxes.PARAMETERS`` is supported in forward mode.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    sim_input: SimulationInput
    config: SimulationConfig
    sensitivity: Optional[InAxes] = None
    sensitivity_method: Literal["reverse", "forward"] = "reverse"

    _simulation_func: Optional[Callable] = PrivateAttr(default=None)

    def __call__(self, y0, parameters, constants, time) -> Any:
        """
        Execute the simulation with given inputs.

        Args:
            y0: Initial conditions array with shape (n_states,)
            parameters: Parameter values array with shape (n_parameters,)
            constants: Constant values array with shape (n_constants,)
            time: Time points array for solution evaluation

        Returns:
            Simulation results at specified time points

        Raises:
            ValueError: If simulation function is not initialized
        """
        if self._simulation_func is None:
            raise ValueError("Simulation function not initialized")

        return self._simulation_func(y0, parameters, constants, time)

    def _prepare_func(
        self,
        in_axes: Optional[Tuple] = None,
    ) -> Tuple[Callable, BaseStack | MixedStack]:
        """
        Prepare and configure the simulation function with all necessary transformations.

        This method orchestrates the complete simulation setup process:
        1. Creates the appropriate stack type (ODE, reaction, or mixed)
        2. Sets up the stepsize controller based on solver type
        3. Creates the core simulation function
        4. Applies transformations for sensitivity analysis or vectorization
        5. JIT-compiles the final function for performance

        Args:
            in_axes: Optional tuple specifying which axes to vectorize over for batch processing.
                    Should contain 4 elements corresponding to (y0, parameters, constants, time).
                    Use None for no vectorization, or specify axis indices (e.g., (0, None, None, None)
                    to vectorize over the first dimension of y0).

        Returns:
            Tuple containing:
                - The prepared simulation function (JIT-compiled and potentially vectorized)
                - The stack object used for the simulation

        Raises:
            ValueError: If an invalid stack type is determined or if the system has no ODEs or reactions
        """
        if self.sensitivity_method == "forward":
            if self.sensitivity is not InAxes.PARAMETERS:
                raise NotImplementedError(
                    f"Forward sensitivities support only sensitivity=InAxes.PARAMETERS, "
                    f"got {self.sensitivity}. Use sensitivity_method='reverse' for "
                    "other axes."
                )
            forward_system = self._create_forward_sensitivity_system(
                self.sim_input.stack,
                self._create_controller(),
            )
            return self._prepare_standard_func(forward_system, in_axes)  # type: ignore

        simulate_system = self._create_simulate_system(
            self.sim_input.stack,
            self._create_controller(),
        )

        if self.sensitivity is not None:
            return self._prepare_sensitivity_func(simulate_system, in_axes)  # type: ignore

        return self._prepare_standard_func(simulate_system, in_axes)  # type: ignore

    def _create_controller(self) -> ConstantStepSize | PIDController:
        """
        Create the appropriate stepsize controller based on the solver type.

        Non-adaptive solvers (like Euler and Heun) use a constant step size controller,
        while adaptive solvers use a PID controller with specified relative and absolute
        tolerances for automatic error control and step size adjustment.

        Returns:
            ConstantStepSize for non-adaptive solvers, PIDController for adaptive solvers
        """
        if self.config.solver in NON_ADAPTIVE_SOLVERS:
            return ConstantStepSize()
        else:
            return PIDController(rtol=self.config.rtol, atol=self.config.atol)

    def _create_simulate_system(
        self,
        stack: BaseStack | MixedStack,
        controller: ConstantStepSize | PIDController,
    ) -> Callable:
        """
        Create the core simulation function that solves the ODE system.

        This method creates a closure that encapsulates the diffrax solver configuration
        and returns a function that can be called with initial conditions, parameters,
        constants, and time points to solve the ODE system using the specified numerical
        integration method.

        Args:
            stack: Stack object containing the ODE system (ODEStack, ReactionStack, or MixedStack)
            controller: Stepsize controller for the solver (ConstantStepSize or PIDController)

        Returns:
            Simulation function that takes initial conditions, parameters, constants, and time
            and returns the solution at specified time points
        """

        def _simulate_system(
            y0: jax.Array,
            parameters: jax.Array,
            constants: jax.Array,
            time: jax.Array,
        ) -> PyTree:
            """
            Internal simulation function that solves the ODE system.

            Uses diffrax to solve the ODE system defined by the stack with the given
            initial conditions, parameters, and constants over the specified time points.
            Handles both successful solutions and error cases based on the throw configuration.

            Args:
                y0: Initial conditions array with shape (n_states,)
                parameters: Parameter values array with shape (n_parameters,)
                constants: Constant values array with shape (n_constants,)
                time: Time points array with shape (n_timepoints,)

            Returns:
                Solution array at specified time points with shape (n_timepoints, n_states).
                If throw=False and integration fails, returns array filled with inf values.
            """
            return self._solve(stack, y0, (parameters, constants), time, controller)

        return _simulate_system

    def _create_forward_sensitivity_system(
        self,
        stack: BaseStack | MixedStack,
        controller: ConstantStepSize | PIDController,
    ) -> Callable:
        """
        Create a simulation function that integrates parameter sensitivities as states.

        The right-hand side ``f = stack(t, y, (parameters, constants))`` is augmented
        with ``dS/dt = J_y S + df/dparameters`` for ``S = dy/dparameters``, with
        ``S(0) = 0``. Both terms come from one ``jax.jvp`` per parameter, with tangent
        ``(S[:, j], e_j)``. Unlike ``jax.jacobian`` through the solve, the result can
        be differentiated again with respect to ``y0``, ``constants`` and ``time``.

        Args:
            stack: Stack object containing the ODE system
            controller: Stepsize controller for the solver

        Returns:
            Function of ``(y0, parameters, constants, time)`` returning ``(ys, S)`` with
            shapes ``(n_timepoints, n_states)`` and
            ``(n_timepoints, n_states, n_parameters)``. If throw=False and integration
            fails, both are filled with inf values.
        """

        def augmented(t, state, args):
            (y, S), (parameters, constants) = state, args

            def f(y, parameters):
                return stack(t, y, (parameters, constants))

            # ponytail: sensitivities are computed for fixed parameters too; restrict
            # the tangent basis to the free ones when a model has many fixed parameters.
            dS = jax.vmap(
                lambda s, e: jax.jvp(f, (y, parameters), (s, e))[1],
                in_axes=1,
                out_axes=1,
            )(S, jnp.eye(parameters.shape[0], dtype=S.dtype))
            return f(y, parameters), dS

        def _simulate_with_sensitivities(y0, parameters, constants, time):
            S0 = jnp.zeros(
                (y0.shape[0], parameters.shape[0]),
                dtype=jnp.result_type(y0, parameters, float),
            )
            return self._solve(
                augmented, (y0, S0), (parameters, constants), time, controller
            )

        return _simulate_with_sensitivities

    def _solve(self, vector_field, y0, args, time, controller) -> PyTree:
        """
        Solve ``dy/dt = vector_field(t, y, args)`` from ``t = 0`` to ``time[-1]``.

        Honours the solver, ``dt0``, ``max_steps`` and ``throw`` of the configuration.

        Returns:
            The solution at ``time``, with the pytree structure of ``y0``.
            If throw=False and integration fails, it is filled with inf values.
        """
        sol = diffeqsolve(
            terms=dfx.ODETerm(vector_field),
            solver=self.config.solver(),
            t0=0,
            t1=time[-1],
            dt0=self.config.dt0,
            y0=y0,
            args=args,
            saveat=SaveAt(ts=time),
            stepsize_controller=controller,
            max_steps=self.config.max_steps,
            throw=self.config.throw,
        )

        return sol.ys

    def _prepare_sensitivity_func(self, simulate_system, in_axes):
        """
        Prepare simulation function with sensitivity analysis capabilities.

        This method wraps the simulation function with JAX's automatic differentiation
        to compute sensitivities (Jacobians) with respect to specified arguments.
        The sensitivity analysis can be vectorized for batch processing, enabling
        efficient computation of parameter sensitivities across multiple conditions.

        Args:
            simulate_system: Base simulation function to differentiate
            in_axes: Vectorization axes specification for batch processing.
                    If None, no vectorization is applied. Should be a tuple of 4 elements
                    corresponding to (y0, parameters, constants, time) axes.

        Returns:
            Tuple containing:
                - The sensitivity-enabled simulation function (JIT-compiled and optionally vectorized)
                - Self reference for method chaining

        Raises:
            AssertionError: If sensitivity is not an instance of InAxes
        """
        assert isinstance(
            self.sensitivity, InAxes
        ), "Expected sensitivity to be an instance of 'InAxes'"

        def sens_fun(y0s, parameters, constants, time):
            """Sensitivity analysis wrapper function."""
            return simulate_system(y0s, parameters, constants, time)

        index = self.sensitivity.value.index(0)

        if in_axes is not None:
            return (
                eqx.filter_jit(
                    jax.vmap(jax.jacobian(sens_fun, argnums=index), in_axes=in_axes)
                ),
                self,
            )

        return (
            eqx.filter_jit(jax.jacobian(sens_fun, argnums=int(index))),
            self,
        )

    def _prepare_standard_func(self, simulate_system, in_axes):
        """
        Prepare standard simulation function without sensitivity analysis.

        This method wraps the simulation function with JIT compilation and optional
        vectorization for batch processing. This is the standard path when no
        sensitivity analysis is required, providing optimal performance for regular
        simulation tasks.

        Args:
            simulate_system: Base simulation function to prepare
            in_axes: Vectorization axes specification for batch processing.
                    If None, no vectorization is applied. Should be a tuple of 4 elements
                    corresponding to (y0, parameters, constants, time) axes.
                    Example: (0, None, None, None) vectorizes over the first dimension of y0.

        Returns:
            Tuple containing:
                - The prepared simulation function (JIT-compiled and optionally vectorized)
                - Self reference for method chaining
        """
        if in_axes is not None:
            return (
                eqx.filter_jit(jax.vmap(simulate_system, in_axes=in_axes)),
                self,
            )
        else:
            return (
                eqx.filter_jit(simulate_system),
                self,
            )
