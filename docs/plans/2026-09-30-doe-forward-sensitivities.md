# Plan: differentiable Fisher information via forward sensitivities (catalax.doe, PR 2a)

## Goal

Make the Fisher information differentiable with respect to the **design**
(initial conditions and sampling times), which `optimise_design` needs. This adds
no optimiser and no new public API. `fisher_information` and `evaluate_design`
keep their signatures and results.

## Why (measured)

- **The current path cannot do it.** `Simulation(sensitivity=InAxes.PARAMETERS)`
  gets `S = dy/dθ` with `jax.jacobian`, i.e. backwards through diffrax's solve.
  `jax.grad` of any function of `S` w.r.t. `y0` then raises:
  `ValueError: Reverse-mode differentiation does not work for lax.while_loop…`
  (measured on `feat/doe` at 56aa295; same result as progress-doe
  `research/scripts/prototype_nested_ad.py`).
- **Integrating the sensitivities as extra ODE states works** with diffrax's default
  adjoint (`progress-doe/research/scripts/prototype_forward_sensitivities.py`):
  - `S` matches `jacfwd` of a plain solve to 8e-9.
  - Gradients vs central finite differences: `y0` 9.5e-7 (Tsit5) / 7.7e-7
    (Kvaerno5) relative; sampling times ≤ 1e-9 absolute, including traced `t1`.
  - Cost per gradient: Tsit5 75 steps / 2 ms, Kvaerno5 310 steps / 34 ms (17x).
    Stiff solvers work but are expensive: Newton runs on the whole augmented system.

## Checked before planning (2026-09-30)

- **Rule: every initial condition is a design variable, state or constant, and no
  species gets a default role.** Whether a column is varied is the design spec's
  choice (fixed/free, PR 2b), never its Catalax type. `catalax.doe` never
  distinguishes enzyme, substrate or product: each state's `observable` is declared
  explicitly in the model, and examples and new tests pass it on every `add_ode`.
  Measured on the current code: MAT with `E` as a constant and as a zero-rate
  unobservable state gives the same Fisher matrix to **1.2e-8**; with
  `"-k_inact*E"`, `k_inact` becomes a free parameter automatically. That model is
  also the first real case of the fixed observable indices (`E` sorts before `P`,
  so `P` is index 3). The gradient must therefore cover both paths: `y0s` (any
  state, observable or not) and `constants`. For a constant, the augmented solve
  matches finite differences to **1.1e-6**; for `y0` see above.
- **A failed draw does not poison the gradient.** Mean over two θ draws, one of
  which exhausts `max_steps` (`throw=False`, padded with inf, masked by `finite`):
  the gradient is finite and equals the valid draw's alone (−34.356). This is
  measured for step-limit failures only, not for blow-ups (non-finite right-hand
  side), so the test below adds a failing draw. **If every draw fails, the mean is
  0/0 = NaN**; `evaluate_design` already documents that, and PR 2b must guard it.
- **Free to change the sensitivity return shape.** Nothing but `doe/information.py`
  constructs `Simulation(sensitivity=…)`. `Model._sensitivity` (`model.py:101`) is
  declared and never read: dead, leave it (not this PR's business).
- **`SimulationConfig.t0`/`t1` are ignored by `Simulation`**, which hard-codes
  `t0=0` and uses `t1=time[-1]`. Only `Model.simulate` reads them (for `nsteps`
  grids). So the traced times are what matter; `doe` must never read `config.t1`
  for anything. Its `float(times.max())` default is fine here, because times are
  extracted at the boundary, but it will fail on a tracer in 2b: build the default
  config from the initial design outside traced code.

Deferred, recorded so they aren't rediscovered:
- 2b: `SaveAt(ts)` needs non-decreasing times, so optimised times need an ordered
  parameterisation (e.g. cumulative softplus increments). `to_time_matrix` requires
  equal numbers of samples per arm.
- PR 4: `mcmc.py` aligns `yerrs` assuming non-observable states sort last (the same
  class of bug as the fixed observable indices).

## Design

### 1. `catalax/tools/simulation.py`: forward-sensitivity mode

Add the mode to `Simulation` itself, not privately in `doe`, so there is one
simulation path that honours `SimulationConfig` (solver, controller, `dt0`,
`max_steps`, `throw`).

- New field: `sensitivity_method: Literal["reverse", "forward"] = "reverse"`.
  `"reverse"` is today's behaviour, unchanged.
- `"forward"` with `sensitivity=InAxes.PARAMETERS`: wrap the existing
  `stack(t, y, args)` in an augmented right-hand side over `(y, S)`:
  `dS/dt = J_y·S + ∂f/∂θ`, with `J_y·S` via `jax.vmap(jax.jvp(...))` over the
  columns of `S` and `∂f/∂θ` via `jax.jacfwd`. `S(0) = 0`.
  Reuse `_create_controller` and the `diffeqsolve` call of `_create_simulate_system`
  (factor out the shared call rather than copying it).
- Returns `(ys, S)`, shapes `(..., n_t, n_states)` and `(..., n_t, n_states, n_params)`.
  The states come from the same solve, which removes the double solve in
  `doe/information.py`.
- `"forward"` with any other `InAxes` → `NotImplementedError` naming
  `InAxes.PARAMETERS` as the supported case. Y0 and CONSTANTS (needed for PR 3,
  prep-error nuisances) are the same construction with `S(0) = I` or a `∂f/∂c`
  forcing term; add them then.

### 2. `catalax/doe/information.py`

- Replace the two `Simulation` objects with one
  `Simulation(sensitivity=InAxes.PARAMETERS, sensitivity_method="forward")`.
- `_prepare_information_function` returns a function of
  `(theta_free, y0s, constants, times)` instead of closing over the design arrays,
  so it can be differentiated w.r.t. the design. The validation in
  `_extract_design_arrays` stays at the boundary (outside traced code).
- `fisher_information` and `evaluate_design` pass the extracted arrays. Their
  public signatures don't change.
- Default config: Tsit5 stays the default. Docstring notes the Kvaerno5 cost
  measurement above.

### 3. Out of scope

- Objective, `DesignSpec`, box transform, Adam multistart, `optimise_design`,
  `DesignResult`: PR 2b.
- Sensitivities for only the free parameters (the fixed columns are wasted work).
  Mark it with a `ponytail:` comment; restrict the tangent basis when a model has
  many fixed parameters.

## Tests

In `../Catalax`, x64 fixture as in `tests/unit/doe/`.

- **Unchanged and must still pass at the same errors:**
  `tests/integration/doe/test_doe_oracle.py` (curves 8.3e-8, Fisher Frobenius
  3.4e-8, direction-aware 3.4e-7). This proves the swap changed nothing. Never
  re-tolerance it. Parametrise it over `E` as a constant **and** as an
  unobservable zero-rate state; both must hit the oracle.
- `tests/unit/tools/test_forward_sensitivity.py` (new):
  - `sensitivity_method="forward"` equals `"reverse"` on the conftest model
    (states and `S`), rtol ~1e-6.
  - Non-PARAMETERS `InAxes` with `"forward"` raises `NotImplementedError`.
- `tests/unit/doe/test_design_gradient.py` (new):
  - `jax.grad` of `logdet(F + I)` w.r.t. `y0s` (observable and unobservable states), `constants` and `times` matches central finite differences (h = 1e-5). Use an absolute floor for plateau times:
    finite differences give exactly 0 there.
  - A mean over draws where one draw fails (`max_steps` too small for it) has a
    finite gradient equal to the valid draws' gradient.
  - The information function works under `jax.vmap` over θ draws, as
    `evaluate_design` uses it.
- Full suite: `uv run pytest -m "not expensive"` stays at 80 passed + new tests.
  Run `ruff format`/`ruff check` **only on new files**; never on whole pre-existing
  Catalax files (it rewrites unrelated lines).

## Verification

1. `uv run pytest tests/unit/doe tests/unit/tools tests/integration/doe -vv`
2. `uv run pytest -m "not expensive" -q`: 80 + new, 0 failed.
3. `git diff --stat`: only `simulation.py`, `doe/information.py`, `doe/evaluate.py`
   (if touched) and the new tests.
