# optimize(): least squares on observed states Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ctx.optimize` a correct least-squares fit for partially observed models with missing points, then use it for Model 1 of the NmNeuS reanalysis.

**Architecture:** `catalax/tools/optimization.py` is changed to do what `catalax/mcmc/mcmc.py` already does: take data and simulation only for `model.get_observable_state_order()`. Hidden states need only initial conditions. The residual becomes a plain vector `simulation − data` over the measured points (NaN gaps are dropped through a fixed mask), and the default solver becomes lmfit's `least_squares`. A custom per-point loss `objective_fun(y_true, y_pred)` still works: the residual becomes `sqrt(loss)`, so lmfit's sum of squares equals the total loss. The mervan side then fits Model 1 with no workarounds.

**Tech Stack:** Python 3.13, JAX, lmfit, pytest, uv. Catalax at `~/code/Catalax` (branch `feat/doe`); mervan at `~/code/mervan` (branch `pub`), which uses Catalax as an editable checkout.

**Spec:** No separate spec document. The design was agreed in conversation on 2026-10-12 and is summarised under "Design" below. The executor reads this plan only.

## Global Constraints

- Work in `~/code/Catalax` on branch `feat/doe`, and in `~/code/mervan` on branch `pub`. Do not switch branches; do not push.
- Leave the untracked `docs/plans/2026-09-15-fix-observable-broadcast-0.5.6.md` in Catalax untouched and out of every commit.
- Public signature of `ctx.optimize` keeps all parameter names: `model, dataset, global_upper_bound, global_lower_bound, dt0, objective_fun, max_steps, method`. Only two defaults change: `objective_fun=None`, `method="least_squares"`.
- `objective_fun`, when given, keeps its existing call convention `objective_fun(y_true, y_pred)` returning an elementwise, non-negative loss array (as `ctx.l1_loss` and `optax.log_cosh` do).
- Run tests with `uv run pytest ...` from the repository root.
- Commit messages end with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Do not change ChromHandler, PyEnzyme, or the EnzymeML documents in `~/code/mervan/data/processed/`.

## Design

What is wrong with `optimize` today (all three reproduced on 2026-10-12):

1. **Hidden states are not supported.** It reads data for `model.get_state_order()` (all states), so a model with an unmeasured substrate fails with `ValueError: ... Missing ['S'] in measurement ...`. `run_mcmc` reads only `model.get_observable_state_order()`; `optimize` must do the same.
2. **Missing points crash the fit.** `Dataset.from_enzymeml` pads shorter measurements and unsampled points with NaN; the residual then contains NaN and lmfit raises `NaN values detected`.
3. **It is not least squares.** The residual returned to lmfit is `optax.l2_loss(data, sim)` = ½·r², and lmfit squares that array again, so the fit minimises Σ ¼·r⁴. With default BFGS on noisy straight-line data (`P' = k`, true k = 0.5) it returns k = 0.760 where the least-squares answer is 0.514.

Masking pitfalls, checked on 2026-10-12 with the straight-line problem:

- **Fixed length.** `measured` comes from the data, which never changes during a fit, so the
  flattened residual has the same length on every lmfit call even when measurements have
  different numbers of real points (e.g. 6 vs 4).
- **Drop, do not zero.** Masked points must be removed, not set to 0: zeros would count in
  lmfit's `ndata`, inflate the degrees of freedom and shrink the reduced χ² and the standard
  errors. Dropped, one masked point of six gives `ndata = 5` and the exact standard error.
- **Equal weight per point.** Least squares sums over points, so a measurement with fewer real
  points weighs less. Nothing may average per measurement. A custom `objective_fun` must
  therefore be elementwise; a reducing loss such as `ctx.mean_absolute_error` (returns a scalar)
  is rejected with a clear error.
- **float32 finite differences.** Catalax runs JAX in float32 by default. scipy's default
  relative finite-difference step (≈1.5·10⁻⁸) is below float32 resolution (≈6·10⁻⁸), so the
  Jacobian is noise: on the straight line, k came out 0.101 instead of 0.514. With
  `diff_step = sqrt(eps)` of the data dtype (3.5·10⁻⁴ in float32, 1.5·10⁻⁸ in float64) the fit
  returns k = 0.51365 and the standard error to 0.04 %; in float64 it is exact either way.
- **Simulation NaN is not masked.** Only NaN in the *data* is dropped. If the solver fails and
  returns NaN at a measured point, lmfit still raises: a failed simulation must not silently
  disappear from the fit.
- **Padded time points are still simulated.** Padding beyond the last real time costs
  integration time and is masked out; it does not affect the result.

Fix: residual = `(y_pred − y_true)[measured]` with `measured = ~isnan(y_true)` computed once, so its length is fixed across lmfit calls and lmfit's `ndata` counts only real points (correct reduced χ² and standard errors). With a custom loss: `sqrt(objective_fun(y_true, y_pred))[measured]`. Default method `least_squares` (scipy trust-region, honours bounds, returns covariance).

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `~/code/Catalax/catalax/tools/optimization.py` | `optimize` and its residual | Modify: observed states, mask, residual, defaults, docstring |
| `~/code/Catalax/tests/unit/tools/test_optimize.py` | Behaviour of `optimize` on small analytic models | Create |
| `~/code/Catalax/docs/basic/parameter-opt.mdx` | User docs for `optimize` | Modify: defaults and observed-state behaviour |
| `~/code/mervan/scripts/fit_model1_lsq.py` | Model 1 least-squares fit on the concentration document | Create |

---

### Task 1: Least-squares residual and default solver

**Files:**
- Modify: `~/code/Catalax/catalax/tools/optimization.py` (whole module, ~200 lines)
- Create: `~/code/Catalax/tests/unit/tools/test_optimize.py`
- Modify: `~/code/Catalax/docs/basic/parameter-opt.mdx:57-66`

**Interfaces:**
- Consumes: `Model.get_observable_state_order(as_indices: bool = False)` (exists, `catalax/model/model.py:967`), `Dataset.to_jax_arrays(state_order)` (exists, returns `(data[n_meas, n_time, n_states], times[n_meas, n_time], inits)`).
- Produces: `optimize(model, dataset, global_upper_bound=1e5, global_lower_bound=1e-6, dt0=0.01, objective_fun=None, max_steps=64**4, method="least_squares") -> tuple[MinimizerResult, Model]`. Tasks 2–4 rely on this signature and on `result.params[name].value`, `.stderr`, `result.chisqr`, `result.ndata`.

- [ ] **Step 1: Write the failing tests**

Create `~/code/Catalax/tests/unit/tools/test_optimize.py`:

```python
"""Tests for ctx.optimize: a least-squares fit on the observed states of a model."""

import jax.numpy as jnp
import numpy as np
import pytest

import catalax as ctx

NOISE = jnp.array([0.0, 0.3, -0.2, 0.9, -0.4, 0.1])


def _bound(model: ctx.Model, name: str, initial: float) -> None:
    parameter = model.parameters[name]
    parameter.initial_value = initial
    parameter.lower_bound = 1e-3
    parameter.upper_bound = 10.0


def linear_problem() -> tuple[ctx.Model, ctx.Dataset]:
    """P' = k, P(0) = 0, sampled at t = 0, 2, ..., 10 with fixed noise; true k = 0.5."""
    model = ctx.Model(name="linear")
    model.add_state("P")
    model.add_ode("P", "k")
    model.parameters["k"].value = 0.5
    _bound(model, "k", initial=0.1)

    dataset = ctx.Dataset.from_model(model)
    dataset.add_initial(P=0.0)
    dataset = model.simulate(dataset=dataset, config=ctx.SimulationConfig(t1=10, nsteps=6))
    measurement = dataset.measurements[0]
    measurement.data["P"] = measurement.data["P"] + NOISE
    return model, dataset


def least_squares_slope(dataset: ctx.Dataset) -> float:
    """Closed-form least-squares k for P = k·t over the measured (non-NaN) points."""
    measurement = dataset.measurements[0]
    t, y = np.asarray(measurement.time), np.asarray(measurement.data["P"])
    keep = np.isfinite(y)
    return float(t[keep] @ y[keep] / (t[keep] @ t[keep]))


def least_squares_stderr(dataset: ctx.Dataset) -> float:
    """Closed-form standard error of k, with n - 1 degrees of freedom over measured points."""
    measurement = dataset.measurements[0]
    t, y = np.asarray(measurement.time), np.asarray(measurement.data["P"])
    keep = np.isfinite(y)
    t, y = t[keep], y[keep]
    k = t @ y / (t @ t)
    return float(np.sqrt(np.sum((y - k * t) ** 2) / (len(t) - 1) / (t @ t)))


def test_default_fit_is_ordinary_least_squares():
    model, dataset = linear_problem()
    result, fitted = ctx.optimize(model, dataset)
    assert result.params["k"].value == pytest.approx(least_squares_slope(dataset), rel=1e-4)
    assert fitted.parameters["k"].value == pytest.approx(result.params["k"].value)


def test_standard_error_matches_least_squares():
    """Fails in float32 if finite differences use scipy's default step (k stays near 0.1)."""
    model, dataset = linear_problem()
    result, _ = ctx.optimize(model, dataset)
    assert result.ndata == 6
    assert result.params["k"].stderr == pytest.approx(least_squares_stderr(dataset), rel=1e-2)


def test_squared_error_objective_matches_default():
    """A per-point loss is minimised as a sum: (y - p)^2 must give the least-squares k."""
    model, dataset = linear_problem()
    result, _ = ctx.optimize(model, dataset, objective_fun=lambda y_true, y_pred: (y_true - y_pred) ** 2)
    assert result.params["k"].value == pytest.approx(least_squares_slope(dataset), rel=1e-4)


def test_reducing_objective_is_rejected():
    """A loss that averages over points cannot be split into per-point residuals."""
    model, dataset = linear_problem()
    with pytest.raises(ValueError, match="one loss per data point"):
        ctx.optimize(model, dataset, objective_fun=ctx.mean_absolute_error)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd ~/code/Catalax && uv run pytest tests/unit/tools/test_optimize.py -v`
Expected: the three value/stderr tests FAIL on their `approx` assertions (the default fit returns
k ≈ 0.76 instead of ≈ 0.514); `test_reducing_objective_is_rejected` FAILS because no
`ValueError` with that message is raised.

- [ ] **Step 3: Rewrite `optimize` and `_residual`**

In `~/code/Catalax/catalax/tools/optimization.py`:

Replace the imports at the top (lines 1–11) with:

```python
from copy import deepcopy
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from lmfit import Parameters, minimize
from lmfit.minimizer import MinimizerResult

from catalax.dataset.dataset import Dataset
from catalax.model import Model, SimulationConfig
```

Replace the whole `optimize` function (from `def optimize(` to its final `return (... new_model, )`) with:

```python
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
        method (str, optional): lmfit method. Defaults to "least_squares".

    Returns:
        Tuple[MinimizerResult, Model]: lmfit result and the fitted copy of the model.
    """

    params = _initialize_params(model, global_upper_bound, global_lower_bound)

    data, times, _ = dataset.to_jax_arrays(model.get_observable_state_order())
    observed = jnp.array(model.get_observable_state_order(as_indices=True))
    measured = ~jnp.isnan(data)

    fit_kws = {}
    if method == "least_squares":
        # The finite-difference step must exceed the simulation's float precision: in float32
        # (Catalax's default) scipy's default relative step of 1.5e-8 cannot be resolved.
        fit_kws["diff_step"] = float(jnp.finfo(data.dtype).eps) ** 0.5

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

    new_model = deepcopy(model)

    for name, parameter in result.params.items():  # type: ignore
        new_model.parameters[name].value = parameter.value

    return (
        result,
        new_model,
    )
```

Replace the whole `_residual` function with:

```python
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
```

Leave `_ipython_print`, `_ipython_flush` and `_initialize_params` unchanged.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd ~/code/Catalax && uv run pytest tests/unit/tools/test_optimize.py tests/integration/test_optimization.py -v`
Expected: 5 passed (the four new tests and the existing `test_optimize_runs`).

- [ ] **Step 5: Update the user docs**

In `~/code/Catalax/docs/basic/parameter-opt.mdx`, inside the existing `python` code block that
starts with `# Perform optimization` (lines 57–66), make two line edits:

- `# Perform optimization` becomes `# Perform optimization (ordinary least squares by default)`
- `    method="lbfgs",                  # Optimization algorithm` becomes
  `    method="least_squares",          # lmfit method (default); reports standard errors`

Then, directly after that code block's closing fence (before `## Model selection`), insert this
paragraph:

> Only the states of the model that are observable are compared with data: a state whose ODE
> was added with `observable=False` needs an initial condition but no data. NaN entries in the
> data (unsampled points, padding of shorter measurements) are left out of the fit. To fit with
> another per-point loss, pass `objective_fun(y_true, y_pred)` returning a non-negative array,
> e.g. `objective_fun=ctx.l1_loss`; the fit then minimises the sum of that loss.

(Insert it as plain text, without the `>` quote markers.)

- [ ] **Step 6: Commit**

```bash
cd ~/code/Catalax
git add catalax/tools/optimization.py tests/unit/tools/test_optimize.py docs/basic/parameter-opt.mdx
git commit -m "fix(optimize): ordinary least squares on observed states

The residual handed to lmfit was optax.l2_loss, which lmfit squares again,
so the fit minimised the sum of r^4; with BFGS on noisy data it also stopped
far from the optimum. Return simulation - data (or sqrt of a custom per-point
loss), compare only the model's observable states, drop NaN points through a
fixed mask, and default to lmfit least_squares with a finite-difference step
suited to the data's float precision (float32 could not resolve scipy's
default step). Reject objective functions that average over points.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Hidden states need no data

**Files:**
- Modify: `~/code/Catalax/tests/unit/tools/test_optimize.py` (append)
- No production change expected: Task 1 already reads data for observable states only. This task proves it.

**Interfaces:**
- Consumes: `optimize(...)` from Task 1; `Model.add_ode(state, equation, observable=False)` (exists, `catalax/model/model.py:161`).
- Produces: nothing new.

- [ ] **Step 1: Write the test**

Append to `~/code/Catalax/tests/unit/tools/test_optimize.py`:

```python
def test_hidden_state_needs_no_data():
    """S -> P with only P measured: S has an initial condition but no data array."""
    model = ctx.Model(name="hidden substrate")
    model.add_state("S, P")
    model.add_ode("S", "-k * S", observable=False)
    model.add_ode("P", "k * S")
    model.parameters["k"].value = 0.3
    _bound(model, "k", initial=0.1)

    dataset = ctx.Dataset.from_model(model)
    dataset.add_initial(S=2.0, P=0.0)
    dataset = model.simulate(dataset=dataset, config=ctx.SimulationConfig(t1=10, nsteps=6))
    for measurement in dataset.measurements:
        measurement.data.pop("S")

    result, _ = ctx.optimize(model, dataset)
    assert result.params["k"].value == pytest.approx(0.3, rel=1e-3)
```

- [ ] **Step 2: Run it**

Run: `cd ~/code/Catalax && uv run pytest tests/unit/tools/test_optimize.py::test_hidden_state_needs_no_data -v`
Expected: PASS.

- [ ] **Step 3: Verify the test fails on the old code**

HEAD is the Task 1 commit, so `HEAD~1` holds the old `optimization.py`:

```bash
cd ~/code/Catalax
git checkout HEAD~1 -- catalax/tools/optimization.py
uv run pytest tests/unit/tools/test_optimize.py::test_hidden_state_needs_no_data -v
git checkout HEAD -- catalax/tools/optimization.py
```

Expected: FAIL with `ValueError: The measurement states are inconsistent ... Missing ['S']`,
and after the last command `git status --short catalax/` shows nothing.

- [ ] **Step 4: Commit**

```bash
cd ~/code/Catalax
git add tests/unit/tools/test_optimize.py
git commit -m "test(optimize): hidden states need initial conditions, not data

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Missing points are left out

**Files:**
- Modify: `~/code/Catalax/tests/unit/tools/test_optimize.py` (append)
- No production change expected: Task 1 masks NaN.

**Interfaces:**
- Consumes: `optimize(...)`, `linear_problem()`, `least_squares_slope(dataset)` from Task 1's test module.
- Produces: nothing new.

- [ ] **Step 1: Write the test**

Append to `~/code/Catalax/tests/unit/tools/test_optimize.py`:

```python
def test_nan_points_are_left_out():
    """A NaN data point is skipped: the fit equals least squares over the other points."""
    model, dataset = linear_problem()
    measurement = dataset.measurements[0]
    measurement.data["P"] = measurement.data["P"].at[3].set(jnp.nan)

    result, _ = ctx.optimize(model, dataset)

    assert result.ndata == 5  # dropped, not counted as a zero residual
    assert result.params["k"].value == pytest.approx(least_squares_slope(dataset), rel=1e-4)
    assert result.params["k"].stderr == pytest.approx(least_squares_stderr(dataset), rel=1e-2)
```

- [ ] **Step 2: Run it**

Run: `cd ~/code/Catalax && uv run pytest tests/unit/tools/test_optimize.py -v`
Expected: 6 passed.

Verify it fails on the old code (HEAD is the Task 2 commit; the old `optimization.py` is at `HEAD~2`):

```bash
cd ~/code/Catalax
git checkout HEAD~2 -- catalax/tools/optimization.py
uv run pytest tests/unit/tools/test_optimize.py::test_nan_points_are_left_out -v
git checkout HEAD -- catalax/tools/optimization.py
```

Expected: FAIL with `ValueError: NaN values detected ...`; afterwards `git status --short catalax/` shows nothing.

- [ ] **Step 3: Run the whole Catalax test suite for regressions**

Run: `cd ~/code/Catalax && uv run pytest -m "not expensive" -q`
Expected: no failures. If any test outside `tests/unit/tools/test_optimize.py` calls `ctx.optimize` and relied on the old default method or objective, report it instead of changing that test.

- [ ] **Step 4: Commit**

```bash
cd ~/code/Catalax
git add tests/unit/tools/test_optimize.py
git commit -m "test(optimize): NaN points are left out of the fit

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Model 1 by least squares in mervan

**Files:**
- Create: `~/code/mervan/scripts/fit_model1_lsq.py`
- Output (generated, commit it): `~/code/mervan/figs/fit_model1_lsq.png`

**Interfaces:**
- Consumes: `ctx.optimize(model, dataset)` from Task 1 (default least squares); `ctx.Dataset.from_enzymeml(doc)`; `pe.read_enzymeml(path)`; `Dataset.plot(ncols, path, predictor)` (exists, `catalax/dataset/dataset.py:957`); `data/processed/neus_kinetics_concentration.json` (exists; species `ManNAc, PEP, Neu5Ac, H2O, Pi, Co`, protein `NmNeuS`; every measurement carries `prepared` for ManNAc, PEP, Neu5Ac, Co and NmNeuS).
- Produces: printed parameter table; `figs/fit_model1_lsq.png`.

Model 1 as on the workshop slides: the four-parameter rate law below, product (Neu5Ac) observed, ManNAc and PEP hidden, one enzyme activity for all series, starting concentrations = prepared (nominal) amounts, data = pre-calibrated concentrations. Units: concentrations in mM, time in min, NmNeuS in mg/mL, so `k_cat` comes out in U/mg (µmol·min⁻¹·mg⁻¹); turnover in s⁻¹ = `k_cat · 38649.8 / 60000` (NmNeuS monomer mass in g/mol used on branch `investigate`).

- [ ] **Step 1: Write the script**

Create `~/code/mervan/scripts/fit_model1_lsq.py`:

```python
"""Model 1 by least squares: one rate law fitted to the progress curves of all 44 reactions.

Product (Neu5Ac) observed, ManNAc and PEP hidden, one enzyme activity for all series. Data are
the pre-calibrated concentrations (neus_kinetics_concentration); the starting concentrations
are the prepared amounts, as in the initial-rate analysis this is compared with.

Run from the repository root:
    uv run python scripts/fit_model1_lsq.py
"""

from pathlib import Path

import catalax as ctx
import pyenzyme as pe

DOCUMENT = Path("data/processed/neus_kinetics_concentration.json")
FIGURE = Path("figs/fit_model1_lsq.png")
M_NEUS = 38649.8  # g/mol, NmNeuS monomer; turnover [1/s] = k_cat [U/mg] * M_NEUS / 60000

RATE = (
    "k_cat * NmNeuS * ManNAc * PEP / ((Km_ManNAc + ManNAc) * (Km_PEP + PEP))"
    " * Co / (Km_Co + Co)"
)
# initial value, lower bound, upper bound; concentrations in mM, k_cat in U/mg
PARAMETERS = {
    "k_cat": (10.0, 1.0, 100.0),
    "Km_ManNAc": (1.0, 1e-3, 50.0),
    "Km_PEP": (1.0, 1e-3, 50.0),
    "Km_Co": (3e-4, 1e-6, 0.1),
}


def load_dataset(path: Path) -> ctx.Dataset:
    """Catalax dataset whose starting concentrations are the prepared amounts.

    Catalax takes `initial` (the first measured point) as the starting concentration; the
    prepared amount is what went into the reaction, so it replaces it here.
    """
    doc = pe.read_enzymeml(path)
    prepared = {
        measurement.id: {d.species_id: d.prepared for d in measurement.species_data if d.prepared is not None}
        for measurement in doc.measurements
    }
    dataset = ctx.Dataset.from_enzymeml(doc)
    for measurement in dataset.measurements:
        measurement.initial_conditions = {**measurement.initial_conditions, **prepared[measurement.id]}
        measurement.data.pop("ManNAc")  # Model 1 fits the product only
    return dataset


def model_1() -> ctx.Model:
    model = ctx.Model(name="Model 1")
    model.add_state("ManNAc, PEP, Neu5Ac")
    model.add_constant("Co, NmNeuS")
    model.add_ode("ManNAc", f"-({RATE})", observable=False)
    model.add_ode("PEP", f"-({RATE})", observable=False)
    model.add_ode("Neu5Ac", RATE)
    for name, (initial, lower, upper) in PARAMETERS.items():
        parameter = model.parameters[name]
        parameter.initial_value = initial
        parameter.lower_bound = lower
        parameter.upper_bound = upper
    return model


def main() -> None:
    dataset = load_dataset(DOCUMENT)
    result, fitted = ctx.optimize(model_1(), dataset)

    print(f"success: {result.success}, {result.ndata} points, residual SD "
          f"{(result.chisqr / result.nfree) ** 0.5:.4f} mM")
    for name, parameter in result.params.items():
        stderr = parameter.stderr if parameter.stderr is not None else float("nan")
        print(f"  {name:10s} {parameter.value:10.4g} ± {stderr:.2g}")
    print(f"  turnover   {result.params['k_cat'].value * M_NEUS / 60000:10.4g} 1/s")

    FIGURE.parent.mkdir(exist_ok=True)
    dataset.plot(ncols=8, path=str(FIGURE), predictor=fitted)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `cd ~/code/mervan && uv run python scripts/fit_model1_lsq.py`
Expected:
- `success: True, 253 points` (253 measured Neu5Ac points).
- Four parameters with finite standard errors. Plausibility ranges (from the workshop deck, Model 1 by Bayesian inference: K_M,PEP 0.54 mM, turnover 6.1 s⁻¹): `Km_ManNAc` 0.1–5 mM, `Km_PEP` 0.1–3 mM, `Km_Co` 1e-5–1e-2 mM, turnover 2–20 s⁻¹. Values outside these ranges are reported, not tuned.
- `figs/fit_model1_lsq.png` written; open it and check that the fitted curves follow the product data, the PEP-limited curves (PEP_H … PEP_N) being the known weak spot of Model 1.

- [ ] **Step 3: Commit**

```bash
cd ~/code/mervan
git add scripts/fit_model1_lsq.py figs/fit_model1_lsq.png
git commit -m "Fit Model 1 by least squares with Catalax

Product observed, substrates hidden, prepared concentrations as starting
values, pre-calibrated concentrations as data.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Note: other uncommitted work on `pub` (window change, notes, check script, PyEnzyme pin, rebuilt documents) is not part of this commit; stage only the two paths above.
