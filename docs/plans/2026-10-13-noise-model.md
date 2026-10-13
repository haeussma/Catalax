# One noise model for least squares, Bayesian inference and design — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `ctx.optimize`, `HMC` and the design code share one observation-noise description, so a weighted least-squares fit and a Bayesian fit can use identical noise (phase 1), and the Bayesian fit can sample the noise parameters with priors (phase 2).

**Architecture:** `catalax/doe/noise.py` moves to `catalax/noise.py` (the old path keeps re-exporting). A noise model is anything with `scale(prediction) -> sd`. A bare array or float means `Homoskedastic(that)`, i.e. fixed per-point standard deviations. `optimize(noise=...)` evaluates the noise model once on the measured data and runs weighted least squares with those SDs taken as absolute (`scale_covar=False`). `HMC.run(noise=...)` evaluates it on the model prediction inside the likelihood and converts the SD to the likelihood family's scale. In phase 2 the fields of `Homoskedastic`/`Proportional` may be `catalax.mcmc.priors` objects; HMC samples them, the design code refuses them.

**Tech Stack:** Python 3.13, JAX, equinox, numpyro, lmfit, pytest, uv. Repository `~/code/Catalax`, branch `feat/doe`.

**Spec:** No separate spec. Design agreed in conversation on 2026-10-13, summarised under "Design".

## Global Constraints

- Work in `~/code/Catalax` on branch `feat/doe`. Do not switch branches; do not push.
- Leave the untracked `docs/plans/2026-09-15-fix-observable-broadcast-0.5.6.md` untouched and out of every commit.
- Existing behaviour is unchanged when `noise` is not given: `optimize` stays ordinary least squares with `scale_covar=True`; `HMC` keeps `sigma ~ HalfNormal(mean(yerrs))` and the optional `noise_cv` term. `tests/unit/mcmc/test_sigma_prior.py` and `tests/unit/tools/test_optimize.py` must keep passing unchanged.
- `from catalax.doe import Proportional, Homoskedastic, NoiseModel` and `import catalax.doe.noise` keep working.
- A noise SD always means a standard deviation, in data units. Never pass it unconverted as the scale of a non-Normal family.
- Run tests with `uv run pytest <path> -q` from the repository root. Unit tests: `uv run pytest tests/unit -q`.
- Commit messages end with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Design

- **Noise model = per-point SD.** `Homoskedastic(sigma)`: `sd = sigma` broadcast to the data shape; `sigma` may be a scalar or a full per-point array (that is how fixed, externally computed SDs are passed). `Proportional(cv, floor)`: `sd = sqrt(floor**2 + (cv * y)**2)`. Gaussian, independent per point.
- **Least squares (`optimize`)** evaluates `noise.scale(data)` — on the measured values, once, before the fit — and minimises `sum(((ŷ - y) / sd)**2)`. Weights never change during the fit (no reweighting loop). The SDs are taken as absolute: lmfit runs with `scale_covar=False`. Without `noise`, plain least squares with `scale_covar=True` as today. `noise` together with `objective_fun` is an error.
- **HMC** evaluates `noise.scale(prediction)` inside the likelihood. For `Homoskedastic(array)` that is the fixed array, so passing the same array to `optimize` and `HMC` gives both the identical Gaussian likelihood. The SD is converted to the family's scale: `Normal` ×1, `SoftLaplace` ×2/π (its SD is π/2·scale), `Laplace` ×1/√2; any other family raises `ValueError`. No `sigma` site is sampled when `noise` is given. `noise` with `noise_cv` or with a surrogate raises `ValueError`.
- **Phase 2:** a field of `Homoskedastic`/`Proportional` may be a `catalax.mcmc.priors.Prior`. HMC samples it as a site named after the field (`sigma`, `cv`, `floor`); a name clash with a model parameter raises. The design code (`fisher_information`, `evaluate_design`, `optimize_design`) raises for noise with priors. `noise_cv` emits a `DeprecationWarning` pointing to `noise=Proportional(...)`.
- `catalax/noise.py` must not import `catalax.mcmc` (circular: `catalax/mcmc/mcmc.py` imports `catalax.noise`). Priors are recognised by duck typing: `hasattr(value, "_distribution_fun")`.

---

## Phase 1 — fixed noise in least squares and HMC

### Task 1: Move the noise module to `catalax/noise.py`

**Files:**
- Create: `catalax/noise.py` (content of `catalax/doe/noise.py`, docstring generalised)
- Modify: `catalax/doe/noise.py` (becomes a re-export), `catalax/doe/information.py:19`, `catalax/doe/optimize.py:20`, `catalax/doe/evaluate.py:19`, `catalax/__init__.py`
- Test: `tests/unit/test_noise.py`

**Interfaces:**
- Produces: `catalax.noise.{NoiseModel, Homoskedastic, Proportional, as_noise_model}`; `ctx.noise` module attribute. Same signatures as today.

- [ ] **Step 1: Write the failing test** — `tests/unit/test_noise.py`:

```python
"""catalax.noise: the shared observation-noise models."""

import jax.numpy as jnp
import pytest

import catalax as ctx
import catalax.doe as cdoe


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
```

- [ ] **Step 2: Run it** — `uv run pytest tests/unit/test_noise.py -q`. Expected: FAIL, `AttributeError: module 'catalax' has no attribute 'noise'`.

- [ ] **Step 3: Implement.**
  - `git mv catalax/doe/noise.py catalax/noise.py`. In its module docstring replace the first line with `"""Observation-noise models, shared by least squares, Bayesian inference and design.` and add one paragraph after it: `A noise model gives the standard deviation of every observation, ``scale(prediction)``. ``optimize`` evaluates it on the measured data (fixed weights), ``HMC`` on the model prediction, the design code uses it to whiten sensitivities.` Keep the rest. In `Homoskedastic`'s docstring change the `sigma` attribute line to `sigma: Noise standard deviation in data units; a scalar, or one value per data point (an array broadcastable to the data).`
  - New `catalax/doe/noise.py`:

```python
"""Moved to ``catalax.noise``; kept so existing imports work."""

from catalax.noise import Homoskedastic, NoiseModel, Proportional, as_noise_model

__all__ = ["Homoskedastic", "NoiseModel", "Proportional", "as_noise_model"]
```

  - In `catalax/doe/information.py`, `optimize.py`, `evaluate.py`: `from catalax.doe.noise import ...` → `from catalax.noise import ...`.
  - `catalax/__init__.py`: add `from . import noise` next to the other imports (after `from .tools.optimization import optimize`).

- [ ] **Step 4: Run** `uv run pytest tests/unit/test_noise.py tests/unit/doe -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add catalax/noise.py catalax/doe/noise.py catalax/doe/information.py catalax/doe/optimize.py catalax/doe/evaluate.py catalax/__init__.py tests/unit/test_noise.py
git commit -m "refactor: move noise models to catalax.noise

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2: Weighted least squares in `optimize(noise=...)`

**Files:**
- Modify: `catalax/tools/optimization.py`
- Test: `tests/unit/tools/test_optimize_noise.py`

**Interfaces:**
- Consumes: `catalax.noise.as_noise_model`, `NoiseModel` (Task 1).
- Produces: `ctx.optimize(model, dataset, ..., method="least_squares", noise: NoiseModel | jax.Array | float | None = None)`.

- [ ] **Step 1: Write the failing tests** — `tests/unit/tools/test_optimize_noise.py`:

```python
"""ctx.optimize(noise=...): weighted least squares with absolute standard deviations."""

import jax.numpy as jnp
import numpy as np
import pytest

import catalax as ctx

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


def _arrays(model, dataset):
    data, times, _ = dataset.to_jax_arrays(model.get_observable_state_order())
    return data, times


def _sigma(model, dataset):
    _, times = _arrays(model, dataset)
    return (0.1 + 0.05 * times)[..., None]  # grows with t, shape of the data


def _closed_form(model, dataset, sigma):
    """Weighted least squares for P = k t: estimate and absolute standard error."""
    data, times = _arrays(model, dataset)
    t, y, s = (np.asarray(a).ravel() for a in (times, data[..., 0], sigma))
    w = 1 / s**2
    return float(np.sum(w * t * y) / np.sum(w * t * t)), float(1 / np.sqrt(np.sum(w * t * t)))


def test_weighted_estimate_and_absolute_stderr():
    model, dataset = linear_problem()
    sigma = _sigma(model, dataset)
    result, _ = ctx.optimize(model, dataset, noise=sigma)
    k, se = _closed_form(model, dataset, sigma)
    assert result.params["k"].value == pytest.approx(k, rel=1e-4)
    assert result.params["k"].stderr == pytest.approx(se, rel=1e-2)


def test_stderr_follows_the_given_sigma():
    """Absolute SDs: ten times the noise means ten times the standard error, same estimate."""
    model, dataset = linear_problem()
    sigma = _sigma(model, dataset)
    small, _ = ctx.optimize(model, dataset, noise=sigma)
    large, _ = ctx.optimize(model, dataset, noise=10 * sigma)
    assert large.params["k"].value == pytest.approx(small.params["k"].value, rel=1e-4)
    assert large.params["k"].stderr == pytest.approx(10 * small.params["k"].stderr, rel=1e-2)


def test_noise_model_is_evaluated_on_the_data():
    model, dataset = linear_problem()
    noise = ctx.noise.Proportional(cv=0.1, floor=0.2)
    data, _ = _arrays(model, dataset)
    by_model, _ = ctx.optimize(model, dataset, noise=noise)
    by_array, _ = ctx.optimize(model, dataset, noise=noise.scale(data))
    assert by_model.params["k"].value == pytest.approx(by_array.params["k"].value, rel=1e-6)


def test_missing_points_are_left_out():
    model, dataset = linear_problem()
    dataset.measurements[0].data["P"] = dataset.measurements[0].data["P"].at[3].set(jnp.nan)
    sigma = _sigma(model, dataset)
    result, _ = ctx.optimize(model, dataset, noise=sigma)
    assert result.ndata == 5
    assert np.isfinite(result.params["k"].value)


def test_noise_and_objective_fun_conflict():
    model, dataset = linear_problem()
    with pytest.raises(ValueError, match="noise"):
        ctx.optimize(model, dataset, noise=0.1, objective_fun=lambda y, p: (y - p) ** 2)


def test_non_positive_sd_at_a_measured_point_raises():
    model, dataset = linear_problem()
    sigma = _sigma(model, dataset).at[0, 2, 0].set(0.0)
    with pytest.raises(ValueError, match="positive"):
        ctx.optimize(model, dataset, noise=sigma)
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/tools/test_optimize_noise.py -q`. Expected: FAIL, `TypeError: optimize() got an unexpected keyword argument 'noise'`.

- [ ] **Step 3: Implement** in `catalax/tools/optimization.py`:
  - Import: `from catalax.noise import NoiseModel, as_noise_model`.
  - Signature: add `noise: Optional[NoiseModel | jax.Array | float] = None` after `method`.
  - Docstring: replace the paragraph "By default this is ordinary least squares: ..." with:

```text
    By default this is ordinary least squares: lmfit receives the residual vector
    ``simulation - data`` and minimises its sum of squares; the noise level is estimated
    from the residuals, so the standard errors are scaled by the reduced chi-square.

    With ``noise`` it is weighted least squares. ``noise`` is a noise model (see
    ``catalax.noise``) or the standard deviations themselves (a float, or an array shaped
    like the observable data). It is evaluated once on the measured data, ``sd =
    noise.scale(data)``, and the residual becomes ``(simulation - data) / sd``. The
    standard deviations are taken as absolute: the standard errors are not rescaled, so
    report ``result.redchi`` (about 1 when the noise model fits the residuals).

    A per-point loss can be given instead as ``objective_fun(y_true, y_pred)`` returning a
    non-negative array; the fit then minimises the sum of that loss over the measured
    points. It cannot be combined with ``noise``.
```

  - Args entry: `noise (NoiseModel | Array | float, optional): Observation noise, as absolute standard deviations; see above. Defaults to None (ordinary least squares).`
  - Body, after `measured = ~jnp.isnan(data)`:

```python
    sigma = None
    if noise is not None:
        if objective_fun is not None:
            raise ValueError("Give either noise (weighted least squares) or objective_fun, not both.")
        # Evaluated on the measured values, once: fixed weights cannot be gamed by the fit.
        sigma = jnp.broadcast_to(as_noise_model(noise).scale(data), data.shape)
        if not bool(jnp.all(jnp.isfinite(sigma[measured]) & (sigma[measured] > 0))):
            raise ValueError("noise must give a finite, positive standard deviation at every measured point.")
```

  - Add `sigma` to `minimize_args` after `objective_fun`, and to `_residual`'s signature as `sigma: Optional[jax.Array]` (Args: `sigma (Optional[jax.Array]): Per-point standard deviations, or None for unweighted residuals.`).
  - `minimize(..., method=method, scale_covar=noise is None, **fit_kws)`.
  - In `_residual`: `residual = predicted - data` becomes

```python
    if objective_fun is None:
        residual = predicted - data
        if sigma is not None:
            residual = residual / sigma
```

- [ ] **Step 4: Run** `uv run pytest tests/unit/tools -q`. Expected: PASS (new file and the unchanged `test_optimize.py`).

- [ ] **Step 5: Commit**

```bash
git add catalax/tools/optimization.py tests/unit/tools/test_optimize_noise.py
git commit -m "feat(optimize): weighted least squares with absolute noise SDs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 3: Fixed noise in HMC

**Files:**
- Modify: `catalax/mcmc/mcmc.py` (`HMC.run`, `run_mcmc`, `BayesianModel.__init__`, `BayesianModel.__call__`)
- Test: `tests/unit/mcmc/test_noise_likelihood.py`

**Interfaces:**
- Consumes: `catalax.noise.as_noise_model`, `NoiseModel` (Task 1).
- Produces: `HMC.run(model, dataset, yerrs=None, surrogate=None, pre_model=None, post_model=None, abort_on_error=False, noise=None)`; `run_mcmc(model, dataset, yerrs, config, surrogate=None, pre_model=None, post_model=None, noise=None)` (`yerrs` may be None when `noise` is given); `BayesianModel(..., noise: NoiseModel | None = None, noise_factor: float = 1.0)`; module constant `SD_TO_SCALE: dict[type, float]`.

- [ ] **Step 1: Write the failing tests** — `tests/unit/mcmc/test_noise_likelihood.py`:

```python
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
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/mcmc/test_noise_likelihood.py -q`. Expected: FAIL, `ImportError: cannot import name 'SD_TO_SCALE'`.

- [ ] **Step 3: Implement** in `catalax/mcmc/mcmc.py`:
  - Imports: `import math` and `from catalax.noise import NoiseModel, as_noise_model`.
  - Module constant after the imports:

```python
# A noise model gives standard deviations; a likelihood family takes a scale. SD of
# SoftLaplace(0, s) is pi/2 * s, of Laplace(0, s) is sqrt(2) * s.
SD_TO_SCALE = {dist.Normal: 1.0, dist.SoftLaplace: 2 / math.pi, dist.Laplace: 1 / math.sqrt(2)}
```

  - `HMC.run`: signature `yerrs: Optional[Union[float, Array]] = None`, add `noise: Optional[Union[NoiseModel, Array, float]] = None` as the last parameter; Args: `yerrs: Standard deviation of observed data; sets the scale of the sampled sigma's prior. Not used with noise.` and `noise: Observation noise as standard deviations (see catalax.noise): fixed per-point SDs (array or float) or a noise model evaluated on the prediction. Replaces the sampled sigma.` Before `config = self.config.to_simulation_config()` insert:

```python
        noise_factor = 1.0
        if noise is not None:
            if self.config.noise_cv is not None:
                raise ValueError("Give either noise or noise_cv, not both.")
            if surrogate is not None:
                raise ValueError("noise is not supported with a surrogate model.")
            if self.config.likelihood not in SD_TO_SCALE:
                raise ValueError(
                    f"noise gives standard deviations; the likelihood {self.config.likelihood.__name__} "
                    f"has no known SD-to-scale conversion. Use one of {[d.__name__ for d in SD_TO_SCALE]}."
                )
            noise = as_noise_model(noise)
            noise_factor = SD_TO_SCALE[self.config.likelihood]
        elif yerrs is None:
            raise ValueError("Give yerrs (scale of the sampled sigma's prior) or noise.")
```

    Wrap the existing `if isinstance(yerrs, float) ... else raise` block in `if yerrs is not None:` and set `yerrs_ = None` otherwise. Pass `noise=noise, noise_factor=noise_factor` to `BayesianModel(...)`.
  - `run_mcmc`: add `noise=None` as the last parameter, type `Optional[Union[NoiseModel, Array, float]]`, yerrs type `Optional[Union[float, Array]]`; pass `noise=noise` to `hmc.run`. Docstring Args line for noise as in `HMC.run`.
  - `BayesianModel.__init__`: `yerrs: Optional[Array]`, add `noise: Optional[NoiseModel] = None, noise_factor: float = 1.0`, store as `self.noise`, `self.noise_factor`; Args lines: `noise: Observation noise model; replaces the sampled sigma when given` and `noise_factor: Factor from standard deviation to the likelihood family's scale`.
  - `BayesianModel.__call__`: directly before `# Sample noise parameter` insert:

```python
        if self.noise is not None:
            predicted = states[..., self.observables]
            sd = jnp.broadcast_to(self.noise.scale(predicted), predicted.shape)
            # unmeasured points are masked out; give them a harmless scale so no NaN reaches the gradient
            scale = jnp.where(mask, sd, 1.0) * self.noise_factor
            with numpyro.handlers.mask(mask=mask):
                numpyro.sample("y", self.likelihood(predicted, scale), obs=data)  # type: ignore
            return
```

- [ ] **Step 4: Run** `uv run pytest tests/unit/mcmc -q`. Expected: PASS, including the unchanged `test_sigma_prior.py`.

- [ ] **Step 5: Commit**

```bash
git add catalax/mcmc/mcmc.py tests/unit/mcmc/test_noise_likelihood.py
git commit -m "feat(mcmc): fixed observation noise via noise=, converted to the family's scale

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 4: Least squares and HMC agree under identical noise

**Files:**
- Test: `tests/integration/test_noise_agreement.py`

**Interfaces:**
- Consumes: `ctx.optimize(..., noise=)` (Task 2), `HMC.run(..., noise=)` (Task 3).

- [ ] **Step 1: Write the test.** For a model linear in its parameter with a flat prior and Normal likelihood, the posterior is exactly Gaussian with mean = weighted least-squares estimate and sd = its absolute standard error.

```python
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
```

- [ ] **Step 2: Run** `uv run pytest tests/integration/test_noise_agreement.py -q -m expensive`. Expected: PASS. If it fails, report the two means and SDs; do not loosen tolerances without saying so.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_noise_agreement.py
git commit -m "test: weighted least squares and HMC agree under identical fixed noise

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Phase 2 — noise parameters with priors

### Task 5: Priors as noise parameters, sampled by HMC, refused by the design code

**Files:**
- Modify: `catalax/noise.py`, `catalax/mcmc/mcmc.py`, `catalax/doe/information.py:126`
- Test: `tests/unit/test_noise.py` (extend), `tests/unit/mcmc/test_noise_likelihood.py` (extend)

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: `catalax.noise.prior_fields(noise) -> dict[str, Prior]`, `catalax.noise.sample_noise(noise) -> NoiseModel`; `as_noise_model(noise, fixed: bool = False)`.

- [ ] **Step 1: Write the failing tests.** Append to `tests/unit/test_noise.py`:

```python
import catalax.mcmc as cmm


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
```

Append to `tests/unit/mcmc/test_noise_likelihood.py`:

```python
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
```

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_noise.py tests/unit/mcmc/test_noise_likelihood.py -q`. Expected: FAIL, `AttributeError: module 'catalax.noise' has no attribute 'prior_fields'`.

- [ ] **Step 3: Implement.**
  - `catalax/noise.py`: add `import dataclasses` and `import numpyro`. Add:

```python
def _is_prior(value) -> bool:
    # Duck-typed: importing catalax.mcmc here would be circular (catalax.mcmc imports this module).
    return hasattr(value, "_distribution_fun")


def prior_fields(noise) -> dict:
    """The fields of a noise model that are priors (to be sampled), by name."""
    return {
        field.name: getattr(noise, field.name)
        for field in dataclasses.fields(noise)
        if _is_prior(getattr(noise, field.name))
    }


def sample_noise(noise):
    """Inside a numpyro model: the noise model with every prior field replaced by a sample.

    Each prior is sampled as a site named after its field (``sigma``, ``cv``, ``floor``).
    A noise model without priors is returned unchanged.
    """
    priors = prior_fields(noise)
    if not priors:
        return noise
    values = [numpyro.sample(name, prior._distribution_fun()) for name, prior in priors.items()]
    return eqx.tree_at(lambda m: [getattr(m, name) for name in priors], noise, values)
```

    In `Homoskedastic.__init__`: `self.sigma = sigma if _is_prior(sigma) else jnp.asarray(sigma, dtype=float)`. In `Proportional.__init__`: convert and check `floor` only if not a prior; `self.cv = cv if _is_prior(cv) else jnp.asarray(cv, dtype=float)`. Docstrings of both classes: add `Any field may be a catalax.mcmc.priors prior instead of a number; HMC then samples it. The design code needs numbers.`
    `as_noise_model(noise, fixed: bool = False)`: after building the model, `if fixed and prior_fields(model): raise ValueError(f"The design needs fixed noise values; got priors for {list(prior_fields(model))}.")`. Args line: `fixed: Raise if any field is a prior (the design code needs numbers).`
    Add `prior_fields`, `sample_noise` to `__all__`.
  - `catalax/doe/information.py:126`: `noise = as_noise_model(noise, fixed=True)`.
  - `catalax/mcmc/mcmc.py`: import `prior_fields, sample_noise`. In `BayesianModel.__init__`, after storing `self.noise`:

```python
        if noise is not None:
            clash = set(prior_fields(noise)) & set(model.parameters)
            if clash:
                raise ValueError(f"Noise prior names {sorted(clash)} clash with model parameters.")
```

    In `__call__`'s noise branch: `sd = jnp.broadcast_to(sample_noise(self.noise).scale(predicted), predicted.shape)`.

- [ ] **Step 4: Run** `uv run pytest tests/unit -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add catalax/noise.py catalax/mcmc/mcmc.py catalax/doe/information.py tests/unit/test_noise.py tests/unit/mcmc/test_noise_likelihood.py
git commit -m "feat(noise): priors as noise parameters, sampled by HMC

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 6: Deprecate `noise_cv`, update the docs

**Files:**
- Modify: `catalax/mcmc/mcmc.py` (`HMC.__init__`), `docs/basic/parameter-opt.mdx`, `docs/hmc/mcmc-basic.mdx`, `docs/doe/overview.mdx:19,223`
- Test: `tests/unit/mcmc/test_noise_likelihood.py` (extend)

- [ ] **Step 1: Write the failing test** (append):

```python
def test_noise_cv_is_deprecated():
    with pytest.warns(DeprecationWarning, match="Proportional"):
        cmm.HMC(num_warmup=1, num_samples=1, noise_cv=0.05, verbose=0)
```

- [ ] **Step 2: Run** it. Expected: FAIL (no warning).

- [ ] **Step 3: Implement.** In `HMC.__init__`, before building the config:

```python
        if noise_cv is not None:
            warnings.warn(
                "noise_cv is deprecated; pass noise=catalax.noise.Proportional(cv=..., floor=...) "
                "to HMC.run, with priors as fields to sample them.",
                DeprecationWarning,
                stacklevel=2,
            )
```

  (`import warnings` at the top.) Docs, each a short section in the style of the surrounding text:
  - `docs/basic/parameter-opt.mdx`: "Weighted least squares": `noise=` takes a noise model or per-point SDs; evaluated on the measured data once; SDs are absolute (no rescaling), report `result.redchi`; cannot be combined with `objective_fun`. Example with `ctx.noise.Proportional(cv=0.035, floor=0.006)`.
  - `docs/hmc/mcmc-basic.mdx`: "Observation noise": `noise=` replaces the sampled `sigma`; fixed array → same likelihood as weighted least squares; noise model → evaluated on the prediction; fields as priors are sampled (`cv`, `floor`); SDs converted to the family's scale (Normal, SoftLaplace, Laplace); `noise_cv` deprecated.
  - `docs/doe/overview.mdx`: line 19 mention `catalax.noise` and that the same objects go to `optimize` and `HMC`; line 223 replace the `noise_cv` advice with `HMC.run(..., noise=<the design's noise model>)` and `likelihood=dist.Normal`.

- [ ] **Step 4: Run** `uv run pytest tests/unit -q`. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add catalax/mcmc/mcmc.py docs/basic/parameter-opt.mdx docs/hmc/mcmc-basic.mdx docs/doe/overview.mdx tests/unit/mcmc/test_noise_likelihood.py
git commit -m "docs: one noise model across optimize, HMC and design; deprecate noise_cv

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

## Out of scope (later, in mervan)

- `build_enzymeml.py`: subtract the average t = 0 background (about 0.042 mM) instead of the calibration intercept.
- Check the early Co t = 0 points (6–9 July) before weighted fits.
- `scripts/fit_bayes.py` with the `MODELS` registry; the fixed comparison of Model 1.1 with `sigma = Proportional(cv=0.035, floor=0.006).scale(data)` passed to both `optimize` and `HMC.run`.
