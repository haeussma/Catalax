# Plan: close the loop (catalax.doe, PR 4)

## Context

PR 2b (f47cbe5) designs round 1 against the model's priors. PR 4 closes the loop:
1. Fit the measured round-1 data with `run_mcmc`.
2. Design round 2 against that posterior instead of the priors.
3. Check how far the Laplace approximation behind every efficiency number agrees with
   a real NUTS posterior.

There is no lab data yet, so the PR is built on **simulated data**. That is the only
setup where the truth is known, so coverage and the Laplace check can be verified at
all. When real data arrives, it replaces the simulated `Dataset`, and nothing else
changes.

**Limit, stated in the docs:** simulated data comes from the fitted model itself, so it
can't reveal a wrong rate law. Checking the mechanism belongs to real data and is out of
scope.

## Measured before planning (2026-09-30, scratchpad `probe_*.py`)

**Setup:** MAT with E as an unobservable state (fixture `doe_mat_design_oracle.json`);
truth at the geometric midpoints of the Uniform priors; `Proportional(0.05, 5)`;
6 time points.

1. **The loop runs today with no new mcmc API.** The pieces are:
   - `model.simulate(design, config, saveat=design.to_time_matrix())` returns a
     `Dataset`.
   - Add noise with `noise.scale(y)`. The probe dropped the unobservable species from
     `meas.data`. `run_mcmc` should read observables only, so leaving them in is
     probably fine; the implementer must verify that.
   - `run_mcmc(..., MCMCConfig(likelihood=dist.Normal, noise_cv=0.1))`. One chain,
     500/1000, takes **35 s** at 4 arms.
   - `HMCResults.get_samples()` gives `{kcat, ki_u, km_a, km_b, sigma, cv}` in natural
     units.
2. **The `yerrs` bug in `mcmc.py:465` is real.**
   - Reproducer: states A (unobservable), B and C (observable), `yerrs = [1, 100]`.
     The sigma prior scale comes out as **100**; the correct value is **50.5**.
   - Cause: `self.yerrs` is already in observable order, but it is indexed with
     *modeled* positions [1, 2]. JAX clamps the out-of-range index 2 to 1.
   - Impact is limited to the HalfNormal scale of the `sigma` prior. A scalar `yerrs`
     is unaffected. The surrogate arrays (`sigma_surr`, `rate_sigma`) are a separate
     path and are not touched.
3. **Laplace is overconfident here, against progress-doe's 0.99–1.05.** The table gives
   the Laplace/NUTS posterior-sd ratio at the truth, per parameter
   (kcat, ki_u, km_a, km_b), with the NUTS noise fixed at the truth via
   `numpyro.handlers.condition`:

   | design | seed 3 | seed 4 | seed 5 |
   |---|---|---|---|
   | 4 arms | .73 .73 .83 .86 | .67 .68 .83 .83 | .71 .72 .85 .77 |
   | 16 random arms | .84 .83 .95 1.02 | .76 .78 .89 .94 | — |

   - It moves toward 1 with more data. The worst case is the kcat/ki_u pair, whose
     posterior log correlation is −0.97, so the posterior is a ridge.
   - The likely suspects are the Uniform prior (progress-doe fitted a Normal in log
     space) and a design too weak for the likelihood to dominate. This PR diagnoses the
     cause (see §6); it does not fix it.
4. **Letting NUTS infer the noise confounds the check.** In seed 3 the fitted floor
   `sigma` was 1.4 against a true 5, which pushed the ratio to 1.34 in the opposite
   direction. So the Laplace check fixes the noise, while the user-facing loop keeps
   `run_mcmc`'s inferred noise, which is the realistic setting.
5. **Parallel chains work:** 4 chains take 80 s sequentially and **30 s** with
   `ctx.set_host_count(4)` + `chain_method="parallel"`. There's nothing to build.

## Design

### 1. `catalax/mcmc/mcmc.py`: fix the `yerrs` alignment (test first)

- Line 465 becomes `dist.HalfNormal(jnp.mean(self.yerrs))`.
- Rewrite the `ponytail:` comment at lines 397-398. `self.observables` indexes
  `states` only, which is correct.
- Leave the surrogate arrays alone.
- Write the failing test first: `tests/unit/mcmc/`, using the A/B/C reproducer above
  and reading `trace["sigma"]["fn"].scale` from `numpyro.handlers.trace`.

### 2. `catalax/doe/evaluate.py`: a belief from samples

Add `belief: Mapping[str, ArrayLike] | None = None` to `evaluate_design` and
`_prepare_efficiency`. It takes posterior samples in natural units, e.g.
`results.get_samples()`, and `None` means the priors.

**The `None` path stays bit-identical:** same key splits, and the existing tests are
unchanged.

**Samples path:**
- Validation, at the boundary:
  - every free parameter must be present; extra keys such as `sigma`, `cv` or fixed
    parameters are ignored;
  - all arrays are 1-D and the same length;
  - all values are finite and > 0;
  - `n_samples > n_free`, since the covariance needs it.

  Each error names the parameter and the fix.
- `log_samples = log(stack(...))`.
- Draws: `jax.random.choice(draw_key, n_samples, (n_draws,), replace=True)`. This is a
  bootstrap resample, as in progress-doe `beliefs._resample_rows`. Truncating would
  bias the draws toward one autocorrelated stretch of the chain.
- `Σ₀ = jnp.cov(log_samples, rowvar=False)`, the **full** covariance, because
  posteriors are correlated (kcat/ki_u was −0.97). Then `prior_precision = inv(Σ₀)`
  and `prior_var = diag(Σ₀)`. The efficiency formula is unchanged.
- **Efficiency is now relative to the belief.** `DesignReport.efficiency` means
  "shrinkage beyond what round 1 already taught", as the docstring will say.

### 3. `catalax/doe/optimize.py`

Add `belief=None`, passed through to both `_prepare_efficiency` calls and to the final
`evaluate_design`. That's the whole change.

### 4. Simulated data: reuse `model.simulate`, no new function

The truth is the model's parameter `.value`s. Noise comes from the **same**
`NoiseModel` the design used, via `noise.scale`; a mismatch there would show up as a
spurious Laplace gap. The `Dataset.augment` path can't do floor + cv, and it reuses one
seed for every species. The docs and tests use:

```python
data = model.simulate(design, config, saveat=design.to_time_matrix())
for meas in data.measurements:
    key, k = jax.random.split(key)
    y = meas.data["P"]
    meas.data["P"] = y + noise.scale(y) * jax.random.normal(k, y.shape)
```

### 5. Docs: new section "Closing the loop" in `docs/doe/overview.mdx`

- The round-1 → `model.simulate` + noise → `run_mcmc` → `belief=` → round-2 recipe.
  Use `likelihood=dist.Normal` and `noise_cv` set to match the design's noise model;
  the default `SoftLaplace` only rescales the information, but it should match.
- Merge the rounds with `add_measurement` and **refit all data under the original
  priors**, never "last posterior as the new prior". This is progress-doe's
  `inference.py` rule: it stays exact with finite draws.
- `ctx.set_host_count(n)` must be called before any JAX op, with
  `chain_method="parallel"` (measured 80 s → 30 s for 4 chains).
- **How far to trust the numbers:** the Laplace/NUTS table above, extended by the
  measurement in step 6. Add a note that progress-doe found greedy rounds lose to one
  joint design (§7, `research-directions.md`). The loop earns its keep when the prior
  is wrong, not by default.

### 6. The Laplace check: a measurement, not an assertion

This produces numbers; it doesn't enforce a tolerance.

- Script in the session scratchpad, pasted into the PR description as in PR 2b.
- Fixed-noise NUTS via `numpyro.handlers.condition(res.bayesian_model, {"sigma", "cv"})`,
  so no mcmc API change is needed.
- Grid: {4, 16} arms × 3 seeds × {Uniform, LogNormal prior matched in log mean and
  variance}.
- The prior arm is the diagnosis:
  - if LogNormal brings the ratio to about 1, the Uniform prior's Gaussian stand-in is
    the cause;
  - if not, it's the ridge or the design.
- The result goes into the docs.
- The progress-doe side, a note in `research-directions.md` §7, **only with the user's
  approval**, because that file has uncommitted shared edits.

### Out of scope

- A known-noise option in `run_mcmc`. Revisit if the step 6 result says the inferred
  noise matters for design.
- Non-Gaussian beliefs (e.g. KDE), and any fix for the Laplace gap.
- Parallel restarts in `optimize_design`. This is **PR 2c**, below.
- `constant=True` parameters: `run_mcmc` still samples them from their priors, while
  doe holds them fixed. Document it; don't fix it.

## Tests (x64, 16 draws, as in 2b)

**Unit, `tests/unit/doe/test_belief.py`:**
- Samples drawn from independent LogNormals match the LogNormal-prior path within
  Monte Carlo error. State the tolerance as 3 standard errors, with the SE computed
  in the test.
- A correlated belief uses the off-diagonal terms: its result differs from the same
  samples with the columns shuffled independently.
- Validation: a missing parameter, a sample ≤ 0, ragged lengths, too few samples.
- `optimize_design(..., belief=...)` runs and is deterministic by key.

**Unit, mcmc:** the `yerrs` test from §1.

**Integration, `tests/integration/doe/test_close_the_loop.py`** (`@expensive`, about
3 min):
1. MAT round 1 with `optimize_design`, then `model.simulate` plus noise, then
   `run_mcmc`.
2. Assert the truth lies inside the posterior (|z| < 3 per parameter).
3. Round 2 with `belief=results.get_samples()`, then simulate, merge, and refit under
   the priors.
4. Assert that the worst-parameter posterior sd shrinks from round 1 to round 2.
5. Print the Laplace/NUTS ratios for the PR description; don't assert them.

**Full suite:** `uv run pytest -m "not expensive" -q` gives 102 plus the new tests. Run
ruff only on new files.

## Verification

1. The `yerrs` test fails before the fix and passes after it.
2. `uv run pytest tests/unit/doe tests/unit/mcmc -q`, then `-m expensive
   tests/integration/doe`.
3. The full suite, with nothing lost.
4. Run the docs example end to end.
5. The step 6 table is filled in, and the docs say which regime the Laplace numbers
   hold in.

## Next: PR 2c, cores for `optimize_design` (your question)

- **numpyro: yes, the core count must be set explicitly, and it's already possible.**
  Call `ctx.set_host_count(n)` before JAX starts. It can't be done inside `run_mcmc`,
  because by then it's too late.
- **Optimiser: today it effectively uses about 1 core.** The restarts are vmapped, and
  progress-doe measured only 1.2× for 16 restarts that way. progress-doe's fix was a
  thread pool of per-restart jitted ascents, since JAX releases the GIL: 3.7× at
  4 workers, 6.3× at 8, and **slower at 12**, because the extra threads land on the
  M4 Pro's efficiency cores.
  - So yes, a worker cap: `n_workers = min(n_restarts, 8, os.cpu_count())`, exposed as
    a keyword.
  - Also try `pmap`: it failed in progress-doe only because of optimistix's root-find,
    which Catalax's ODE path doesn't have.
- Measure both, keep the faster one, and keep results bit-identical to the vmapped
  path.

