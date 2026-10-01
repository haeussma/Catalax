# Plan: honest design report (catalax.doe, PR 4b)

## Context

`evaluate_design` reports the Laplace efficiency, `Σ_post = (F + Σ₀⁻¹)⁻¹`. With
Uniform priors (the user's default), Σ₀⁻¹ replaces the box with a round Gaussian. That
makes the reported numbers optimistic: the 2-arm reference is predicted at 0.182
worst-parameter efficiency, and NUTS measures 0.097.

The cross-evaluation (2026-10-01, scratchpad `rw_*.py`/`.log`) showed this affects the
**numbers, not the design**:
- A search on a corrected criterion lands in the same basin.
- Under NUTS the two designs tie: −0.003 ± 0.007 over 7 truths.
- Laplace ranks designs correctly (Spearman 0.97 within a truth).

So **the search stays Laplace**, and only the **report** changes. It switches to a
reweighted estimator: the Gaussian likelihood built from F, combined with the *exact*
prior through importance weights over prior samples. The report also states what the
user actually wants to know: the expected ±% uncertainty per parameter after the
experiment.

**Precondition:** PR 4 is committed first. It is still uncommitted, and the user's OK
is pending.

## Measured (scratchpad `rw_step1/2/5.log`)

**Predicted sd / fixed-noise NUTS sd** (kcat, ki_u, km_a, km_b), midpoint truth,
3 seeds, M = 65536:

| estimator | 4 arms | 16 arms |
|---|---|---|
| Laplace | .70 .71 .84 .82 | .79 .80 .87 .96 |
| reweighted, data-averaged (b) | .91 .92 .96 .92 | 1.01 1.02 1.03 1.00 |
| reweighted, at truth (a) | 1.00 1.01 1.03 .99 | 1.11 1.12 1.13 1.05 |

**Over 28 design × truth cells:**

| | Laplace | (b) |
|---|---|---|
| sd ratio, 10–90% | 0.81–1.55 | 0.84–1.12 |
| bias in worst-parameter e | +0.036 | +0.030 |

**Effective sample size (ESS) at M = 4096**, median over draws: 61 for 4 arms and 24
for 16 arms. M = 65536 raises that about 16×.

## Design

### 1. `catalax/doe/evaluate.py`

- **`_prepare_efficiency` (the search path) is unchanged and stays bit-identical.**
- **Same outer draws as now for a given key.** Keep the `draw_key, cov_key` split, so
  Laplace and reweighted numbers are comparable draw for draw.
- **New private `_reweighted_sd(...)`**, used only by `evaluate_design`. Variant (b),
  as prototyped in `scratchpad/rw_common.py:rw_sd`:
  - `L = chol(F_i + 1e-6·I)`.
  - `w_k ∝ exp(−½‖Lᵀ(u_k − log θ_i) − z_i‖²)`, normalised with logsumexp. `z_i` is a
    fixed standard normal, so the weights average over where noisy data would land.
  - Posterior `sd_ij` is the weighted sd of `u_k`.
  - Inner samples `u_k`:
    - prior path: M = 65536 log-samples from the priors (key derived from `cov_key`);
    - belief path: all belief samples in log space, without resampling.
  - `sd_prior_j` is the sd of the inner samples, in both paths.
  - Map over outer draws with `lax.map`, not `vmap`. The vmapped `(256, 65536, p)`
    array is about 0.5 GB.
- **Why (b) over (a):** the report means the expected posterior sd over possible
  datasets, which is variant (b) by definition. (b) also matches NUTS best at
  converged M; the step 2 table above shows it.
- **Fallback:** when a draw's ESS < 100, use Laplace sd for that draw. Low ESS means
  the data dominate, which is where Laplace is accurate. The report needs no gradient,
  so a hard switch is fine. Count the fallbacks.
  - `# ponytail:` with a belief of about 1000 samples, the fallback fires often. The
    upgrade path is longer chains.
- **`DesignReport`**, all numbers from the reweighted sd:
  - `efficiency`, `maximin`: same meaning as now. `efficiency_j = 1 − relative_sd_j /
    prior_relative_sd_j` holds exactly.
  - new `relative_sd: dict[str, float]`: expected posterior sd of log θ. That's about
    ±relative uncertainty at one sd; the multiplicative form is ×/÷ exp(sd).
  - new `prior_relative_sd: dict[str, float]`.
  - new `n_fallback: int`.
  - a short `__str__`, one line per parameter, e.g.
    `kcat  ±21% (×/÷1.23)   prior ±38%   efficiency 0.44`, plus the maximin line.
- **The docstring** says the report is reweighted and the search is Laplace, and why,
  quoting the measurement.

### 2. `catalax/doe/optimize.py`

No logic change. `result.report` is `evaluate_design`, so it becomes reweighted
automatically. `restart_scores` stays Laplace, from the search's hard re-rank; its
docstring says so.

### 3. Docs, `docs/doe/overview.mdx`

- **"Posterior shrinkage and efficiency":** how to read `relative_sd`, with the worked
  example (prior ±38% → about ±21%). The report is reweighted (the exact prior); the
  search is Laplace (cheap, ranks correctly).
- **Replace "How far to trust the efficiency numbers"** with:
  - the table above;
  - the cross-evaluation result: Laplace and reweighted search find the same design
    under NUTS;
  - the remaining limits: likelihood curvature about ±10%, the model and noise model
    assumed correct, and the truth inside the box.
- Re-run every docs example and update the printed numbers.

### Out of scope

- Using the reweighted estimator in the search (measured: no gain).
- Nested importance sampling with the true likelihood (finalist re-rank). Revisit if
  F-based numbers prove too coarse on a model with stronger nonlinearity.
- The progress-doe `research-directions.md` note, which needs separate approval.

## Tests

**Unit, `tests/unit/doe/test_reweighted_report.py`:**

- **Exact 1-D oracle.** Model `dP/dt = k` with P observable, `Proportional(cv, floor=1e-6)`
  and times > 0. Then F = n/cv², independent of θ.
  - With a Uniform prior on k, the reweighted `relative_sd` must equal the 1-D
    quadrature of E_{u~prior, û~N(u, 1/F)}[sd(u | û)] (scipy `quad`; prior density
    ∝ e^u on the log box).
  - Tolerance is 3 Monte Carlo SE, computed in the test.
  - This is the independent reference.
- **LogNormal oracle.** With LogNormal priors, the reweighted result must equal Laplace,
  since Gaussian × Gaussian makes the posterior sd independent of the data. Tolerance is
  3 SE.
- **Consistency:**
  - `efficiency_j == 1 − relative_sd_j / prior_relative_sd_j`;
  - `maximin <= min(efficiency)`;
  - same key gives the same report.
- **Fallback:** a near-noiseless design gives `n_fallback > 0`, and its numbers match
  Laplace.
- **Belief path:** runs, and `prior_relative_sd` equals the sd of the belief's log
  samples.

**Changes to existing tests (all stated in the PR):**

- `test_doe.py::test_efficiency_in_unit_interval_and_monotone_in_design`: drop
  `0 <= e`. Laplace guaranteed it by construction; the true posterior doesn't (the
  reweighted RAND row came out at −0.009). Keep `e <= 1`, `maximin <= min(e)` and the
  monotonicity checks.
- `test_optimize_design.py::test_optimum_beats_the_midpoint_design` and the
  `test_optimize_oracle.py` gap now go through the reweighted scorer. Run them as they
  are. If either fails, investigate; **never re-tolerance**. Report the oracle gap
  under both scorers.
- `_prepare_efficiency`-based tests (`test_belief.py`) are unchanged.

**Full suite:** `uv run pytest -m "not expensive" -q` gives 113 plus the new tests.
Run ruff only on new files.

## Verification

1. Unit tests, and the bit-identity of `_prepare_efficiency` (reuse
   `scratchpad/bitid.py`, restricted to the search path and `restart_scores`).
2. `-m expensive tests/integration/doe -s`. Print each design's report.
3. Spot-check against NUTS: the report's `relative_sd` for the 4-arm design L, against
   the mean NUTS log-sd over the 7 truths in `rw_step4_nuts_*.log`. Record it in the
   PR; don't assert it.
4. Time `evaluate_design` at 256 draws with M = 65536 and put the number in the
   docstring.
5. Run the docs examples end to end.


## Deviation (2026-10-01)

**No Laplace fallback.** A switch at ESS < 100, and later < 30, was tried. It was
dropped because any threshold is arbitrary and decides which formula produces the
number. Measurement also showed that low ESS does not mean Laplace is accurate. The
report returns `ess_min` and `ess_median` instead, so the user can judge.
