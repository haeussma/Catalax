# Plan: proposal-based inner sampling in `evaluate_design` (catalax.doe, PR 4c)

After approval this plan is copied to
`Catalax/docs/plans/2026-10-02-doe-proposal-inner-sampling.md`.

## Context

`evaluate_design` reports the expected posterior sd per parameter (the ±% the user reads).
It does this in two loops:

- **Outer loop:** one draw per plausible enzyme θ_i from the priors.
- **Inner loop:** 65,536 candidates u_k, which `_reweighted_sd` weights by the Gaussian
  likelihood built from F(θ_i).

The inner candidates come from the **whole prior**. When a design pins the parameters far
below the prior width, almost none of them land in the posterior, and the effective sample
size (ESS) collapses. A collapsed ESS biases the sd low, i.e. **optimistic**.

Measured on the competitive product inhibition production run (log-uniform priors
spanning 40–100×):

| arms | 2 | 3 | 4 | 6 |
|---|---|---|---|---|
| ESS median | 78 | 32 | 19 | 11 |
| ESS min | 12 | 4 | 2 | 2 |

So the more informative the design, the less the report can be trusted. That's exactly
backwards for comparing 2 vs 6 arms.

**Fix:** importance sampling with a proposal centred on each draw's Laplace posterior.
- It computes the same quantity: the weighted sd under the exact prior × the
  F-Gaussian likelihood.
- Almost every sample now carries weight, so the ESS no longer falls as designs improve.
- It applies to the **prior path only**. A belief (MCMC samples) has no density, so the
  belief path keeps today's estimator, where its samples are the candidates.

## Design: `catalax/doe/evaluate.py` only

### The estimator, per draw i

Notation:
- u = log θ, with draw u_i;
- F_J = F_i + 1e-6·I and L = chol(F_J), as in `_reweighted_sd`. Use F_J
  **throughout**, so the Gaussian part targets exactly the likelihood that is weighted.
- The likelihood is `‖Lᵀ(u − u_i) − z‖² = (u − û)ᵀ F_J (u − û)` with
  û = u_i + L⁻ᵀz. **Never form û:** in weakly identified directions L⁻ᵀz is of order
  1e3 because of the jitter. Use F_J·û = F_J·u_i + L·z instead.
- Prior in log space: μ₀ and P₀ = diag(1/var), from the log of the existing prior inner
  samples.

**Proposal,** a defensive mixture (Hesterberg 1995):

`q(u) = (n_g/M)·N(u; m_i, c²·P⁻¹) + (n_p/M)·p(u)`

- **Gaussian part:** the Laplace posterior.
  - P = F_J + P₀, and m_i = P⁻¹(F_J·u_i + L·z + P₀·μ₀).
  - One Cholesky L_P of P serves everything:
    - m_i by two triangular solves;
    - samples m_i + c·L_P⁻ᵀε;
    - log-density from ‖L_Pᵀ(u − m)‖²/c² and Σ log diag(L_P).
  - Never form P⁻¹: at near-noiseless F it's ill-conditioned.
  - With c = 1 and a LogNormal prior, the Gaussian part is the exact posterior.
- **Prior part:** cheap insurance for a box corner or a near-singular F. It bounds the
  weights: q ≥ (n_p/M)·p gives w ≤ ℓ·M/n_p ≤ M/n_p.
  - What this guarantees: asymptotically, the ESS is at least that of prior sampling
    with n_p samples. That's finite variance, not a high ESS.
  - The docstring states the bound this way.
- **Validity:** this is a deterministic-mixture (balance-heuristic) estimator. Its
  coefficients are the **actual sample counts** n_g/M and n_p/M after rounding, not the
  nominal α.
  - With those, the normaliser estimate is unbiased, with variance no worse than a
    random mixture (Owen & Zhou 2000).
  - The only bias is the usual self-normalised O(1/ESS).

**Samples:**
- **n_g = M − n_p Gaussian samples.** ε = normal(fold_in(eps_key, i)) per draw inside
  `lax.map`. That's independent across draws, so the inner error averages out over the
  draws; memory is unchanged.
- **n_p = round(α·M) samples** from the existing prior inner samples, shared across draws.

**Why the target is benign** (from the review): in u, prior × likelihood is exactly a
truncated Gaussian, so there are no modes to miss.
- Uniform prior: N(û + F_J⁻¹·1, F_J⁻¹) on the log box.
- LogUniform prior: N(û, F_J⁻¹) on the box.
- LogNormal prior: a Gaussian with precision F_J + σ⁻².

So the estimator is consistent for any c > 0 and α ≥ 0.

**Weights:** `log w = log p(u) + log ℓ_i(u) − log q(u)`, then softmax. The sd and the
ESS = 1/Σw² are computed exactly as now.

**`log p(u)`** is the sum over parameters of `prior.log_prob(e^u) + u` (the Jacobian
into log space), **masked to −inf outside `prior.support`**. numpyro returns finite
values outside a Uniform or LogUniform box; checked: `Uniform(0.1, 5).log_prob(6.0)`
gives −1.59.

### Code changes

1. **New private `_proposal_sd(log_theta, z, F, eps, prior_inner, mu0, prec0, log_prior)
   -> (sd, ess)`.**
   - It is the prior-path twin of `_reweighted_sd`, with the same signature style and
     the same outputs.
   - `log_prior` is a closure built once from `_extract_priors`.
   - Mask with `jnp.where(support(θ), log_prob, -inf)`. It also guards
     `LogNormal.log_prob(0) = NaN`.
   - The prior part guarantees at least one finite log weight, so the softmax is never
     NaN.
   - `mvn` log-density via `jax.scipy.stats.multivariate_normal.logpdf` (or the
     Cholesky by hand).
2. **`evaluate_design`, prior path:**
   - split `inner_key` once more to get the ε block;
   - μ₀ = `inner.mean(0)`, `prec0 = diag(1 / inner.var(0))`;
   - `lax.map` over draws now calls `_proposal_sd`.
   - The belief path is unchanged and **bit-identical**.
   - `_prepare_efficiency` and the search are untouched and **bit-identical**.
3. **Constants:** `_PROPOSAL_SCALE` (c) and `_DEFENSIVE_FRACTION` (α).
   - They affect efficiency only, never the limit: the estimator is consistent for any
     c > 0 and α ≥ 0, with finite weight variance given the bounded Gaussian likelihood.
     That makes them allowed constants under the no-magic-thresholds rule.
   - Their values come from step 1 of Verification.
   - Each docstring carries its measurement table (house rule).
4. **Docstrings:**
   - `evaluate_design`, `_reweighted_sd` (now "belief path") and the
     `DesignReport.ess_*` fields: what ESS means now, and that for priors it no longer
     falls with design quality.
   - The old M = 4096 / 65536 tables stay, marked as the prior-sampling estimator,
     since they still describe the belief path.

### Docs: `docs/doe/overview.mdx`, "How far to trust the numbers"

- Rewrite the ESS paragraph:
  - with priors, the ESS stays high and the ±% stay trustworthy at any number of arms;
  - with a belief, the old guidance applies (longer chains).
- Update printed example outputs that change; the ESS numbers will.

### Out of scope

- A density for beliefs (KDE or Gaussian fit), which would bring the fix to round 2.
  It's an extra approximation; revisit if round-2 ESS matters in practice.
- Using the reweighted estimator in the search, which was measured to give no gain.
- Changing M = 65,536. Measure whether fewer samples suffice now; record it, don't
  change it in this PR.

## Tests: `tests/unit/doe/test_reweighted_report.py`

- **The existing 1-D quadrature oracle and LogNormal oracle must pass unchanged.** Their
  tolerances are 3 SE computed in the test, which is now smaller, so they get stricter
  automatically.
- **New oracle with teeth: a 3-parameter product model.**
  - Setup: three observed states, dP_j/dt = k_j, Uniform(1, 10) priors,
    `Proportional(cv=0.05, floor=1e-6)`, times [1, 2].
  - Then F = (2/cv²)·I = 800·I. The posterior factorises, so each parameter's exact sd
    is the existing 1-D quadrature at F = 800.
  - The new estimate must match within 3 SE, with the SE over replicate keys computed in
    the test.
  - Also assert the old prior-sampling estimator is biased low by more than 3 SE.
  - The reviewer's prototype gave:
    - old: −4.4 to −6.5 SE, ESS median 19 and minimum 1;
    - new (c = 1.5, α = 0.25): +0.04, +0.93 and −0.90 SE, ESS median 28,205.
  - A 1-D version has no teeth: the collapse is roughly M·Π(σ_post/σ_prior), so it needs
    several parameters. At cv 0.02 in 1-D, the old estimator was only −0.98 SE off.
- **Replace `test_near_noiseless_design_shows_a_collapsed_ess`.** Its premise is the
  defect being fixed, so this is a behaviour change, not a re-tolerance; the PR says so.
  - New version: the near-noiseless design keeps `ess_median` above 10% of the inner
    sample count. The ceiling is about 0.48·(1 − α)·M at 4 parameters with c = 1.5, so
    this is safe.
  - Its `relative_sd` matches sqrt(diag(inv(F_J))) per draw.
    - That's the exact sd away from the box edges, because the target is a truncated
      Gaussian. It is not Laplace, which adds a prior precision the box doesn't have.
    - Paired draw for draw over N_REPLICATES keys, like
      `test_lognormal_prior_gives_the_laplace_sd`: the SE of the paired difference is
      computed in the test.
    - Use a design whose draws sit away from the edges, or exclude draws within 3
      posterior sd of an edge, so truncation doesn't make the result depend on the key.
- **Mask test:** a Uniform-prior draw whose Laplace mean lies outside the box gives
  finite, in-box results. Proposal samples outside the box get weight 0.
- **Belief path:** the existing `test_belief_path_weights_the_belief_samples` runs
  unchanged. Bit-identity is checked in Verification, not pinned in a test.

Full suite: `uv run pytest -m "not expensive" -q` gives 124 plus the new tests. Run ruff
only on new files.

## Verification

1. **Choose c and α by measurement** (scratchpad script):
   - **Grid:** c ∈ {1, 1.25, 1.5, 2}, α ∈ {0.05, 0.1, 0.25}. Expect a small α (about
     0.1) to win: prior samples carry almost no weight when the posterior is narrow.
   - **Workloads:**
     - the competitive product inhibition designs for 2 and 6 arms (cached in
       `progress-doe/research/figs/_competitive_pi_arms.json`);
     - MAT for 4 and 16 arms.
   - **Reference:** the new estimator at large M (about 4M) with a different (c, α) and
     independent keys.
     - Prior sampling at 4M is too weak: at 6 arms its minimum ESS would be about 128,
       still biased in exactly the draws that decide the choice.
     - Also report the old reference's own ESS as a cross-check.
   - **Per cell:** ESS median and minimum, and the sd ratio to the reference, over
     replicate keys, so "within MC error" has a number.
   - Pick the cell with the best minimum ESS whose ratios all sit within that error.
     Put the table in the docstring.
2. **Bit-identity:** `_prepare_efficiency` and `optimize_design`'s `restart_scores`
   (reuse `scratchpad/bitid.py`), and the belief path.
3. **Compare against NUTS:** rerun `catalax_competitive_pi_arms.py`. It's cached, so
   only rescoring, MCMC and the plot run, about 1 min.
   - Expect the ESS in the thousands for every arm count.
   - Record the predicted ±% against NUTS per arm count in the PR and in
     `progress-doe/research/README.md`.
   - Also redo the 4 designs × 7 truths table of the reweighted report on MAT (scratchpad
     `rw_step4`), to check that the agreement with NUTS hasn't degraded.
4. **Timing:** `evaluate_design` at 1024 draws, before and after. Put it in the
   docstring.
5. Run the docs examples end to end. Run the expensive DOE tests.
