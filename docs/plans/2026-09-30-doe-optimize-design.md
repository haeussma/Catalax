# Plan: `optimize_design` (catalax.doe, PR 2b)

## Goal

Find the initial conditions of `n_arms` experiments that maximise the expected
worst-parameter posterior shrinkage (maximin efficiency) under the model's priors,
and return them as a lab-ready `Dataset`. It builds on PR 2a (28750a9), where the
information function is differentiable in `(y0s, constants, times)`.

```python
spec = {"A": (10, 2000), "B": (10, 1000), "E": (0.2, 0.2), "P": (0, 0)}
result = cdoe.optimize_design(model, spec, noise, n_arms=4, times=times, key=key)
result.dataset      # one add_initial(...) arm per experiment
result.report       # DesignReport of the winner, scored on fresh draws
```

Named `optimize_*` to match Catalax (`catalax/tools/optimization.py: optimize`),
not progress-doe's `optimise_*`.

## Decisions (agreed 2026-09-30)

- **Sampling times are fixed.** `times` is one grid, shared by every arm. progress-doe
  never optimised times either, and measured why not
  (`examples/optimise_four_arms.py`): an optimised schedule specialises toward the
  prior median. At 3 arms it lifts the mean 6.4% but raises the fraction of prior
  draws below 0.05 efficiency from 0.4% to 2.2%. This also removes the
  ordered-times parameterisation that PR 2a deferred.
- **Spec is a plain mapping `name -> (low, high)`**, one entry for **every** state
  and constant of the model. There are no defaults, and enzyme, substrate and
  product are treated alike. **`low == high` means fixed**: the column is left out
  of the optimised vector and filled in at that value. There is no `Fixed`/`Free`
  class and no `DesignSpec` class. PR 3 (prep-error CVs) may need a richer
  per-column value; add it then.
- **Development and tests use 16 prior draws.**

## Measured before planning (2026-09-30)

One `value_and_grad` of the smoothed maximin with respect to `y0s`: MAT (A, B, E
state, P), 4 arms × 9 times, Tsit5, forward sensitivities, CPU, x64. Probe:
`probe_step_cost.py` in the session scratchpad, pasted into the PR description.

| draws | per step | 8 restarts × 300 steps | 24 × 600 (progress-doe defaults) |
|---|---|---|---|
| 16 | 27 ms | ~1 min | 6.4 min |
| 64 | 105 ms | ~4 min | 25 min |
| 256 | 499 ms | 20 min | 2 h |

The first compile takes about 4.5 s, and every gradient was finite. Cost is linear
in draws. progress-doe's defaults were tuned on the closed form, and on this ODE
path they cost hours, so the defaults below are smaller. Put this table in the
`n_draws` docstring.

## Lessons ported from progress-doe (`progress_doe/optimize.py`, measured there)

1. **Multistart is mandatory.** The maximin landscape has several basins. In one
   configuration, 8 of 24 Latin-hypercube starts landed in the lower basin, 14%
   short. `n_restarts >= 1` is validated, and the docstring says 1 is not useful.
2. **Latin-hypercube starts.** Each free column's range is cut into
   `n_restarts × n_arms` strata, and each stratum gets exactly one start point.
   Arms are drawn independently, so no start has two identical arms. Identical
   arms get identical gradients forever and stay stuck at about 0.25.
   **`starts=` is not exposed in 2b**, so that trap can't be entered.
3. **Sigmoid box, never a clip**: `d = low + (high - low) · sigmoid(z)`. Optima sit
   on bounds (enzyme ceiling, stock ceiling), and a clip gives an exactly-zero
   gradient there.
4. **Common random numbers.** Every restart and every step scores against the
   same fixed prior draws, so the objective is a fixed surface and restarts can
   be compared draw for draw.
5. **Keep each restart's best point along its trajectory** (by smoothed value),
   not its last point, because Adam can step off the ridge on its final step.
6. **Ascend on the smoothed objective, rank on the hard one.**
   - Smoothed: softmin `−T · logsumexp(−e/T)`, T = 0.01, which costs under 0.001
     of efficiency.
   - Hard: `mean_i min_j e_ij`, on a **separate, larger** draw set
     (`n_rank_draws`). Ranking runs once per restart, so it is cheap.
7. **Failed draws are a reward-hacking channel.** Dropping non-finite draws and
   averaging over the rest can make a failing design score *higher*.
   - In the ascent: the mean over valid draws uses a denominator of
     `max(n_valid, 1)`. When every draw fails, the value is 0 with zero gradient,
     not NaN.
   - In the ranking: a restart with **any** invalid ranking draw is dropped.
   - If no clean restart is left, raise `RuntimeError`. The message names the
     fixes: narrow the bounds, raise `config.max_steps`, or use a stiff solver.

## Design

### 1. `catalax/doe/evaluate.py`: share the efficiency function

Factor out of `evaluate_design` one private helper, for example
`_prepare_efficiency(model, design, noise, *, key, n_draws, config)`. It returns:
- `efficiency(y0s, constants) -> (e[n_draws, n_free], valid[n_draws])` over a
  fixed set of prior draws;
- `parameter_order`;
- the design arrays.

`evaluate_design` must give **bit-identical** numbers afterwards: same key
splitting, and existing tests unchanged. `optimize_design` calls the helper twice,
once for the ascent draws and once for the ranking draws. Building the `Simulation`
twice is cheap; don't share it speculatively.

### 2. `catalax/doe/optimize.py` (new)

```python
def optimize_design(
    model: Model,
    spec: Mapping[str, tuple[float, float]],
    noise: NoiseModel | float,
    *,
    n_arms: int,
    times: Sequence[float] | jax.Array,
    key: jax.Array,
    n_restarts: int = 8,
    n_steps: int = 300,
    n_draws: int = 64,
    n_rank_draws: int = 256,
    learning_rate: float = 0.1,
    temperature: float = 0.01,
    config: SimulationConfig | None = None,
) -> DesignResult
```

**Boundary validation** (plain Python, before anything is traced). Each error
names the column and the fix:
- The spec's keys must equal the model's states plus constants exactly. Missing
  or unknown names raise.
- `low > high` raises. `n_arms`, `n_restarts`, `n_steps` and the draw counts must
  each be ≥ 1.
- At least one column must be free (`low < high`); otherwise suggest
  `evaluate_design`.
- Build a **template `Dataset`**: `n_arms` arms at the box midpoints, via
  `Dataset.from_model(model)` and `add_initial(time=times, **ics)`. This reuses
  `_extract_design_arrays`'s checks on times, and fixes the default config
  (`t1 = max(times)`) outside traced code.

**Traced search:**
- Free columns map to `(target, index)`, where the target is the state order
  (`y0s`) or the constant order (`constants`). `z` has shape
  `(n_arms, n_free_columns)`. `_to_design(z)` applies the sigmoid box and then
  `.at[:, idx].set(...)` onto the template arrays.
- `key` splits into `start_key`, `ascent_key`, `rank_key` and `report_key`.
- The ascent is a `lax.scan` over `n_steps` optax-Adam updates on `−smoothed(z)`,
  carrying `(z, opt_state, best_value, best_z)`. It is `jax.vmap`ped over
  restarts and `jit`ted once.
  - `# ponytail:` no progress bar or chunked scan. Add one if people wait more
    than a few minutes, following progress-doe's `_ascent._STEP_CHUNK`.
  - Restarts are vmapped, not threaded, so the cost is `n_restarts ×` the table
    above.
- Ranking: score each restart's `best_z` on the hard criterion with the rank
  draws, drop unclean restarts, and take the argmax.

**`DesignResult`** is a `@dataclass`, like `DesignReport`:
- `dataset`: `Dataset`, one `add_initial` arm per experiment, with fixed columns
  at their value.
- `report`: `DesignReport` from `evaluate_design(model, dataset, noise,
  key=report_key, n_draws=n_rank_draws, config=config)`. The draws are fresh
  because the winner was selected on the rank draws, which would bias its own
  score upwards.
- `restart_scores`: `list[float]`, the hard maximin of each restart (NaN if
  dropped), so the spread across basins is visible.

Export `DesignResult` and `optimize_design` in `catalax/doe/__init__.py`.

### 3. Docs

In `docs/doe/overview.mdx`, add a section "Optimising a design": the spec
mapping, `low == high` meaning fixed, fixed times, the cost table and what the
restart spread means. Extend the existing Michaelis–Menten example, with
`observable=` passed on every `add_ode`.

### 4. Out of scope

- Optimised sampling times, and caller-supplied `starts`.
- Excluding parameters from the objective (progress-doe's `optimise=`). In
  Catalax, set `constant=True` on the parameter instead.
- Trajectory output, plots, `sweep_n_arms`, adding one arm at a time, a progress
  bar.
- Sensitivities for free parameters only (the `ponytail:` note from PR 2a stays).
- Prep-error CVs (PR 3).

## Tests

x64 fixture, `tests/unit/doe/` and `tests/integration/doe/`. Always 16 draws.

**Unit, `tests/unit/doe/test_optimize_design.py`.** Tiny model: one-substrate
Michaelis–Menten, `s`, `p` and `e` as states, `e` with `observable=False`,
Uniform priors. Use 4 restarts × 50 steps.
- `low == high` columns come back exactly at their value, and the free columns lie
  inside their bounds. Cover both paths: a fixed state **and** a free constant.
- The optimum's hard maximin is ≥ that of the midpoint template design, both
  scored with `evaluate_design` on the same key.
- The same key gives the same design.
- `result.dataset` has `n_arms` measurements at `times`, and round-trips through
  `evaluate_design`.
- Validation errors each name the column or argument: a missing column, an
  unknown column, `low > high`, no free column, `n_restarts = 0`.
- `max_steps` too small for every draw raises `RuntimeError`, and the message
  names `max_steps`.
- `evaluate_design` numbers are unchanged after the refactor (existing tests
  cover this).

**Oracle, `tests/integration/doe/test_optimize_oracle.py`** (`@pytest.mark.expensive`).
Cross-evaluation against progress-doe:
- The fixture comes from a new progress-doe script,
  `research/scripts/export_catalax_design_oracle.py`, and is written to
  `tests/fixtures/doe_mat_design_oracle.json`. The setup is the MAT closed form:
  2 arms, 6 uniform times over 2 h, Uniform priors on all four parameters,
  `Proportional(0.2, 10)`, boxes A (0, 2000), B (0, 1000), E (0.01, 5), P fixed at
  0. It is run with progress-doe's own defaults. The fixture records the spec,
  priors, noise, times, progress-doe's best design and its closed-form hard
  maximin.
- The test builds MAT with `E` as an unobservable state, runs `optimize_design`
  with 16 draws × 8 restarts × 300 steps, then scores both designs with
  `evaluate_design` on one key at 256 draws.
- Pass condition: the Catalax design is at most 0.01 below progress-doe's. If it
  loses by more, the search failed; fix the search, never the tolerance. Report
  the gap both ways in the PR.
- The independent direction runs once, by hand: progress-doe's closed form scores
  the Catalax design, and the number goes in the PR description. Both
  criteria must agree on which design wins, within Monte Carlo error.

**Full suite:** `uv run pytest -m "not expensive" -q` must give 92 passed plus the
new tests. Run `ruff format`/`ruff check` **only on new files**.

## Verification

1. `uv run pytest tests/unit/doe tests/integration/doe -vv`, including the oracle
   test with `-m expensive`.
2. `uv run pytest -m "not expensive" -q`: 92 plus the new tests, 0 failed.
3. Run the docs example; it must finish in about a minute at 16 draws.
4. `git diff --stat`: `doe/evaluate.py`, `doe/optimize.py`, `doe/__init__.py`,
   `docs/doe/overview.mdx`, the new tests and the fixture. In progress-doe, only
   the export script and its `research/README.md` entry.
