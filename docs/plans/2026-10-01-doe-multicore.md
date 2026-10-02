# Plan: use every core (catalax.doe, PR 2c)

After approval this plan is copied to `Catalax/docs/plans/2026-10-01-doe-multicore.md`.

## Context

`optimize_design` uses one CPU core today. The restarts are `jax.vmap`ped, and on CPU
that doesn't run them in parallel: XLA walks the wider arrays on one core. At 64 draws
the MAT search takes about 4 minutes, and longer on a bigger model.

Separately, `run_mcmc(chain_method="parallel")` only runs in parallel after
`ctx.set_host_count(n)` has been called before any JAX operation. Without that call,
numpyro warns and quietly runs the chains one after another. The Catalax docs
(`docs/hmc/surrogate-hmc.mdx`, `docs/doe/overview.mdx`) use `"parallel"` without it.

The user asked whether this can be set up at `import catalax`. It can, and it's
harmless (measured below).

## Measured before planning (2026-10-01; MM model, 64 draws, 8 restarts × 40 steps of `value_and_grad`, float64, M4 Pro 10P+4E)

| CPU devices | `vmap` (today) | one after another | threads 4 | threads 8 | threads 10 |
|---|---|---|---|---|---|
| 1 | 5.71 s | 5.85 s | 1.69 s | **0.93 s (6.1×)** | 0.92 s |
| 8 | 5.71 s | 5.97 s | 1.70 s | 0.94 s | 0.95 s |

- **Threads give the whole gain without any device setup.** JAX releases the GIL while
  compiled code runs, so a `ThreadPoolExecutor` over per-restart jitted calls runs them
  in parallel. `vmap` gains nothing over a plain loop.
- **Threaded results are bitwise identical to the `vmap` path** (max difference 0.0).
- **Forcing 8 CPU devices costs single-device code nothing** (5.71 s both ways).
- **`jax.config.update("jax_num_cpu_devices", n)` (JAX 0.7):**
  - it works after `import catalax`: importing catalax, `catalax.doe` or `catalax.mcmc`
    doesn't start the JAX backend;
  - called after any JAX operation, it raises a clear `RuntimeError`, where numpyro's
    `set_host_device_count` silently does nothing;
  - it overrides `XLA_FLAGS=--xla_force_host_platform_device_count`;
  - its default is -1, meaning unset.
- **`pmap` isn't needed.** Threads already reach about 6×, and progress-doe measured
  `pmap` breaking on closure-converted root finds, which a stiff solver such as
  Kvaerno5 brings in.

## Design

### 0. How many cores: the cores this process may use, not the machine's

- `os.cpu_count()` counts the whole machine. On a SLURM node or a shared Linux server
  that's 128 cores for an 8-core job, which oversubscribes everything.
- So both defaults use the cores the process is allowed to run on:

  ```python
  def _usable_cores() -> int:
      if hasattr(os, "process_cpu_count"):  # Python 3.13+, honours affinity
          return os.process_cpu_count() or 1
      if hasattr(os, "sched_getaffinity"):  # Linux on 3.12 (SLURM, taskset)
          return len(os.sched_getaffinity(0))
      return os.cpu_count() or 1
  ```

  It lives in `catalax/__init__.py`, and `optimize.py` imports it.
- **Limits** (in the docs, not handled in code):
  - a Docker `--cpus` quota isn't seen by any of these, so set `n_workers` and
    `ctx.set_host_count` yourself;
  - hyperthreads count double on Linux x86;
  - Apple efficiency cores count as cores.

### 1. `catalax/__init__.py`: one CPU device per core at import

Put this at the very top, before the submodule imports:

```python
import os

import jax

# One CPU device per core, so `run_mcmc(chain_method="parallel")` runs its chains
# in parallel. Single-device code is unaffected (measured: 5.71 s at 1 and at 8
# devices). The user's own setting wins: XLA_FLAGS or jax_num_cpu_devices set
# before import.
if jax.config.jax_num_cpu_devices == -1 and (
    "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", "")
):
    try:
        jax.config.update("jax_num_cpu_devices", _usable_cores())
    except RuntimeError:
        pass  # JAX already ran an operation; its device count is fixed.
```

- Rewrite `set_host_count(n)` as `jax.config.update("jax_num_cpu_devices", n)`:
  - it still overrides the import default when called before any JAX operation;
  - called too late, it now raises JAX's message instead of silently doing nothing;
  - the docstring says what the import already does.
- `set_platform` and `enable_x64` stay as they are.

### 2. `catalax/doe/optimize.py`: restarts in a thread pool

- New keyword `n_workers: int | None = None`. `None` means
  `min(n_restarts, _usable_cores())`. `_validate` checks `n_workers >= 1`.
- One jitted function per restart: `def restart(z0): z = ascend(z0); return z, *hard(z)`,
  then `run = jax.jit(restart)`. It returns `best_z`, the score and `clean`, so the
  ranking runs in the pool too.
- Replace the two `jax.jit(jax.vmap(...))` lines with:

  ```python
  with ThreadPoolExecutor(n_workers) as pool:
      results = list(pool.map(run, starts))
  ```

  `pool.map` keeps start order, so the result doesn't depend on `n_workers`.
- Warm the compile once on `starts[0]` before the pool. Otherwise all workers hit the
  first-call compile together; progress-doe pitfall §6 measured that serialising.
- The docstring for `n_workers`:
  - the measured table;
  - the progress-doe finding that workers beyond the performance-core count were
    slower (12 vs 8 on 10P+4E), so set `n_workers` to the P-core count when
    `n_restarts` exceeds it;
  - **no hard-coded cap**, because that's a machine-specific magic number.
- Update the `n_draws` cost table's "8 restarts" column from about 1 min to
  about 1/6 of that, measured on MAT.

### 3. Docs, `docs/doe/overview.mdx`

- **The parallel-chains note** (about line 224) becomes: catalax sets one CPU device
  per core at import; override with `ctx.set_host_count(n)` before any JAX operation.
- **One line on `n_workers`** in "Optimising a design".
- **A short "Cores" note** covering the §0 limits: SLURM and taskset are honoured; a
  Docker `--cpus` quota isn't, so set the count yourself; on an Apple chip, set
  `n_workers` to the performance-core count when `n_restarts` exceeds it.

### Out of scope

- `pmap` or `shard_map` over devices (measured unnecessary; it breaks on stiff
  solvers).
- Threading `evaluate_design`'s `lax.map`, which takes about 1 s.
- A progress bar (the existing `ponytail:` note stays).
- Changing `MCMCConfig.chain_method`'s default.

## Tests

**`tests/unit/doe/test_optimize_design.py`:**
- `n_workers=1` and `n_workers=4` give an equal `DesignResult`: the same dataset and
  `restart_scores`, compared bitwise.
- `n_workers=0` raises `ValueError`.
- The existing determinism and oracle tests run unchanged. Never re-tolerance them.

**`tests/unit/test_host_devices.py`** runs in subprocesses, because device count is
per process:
- `import catalax; len(jax.devices()) == catalax._usable_cores()`.
- `XLA_FLAGS=--xla_force_host_platform_device_count=3` before import gives 3.
- `ctx.set_host_count(2)` after import, before any op, gives 2.
- `set_host_count` after a JAX op raises `RuntimeError`.

**Full suite:** `uv run pytest -m "not expensive" -q` gives 118 plus the new tests. Run
ruff only on new files.

## Verification

1. **Bitwise:** before the change, save `restart_scores` and the winner's dataset from
   `optimize_design` on the MAT oracle fixture (16 draws, scratchpad). Afterwards they
   must match exactly.
2. **Timing on MAT, 64 draws, 8 restarts × 300 steps,** before and after. Put the
   number in the docstring and in the PR.
3. **`-m expensive tests/integration/doe`** passes. `test_close_the_loop` now gets
   parallel chains if it uses them; record its wall time.
4. **Suite wall time with 14 devices at import vs before:** no regression. That's the
   check that 14 forced devices are as harmless as 8.
