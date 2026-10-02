"""``import catalax`` sets one CPU device per usable core; the user's setting wins.

The device count is fixed per process once JAX starts, so every case runs in a
fresh interpreter.
"""

from __future__ import annotations

import os
import subprocess
import sys


def _run(code: str, xla_flags: str | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "XLA_FLAGS"}
    if xla_flags is not None:
        env["XLA_FLAGS"] = xla_flags
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _devices(code: str, xla_flags: str | None = None) -> str:
    result = _run(code + "\nprint(len(jax.devices()))", xla_flags)
    assert result.returncode == 0, result.stderr
    return result.stdout.split()[-1]


def test_import_sets_one_device_per_usable_core():
    code = "import jax, catalax\nprint(catalax._usable_cores(), len(jax.devices()))"
    result = _run(code)
    assert result.returncode == 0, result.stderr
    cores, devices = result.stdout.split()[-2:]
    assert devices == cores


def test_xla_flags_set_before_import_win():
    flags = "--xla_force_host_platform_device_count=3"
    assert _devices("import jax, catalax", xla_flags=flags) == "3"


def test_set_host_count_after_import_overrides():
    # Importing the submodules must not start JAX either.
    code = (
        "import jax, catalax as ctx, catalax.doe, catalax.mcmc\nctx.set_host_count(2)"
    )
    assert _devices(code) == "2"


def test_set_host_count_after_a_jax_operation_raises():
    code = (
        "import jax.numpy as jnp, catalax as ctx\njnp.zeros(1)\nctx.set_host_count(2)"
    )
    result = _run(code)
    assert result.returncode != 0
    assert "RuntimeError" in result.stderr
    assert "right after `import catalax`" in result.stderr
