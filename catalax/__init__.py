import os

import jax


def _usable_cores() -> int:
    """The cores this process may run on: on a SLURM node, its job's, not the node's.

    A Docker ``--cpus`` quota is not seen, hyperthreads count double on Linux
    x86, and Apple efficiency cores count as cores.
    """
    if hasattr(os, "process_cpu_count"):  # Python 3.13+, honours affinity
        return os.process_cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):  # Linux on 3.12 (SLURM, taskset)
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


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

import matplotlib as _mpl
from sympy import Symbol  # noqa: F401

from .dataset import Dataset, Measurement
from .model import InAxes, Model, SimulationConfig
from .objectives import l1_loss, mean_absolute_error
from .tools.enzymeml import dataset_and_model_from_enzymeml as from_enzymeml
from .tools.optimization import optimize

_mpl.rcParams["figure.dpi"] = 300
_mpl.rcParams["savefig.dpi"] = 300

__all__ = [
    "SimulationConfig",
    "Dataset",
    "Measurement",
    "InAxes",
    "Model",
    "optimize",
    "mean_absolute_error",
    "l1_loss",
    "from_enzymeml",
]

__version__ = "0.6.0"

PARAMETERS = InAxes.PARAMETERS
TIME = InAxes.TIME
INITS = InAxes.Y0


def set_host_count(n: int):
    """
    Sets the number of CPU devices JAX uses, e.g. for parallel MCMC chains.

    ``import catalax`` already sets one device per usable core, so call this only
    to override that: right after ``import catalax``, before anything is simulated
    or fitted. Importing submodules and building models do not start JAX, so their
    order does not matter.

    Args:
        n (int): The number of CPU devices.

    Raises:
        RuntimeError: If JAX has already run an operation, which fixes the count.
    """
    try:
        jax.config.update("jax_num_cpu_devices", n)
    except RuntimeError as error:
        raise RuntimeError(
            "JAX has already run an operation, so its device count is fixed. Call "
            "ctx.set_host_count(n) right after `import catalax`, before simulating "
            "or fitting; in a notebook, restart the kernel first."
        ) from error


def set_platform(platform: str = "cpu"):
    """
    Sets the platform for JAX.

    Args:
        platform (str): The platform to use. Must be one of 'cpu' or 'gpu'. Defaults to 'cpu'.

    Raises:
        AssertionError: If the platform is not 'cpu' or 'gpu'.
    """
    import numpyro

    assert platform in ["cpu", "gpu"], "platform must be one of 'cpu' or 'gpu'"

    numpyro.set_platform(platform)


def enable_x64(use_x64: bool = True):
    """
    Enables the use of 64-bit precision in JAX.

    Args:
        use_x64 (bool, optional): Whether to enable 64-bit precision. Defaults to True.
    """
    import numpyro

    numpyro.enable_x64(use_x64=use_x64)
