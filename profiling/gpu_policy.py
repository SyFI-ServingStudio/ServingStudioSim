"""The no-GPU guarantee: with ``SERVINGSTUDIO_NO_GPU`` set, nothing here uses a GPU.

A simulation or prediction over a warm ``profile.db`` needs no GPU. The only
code that reaches one is profiling a missing row (JIT, a cache build,
``kernel-profile run`` / ``measure``), selecting cards (``nvidia-smi``), and an
alignment capture. Each of those calls `require_gpu` first, so under the
variable a missing row is an error that names what needed the GPU instead of a
silent measurement.

``python -m launcher --no-gpu`` sets the variable for its own process and every
child. It also empties ``CUDA_VISIBLE_DEVICES``, so a CUDA call these checks
did not anticipate sees no device.
"""

from __future__ import annotations

import os
from collections.abc import MutableMapping

NO_GPU_ENV = "SERVINGSTUDIO_NO_GPU"
_OFF = frozenset({"", "0", "false", "no", "off"})


class GpuDisabledError(RuntimeError):
    """A step needed a GPU while ``SERVINGSTUDIO_NO_GPU`` was set."""


def gpu_disabled() -> bool:
    """Whether ``SERVINGSTUDIO_NO_GPU`` is set to anything but an off value."""
    return os.environ.get(NO_GPU_ENV, "").strip().lower() not in _OFF


def require_gpu(action: str) -> None:
    """Raise `GpuDisabledError` if GPUs are disabled; ``action`` names the step."""
    if gpu_disabled():
        raise GpuDisabledError(
            f"{action} needs a GPU, but {NO_GPU_ENV} is set. Fill the missing "
            f"profile.db rows on a GPU host, or unset {NO_GPU_ENV} (drop --no-gpu)."
        )


def disable_gpus(environ: MutableMapping[str, str] | None = None) -> None:
    """Set the variable and hide every CUDA device, for this process and its children."""
    environ = os.environ if environ is None else environ
    environ[NO_GPU_ENV] = "1"
    environ["CUDA_VISIBLE_DEVICES"] = ""
