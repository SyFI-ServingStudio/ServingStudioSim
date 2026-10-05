"""The device and runtime checks every runner shares.

Every backend times a CUDA device, and a backend's capability requirement is
declared once, as its ``BackendSupport`` (``min_compute_capability`` /
``sm_targets``). The worker checks both against the real device before it
loads the runner (``unsupported_device``). A runner therefore repeats neither;
it keeps only checks that depend on the spec's shape, such as which head counts
a kernel build has on this device.

Loaded only inside the worker subprocess; torch is imported lazily.
"""

from __future__ import annotations

from typing import Any

from profiling.db.registry import BackendSupport
from profiling.runners.exceptions import ProfilerNotImplemented


def require_cuda_toolkit(torch: Any, minimum: tuple[int, int], label: str) -> None:
    """Require torch's CUDA toolkit to be at least ``minimum``: the floor of a
    binding the runner JIT-builds against it."""
    version = _major_minor(getattr(torch.version, "cuda", None))
    if version is None or version < minimum:
        raise ProfilerNotImplemented(
            f"{label} requires CUDA >= {minimum[0]}.{minimum[1]}, "
            f"got {getattr(torch.version, 'cuda', None)}"
        )


def require_cutedsl(label: str) -> None:
    """Require vLLM's CuTe DSL path (``has_cutedsl``), which a CuTe DSL backend
    launches through."""
    try:
        from vllm.utils.import_utils import has_cutedsl
    except ImportError as exc:
        raise ProfilerNotImplemented(f"{label} requires pinned vLLM") from exc
    if not has_cutedsl():
        raise ProfilerNotImplemented(f"{label} requires vLLM's CuTe DSL path")


def _major_minor(version: object) -> tuple[int, int] | None:
    major, _, rest = str(version).partition(".")
    minor = rest.partition(".")[0]
    if version is None or not (major.isdigit() and minor.isdigit()):
        return None
    return int(major), int(minor)


def unsupported_device(supports: BackendSupport, label: str) -> str | None:
    """Why this process's CUDA device 0 cannot run a backend declaring
    ``supports``, or ``None`` when it can. No CUDA device runs any backend.

    Reads the real device rather than resolving the requested GPU name through
    the catalog, so a GPU the catalog does not know is still checked.
    """
    try:
        import torch
    except ImportError:
        return f"{label} needs torch to read the device"
    if not torch.cuda.is_available():
        return f"CUDA is required for {label}"
    rule = supports.device_rule()
    if rule is None:
        return None
    major, minor = (int(part) for part in torch.cuda.get_device_capability(0))
    if supports.allows_compute_capability((major, minor)):
        return None
    name = torch.cuda.get_device_name(0)
    return f"{label} needs {rule}, got {name} with SM{major}{minor}"
