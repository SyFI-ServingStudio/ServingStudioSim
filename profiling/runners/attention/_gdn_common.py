"""Shared, runtime-light mechanics for GDN profiling runners.

Keep only contracts that are identical across GDN kernels here. Tensor shapes,
upstream call signatures, correctness oracles, and timing closures remain in
the concrete runner so the measured operation stays visible at its call site.
This module deliberately does not import torch or any environment-specific
vLLM module at import time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from profiling.runners.exceptions import ProfilerNotImplemented


def exact_int(name: str, value: object) -> int:
    """Return an exact integer while rejecting bool and numeric coercions."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an exact integer, got {value!r}")
    return value


def require_cuda(torch: Any, *, backend: str) -> None:
    """Require a CUDA device; these runners launch portable Triton/CUDA kernels."""
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"CUDA is required for {backend}")


def require_compute_capability(
    torch: Any,
    *,
    backend: str,
    capability: tuple[int, int],
    reason: str,
) -> None:
    """Require CUDA and one compute capability for an arch-specific kernel build."""
    require_cuda(torch, backend=backend)
    device = torch.cuda.current_device()
    actual = tuple(torch.cuda.get_device_capability(device))
    if actual != capability:
        gpu_name = str(torch.cuda.get_device_name(device))
        raise ProfilerNotImplemented(
            f"{backend} requires SM{capability[0]}{capability[1]} ({reason}), "
            f"got {gpu_name} with SM{actual[0]}{actual[1]}"
        )


def load_required_callable(
    import_module: Callable[[str], Any],
    *,
    backend: str,
    module_name: str,
    callable_name: str,
    environment_label: str = "the repository vllm_env",
    missing_message: str | None = None,
) -> Any:
    """Lazily import one upstream callable inside the selected worker env."""
    try:
        module = import_module(module_name)
    except Exception as exc:
        raise ProfilerNotImplemented(f"{backend} requires {environment_label}") from exc
    required_callable = getattr(module, callable_name, None)
    if not callable(required_callable):
        message = missing_message or f"{backend} requires {module_name}.{callable_name}"
        raise ProfilerNotImplemented(message)
    return required_callable
