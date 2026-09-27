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


def require_exact_gpu(
    torch: Any,
    *,
    backend: str,
    required_gpu: str,
) -> None:
    """Enforce the hardware boundary under which a runner was validated."""
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented(f"CUDA is required for {backend}")
    gpu_name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if gpu_name != required_gpu:
        raise ProfilerNotImplemented(
            f"{backend} is verified only on {required_gpu}, got {gpu_name}"
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
