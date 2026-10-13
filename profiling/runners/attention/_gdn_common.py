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


def conv_state_dim_first() -> bool:
    """Whether the installed vLLM stores conv state dim-first (``DS``)."""
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

    return bool(is_conv_state_dim_first())


def conv_state_pool(
    torch: Any,
    shape: tuple[int, int, int],
    *,
    dtype: Any,
    device: Any,
    dim_first: bool = False,
) -> Any:
    """Allocate a ``[slots, channels, width - 1]`` conv-state pool in vLLM's layout.

    vLLM stores each slot either dim-first (``DS``: ``[channels, width - 1]``) or
    state-first (``SD``: ``[width - 1, channels]``, its default) and hands the
    conv kernels a ``[slots, channels, width - 1]`` view of it, transposed for
    ``SD``. The kernels' state reads and writes are strided accordingly, which
    moves their time by 12-25% at decode and long-prefill shapes, so runners
    pass the layout the installed vLLM selects (``conv_state_dim_first``).
    """
    slots, channels, state_len = shape
    if dim_first:
        return torch.empty((slots, channels, state_len), dtype=dtype, device=device)
    return torch.empty((slots, state_len, channels), dtype=dtype, device=device).transpose(-1, -2)


def tokens_per_request(batch_size: int, num_tokens: int | None) -> int:
    """Uniform tokens per request of a decode spec; 1 when ``num_tokens`` is absent.

    ``num_tokens`` is the optional ``same_as="batch_size"`` arg of the decode
    kinds: a speculative verify step carries the accepted token plus the drafts
    of every request, an equal count each.
    """
    if num_tokens is None:
        return 1
    num_tokens = int(num_tokens)
    if num_tokens <= 0 or batch_size <= 0 or num_tokens % batch_size != 0:
        raise ValueError(
            f"num_tokens must be a positive multiple of batch_size, got {num_tokens} "
            f"for batch_size={batch_size}"
        )
    return num_tokens // batch_size


def reject_verify_tokens(backend: str, batch_size: int, num_tokens: int | None) -> None:
    """Refuse a multi-token (verify) spec on a backend that decodes one token per request."""
    if tokens_per_request(batch_size, num_tokens) != 1:
        raise ValueError(
            f"{backend} decodes one token per request; num_tokens={num_tokens} with "
            f"batch_size={batch_size} is a verify step"
        )
