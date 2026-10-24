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


def accepted_counts(batch_size: int, per_request: int) -> tuple[int, ...]:
    """Deterministic accepted counts covering 1..q (the verify kernels read from ``a − 1``)."""
    return tuple(request % per_request + 1 for request in range(batch_size))


def checkpoint_indices(batch_size: int, per_request: int) -> tuple[tuple[int, ...], ...]:
    """vLLM's ``[B, q]`` per-token checkpoint slots; slot zero stays reserved."""
    return tuple(
        tuple(request * per_request + token + 1 for token in range(per_request))
        for request in range(batch_size)
    )


def canonical_sequence_boundaries(num_tokens: int, max_sequence_length: int) -> list[int]:
    """`num_tokens // max` full-length sequences, then the remainder.

    This is the realization chunked prefill produces, not a modelling
    convenience: vLLM fills its token budget with whole chunks and leaves one
    short tail, so a capture iteration of 8185 tokens arrives as 8163 + 22. A
    *balanced* partition of the same two numbers would halve the sequential
    inter-chunk critical path and under-measure the kernel by ~2x, which is why
    the second axis of ``gdn_chunk_delta_rule`` is a length rather than a
    sequence count.
    """
    full_count, remainder = divmod(num_tokens, max_sequence_length)
    lengths = [max_sequence_length] * full_count
    if remainder:
        lengths.append(remainder)
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    return boundaries


# ``gdn_chunk_delta_rule``'s documented FLOP estimate uses a 64-token chunk
# width whatever the backend launches; it only sizes the estimate.
_SEMANTIC_CHUNK = 64


def chunk_delta_rule_semantic_flops(args: Any) -> float:
    """Logical MACs×2 of the chunked delta rule, at the canonical chunk width.

    Per token the recurrence touches the whole `key_head_dim × value_head_dim`
    state twice (scan-in and read-out); the intra-chunk terms (K·Kᵀ, the
    triangular solve, and the attention-like output) each scale with the chunk
    width rather than the sequence, which is exactly why chunking exists. The
    kind's DOC formula; every backend reports it so their TFLOPS compare.
    """
    tokens = float(args.num_tokens)
    state = 2.0 * args.key_head_dim * args.value_head_dim * args.num_heads
    intra = (
        2.0
        * _SEMANTIC_CHUNK
        * (args.num_key_heads * args.key_head_dim + args.num_heads * args.value_head_dim)
    )
    return tokens * 2.0 * (state + intra)


def chunk_delta_rule_logical_bytes(args: Any) -> float:
    """Minimum traffic the operation implies: operands in, output and state out."""
    qk = 2.0 * args.num_tokens * args.num_key_heads * args.key_head_dim * 2
    value_and_output = 2.0 * args.num_tokens * args.num_heads * args.value_head_dim * 2
    gates = 2.0 * args.num_tokens * args.num_heads * 4
    sequence_count = -(-args.num_tokens // args.max_sequence_length)
    states = 2.0 * sequence_count * args.num_heads * args.key_head_dim * args.value_head_dim * 4
    return qk + value_and_output + gates + states
