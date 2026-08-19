"""Torch semantics for Qwen GDN ragged chunk-local cumulative decay."""

from __future__ import annotations

import torch

__all__ = ["gdn_chunk_local_cumsum_reference"]

_CHUNK_SIZE = 64


def gdn_chunk_local_cumsum_reference(
    g: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Compute Qwen's FP32 chunk-local inclusive cumulative decay.

    ``g`` is token-major ``[T, H]`` FP32 decay and ``cu_seqlens`` is an int32
    prefix array ``[N + 1]`` describing strictly positive ragged sequences.
    Within each sequence, an independent inclusive cumulative sum is computed
    for every consecutive block of at most 64 tokens. The sum therefore resets
    both at sequence boundaries and at 64-token chunk boundaries.

    Chunk size 64, forward/token-first scalar mode, and FP32 output are frozen
    Qwen semantics rather than options. Strided CPU tensors are accepted. The
    returned ``[T, H]`` tensor is newly allocated and contiguous; neither input
    is mutated.
    """
    boundaries = _validate(g, cu_seqlens)
    output = torch.empty(g.shape, dtype=torch.float32, device=g.device)

    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        for chunk_start in range(sequence_start, sequence_end, _CHUNK_SIZE):
            chunk_end = min(chunk_start + _CHUNK_SIZE, sequence_end)
            # Keep every accumulation local to its production chunk. A global
            # cumsum followed by subtraction can change FP32 rounding order.
            output[chunk_start:chunk_end].copy_(
                torch.cumsum(
                    g[chunk_start:chunk_end],
                    dim=0,
                    dtype=torch.float32,
                )
            )

    return output


def _validate(g: object, cu_seqlens: object) -> list[int]:
    tensors = {"g": g, "cu_seqlens": cu_seqlens}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    assert isinstance(g, torch.Tensor)
    assert isinstance(cu_seqlens, torch.Tensor)

    for name, tensor in tensors.items():
        if tensor.layout is not torch.strided:
            raise ValueError(f"{name} must have torch.strided layout")

    if g.device != cu_seqlens.device:
        raise ValueError(
            f"g and cu_seqlens must be on the same device, got {g.device} and {cu_seqlens.device}"
        )
    if g.device.type == "meta":
        raise ValueError("meta tensors are not supported")
    if g.device.type != "cpu":
        raise ValueError(f"semantic reference requires CPU tensors, got {g.device.type}")

    if g.ndim != 2:
        raise ValueError(f"g must be rank 2, got rank {g.ndim}")
    if cu_seqlens.ndim != 1:
        raise ValueError(f"cu_seqlens must be rank 1, got rank {cu_seqlens.ndim}")
    if g.dtype is not torch.float32:
        raise TypeError(f"g dtype must be torch.float32, got {g.dtype}")
    if cu_seqlens.dtype is not torch.int32:
        raise TypeError(f"cu_seqlens dtype must be torch.int32, got {cu_seqlens.dtype}")

    num_tokens, num_heads = g.shape
    if num_tokens <= 0:
        raise ValueError(f"g token dimension must be positive, got {num_tokens}")
    if num_heads <= 0:
        raise ValueError(f"g head dimension must be positive, got {num_heads}")
    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain at least one sequence")

    boundaries = [int(boundary) for boundary in cu_seqlens.tolist()]
    if any(boundary < 0 or boundary > num_tokens for boundary in boundaries):
        raise ValueError(f"cu_seqlens boundaries must be in [0, {num_tokens}]")
    if boundaries[0] != 0:
        raise ValueError(f"cu_seqlens must start at zero, got {boundaries[0]}")
    if boundaries[-1] != num_tokens:
        raise ValueError(f"cu_seqlens must end at token count {num_tokens}, got {boundaries[-1]}")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("cu_seqlens must be strictly increasing with no empty sequences")
    return boundaries
