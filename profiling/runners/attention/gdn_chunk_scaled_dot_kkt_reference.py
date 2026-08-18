"""Torch semantics for Qwen GDN chunk-local scaled-dot KKT construction."""

from __future__ import annotations

import torch

__all__ = ["gdn_chunk_scaled_dot_kkt_reference"]

_CHUNK_SIZE = 64


def gdn_chunk_scaled_dot_kkt_reference(
    k: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Build Qwen's positive strict-lower chunk-local KKT matrix.

    ``k`` is BF16 ``[T, Hg, K]`` while ``beta`` and the already chunk-local
    cumulative ``g_cumsum`` are FP32 ``[T, H]``. For each ragged sequence and
    each consecutive block of at most 64 tokens, output head ``h`` uses key
    head ``h // (H // Hg)``. Row ``i`` owns beta, and valid strict-lower entry
    ``(i, j)`` is ``beta_i * dot(k_i, k_j) * exp(g_i - g_j)``.

    K is promoted from its BF16-rounded representation to FP32 before the
    beta scaling and dot. Dot accumulation, decay, and output remain FP32.
    This is the positive matrix consumed by ``solve_tril``; Transformers'
    combined fallback negates it while performing that later solve.

    Chunk size 64 and FP32 gate/output dtypes are frozen Qwen semantics.
    Strided CPU tensors are accepted. The returned ``[T, H, 64]`` tensor is
    newly allocated and contiguous, and no input is mutated. This CPU
    reference expresses mathematical FP32 semantics; it intentionally does
    not emulate the production H200 kernel's TF32 dot-input rounding.
    """
    boundaries = _validate(k, beta, g_cumsum, cu_seqlens)
    num_tokens, num_key_heads, _ = k.shape
    num_heads = beta.shape[1]
    heads_per_key = num_heads // num_key_heads
    output = torch.zeros(
        (num_tokens, num_heads, _CHUNK_SIZE),
        dtype=torch.float32,
        device="cpu",
    )

    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        for chunk_start in range(sequence_start, sequence_end, _CHUNK_SIZE):
            chunk_end = min(chunk_start + _CHUNK_SIZE, sequence_end)
            chunk_length = chunk_end - chunk_start

            # Keep dot reductions and reset boundaries local to one production
            # chunk. A global T×T product followed by masking would compute a
            # different operation and could change FP32 accumulation behavior.
            key_chunk = k[chunk_start:chunk_end].float()
            key_by_output_head = key_chunk.repeat_interleave(
                heads_per_key,
                dim=1,
            ).permute(1, 0, 2)
            row_beta = beta[chunk_start:chunk_end].transpose(0, 1)
            scaled_key = key_by_output_head * row_beta.unsqueeze(-1)
            dots = torch.matmul(
                scaled_key,
                key_by_output_head.transpose(-1, -2),
            )

            gate = g_cumsum[chunk_start:chunk_end].transpose(0, 1)
            decay = torch.exp(gate.unsqueeze(-1) - gate.unsqueeze(-2))
            strict_lower = torch.tril(
                torch.ones(
                    (chunk_length, chunk_length),
                    dtype=torch.bool,
                    device="cpu",
                ),
                diagonal=-1,
            )
            block = torch.where(strict_lower, dots * decay, 0.0)
            output[chunk_start:chunk_end, :, :chunk_length].copy_(block.permute(1, 0, 2))

    return output


def _validate(
    k: object,
    beta: object,
    g_cumsum: object,
    cu_seqlens: object,
) -> list[int]:
    tensors = {
        "k": k,
        "beta": beta,
        "g_cumsum": g_cumsum,
        "cu_seqlens": cu_seqlens,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    assert all(isinstance(tensor, torch.Tensor) for tensor in tensors.values())
    k = tensors["k"]
    beta = tensors["beta"]
    g_cumsum = tensors["g_cumsum"]
    cu_seqlens = tensors["cu_seqlens"]

    for name, tensor in tensors.items():
        if tensor.layout is not torch.strided:
            raise ValueError(f"{name} must have torch.strided layout")

    devices = {tensor.device for tensor in tensors.values()}
    if len(devices) != 1:
        rendered = ", ".join(f"{name}={tensor.device}" for name, tensor in tensors.items())
        raise ValueError(f"all inputs must be on the same device, got {rendered}")
    device = next(iter(devices))
    if device.type == "meta":
        raise ValueError("meta tensors are not supported")
    if device.type != "cpu":
        raise ValueError(f"semantic reference requires CPU tensors, got {device.type}")

    expected_ranks = {"k": 3, "beta": 2, "g_cumsum": 2, "cu_seqlens": 1}
    for name, tensor in tensors.items():
        expected_rank = expected_ranks[name]
        if tensor.ndim != expected_rank:
            raise ValueError(f"{name} must be rank {expected_rank}, got rank {tensor.ndim}")

    if k.dtype is not torch.bfloat16:
        raise TypeError(f"k dtype must be torch.bfloat16, got {k.dtype}")
    for name, tensor in (("beta", beta), ("g_cumsum", g_cumsum)):
        if tensor.dtype is not torch.float32:
            raise TypeError(f"{name} dtype must be torch.float32, got {tensor.dtype}")
    if cu_seqlens.dtype is not torch.int32:
        raise TypeError(f"cu_seqlens dtype must be torch.int32, got {cu_seqlens.dtype}")

    num_tokens, num_key_heads, key_head_dim = k.shape
    if num_tokens <= 0:
        raise ValueError(f"k token dimension must be positive, got {num_tokens}")
    if num_key_heads <= 0:
        raise ValueError(f"k head dimension must be positive, got {num_key_heads}")
    if key_head_dim <= 0:
        raise ValueError(f"k feature dimension must be positive, got {key_head_dim}")

    expected_gate_tokens = num_tokens
    if beta.shape[0] != expected_gate_tokens:
        raise ValueError(
            f"beta token dimension must equal k token count {num_tokens}, got {beta.shape[0]}"
        )
    num_heads = beta.shape[1]
    if num_heads <= 0:
        raise ValueError(f"beta head dimension must be positive, got {num_heads}")
    if g_cumsum.shape != beta.shape:
        raise ValueError(
            f"g_cumsum must have shape {tuple(beta.shape)}, got {tuple(g_cumsum.shape)}"
        )
    if num_heads % num_key_heads != 0:
        raise ValueError(
            "output head count must be divisible by key head count, got "
            f"H={num_heads}, Hg={num_key_heads}"
        )

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
