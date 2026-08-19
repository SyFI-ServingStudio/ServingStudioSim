"""Torch semantics for Qwen GDN chunk-local WY recomputation."""

from __future__ import annotations

import torch

__all__ = ["gdn_chunk_recompute_w_u_reference"]

_CHUNK_SIZE = 64


def gdn_chunk_recompute_w_u_reference(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recompute Qwen's chunk-local BF16 WY factors ``(w, u)``.

    Inputs are token-major BF16 ``k [T, Hg, K]``, BF16 ``v [T, H, V]``,
    FP32 ``beta`` and chunk-local inclusive ``g_cumsum [T, H]``, BF16 solved
    lower-triangular ``A [T, H, 64]``, and int32 ragged sequence boundaries.
    Output head ``h`` uses key head ``h // (H // Hg)``.

    Within each sequence-local chunk, production first rounds the factors
    ``beta * v`` and ``beta * exp(g) * k`` to BF16. It then multiplies the
    solved matrix by each rounded factor with FP32 accumulation and casts the
    resulting W and U once to BF16. Chunk and sequence boundaries reset the
    operation. The returned tensors are fresh contiguous CPU storage, and no
    input is mutated.
    """
    boundaries = _validate(k, v, beta, g_cumsum, A, cu_seqlens)
    num_tokens, num_key_heads, key_head_dim = k.shape
    num_heads, value_head_dim = v.shape[1:]
    heads_per_key = num_heads // num_key_heads
    w = torch.empty((num_tokens, num_heads, key_head_dim), dtype=torch.bfloat16)
    u = torch.empty((num_tokens, num_heads, value_head_dim), dtype=torch.bfloat16)

    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        for chunk_start in range(sequence_start, sequence_end, _CHUNK_SIZE):
            chunk_end = min(chunk_start + _CHUNK_SIZE, sequence_end)
            chunk_length = chunk_end - chunk_start

            solved = A[chunk_start:chunk_end, :, :chunk_length].permute(1, 0, 2).float()
            row_beta = beta[chunk_start:chunk_end].transpose(0, 1).unsqueeze(-1)

            # Match the Triton `.to(input.dtype)` cast before tl.dot. This is
            # intentionally not an all-FP32-until-the-final-store reference.
            v_factor = (v[chunk_start:chunk_end].permute(1, 0, 2).float() * row_beta).to(
                torch.bfloat16
            )
            key_by_output_head = (
                k[chunk_start:chunk_end]
                .repeat_interleave(heads_per_key, dim=1)
                .permute(1, 0, 2)
                .float()
            )
            gate_scale = torch.exp(g_cumsum[chunk_start:chunk_end].transpose(0, 1)).unsqueeze(-1)
            k_factor = (key_by_output_head * row_beta * gate_scale).to(torch.bfloat16)

            u[chunk_start:chunk_end].copy_(
                torch.matmul(solved, v_factor.float()).permute(1, 0, 2).to(torch.bfloat16)
            )
            w[chunk_start:chunk_end].copy_(
                torch.matmul(solved, k_factor.float()).permute(1, 0, 2).to(torch.bfloat16)
            )

    return w, u


def _validate(
    k: object,
    v: object,
    beta: object,
    g_cumsum: object,
    A: object,
    cu_seqlens: object,
) -> list[int]:
    tensors = {
        "k": k,
        "v": v,
        "beta": beta,
        "g_cumsum": g_cumsum,
        "A": A,
        "cu_seqlens": cu_seqlens,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    assert all(isinstance(tensor, torch.Tensor) for tensor in tensors.values())
    k = tensors["k"]
    v = tensors["v"]
    beta = tensors["beta"]
    g_cumsum = tensors["g_cumsum"]
    A = tensors["A"]
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

    expected_ranks = {
        "k": 3,
        "v": 3,
        "beta": 2,
        "g_cumsum": 2,
        "A": 3,
        "cu_seqlens": 1,
    }
    for name, tensor in tensors.items():
        expected_rank = expected_ranks[name]
        if tensor.ndim != expected_rank:
            raise ValueError(f"{name} must be rank {expected_rank}, got rank {tensor.ndim}")

    for name, tensor in (("k", k), ("v", v), ("A", A)):
        if tensor.dtype is not torch.bfloat16:
            raise TypeError(f"{name} dtype must be torch.bfloat16, got {tensor.dtype}")
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
    if v.shape[0] != num_tokens:
        raise ValueError(f"v token dimension must equal {num_tokens}, got {v.shape[0]}")
    num_heads, value_head_dim = v.shape[1:]
    if num_heads <= 0:
        raise ValueError(f"v head dimension must be positive, got {num_heads}")
    if value_head_dim <= 0:
        raise ValueError(f"v feature dimension must be positive, got {value_head_dim}")
    if num_heads % num_key_heads != 0:
        raise ValueError(
            "output head count must be divisible by key head count, got "
            f"H={num_heads}, Hg={num_key_heads}"
        )
    expected_gate_shape = (num_tokens, num_heads)
    for name, tensor in (("beta", beta), ("g_cumsum", g_cumsum)):
        if tensor.shape != expected_gate_shape:
            raise ValueError(
                f"{name} must have shape {expected_gate_shape}, got {tuple(tensor.shape)}"
            )
    expected_A_shape = (num_tokens, num_heads, _CHUNK_SIZE)
    if A.shape != expected_A_shape:
        raise ValueError(f"A must have shape {expected_A_shape}, got {tuple(A.shape)}")

    for name, tensor in (("k", k), ("v", v), ("beta", beta), ("g_cumsum", g_cumsum), ("A", A)):
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} must contain only finite values")

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

    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        for chunk_start in range(sequence_start, sequence_end, _CHUNK_SIZE):
            chunk_end = min(chunk_start + _CHUNK_SIZE, sequence_end)
            chunk_length = chunk_end - chunk_start
            local = A[chunk_start:chunk_end]
            for row in range(chunk_length):
                if not torch.all(local[row, :, row] == 1).item():
                    raise ValueError("A must have an exact unit diagonal in every local chunk")
                if torch.count_nonzero(local[row, :, row + 1 :]).item() != 0:
                    raise ValueError(
                        "A must be exactly lower triangular; upper and unused columns must be zero"
                    )
    return boundaries
