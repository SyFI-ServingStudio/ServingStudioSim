"""Torch CPU semantics for Qwen GDN's fused chunk output kernel."""

from __future__ import annotations

import torch

__all__ = ["gdn_chunk_output_reference"]

_CHUNK_SIZE = 64


def gdn_chunk_output_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v_new: torch.Tensor,
    h: torch.Tensor,
    g_cumsum: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the ragged Qwen chunk-output equation on CPU.

    Inputs are token-major BF16 ``q, k [T, Hg, K]``, BF16
    ``v_new [T, H, V]``, BF16 incoming-state snapshots
    ``h [C, H, V, K]`` in global chunk order, FP32 chunk-local inclusive
    ``g_cumsum [T, H]``, and int32 sequence boundaries. Chunk size 64 and
    scale ``K**-0.5`` are fixed production semantics.

    Production forms each gated causal QK score in FP32, rounds that score to
    BF16 before the value product, accumulates both QH and AV products in
    FP32, then rounds the combined scaled result once to BF16. Sequence and
    chunk boundaries reset the causal term. Outputs are fresh contiguous CPU
    storage and inputs are never mutated.
    """
    boundaries = _validate(q, k, v_new, h, g_cumsum, cu_seqlens)
    num_tokens, num_key_heads, key_head_dim = q.shape
    num_heads, value_head_dim = v_new.shape[1:]
    heads_per_key = num_heads // num_key_heads
    scale = key_head_dim**-0.5
    output = torch.empty(
        (num_tokens, num_heads, value_head_dim),
        dtype=torch.bfloat16,
        device="cpu",
    )

    global_chunk = 0
    for sequence_start, sequence_end in zip(boundaries, boundaries[1:]):
        for chunk_start in range(sequence_start, sequence_end, _CHUNK_SIZE):
            chunk_end = min(chunk_start + _CHUNK_SIZE, sequence_end)
            for head in range(num_heads):
                key_head = head // heads_per_key
                snapshot = h[global_chunk, head]
                for row in range(chunk_start, chunk_end):
                    query = q[row, key_head].float()
                    gate = g_cumsum[row, head]
                    state = torch.mv(snapshot.float(), query) * torch.exp(gate)
                    causal = torch.zeros(value_head_dim, dtype=torch.float32)
                    for source in range(chunk_start, row + 1):
                        score = torch.dot(query, k[source, key_head].float())
                        score = score * torch.exp(gate - g_cumsum[source, head])
                        rounded_score = score.to(torch.bfloat16)
                        causal.add_(rounded_score.float() * v_new[source, head].float())
                    output[row, head].copy_(((state + causal) * scale).to(torch.bfloat16))
            global_chunk += 1

    return output.contiguous()


def _validate(
    q: object,
    k: object,
    v_new: object,
    h: object,
    g_cumsum: object,
    cu_seqlens: object,
) -> list[int]:
    tensors = {
        "q": q,
        "k": k,
        "v_new": v_new,
        "h": h,
        "g_cumsum": g_cumsum,
        "cu_seqlens": cu_seqlens,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    assert all(isinstance(tensor, torch.Tensor) for tensor in tensors.values())

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

    expected_ranks = {"q": 3, "k": 3, "v_new": 3, "h": 4, "g_cumsum": 2, "cu_seqlens": 1}
    for name, tensor in tensors.items():
        if tensor.ndim != expected_ranks[name]:
            raise ValueError(f"{name} must be rank {expected_ranks[name]}, got rank {tensor.ndim}")
    for name in ("q", "k", "v_new", "h"):
        tensor = tensors[name]
        if tensor.dtype is not torch.bfloat16:
            raise TypeError(f"{name} dtype must be torch.bfloat16, got {tensor.dtype}")
    if tensors["g_cumsum"].dtype is not torch.float32:
        raise TypeError(f"g_cumsum dtype must be torch.float32, got {tensors['g_cumsum'].dtype}")
    if tensors["cu_seqlens"].dtype is not torch.int32:
        raise TypeError(f"cu_seqlens dtype must be torch.int32, got {tensors['cu_seqlens'].dtype}")

    q = tensors["q"]
    k = tensors["k"]
    v_new = tensors["v_new"]
    h = tensors["h"]
    g_cumsum = tensors["g_cumsum"]
    cu_seqlens = tensors["cu_seqlens"]
    num_tokens, num_key_heads, key_head_dim = q.shape
    if min(num_tokens, num_key_heads, key_head_dim) <= 0:
        raise ValueError("q token, head, and feature dimensions must be positive")
    if k.shape != q.shape:
        raise ValueError(f"k must have shape {tuple(q.shape)}, got {tuple(k.shape)}")
    if v_new.shape[0] != num_tokens:
        raise ValueError(f"v_new token dimension must equal {num_tokens}, got {v_new.shape[0]}")
    num_heads, value_head_dim = v_new.shape[1:]
    if min(num_heads, value_head_dim) <= 0:
        raise ValueError("v_new head and feature dimensions must be positive")
    if num_heads % num_key_heads:
        raise ValueError(
            f"output heads must be divisible by key heads, got H={num_heads}, Hg={num_key_heads}"
        )
    if g_cumsum.shape != (num_tokens, num_heads):
        raise ValueError(
            f"g_cumsum must have shape {(num_tokens, num_heads)}, got {tuple(g_cumsum.shape)}"
        )

    if cu_seqlens.numel() < 2:
        raise ValueError("cu_seqlens must contain at least one sequence")
    boundaries = [int(value) for value in cu_seqlens.tolist()]
    if boundaries[0] != 0:
        raise ValueError(f"cu_seqlens must start at zero, got {boundaries[0]}")
    if boundaries[-1] != num_tokens:
        raise ValueError(f"cu_seqlens must end at token count {num_tokens}, got {boundaries[-1]}")
    if any(value < 0 or value > num_tokens for value in boundaries):
        raise ValueError(f"cu_seqlens boundaries must be in [0, {num_tokens}]")
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError("cu_seqlens must be strictly increasing with no empty sequences")
    num_chunks = sum(
        (end - start + _CHUNK_SIZE - 1) // _CHUNK_SIZE
        for start, end in zip(boundaries, boundaries[1:])
    )
    expected_h = (num_chunks, num_heads, value_head_dim, key_head_dim)
    if h.shape != expected_h:
        raise ValueError(f"h must have shape {expected_h}, got {tuple(h.shape)}")

    for name in ("q", "k", "v_new", "h", "g_cumsum"):
        if not torch.isfinite(tensors[name]).all().item():
            raise ValueError(f"{name} must contain only finite values")
    return boundaries
