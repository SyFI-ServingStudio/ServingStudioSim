"""Torch semantics for the DSA prefill MQA-logits operation."""

from __future__ import annotations

import torch

_SUPPORTED_QK_DTYPES = (
    torch.float8_e4m3fn,
    torch.bfloat16,
    torch.float16,
    torch.float32,
)
_SPAN_DTYPES = (torch.int32, torch.int64)


def dsa_mqa_logits_prefill_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    k_scale: torch.Tensor,
    weights: torch.Tensor,
    k_start: torch.Tensor,
    k_end: torch.Tensor,
    *,
    clean_logits: bool = False,
) -> torch.Tensor:
    """Return FP32 DSA logits for the valid key span of every query.

    Dot products, ReLU, head weighting/reduction, and K scaling are evaluated
    in FP32. Invalid output positions are ``-inf`` when ``clean_logits`` is
    true; otherwise they are NaN sentinels because production does not define
    their contents.

    This models the math and valid-output contract of vLLM v0.23.0 commit
    0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665 and DeepGEMM commit
    891d57b4db1071624b5c8fa0d1e51cb317fa709f,
    ``fp8_fp4_mqa_logits``. Later profiling backends construct the exact
    DeepGEMM layouts and time the production kernel.
    """
    _validate(q, k, k_scale, weights, k_start, k_end, clean_logits)

    q_fp32 = q.float()
    k_fp32 = k.float()
    dot_products = torch.einsum("mhd,nd->mhn", q_fp32, k_fp32)
    reduced = (torch.relu(dot_products) * weights.unsqueeze(-1)).sum(dim=1) * k_scale.unsqueeze(0)

    invalid_value = float("-inf") if clean_logits else float("nan")
    output = torch.full(
        (q.shape[0], k.shape[0]),
        invalid_value,
        dtype=torch.float32,
        device=q.device,
    )
    positions = torch.arange(k.shape[0], device=q.device)
    valid = (positions.unsqueeze(0) >= k_start.unsqueeze(1)) & (
        positions.unsqueeze(0) < k_end.unsqueeze(1)
    )
    output[valid] = reduced[valid]
    return output


def _validate(
    q: object,
    k: object,
    k_scale: object,
    weights: object,
    k_start: object,
    k_end: object,
    clean_logits: object,
) -> None:
    tensors = (
        ("q", q),
        ("k", k),
        ("k_scale", k_scale),
        ("weights", weights),
        ("k_start", k_start),
        ("k_end", k_end),
    )
    for name, tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
    if not isinstance(clean_logits, bool):
        raise TypeError("clean_logits must be a bool")

    assert isinstance(q, torch.Tensor)
    assert isinstance(k, torch.Tensor)
    assert isinstance(k_scale, torch.Tensor)
    assert isinstance(weights, torch.Tensor)
    assert isinstance(k_start, torch.Tensor)
    assert isinstance(k_end, torch.Tensor)

    expected_ranks = {
        "q": 3,
        "k": 2,
        "k_scale": 1,
        "weights": 2,
        "k_start": 1,
        "k_end": 1,
    }
    for name, tensor in tensors:
        expected_rank = expected_ranks[name]
        if tensor.ndim != expected_rank:
            raise ValueError(f"{name} must be rank {expected_rank}, got rank {tensor.ndim}")
        if any(dimension <= 0 for dimension in tensor.shape):
            raise ValueError(f"{name} dimensions must be positive, got {tuple(tensor.shape)}")

    m, num_heads, head_dim = q.shape
    num_keys, key_dim = k.shape
    if key_dim != head_dim:
        raise ValueError(f"q and k head dimensions must match, got {head_dim} and {key_dim}")
    if k_scale.shape != (num_keys,):
        raise ValueError(f"k_scale must have shape ({num_keys},), got {tuple(k_scale.shape)}")
    if weights.shape != (m, num_heads):
        raise ValueError(f"weights must have shape ({m}, {num_heads}), got {tuple(weights.shape)}")
    for name, spans in (("k_start", k_start), ("k_end", k_end)):
        if spans.shape != (m,):
            raise ValueError(f"{name} must have shape ({m},), got {tuple(spans.shape)}")

    if q.dtype not in _SUPPORTED_QK_DTYPES:
        raise TypeError(
            "q dtype must be torch.float8_e4m3fn, torch.bfloat16, "
            f"torch.float16, or torch.float32, got {q.dtype}"
        )
    if k.dtype != q.dtype:
        raise TypeError(f"k dtype must match q dtype {q.dtype}, got {k.dtype}")
    if k_scale.dtype is not torch.float32:
        raise TypeError(f"k_scale dtype must be torch.float32, got {k_scale.dtype}")
    if weights.dtype is not torch.float32:
        raise TypeError(f"weights dtype must be torch.float32, got {weights.dtype}")
    for name, spans in (("k_start", k_start), ("k_end", k_end)):
        if spans.dtype not in _SPAN_DTYPES:
            raise TypeError(f"{name} dtype must be torch.int32 or torch.int64")
        if spans.layout is not torch.strided or not spans.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if q.device.type == "meta":
        raise ValueError("meta tensors are not supported")
    for name, tensor in tensors[1:]:
        if tensor.device != q.device:
            raise ValueError(
                f"{name} must be on the same device as q, got {tensor.device} and {q.device}"
            )

    for name, tensor in tensors[:4]:
        if tensor.layout is not torch.strided:
            raise ValueError(f"{name} must have torch.strided layout")
        if not bool(torch.isfinite(tensor.float()).all().item()):
            raise ValueError(f"{name} must contain only finite values")
    if not bool(torch.all(k_scale > 0).item()):
        raise ValueError("k_scale values must be strictly positive")

    if bool(torch.any(k_start < 0).item()):
        raise ValueError("k_start values must be nonnegative")
    if bool(torch.any(k_end > num_keys).item()):
        raise ValueError(f"k_end values must not exceed the key count {num_keys}")
    if bool(torch.any(k_start > k_end).item()):
        raise ValueError("each k_start value must be less than or equal to k_end")
