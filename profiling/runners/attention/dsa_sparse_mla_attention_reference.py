"""Torch semantics for GLM-5.2 BF16 selected sparse MLA attention."""

from __future__ import annotations

import math

import torch

__all__ = ["dsa_sparse_mla_attention_reference"]

_LATENT_DIM = 512
_ROPE_DIM = 64
_SCORE_DIM = _LATENT_DIM + _ROPE_DIM
_VALUE_DIM = _LATENT_DIM
_QUERY_CHUNK_SIZE = 8


def dsa_sparse_mla_attention_reference(
    q: torch.Tensor,
    cache: torch.Tensor,
    selected_indices: torch.Tensor,
    *,
    softmax_scale: float,
) -> torch.Tensor:
    """Return BF16 selected MLA attention in a fresh ``[Q, H, 512]`` tensor.

    The score and value paths use FP32: QK consumes all 576 dimensions (the
    512-dimensional latent part followed by 64 RoPE dimensions), while PV uses
    the first 512 cache dimensions as values. Every negative index and every
    index outside the cache is masked. Duplicate valid indices deliberately
    retain repeated softmax mass, and rows with no valid index return zeros.

    This is the mathematical/valid-output boundary derived from installed
    Transformers 5.13 ``glm_moe_dsa`` attention and vLLM v0.23.0 commit
    0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665 with FlashMLA commit
    a6ec2ba7bd0a7dff98b3f4d3e6b52b159c48d78b. A later backend will construct
    production operands and time ``flash_mla_sparse_fwd``; this reference never
    imports or calls vLLM, FlashMLA, or custom CUDA.

    Query chunking bounds the largest gathered tensor by
    ``_QUERY_CHUNK_SIZE * K * 576`` and does not change the row-independent
    result. Invalid entries are zeroed before QK and PV, so an invalid
    placeholder can never expose NaN/Inf from an unrelated cache row.
    """
    _validate(q, cache, selected_indices, softmax_scale=softmax_scale)

    output = torch.empty(
        (q.shape[0], q.shape[1], _VALUE_DIM),
        dtype=torch.bfloat16,
        device=q.device,
    )
    cache_rows = cache[:, 0, :]

    for query_start in range(0, q.shape[0], _QUERY_CHUNK_SIZE):
        query_end = min(query_start + _QUERY_CHUNK_SIZE, q.shape[0])
        query = q[query_start:query_end].float()
        indices = selected_indices[query_start:query_end, 0, :]
        valid = (indices >= 0) & (indices < cache.shape[0])

        safe_indices = indices.masked_fill(~valid, 0).to(torch.int64)
        gathered = cache_rows.index_select(0, safe_indices.reshape(-1)).reshape(
            query_end - query_start,
            selected_indices.shape[2],
            _SCORE_DIM,
        )
        gathered = gathered.float()
        gathered.masked_fill_(~valid.unsqueeze(-1), 0.0)

        scores = torch.einsum("qhd,qkd->qhk", query, gathered)
        scores.mul_(softmax_scale)
        scores.masked_fill_(~valid.unsqueeze(1), float("-inf"))

        # Softmax over all -inf is NaN. Temporarily make those rows finite,
        # then explicitly zero every invalid probability after softmax.
        has_valid = valid.any(dim=-1)
        scores.masked_fill_(~has_valid[:, None, None], 0.0)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        probabilities.masked_fill_(~valid.unsqueeze(1), 0.0)

        chunk_output = torch.einsum(
            "qhk,qkv->qhv",
            probabilities,
            gathered[..., :_VALUE_DIM],
        )
        output[query_start:query_end].copy_(chunk_output.to(torch.bfloat16))

    return output


def _validate(
    q: object,
    cache: object,
    selected_indices: object,
    *,
    softmax_scale: object,
) -> None:
    for name, tensor in (
        ("q", q),
        ("cache", cache),
        ("selected_indices", selected_indices),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    assert isinstance(q, torch.Tensor)
    assert isinstance(cache, torch.Tensor)
    assert isinstance(selected_indices, torch.Tensor)

    if type(softmax_scale) is not float:
        raise TypeError("softmax_scale must be a Python float")
    if not math.isfinite(softmax_scale):
        raise ValueError("softmax_scale must be finite")
    if softmax_scale <= 0:
        raise ValueError("softmax_scale must be positive")

    _validate_ranks_and_shapes(q, cache, selected_indices)
    _validate_dtypes_and_devices(q, cache, selected_indices)
    _validate_layouts(q, cache, selected_indices)
    _validate_finite_domain(q, cache, selected_indices)


def _validate_ranks_and_shapes(
    q: torch.Tensor,
    cache: torch.Tensor,
    selected_indices: torch.Tensor,
) -> None:
    for name, tensor in (
        ("q", q),
        ("cache", cache),
        ("selected_indices", selected_indices),
    ):
        if tensor.ndim != 3:
            raise ValueError(f"{name} must be rank 3, got rank {tensor.ndim}")

    if q.shape[0] <= 0:
        raise ValueError("q must contain at least one query")
    if q.shape[1] <= 0:
        raise ValueError("q must contain at least one head")
    if q.shape[2] != _SCORE_DIM:
        raise ValueError(f"q score width must be {_SCORE_DIM}, got {q.shape[2]}")

    if cache.shape[0] <= 0:
        raise ValueError("cache must contain at least one token")
    if cache.shape[1] != 1:
        raise ValueError(f"cache must have exactly one MQA head, got {cache.shape[1]}")
    if cache.shape[2] != _SCORE_DIM:
        raise ValueError(f"cache score width must be {_SCORE_DIM}, got {cache.shape[2]}")

    if selected_indices.shape[0] != q.shape[0]:
        raise ValueError(
            "selected_indices query dimension must match q, "
            f"got {selected_indices.shape[0]} and {q.shape[0]}"
        )
    if selected_indices.shape[1] != 1:
        raise ValueError(
            f"selected_indices must have exactly one MQA head, got {selected_indices.shape[1]}"
        )
    if selected_indices.shape[2] <= 0:
        raise ValueError("selected_indices K dimension must be positive")


def _validate_dtypes_and_devices(
    q: torch.Tensor,
    cache: torch.Tensor,
    selected_indices: torch.Tensor,
) -> None:
    if q.dtype is not torch.bfloat16:
        raise TypeError(f"q dtype must be torch.bfloat16, got {q.dtype}")
    if cache.dtype is not torch.bfloat16:
        raise TypeError(f"cache dtype must be torch.bfloat16, got {cache.dtype}")
    if selected_indices.dtype is not torch.int32:
        raise TypeError(f"selected_indices dtype must be torch.int32, got {selected_indices.dtype}")

    tensors = (q, cache, selected_indices)
    if any(tensor.device.type == "meta" for tensor in tensors):
        raise ValueError("meta tensors are not supported")
    for name, tensor in (("cache", cache), ("selected_indices", selected_indices)):
        if tensor.device != q.device:
            raise ValueError(
                f"{name} must be on the same device as q, got {tensor.device} and {q.device}"
            )


def _validate_layouts(
    q: torch.Tensor,
    cache: torch.Tensor,
    selected_indices: torch.Tensor,
) -> None:
    for name, tensor in (
        ("q", q),
        ("cache", cache),
        ("selected_indices", selected_indices),
    ):
        if tensor.layout is not torch.strided:
            raise ValueError(f"{name} must have torch.strided layout")
        # Status 1 is definite overlap. Status 2 is indeterminate and remains
        # accepted for padded outer-stride views, matching nearby references.
        if int(torch._debug_has_internal_overlap(tensor)) == 1:
            raise ValueError(f"{name} must not have internal overlap")
        if tensor.stride(-1) != 1:
            raise ValueError(f"{name} innermost stride must be 1")


def _validate_finite_domain(
    q: torch.Tensor,
    cache: torch.Tensor,
    selected_indices: torch.Tensor,
) -> None:
    if not bool(torch.isfinite(q).all().item()):
        raise ValueError("q must contain only finite values")

    valid = (selected_indices >= 0) & (selected_indices < cache.shape[0])
    if not bool(valid.any().item()):
        return
    selected_rows = torch.unique(selected_indices[valid].to(torch.int64))
    selected_cache = cache.index_select(0, selected_rows)
    if not bool(torch.isfinite(selected_cache).all().item()):
        raise ValueError("valid selected cache rows must contain only finite values")
