"""Torch semantics for Qwen3.5/3.6 Gated DeltaNet recurrent decode."""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["gdn_recurrent_decode_reference"]

_L2_NORM_EPS = 1e-6


def gdn_recurrent_decode_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one non-speculative Gated DeltaNet decode step.

    Inputs are semantic tensors, not vLLM's packed QKV or slot-indexed state:
    Q/K are ``[B, Hq, Dk]``, V is ``[B, Hv, Dv]``, raw ``a``/``b`` are
    ``[B, Hv]``, learned ``A_log``/``dt_bias`` are ``[Hv]``, and state is
    ``[B, Hv, Dk, Dv]``. Q/K/V/a/b and the returned core output are BF16;
    parameters and recurrent state are FP32.

    For ``Qwen/Qwen3.6-35B-A3B-FP8``, ``Hq=16``, ``Hv=32``, and
    ``Dk=Dv=128``, so state is ``[B, 32, 128, 128]``. The implementation stays
    dimension-parameterized for smaller correctness tests and compatible GDNs.

    Q/K heads are repeated consecutively to value heads, matching Qwen's
    ``repeat_interleave(Hv // Hq)``. The recurrent state is updated in place and
    returned by identity, as in vLLM's packed decode kernel. All validation is
    completed before mutation.

    This first reference intentionally excludes multi-token/speculative decode,
    interleaved or packed QKV layouts, non-FP32 state/parameters, non-BF16
    activations, and head layouts where ``Hv`` is not divisible by ``Hq``.

    Semantics follow Transformers 5.13 ``torch_recurrent_gated_delta_rule`` and
    vLLM's ``fused_recurrent_gated_delta_rule_packed_decode_kernel``. The packed
    kernel computes sigmoid in FP32, rounds beta through the activation dtype,
    and then applies the recurrence in FP32; this reference preserves that
    rounding point.
    """
    _validate(query, key, value, a, b, A_log, dt_bias, state)

    num_value_heads = value.shape[1]
    repeats = num_value_heads // query.shape[1]

    query_fp32 = query.float()
    key_fp32 = key.float()
    query_fp32 = query_fp32 / torch.sqrt(
        torch.sum(query_fp32 * query_fp32, dim=-1, keepdim=True) + _L2_NORM_EPS
    )
    key_fp32 = key_fp32 / torch.sqrt(
        torch.sum(key_fp32 * key_fp32, dim=-1, keepdim=True) + _L2_NORM_EPS
    )
    query_fp32 = query_fp32.repeat_interleave(repeats, dim=1)
    key_fp32 = key_fp32.repeat_interleave(repeats, dim=1)
    query_fp32 = query_fp32 * (query.shape[-1] ** -0.5)

    decay_log = -torch.exp(A_log) * F.softplus(a.float() + dt_bias)
    decay = torch.exp(decay_log).unsqueeze(-1).unsqueeze(-1)
    beta = torch.sigmoid(b.float()).to(b.dtype).float().unsqueeze(-1)

    decayed_state = state * decay
    memory = torch.sum(decayed_state * key_fp32.unsqueeze(-1), dim=-2)
    delta = (value.float() - memory) * beta
    updated_state = decayed_state + key_fp32.unsqueeze(-1) * delta.unsqueeze(-2)
    output = torch.sum(updated_state * query_fp32.unsqueeze(-1), dim=-2)

    state.copy_(updated_state)
    return output.to(value.dtype), state


def _validate(
    query: object,
    key: object,
    value: object,
    a: object,
    b: object,
    A_log: object,
    dt_bias: object,
    state: object,
) -> None:
    tensors = {
        "query": query,
        "key": key,
        "value": value,
        "a": a,
        "b": b,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "state": state,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    assert all(isinstance(tensor, torch.Tensor) for tensor in tensors.values())
    query = tensors["query"]
    key = tensors["key"]
    value = tensors["value"]
    a = tensors["a"]
    b = tensors["b"]
    A_log = tensors["A_log"]
    dt_bias = tensors["dt_bias"]
    state = tensors["state"]

    expected_ranks = {
        "query": 3,
        "key": 3,
        "value": 3,
        "a": 2,
        "b": 2,
        "A_log": 1,
        "dt_bias": 1,
        "state": 4,
    }
    for name, tensor in tensors.items():
        expected_rank = expected_ranks[name]
        if tensor.ndim != expected_rank:
            raise ValueError(f"{name} must be rank {expected_rank}, got rank {tensor.ndim}")
        if any(dimension <= 0 for dimension in tensor.shape):
            raise ValueError(f"{name} dimensions must be positive")

    for name in ("query", "key", "value", "a", "b"):
        tensor = tensors[name]
        if tensor.dtype is not torch.bfloat16:
            raise TypeError(f"{name} dtype must be torch.bfloat16, got {tensor.dtype}")
    for name in ("A_log", "dt_bias", "state"):
        tensor = tensors[name]
        if tensor.dtype is not torch.float32:
            raise TypeError(f"{name} dtype must be torch.float32, got {tensor.dtype}")

    for name, tensor in tensors.items():
        if tensor.layout is not torch.strided:
            raise ValueError(f"{name} must have torch.strided layout")
        if tensor.device != query.device:
            raise ValueError(
                f"{name} must be on the same device as query, "
                f"got {tensor.device} and {query.device}"
            )
    if query.device.type == "meta":
        raise ValueError("meta tensors are not supported")

    if key.shape != query.shape:
        raise ValueError(
            f"key shape must match query shape {tuple(query.shape)}, got {tuple(key.shape)}"
        )

    batch, num_query_heads, key_dim = query.shape
    num_value_heads, value_dim = value.shape[1:]
    if value.shape[0] != batch:
        raise ValueError("value batch dimension must match query")
    if num_value_heads % num_query_heads != 0:
        raise ValueError(
            "value head count must be divisible by query/key head count, "
            f"got {num_value_heads} and {num_query_heads}"
        )

    expected_gate_shape = (batch, num_value_heads)
    for name, tensor in (("a", a), ("b", b)):
        if tensor.shape != expected_gate_shape:
            raise ValueError(
                f"{name} must have shape {expected_gate_shape}, got {tuple(tensor.shape)}"
            )
    expected_parameter_shape = (num_value_heads,)
    for name, tensor in (("A_log", A_log), ("dt_bias", dt_bias)):
        if tensor.shape != expected_parameter_shape:
            raise ValueError(
                f"{name} must have shape {expected_parameter_shape}, got {tuple(tensor.shape)}"
            )

    expected_state_shape = (batch, num_value_heads, key_dim, value_dim)
    if state.shape != expected_state_shape:
        raise ValueError(f"state must have shape {expected_state_shape}, got {tuple(state.shape)}")
