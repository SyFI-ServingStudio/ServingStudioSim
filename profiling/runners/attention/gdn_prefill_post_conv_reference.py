"""Torch semantics for Qwen3.5/3.6 GDN prefill post-convolution prep."""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["gdn_prefill_post_conv_reference"]

_L2NORM_EPSILON = 1e-6
_SOFTPLUS_THRESHOLD = 20.0


def gdn_prefill_post_conv_reference(
    conv_output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    num_qk_heads: int,
    num_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare normalized Q/K, V, decay, and beta for chunked GDN prefill.

    ``conv_output`` is the raw token-major packed ``[Q, K, V]`` tensor with
    shape ``[L, 2 * H * K + HV * V]``. Activations ``conv_output``, ``a``, and
    ``b`` are BF16; learned ``A_log`` and ``dt_bias`` are FP32. Q/K L2
    statistics, softplus decay, and sigmoid beta are computed in FP32. Q/K are
    rounded once to BF16 after normalization, V is copied bit-exactly, and
    ``g``/``beta`` remain FP32.

    This freezes Qwen's production options: L2 normalization with epsilon
    ``1e-6``, beta-one softplus with threshold ``20``, and log-space ``g``
    output (no final exponentiation). Strided semantic tensors are accepted;
    production contiguity and device constraints belong to its backend. All
    outputs are newly allocated, contiguous, and inputs are never mutated.
    """
    dimensions = _validate(
        conv_output,
        a,
        b,
        A_log,
        dt_bias,
        num_qk_heads=num_qk_heads,
        num_value_heads=num_value_heads,
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
    )
    num_tokens, qk_heads, value_heads, key_dim, value_dim = dimensions

    q_width = qk_heads * key_dim
    value_width = value_heads * value_dim
    q_flat, k_flat, v_flat = torch.split(
        conv_output,
        [q_width, q_width, value_width],
        dim=-1,
    )

    q_fp32 = q_flat.reshape(num_tokens, qk_heads, key_dim).float()
    k_fp32 = k_flat.reshape(num_tokens, qk_heads, key_dim).float()
    q_rstd = torch.rsqrt(q_fp32.square().sum(dim=-1, keepdim=True) + _L2NORM_EPSILON)
    k_rstd = torch.rsqrt(k_fp32.square().sum(dim=-1, keepdim=True) + _L2NORM_EPSILON)
    q = (q_fp32 * q_rstd).to(torch.bfloat16).contiguous()
    k = (k_fp32 * k_rstd).to(torch.bfloat16).contiguous()

    # ``contiguous`` alone could retain storage when the packed input already
    # has the target layout. Clone explicitly because the semantic contract
    # requires a fresh V output just like the production wrapper's allocation.
    v = v_flat.reshape(num_tokens, value_heads, value_dim).clone().contiguous()

    gate_input = a.float() + dt_bias.float()
    softplus = F.softplus(
        gate_input,
        beta=1.0,
        threshold=_SOFTPLUS_THRESHOLD,
    )
    g = (-torch.exp(A_log.float()) * softplus).contiguous()
    beta = torch.sigmoid(b.float()).contiguous()
    return q, k, v, g, beta


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _validate(
    conv_output: object,
    a: object,
    b: object,
    A_log: object,
    dt_bias: object,
    *,
    num_qk_heads: object,
    num_value_heads: object,
    key_head_dim: object,
    value_head_dim: object,
) -> tuple[int, int, int, int, int]:
    qk_heads = _positive_int("num_qk_heads", num_qk_heads)
    value_heads = _positive_int("num_value_heads", num_value_heads)
    key_dim = _positive_int("key_head_dim", key_head_dim)
    value_dim = _positive_int("value_head_dim", value_head_dim)

    tensors = {
        "conv_output": conv_output,
        "a": a,
        "b": b,
        "A_log": A_log,
        "dt_bias": dt_bias,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    assert all(isinstance(tensor, torch.Tensor) for tensor in tensors.values())
    conv_output = tensors["conv_output"]
    a = tensors["a"]
    b = tensors["b"]
    A_log = tensors["A_log"]
    dt_bias = tensors["dt_bias"]

    expected_ranks = {"conv_output": 2, "a": 2, "b": 2, "A_log": 1, "dt_bias": 1}
    for name, tensor in tensors.items():
        expected_rank = expected_ranks[name]
        if tensor.ndim != expected_rank:
            raise ValueError(f"{name} must be rank {expected_rank}, got rank {tensor.ndim}")
        if any(dimension <= 0 for dimension in tensor.shape):
            raise ValueError(f"{name} dimensions must be positive")
        if tensor.layout is not torch.strided:
            raise ValueError(f"{name} must have torch.strided layout")

    if any(tensor.device.type == "meta" for tensor in tensors.values()):
        raise ValueError("meta tensors are not supported")
    for name, tensor in tensors.items():
        if tensor.device != conv_output.device:
            raise ValueError(
                f"{name} must be on the same device as conv_output, got "
                f"{tensor.device} and {conv_output.device}"
            )

    for name in ("conv_output", "a", "b"):
        tensor = tensors[name]
        if tensor.dtype is not torch.bfloat16:
            raise TypeError(f"{name} dtype must be torch.bfloat16, got {tensor.dtype}")
    for name in ("A_log", "dt_bias"):
        tensor = tensors[name]
        if tensor.dtype is not torch.float32:
            raise TypeError(f"{name} dtype must be torch.float32, got {tensor.dtype}")

    num_tokens = conv_output.shape[0]
    packed_width = 2 * qk_heads * key_dim + value_heads * value_dim
    if conv_output.shape != (num_tokens, packed_width):
        raise ValueError(
            f"conv_output must have shape ({num_tokens}, {packed_width}), "
            f"got {tuple(conv_output.shape)}"
        )
    expected_gate_shape = (num_tokens, value_heads)
    for name, tensor in (("a", a), ("b", b)):
        if tensor.shape != expected_gate_shape:
            raise ValueError(
                f"{name} must have shape {expected_gate_shape}, got {tuple(tensor.shape)}"
            )
    expected_parameter_shape = (value_heads,)
    for name, tensor in (("A_log", A_log), ("dt_bias", dt_bias)):
        if tensor.shape != expected_parameter_shape:
            raise ValueError(
                f"{name} must have shape {expected_parameter_shape}, got {tuple(tensor.shape)}"
            )
    return num_tokens, qk_heads, value_heads, key_dim, value_dim
