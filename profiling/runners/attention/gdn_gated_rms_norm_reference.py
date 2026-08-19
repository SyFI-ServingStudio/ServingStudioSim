"""Torch semantics for Qwen3.5/3.6 GDN gated RMS normalization."""

from __future__ import annotations

import torch

__all__ = ["gdn_gated_rms_norm_reference"]

_EPSILON = 1e-6


def gdn_gated_rms_norm_reference(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Apply Qwen's norm-before-SiLU-gate output normalization.

    ``x`` and output gate ``z`` are ``[M, hidden]`` and ``weight`` is
    ``[hidden]``. Qwen3.6 reshapes its GDN output to ``hidden=128`` and
    ``M=num_tokens * 32``. All semantic inputs and the returned output are
    BF16. RMS statistics, affine multiplication, SiLU, and gating are computed
    in FP32; only the final output is rounded to BF16.

    This reference freezes epsilon to ``1e-6``, has no bias or groups, and
    always normalizes before applying a SiLU gate. It intentionally excludes
    non-BF16 data, sigmoid-only or alternate activations, norm-after-gate, and
    alternate epsilon. Non-contiguous strided tensors are supported by the
    semantic math; contiguous feature storage is a production Triton backend
    constraint rather than a semantic axis. Inputs are not mutated, and the
    returned tensor has fresh storage.
    """
    _validate(x, z, weight)

    x_fp32 = x.float()
    z_fp32 = z.float()
    weight_fp32 = weight.float()
    rstd = torch.rsqrt(x_fp32.square().mean(dim=-1, keepdim=True) + _EPSILON)
    normalized = x_fp32 * rstd * weight_fp32
    gate = z_fp32 * torch.sigmoid(z_fp32)
    return (normalized * gate).to(torch.bfloat16)


def _validate(x: object, z: object, weight: object) -> None:
    tensors = {"x": x, "z": z, "weight": weight}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    assert all(isinstance(tensor, torch.Tensor) for tensor in tensors.values())
    x = tensors["x"]
    z = tensors["z"]
    weight = tensors["weight"]

    expected_ranks = {"x": 2, "z": 2, "weight": 1}
    for name, tensor in tensors.items():
        expected_rank = expected_ranks[name]
        if tensor.ndim != expected_rank:
            raise ValueError(f"{name} must be rank {expected_rank}, got rank {tensor.ndim}")
        if any(dimension <= 0 for dimension in tensor.shape):
            raise ValueError(f"{name} dimensions must be positive")
        if tensor.dtype is not torch.bfloat16:
            raise TypeError(f"{name} dtype must be torch.bfloat16, got {tensor.dtype}")
        if tensor.layout is not torch.strided:
            raise ValueError(f"{name} must have torch.strided layout")
        if tensor.device != x.device:
            raise ValueError(
                f"{name} must be on the same device as x, got {tensor.device} and {x.device}"
            )

    if x.device.type == "meta":
        raise ValueError("meta tensors are not supported")
    if z.shape != x.shape:
        raise ValueError(f"z shape must match x shape {tuple(x.shape)}, got {tuple(z.shape)}")
    expected_weight_shape = (x.shape[1],)
    if weight.shape != expected_weight_shape:
        raise ValueError(
            f"weight must have shape {expected_weight_shape}, got {tuple(weight.shape)}"
        )
