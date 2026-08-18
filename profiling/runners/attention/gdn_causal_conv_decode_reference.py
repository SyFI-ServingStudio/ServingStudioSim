"""Torch semantics for Qwen3.5/3.6 GDN causal-convolution decode."""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["gdn_causal_conv_decode_reference"]

_SUPPORTED_KERNEL_SIZES = frozenset(range(2, 7))


def gdn_causal_conv_decode_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    slot_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one non-speculative GDN depthwise-convolution decode step.

    ``x`` is ``[B, C]``, depthwise ``weight`` is ``[C, W]``, dim-first
    slot-indexed ``state`` is ``[S, C, W - 1]``, and ``slot_indices`` is
    ``[B]``. Qwen3.6 uses ``C=8192`` and ``W=4``. Activations, weights, state,
    and output are BF16, while the channelwise dot product and SiLU are FP32.

    Slot zero is reserved by vLLM's cache layout, so every active row must name
    a distinct slot in ``[1, S)``. Selected state slots are shifted left and
    receive ``x`` as their newest sample. State is mutated in place and returned
    by identity; output is newly allocated, and ``x`` and ``weight`` are not
    mutated. All validation completes before state mutation.

    The first reference intentionally excludes speculative or multi-token
    decode, varlen metadata, bias, activations other than SiLU, non-BF16 data,
    duplicate/invalid slots, non-dim-first state, and kernel widths outside the
    production Triton implementation's supported range of 2 through 6.
    """
    _validate(x, weight, state, slot_indices)

    selected_state = state.index_select(0, slot_indices.to(torch.int64))
    window = torch.cat((selected_state, x.unsqueeze(-1)), dim=-1)
    accumulator = torch.sum(window.float() * weight.float().unsqueeze(0), dim=-1)
    output = F.silu(accumulator).to(torch.bfloat16)
    updated_state = window[..., 1:]

    state.index_copy_(0, slot_indices.to(torch.int64), updated_state)
    return output, state


def _validate(
    x: object,
    weight: object,
    state: object,
    slot_indices: object,
) -> None:
    tensors = {
        "x": x,
        "weight": weight,
        "state": state,
        "slot_indices": slot_indices,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")

    assert all(isinstance(tensor, torch.Tensor) for tensor in tensors.values())
    x = tensors["x"]
    weight = tensors["weight"]
    state = tensors["state"]
    slot_indices = tensors["slot_indices"]

    expected_ranks = {"x": 2, "weight": 2, "state": 3, "slot_indices": 1}
    for name, tensor in tensors.items():
        expected_rank = expected_ranks[name]
        if tensor.ndim != expected_rank:
            raise ValueError(f"{name} must be rank {expected_rank}, got rank {tensor.ndim}")
        if any(dimension <= 0 for dimension in tensor.shape):
            raise ValueError(f"{name} dimensions must be positive")

    for name in ("x", "weight", "state"):
        tensor = tensors[name]
        if tensor.dtype is not torch.bfloat16:
            raise TypeError(f"{name} dtype must be torch.bfloat16, got {tensor.dtype}")
    if slot_indices.dtype is not torch.int32:
        raise TypeError(f"slot_indices dtype must be torch.int32, got {slot_indices.dtype}")

    for name, tensor in tensors.items():
        if tensor.layout is not torch.strided:
            raise ValueError(f"{name} must have torch.strided layout")
        if tensor.device != x.device:
            raise ValueError(
                f"{name} must be on the same device as x, got {tensor.device} and {x.device}"
            )
    if x.device.type == "meta":
        raise ValueError("meta tensors are not supported")

    batch_size, channels = x.shape
    if slot_indices.shape != (batch_size,):
        raise ValueError(
            f"slot_indices must have shape ({batch_size},), got {tuple(slot_indices.shape)}"
        )
    if weight.shape[0] != channels:
        raise ValueError(f"weight must have {channels} channels, got {weight.shape[0]}")

    kernel_size = weight.shape[1]
    if kernel_size not in _SUPPORTED_KERNEL_SIZES:
        raise ValueError(
            "kernel_size must be supported by the production Triton kernel "
            f"({min(_SUPPORTED_KERNEL_SIZES)}..{max(_SUPPORTED_KERNEL_SIZES)}), "
            f"got {kernel_size}"
        )
    expected_state_shape = (state.shape[0], channels, kernel_size - 1)
    if state.shape != expected_state_shape:
        raise ValueError(
            f"state must use dim-first shape {expected_state_shape}, got {tuple(state.shape)}"
        )

    if torch.any(slot_indices <= 0).item():
        raise ValueError("slot_indices must not select reserved slot zero or negative slots")
    if torch.any(slot_indices >= state.shape[0]).item():
        raise ValueError(f"slot_indices must be less than slot count {state.shape[0]}")
    if torch.unique(slot_indices).numel() != batch_size:
        raise ValueError("slot_indices must be unique for independent state mutation")
