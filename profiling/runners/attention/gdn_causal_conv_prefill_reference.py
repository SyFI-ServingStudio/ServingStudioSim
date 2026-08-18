"""Torch semantics for fresh Qwen3.5/3.6 GDN causal-convolution prefill."""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["gdn_causal_conv_prefill_reference"]

_SUPPORTED_KERNEL_SIZES = frozenset(range(2, 5))


def gdn_causal_conv_prefill_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    state: torch.Tensor,
    slot_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run fresh homogeneous GDN depthwise-convolution prefill.

    ``x`` is ``[B, L, C]``, depthwise ``weight`` is ``[C, W]``, dim-first
    slot-indexed ``state`` is ``[S, C, W - 1]``, and ``slot_indices`` is
    ``[B]``. Qwen3.6 uses ``C=8192`` and ``W=4``. Activations, weights, state,
    and output are BF16, while the channelwise dot product and SiLU are FP32.

    This reference models fresh prefill only: each sequence starts with
    ``W - 1`` zeros, so prior selected state is not read. Each selected slot is
    overwritten with the last ``W - 1`` raw input samples, left-zero-padded
    when the sequence is shorter. Slot zero is reserved, active slots must be
    distinct, state is mutated in place and returned by identity, and output is
    newly allocated. Validation completes before state mutation.

    The semantic reference accepts strided non-contiguous tensors. Packing and
    contiguity are constraints of a later production backend, not of this math.
    It intentionally excludes initial-state/APC continuation, variable-length
    or padded batches, speculative/mixed decode, bias, non-SiLU activation,
    non-BF16 data, and widths outside the production prefill branches 2..4.
    """
    _validate(x, weight, state, slot_indices)

    batch_size, sequence_length, channels = x.shape
    kernel_size = weight.shape[1]
    state_length = kernel_size - 1

    channel_first = x.float().permute(0, 2, 1)
    padded = F.pad(channel_first, (state_length, 0))
    windows = padded.unfold(dimension=-1, size=kernel_size, step=1)
    accumulator = torch.sum(windows * weight.float().view(1, channels, 1, kernel_size), dim=-1)
    output = F.silu(accumulator).permute(0, 2, 1).contiguous().to(torch.bfloat16)

    if sequence_length >= state_length:
        updated_state = x[:, -state_length:, :].permute(0, 2, 1)
    else:
        updated_state = torch.zeros(
            (batch_size, channels, state_length), dtype=x.dtype, device=x.device
        )
        updated_state[..., -sequence_length:] = x.permute(0, 2, 1)

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

    expected_ranks = {"x": 3, "weight": 2, "state": 3, "slot_indices": 1}
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

    batch_size, _, channels = x.shape
    if slot_indices.shape != (batch_size,):
        raise ValueError(
            f"slot_indices must have shape ({batch_size},), got {tuple(slot_indices.shape)}"
        )
    if weight.shape[0] != channels:
        raise ValueError(f"weight must have {channels} channels, got {weight.shape[0]}")

    kernel_size = weight.shape[1]
    if kernel_size not in _SUPPORTED_KERNEL_SIZES:
        raise ValueError(
            "kernel_size must be supported by the production prefill Triton kernel "
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
