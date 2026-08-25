"""Independent value reference for DeepSeek's clamped SwiGLU."""

from typing import Any

CLAMP_LIMIT = 10.0


def clamped_swiglu_reference(torch: Any, input_tensor: Any) -> Any:
    if input_tensor.ndim != 2 or input_tensor.shape[1] % 2:
        raise ValueError("input_tensor must have shape [rows, 2 * hidden_dim]")
    if input_tensor.dtype is not torch.bfloat16:
        raise TypeError("input_tensor must use bfloat16")
    if not input_tensor.is_contiguous() or any(size <= 0 for size in input_tensor.shape):
        raise ValueError("input_tensor must be non-empty contiguous storage")
    gate, up = input_tensor.chunk(2, dim=1)
    gate = torch.clamp(gate, max=CLAMP_LIMIT)
    up = torch.clamp(up, min=-CLAMP_LIMIT, max=CLAMP_LIMIT)
    return torch.nn.functional.silu(gate) * up


__all__ = ["CLAMP_LIMIT", "clamped_swiglu_reference"]
