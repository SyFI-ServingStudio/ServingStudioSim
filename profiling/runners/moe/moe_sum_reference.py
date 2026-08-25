"""Independent value reference for DeepSeek's routed-expert reduction."""

from typing import Any


def moe_sum_reference(torch: Any, input_tensor: Any) -> Any:
    if input_tensor.ndim != 3:
        raise ValueError("input_tensor must have shape [tokens, top_k, hidden_dim]")
    if input_tensor.dtype is not torch.bfloat16:
        raise TypeError("input_tensor must use bfloat16")
    if not input_tensor.is_contiguous() or any(size <= 0 for size in input_tensor.shape):
        raise ValueError("input_tensor must be non-empty contiguous storage")
    return input_tensor.float().sum(dim=1).to(torch.bfloat16)


__all__ = ["moe_sum_reference"]
