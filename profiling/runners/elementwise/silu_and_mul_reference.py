"""Independent value reference for SiLU-and-mul."""

from typing import Any


def silu_and_mul_reference(torch: Any, input_tensor: Any) -> Any:
    gate, up = input_tensor.float().chunk(2, dim=-1)
    return (torch.nn.functional.silu(gate) * up).to(input_tensor.dtype)


__all__ = ["silu_and_mul_reference"]
