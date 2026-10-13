"""Independent value reference for static per-tensor FP8 quantization."""

from typing import Any

FP8_E4M3_MAX = 448.0


def static_fp8_quant_reference(torch: Any, values: Any, gate: Any | None, scale: Any) -> Any:
    values = values.float()
    if gate is not None:
        values = values * torch.sigmoid(gate.float())
    return (values / scale.float()).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)


__all__ = ["static_fp8_quant_reference"]
