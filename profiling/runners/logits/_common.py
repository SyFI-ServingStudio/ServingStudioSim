"""Input contracts shared by logits profiling runners."""

from typing import Any

from profiling.db.args import DType
from profiling.runners.exceptions import ProfilerNotImplemented


def positive_int(name: str, value: int) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def logits_dtype(value: DType | str) -> DType:
    dtype = DType.from_value(value)
    if dtype not in (DType.BF16, DType.FP32):
        raise ProfilerNotImplemented("logits profiling supports only bf16 and fp32")
    return dtype


def validate_shape(num_rows: int, vocab_size: int, row_stride: int) -> None:
    positive_int("num_rows", num_rows)
    positive_int("vocab_size", vocab_size)
    positive_int("row_stride", row_stride)
    if row_stride < vocab_size:
        raise ValueError("row_stride must be >= vocab_size")


def require_b200(torch: Any) -> None:
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("logits profiling requires CUDA")
    name = str(torch.cuda.get_device_name(torch.cuda.current_device()))
    if name != "NVIDIA B200":
        raise ProfilerNotImplemented(
            f"logits profiling currently targets NVIDIA B200, got {name!r}"
        )


def make_logits(
    torch: Any, num_rows: int, vocab_size: int, row_stride: int, dtype: Any, *, device: Any
) -> Any:
    storage = torch.full((num_rows, row_stride), float("nan"), dtype=dtype, device=device)
    logits = storage[:, :vocab_size]
    generator = torch.Generator(device=device).manual_seed(0)
    logits.copy_(
        torch.randn((num_rows, vocab_size), dtype=dtype, device=device, generator=generator)
    )
    return logits
