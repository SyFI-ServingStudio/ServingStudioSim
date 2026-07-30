"""Layout-agnostic Torch semantic reference for batched matrix multiplication."""

from __future__ import annotations

import torch

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)


def batched_gemm_reference(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute ``[B, M, K] @ [B, K, N]`` with FP32 accumulation.

    The optional output may be an arbitrary non-overlapping strided view. When
    supplied, it is updated and returned without mutating either input.
    """
    # Source boundary: vLLM v0.23.0 commit 0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
    # uses BMM for MLA Q absorption (H,M,192)x(H,192,512) and V-up
    # (H,M,512)x(H,512,256). Profiling backends construct those GLM layouts.
    _validate_inputs(lhs, rhs)
    expected_shape = (lhs.shape[0], lhs.shape[1], rhs.shape[2])
    if out is not None:
        _validate_out(out, lhs, rhs, expected_shape)

    result_fp32 = torch.einsum(
        "bmk,bkn->bmn",
        lhs.to(dtype=torch.float32),
        rhs.to(dtype=torch.float32),
    )
    result = result_fp32.to(dtype=lhs.dtype)
    if out is None:
        return result

    try:
        out.copy_(result)
    except RuntimeError as exc:
        raise ValueError("out must be writable") from exc
    return out


def _validate_inputs(lhs: object, rhs: object) -> None:
    if not isinstance(lhs, torch.Tensor):
        raise TypeError("lhs must be a torch.Tensor")
    if not isinstance(rhs, torch.Tensor):
        raise TypeError("rhs must be a torch.Tensor")

    if lhs.ndim != 3:
        raise ValueError(f"lhs must be rank 3, got rank {lhs.ndim}")
    if rhs.ndim != 3:
        raise ValueError(f"rhs must be rank 3, got rank {rhs.ndim}")
    if any(dim <= 0 for dim in lhs.shape):
        raise ValueError(f"lhs dimensions must be positive, got {tuple(lhs.shape)}")
    if any(dim <= 0 for dim in rhs.shape):
        raise ValueError(f"rhs dimensions must be positive, got {tuple(rhs.shape)}")

    if lhs.shape[0] != rhs.shape[0]:
        raise ValueError(
            f"lhs and rhs must have the same batch dimension, got {lhs.shape[0]} "
            f"and {rhs.shape[0]}"
        )
    if lhs.shape[2] != rhs.shape[1]:
        raise ValueError(
            f"lhs and rhs inner K dimensions must match, got {lhs.shape[2]} "
            f"and {rhs.shape[1]}"
        )

    if lhs.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "lhs dtype must be torch.bfloat16, torch.float16, or torch.float32, "
            f"got {lhs.dtype}"
        )
    if rhs.dtype != lhs.dtype:
        raise TypeError(
            f"lhs and rhs must have the same dtype, got {lhs.dtype} and {rhs.dtype}"
        )
    if rhs.device != lhs.device:
        raise ValueError(
            f"lhs and rhs must be on the same device, got {lhs.device} and {rhs.device}"
        )
    if lhs.layout is not torch.strided or rhs.layout is not torch.strided:
        raise ValueError("lhs and rhs must have torch.strided layout")


def _validate_out(
    out: object,
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    expected_shape: tuple[int, int, int],
) -> None:
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    if out.ndim != 3:
        raise ValueError(f"out must be rank 3, got rank {out.ndim}")
    if tuple(out.shape) != expected_shape:
        raise ValueError(f"out must have shape {expected_shape}, got {tuple(out.shape)}")
    if out.dtype != lhs.dtype:
        raise TypeError(f"out dtype must match input dtype {lhs.dtype}, got {out.dtype}")
    if out.device != lhs.device:
        raise ValueError(
            f"out must be on the same device as inputs, got {out.device} and {lhs.device}"
        )
    if out.layout is not torch.strided:
        raise ValueError("out must have torch.strided layout")

    # Status 1 is definite overlap; status 2 means Torch could not prove either
    # outcome and must remain accepted for valid arbitrary-strided views.
    if int(torch._debug_has_internal_overlap(out)) == 1:
        raise ValueError("out must not have internal overlap")
    if torch._C._overlaps(out, lhs) or torch._C._overlaps(out, rhs):
        raise ValueError("out must not alias lhs or rhs")
