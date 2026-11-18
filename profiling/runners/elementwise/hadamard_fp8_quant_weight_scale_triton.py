"""Profile the vendored Triton Hadamard quant + head-weight scale epilogue (backend ``triton``).

The kernel (``hadamard_wscale_vendor/triton_post_b.py``, ``post_b``) reads the
indexer query heads in place from the concatenated ``[q_b | wq_b]`` GEMM output
(row stride ``lead_cols + 32 * 128``, heads from column ``lead_cols``), runs
the vLLM fork's ``fwht128_quant_fp8`` butterflies and UE8M0 FP8 quant on them,
and writes the head weights times each head's scale times
``128^-0.5 * 32^-0.5`` -- what the fork does as
``fwht128_quant_fp8`` on the contiguous ``wq_b`` output plus the Inductor
weight-scale kernel.

Gate before timing: the FP8 heads, the scales and the scaled weights equal,
byte for byte, the fork's ``fwht128_quant_fp8`` on a contiguous copy of the
span followed by the weight scale (power-of-two factors, so any multiply order
is exact). The FP8 heads are also checked against the float64 dense-Hadamard
reference of ``hadamard_fp8_quant``.
"""

from __future__ import annotations

import importlib

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, OOMError, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "hadamard_fp8_quant_weight_scale:triton"
_FORK_MODULE = "vllm.models.glm5next.nvidia.ops.kpool_compress"
# (num_heads, head_dim) compiled into _post_b (rows // 32, 128 columns, 0.015625 weight scale).
_COMPILED_HEADS = (32, 128)
# _post_b forms the input offset tok * ldy in int32: past 2^31 elements it wraps
# (seen at 131072 tokens x 20480 columns: 80% of the FP8 bytes wrong).
_MAX_INPUT_ELEMENTS = 2**31


def _validate_args(
    num_tokens: int, num_heads: int, head_dim: int, lead_cols: int, input_dtype: DType | str
) -> None:
    for name, value in (
        ("num_tokens", num_tokens),
        ("num_heads", num_heads),
        ("head_dim", head_dim),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    if type(lead_cols) is not int or lead_cols < 0:
        raise ValueError(f"lead_cols must be a non-negative integer, got {lead_cols!r}")
    if DType.from_value(input_dtype) is not DType.BF16:
        raise ProfilerNotImplemented(f"{_BACKEND} requires input_dtype=bf16")
    if (num_heads, head_dim) != _COMPILED_HEADS:
        raise ProfilerNotImplemented(
            f"{_BACKEND} has (num_heads, head_dim) = {_COMPILED_HEADS} compiled in, "
            f"got {(num_heads, head_dim)}"
        )
    if num_tokens * (lead_cols + num_heads * head_dim) >= _MAX_INPUT_ELEMENTS:
        raise ProfilerNotImplemented(
            f"{_BACKEND} indexes its input with int32 offsets: num_tokens * row width must "
            f"stay below 2^31, got {num_tokens} x {lead_cols + num_heads * head_dim}"
        )


def logical_bytes(num_tokens: int, num_heads: int, head_dim: int) -> int:
    """BF16 heads and FP32 weights in; FP8 heads, FP32 scales and FP32 weights out."""
    return num_tokens * num_heads * (head_dim * (2 + 1) + 3 * 4)


def fork_pair(torch, fwht128_quant_fp8, y, lead_cols, hw, num_heads, head_dim):
    """The fork's launches on a contiguous copy of the span: FWHT quant, then the weight scale."""
    span = y[:, lead_cols : lead_cols + num_heads * head_dim].contiguous()
    q_fp8, q_scale = fwht128_quant_fp8(span.view(-1, head_dim))
    factor = head_dim**-0.5 * num_heads**-0.5
    weights = (hw.unsqueeze(-1) * q_scale.view(-1, num_heads, 1) * factor).squeeze(-1)
    return q_fp8.view(torch.uint8), q_scale.view(-1), weights.reshape(-1)


def profile_hadamard_fp8_quant_weight_scale_triton(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    lead_cols: int,
    input_dtype: DType | str,
) -> ComputeMetrics:
    _validate_args(num_tokens, num_heads, head_dim, lead_cols, input_dtype)
    try:
        import torch

        from profiling.runners.elementwise.hadamard_fp8_quant_vllm_triton import reference
        from profiling.runners.elementwise.hadamard_wscale_vendor.triton_post_b import post_b

        fwht128_quant_fp8 = importlib.import_module(_FORK_MODULE).fwht128_quant_fp8
    except (ImportError, AttributeError) as exc:
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires Torch, Triton and the vLLM fork ({_FORK_MODULE})"
        ) from exc

    rows = num_tokens * num_heads
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        g = torch.Generator(device=device).manual_seed(0)
        y = torch.randn(
            (num_tokens, lead_cols + num_heads * head_dim), device=device, generator=g
        ).to(torch.bfloat16)
        hw = torch.randn((num_tokens, num_heads), dtype=torch.float32, device=device, generator=g)
        q_out = torch.empty((rows, head_dim), dtype=torch.uint8, device=device)
        s_out = torch.empty(rows, dtype=torch.float32, device=device)
        w_out = torch.empty(rows, dtype=torch.float32, device=device)

        def launch() -> None:
            post_b(y, lead_cols, hw, q_out.view(torch.float8_e4m3fn), s_out, w_out)

        q_out.fill_(0x7F)
        s_out.fill_(float("nan"))
        w_out.fill_(float("nan"))
        launch()
        want_q, want_s, want_w = fork_pair(
            torch, fwht128_quant_fp8, y, lead_cols, hw, num_heads, head_dim
        )
        torch.cuda.synchronize()
        for name, got, want in (
            ("FP8 heads", q_out, want_q),
            ("scales", s_out, want_s),
            ("scaled weights", w_out, want_w),
        ):
            if not torch.equal(got.view(torch.uint8), want.view(torch.uint8)):
                same = (got.view(torch.uint8) == want.view(torch.uint8)).float().mean().item()
                raise AssertionError(
                    f"{_BACKEND} {name} are not byte-identical to the fork's fwht128_quant_fp8 "
                    f"+ weight scale ({same:.6f} of bytes equal)"
                )
        # Independent oracle on a bounded prefix plus the ragged tail.
        idx = torch.cat(
            (
                torch.arange(min(rows, 4096), device=device),
                torch.arange(max(rows - 64, min(rows, 4096)), rows, device=device),
            )
        )
        span = y[:, lead_cols:].reshape(rows, head_dim)
        ref_q, ref_s = reference(torch, span[idx])
        got_s = s_out[idx].double().view(-1, 1)
        if int((got_s != ref_s).sum()) > max(1, idx.numel() // 1000):
            raise AssertionError("head scales differ from the float64 Hadamard reference")
        got = q_out[idx].view(torch.float8_e4m3fn).double() * got_s
        want = ref_q * ref_s
        if bool(((got - want).abs() > want.abs() * 2.0**-3 + got_s * 2.0**-8).any()):
            raise AssertionError("FP8 heads differ from the float64 Hadamard reference")
        del want_q, want_s, want_w, span, ref_q, ref_s, got, want
        time_ms = Timer.cupti(launch, warmup=3, kernel_name=None)
        energy_j = Energy.perf(launch, warmup=3, per_iter_time_ms=time_ms)
    except torch.OutOfMemoryError as exc:
        raise OOMError(f"{_BACKEND} ran out of CUDA memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {type(exc).__name__}: {exc}") from exc

    seconds = time_ms / 1000.0
    flops = rows * head_dim * 7  # log2(128) butterfly add/sub per element
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=flops / seconds / 1e12,
        memory_bandwidth_gbps=logical_bytes(num_tokens, num_heads, head_dim) / seconds / 1e9,
        energy_j=float(energy_j),
    )


__all__ = ["fork_pair", "logical_bytes", "profile_hadamard_fp8_quant_weight_scale_triton"]
