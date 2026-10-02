"""Torch-on-ROCm runner for the GLM-5.3-Flash KDA recurrent decode on MI300X.

The measured boundary is one call of the vendored *AMD* KDA entry point
``vllm.models.glm5next.amd.ops.third_party.kda.fused_recurrent_kda`` -- the one
``Glm5NextLinearAttention._forward`` dispatches to on ROCm (the ``is_rocm()``
branch in ``vllm/models/glm5next/common/kda.py``), made with exactly the
keyword set and operand layout the NVIDIA ``vllm_triton`` backend uses. The
signature is identical to the NVIDIA callable (verified on the pinned image,
v0.3.1.dev190); only the kernels behind it change (AMD Triton instead of the
NVIDIA fork's). This is the ROCm counterpart of
``kda_recurrent_decode_vllm_triton`` and the second kernel to prove the ROCm
profiling path end to end after ``rms_norm`` -- the primary KDA linear-attention
compute kernel, covering the decode step of the 34 KDA layers.

Timing is kernel-only via rocprofv3 (``measure_registered_via_rocprofv3``), the
ROCm counterpart of the CUPTI path the NVIDIA backend uses. The NVIDIA backend
sums *every* dispatch of the call (the four input ``.contiguous()`` copies plus
the recurrent kernel) with ``Timer.cupti(..., kernel_name=None)``. To match that
on ROCm, the call is captured under a whole-process rocprofv3 kernel trace and
the per-call time is the sum of that call's dispatches, averaged over the timed
reps (``fold_per_launch=True``). To make the dispatch stream contain *only* the
call's kernels -- so the per-call dispatch count is constant and the fold is
exact -- the operands are built on the host and moved to the device with
``.to()`` (a host-to-device copy, not a kernel), so tensor setup contributes no
kernel dispatches to the ``--kernel-trace``. The recurrent kernel on this path
is not ``@autotune``d, so each call launches a fixed number of dispatches.

FLOPs and bytes are the same semantic logical counts the NVIDIA backend reports,
imported from the ``vllm_triton`` runner so the two never drift.
"""

from __future__ import annotations

import importlib
from typing import Any

from profiling.db.args import DType
from profiling.runners.attention.kda_recurrent_decode_vllm_triton import (
    LOWER_BOUND,
    OUTPUT_RMSE_RATIO_TOL,
    STATE_RMSE_RATIO_TOL,
    KdaRecurrentDecodeShape,
    logical_bytes,
    semantic_flops,
    state_slots,
    validate_args,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "kda_recurrent_decode:torch_rocm"
_KIND = "kda_recurrent_decode"
# The ROCm (is_rocm()) branch of common/kda.py imports the KDA entry points from
# here; the NVIDIA backend uses the parallel ``nvidia`` subtree.
_MODULE = "vllm.models.glm5next.amd.ops.third_party.kda"
_CALLABLE = "fused_recurrent_kda"
# torch's ROCm build reports MI300X as some "...MI300X..." marketing string
# ("AMD Instinct MI300X" on the pinned image); match the SKU token rather than
# an exact name so an OAM/VF suffix does not spuriously reject the verified GPU.
_GPU_SKU_TOKEN = "MI300X"
_WARMUP = 5
_REP = 20


def build_kda_recurrent_decode_kernel(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> dict[str, Any]:
    """Build the timed callable + fixed operands for the AMD KDA decode call.

    Shared by the runner's correctness check and the rocprofv3 launch driver
    (``profiling.profilers.rocprof_run``) so both replay the identical call from
    the same fixed seed. Operands are built on the host (CPU) and moved to the
    device, so setup launches no GPU kernels -- the only dispatches the trace
    sees are the call's own (its four ``.contiguous()`` copies and the recurrent
    kernel). q/k/v stay strided views of the merged projection, as the production
    path hands them over, so the call's copies are real and counted.
    """
    shape = validate_args(batch_size, num_heads, head_dim, dtype)
    import torch

    callable_ = _load_amd_callable(torch)
    device = torch.device("cuda", torch.cuda.current_device())
    operands = _build_operands_host_to_device(torch, shape, device=device)

    def kernel() -> Any:
        return _invoke(callable_, operands)

    return {
        "torch": torch,
        "kernel": kernel,
        "callable": callable_,
        "operands": operands,
        "shape": shape,
        "warmup": _WARMUP,
        "rep": _REP,
    }


def _device_name(torch: Any) -> str:
    try:
        return str(torch.cuda.get_device_name(torch.cuda.current_device()))
    except Exception:
        return ""


def _device_arch(torch: Any) -> str:
    # torch's ROCm build exposes the CDNA ISA as gcnArchName (e.g. "gfx942" for
    # the MI300 series). Under ROCR_VISIBLE_DEVICES pinning get_device_name can
    # return an empty string, so the arch is the reliable identity fallback.
    try:
        return str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", ""))
    except Exception:
        return ""


def _is_mi300_series(torch: Any) -> bool:
    import os

    name = _device_name(torch)
    if _GPU_SKU_TOKEN in name:
        return True
    # On this image torch masks the device name under ROCR_VISIBLE_DEVICES; the
    # CDNA3 ISA (gfx942) is the reliable fallback, and the caller may also pass
    # the authoritative rocminfo Marketing Name through VIBESIM_OBSERVED_GPU.
    if _device_arch(torch).split(":")[0] == "gfx942":
        return True
    return _GPU_SKU_TOKEN in os.environ.get("VIBESIM_OBSERVED_GPU", "")


def _load_amd_callable(torch: Any) -> Any:
    if getattr(torch.version, "hip", None) is None:
        raise ProfilerNotImplemented(f"{_BACKEND} requires a ROCm (HIP) torch build")
    try:
        module = importlib.import_module(_MODULE)
    except Exception as exc:  # pragma: no cover - requires the vLLM-ROCm image
        raise ProfilerNotImplemented(
            f"{_BACKEND} requires the vllm_rocm_env glm5next plugin ({_MODULE})"
        ) from exc
    fn = getattr(module, _CALLABLE, None)
    if not callable(fn):
        raise ProfilerNotImplemented(f"{_BACKEND} requires {_MODULE}.{_CALLABLE}")
    return fn


def _build_operands_host_to_device(torch: Any, shape: KdaRecurrentDecodeShape, *, device: Any):
    """Mirror the NVIDIA runner's operands, built on host then copied to device.

    Same fixed seed (42) and same layout as ``kda_recurrent_decode_vllm_triton``;
    the only difference is the host->device move, which keeps tensor setup out of
    the kernel trace.
    """
    batch, heads, dim, projection = (
        shape.batch_size,
        shape.num_heads,
        shape.head_dim,
        shape.projection,
    )
    gen = torch.Generator(device="cpu").manual_seed(42)

    def normal(*size: int, dtype: Any) -> Any:
        return torch.randn(*size, generator=gen, dtype=torch.float32).to(dtype)

    projected = normal(batch, shape.projected_width, dtype=torch.bfloat16).to(device)
    qkv = projected[:, : 3 * projection]
    q, k, v = (part.reshape(1, batch, heads, dim) for part in qkv.split(projection, dim=-1))
    beta = projected[:, 3 * projection : 3 * projection + heads].unsqueeze(0)
    g = normal(batch, projection, dtype=torch.bfloat16).reshape(1, batch, heads, dim).to(device)
    a_log = (
        torch.empty(1, 1, heads, 1, dtype=torch.float32)
        .uniform_(1.0, 16.0, generator=gen)
        .log()
        .to(device)
    )
    dt_bias = (normal(projection, dtype=torch.float32) * 0.1).to(device)
    state_pool = normal(batch + 1, heads, dim, dim, dtype=torch.float32).to(device)
    state_indices = torch.tensor(state_slots(batch), dtype=torch.int32, device=device)
    cu_seqlens = torch.arange(batch + 1, dtype=torch.int32, device=device)
    out = torch.empty(1, batch, heads, dim, dtype=torch.bfloat16, device=device)
    return {
        "projected": projected,
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "a_log": a_log,
        "dt_bias": dt_bias,
        "state_pool": state_pool,
        "state_indices": state_indices,
        "cu_seqlens": cu_seqlens,
        "out": out,
    }


def _invoke(callable_: Any, op: dict[str, Any]) -> Any:
    """The exact keyword set common/kda.py passes on the plain-decode branch."""
    return callable_(
        q=op["q"],
        k=op["k"],
        v=op["v"],
        g=op["g"],
        beta=op["beta"],
        initial_state=op["state_pool"],
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=op["cu_seqlens"],
        ssm_state_indices=op["state_indices"],
        out=op["out"],
        sigmoid_beta=True,
        a_log=op["a_log"],
        g_bias=op["dt_bias"],
        compute_gate=True,
        lower_bound=LOWER_BOUND,
    )


def _check_correctness(torch: Any, built: dict[str, Any]) -> tuple[float, float]:
    """One call vs the naive oracle; restore the pool; return (o, state) RMSE ratios."""
    from profiling.runners.attention.kda_chunk_prefill_reference import rmse_ratio
    from profiling.runners.attention.kda_recurrent_decode_reference import (
        kda_recurrent_decode_reference,
    )

    op = built["operands"]
    shape = built["shape"]
    pool_snapshot = op["state_pool"].clone()
    expected_o, expected_pool = kda_recurrent_decode_reference(
        op["q"].squeeze(0),
        op["k"].squeeze(0),
        op["v"].squeeze(0),
        op["g"].squeeze(0),
        op["beta"].squeeze(0),
        op["a_log"],
        op["dt_bias"],
        pool_snapshot,
        op["state_indices"],
        lower_bound=LOWER_BOUND,
    )
    try:
        output, final_state = built["kernel"]()
        torch.cuda.synchronize()
        if output is not op["out"] or final_state is not op["state_pool"]:
            raise AssertionError("fused_recurrent_kda must write out/state in place")
        slots = op["state_indices"].long()
        actual_state = op["state_pool"][slots]
        if not (torch.isfinite(output).all() and torch.isfinite(actual_state).all()):
            raise AssertionError("non-finite KDA output or state")
        o_err = rmse_ratio(expected_o, output.squeeze(0))
        state_err = rmse_ratio(expected_pool[slots], actual_state)
        if not (o_err < OUTPUT_RMSE_RATIO_TOL and state_err < STATE_RMSE_RATIO_TOL):
            raise AssertionError(
                f"AMD KDA decode mismatch vs oracle: o {o_err:.2e} "
                f"(tol {OUTPUT_RMSE_RATIO_TOL:.0e}), state {state_err:.2e} "
                f"(tol {STATE_RMSE_RATIO_TOL:.0e})"
            )
    finally:
        op["state_pool"].copy_(pool_snapshot)
    return o_err, state_err


def profile_kda_recurrent_decode_torch_rocm(
    batch_size: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile one GLM-5.3-Flash KDA recurrent-decode call on an MI300X."""
    shape = validate_args(batch_size, num_heads, head_dim, dtype)
    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented(f"PyTorch is required for {_BACKEND}") from exc
    try:
        if not torch.cuda.is_available():
            raise ProfilerNotImplemented(f"a ROCm device is required for {_BACKEND}")
        if getattr(torch.version, "hip", None) is None:
            raise ProfilerNotImplemented(f"{_BACKEND} requires a ROCm (HIP) torch build")
        if not _is_mi300_series(torch):
            raise ProfilerNotImplemented(
                f"{_BACKEND} is verified only on {_GPU_SKU_TOKEN} (CDNA3 gfx942), got "
                f"name={_device_name(torch)!r} arch={_device_arch(torch)!r}"
            )
        from profiling.profilers.rocprof_kernel_profiler import measure_registered_via_rocprofv3

        built = build_kda_recurrent_decode_kernel(batch_size, num_heads, head_dim, dtype)
        _check_correctness(torch, built)

        # Whole-process rocprofv3 capture of the identical call, replayed from the
        # same spec under the tracer. Every dispatch of the call is counted and
        # summed per call (fold_per_launch), matching the NVIDIA CUPTI path's
        # kernel_name=None sum; the warmup calls are dropped.
        time_ms = measure_registered_via_rocprofv3(
            kind=_KIND,
            backend="torch_rocm",
            spec={
                "batch_size": shape.batch_size,
                "num_heads": shape.num_heads,
                "head_dim": shape.head_dim,
                "dtype": shape.dtype.value,
            },
            kernel_name_contains=None,
            warmup=_WARMUP,
            rep=_REP,
            fold_per_launch=True,
        )
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        raise OOMError(f"{_BACKEND} ran out of device memory") from exc
    except ProfilerNotImplemented:
        raise
    except Exception as exc:
        raise KernelLaunchFailed(f"{_BACKEND} failed: {exc}") from exc

    elapsed = time_ms / 1000.0
    return ComputeMetrics(
        time_ms=float(time_ms),
        tflops=semantic_flops(shape) / elapsed / 1e12 if elapsed > 0 else 0.0,
        memory_bandwidth_gbps=logical_bytes(shape) / elapsed / 1e9 if elapsed > 0 else 0.0,
        energy_j=0.0,
    )


__all__ = [
    "build_kda_recurrent_decode_kernel",
    "profile_kda_recurrent_decode_torch_rocm",
]
