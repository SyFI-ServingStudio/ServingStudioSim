"""Torch-on-ROCm runner for the GLM-5.3-Flash KDA chunked prefill on MI300X.

The measured boundary is one call of the vendored *AMD* KDA entry point
``vllm.models.glm5next.amd.ops.third_party.kda.chunk_kda_with_fused_gate`` -- the
one ``Glm5NextLinearAttention._forward`` dispatches to on ROCm (the ``is_rocm()``
branch in ``vllm/models/glm5next/common/kda.py``), made with exactly the keyword
set and operand layout the NVIDIA ``vllm_triton`` backend uses. The callable's
signature is identical to the NVIDIA one (verified on the pinned image,
v0.3.1.dev190); only the kernels behind it change. The NVIDIA chunk path reaches
``torch.ops._flashkda_C`` (an NVIDIA-only C extension); the AMD module's
``chunk_kda_with_fused_gate`` is the ROCm counterpart built on FLA Triton
kernels, confirmed to import and run on CDNA3 (gfx942) with no ``_flashkda_C``
dependency. This is the prefill sibling of ``kda_recurrent_decode_torch_rocm``.

Timing is kernel-only via rocprofv3 (``measure_registered_via_rocprofv3``), the
ROCm counterpart of the CUPTI path the NVIDIA backend uses. The NVIDIA backend
sums *every* dispatch of the call (the q/k/v ``.contiguous()`` copies plus the
FLA chunk-kernel chain) with ``Timer.cupti(..., kernel_name=None)``. To match
that on ROCm, the call is captured under a whole-process rocprofv3 kernel trace
and the per-call time is the sum of that call's dispatches, averaged over the
timed reps.

Autotune / fold. Unlike ``fused_recurrent_kda``, the FLA chunk kernels this call
reaches are ``@autotune``d: the *first* call in a process benchmarks candidate
configs, emitting a variable burst of extra dispatches, so the constant-prefix
fold the decode runner uses (``fold_per_launch``, which recovers a single ``D``
by division over the whole stream) would mis-count. Instead this runner measures
with ``dispatches_per_launch=D``: the Triton autotuner caches its winner
in-process after the first warm-up call, so every call from warm-up #2 onward
launches the SAME constant ``D`` dispatches, and the harness folds the *trailing*
``rep`` launches (the last ``rep*D`` dispatches) into per-call sums. Taking the
tail makes the measurement robust to however large the leading autotune burst
is. ``D`` is the per-call dispatch count measured directly on the MI300X image
(see ``_DISPATCHES_PER_LAUNCH``); the harness asserts the captured stream has at
least ``rep*D`` dispatches, so a wrong ``D`` or an un-warmed autotuner fails loud
rather than returning a fabricated time. Operands are built on the host and moved
to the device with ``.to()`` (a host-to-device copy, not a kernel), so tensor
setup emits no kernel dispatches and the per-call count stays exactly ``D``.

FLOPs and bytes are the same semantic logical counts the NVIDIA backend reports,
imported from the ``vllm_triton`` runner so the two never drift.
"""

from __future__ import annotations

import importlib
from typing import Any

from profiling.db.args import DType
from profiling.runners.attention.kda_chunk_prefill_vllm_triton import (
    LOWER_BOUND,
    RMSE_RATIO_TOL,
    KdaChunkPrefillShape,
    _Operands,
    check_correctness,
    guard_shape,
    invoke,
    logical_bytes,
    semantic_flops,
    sequence_boundaries,
    sequence_lengths,
    validate_args,
)
from profiling.runners.attention.kda_recurrent_decode_torch_rocm import (
    _device_arch,
    _device_name,
    _is_mi300_series,
)
from profiling.runners.exceptions import (
    KernelLaunchFailed,
    OOMError,
    ProfilerNotImplemented,
)
from profiling.runners.metrics import ComputeMetrics

_BACKEND = "kda_chunk_prefill:torch_rocm"
_KIND = "kda_chunk_prefill"
# The ROCm (is_rocm()) branch of common/kda.py imports the KDA entry points from
# here; the NVIDIA backend uses the parallel ``nvidia`` subtree.
_MODULE = "vllm.models.glm5next.amd.ops.third_party.kda"
_CALLABLE = "chunk_kda_with_fused_gate"
_GPU_SKU_TOKEN = "MI300X"
_WARMUP = 5
_REP = 20
# Per-call GPU-dispatch count of chunk_kda_with_fused_gate on the AMD path, once
# the FLA Triton autotuners have cached their winners (every call from warm-up #2
# onward). Measured directly on the pinned MI300X image; see the module docstring
# for why the trailing fold uses it instead of the decode runner's division fold.
# Filled from the MI300X inspection capture.
_DISPATCHES_PER_LAUNCH = 15


def build_kda_chunk_prefill_kernel(
    num_tokens: int,
    max_sequence_length: int,
    num_decode_sequences: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> dict[str, Any]:
    """Build the timed callable + fixed operands for the AMD KDA prefill call.

    Shared by the runner's correctness check and the rocprofv3 launch driver
    (``profiling.profilers.rocprof_run``) so both replay the identical call from
    the same fixed seed. Operands are built on the host (CPU) and moved to the
    device, so setup launches no GPU kernels -- the only dispatches the trace
    sees are the call's own (its q/k/v ``.contiguous()`` copies and the FLA chunk
    chain). q/k/v stay strided views of the merged projection, as the production
    path hands them over, so the call's copies are real and counted.
    """
    shape = validate_args(
        num_tokens, max_sequence_length, num_decode_sequences, num_heads, head_dim, dtype
    )
    import torch

    callable_ = _load_amd_callable(torch)
    device = torch.device("cuda", torch.cuda.current_device())
    operands = _build_operands_host_to_device(torch, shape, device=device)

    def kernel() -> Any:
        return invoke(callable_, operands)

    return {
        "torch": torch,
        "kernel": kernel,
        "callable": callable_,
        "operands": operands,
        "shape": shape,
        "warmup": _WARMUP,
        "rep": _REP,
    }


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


def _build_operands_host_to_device(
    torch: Any, shape: KdaChunkPrefillShape, *, device: Any
) -> _Operands:
    """Mirror the NVIDIA runner's operands, built on host then copied to device.

    Same fixed seed (42) and same layout as ``kda_chunk_prefill_vllm_triton``;
    the only difference is the host->device move, which keeps tensor setup out of
    the kernel trace (so the per-call dispatch count is exactly the call's own).
    q/k/v are strided views of the device-resident contiguous qkv, so the call's
    three ``.contiguous()`` copies are real, as in production.
    """
    tokens, heads, dim = shape.num_tokens, shape.num_heads, shape.head_dim
    lengths = sequence_lengths(shape)
    boundaries = sequence_boundaries(shape)
    gen = torch.Generator(device="cpu").manual_seed(42)

    def normal(*size: int, dtype: Any) -> Any:
        return torch.randn(*size, generator=gen, dtype=torch.float32).to(dtype)

    projection = heads * dim
    qkv = normal(tokens, 3 * projection, dtype=torch.bfloat16).to(device)
    q, k, v = (part.reshape(1, tokens, heads, dim) for part in qkv.split(projection, dim=-1))
    raw_g = normal(1, tokens, heads, dim, dtype=torch.bfloat16).to(device)
    beta = torch.sigmoid(normal(tokens, heads, dtype=torch.float32)).unsqueeze(0).to(device)
    # Mamba/Kimi-style A init: exp(A_log) uniform in [1, 16].
    a_log = (
        torch.empty(1, 1, heads, 1, dtype=torch.float32)
        .uniform_(1.0, 16.0, generator=gen)
        .log()
        .to(device)
    )
    dt_bias = (normal(projection, dtype=torch.float32) * 0.1).to(device)
    initial_state = normal(len(lengths), heads, dim, dim, dtype=torch.float32)
    # gather_initial_states zero-fills rows without prior state (fresh prefills).
    initial_state[shape.num_decode_sequences :] = 0
    initial_state = initial_state.to(device)
    cu_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=device)
    return _Operands(
        qkv=qkv,
        q=q,
        k=k,
        v=v,
        raw_g=raw_g,
        beta=beta,
        a_log=a_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        boundaries=boundaries,
    )


def profile_kda_chunk_prefill_torch_rocm(
    num_tokens: int,
    max_sequence_length: int,
    num_decode_sequences: int,
    num_heads: int,
    head_dim: int,
    dtype: DType | str,
) -> ComputeMetrics:
    """Profile one GLM-5.3-Flash KDA chunked-prefill call on an MI300X."""
    shape = validate_args(
        num_tokens, max_sequence_length, num_decode_sequences, num_heads, head_dim, dtype
    )
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
        from profiling.profilers.rocprof_kernel_profiler import (
            measure_registered_via_rocprofv3,
        )

        built = build_kda_chunk_prefill_kernel(
            num_tokens, max_sequence_length, num_decode_sequences, num_heads, head_dim, dtype
        )
        # Correctness: a bounded witness (few decodes, multi-chunk prefill, tail)
        # against the naive oracle, restoring nothing (the call reads no state it
        # writes). Uses the shared check from the NVIDIA runner.
        guard = guard_shape(shape)
        guard_built = build_kda_chunk_prefill_kernel(
            guard.num_tokens,
            guard.max_sequence_length,
            guard.num_decode_sequences,
            guard.num_heads,
            guard.head_dim,
            dtype,
        )
        check_correctness(torch, guard_built["callable"], guard_built["operands"], guard)

        # Whole-process rocprofv3 capture of the identical call. The FLA chunk
        # kernels autotune on the first warm-up call (a variable dispatch burst);
        # the autotuner then caches its winner in-process, so every later call
        # launches a constant D dispatches. The harness folds the TRAILING rep
        # launches (last rep*D dispatches), which skips the autotune burst and
        # matches the NVIDIA CUPTI path's kernel_name=None per-launch sum.
        time_ms = measure_registered_via_rocprofv3(
            kind=_KIND,
            backend="torch_rocm",
            spec={
                "num_tokens": shape.num_tokens,
                "max_sequence_length": shape.max_sequence_length,
                "num_decode_sequences": shape.num_decode_sequences,
                "num_heads": shape.num_heads,
                "head_dim": shape.head_dim,
                "dtype": shape.dtype.value,
            },
            kernel_name_contains=None,
            warmup=_WARMUP,
            rep=_REP,
            dispatches_per_launch=_DISPATCHES_PER_LAUNCH,
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
    "build_kda_chunk_prefill_kernel",
    "profile_kda_chunk_prefill_torch_rocm",
    "LOWER_BOUND",
    "RMSE_RATIO_TOL",
]
