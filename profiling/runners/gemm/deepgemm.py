"""DeepGEMM FP8 grouped-GEMM runner (the `deepgemm` backend of `grouped_gemm`).

L1a-only: allocate tensors, time one kernel, return metrics. Mirrors
ref/profile/gemm/grouped_gemm_deepgemm.py (FP8 m-grouped contiguous kernel) but
generalizes the ref's uniform `batch_size`-per-expert to MLSim's variable
`per_group_batches` vector — the distribution-sensitive cost driver (§2.8).

DeepGEMM needs each expert's rows aligned to the contiguous-layout alignment;
real rows carry the expert id in `m_indices`, padding rows carry -1 and the
kernel skips them. FP8: A is per-token cast, each expert's weight is per-block
cast. Requires a Hopper GPU + the pinned `deep_gemm` wheel (see CLAUDE.md /
`just sync`); absent it, raises ProfilerNotImplemented.
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics


def profile_grouped_gemm(
    n: int,
    k: int,
    dtype: DType | str,
    num_local_experts: int,
    per_group_batches: tuple[int, ...] | list[int],
) -> ComputeMetrics:
    # The deepgemm backend is FP8-only (fp8 in / bf16 out): there is no bf16/fp16
    # grouped-GEMM path on this kernel. Reject any non-FP8 compute dtype loudly so
    # a high-precision config can never silently execute as FP8 — bf16/fp16 grouped
    # GEMMs must route to the `torch` backend instead.
    dtype = DType.from_value(dtype)
    if dtype not in (DType.FP8_E4M3, DType.FP8_E5M2):
        raise ValueError(
            f"deepgemm grouped-GEMM is FP8-only but got dtype={dtype.value}; "
            "use the torch backend for bf16/fp16 grouped GEMM"
        )
    batches = [int(b) for b in per_group_batches]
    if len(batches) != int(num_local_experts):
        raise ValueError(
            f"per_group_batches has {len(batches)} entries, expected num_local_experts="
            f"{num_local_experts}"
        )
    total_real = sum(batches)
    if total_real == 0:
        raise ValueError("per_group_batches sums to 0; no tokens to profile")

    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch required for the deepgemm grouped-GEMM runner") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA required for the deepgemm grouped-GEMM runner")
    try:
        import deep_gemm
        from deep_gemm.utils import (
            align,
            ceil_div,
            get_mk_alignment_for_contiguous_layout,
            per_block_cast_to_fp8,
            per_token_cast_to_fp8,
        )
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "deep_gemm required for the deepgemm grouped-GEMM runner (see CLAUDE.md / `just sync`)"
        ) from exc

    try:
        # Contiguous layout: each expert occupies an `align`-padded row block;
        # real rows carry the expert id, padding rows carry -1 (kernel skips).
        alignment = get_mk_alignment_for_contiguous_layout()
        aligned = [align(m, alignment) if m > 0 else 0 for m in batches]
        total_tokens = sum(aligned)

        a_bf16 = torch.randn(total_tokens, k, device="cuda", dtype=torch.bfloat16)
        b_bf16 = torch.randn(num_local_experts, n, k, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(total_tokens, n, device="cuda", dtype=torch.bfloat16)

        # A: per-token FP8 (use_ue8m0=False for SM90/Hopper). B: per-block FP8 per expert.
        a_fp8 = per_token_cast_to_fp8(a_bf16, use_ue8m0=False)
        b_data = torch.empty(num_local_experts, n, k, device="cuda", dtype=torch.float8_e4m3fn)
        b_scale = torch.empty(
            num_local_experts,
            ceil_div(n, 128),
            ceil_div(k, 128),
            device="cuda",
            dtype=torch.float32,
        )
        for i in range(num_local_experts):
            b_data[i], b_scale[i] = per_block_cast_to_fp8(b_bf16[i], use_ue8m0=False)
        b_fp8 = (b_data, b_scale)

        m_indices = torch.empty(total_tokens, device="cuda", dtype=torch.int32)
        pos = 0
        for expert, (m_g, am) in enumerate(zip(batches, aligned, strict=True)):
            m_indices[pos : pos + m_g] = expert
            m_indices[pos + m_g : pos + am] = -1
            pos += am

        def kernel():
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a_fp8, b_fp8, out, m_indices)
            return out

        # DeepGEMM's first launch can transiently fail ("doesn't have storage");
        # prime before timing.
        _prime(kernel, torch)

        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)

        # FLOPs over real tokens (padding rows do no useful work); bytes over the
        # padded layout actually moved: A fp8 (1B), B fp8 (1B), out bf16 (2B).
        flops = 2 * total_real * n * k
        tflops = (flops / (time_ms / 1000.0)) / 1e12 if time_ms > 0 else 0.0
        bytes_accessed = total_tokens * k + num_local_experts * n * k + total_tokens * n * 2
        bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


def _prime(fn, torch, attempts: int = 3) -> None:
    last_error: RuntimeError | None = None
    for _ in range(attempts):
        try:
            fn()
            torch.cuda.synchronize()
            return
        except RuntimeError as exc:
            last_error = exc
            if "doesn't have storage" not in str(exc):
                raise
            torch.cuda.synchronize()
    if last_error is not None:
        raise last_error
