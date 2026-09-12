"""DeepGEMM FP8 GEMM runners — the `deepgemm` backend of both `single_gemm`
(dense) and `grouped_gemm` (MoE experts).

L1a-only: allocate tensors, time one kernel, return metrics. Both mirror
ref/profile/gemm/grouped_gemm_deepgemm.py: `profile_single_gemm` ports the ref's
dense `fp8_gemm_nt`; `profile_grouped_gemm` ports the m-grouped contiguous kernel
but generalizes the ref's uniform `batch_size`-per-expert to ServingStudioSim's variable
`per_group_batches` vector — the distribution-sensitive cost driver (§2.8).

DeepGEMM grouped needs each expert's rows aligned to the contiguous-layout
alignment; real rows carry the expert id in `m_indices`, padding rows carry -1
and the kernel skips them. FP8: A is per-token cast, each weight is per-block
cast. Both runners require a Hopper GPU + the pinned `deep_gemm` wheel (see
CLAUDE.md / `just sync`); absent it, they raise ProfilerNotImplemented.
"""

from __future__ import annotations

from profiling.db.args import DType
from profiling.profilers.energy import Energy
from profiling.profilers.timer import Timer
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.metrics import ComputeMetrics


def _grouped_gemm_logical_bytes(
    n: int,
    k: int,
    batches: list[int],
    aligned: list[int],
) -> int:
    """Logical bytes touched by the FP8 grouped launch.

    Activation/output traffic follows the aligned contiguous layout. Weight
    traffic includes one FP8 ``n x k`` matrix per active expert; experts with no
    real rows have no id in ``m_indices`` and their allocated weights are skipped.
    """
    total_tokens = sum(aligned)
    active_experts = sum(batch > 0 for batch in batches)
    return total_tokens * k + active_experts * n * k + total_tokens * n * 2


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

        # FLOPs cover real tokens; activation/output bytes cover the padded
        # layout, while weights include only experts with real rows.
        flops = 2 * total_real * n * k
        tflops = (flops / (time_ms / 1000.0)) / 1e12 if time_ms > 0 else 0.0
        bytes_accessed = _grouped_gemm_logical_bytes(n, k, batches, aligned)
        bandwidth_gbps = (bytes_accessed / (time_ms / 1000.0)) / 1e9 if time_ms > 0 else 0.0
        return ComputeMetrics(
            time_ms=float(time_ms),
            tflops=float(tflops),
            memory_bandwidth_gbps=float(bandwidth_gbps),
            energy_j=float(energy_j),
        )
    except RuntimeError as exc:
        raise KernelLaunchFailed(str(exc)) from exc


def profile_single_gemm(
    m: int,
    n: int,
    k: int,
    dtype: DType | str,
) -> ComputeMetrics:
    # Dense FP8 GEMM: (M×K) @ (K×N)^T -> (M×N), fp8 in / bf16 out. Same FP8-only
    # policy as the grouped path — bf16/fp16 dense GEMMs must route to the `torch`
    # backend, so reject any non-FP8 compute dtype loudly. Mirrors
    # ref/profile/gemm/grouped_gemm_deepgemm.py::profile_deepgemm_dense.
    dtype = DType.from_value(dtype)
    if dtype not in (DType.FP8_E4M3, DType.FP8_E5M2):
        raise ValueError(
            f"deepgemm single-GEMM is FP8-only but got dtype={dtype.value}; "
            "use the torch backend for bf16/fp16 dense GEMM"
        )

    try:
        import torch
    except ImportError as exc:
        raise ProfilerNotImplemented("torch required for the deepgemm single-GEMM runner") from exc
    if not torch.cuda.is_available():
        raise ProfilerNotImplemented("CUDA required for the deepgemm single-GEMM runner")
    try:
        import deep_gemm
        from deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8
    except ImportError as exc:
        raise ProfilerNotImplemented(
            "deep_gemm required for the deepgemm single-GEMM runner (see CLAUDE.md / `just sync`)"
        ) from exc

    try:
        # NT layout: A is (M, K) row-major, B is (N, K) row-major (i.e. stored
        # transposed), out is (M, N) bf16. A: per-token FP8, B: per-block FP8
        # (use_ue8m0=False for SM90/Hopper).
        a_bf16 = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        b_bf16 = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

        a_fp8 = per_token_cast_to_fp8(a_bf16, use_ue8m0=False)
        b_fp8 = per_block_cast_to_fp8(b_bf16, use_ue8m0=False)

        def kernel():
            deep_gemm.fp8_gemm_nt(a_fp8, b_fp8, out)
            return out

        # DeepGEMM's first launch can transiently fail ("doesn't have storage");
        # prime before timing.
        _prime(kernel, torch)

        time_ms = Timer.cupti(kernel)
        energy_j = Energy.perf(kernel, warmup=5, per_iter_time_ms=time_ms)

        # Bytes actually moved: A fp8 (1B), B fp8 (1B), out bf16 (2B).
        flops = 2 * m * n * k
        tflops = (flops / (time_ms / 1000.0)) / 1e12 if time_ms > 0 else 0.0
        bytes_accessed = m * k + n * k + m * n * 2
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
