"""TRT-LLM FP8 block-scale GroupedWithOffset GEMM kernel kind.

This is intentionally separate from ``grouped_gemm``.  The generic Torch and
DeepGEMM backends use different activation-scale and routing ABIs and their
existing table has no ``num_input_tokens`` / ``experts_per_token`` recipe axes.
The direct FlashInfer wrapper keeps the production per-expert offsets, global
capacity, and recipe selection as one independently cacheable kernel contract.
"""

from __future__ import annotations

from profiling.db.args import DType, Fp8BlockscaleGroupedGemmArgs
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "fp8_blockscale_grouped_gemm"


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm",
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA H100", "NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.gemm.flashinfer_trtllm_blockscale",
            function_name=("profile_fp8_blockscale_grouped_gemm_flashinfer_trtllm"),
        ),
        table_name=KIND,
        args_schema=Fp8BlockscaleGroupedGemmArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="flashinfer_pip_env",
    )
)
