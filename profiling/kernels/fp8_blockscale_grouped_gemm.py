"""TRT-LLM FP8 block-scale GroupedWithOffset GEMM kernel kind.

This is intentionally separate from ``grouped_gemm``.  The generic Torch and
DeepGEMM backends use different activation-scale and routing ABIs and their
existing table has no ``num_input_tokens`` / ``experts_per_token`` recipe axes.
The direct FlashInfer wrapper keeps the production per-expert offsets, global
capacity, and recipe selection as one independently cacheable kernel contract.
"""

from __future__ import annotations

from profiling.db.args import DType, Fp8BlockscaleGroupedGemmArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "fp8_blockscale_grouped_gemm"


DOC = KernelDoc(
    title="FP8 block-scale grouped GEMM",
    summary="Multiply routed fp8 token rows by block-scaled expert weights and write bf16 output.",
    description=(
        "TensorRT-LLM's fused MoE, as vendored by FlashInfer, runs the routed "
        "experts' gate-up projection as this grouped GEMM. per_group_batches lists "
        "the local rows per expert. num_input_tokens and experts_per_token matter "
        "too: the kernel recipe is chosen from the global input-token count, and "
        "buffers are sized for num_input_tokens · experts_per_token routed rows "
        "before expert parallelism decides which rows are local."
    ),
    category="MoE",
    subcategory="Expert compute",
    formula=(
        "C_g[m_g, n] = A_g[m_g, k] · B_g[k, n], with m_g = per_group_batches[g]",
        "M = Σ m_g; E_active = number of experts with m_g > 0",
        "TFLOPS = 2·M·n·k / time",
        "GB/s = [M·k + 4·M·(k/128) + E_active·n·k + 4·E_active·(n/128)·(k/128) + 2·M·n] / time",
    ),
    default_metric="tflops",
    method=(
        f"{CUPTI_METHOD} One untimed launch and a check of the recipe's kernel "
        "names run first. Only the GroupedWithOffset GEMM kernel for the shape is "
        "counted."
    ),
    caveats=(
        "The fp8 activation and weight buffers are uninitialized and every scale is 1.",
        "GB/s counts the real local rows and the weights of experts that have "
        "rows, not the allocated capacity.",
    ),
    # The direct FlashInfer binding has no separate PyTorch reference module.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm",
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            sm_targets=frozenset({"sm_90a"}),
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
        doc=BackendDoc(
            summary=(
                "The TensorRT-LLM grouped_gemm_dispatch vendored in FlashInfer, "
                "called through a local binding without the rest of the MoE."
            ),
        ),
    )
)
