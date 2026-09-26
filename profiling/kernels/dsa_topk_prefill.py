"""DSA prefill top-k index-selection kernel kind."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND: str = "dsa_topk_prefill"


@dataclass(frozen=True)
class DsaTopkPrefillArgs(KernelArgs):
    num_queries: int = arg(unit="tokens", doc="Query rows whose index scores are selected.")
    num_keys: int = arg(unit="tokens", doc="Index-key columns in each score row.")
    num_sequences: int = arg(unit="requests", doc="Sequences represented by the score matrix.")
    top_k: int = arg(unit="tokens", doc="Selected key positions written per query.")
    logits_row_stride: int = arg(unit="elements", doc="Allocated elements between score rows.")
    logits_dtype: DType = arg(doc="Element type of the input index scores.")
    index_dtype: str = arg(doc="Element type of selected key positions.")
    span_mode: str = arg(doc="Rule defining the valid key range of each query.")


DOC = KernelDoc(
    title="Prefill indexer top-k",
    summary="Choose the highest-scoring index-key positions for each prefill query.",
    description=(
        "After the prefill scores, GLM-5.2's DSA indexer keeps the top_k "
        "highest-scoring keys for each query; sparse attention then reads only "
        "those. The measurement uses one sequence whose last num_queries tokens"
        " are the queries, so query i has num_keys − num_queries + i + 1 valid "
        "scores. Scores are FP32 in rows padded to logits_row_stride."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "indices[q] = top_k(valid logits[q])",
        "S = num_queries · (num_keys − num_queries + 1 + num_keys) / 2",
        "GB/s = (4 · S + 8 · num_queries + 4 · num_queries · top_k) / time",
        "SGLang GB/s = (4 · S + 8 · num_queries + 8 · num_queries · top_k) / time",
    ),
    default_metric="time_ms",
    method=(
        "torch masks and selects as separate launches, timed with CUDA events: "
        "five warm-up calls, then a loop of back-to-back calls, taking the "
        "median of three runs. vllm_cuda and sglang_cuda run once before the "
        "capture; CUPTI then counts only topKPerRowPrefill or "
        "topk_transform_prefill_kernel launches, with the L2 cache flushed "
        "before each."
    ),
    caveats=(
        "Only one sequence with top_k = 2048 is measured, on scores that are "
        "deterministic and free of ties.",
        "GB/s counts valid scores, row bounds and output indices, not padding "
        "or torch intermediates; SGLang also counts its page-table output.",
    ),
    reference="profiling.runners.attention.dsa_topk_prefill_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(
            compute=frozenset({DType.FP32}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_topk_prefill",
            function_name="profile_dsa_topk_prefill_torch",
        ),
        table_name=KIND,
        args_schema=DsaTopkPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(summary="PyTorch masking and topk selection as separate launches."),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.FP32}),
            gpus=frozenset({"NVIDIA H200", "NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_topk_prefill",
            function_name="profile_dsa_topk_prefill_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=DsaTopkPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary="vLLM topKPerRowPrefill selects positions from each causal score row.",
            url="https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/sampler.cu",
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.FP32}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_topk_prefill",
            function_name="profile_dsa_topk_prefill_sglang_cuda",
        ),
        table_name=KIND,
        args_schema=DsaTopkPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_env",
        doc=BackendDoc(
            summary=(
                "SGLang fast_topk_transform_fused: top-k selection that also maps "
                "positions through the page table."
            ),
            url="https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/aot/csrc/elementwise/topk.cu",
        ),
    )
)
