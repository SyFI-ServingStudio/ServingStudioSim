"""DSA prefill MQA-logits kernel kind."""

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

KIND: str = "dsa_mqa_logits_prefill"


@dataclass(frozen=True)
class DsaMqaLogitsPrefillArgs(KernelArgs):
    num_queries: int = arg(unit="tokens", doc="Query tokens in the prefill span.")
    num_keys: int = arg(unit="tokens", doc="Cached index keys available to the final query.")
    num_sequences: int = arg(unit="requests", doc="Sequences represented by the span.")
    num_heads: int = arg(unit="heads", doc="Indexer query heads scored against each key.")
    head_dim: int = arg(unit="elements", doc="Elements in each query and key head.")
    q_dtype: DType = arg(doc="Element type of indexer queries.")
    k_dtype: DType = arg(doc="Element type of cached index keys.")
    k_scale_dtype: DType = arg(doc="Element type of per-key quantization scales.")
    weight_dtype: DType = arg(doc="Element type of per-query, per-head weights.")
    output_dtype: DType = arg(doc="Element type of the score matrix.")
    span_mode: str = arg(doc="Rule defining the valid key range of each query.")
    clean_logits: bool = arg(doc="Whether invalid score positions are initialized.")


DOC = KernelDoc(
    title="Prefill indexer logits",
    summary="Score each prefill query against causal index keys and reduce across indexer heads.",
    description=(
        "In prefill, the DSA indexer scores every visible key for each "
        "query: per head, the ReLU of the query-key dot product, weighted by "
        "head and summed, then multiplied by the key's scale. The measurement "
        "uses one sequence whose last num_queries tokens are the queries, so "
        "query i sees keys 0 through num_keys − num_queries + i. Inputs are "
        "deterministic FP8 values with FP32 scales."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "logit[q, k] = scale[k] · Σₕ weight[q, h] · ReLU(query[q, h] · key[k])",
        "W = Σ over two-query tiles ⌈(num_keys − num_queries + 1 + final query index) / 256⌉ · 256",
        "C = 2 · W",
        "TFLOPS = 2 · C · num_heads · head_dim / time",
        "B = num_queries · num_heads · head_dim + num_keys · head_dim + 4 · num_keys "
        "+ 4 · num_queries · num_heads + 8 · num_queries + 4 · C",
        "GB/s = B / time",
    ),
    default_metric="tflops",
    method=(
        "torch runs the math in FP32 as separate launches and is timed with "
        "CUDA events: five warm-up calls, then a loop of back-to-back calls, "
        "taking the median of three runs. deepgemm_fp8 runs once for setup, "
        "then CUPTI counts only the MQA-logits kernel launches, with the L2 cache "
        "flushed before each."
    ),
    caveats=(
        "Only one sequence with clean_logits = false is measured; the torch "
        "backend marks invalid positions with NaN.",
        "Both backends' TFLOPS and the logits term of GB/s use DeepGEMM's schedule "
        "of two-query, 256-key tiles, not the torch launches.",
        "GB/s counts each key once. Every two-query tile reads its key window "
        "again, but the one sequence's FP8 keys stay in L2 across tiles, so those "
        "re-reads are not HBM traffic.",
    ),
    reference="profiling.runners.attention.dsa_mqa_logits_prefill_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            kv=frozenset({DType.FP8_E4M3}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_mqa_logits_prefill",
            function_name="profile_dsa_mqa_logits_prefill_torch",
        ),
        table_name=KIND,
        args_schema=DsaMqaLogitsPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "PyTorch FP32 dot products, ReLU, head reduction and key scaling as "
                "separate launches."
            ),
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="deepgemm_fp8",
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            kv=frozenset({DType.FP8_E4M3}),
            sm_targets=frozenset({"sm_90a", "sm_100f", "sm_120f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_mqa_logits_prefill",
            function_name="profile_dsa_mqa_logits_prefill_deepgemm_fp8",
        ),
        table_name=KIND,
        args_schema=DsaMqaLogitsPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "DeepGEMM fp8_mqa_logits through vLLM's deep_gemm wrapper: FP8 scoring "
                "and head reduction in one kernel."
            ),
        ),
    )
)
