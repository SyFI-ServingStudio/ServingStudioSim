"""DSA persistent decode top-k index-selection kernel kind."""

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

KIND: str = "dsa_persistent_topk_decode"


@dataclass(frozen=True)
class DsaPersistentTopkDecodeArgs(KernelArgs):
    batch_size: int = arg(unit="requests", doc="Requests decoded together.")
    context_len: int = arg(unit="tokens", doc="Longest valid logit prefix per request.")
    next_n: int = arg(unit="tokens", doc="Query rows per request in a decode step.")
    max_model_len: int = arg(unit="tokens", doc="Visible width of each logit row.")
    top_k: int = arg(unit="tokens", doc="Selected request-local positions per query row.")
    logits_row_stride: int = arg(unit="elements", doc="Physical spacing between logit rows.")
    logits_dtype: DType = arg(doc="Element type of the indexer logits.")
    index_dtype: str = arg(doc="Element type of the selected positions.")
    context_mode: str = arg(doc="Pattern of valid prefix lengths across requests.")


DOC = KernelDoc(
    title="Decode indexer top-k",
    summary="Select request-local token positions from indexer logits for each DSA decode query.",
    description=(
        "In decode, GLM-5.2's DSA indexer keeps the top_k highest-scoring "
        "positions of each query row; sparse MLA attention then reads only "
        "those. Each request has next_n rows whose valid lengths run from "
        "context_len − next_n + 1 to context_len. A row no longer than top_k "
        "returns its positions in order, padded with -1. The logits are FP32 in"
        " rows padded to logits_row_stride, and every row holds the same "
        "increasing sequence."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "length[j] = context_len − next_n + 1 + (j mod next_n)",
        "output[j] = top_k positions from logits[j, :length[j]], or natural positions "
        "when length[j] ≤ top_k",
        "logical bytes = 4·batch_size·next_n·[(2·context_len − next_n + 1)/2 + 1 + top_k]",
        "GB/s = logical bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        "torch is timed with CUDA events: five warm-up calls, then a loop of "
        "back-to-back calls, taking the median of three runs. vllm_cuda uses "
        "CUPTI kernel time with the L2 cache flushed before each call and "
        "counts every launch, the workspace memset and the persistent kernel. "
        "Operand construction and the output checks run before timing."
    ),
    caveats=(
        "The increasing test logits make the selection predictable; they do not"
        " follow a live indexer's score distribution.",
        "On B200 the check before timing verifies that long-row selections are "
        "in range and unique, but does not compare them with the reference.",
        "GB/s counts the valid FP32 logits, one int32 length and top_k int32 "
        "outputs per row, not padding, scratch space or physical transactions.",
    ),
    reference="profiling.runners.attention.dsa_persistent_topk_decode_reference",
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
            module_name="profiling.runners.attention.dsa_persistent_topk_decode",
            function_name="profile_dsa_persistent_topk_decode_torch",
        ),
        table_name=KIND,
        args_schema=DsaPersistentTopkDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary="PyTorch's vectorized topk composite, with natural-index output for short rows."
        ),
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
            module_name="profiling.runners.attention.dsa_persistent_topk_decode",
            function_name="profile_dsa_persistent_topk_decode_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=DsaPersistentTopkDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's persistent_topk: the vLLM build's op on B200, and a corrected "
                "build of the vLLM v0.23 kernel on H200."
            ),
        ),
    )
)
