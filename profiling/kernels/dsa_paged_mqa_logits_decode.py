"""DSA paged-decode MQA-logits kernel kind."""

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

KIND: str = "dsa_paged_mqa_logits_decode"


@dataclass(frozen=True)
class DsaPagedMqaLogitsDecodeArgs(KernelArgs):
    batch_size: int = arg(unit="requests", doc="Requests decoded together.")
    context_len: int = arg(unit="tokens", doc="Cached index keys per request.")
    next_n: int = arg(unit="tokens", doc="New query tokens per request.")
    max_model_len: int = arg(unit="tokens", doc="Allocated width of each score row.")
    num_heads: int = arg(unit="heads", doc="Indexer query heads scored against each key.")
    head_dim: int = arg(unit="elements", doc="Elements in each query and key head.")
    block_size: int = arg(unit="tokens", doc="Index-key tokens per cache page.")
    q_dtype: DType = arg(doc="Element type of indexer queries.")
    cache_dtype: DType = arg(doc="Element type of cached index keys.")
    scale_dtype: DType = arg(doc="Element type of per-key quantization scales.")
    weight_dtype: DType = arg(doc="Element type of per-query, per-head weights.")
    output_dtype: DType = arg(doc="Element type of the score matrix.")
    context_mode: str = arg(doc="Distribution of context lengths across requests.")
    page_mapping: str = arg(doc="Assignment of logical pages to physical cache pages.")
    cache_format: str = arg(doc="Page-planar layout of cached keys and scales.")
    clean_logits: bool = arg(doc="Whether unused score positions are initialized.")


DOC = KernelDoc(
    title="Decode indexer logits",
    summary="Score decode queries against paged index keys and reduce across indexer heads.",
    description=(
        "In decode, GLM-5.2's DSA indexer scores each request's cached keys: "
        "per head, the ReLU of the query-key dot product, weighted by head and "
        "summed, then multiplied by the key's scale. Keys live in scattered FP8"
        " pages with a separate FP32 scale per key. Every request has the same "
        "context_len and next_n new queries; max_model_len only sets the width "
        "of the allocated score rows, and the work is counted in whole pages."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "logit[q, k] = scale[k] · Σₕ weight[q, h] · ReLU(query[q, h] · key[k])",
        "P = ⌈context_len / block_size⌉",
        "C = batch_size · next_n · P · block_size",
        "TFLOPS = 2 · C · num_heads · head_dim / time",
        "B = batch_size · next_n · num_heads · head_dim "
        "+ 4 · batch_size · next_n · num_heads + (head_dim + 4) · C "
        "+ 4 · batch_size · P + 4 · batch_size · next_n + 4 · C",
        "GB/s = B / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        "torch gathers the pages and runs the math in FP32 as separate "
        "launches, timed with CUDA events: five warm-up calls, then a loop of "
        "back-to-back calls, taking the median of three runs. deepgemm_fp8 "
        "builds its scheduling metadata and runs once before the capture; CUPTI"
        " then counts only fp8_paged_mqa_logits launches, with the L2 cache "
        "flushed before each."
    ),
    caveats=(
        "All requests have the same context length, and every logical page maps"
        " to its own physical page.",
        "Both backends' TFLOPS and GB/s use the page-rounded DeepGEMM schedule;"
        " GB/s excludes the scheduling metadata, torch intermediates and "
        "physical transactions.",
    ),
    reference="profiling.runners.attention.dsa_paged_mqa_logits_decode_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA H200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_paged_mqa_logits_decode",
            function_name="profile_dsa_paged_mqa_logits_decode_torch",
        ),
        table_name=KIND,
        args_schema=DsaPagedMqaLogitsDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary=(
                "PyTorch paged-key gather and FP32 query-key scoring as a multi-launch composite."
            )
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
            gpus=frozenset({"NVIDIA H200", "NVIDIA B200"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_paged_mqa_logits_decode",
            function_name="profile_dsa_paged_mqa_logits_decode_deepgemm_fp8",
        ),
        table_name=KIND,
        args_schema=DsaPagedMqaLogitsDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "DeepGEMM fp8_paged_mqa_logits through vLLM's deep_gemm wrapper, "
                "reading paged FP8 keys."
            ),
        ),
    )
)
