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
    context_len: int = arg(unit="tokens", doc="Index keys cached by the longest request.")
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
    cache_format: str = arg(doc="Page layout and scale format of the cached keys.")
    clean_logits: bool = arg(doc="Whether unused score positions are initialized.")


DOC = KernelDoc(
    title="Decode indexer logits",
    summary="Score decode queries against paged index keys and reduce across indexer heads.",
    description=(
        "In decode, the DSA indexer scores each request's cached keys: per "
        "head, the ReLU of the query-key dot product, weighted by head and "
        "summed, then multiplied by the key's scale. Keys are FP8 in pages "
        "that hold their keys followed by one FP32 scale per key. Two layouts "
        "are measured. With page_mapping unique_scattered and context_mode "
        "uniform, every request has context_len keys in scattered pages. With "
        "request_contiguous and max_ragged, request b has max(1, context_len − "
        "b) keys in consecutive pages padded to a 576-byte stride. max_model_len"
        " only sets the width of the score rows, and the work is counted in "
        "whole pages."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "logit[q, k] = scale[k] · Σₕ weight[q, h] · ReLU(query[q, h] · key[k])",
        "length[b] = context_len (uniform) or max(1, context_len − b) (max_ragged)",
        "C = next_n · Σ_b block_size · ⌈length[b] / block_size⌉; P = ⌈context_len / block_size⌉",
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
        "builds its scheduling metadata and runs once before the capture; for "
        "scattered pages that run's first request is checked against the torch"
        " composite. CUPTI then counts, with the L2 cache flushed before each "
        "launch, only the paged_mqa_logits kernel (sm90_fp8_paged_mqa_logits on"
        " H200, sm100_paged_mqa_logits on B200) for scattered pages and every "
        "launch of the call for request-contiguous pages."
    ),
    caveats=(
        "Uniform rows give every request the same context length and every "
        "logical page its own physical page.",
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
            sm_targets=frozenset({"sm_90a", "sm_100f", "sm_120f"}),
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
                "DeepGEMM's paged MQA logits through vLLM's deep_gemm wrapper "
                "(fp8_paged_mqa_logits, named fp8_fp4_paged_mqa_logits in the vLLM "
                "fork), reading paged FP8 keys."
            ),
            url="https://github.com/deepseek-ai/DeepGEMM",
        ),
    )
)
