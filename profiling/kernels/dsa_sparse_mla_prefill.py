"""Varlen sparse-MLA prefill over one request batch."""

from __future__ import annotations

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.doc import CUPTI_METHOD, BackendDoc, KernelDoc, arg
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND = "dsa_sparse_mla_prefill"


@dataclass(frozen=True)
class DsaSparseMlaPrefillArgs(KernelArgs):
    query_context_pairs: tuple[tuple[int, int], ...] = arg(
        doc="Query-token count and ending context length for each prefill request."
    )
    num_heads: int = arg(unit="heads", doc="Query heads on this GPU.")
    num_kv_heads: int = arg(unit="heads", doc="KV heads shared by the query heads.")
    selected_k: int = arg(unit="tokens", doc="Selected index slots per query.")
    latent_dim: int = arg(unit="elements", doc="Compressed KV latent width.")
    rope_dim: int = arg(unit="elements", doc="Rotary-position width appended to the latent.")
    value_dim: int = arg(unit="elements", doc="Value and output width per head.")
    softmax_scale: float = arg(unit="multiplier", doc="Multiplier applied to attention scores.")
    q_dtype: DType = arg(doc="Query element type.")
    cache_dtype: DType = arg(doc="Compressed KV cache element type.")
    index_dtype: str = arg(doc="Element type of selected indices.")
    output_dtype: DType = arg(doc="Attention output element type.")
    index_distribution: str = arg(doc="Pattern used to construct selected indices.")
    cache_layout: str = arg(doc="Layout of compressed KV values in the cache.")


DOC = KernelDoc(
    title="Sparse MLA prefill",
    summary="Attend to selected compressed KV tokens for a batch of prefill queries.",
    description=(
        "Sparse MLA attention for a prefill batch on B200, where "
        "requests differ in new-query count and context length. Each "
        "query_context_pairs entry gives one request's query count and final "
        "context length; its query rows are the last positions of that context,"
        " and each attends to up to selected_k causal positions. All rows go to"
        " FlashInfer in one call, each as a one-token decode row."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "valid_count[position] = min(position + 1, selected_k), per request and "
        "zero-based position",
        "O = softmax(softmax_scale · q · K_selectedᵀ) · V_selected",
        "Q = Σ request query counts; H = num_heads; K = selected_k; D = latent_dim + rope_dim",
        "TFLOPS = 2·Q·H·K·(D + value_dim) / time",
        "logical bytes = Q·H·D + 4·Q·K + Σvalid_counts·D + 2·Q·H·value_dim + 8·Q·H",
        "GB/s = logical bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "Only the fmhaSm100 launch is counted. Query, cache, indices and "
        "workspace are built first, and one call checks the output before "
        "timing."
    ),
    caveats=(
        "Selected positions follow index_distribution within each row's causal "
        "range; they are not recorded indexer selections.",
        "The output check covers shape and finiteness only; there is no "
        "numerical comparison with a reference.",
        "TFLOPS counts all selected_k slots even when fewer are valid; GB/s "
        "includes 8 bytes per query head for outputs this call does not return.",
    ),
    # This kind has no separate PyTorch reference implementation.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm_fp8",
        # TRTLLM-GEN sparse MLA (fmhaSm100) runs only on the SM10x family.
        supports=BackendSupport(
            compute=frozenset({DType.FP8_E4M3}),
            kv=frozenset({DType.FP8_E4M3}),
            sm_targets=frozenset({"sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_sparse_mla_prefill",
            function_name="profile_dsa_sparse_mla_prefill_flashinfer_trtllm_fp8",
        ),
        table_name=KIND,
        args_schema=DsaSparseMlaPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "FlashInfer trtllm_batch_decode_with_kv_cache_mla in sparse mode over a"
                " ragged prefill batch, FP8 query and paged cache."
            ),
            url="https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/mla/_core.py",
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="rocm_triton_mla_sparse",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16}),
            arch_targets=frozenset({"CDNA3"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_sparse_mla_prefill_rocm_triton",
            function_name="profile_dsa_sparse_mla_prefill_rocm_triton",
        ),
        table_name=KIND,
        args_schema=DsaSparseMlaPrefillArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_rocm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's rope-free Triton ragged sparse-MLA kernel "
                "(rocm_sparse_attn_prefill, head_dim=512 nope=512 rope=0) over a BF16 "
                "ragged prefill batch, the GLM-5.3-Flash DSA prefill path on MI300X."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py",
        ),
    )
)

__all__ = ["DsaSparseMlaPrefillArgs", "KIND"]
