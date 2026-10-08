"""Sparse MLA attention kernel kind over DSA-selected cache tokens."""

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

KIND: str = "dsa_sparse_mla_attention"


@dataclass(frozen=True)
class DsaSparseMlaAttentionArgs(KernelArgs):
    num_queries: int = arg(unit="tokens", doc="Query tokens in the attention batch.")
    num_cache_tokens: int = arg(
        unit="tokens", doc="Cache tokens available to the selected indices."
    )
    num_heads: int = arg(unit="heads", doc="Query heads on this GPU.")
    num_kv_heads: int = arg(unit="heads", doc="KV heads shared by the query heads.")
    selected_k: int = arg(unit="tokens", doc="Selected index slots per query.")
    latent_dim: int = arg(unit="elements", doc="Compressed KV latent width.")
    rope_dim: int = arg(
        unit="elements", doc="Rotary-position width appended to the latent; 0 when there is none."
    )
    value_dim: int = arg(unit="elements", doc="Value and output width per head.")
    softmax_scale: float = arg(unit="multiplier", doc="Multiplier applied to attention scores.")
    q_dtype: DType = arg(doc="Query element type.")
    cache_dtype: DType = arg(doc="Compressed KV cache element type.")
    index_dtype: str = arg(doc="Element type of selected indices.")
    output_dtype: DType = arg(doc="Attention output element type.")
    valid_counts: str = arg(doc="Encoded valid selected-index count per query.")
    index_distribution: str = arg(doc="Pattern used to construct selected indices.")
    cache_layout: str = arg(doc="Layout of compressed KV values in the cache.")


DOC = KernelDoc(
    title="Sparse MLA attention",
    summary="Attend to the DSA-selected cache tokens for each query head.",
    description=(
        "In sparse MLA attention, each query head attends only to the cache "
        "tokens the DSA indexer selected. Scores use the compressed "
        "latent and the rotary part, if any; values are the latent. valid_counts sets "
        "how many of the selected_k slots each query uses, and the rest are "
        "masked. Queries and cache values are synthetic, and the selected "
        "indices follow index_distribution."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "O = softmax(softmax_scale · q · K_selectedᵀ) · V_selected",
        "D = latent_dim + rope_dim; Q = num_queries; H = num_heads; K = selected_k",
        "TFLOPS = 2·Q·H·K·(D + value_dim) / time",
        "logical bytes = bq·Q·H·D + 4·Q·K + bc·Σvalid_counts·D + 2·Q·H·value_dim + 8·Q·H",
        "GB/s = logical bytes / time; bq = bc = 2 for BF16, 1 for FP8",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "torch counts every launch of its composite; flashinfer_trtllm_fp8 "
        "counts only fmhaSm100, and vllm_flashmla_bf16 only "
        "sparse_attn_fwd_kernel. Setup and output checks run before timing."
    ),
    caveats=(
        "Selected positions and tensor values are constructed, not taken from "
        "an indexer or a serving KV cache.",
        "TFLOPS counts all selected_k slots even when fewer are valid.",
        "torch and vllm_flashmla_bf16 accept only rope_dim = 64; "
        "flashinfer_trtllm_fp8 also accepts rope_dim = 0, with its own cache "
        "layout. vllm_flashmla_bf16 needs selected_k to be a multiple of "
        "FlashMLA's top-k tile: 128 on SM90, and 64 at 64 heads or 128 otherwise "
        "on SM10x.",
        "GB/s includes 8 bytes per query head for the max-logit and log-sum-exp"
        " outputs, even for backends that return only the attention output.",
    ),
    reference="profiling.runners.attention.dsa_sparse_mla_attention_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_sparse_mla_attention",
            function_name="profile_dsa_sparse_mla_attention_torch",
        ),
        table_name=KIND,
        args_schema=DsaSparseMlaAttentionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary="A PyTorch gather, score, softmax and value-reduction composite on BF16 inputs."
        ),
    )
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
            module_name="profiling.runners.attention.dsa_sparse_mla_attention",
            function_name="profile_dsa_sparse_mla_attention_flashinfer_trtllm_fp8",
        ),
        table_name=KIND,
        args_schema=DsaSparseMlaAttentionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "FlashInfer trtllm_batch_decode_with_kv_cache_mla in sparse mode, FP8 "
                "query and paged cache, on B200."
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
            module_name="profiling.runners.attention.dsa_sparse_mla_attention_rocm_triton",
            function_name="profile_dsa_sparse_mla_attention_rocm_triton",
        ),
        table_name=KIND,
        args_schema=DsaSparseMlaAttentionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_rocm_env",
        doc=BackendDoc(
            summary=(
                "vLLM's rope-free Triton ragged sparse-MLA kernel "
                "(rocm_sparse_attn_prefill, head_dim=512 nope=512 rope=0) on a BF16 "
                "latent, the GLM-5.3-Flash DSA decode path on MI300X."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/rocm_aiter_mla_sparse.py",
        ),
    )
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_flashmla_bf16",
        # FlashMLA's Arch::is_sm90a() / is_sm100f(): exactly SM90, or any SM10x.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16}),
            sm_targets=frozenset({"sm_90a", "sm_100f"}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_sparse_mla_attention",
            function_name="profile_dsa_sparse_mla_attention_vllm_flashmla_bf16",
        ),
        table_name=KIND,
        args_schema=DsaSparseMlaAttentionArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=("vLLM's flash_mla_sparse_fwd on BF16 selected KV tokens."),
            url="https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/ops/flashmla.py",
        ),
    )
)
