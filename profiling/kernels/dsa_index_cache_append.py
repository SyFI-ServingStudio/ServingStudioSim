"""DSA index-key quantization and page-planar cache append kernel kind."""

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

KIND: str = "dsa_index_cache_append"


@dataclass(frozen=True)
class DsaIndexCacheAppendArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="New index-key tokens written to the cache.")
    index_dim: int = arg(unit="elements", doc="Elements in each index key.")
    block_size: int = arg(unit="tokens", doc="Tokens stored in each cache page.")
    quant_block_size: int = arg(unit="elements", doc="Key elements sharing one FP8 scale.")
    input_dtype: DType = arg(doc="Element type of the input index keys.")
    cache_dtype: DType = arg(doc="Element type of the cached index keys.")
    scale_format: str = arg(doc="Encoding of each quantization scale.")
    cache_format: str = arg(doc="Page-planar layout of cached keys and scales.")


DOC = KernelDoc(
    title="Indexer key cache append",
    summary="Quantize new DSA index keys and write them into a paged FP8 cache.",
    description=(
        "The DSA indexer stores each new index key in a paged FP8 cache "
        "for later scoring. Within a page, keys and their scales sit in "
        "separate planes, and a slot mapping scatters tokens across pages. "
        "torch and vllm_cuda quantize BF16 keys as given; SGLang's fused call "
        "also applies LayerNorm and RoPE to the keys before storing them."
    ),
    category="Attention",
    subcategory="DSA",
    formula=(
        "cache[slot] = FP8(key / scale), one scale per quant_block_size elements",
        "groups = index_dim / quant_block_size",
        "B = num_tokens · (2 · index_dim + 8 + index_dim + 4 · groups)",
        "GB/s = B / time; SGLang adds 4 · 64 bytes per token of RoPE-table reads",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        "torch is timed with CUDA events: five warm-up calls, then a loop of "
        "back-to-back calls, taking the median of three runs, so the gaps "
        "between its launches are included. vllm_cuda and "
        "sglang_fused_norm_rope_store use CUPTI kernel time with the L2 cache "
        "flushed before each launch: vllm_cuda counts only "
        "indexer_k_quant_and_cache_kernel, and SGLang counts every launch of "
        "its call after one setup call."
    ),
    caveats=(
        "torch and vllm_cuda round scales to powers of two (UE8M0); SGLang "
        "keeps FP32 scales and also runs LayerNorm and RoPE, so its rows cover "
        "more work.",
        "torch rows are warm-cache elapsed time and the others are cold-cache "
        "kernel time, so the backends are not directly comparable.",
        "GB/s counts logical key, slot, scale and cache bytes, not torch "
        "intermediates or physical transactions.",
    ),
    reference="profiling.runners.attention.dsa_index_cache_append_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP8_E4M3}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_index_cache_append",
            function_name="profile_dsa_index_cache_append_torch",
        ),
        table_name=KIND,
        args_schema=DsaIndexCacheAppendArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        doc=BackendDoc(
            summary="PyTorch quantization and scattered page writes as a multi-launch composite."
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP8_E4M3}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_index_cache_append",
            function_name="profile_dsa_index_cache_append_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=DsaIndexCacheAppendArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "vLLM indexer_k_quant_and_cache: one CUDA launch for FP8 "
                "quantization and page writes."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/cache_kernels.cu",
        ),
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="sglang_fused_norm_rope_store",
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP8_E4M3}),
        ),
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.dsa_index_cache_append",
            function_name="profile_dsa_index_cache_append_sglang_fused_norm_rope_store",
        ),
        table_name=KIND,
        args_schema=DsaIndexCacheAppendArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        subprocess_env="sglang_env",
        doc=BackendDoc(
            summary=(
                "SGLang fused_k_indexer_norm_rope_store: LayerNorm, RoPE, FP8 "
                "quantization and page writes in one call."
            ),
            url="https://github.com/sgl-project/sglang/blob/main/python/sglang/kernels/ops/quantization/dsv32/elementwise.py",
        ),
    )
)
