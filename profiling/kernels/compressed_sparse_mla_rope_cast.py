"""DeepSeek V4.1 FlashMLA mega attention: one layer's decode or prefill segment.

Source: alignment fork ``servingstudio-alignment-v41``. The kernel is
``vllm/models/deepseek_v41/nvidia/flash_mla_mega_attn.py``
(``DeepseekV4MegaAttnAttention``), which ``nvidia/model.py`` selects by
default on SM100. Per layer, ``forward_mqa`` makes one call per segment: the
prefill tokens and the decode tokens write disjoint ranges of one output
buffer. ``mode`` picks the segment:

- ``decode`` is ``_forward_decode_mega``, one
  ``fused_norm_rope_attn_rope_cast_decode`` launch over the paged caches;
- ``prefill`` is ``_forward_prefill_mega``. For each chunk of up to
  ``prefill_chunk_size`` requests, it launches the compressed-cache gather
  (``_dequantize_and_gather_k_nvfp4`` when ratio > 0), the MXFP8 SWA gather
  (CuTe DSL), ``combine_topk_swa_indices`` and one
  ``fused_norm_rope_attn_rope_cast_fwd``. All of these launches are in the slot.

The kernel math has Q RoPE (GPT-J pairs, last 64 of 512 dims) and 1-KV-head
sparse MLA with an attention sink. It attends over the 128-token sliding
window (MXFP8, 528 B/token) plus, for ``compress_ratio`` 1 or 2, up to
``index_topk`` indexer-selected compressed rows (NVFP4 ``nvfp4_ds_mla``,
288 B/token). The output then gets the inverse RoPE and an FP8 E4M3 cast with
UE8M0 per-32 scales, written in ``wo_a``'s permuted layout. The fork passes
``enable_q_norm=False``, so the kernel does no Q RMSNorm: V4.1 norms ``qr``
before ``wq_b``. Outside the slot: the fused Q-pad/KV insert, the decode
global top-k remap (``compute_global_topk_indices_and_lens``), and
``wo_a``/``wo_b``.

``num_heads`` is the kernel's padded Q head count (64 or 128). TP4 on the
64-head checkpoint leaves 16 live heads, padded to 64. The kernel runs, and
writes output for, all padded heads, so the live count does not change the
execution. ``query_context_pairs`` holds ``(query_len, context_len)`` per
request, with context including the query. In decode mode every query token
is one flattened row. ``max_model_len``, ``max_num_batched_tokens`` and
``prefill_chunk_size`` drive the prefill chunk planner, and decode ignores them.
"""

from dataclasses import dataclass

from profiling.db.args import DType, KernelArgs
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import (
    BackendSupport,
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    register,
)

KIND = "compressed_sparse_mla_rope_cast"


@dataclass(frozen=True)
class CompressedSparseMlaRopeCastArgs(KernelArgs):
    mode: str
    query_context_pairs: tuple[tuple[int, int], ...]
    compress_ratio: int
    window_size: int
    index_topk: int
    num_heads: int
    head_dim: int
    rope_dim: int
    max_model_len: int
    max_num_batched_tokens: int
    prefill_chunk_size: int
    q_dtype: DType
    swa_cache_format: str
    compressed_cache_format: str
    output_dtype: DType


_SUPPORT = BackendSupport(
    compute=frozenset({DType.BF16}),
    kv=frozenset({DType.FP8_E4M3}),
    gpus=frozenset({"NVIDIA B200"}),
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="torch",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.compressed_sparse_mla_rope_cast",
            function_name="profile_compressed_sparse_mla_rope_cast_torch",
        ),
        table_name=KIND,
        args_schema=CompressedSparseMlaRopeCastArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=_SUPPORT,
    )
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashmla_mega",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.compressed_sparse_mla_rope_cast",
            function_name="profile_compressed_sparse_mla_rope_cast_flashmla_mega",
        ),
        table_name=KIND,
        args_schema=CompressedSparseMlaRopeCastArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=_SUPPORT,
        subprocess_env="vllm_fork_env",
    )
)

__all__ = ["CompressedSparseMlaRopeCastArgs", "KIND"]
