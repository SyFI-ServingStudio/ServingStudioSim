"""DeepSeek V4.1 fused Q pad and MXFP8 sliding-window KV insert.

Source: alignment fork ``servingstudio-alignment-v41``,
``vllm/models/deepseek_v41/attention.py`` (``_fused_qnorm_rope_kv_insert``),
which calls ``torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert``
once per layer on the SM100 FlashMLA mega-attention path. That path passes
``apply_q_norm=False`` (``qr`` is normed before ``wq_b``), ``apply_q_rope=False``
and ``is_q_interleaved=True`` (the attention kernel does Q RoPE itself and reads
Q in its 16-dim chunk-interleaved layout), and ``kv_mxfp8=True``. So:

- Q side: copy the ``num_heads`` live heads and zero-fill up to
  ``padded_heads``, in the chunk-interleaved layout. No norm, no rotation.
  ``padded_heads == 0`` is the KV-only launch the layer makes once its shard
  is already ``padded_heads`` wide (TP1 on the 64-head checkpoint).
- KV side: GPT-J RoPE on the last 64 of 512 dims, MXFP8 quantization of the
  whole row (FP8 E4M3 with one UE8M0 scale per 32 dims), and a paged insert
  into the sliding-window cache (``swa_cache_format="mxfp8"``, 528 B/token:
  each page stores all 512-byte data rows, then all 16-byte scale rows).
  Only the first ``num_insert_tokens`` rows are inserted, as with DP padding.

This is a separate kind from ``deepseek_v4_qnorm_rope_kv_insert`` because the
V4 schema has no axis for the Q transform: its rows mean Q RMSNorm plus Q RoPE
in the head-major layout. The V4.1 launch does neither, so carrying it there
would make ``cache_dtype`` silently imply a different Q operation and leave
``rms_eps`` dead.

Launch variants, chosen inside the op: ``num_tokens >= 1024`` with
``padded_heads > 0`` runs ``...InsertKernelReducedGrid`` (one CTA per token
iterating its slots); otherwise the warp-per-(token, slot)
``...InsertKernel``. Both are one launch. The production SWA page is
32 tokens (``DeepseekV4SWACache(block_size=32)``).
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

KIND = "deepseek_v41_qnorm_rope_kv_insert"


@dataclass(frozen=True)
class DeepseekV41QnormRopeKvInsertArgs(KernelArgs):
    num_tokens: int
    num_insert_tokens: int
    num_heads: int
    padded_heads: int
    block_size: int
    input_dtype: DType
    swa_cache_format: str


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.deepseek_v41_qnorm_rope_kv_insert",
            function_name="profile_deepseek_v41_qnorm_rope_kv_insert_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=DeepseekV41QnormRopeKvInsertArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP8_E4M3}),
            gpus=frozenset({"NVIDIA B200"}),
        ),
        subprocess_env="vllm_fork_env",
    )
)

__all__ = ["DeepseekV41QnormRopeKvInsertArgs", "KIND"]
