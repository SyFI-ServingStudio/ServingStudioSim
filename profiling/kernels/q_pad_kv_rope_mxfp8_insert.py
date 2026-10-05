"""Fused query head padding with KV RoPE, MXFP8 quantization and sliding-window insert.

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

This is a separate kind from ``qnorm_rope_kv_insert`` (the same CUDA op with
other flags) because that schema has no axis for the Q transform: its rows mean
Q RMSNorm plus Q RoPE in the head-major layout and a mixed FP8/bf16 MLA cache.
This launch does neither Q step, so carrying it there would make
``cache_dtype`` silently imply a different Q operation and leave ``rms_eps``
dead.

Launch variants, chosen inside the op: ``num_tokens >= 1024`` with
``padded_heads > 0`` runs ``...InsertKernelReducedGrid`` (one CTA per token
iterating its slots); otherwise the warp-per-(token, slot)
``...InsertKernel``. Both are one launch. The production SWA page is
32 tokens (``DeepseekV4SWACache(block_size=32)``).
"""

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

KIND = "q_pad_kv_rope_mxfp8_insert"


@dataclass(frozen=True)
class QPadKvRopeMxfp8InsertArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Query and KV rows the call reads.")
    num_insert_tokens: int = arg(
        unit="tokens", doc="Leading KV rows inserted into the sliding-window cache."
    )
    num_heads: int = arg(unit="heads", doc="Live query heads per token on this GPU.")
    padded_heads: int = arg(
        unit="heads",
        doc="Query heads in the zero-padded output; 0 when no padding is written.",
    )
    block_size: int = arg(unit="tokens", doc="Tokens per sliding-window cache page.")
    input_dtype: DType = arg(doc="Element type of the query and KV inputs.")
    swa_cache_format: str = arg(doc="Record format of the sliding-window KV cache.")


DOC = KernelDoc(
    title="Query padding, KV RoPE and MXFP8 window insert",
    summary=(
        "Copy the live query heads into a zero-padded buffer while rotating, "
        "quantizing to MXFP8 and inserting KV rows into a sliding-window cache."
    ),
    description=(
        "Between the attention projections and a fused attention kernel that "
        "applies query RoPE itself, one CUDA call prepares both inputs. Query "
        "side: the num_heads live heads are copied and zero-filled up to "
        "padded_heads in the attention kernel's 16-element chunk-interleaved "
        "layout, with no norm and no rotation. KV side: each 512-element row gets "
        "GPT-J RoPE on its last 64 elements and MXFP8 quantization (FP8 E4M3, one "
        "UE8M0 scale per 32 elements), and the first num_insert_tokens rows are "
        "written to the paged cache, each page holding its 512-byte data rows "
        "and then its 16-byte scale rows. padded_heads = 0 is the KV-only launch."
    ),
    category="Attention",
    subcategory="Compressed sparse MLA",
    formula=(
        "q_out[:, :num_heads] = q; q_out[:, num_heads:padded_heads] = 0",
        "cache[slot] = MXFP8(RoPE(kv)), 528 bytes per inserted row",
        "GB/s = [(2 · 512 · num_tokens · (num_heads + padded_heads) if padded_heads > 0) "
        "+ num_insert_tokens · (2 · 512 + 16 + 4 · 64 + 528)] / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "The call is one launch: a one-CTA-per-token grid when num_tokens ≥ 1024 "
        "and padded_heads > 0, a warp per (token, slot) otherwise. Before timing, "
        "the padded query must match bit for bit and every inserted record must "
        "decode to the rotated KV row within FP8 tolerance."
    ),
    caveats=(
        "Only head_dim 512, rope_dim 64, 32-token pages, bf16 inputs, an MXFP8 "
        "cache (528 B per token), live heads 8 to 128 padded to 64 or 128, and up "
        "to 65,536 tokens are measured, on B200.",
        "Positions and insert slots are random and distinct, as for decode tokens "
        "of different requests.",
        "TFLOPS is not computed. GB/s counts logical query, KV, position, "
        "RoPE-table and record bytes.",
    ),
    # The runner checks outputs with PyTorch, but has no separate full reference.
    reference=None,
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_cuda",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.q_pad_kv_rope_mxfp8_insert_vllm_cuda",
            function_name="profile_q_pad_kv_rope_mxfp8_insert_vllm_cuda",
        ),
        table_name=KIND,
        args_schema=QPadKvRopeMxfp8InsertArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # The same vLLM op as qnorm_rope_kv_insert, built for every CUDA arch; its
        # host launcher refuses below SM80 (the bf16 body compiles to a no-op there).
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.FP8_E4M3}),
            min_compute_capability=(8, 0),
        ),
        subprocess_env="vllm_upstream_fork_env",
        doc=BackendDoc(
            summary=(
                "vLLM's fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert CUDA op with "
                "query norm and query RoPE off, chunk-interleaved query and MXFP8 KV "
                "records."
            ),
            url="https://github.com/vllm-project/vllm/blob/main/csrc/libtorch_stable/fused_qnorm_rope_kv_insert_kernel.cu",
        ),
    )
)

__all__ = ["QPadKvRopeMxfp8InsertArgs", "KIND"]
