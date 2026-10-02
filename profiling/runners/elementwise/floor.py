"""Elementwise byte-placeholder *floor* adapters for the GLM-5.3-Flash MI300X port.

Several composed kinds of the ``glm53_flash_vllm_fp8_kda_dsa_moe`` arch carry a
negligible predicted share of iteration time (Phase-4 ranking, decision #38) yet
have no MI300X-native backend written. Rather than leave them pinned to an
NVIDIA-only backend — which a real MI300X ``timing-predict`` would reject at
``BackendSupport.allows`` — each is floored here onto the *already-measured*
MI300X ``elementwise`` byte-mover (``profiling.runners.elementwise.torch_rocm``).

Mechanism (decision #37 "mechanism B"): a per-kind ``elementwise_floor`` backend,
``gpus={MI300X}`` and compute-dtype-agnostic, whose runner derives the kind's
memory-bound byte footprint from its *shape* args and times the elementwise
byte-mover for that many bytes. The number is therefore a genuine measured
memory-bound floor, not a hand-written constant: it points at the measured
``elementwise`` ``torch_rocm`` row for the equivalent byte traffic. The floored
kind's profile.db row still lives under its own ``table_name`` (= kernel_kind),
keyed by its own args, so the cross-language facade is unaffected.

These are intentionally approximate: a memory-bound read-inputs + write-outputs
estimate is the designed floor for a negligible-share kind. A kind whose share
later proves material is promoted to a real backend (decision #29).

Every adapter delegates timing to ``profile_elementwise_torch_rocm``, which does
the rocprofv3 capture via the registered ``("elementwise", "torch_rocm")`` launch
driver, so no new ``profiling.profilers.rocprof_run._BUILDERS`` entry is needed.
"""

from __future__ import annotations

import math
from typing import Any

from profiling.db.args import DType
from profiling.runners.elementwise.torch_rocm import profile_elementwise_torch_rocm
from profiling.runners.metrics import ComputeMetrics


def _dt_bytes(value: Any) -> float:
    """Byte width of a dtype arg given as a wire string or ``DType``."""
    return DType.from_value(value).size_bytes()


def _index_bytes(index_dtype: str) -> int:
    """Byte width of an integer index dtype arg (a plain string, not ``DType``)."""
    token = str(index_dtype).lower()
    if "64" in token:
        return 8
    if "16" in token:
        return 2
    return 4  # int32 / uint32 and anything unrecognized


def _floor(input_size_bytes: float, output_size_bytes: float) -> ComputeMetrics:
    """Time the measured MI300X elementwise byte-mover for a derived footprint.

    Bytes are whole (the elementwise contract is ``uint8``); the output is forced
    to at least one byte so the byte-mover always has a write pass to time.
    """
    out_bytes = max(1, int(math.ceil(output_size_bytes)))
    in_bytes = max(0, int(math.ceil(input_size_bytes)))
    return profile_elementwise_torch_rocm(in_bytes, out_bytes)


# ── fp8 activation quantization ──────────────────────────────────────────────


def profile_fp8_per_token_group_quant_floor(
    *, num_tokens: int, hidden_size: int, group_size: int, input_dtype: Any, scale_format: Any, **_: Any
) -> ComputeMetrics:
    rows = num_tokens * hidden_size
    in_bytes = rows * _dt_bytes(input_dtype)
    # fp8 output (1 byte/elem) plus one fp32 scale per group.
    scales = num_tokens * math.ceil(hidden_size / max(1, group_size))
    out_bytes = rows * 1 + scales * 4
    return _floor(in_bytes, out_bytes)


# ── dense / router GEMM with fp32 output ─────────────────────────────────────


def profile_gemm_fp32_output_floor(
    *, m: int, n: int, k: int, input_dtype: Any, **_: Any
) -> ComputeMetrics:
    in_bytes = (m * k + n * k) * _dt_bytes(input_dtype)
    out_bytes = m * n * 4  # fp32 output
    return _floor(in_bytes, out_bytes)


# ── DSA indexer logits (decode / prefill) ────────────────────────────────────


def profile_dsa_paged_mqa_logits_decode_floor(
    *,
    batch_size: int,
    context_len: int,
    next_n: int,
    max_model_len: int,
    num_heads: int,
    head_dim: int,
    q_dtype: Any,
    cache_dtype: Any,
    output_dtype: Any,
    **_: Any,
) -> ComputeMetrics:
    queries = batch_size * next_n * num_heads * head_dim * _dt_bytes(q_dtype)
    keys = batch_size * context_len * head_dim * _dt_bytes(cache_dtype)
    out_bytes = batch_size * next_n * max_model_len * _dt_bytes(output_dtype)
    return _floor(queries + keys, out_bytes)


def profile_dsa_mqa_logits_prefill_floor(
    *,
    num_queries: int,
    num_keys: int,
    num_heads: int,
    head_dim: int,
    q_dtype: Any,
    k_dtype: Any,
    output_dtype: Any,
    **_: Any,
) -> ComputeMetrics:
    queries = num_queries * num_heads * head_dim * _dt_bytes(q_dtype)
    keys = num_keys * head_dim * _dt_bytes(k_dtype)
    out_bytes = num_queries * num_keys * _dt_bytes(output_dtype)
    return _floor(queries + keys, out_bytes)


# ── DSA top-k selection (decode / prefill) ───────────────────────────────────


def profile_dsa_persistent_topk_decode_floor(
    *,
    batch_size: int,
    next_n: int,
    max_model_len: int,
    top_k: int,
    logits_dtype: Any,
    index_dtype: str,
    **_: Any,
) -> ComputeMetrics:
    in_bytes = batch_size * next_n * max_model_len * _dt_bytes(logits_dtype)
    out_bytes = batch_size * next_n * top_k * _index_bytes(index_dtype)
    return _floor(in_bytes, out_bytes)


def profile_dsa_topk_prefill_floor(
    *,
    num_queries: int,
    num_keys: int,
    top_k: int,
    logits_dtype: Any,
    index_dtype: str,
    **_: Any,
) -> ComputeMetrics:
    in_bytes = num_queries * num_keys * _dt_bytes(logits_dtype)
    out_bytes = num_queries * top_k * _index_bytes(index_dtype)
    return _floor(in_bytes, out_bytes)


# ── MLA cache append ─────────────────────────────────────────────────────────


def profile_mla_cache_append_floor(
    *,
    num_tokens: int,
    kv_lora_rank: int,
    rope_dim: int,
    input_dtype: Any,
    kv_dtype: Any,
    **_: Any,
) -> ComputeMetrics:
    width = kv_lora_rank + rope_dim
    in_bytes = num_tokens * width * _dt_bytes(input_dtype)
    out_bytes = num_tokens * width * _dt_bytes(kv_dtype)
    return _floor(in_bytes, out_bytes)


# ── DSA sparse index remap ───────────────────────────────────────────────────


def profile_dsa_sparse_index_remap_floor(
    *, num_queries: int, selected_k: int, index_dtype: str, **_: Any
) -> ComputeMetrics:
    slot_bytes = num_queries * selected_k * _index_bytes(index_dtype)
    # Read local indices, write mapped indices: one pass each.
    return _floor(slot_bytes, slot_bytes)


# ── mHC residual-stream RMS norms ────────────────────────────────────────────


def profile_mhc_pre_rms_norm_floor(
    *, num_tokens: int, hidden_size: int, hc_mult: int, hidden_dtype: Any, **_: Any
) -> ComputeMetrics:
    stream_bytes = num_tokens * hidden_size * hc_mult * _dt_bytes(hidden_dtype)
    return _floor(stream_bytes, stream_bytes)


def profile_mhc_fused_post_pre_rms_norm_floor(
    *, num_tokens: int, hidden_size: int, hc_mult: int, hidden_dtype: Any, **_: Any
) -> ComputeMetrics:
    stream_bytes = num_tokens * hidden_size * hc_mult * _dt_bytes(hidden_dtype)
    # Fused post+pre reads the prior stream and its residual, writes the normed
    # stream: two read passes, one write.
    return _floor(2 * stream_bytes, stream_bytes)


# ── TP all-reduce fusion ─────────────────────────────────────────────────────


def profile_all_reduce_fusion_floor(
    *, num_tokens: int, hidden_dim: int, dtype: Any, **_: Any
) -> ComputeMetrics:
    tensor_bytes = num_tokens * hidden_dim * _dt_bytes(dtype)
    # A ring all-reduce moves roughly twice the tensor over the fabric; floor it
    # as two read passes and one write of the reduced tensor.
    return _floor(2 * tensor_bytes, tensor_bytes)
