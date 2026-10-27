"""Paged batch decode attention kernel kind.

All Python-side per-kernel knowledge for ``paged_attn_batch_decode`` lives
here: the wire string ``KIND``, the ``PagedAttnBatchDecodeArgs`` schema, and
the ``register(...)`` call that wires the kind into ``profiling.db.registry``.

Source: vLLM ``FlashInferImpl.forward`` (``vllm/v1/attention/backends/
flashinfer.py``) on a TRTLLM-attention deployment (quantized query). The
metadata builder splits every step into decode requests (one query token) and
prefill requests (``split_decodes_and_prefills``); the decode part is ONE
``flashinfer.decode.trtllm_batch_decode_with_kv_cache`` call per layer over a
paged cache in HND layout whose last dimension packs a head's K then V
channels. The prefill part of a mixed step is a separate call
(``paged_attn_batch_prefill``).

The call issues FlashInfer's zeroing of its multi-CTA KV counter (vLLM passes
no counter buffer, so FlashInfer allocates and clears one per call) and the
trtllm-gen decode kernel, which splits each request's KV range over up to
``ceil(max_seq_len / 512)`` CTAs bounded by the GPU's SM count, and reduces
the splits inside the kernel. Both launches are the L1 unit.

The batch is keyed by ``(batch_size, max_kv_len, total_kv_len)``: the time is
set by the longest request (its splits run serially once the SMs are shared)
and by the KV bytes of the whole batch, and batches with the same three values
measure within about 10%. The runner times the canonical batch: one request
at ``max_kv_len`` and the other ``batch_size - 1`` sharing the rest evenly
(each at least one token, as a decode request reads its own).

``flashinfer_fa2`` is the same decode call where TRTLLM attention is
unavailable (sm_8x, e.g. L40S): ``FlashInferImpl.forward`` runs the decode
requests through ``BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True)``
with a BF16 query over the FP8 cache, i.e. FlashInfer's batch prefill kernel
with a 16-row query tile plus its split-KV merge kernel. Both launches are the
L1 unit, keyed and canonicalized as above.
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

KIND = "paged_attn_batch_decode"


@dataclass(frozen=True)
class PagedAttnBatchDecodeArgs(KernelArgs):
    num_qo_heads: int = arg(unit="heads", doc="Query and output heads on this GPU.")
    num_kv_heads: int = arg(unit="heads", doc="KV heads on this GPU.")
    head_dim: int = arg(unit="elements", doc="Dimension of each query, key and value head.")
    q_dtype: DType = arg(doc="Query element type.")
    kv_dtype: DType = arg(doc="Key and value cache element type.")
    o_dtype: DType = arg(doc="Output element type.")
    batch_size: int = arg(
        unit="requests",
        doc="Rows in the call; a CUDA-graph replay counts its zero-length padding rows.",
    )
    max_kv_len: int = arg(unit="tokens", doc="Context of the longest request, its query included.")
    total_kv_len: int = arg(unit="tokens", doc="Context tokens summed over the batch.")


DOC = KernelDoc(
    title="Paged batch decode attention",
    summary=(
        "One-token-per-request GQA attention over a paged KV cache for a whole "
        "decode batch, keyed by the batch's longest context and total context."
    ),
    description=(
        "One call reads every decode request's cached keys and values through its "
        "page table. The kernel splits each request's KV range over several CTAs "
        "while the batch leaves SMs free and merges the splits itself, so the "
        "longest request bounds the step when the batch is small, and the total KV "
        "bytes when it is large. The timed batch is one request at max_kv_len and "
        "the rest sharing the remaining context evenly."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "O[b, h] = softmax(Q[b, h] · K[b, h // G]ᵀ · scale) · V[b, h // G]",
        "TFLOPS = 4 · num_qo_heads · head_dim · total_kv_len / time",
        "GB/s = (Q + O + every request's K and V once) bytes / time",
    ),
    default_metric="memory_bandwidth_gbps",
    method=(
        f"{CUPTI_METHOD} "
        "Every launch of the call counts. trtllm-gen: FlashInfer's counter memset "
        "and the decode kernel; the planning bound max_seq_len is the larger of "
        "max_kv_len and 512 tokens per SM, which saturates the split choice as "
        "vLLM's FULL CUDA graph captured at max_model_len does. fa2: the batch "
        "prefill kernel and the split-KV merge, planned as vLLM's FULL-CUDA-graph "
        "decode wrapper is (that plan always splits and merges). Before timing, "
        "the output is checked against a PyTorch reference on sampled requests."
    ),
    caveats=(
        "Queries, keys and values are random; the quantization scales are 1.",
        "Pages hold 32 tokens: 16, 32 and 64 measured within 1% on both backends, "
        "so the page size is not part of the key.",
        "fa2: a mixed step's decode rows run vLLM's eager wrapper, which skips the "
        "merge when it does not split; measured equal at c8 contexts and about "
        "2 us faster for one short request.",
        "An eager mixed step passes the whole batch's longest context as the "
        "bound; below the saturating bound that is up to about 3 us faster for "
        "batches of at most 4 requests with contexts of at most 4096 tokens.",
        "A batch whose context is spread unlike the canonical one can differ by "
        "about 10% at the same key.",
        "CUDA-graph padding rows hold no context; zero-length rows measured within "
        "10% of one-token rows.",
    ),
    reference="profiling.runners.attention.fa4_varlen_attention_reference",
)


# trtllm-gen decode cubins ship for sm_100f; vLLM routes FP8 queries over an
# FP8 cache to them (flashinfer.py use_trtllm_attention).
_SUPPORT = BackendSupport(
    compute=frozenset({DType.FP8_E4M3}),
    kv=frozenset({DType.FP8_E4M3}),
    sm_targets=frozenset({"sm_100f"}),
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_trtllm_gen",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.paged_attn_batch_trtllm_gen",
            function_name="profile_paged_attn_batch_decode",
        ),
        table_name=KIND,
        args_schema=PagedAttnBatchDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=_SUPPORT,
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "FlashInfer's trtllm_batch_decode_with_kv_cache(backend='trtllm-gen') "
                "as vLLM's FlashInferImpl.forward calls it for the decode requests."
            ),
        ),
    )
)


# FlashInfer's FA2 batch prefill kernel, which vLLM's native FlashInfer path runs
# for the decode requests where TRTLLM attention is unavailable (sm_8x, e.g.
# L40S). FlashInfer JIT-builds it for sm_80 and newer and converts FP8 KV in
# software below sm_89 (vec_dtypes.cuh FLASHINFER_HARDWARE_FP8_CONVERSION_ENABLED);
# vLLM keeps the query in the model dtype on fa2-only architectures
# (flashinfer.py get_q_data_type).
_FA2_SUPPORT = BackendSupport(
    compute=frozenset({DType.BF16}),
    kv=frozenset({DType.FP8_E4M3}),
    min_compute_capability=(8, 0),
)

register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="flashinfer_fa2",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.paged_attn_batch_fa2",
            function_name="profile_paged_attn_batch_decode",
        ),
        table_name=KIND,
        args_schema=PagedAttnBatchDecodeArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        supports=_FA2_SUPPORT,
        subprocess_env="vllm_env",
        doc=BackendDoc(
            summary=(
                "FlashInfer's BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True) "
                "on its fa2 backend, as vLLM's FlashInferImpl.forward runs it for the "
                "decode requests off TRTLLM: the batch prefill kernel with a 16-row "
                "query tile and the split-KV merge, a BF16 query over the FP8 cache in "
                "32-token pages."
            ),
        ),
    )
)

__all__ = ["KIND", "PagedAttnBatchDecodeArgs"]
