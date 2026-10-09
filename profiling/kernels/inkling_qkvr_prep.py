"""Inkling's fused QKVR prep: K/V short conv, q/k norms, log scaling, KV insert, rel logits.

Source: alignment fork, ``vllm/models/inkling/nvidia/ops/qkvr_prep.py``
(``fused_qkvr_prep``), called once per layer by ``InklingAttention.forward``
(``attention.py``) between the BF16 ``qkvr`` GEMM and the FA4 relative
attention. Per token of the per-rank ``qkvr`` row (q | k | v | r):

- depthwise causal short conv (``sconv_kernel_size`` taps) plus residual on K
  and V; taps of earlier forwards come from the layer's paged conv-state cache
  (``conv_block_size``-token blocks), taps of this forward from ``qkvr``, and the
  raw K/V are written to the conv-state cache;
- per-head RMSNorm on q and on the convolved k; on global layers q is
  multiplied by the log-length scale tau;
- the normed k and convolved v are written to the paged LBNHC attention cache
  (``page_size``-token pages, K and V interleaved per token and head);
- rel logits ``r[t, h, :] @ W_rel`` (``d_rel`` -> ``rel_extent``), times tau
  on global layers.

The call switches implementation at 128 tokens: one ``_qkvr_qkv_kernel`` launch
below; above, ``_kv_kernel`` on vLLM's ``aux_stream()`` in parallel with
``_q_kernel`` plus a rel-projection kernel on the current stream (the
throughput variant once ``8 * num_tokens`` reaches 8192 rows for ``rel_extent``
512 or 2048 rows otherwise). The elementwise byte placeholder cannot express it:
its output is 3.7-6.2x its input, decode is launch-bound, and the fork overlaps
two streams. One row is one whole call, priced by its kernels' interval union.

Tap source: a decode token reads ``W - 1`` cached taps from the conv-state
cache, a prefill token reads them from ``qkvr``. The runner times all-decode
batches (one request per token), which is what production runs below 128
tokens. Measured on B200 with CUDA-graph replays, a single prefill request of
the same size is 5-20% faster below 128 tokens and 5-15% faster from 128
(e.g. 2048 tokens: local 14.3 us decode vs 13.0 prefill / 12.9 chunk plus 32
decodes; global 18.6 vs 17.3 / 17.6), so rows are an upper bound for
prefill-heavy batches. That band did not justify a tap-source or request-count
axis; ``num_tokens`` is the only input.
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

KIND = "inkling_qkvr_prep"


@dataclass(frozen=True)
class InklingQkvrPrepArgs(KernelArgs):
    num_tokens: int = arg(unit="tokens", doc="Rows of the qkvr input in this call.")
    num_q_heads: int = arg(unit="heads", doc="Query heads on this GPU.")
    num_kv_heads: int = arg(unit="heads", doc="KV heads on this GPU.")
    head_dim: int = arg(unit="elements", doc="Width of one q, k or v head.")
    d_rel: int = arg(unit="elements", doc="Width of one head's relative-position branch r.")
    rel_extent: int = arg(
        unit="positions", doc="Relative distances the rel logits cover (columns of W_rel)."
    )
    apply_log_scaling: bool = arg(
        doc="Whether q and the rel logits are multiplied by tau (global layers)."
    )
    sconv_kernel_size: int = arg(unit="tokens", doc="Taps of the K/V short convolution.")
    conv_block_size: int = arg(unit="tokens", doc="Tokens per conv-state cache block.")
    page_size: int = arg(unit="tokens", doc="Tokens per attention KV cache page.")
    dtype: DType = arg(doc="Element type of qkvr, the weights, the outputs and both caches.")


DOC = KernelDoc(
    title="Inkling QKVR prep",
    summary=(
        "Short-conv K and V, RMS-normalize q and k, scale by tau, insert K/V into "
        "the paged cache and project the relative-position logits, in one call."
    ),
    description=(
        "Between the qkvr projection and relative attention, one call turns each "
        "token's q | k | v | r row into the attention inputs. K and V get a "
        "depthwise causal short convolution with residual, reading earlier "
        "tokens' taps from a paged conv-state cache and writing the raw K/V back "
        "to it. q and the convolved k are RMS-normalized per head; on global "
        "layers q and the rel logits are multiplied by the log-length scale tau. "
        "The normalized k and convolved v go into the paged attention cache, and "
        "r is projected to rel_extent relative-position logits per head. Below "
        "128 tokens this is one kernel; from 128 tokens the KV work runs on a "
        "second stream in parallel with the q and rel-projection kernels."
    ),
    category="Attention",
    subcategory="MHA / GQA",
    formula=(
        "k' = bf16(k + Σ_i w_k[i] · k[p − W + 1 + i]), v' likewise",
        "q̂ = RMSNorm(q) · tau, k̂ = RMSNorm(k'), rel = (r @ W_rel) · tau",
        "cache[slot] = (k̂, v'); conv_cache[conv_slot] = (k, v)",
        "GB/s = 2 · [num_tokens · (q_w + 2·kv_w + H_q·d_rel + q_w + H_q·rel_extent + 4·kv_w) "
        "+ d_rel·rel_extent] / time, q_w = H_q·head_dim, kv_w = H_kv·head_dim",
    ),
    default_metric="time_ms",
    method=(
        f"{CUPTI_METHOD} One call is captured into a CUDA graph, as production "
        "runs it, and each graph replay is one launch, priced by the interval "
        "union of its kernels on both streams (the overlap counts once), not "
        "summed. Before "
        "timing, q, rel logits, the attention-cache K/V and the conv-cache "
        "writes must match an independent PyTorch reference, with every other "
        "cache slot untouched."
    ),
    caveats=(
        "Every token is a decode token of its own request, so all W − 1 conv taps "
        "come from the conv-state cache; prefill-heavy batches of the same size "
        "run 5-20% faster on B200.",
        "tau values do not change the work; the runner draws them in [1, 1.3).",
        "head_dim 128, d_rel 16 and bf16 only, as the production call asserts.",
        "TFLOPS is not computed. GB/s counts the qkvr read, q and rel-logit "
        "writes and both caches' K/V writes, not the cached conv taps.",
    ),
    reference="profiling.runners.attention.inkling_qkvr_prep_reference",
)


register(
    KernelProfilerSpec(
        kernel_kind=KIND,
        backend="vllm_triton",
        runner_ref=RunnerRef(
            module_name="profiling.runners.attention.inkling_qkvr_prep_vllm_triton",
            function_name="profile_inkling_qkvr_prep_vllm_triton",
        ),
        table_name=KIND,
        args_schema=InklingQkvrPrepArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
        # Portable Triton kernels; the rel projection is a bf16 tl.dot, which
        # Triton lowers to tensor-core MMA from SM80.
        supports=BackendSupport(
            compute=frozenset({DType.BF16}),
            kv=frozenset({DType.BF16}),
            min_compute_capability=(8, 0),
        ),
        # Inkling exists only in the upstream-rebased alignment fork.
        subprocess_env="vllm_upstream_fork_env",
        doc=BackendDoc(
            summary=(
                "The alignment fork's fused_qkvr_prep Triton call: one kernel below "
                "128 tokens, a two-stream KV ∥ q + rel-projection fork above."
            ),
            url=None,
        ),
    )
)

__all__ = ["InklingQkvrPrepArgs", "KIND"]
