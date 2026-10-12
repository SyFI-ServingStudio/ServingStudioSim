"""Inkling relative-position attention: a sliding-window or a global layer.

Source: the vLLM fork's ``vllm/models/inkling/nvidia/attention.py`` (with
``ops/qkvr_prep.py``, ``ops/fa4_rel_attention.py`` and ``sconv_swa_attn.py``) and
the Inkling-Small-NVFP4 checkpoint's tensor shapes. Per layer and token:

- one QKVR projection ``hidden -> q | k | v | r`` of width
  ``H*d + 2*H_kv*d + H*d_rel`` (checkpoint ``wq_du | wk_dv | wv_dv | wr_du``);
- a depthwise causal short convolution of ``W`` taps on the raw K and on the raw V
  (``k_sconv`` / ``v_sconv``), then the per-head ``q_norm`` / ``k_norm`` RMSNorms;
- the relative branch ``r`` (``H`` heads of ``d_rel``) times the learned
  ``rel_logits_proj [d_rel, E]``: one logit per head and distance ``0..E-1``, added
  to that distance's score. A distance of ``E`` or more adds nothing. ``E`` is
  ``rel_extent`` on global layers and ``sliding_window_size`` on local layers;
- global layers scale ``q`` by the log-scaling ``tau(position)``;
- ``softmax(q.k / d + bias) v`` over the visible keys: the last
  ``sliding_window_size`` keys including the query's own on a local layer (FA4
  window ``(W_s - 1, 0)``), every causal key on a global layer;
- the O projection.

Necessary work beyond the weight matmuls, per request appending ``append`` queries
after ``prefix`` cached tokens (query positions ``prefix .. prefix + append - 1``):

- attention math: ``4*H*d`` FLOPs per visible (query, key) pair —
  ``sum_q min(pos_q + 1, W_s)`` pairs on a local layer, the causal triangle on a
  global one;
- relative logits: ``2*H*d_rel`` FLOPs per reachable (query, distance) pair,
  ``sum_q min(pos_q + 1, E)``. A distance past a query's context is never read, so
  its logit is not necessary; ``d_rel x min(E, longest context)`` weight columns
  are read;
- cached K/V read: the cached keys the request's queries can see,
  ``min(prefix, W_s - 1)`` local and ``prefix`` global. The chunk's own keys are a
  fused activation;
- KV-cache append: ``min(append, W_s - 1)`` tokens local — older tokens of a chunk
  leave the window before any later query reads them — and ``append`` global;
- K/V short-conv state: the conv at position ``p`` reads taps ``p-W+1 .. p``, so a
  request reads ``min(prefix, W - 1)`` raw K and V rows and leaves
  ``min(append, W - 1)`` new ones for the next step.

The q/k norms, ``tau``, the softmax and the conv's residual add are elementwise
and stay out of the pinned FLOPs (model.work convention 4); their learned norm
scales are compulsory weight reads.

Geometry: ``Workload.causal_lm`` and logged per-request geometry carry one causal
interaction per request, which is exact. A collapsed analyzer aggregate (one
``full`` interaction per phase) is labelled through a mean request rebuilt from
the phase's request and token counts — exact for a single request and whenever
every request sits on one side of the window and extent caps. A global layer's
pairs are taken from the interactions themselves, so they stay exact either way.
The relative-logit bytes are the exception: they read ``ceil`` of the mean
request's context, so below the extent they count the mean context, not the
longest. The simulator arch logs per-request decode geometry, so analyzer labels
are exact.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from ..core import LearnedWeightGroup, MatmulGroup, Workload
from .base import AttentionSemantic


@dataclass(frozen=True)
class InklingRequest:
    """``count`` equal requests appending ``append`` queries after ``prefix``
    cached tokens."""

    count: float
    append: float
    prefix: float


def _interpolate(function: Callable[[int], float], x: float) -> float:
    """An integer-domain function at a real point, linearly interpolated: a mean
    request rebuilt from a collapsed aggregate may have fractional lengths."""
    low = math.floor(x)
    value = float(function(low))
    fraction = x - low
    if fraction:
        value += fraction * (function(low + 1) - value)
    return value


def capped_cumulative(n: float, cap: int | None) -> float:
    """``sum_{j=1..n} min(j, cap)``: keys (or distances) visible to the first
    ``n`` positions of a sequence. ``cap=None`` is the causal triangle."""
    if cap is None:
        return n * (n + 1) / 2.0

    def exact(m: int) -> int:
        if m <= cap:
            return m * (m + 1) // 2
        return cap * (cap + 1) // 2 + (m - cap) * cap

    return _interpolate(exact, n)


def _phase_requests(phase: str | None, workload: Workload) -> list[InklingRequest]:
    if not workload.attn:
        return []
    if all(interaction.mask == "causal" for interaction in workload.attn):
        return [
            InklingRequest(
                float(interaction.multiplicity),
                float(interaction.num_query),
                float(interaction.num_cached_key),
            )
            for interaction in workload.attn
        ]
    if any(interaction.mask == "causal" for interaction in workload.attn):
        raise ValueError("Inkling attention cannot mix exact and collapsed geometry")
    requests = float(workload.num_attention_steps)
    tokens = float(workload.matmul_tokens)
    if requests <= 0 or tokens <= 0:
        return []
    cached = sum(
        interaction.num_cached_key * interaction.multiplicity for interaction in workload.attn
    )
    prefix = cached / requests
    if phase == "decode":
        # The collapsed decode aggregate counts each request's own new token as cached.
        prefix = max(prefix - tokens / requests, 0.0)
    return [InklingRequest(requests, tokens / requests, prefix)]


def inkling_requests(workload: Workload) -> list[tuple[str | None, list[InklingRequest]]]:
    """Each attention phase's requests, in ``Workload.attention_phases`` order."""
    return [
        (phase, _phase_requests(phase, phase_workload))
        for phase, phase_workload in workload.attention_phases()
    ]


def state_rows(requests: list[InklingRequest], history: int) -> float:
    """Rows of a ``history``-deep per-request state read plus written: the cached
    rows the first new position reaches, and the newest rows left behind."""
    return sum(
        request.count * (min(request.prefix, history) + min(request.append, history))
        for request in requests
    )


@dataclass
class InklingRelAttention:
    """One Inkling attention layer; ``window=None`` is a global layer."""

    hidden: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    d_rel: int
    #: Relative distances with a learned logit (``rel_logits_proj`` columns).
    rel_extent: int
    #: Keys a query sees including its own; ``None`` for full causal attention.
    window: int | None
    sconv_kernel: int
    kv_dtype_bytes: float
    #: The short-conv state and the rel-logit table are stored at the master dtype.
    master_dtype_bytes: float

    split_attention_phases = True

    @property
    def kv_width(self) -> int:
        return self.num_kv_heads * self.head_dim

    def matmul_groups(self) -> list[MatmulGroup]:
        def group(name: str, n: int, k: int, module: str) -> MatmulGroup:
            return MatmulGroup(name, n=n, k=k, bucket="attn_proj", module=f"attn.{module}")

        qkvr = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        qkvr += self.num_heads * self.d_rel
        return [
            group("qkvr", qkvr, self.hidden, "qkvr"),
            # A depthwise Conv1d is a MatmulGroup with n=channels, k=taps: its
            # parameters and 2*T*channels*taps FLOPs match it exactly (as KDA's).
            group("k_sconv", self.kv_width, self.sconv_kernel, "k_sconv"),
            group("v_sconv", self.kv_width, self.sconv_kernel, "v_sconv"),
            group("o", self.hidden, self.num_heads * self.head_dim, "wo_ud"),
        ]

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return [
            LearnedWeightGroup("q_norm", self.head_dim, 1),
            LearnedWeightGroup("k_norm", self.head_dim, 1),
            # Only the columns a batch's contexts reach are read; the rel_logits
            # row reports them.
            LearnedWeightGroup(
                "rel_logits_proj", self.d_rel * self.rel_extent, 1, "attn", gathered=True
            ),
        ]

    # ------------------------------------------------------------------ geometry

    def _pairs(self, request: InklingRequest, cap: int | None) -> float:
        end = request.prefix + request.append
        return capped_cumulative(end, cap) - capped_cumulative(request.prefix, cap)

    def _cached_keys(self, request: InklingRequest) -> float:
        if self.window is None:
            return request.prefix
        return min(request.prefix, self.window - 1)

    def _appended_keys(self, request: InklingRequest) -> float:
        if self.window is None:
            return request.append
        return min(request.append, self.window - 1)

    # ------------------------------------------------------------ AttentionSpec

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        """Attention per phase, then the relative logits and the two persistent
        states. Every row is always emitted, even at zero work, so one iteration's
        row set does not depend on its batch composition."""
        rows: list[AttentionSemantic] = []
        requests: list[InklingRequest] = []
        kv_entry_bytes = 2.0 * self.kv_width * self.kv_dtype_bytes
        for phase, phase_workload in wl.attention_phases():
            phase_requests = _phase_requests(phase, phase_workload)
            requests.extend(phase_requests)
            if self.window is None:
                # The causal triangle is what every interaction's own pair count
                # states, collapsed or not, so no mean request is needed.
                pairs = sum(interaction.pairs() for interaction in phase_workload.attn)
            else:
                pairs = sum(r.count * self._pairs(r, self.window) for r in phase_requests)
            cached = sum(r.count * self._cached_keys(r) for r in phase_requests)
            rows.append(
                AttentionSemantic(
                    name="attn" if phase is None else f"attn.{phase}",
                    flops=4.0 * self.num_heads * self.head_dim * pairs,
                    bytes=cached * kv_entry_bytes,
                )
            )
        distances = sum(r.count * self._pairs(r, self.rel_extent) for r in requests)
        longest = max((math.ceil(r.prefix + r.append) for r in requests), default=0)
        rows.append(
            AttentionSemantic(
                name="rel_logits",
                bucket="attn_proj",
                byte_kind="weights",
                flops=2.0 * self.num_heads * self.d_rel * distances,
                bytes=self.d_rel * min(self.rel_extent, longest) * self.master_dtype_bytes,
            )
        )
        rows.append(
            AttentionSemantic(
                name="kv_cache_append",
                bucket="embedding",
                bytes=sum(r.count * self._appended_keys(r) for r in requests) * kv_entry_bytes,
            )
        )
        rows.append(
            AttentionSemantic(
                name="kv_sconv_state",
                bucket="embedding",
                bytes=state_rows(requests, self.sconv_kernel - 1)
                * 2.0
                * self.kv_width
                * self.master_dtype_bytes,
            )
        )
        return rows

    # Historical AttentionSpec surface; Model.label uses semantic_segments.
    def internal_flops(self, wl: Workload) -> float:
        return sum(row.flops for row in self.semantic_segments(wl) if row.bucket == "attn_internal")

    def kv_bytes(self, wl: Workload) -> float:
        return sum(row.bytes for row in self.semantic_segments(wl) if row.bucket == "attn_internal")

    def cache_write_bytes(self, wl: Workload) -> float:
        return sum(row.bytes for row in self.semantic_segments(wl) if row.bucket == "embedding")
