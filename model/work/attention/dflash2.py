"""DFlash2 draft attention: a non-causal query block over a sliding window.

A DFlash2 draft (vLLM ``qwen3_dflash.DFlashQwen3Attention``) never runs over its
own context. The context's K/V are written into the draft's cache ahead of the
forward (``precompute_and_store_context_kv``, :class:`ContextKvProjection`), and
the forward attends only the request's query block: the bonus token plus the
mask tokens, ``q = draft_tokens + 1`` rows.

The block attends non-causally (the checkpoint sets ``is_causal: false``) over
the context plus the block itself, inside a sliding window of ``W`` keys. vLLM
makes a non-causal window symmetric (``flash_attn._maybe_symmetrize_window``:
``(W - 1, W - 1)``), as does HF's bidirectional sliding mask
(``|q_pos - k_pos| < W``). With ``S = context + q`` keys and the block at the
end, query ``i`` attends ``min(S, W + q - 1 - i)`` keys, so per request

    pairs = Σ_{t=0}^{q-1} min(S, W + t)

and the cached context keys read are the union over the block,
``min(context, W - 1)``. The block's own K/V are fused activations: vLLM writes
them to cache slots that the next step's context precomputation overwrites, so
they carry no compulsory write.

An interaction carries the honest geometry -- ``num_query = q``,
``num_key = context + q``, ``num_cached_key = context``, ``mask = "full"`` --
and this spec applies the window, which is a property of the mechanism, not of
the workload.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core import AttnInteraction, LearnedWeightGroup, MatmulGroup, Workload
from .base import AttentionSemantic


def window_pairs(num_query: int, num_key: int, window: int) -> int:
    """Exact (query, key) pairs of one non-causal query block at the end of
    ``num_key`` keys under a symmetric sliding window of ``window`` keys."""
    if num_query < 1 or num_key < num_query or window < num_query:
        raise ValueError(
            f"window block needs 1 <= q <= keys and q <= window, got "
            f"q={num_query}, keys={num_key}, window={window}"
        )
    # Terms t with W + t < S are window-bound; the rest see every key.
    bound = min(num_query, max(0, num_key - window))
    return bound * window + bound * (bound - 1) // 2 + (num_query - bound) * num_key


@dataclass
class BlockWindowAttention:
    """One DFlash2 draft layer's attention (Qwen3 GQA with per-head q/k norms)."""

    split_attention_phases = False

    hidden: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    sliding_window: int
    kv_dtype_bytes: float
    #: The engine attends the FP8 KV cache with an FP8 query, so the necessary
    #: attention math runs on the FP8 tensor cores whatever the weight dtype.
    compute_dtype: str | None = None
    module_prefix: str = ""

    def _module(self, path: str) -> str:
        return f"{self.module_prefix}.{path}" if self.module_prefix else path

    def matmul_groups(self) -> list[MatmulGroup]:
        return [
            MatmulGroup(
                "qkv",
                n=(self.num_qo_heads + 2 * self.num_kv_heads) * self.head_dim,
                k=self.hidden,
                bucket="attn_proj",
                module=self._module("self_attn.qkv_proj"),
            ),
            MatmulGroup(
                "o",
                n=self.hidden,
                k=self.num_qo_heads * self.head_dim,
                bucket="attn_proj",
                module=self._module("self_attn.o_proj"),
            ),
        ]

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return [
            LearnedWeightGroup("q_norm", self.head_dim, 1, "norm"),
            LearnedWeightGroup("k_norm", self.head_dim, 1, "norm"),
        ]

    def _pairs(self, interaction: AttnInteraction) -> float:
        if interaction.mask != "full":
            raise ValueError("a DFlash2 query block is non-causal; expected mask='full'")
        if interaction.num_cached_key != interaction.num_key - interaction.num_query:
            raise ValueError("a DFlash2 block attends its whole cached context plus itself")
        return interaction.multiplicity * window_pairs(
            interaction.num_query, interaction.num_key, self.sliding_window
        )

    def internal_flops(self, wl: Workload) -> float:
        # QK^T and P·V: 2·head_dim MACs each per pair per query head.
        return 4.0 * self.num_qo_heads * self.head_dim * sum(self._pairs(i) for i in wl.attn)

    def kv_bytes(self, wl: Workload) -> float:
        per_token = 2.0 * self.num_kv_heads * self.head_dim * self.kv_dtype_bytes
        return per_token * sum(
            min(i.num_cached_key, self.sliding_window - 1) * i.multiplicity for i in wl.attn
        )

    def cache_write_bytes(self, wl: Workload) -> float:
        return 0.0

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        # One row whatever the batch holds: every request drafts one block.
        return [
            AttentionSemantic(
                "attn",
                flops=self.internal_flops(wl),
                bytes=self.kv_bytes(wl),
                compute_dtype=self.compute_dtype,
            )
        ]


@dataclass
class ContextKvProjection:
    """The draft's context K/V for one layer, from the target's hidden states.

    ``precompute_and_store_context_kv`` runs the K and V halves of every draft
    layer's ``qkv_proj`` over the target rows this step scheduled, normalizes K,
    rotates it, and writes both into the draft's paged cache. Those writes are
    the persistent state the draft's attention reads later, so they are
    compulsory; norm and RoPE math are not.
    """

    split_attention_phases = False

    hidden: int
    num_kv_heads: int
    head_dim: int
    kv_dtype_bytes: float
    module_prefix: str = ""

    def matmul_groups(self) -> list[MatmulGroup]:
        prefix = f"{self.module_prefix}." if self.module_prefix else ""
        return [
            MatmulGroup(
                "kv_proj",
                n=2 * self.num_kv_heads * self.head_dim,
                k=self.hidden,
                bucket="attn_proj",
                module=f"{prefix}self_attn.kv_proj",
            )
        ]

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return [LearnedWeightGroup("k_norm", self.head_dim, 1, "norm")]

    def internal_flops(self, wl: Workload) -> float:
        return 0.0

    def kv_bytes(self, wl: Workload) -> float:
        return 0.0

    def cache_write_bytes(self, wl: Workload) -> float:
        return 2.0 * self.num_kv_heads * self.head_dim * self.kv_dtype_bytes * wl.matmul_tokens

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        return [
            AttentionSemantic(
                "kv_cache_append",
                bucket="embedding",
                bytes=self.cache_write_bytes(wl),
            )
        ]
