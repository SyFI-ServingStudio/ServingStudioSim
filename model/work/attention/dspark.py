"""DeepSeek-V4.1 DSpark draft attention: context K/V and a non-causal query block.

Derived from DeepSeek's reference ``inference/model.py`` (``DSparkAttention``,
``get_dspark_topk_idxs``) and the config, never from simulator shapes. A DSpark
draft layer is a DeepSeek-V4.1 sliding-window layer (``compress_ratio`` 0: no
compressor, no indexer) whose attention has two inputs:

- the **context** ``main_x``, one row per target row this step scheduled. Each
  layer projects it with its own ``wkv`` and ``kv_norm`` and appends it to that
  layer's own sliding-window cache (:class:`DSparkContextKv`). Only the last
  ``window`` rows of a request's chunk survive in the ring, so only they are a
  compulsory write; the entry is the target's 528-byte FP8 + UE8M0 window entry.
- the **block**, ``block_size`` query rows per request (the anchor token, then
  noise tokens). They run the layer's full query path (``wq_a`` -> ``q_norm`` ->
  ``wq_b``), project their own keys with ``wkv``, and attend non-causally: every
  block query sees the request's ``min(window, context)`` cached context entries
  plus all ``block_size`` block keys (:class:`DSparkBlockAttention`). The block's
  keys are fused activations, never written to the cache.

So per request and layer the block attends
``block_size * (min(window, context) + block_size)`` (query, key) pairs and reads
``min(window, context)`` cached entries. Each pair costs a 512-wide dot product
plus a 512-wide value update per head, exactly as in the target's attention.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core import AttnInteraction, LearnedWeightGroup, MatmulGroup, Workload
from .base import AttentionSemantic
from .deepseek_v41 import INDEX_NONE, WINDOW_ENTRY_BYTES, DeepseekV41Attention


@dataclass
class DSparkBlockAttention(DeepseekV41Attention):
    """One DSpark draft layer's attention over its query block.

    The projections, norms and sink are the target's sliding-window layer's
    (``compress_ratio`` 0); only the geometry differs. The workload carries one
    ``full`` interaction per group of equal requests: ``num_query`` is the block,
    ``num_cached_key`` the context the request drafts against.
    """

    block_size: int = 0

    split_attention_phases = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.compress_ratio != 0 or self.index_role != INDEX_NONE or self.kv_source:
            raise ValueError("a DSpark draft layer is sliding-window only")
        if self.block_size < 1:
            raise ValueError("DSpark block_size must be positive")

    def _check(self, interaction: AttnInteraction) -> None:
        if interaction.mask != "full":
            raise ValueError("a DSpark query block is non-causal; expected mask='full'")
        if interaction.num_query != self.block_size:
            raise ValueError(
                f"a DSpark block has {self.block_size} query rows, got {interaction.num_query}"
            )
        if interaction.num_cached_key != interaction.num_key - interaction.num_query:
            raise ValueError("a DSpark block attends its cached context plus itself")

    def _context_entries(self, interaction: AttnInteraction) -> float:
        return min(interaction.num_cached_key, self.window)

    def _pairs(self, wl: Workload) -> float:
        total = 0.0
        for interaction in wl.attn:
            self._check(interaction)
            keys = self._context_entries(interaction) + self.block_size
            total += interaction.multiplicity * self.block_size * keys
        return total

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        # One row whatever the batch holds: every request drafts one block.
        return [
            AttentionSemantic(
                name="attn",
                flops=2.0 * self.num_heads * (2 * self.head_dim) * self._pairs(wl),
                bytes=sum(
                    interaction.multiplicity * self._context_entries(interaction)
                    for interaction in wl.attn
                )
                * WINDOW_ENTRY_BYTES,
                compute_dtype="bf16",
            )
        ]

    def internal_flops(self, wl: Workload) -> float:
        return self.semantic_segments(wl)[0].flops

    def kv_bytes(self, wl: Workload) -> float:
        return self.semantic_segments(wl)[0].bytes

    def cache_write_bytes(self, wl: Workload) -> float:
        return 0.0


@dataclass
class DSparkContextKv:
    """One DSpark draft layer's context K/V over the target rows.

    The workload is the target's own: one causal interaction per request, whose
    ``num_query`` is the rows the request appended this step. The ``wkv`` weights
    and ``kv_norm`` scale belong to the draft layer, which counts them as
    parameters; this pass reads them again.
    """

    hidden: int
    head_dim: int
    window: int

    split_attention_phases = False

    def matmul_groups(self) -> list[MatmulGroup]:
        return [
            MatmulGroup(
                "wkv", n=self.head_dim, k=self.hidden, bucket="attn_proj", module="attn.wkv"
            )
        ]

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return [LearnedWeightGroup("kv_norm", self.head_dim, 1)]

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        appended = sum(
            interaction.multiplicity * min(interaction.num_query, self.window)
            for interaction in wl.attn
        )
        return [
            AttentionSemantic(
                name="window_cache_append",
                bucket="embedding",
                bytes=appended * WINDOW_ENTRY_BYTES,
            )
        ]

    def internal_flops(self, wl: Workload) -> float:
        return 0.0

    def kv_bytes(self, wl: Workload) -> float:
        return 0.0

    def cache_write_bytes(self, wl: Workload) -> float:
        return self.semantic_segments(wl)[0].bytes
