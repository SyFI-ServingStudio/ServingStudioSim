"""GLM-5.3-Flash MLA with the kpool-compressed DSA indexer.

The attention is GLM's no-rope MLA (``qk_rope_head_dim == 0``): a query latent
``q_a -> q_b``, the absorbed per-head ``W_UK`` (a 512-wide query per head), attention
over the compressed 512-wide latent cache, and the per-head ``W_UV`` value-up — the
``q_absorb`` / ``v_up`` views of the single learned ``kv_b_proj``.

The indexer (vLLM fork ``glm5next/nvidia/attention.py`` ``Indexer`` +
``SparseAttnIndexerKpool``) differs from GLM-5.2's in what it scores:

- keys are *pooled*: every ``index_kpool`` consecutive tokens' indexer keys are merged
  into one entry by a per-channel softmax gate (``index_kpool_compress_gate`` +
  ``index_kpool_compress_ape``). A query at context ``k`` scores only the
  ``floor(k / kpool)`` complete pools; the in-progress tail pool is always selected
  (``index_kpool_always_select_tail``) and is not scored;
- top-k selects ``index_topk / kpool`` pools, i.e. ``index_topk`` tokens, plus the
  ``k mod kpool`` tail tokens. A query therefore attends to
  ``min(k, index_topk + k mod kpool)`` keys: all of them while at most
  ``index_topk / kpool`` complete pools exist (``k <= index_topk + kpool - 1``).

Per-query work depends on each request's own context, so the rows below need exact
causal interactions. A collapsed aggregate (one ``full`` interaction carrying summed
pairs) cannot recover per-request caps; it is rejected rather than silently labelled.

Semantic rows (per phase: ``indexer``, ``attn``; per iteration: the two cache writes):

- ``indexer``: ``2 * index_heads * index_head_dim`` FLOPs per (query, complete pool),
  FP8 (the index cache is FP8 by construction); reads each cached pooled entry once
  (FP8 key + one FP32 scale per 128-element block).
- ``attn``: ``2 * heads * (kv_lora + rope + kv_lora)`` FLOPs per (query, selected key).
  Bytes: the latent-cache keys the chunk's first query must read — its selected keys
  other than itself. Later queries' selections may add keys, so this is a lower bound
  on distinct reads, never an over-count.
- ``mla_cache_append``: every new token's latent (+rope) entry.
- ``index_cache_append``: one pooled FP8 entry per pool the batch completes.

Pool compression (gated per-channel sum), top-k, remap, norm, and quantization
arithmetic are elementwise/selection work outside the pinned denominator. The tail
buffer (raw keys of the in-progress pool) is O(kpool) per request and excluded, which
keeps the minimum a valid lower bound.

``mla_cache_dtype_bytes`` is a serving choice (``--kv-cache-dtype``), not a checkpoint
fact; the builder follows the config and ``floors.py`` applies the arch's served FP8
cache. The sparse-MLA compute floor follows the cache precision: an FP8 cache runs the
FP8 kernel, a BF16 cache the BF16 one.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core import AttnInteraction, LearnedWeightGroup, MatmulGroup, Workload
from .base import AttentionSemantic


def pooled_prefix_sum(n: int, pool: int) -> int:
    """``sum_{k=1..n} floor(k / pool)`` in O(1)."""
    if n <= 0:
        return 0
    full, remainder = divmod(n, pool)
    return pool * full * (full - 1) // 2 + full * (remainder + 1)


def selected_prefix_sum(n: int, topk: int, pool: int) -> int:
    """``sum_{k=1..n} min(k, topk + k mod pool)`` in O(1).

    ``min(k, topk + k mod pool) == k`` exactly while ``k <= topk + pool - 1`` (at most
    ``topk / pool`` complete pools), and ``topk + k mod pool`` afterwards.
    """
    if n <= 0:
        return 0
    threshold = topk + pool - 1
    dense = min(n, threshold)
    total = dense * (dense + 1) // 2
    if n > threshold:

        def remainder_sum(m: int) -> int:
            return m * (m + 1) // 2 - pool * pooled_prefix_sum(m, pool)

        count = n - threshold
        total += topk * count + remainder_sum(n) - remainder_sum(threshold)
    return total


def selected_keys(k: int, topk: int, pool: int) -> int:
    return min(k, topk + k % pool) if k > 0 else 0


@dataclass
class Glm53KpoolDsaAttention:
    hidden: int
    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    index_kpool: int
    mla_cache_dtype_bytes: float = 2.0
    index_cache_dtype_bytes: float = 1.0
    index_scale_dtype_bytes: float = 4.0
    quant_block_size: int = 128

    split_attention_phases = True

    def __post_init__(self) -> None:
        if self.qk_rope_head_dim != 0:
            raise ValueError("GLM-5.3-Flash MLA is no-rope; qk_rope_head_dim must be 0")
        if self.index_kpool < 1 or self.index_topk % self.index_kpool:
            raise ValueError("index_topk must be a positive multiple of index_kpool")

    @property
    def mla_compute_dtype(self) -> str:
        return "fp8" if self.mla_cache_dtype_bytes == 1.0 else "bf16"

    @property
    def index_entry_bytes(self) -> float:
        blocks = -(-self.index_head_dim // self.quant_block_size)
        return (
            self.index_head_dim * self.index_cache_dtype_bytes
            + blocks * self.index_scale_dtype_bytes
        )

    @property
    def latent_entry_bytes(self) -> float:
        return (self.kv_lora_rank + self.qk_rope_head_dim) * self.mla_cache_dtype_bytes

    def matmul_groups(self) -> list[MatmulGroup]:
        def group(name: str, n: int, k: int, module: str, heads: int = 1) -> MatmulGroup:
            return MatmulGroup(
                name,
                n=n,
                k=k,
                activated_mult=heads,
                total_count=heads,
                bucket="attn_proj",
                module=f"self_attn.{module}",
            )

        qk_head = self.qk_nope_head_dim + self.qk_rope_head_dim
        return [
            group("q_a_proj", self.q_lora_rank, self.hidden, "q_a_proj"),
            group(
                "kv_a_proj",
                self.kv_lora_rank + self.qk_rope_head_dim,
                self.hidden,
                "kv_a_proj_with_mqa",
            ),
            group("q_b_proj", self.num_heads * qk_head, self.q_lora_rank, "q_b_proj"),
            # W_UK / W_UV: the two per-head views of kv_b_proj [H*(nope+v), kv_lora].
            group(
                "q_absorb",
                self.kv_lora_rank,
                self.qk_nope_head_dim,
                "kv_b_proj",
                heads=self.num_heads,
            ),
            group("v_up", self.v_head_dim, self.kv_lora_rank, "kv_b_proj", heads=self.num_heads),
            group("o_proj", self.hidden, self.num_heads * self.v_head_dim, "o_proj"),
            group(
                "indexer.wq_b",
                self.index_n_heads * self.index_head_dim,
                self.q_lora_rank,
                "indexer.wq_b",
            ),
            group("indexer.wk", self.index_head_dim, self.hidden, "indexer.wk"),
            group("indexer.weights_proj", self.index_n_heads, self.hidden, "indexer.weights_proj"),
            group(
                "indexer.kpool_gate",
                self.index_head_dim,
                self.hidden,
                "indexer.index_kpool_compress_gate",
            ),
        ]

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return [
            LearnedWeightGroup("q_a_norm", self.q_lora_rank, 1, "norm"),
            LearnedWeightGroup("kv_a_norm", self.kv_lora_rank, 1, "norm"),
            LearnedWeightGroup(
                "indexer.kpool_ape", self.index_kpool * self.index_head_dim, 1, "attn"
            ),
            # Indexer k_norm is a LayerNorm: weight and bias.
            LearnedWeightGroup("indexer.k_norm", 2 * self.index_head_dim, 1, "norm"),
        ]

    @staticmethod
    def _causal(interaction: AttnInteraction) -> tuple[int, int]:
        if interaction.mask != "causal":
            raise ValueError(
                "GLM-5.3-Flash kpool DSA needs exact per-request causal geometry; "
                f"got a {interaction.mask!r} interaction (collapsed aggregate?)"
            )
        return interaction.num_query, interaction.num_cached_key

    def _phase_work(self, wl: Workload) -> tuple[float, float, float, float]:
        pooled_pairs = selected_pairs = index_reads = latent_reads = 0.0
        pool, topk = self.index_kpool, self.index_topk
        for interaction in wl.attn:
            q, cached = self._causal(interaction)
            weight = interaction.multiplicity
            last = cached + q
            pooled_pairs += weight * (
                pooled_prefix_sum(last, pool) - pooled_prefix_sum(cached, pool)
            )
            selected_pairs += weight * (
                selected_prefix_sum(last, topk, pool) - selected_prefix_sum(cached, topk, pool)
            )
            index_reads += weight * (cached // pool)
            latent_reads += weight * max(selected_keys(cached + 1, topk, pool) - 1, 0)
        return pooled_pairs, selected_pairs, index_reads, latent_reads

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        rows: list[AttentionSemantic] = []
        mla_flops_per_pair = (
            2.0 * self.num_heads * (self.kv_lora_rank + self.qk_rope_head_dim + self.kv_lora_rank)
        )
        for phase, phase_workload in wl.attention_phases():
            suffix = f".{phase}" if phase is not None else ""
            pooled, selected, index_reads, latent_reads = self._phase_work(phase_workload)
            rows.append(
                AttentionSemantic(
                    name=f"indexer{suffix}",
                    flops=2.0 * self.index_n_heads * self.index_head_dim * pooled,
                    bytes=index_reads * self.index_entry_bytes,
                    compute_dtype="fp8",
                )
            )
            rows.append(
                AttentionSemantic(
                    name=f"attn{suffix}",
                    flops=mla_flops_per_pair * selected,
                    bytes=latent_reads * self.latent_entry_bytes,
                    compute_dtype=self.mla_compute_dtype,
                )
            )
        completed_pools = 0.0
        for interaction in wl.attn:
            q, cached = self._causal(interaction)
            completed_pools += interaction.multiplicity * (
                (cached + q) // self.index_kpool - cached // self.index_kpool
            )
        rows.append(
            AttentionSemantic(
                name="mla_cache_append",
                bucket="embedding",
                bytes=wl.matmul_tokens * self.latent_entry_bytes,
            )
        )
        rows.append(
            AttentionSemantic(
                name="index_cache_append",
                bucket="embedding",
                bytes=completed_pools * self.index_entry_bytes,
            )
        )
        return rows

    # The historical AttentionSpec surface; Model.label uses semantic_segments.
    def internal_flops(self, wl: Workload) -> float:
        return sum(row.flops for row in self.semantic_segments(wl) if row.bucket == "attn_internal")

    def kv_bytes(self, wl: Workload) -> float:
        return sum(row.bytes for row in self.semantic_segments(wl) if row.byte_kind == "kv")

    def cache_write_bytes(self, wl: Workload) -> float:
        return sum(row.bytes for row in self.semantic_segments(wl) if row.bucket == "embedding")
