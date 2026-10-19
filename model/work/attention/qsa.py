"""Query-sparse attention (QSA): gated GQA over keys a weight-free group indexer selects.

Mechanism (vLLM fork ``models/qwen4_exp/nvidia/{qsa,indexer_qsa}.py`` and
``ops/qsa_indexer.py``; Qwen3.8-Flash-Next's full-attention layers):

- ``qkv_proj`` yields ``H`` query heads, ``H`` sigmoid output-gate heads and ``KV``
  heads each of K and V (``head_dim`` wide); ``o_proj`` maps ``H * head_dim`` back.
- The indexer's ``index_qk_proj`` yields ``index_heads`` query heads and one key head
  (``index_head_dim`` wide). Every ``ratio`` consecutive tokens' keys are averaged
  (no learned weights), normalized, roped and stored as one compressed entry once
  their group is complete. A query at context ``k`` (itself included) scores the
  ``floor(k / ratio)`` complete groups: ``sum_h relu(q_h . K)``, so
  ``2 * index_heads * index_head_dim`` FLOPs per (query, group).
- Top-k keeps ``budget / ratio`` groups and expands them to tokens; the
  ``k mod ratio`` tokens of the query's open group are always attended. A query
  therefore attends ``min(k, budget + k mod ratio)`` keys: all of them while at most
  ``budget / ratio`` groups are complete.
- Sparse GQA: ``Q K^T`` and ``P V`` cost ``4 * H * head_dim`` FLOPs per (query, key).

That selection rule is exactly GLM-5.3's kpool one (pool = ratio, top-k = budget),
so the geometry is the shared :class:`.pooled_topk.PooledTopkGeometry`.

Semantic rows (per phase: ``indexer``, ``attn``; per iteration: two cache writes):

- ``indexer``: the scoring FLOPs; bytes are every cached compressed entry the
  chunk's first query scores, read once. FP8 math when the index cache is FP8
  (the logits kernel dots FP8 x FP8), else BF16.
- ``attn``: the sparse-GQA FLOPs; bytes are the cached K/V the chunk's first query
  selects other than itself. Later queries may add keys, so this is a lower bound
  on distinct reads. The kernel dequantizes an FP8 cache and computes in BF16, so
  the row makes no FP8 claim.
- ``kv_cache_append``: every new token's K and V.
- ``index_cache_append``: one compressed entry per group the batch completes.

Mean pooling, norms, RoPE, ReLU, top-k, expansion, softmax and the output gate are
elementwise/selection work outside the pinned denominator. The raw-key buffer of
each request's open group is O(ratio) and could be replaced by one running sum, so
it is excluded, which keeps the minimum a valid lower bound.

Every pass, including each recurrent MTP draft step, runs its own indexer: the
checkpoint has no ``index_share_for_mtp_iteration``, so no pass reuses another
pass's selection.

Cache dtypes are serving choices (``--kv-cache-dtype``, the indexer KV dtype), not
checkpoint facts: the builder uses the master dtype and the floors layer serves
what the deployment ran.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core import LearnedWeightGroup, MatmulGroup, Workload
from .base import AttentionSemantic
from .pooled_topk import PhaseWork, PooledTopkGeometry


@dataclass
class QuerySparseAttention(PooledTopkGeometry):
    hidden: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    index_n_heads: int
    index_head_dim: int
    #: Tokens the indexer selects (``indexer_budget``); a multiple of the ratio.
    budget: int
    #: Tokens averaged into one compressed index entry (``indexer_compress_ratio``).
    ratio: int
    kv_dtype_bytes: float = 2.0
    index_cache_dtype_bytes: float = 2.0
    #: Prepended to every checkpoint path (``mtp`` for the MTP layer).
    module_prefix: str = ""

    mechanism = "QSA"

    def __post_init__(self) -> None:
        if self.ratio < 1 or self.budget % self.ratio:
            raise ValueError("QSA budget must be a positive multiple of the compress ratio")

    @property
    def pool(self) -> int:
        return self.ratio

    @property
    def topk(self) -> int:
        return self.budget

    def _module(self, path: str) -> str:
        return f"{self.module_prefix}.{path}" if self.module_prefix else path

    @property
    def kv_entry_bytes(self) -> float:
        return 2 * self.num_kv_heads * self.head_dim * self.kv_dtype_bytes

    @property
    def index_entry_bytes(self) -> float:
        return self.index_head_dim * self.index_cache_dtype_bytes

    def matmul_groups(self) -> list[MatmulGroup]:
        q_and_gate = 2 * self.num_heads * self.head_dim
        return [
            MatmulGroup(
                "qkv_proj",
                n=q_and_gate + 2 * self.num_kv_heads * self.head_dim,
                k=self.hidden,
                bucket="attn_proj",
                module=self._module("self_attn.qkv_proj"),
            ),
            MatmulGroup(
                "o_proj",
                n=self.hidden,
                k=self.num_heads * self.head_dim,
                bucket="attn_proj",
                module=self._module("self_attn.o_proj"),
            ),
            # One indexer key head next to the query heads.
            MatmulGroup(
                "indexer.index_qk_proj",
                n=(self.index_n_heads + 1) * self.index_head_dim,
                k=self.hidden,
                bucket="attn_proj",
                module=self._module("self_attn.indexer.index_qk_proj"),
            ),
        ]

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return [
            LearnedWeightGroup("q_norm", self.head_dim, 1),
            LearnedWeightGroup("k_norm", self.head_dim, 1),
            LearnedWeightGroup("indexer.q_norm", self.index_head_dim, 1),
            LearnedWeightGroup("indexer.k_norm", self.index_head_dim, 1),
        ]

    def _rows(
        self, wl: Workload, phase_work: list[tuple[str | None, PhaseWork]]
    ) -> list[AttentionSemantic]:
        rows: list[AttentionSemantic] = []
        pair_flops = 4.0 * self.num_heads * self.head_dim
        group_flops = 2.0 * self.index_n_heads * self.index_head_dim
        index_dtype = "fp8" if self.index_cache_dtype_bytes == 1.0 else None
        completed_groups = 0.0
        for phase, (scored, selected, index_reads, kv_reads, completed) in phase_work:
            suffix = f".{phase}" if phase is not None else ""
            completed_groups += completed
            rows.append(
                AttentionSemantic(
                    name=f"indexer{suffix}",
                    flops=group_flops * scored,
                    bytes=index_reads * self.index_entry_bytes,
                    compute_dtype=index_dtype,
                )
            )
            rows.append(
                AttentionSemantic(
                    name=f"attn{suffix}",
                    flops=pair_flops * selected,
                    bytes=kv_reads * self.kv_entry_bytes,
                )
            )
        rows.append(
            AttentionSemantic(
                name="kv_cache_append",
                bucket="embedding",
                bytes=wl.matmul_tokens * self.kv_entry_bytes,
            )
        )
        rows.append(
            AttentionSemantic(
                name="index_cache_append",
                bucket="embedding",
                bytes=completed_groups * self.index_entry_bytes,
            )
        )
        return rows
