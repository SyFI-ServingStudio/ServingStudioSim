"""Independent GLM-5.2 MLA/DSA necessary-work accountant.

The compressed MLA path is not ordinary GQA: the model projects a query latent,
absorbs the no-PE key projection into a per-head 512-wide query, attends over the
selected compressed latent/rope cache, and then performs the per-head value-up
projection. Full-index layers additionally build the 32-head DSA index and read/
write its FP8-plus-scale cache; index-share layers reuse those indices.

The FP8 index cache is an attention implementation detail of the mechanism, not
of the checkpoint: it is FP8 (and its logits run on the FP8 tensor cores) even
when the weights are BF16. The MLA latent/rope cache and the FlashMLA kernel
that reads it stay BF16 under every checkpoint precision. Whether the learned
weights themselves are FP8 comes from the config's ``quantization_config`` and is
matched per ``MatmulGroup.module``. Index scale values are FP32 semantics, so
their bytes are accounted in the index-cache rows rather than silently treated as
BF16.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core import MatmulGroup, Workload
from .base import AttentionSemantic


@dataclass
class Glm52DsaAttention:
    """One GLM-5.2 attention layer, either full-index or index-share."""

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
    full_index: bool
    weight_dtype_bytes: float = 2.0
    mla_cache_dtype_bytes: float = 2.0
    index_cache_dtype_bytes: float = 1.0
    index_scale_dtype_bytes: float = 4.0
    quant_block_size: int = 128

    split_attention_phases = True

    def matmul_groups(self) -> list[MatmulGroup]:
        # Fused q/kv-A, q-B, and output projection are the learned MLA matrices.
        # W_UK and W_UV are the two views of kv_b_proj; they are executed as the
        # q_absorb and v_up BMMs, and their dimensions partition the same weight.
        groups = [
            MatmulGroup(
                "fused_qkv_a_proj",
                n=self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                k=self.hidden,
                bucket="attn_proj",
                module="self_attn.q_a_proj",
            ),
            MatmulGroup(
                "q_b_proj",
                n=self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim),
                k=self.q_lora_rank,
                bucket="attn_proj",
                module="self_attn.q_b_proj",
            ),
            MatmulGroup(
                "q_absorb",
                n=self.kv_lora_rank,
                k=self.qk_nope_head_dim,
                activated_mult=self.num_heads,
                total_count=self.num_heads,
                bucket="attn_proj",
                module="self_attn.kv_b_proj",
            ),
            MatmulGroup(
                "v_up",
                n=self.v_head_dim,
                k=self.kv_lora_rank,
                activated_mult=self.num_heads,
                total_count=self.num_heads,
                bucket="attn_proj",
                module="self_attn.kv_b_proj",
            ),
            MatmulGroup(
                "o_proj",
                n=self.hidden,
                k=self.num_heads * self.v_head_dim,
                bucket="attn_proj",
                module="self_attn.o_proj",
            ),
        ]
        if self.full_index:
            groups.extend(
                [
                    MatmulGroup(
                        "indexer.q_proj",
                        n=self.index_n_heads * self.index_head_dim,
                        k=self.q_lora_rank,
                        bucket="attn_proj",
                        module="self_attn.indexer.wq_b",
                    ),
                    MatmulGroup(
                        "indexer.wk",
                        n=self.index_head_dim,
                        k=self.hidden,
                        bucket="attn_proj",
                        module="self_attn.indexer.wk",
                    ),
                    # `indexers_proj` is the head-weight matrix and sits in the FP8
                    # checkpoint's modules_to_not_convert, so it stays at the
                    # master dtype even in a quantized repo.
                    MatmulGroup(
                        "indexer.weights_proj",
                        n=self.index_n_heads,
                        k=self.hidden,
                        bucket="attn_proj",
                        module="self_attn.indexers_proj",
                    ),
                ]
            )
        return groups

    def _selected_pairs(self, workload: Workload) -> float:
        selected = self.index_topk
        total = 0.0
        for interaction in workload.attn:
            if interaction.mask == "causal":
                total += sum(
                    min(selected, interaction.num_cached_key + query + 1)
                    for query in range(interaction.num_query)
                )
            elif interaction.mask in ("full", "cross"):
                total += interaction.num_query * min(selected, interaction.num_key)
            else:
                raise ValueError(f"unknown attention mask {interaction.mask!r}")
        return float(total)

    def _index_cache_row_bytes(self, workload: Workload) -> float:
        # One FP8 index vector plus one FP32 scale per 128-element quant block.
        per_token = self.index_head_dim * self.index_cache_dtype_bytes
        per_token += (
            (self.index_head_dim + self.quant_block_size - 1) // self.quant_block_size
        ) * self.index_scale_dtype_bytes
        return per_token * sum(interaction.num_cached_key for interaction in workload.attn)

    def _mla_cache_read_bytes(self, workload: Workload) -> float:
        per_selected = (self.kv_lora_rank + self.qk_rope_head_dim) * self.mla_cache_dtype_bytes
        total = 0.0
        for interaction in workload.attn:
            if interaction.mask == "causal":
                total += sum(
                    min(self.index_topk, interaction.num_cached_key + query + 1)
                    for query in range(interaction.num_query)
                ) * per_selected
            elif interaction.mask in ("full", "cross"):
                total += (
                    interaction.num_query
                    * min(self.index_topk, interaction.num_key)
                    * per_selected
                )
            else:
                raise ValueError(f"unknown attention mask {interaction.mask!r}")
        return float(total)

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        """Return indexer, sparse-attention, and persistent-cache semantic rows.

        Indexer logits use the exact causal pair count. Sparse MLA uses the
        selected-key count, capped by the available causal context. Norm, rope,
        quantization, top-k, and index remapping arithmetic remain outside the
        pinned FLOP denominator under the shared model.work convention.
        """

        rows: list[AttentionSemantic] = []
        for phase, phase_workload in wl.attention_phases():
            suffix = f".{phase}" if phase is not None else ""
            if self.full_index:
                index_pairs = sum(interaction.pairs() for interaction in phase_workload.attn)
                index_cache_bytes = self._index_cache_row_bytes(phase_workload)
                rows.append(
                    AttentionSemantic(
                        name=f"indexer{suffix}",
                        flops=2.0
                        * self.index_n_heads
                        * self.index_head_dim
                        * index_pairs,
                        bytes=index_cache_bytes,
                        # The index cache is FP8 by construction (see
                        # index_cache_dtype_bytes), so the logits run on the FP8
                        # tensor cores regardless of how the weights were stored.
                        compute_dtype="fp8",
                    )
                )

            selected_pairs = self._selected_pairs(phase_workload)
            rows.append(
                AttentionSemantic(
                    name=f"attn{suffix}",
                    flops=2.0
                    * self.num_heads
                    * (self.kv_lora_rank + self.qk_rope_head_dim + self.kv_lora_rank)
                    * selected_pairs,
                    bytes=self._mla_cache_read_bytes(phase_workload),
                    # The MLA latent/rope cache is BF16, and so is the FlashMLA
                    # kernel that reads it — an FP8 checkpoint does not move this
                    # row onto the FP8 tensor cores.
                    compute_dtype="bf16",
                )
            )

        # The two cache writes are compulsory state traffic and are separate
        # leaves in the unified DSA CostTree.
        if wl.matmul_tokens > 0:
            rows.append(
                AttentionSemantic(
                    name="mla_cache_append",
                    bucket="embedding",
                    flops=0.0,
                    bytes=wl.matmul_tokens
                    * (self.kv_lora_rank + self.qk_rope_head_dim)
                    * self.mla_cache_dtype_bytes,
                )
            )
            if self.full_index:
                per_index_token = self.index_head_dim * self.index_cache_dtype_bytes
                per_index_token += (
                    (self.index_head_dim + self.quant_block_size - 1) // self.quant_block_size
                ) * self.index_scale_dtype_bytes
                rows.append(
                    AttentionSemantic(
                        name="index_cache_append",
                        bucket="embedding",
                        flops=0.0,
                        bytes=wl.matmul_tokens * per_index_token,
                    )
                )
        return rows

    # These methods keep the historical AttentionSpec contract usable for callers
    # that inspect an attention object directly; Model.label uses semantic_segments.
    def internal_flops(self, wl: Workload) -> float:
        return sum(row.flops for row in self.semantic_segments(wl) if row.bucket == "attn_internal")

    def kv_bytes(self, wl: Workload) -> float:
        return sum(row.bytes for row in self.semantic_segments(wl) if row.byte_kind == "kv")

    def cache_write_bytes(self, wl: Workload) -> float:
        per_token = (self.kv_lora_rank + self.qk_rope_head_dim) * self.mla_cache_dtype_bytes
        if self.full_index:
            per_token += self.index_head_dim * self.index_cache_dtype_bytes
            per_token += (
                (self.index_head_dim + self.quant_block_size - 1) // self.quant_block_size
            ) * self.index_scale_dtype_bytes
        return wl.matmul_tokens * per_token
