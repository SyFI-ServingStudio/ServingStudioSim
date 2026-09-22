"""Plain absorbed multi-head latent attention used by Kimi-K3 full layers."""

from __future__ import annotations

from dataclasses import dataclass

from ..core import MatmulGroup, Workload
from .base import AttentionSemantic


@dataclass
class MLA:
    """DeepSeek-V3-style MLA without a DSA indexer.

    The learned ``kv_b_proj`` matrix is executed in two absorbed views: one
    ``q_nope @ W_UK`` projection before attention and one ``attn_out @ W_UV``
    projection after attention. Their parameter counts partition that matrix.
    """

    split_attention_phases = True

    hidden: int
    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    kv_dtype_bytes: float = 2.0
    output_gate: bool = False

    def matmul_groups(self) -> list[MatmulGroup]:
        groups = [
            MatmulGroup(
                "fused_qkv_a_proj",
                n=self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                k=self.hidden,
                bucket="attn_proj",
                module="self_attn.fused_qkv_a_proj_with_mqa",
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
        if self.output_gate:
            groups.append(
                MatmulGroup(
                    "output_gate",
                    n=self.num_heads * self.v_head_dim,
                    k=self.hidden,
                    bucket="attn_proj",
                    module="self_attn.g_proj",
                )
            )
        return groups

    @property
    def cache_width(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    def internal_flops(self, wl: Workload) -> float:
        pairs = sum(interaction.pairs() for interaction in wl.attn)
        # Per pair: latent+rope QK and latent V accumulation. The absorbed
        # W_UK/W_UV products are represented by q_absorb/v_up matmul groups.
        return (
            2.0
            * self.num_heads
            * (self.kv_lora_rank + self.qk_rope_head_dim + self.kv_lora_rank)
            * pairs
        )

    def kv_bytes(self, wl: Workload) -> float:
        per_cached_token = self.cache_width * self.kv_dtype_bytes
        return per_cached_token * sum(
            interaction.num_cached_key * interaction.multiplicity
            for interaction in wl.attn
        )

    def cache_write_bytes(self, wl: Workload) -> float:
        return wl.matmul_tokens * self.cache_width * self.kv_dtype_bytes

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        rows: list[AttentionSemantic] = []
        for phase, phase_workload in wl.attention_phases():
            suffix = f".{phase}" if phase is not None else ""
            rows.append(
                AttentionSemantic(
                    name=f"attn{suffix}",
                    flops=self.internal_flops(phase_workload),
                    bytes=self.kv_bytes(phase_workload),
                )
            )
        if wl.matmul_tokens > 0:
            rows.append(
                AttentionSemantic(
                    name="mla_cache_append",
                    bucket="embedding",
                    flops=0.0,
                    bytes=self.cache_write_bytes(wl),
                )
            )
        return rows
