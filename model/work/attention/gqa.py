"""Grouped-query attention (covers MHA / MQA / GQA — they differ only in num_kv_heads).

MHA: num_kv_heads == num_qo_heads. MQA: num_kv_heads == 1. GQA: in between. All three
share the same QKV/O projections and the same internal math; only the KV width changes.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core import MatmulGroup, Workload


@dataclass
class GQA:
    split_attention_phases = True

    hidden: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    kv_dtype_bytes: float
    output_gate: bool = False  # Qwen3.5/3.6 "gated attention": q_proj also emits a gate
    #: A recipe that runs the prefill (FA3) forward on the FP8 tensor cores (q and
    #: kv both FP8) must label its necessary attention math at the FP8 peak too,
    #: otherwise the necessary floor lands above the measured hardware limit and all
    #: mechanical (quant/norm/…) overhead masquerades as zero excess. ``None`` keeps
    #: the master dtype (decode QK stays BF16 in the FP8 recipe, so decode is not
    #: covered by this pin). Builders that serve an FP8 prefill set this to fp8.
    prefill_compute_dtype: str | None = None

    def phase_compute_dtype(self, phase: str | None) -> str | None:
        """Mechanism-fixed precision for one attention phase, if the recipe pins it."""
        if phase == "prefill":
            return self.prefill_compute_dtype
        return None

    @property
    def attn_dim(self) -> int:
        return self.num_qo_heads * self.head_dim

    def matmul_groups(self) -> list[MatmulGroup]:
        # Fused QKV: hidden -> (num_qo + 2*num_kv) * head_dim. O: attn_dim -> hidden.
        # With output_gate the q side is doubled (q_proj -> num_qo*head_dim*2): the extra
        # half is the sigmoid gate multiplied onto the attention output. The gate matmul
        # is real work (counted here); the elementwise multiply is out of the denominator.
        qkv_out = (self.num_qo_heads + 2 * self.num_kv_heads) * self.head_dim
        if self.output_gate:
            qkv_out += self.num_qo_heads * self.head_dim
        return [
            MatmulGroup(
                "qkv",
                n=qkv_out,
                k=self.hidden,
                bucket="attn_proj",
                module="self_attn.qkv_proj",
            ),
            MatmulGroup(
                "o",
                n=self.hidden,
                k=self.attn_dim,
                bucket="attn_proj",
                module="self_attn.o_proj",
            ),
        ]

    def internal_flops(self, wl: Workload) -> float:
        # QK^T and softmax·V are each 2·head_dim MACs per (query, key) pair per query
        # head -> 4·num_qo_heads·head_dim·pairs. Every query head computes (GQA only
        # shares KV storage, not the score/context matmuls).
        per_pair = 4.0 * self.num_qo_heads * self.head_dim
        return per_pair * sum(interaction.pairs() for interaction in wl.attn)

    def kv_bytes(self, wl: Workload) -> float:
        # Only cached keys are HBM reads; K and V both -> factor 2.
        per_cached_token = 2.0 * self.num_kv_heads * self.head_dim * self.kv_dtype_bytes
        return per_cached_token * sum(interaction.num_cached_key for interaction in wl.attn)

    def cache_write_bytes(self, wl: Workload) -> float:
        """Compulsory persistent K/V writes for every newly processed token."""
        per_new_token = 2.0 * self.num_kv_heads * self.head_dim * self.kv_dtype_bytes
        return per_new_token * wl.matmul_tokens
