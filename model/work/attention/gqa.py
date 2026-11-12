"""Grouped-query attention (covers MHA / MQA / GQA — they differ only in num_kv_heads).

MHA: num_kv_heads == num_qo_heads. MQA: num_kv_heads == 1. GQA: in between. All three
share the same QKV/O projections and the same internal math; only the KV width changes.

Two generic options cover checkpoints that depart from the symmetric layout:

- ``v_head_dim`` — the value (and therefore context/output) head width when it differs
  from the query/key width (MiMo-V2: QK 192, V 128). QK^T costs ``2*d_qk`` and P·V
  ``2*d_v`` FLOPs per (query head, key) pair, and the cache stores ``d_qk + d_v`` per
  KV head and token.
- ``attention_sinks`` — one learned logit per query head that joins each softmax
  denominator (gpt-oss / MiMo-V2 ``attention_sink_bias``). It adds no key, no value and
  no matmul, so its only necessary work is reading the per-head vector once per layer.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core import LearnedWeightGroup, MatmulGroup, Workload


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
    #: Value head width; ``None`` means the symmetric ``head_dim``.
    v_head_dim: int | None = None
    #: Learned per-query-head softmax sink logits, stored at the master dtype.
    attention_sinks: bool = False

    def phase_compute_dtype(self, phase: str | None) -> str | None:
        """Mechanism-fixed precision for one attention phase, if the recipe pins it."""
        if phase == "prefill":
            return self.prefill_compute_dtype
        return None

    @property
    def v_dim(self) -> int:
        return self.head_dim if self.v_head_dim is None else self.v_head_dim

    @property
    def attn_dim(self) -> int:
        """Width of the concatenated per-head attention output, the O input."""
        return self.num_qo_heads * self.v_dim

    @property
    def kv_entry_bytes(self) -> float:
        """Cache bytes of one token at one layer: K and V of every KV head."""
        return self.num_kv_heads * (self.head_dim + self.v_dim) * self.kv_dtype_bytes

    def matmul_groups(self) -> list[MatmulGroup]:
        # Fused QKV: hidden -> num_qo*head_dim + num_kv*(head_dim + v_dim). O: attn_dim -> hidden.
        # With output_gate q_proj also emits one sigmoid gate per attention-output element
        # (attn_dim), multiplied onto the attention output. The gate matmul
        # is real work (counted here); the elementwise multiply is out of the denominator.
        qkv_out = (self.num_qo_heads + self.num_kv_heads) * self.head_dim
        qkv_out += self.num_kv_heads * self.v_dim
        if self.output_gate:
            qkv_out += self.attn_dim
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

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        if not self.attention_sinks:
            return []
        return [LearnedWeightGroup("attention_sinks", self.num_qo_heads, 1, "attn")]

    @property
    def flops_per_pair(self) -> float:
        # QK^T is head_dim MACs and softmax·V is v_dim MACs per (query, key) pair per
        # query head -> 2·num_qo_heads·(head_dim + v_dim). Every query head computes
        # (GQA only shares KV storage, not the score/context matmuls).
        return 2.0 * self.num_qo_heads * (self.head_dim + self.v_dim)

    def internal_flops(self, wl: Workload) -> float:
        return self.flops_per_pair * sum(interaction.pairs() for interaction in wl.attn)

    def kv_bytes(self, wl: Workload) -> float:
        # Only cached keys are HBM reads, K and V both.
        return self.kv_entry_bytes * sum(
            interaction.num_cached_key * interaction.multiplicity for interaction in wl.attn
        )

    def cache_write_bytes(self, wl: Workload) -> float:
        """Compulsory persistent K/V writes for every newly processed token."""
        return self.kv_entry_bytes * wl.matmul_tokens
