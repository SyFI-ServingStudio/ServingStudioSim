"""Gated DeltaNet linear attention (Qwen3-Next / Qwen3.5 / Qwen3.6 hybrid stacks).

A linear-attention layer replaces the quadratic softmax attention with a delta-rule
recurrence over a **fixed-size** state. Two consequences make it worth modeling as its
own :class:`AttentionSpec`, distinct from :class:`GQA`:

- **Compute is O(T), not O(T²).** There is no (query, key) pair blow-up — each token does
  a constant amount of readout + state update, so ``internal_flops`` scales with
  ``matmul_tokens``, never with the causal-triangle pair count.
- **Memory does not grow with context.** The recurrent state is a fixed
  ``num_v_heads · head_k_dim · head_v_dim`` matrix per sequence (not a per-token KV
  cache), so ``kv_bytes`` scales with the number of sequences in the batch, not with how
  many tokens each has attended. This is the whole point of the hybrid design — the
  linear layers keep long-context memory flat.

Projection layout mirrors HF ``Qwen3_5GatedDeltaNet``: packed ``in_proj_qkvz``, packed
``in_proj_ba`` (per-head β / decay scalars), a depthwise ``conv1d`` short convolution
over the qkv channels, and ``out_proj``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core import LearnedWeightGroup, MatmulGroup, Workload
from .base import AttentionSemantic


@dataclass
class GatedDeltaNet:
    split_attention_phases = False

    hidden: int
    num_v_heads: int  # linear_num_value_heads
    num_k_heads: int  # linear_num_key_heads
    head_k_dim: int  # linear_key_head_dim
    head_v_dim: int  # linear_value_head_dim
    conv_kernel: int  # linear_conv_kernel_dim
    state_dtype_bytes: float  # recurrent state dtype (mamba_ssm_dtype, usually fp32 -> 4)
    activation_dtype_bytes: float = 2.0

    @property
    def key_dim(self) -> int:
        return self.num_k_heads * self.head_k_dim

    @property
    def value_dim(self) -> int:
        return self.num_v_heads * self.head_v_dim

    @property
    def conv_dim(self) -> int:
        # conv1d mixes the concatenated q, k, v channels (q and k share key_dim each).
        return self.key_dim * 2 + self.value_dim

    def matmul_groups(self) -> list[MatmulGroup]:
        # All in the attn_proj bucket (they are the linear-attention block's projections).
        # The depthwise conv is modeled as a MatmulGroup with n=conv_dim, k=conv_kernel:
        # params = conv_dim*conv_kernel and flops = 2·T·conv_dim·conv_kernel both match a
        # groups=conv_dim depthwise Conv1d exactly.
        return [
            MatmulGroup(
                "qkvz",
                n=self.key_dim * 2 + self.value_dim * 2,
                k=self.hidden,
                bucket="attn_proj",
                module="linear_attn.in_proj_qkvz",
            ),
            MatmulGroup(
                "ba",
                n=2 * self.num_v_heads,
                k=self.hidden,
                bucket="attn_proj",
                module="linear_attn.in_proj_ba",
            ),
            MatmulGroup(
                "conv1d",
                n=self.conv_dim,
                k=self.conv_kernel,
                bucket="attn_proj",
                module="linear_attn.conv1d",
            ),
            MatmulGroup(
                "out_proj",
                n=self.hidden,
                k=self.value_dim,
                bucket="attn_proj",
                module="linear_attn.out_proj",
            ),
        ]

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return [
            LearnedWeightGroup("a_log", self.num_v_heads, 1, "attn"),
            LearnedWeightGroup("dt_bias", self.num_v_heads, 1, "attn"),
            LearnedWeightGroup("gated_norm", self.head_v_dim, 1, "norm"),
        ]

    def internal_flops(self, wl: Workload) -> float:
        # O(T) delta-rule recurrence, counted directly off HF's reference step
        # (torch_recurrent_gated_delta_rule): per token, per value head, exactly THREE
        # head_k_dim x head_v_dim products over the [d_k, d_v] state S --
        #   kv_mem = (S * k).sum   -> k^T S              (d_k*d_v MACs)
        #   S     += k (x) (beta*(v - kv_mem))  rank-1 update (d_k*d_v MACs)
        #   out    = (S * q).sum   -> q^T S              (d_k*d_v MACs)
        # => 3 * d_k*d_v MACs = 6 * d_k*d_v FLOPs per (token, head). The decay scale
        # S*g_t and the (v - kv_mem) / *beta steps are elementwise, so they stay out of
        # the denominator (same convention as norm/rope). No (query, key) pair blow-up.
        per_token = 6.0 * self.num_v_heads * self.head_k_dim * self.head_v_dim
        return per_token * wl.matmul_tokens

    def kv_bytes(self, wl: Workload) -> float:
        # The persistent recurrent state (HF caches it as `recurrent_states`, shape
        # [num_v_heads, head_k_dim, head_v_dim] per sequence, in mamba_ssm_dtype) is the
        # KV-cache analogue: read then written back each step -> factor 2. It scales with
        # the number of state transactions (one per original causal_lm interaction),
        # NOT with context length. Analyzer workloads may collapse the geometry of
        # many interactions, so `num_attention_steps` retains their original count.
        return sum(row.bytes for row in self.semantic_segments(wl))

    @property
    def recurrent_state_bytes(self) -> float:
        return self.num_v_heads * self.head_k_dim * self.head_v_dim * self.state_dtype_bytes

    @property
    def convolution_state_bytes(self) -> float:
        return self.conv_dim * self.conv_kernel * self.activation_dtype_bytes

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        phases = dict(wl.attention_phases())
        prefill = phases.get("prefill", Workload(0, 0))
        decode = phases.get("decode", Workload(0, 0))
        # Legacy/caller-constructed workloads may not tag phases. Preserve their
        # former read+write-per-transaction behavior by treating them as stateful
        # recurrent steps, without guessing that any were fresh prefill requests.
        if None in phases and not prefill.num_attention_steps and not decode.num_attention_steps:
            decode = phases[None]
        prefill_requests = prefill.num_attention_steps
        stateful = prefill.prefill_stateful_requests
        decode_requests = decode.num_attention_steps
        recurrent = self.recurrent_state_bytes
        convolution = self.convolution_state_bytes
        return [
            AttentionSemantic("attn.prefill", flops=self.internal_flops(prefill)),
            AttentionSemantic("attn.decode", flops=self.internal_flops(decode)),
            AttentionSemantic("recurrent_state.prefill_read", bytes=recurrent * stateful),
            AttentionSemantic("recurrent_state.prefill_write", bytes=recurrent * prefill_requests),
            AttentionSemantic("recurrent_state.decode_read", bytes=recurrent * decode_requests),
            AttentionSemantic("recurrent_state.decode_write", bytes=recurrent * decode_requests),
            AttentionSemantic("conv_state.prefill_read", bytes=convolution * stateful),
            AttentionSemantic("conv_state.prefill_write", bytes=convolution * prefill_requests),
            AttentionSemantic("conv_state.decode_read", bytes=convolution * decode_requests),
            AttentionSemantic("conv_state.decode_write", bytes=convolution * decode_requests),
        ]

    def cache_write_bytes(self, wl: Workload) -> float:
        # `kv_bytes` already counts every recurrent/convolution-state transaction.
        return 0.0
