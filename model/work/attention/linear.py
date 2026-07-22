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

Projection layout mirrors HF ``Qwen3_5GatedDeltaNet`` (the split in_proj variant):
``in_proj_qkv`` (q+k+v mixed), separate ``in_proj_z`` (output gate), tiny ``in_proj_b`` /
``in_proj_a`` (per-head β / decay scalars), a depthwise ``conv1d`` short convolution over
the qkv channels, and ``out_proj``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core import MatmulGroup, Workload


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
                "in_proj_qkv",
                n=self.key_dim * 2 + self.value_dim,
                k=self.hidden,
                bucket="attn_proj",
            ),
            MatmulGroup("in_proj_z", n=self.value_dim, k=self.hidden, bucket="attn_proj"),
            MatmulGroup("in_proj_b", n=self.num_v_heads, k=self.hidden, bucket="attn_proj"),
            MatmulGroup("in_proj_a", n=self.num_v_heads, k=self.hidden, bucket="attn_proj"),
            MatmulGroup("conv1d", n=self.conv_dim, k=self.conv_kernel, bucket="attn_proj"),
            MatmulGroup("out_proj", n=self.hidden, k=self.value_dim, bucket="attn_proj"),
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
        state_elems = self.num_v_heads * self.head_k_dim * self.head_v_dim
        state_per_sequence = state_elems * self.state_dtype_bytes
        return 2.0 * wl.num_attention_steps * state_per_sequence

    def cache_write_bytes(self, wl: Workload) -> float:
        # `kv_bytes` already counts both the recurrent-state read and write.
        return 0.0
