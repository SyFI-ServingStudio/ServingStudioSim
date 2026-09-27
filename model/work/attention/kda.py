"""Kimi Delta Attention (KDA), GLM-5.3-Flash's linear-attention layer.

KDA is a gated delta rule with a *per-channel* decay: per head,

    S_t = (I - beta_t k_t k_t^T) Diag(alpha_t) S_{t-1} + beta_t k_t v_t^T,   o_t = q_t^T S_t

(vLLM fork ``glm5next/nvidia/kda.py`` -> ``fused_recurrent_kda`` /
``chunk_kda_with_fused_gate``). The only difference from Gated DeltaNet's recurrence is
that the decay ``Diag(alpha_t)`` is a vector over the key channels rather than one
scalar per head; applying it is still elementwise over the ``d_k x d_v`` state. The
non-elementwise work is therefore GDN's: per token and head, ``k^T S`` (read-out for the
delta), the rank-1 update, and ``q^T S`` — ``6 * d_k * d_v`` FLOPs, O(T) in sequence
length. So this spec inherits :class:`GatedDeltaNet`'s recurrence and state rows.

**Necessary-work formulation.** The recurrent form above is the minimum for both prefill
and decode. The chunked prefill kernel (intra-chunk ``K K^T``, triangular solve, WY
recompute) is an algorithmic choice that does *more* FLOPs to gain parallelism; its
excess is redundancy for R7 to expose, not necessary work.

Projection layout follows the checkpoint (all BF16 — every KDA module is in the FP8
config's ``modules_to_not_convert``): separate ``q/k/v_proj``, the per-head ``b_proj``
(beta), the low-rank per-channel decay gate ``f_a -> f_b``, the low-rank output gate
``g_a -> g_b``, three depthwise short convolutions, and ``o_proj``. ``A_log`` (per head)
and ``dt_bias`` (per channel) are learned vectors; ``o_norm`` is the gated RMSNorm.

The persistent state per sequence is the fp32 ``[H, d_k, d_v]`` recurrent matrix plus the
short-convolution history. That history is the last ``kernel - 1`` inputs of each q/k/v
channel (vLLM's ``kda_state_shape`` allocates exactly ``conv_kernel_size - 1`` columns
without speculation), not ``kernel`` columns.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core import LearnedWeightGroup, MatmulGroup
from .linear import GatedDeltaNet


@dataclass
class KimiDeltaAttention(GatedDeltaNet):
    """One KDA layer. ``num_k_heads == num_v_heads`` and ``head_k_dim == head_v_dim``."""

    gate_rank: int = 128  # f_a / g_a low-rank width (the checkpoint's head_dim)

    @property
    def projection(self) -> int:
        return self.num_v_heads * self.head_v_dim

    def matmul_groups(self) -> list[MatmulGroup]:
        # A depthwise Conv1d is a MatmulGroup with n=channels, k=kernel: its params
        # and 2*T*channels*kernel FLOPs match the grouped convolution exactly.
        def group(name: str, n: int, k: int) -> MatmulGroup:
            return MatmulGroup(name, n=n, k=k, bucket="attn_proj", module=f"self_attn.{name}")

        width = self.projection
        return [
            group("q_proj", width, self.hidden),
            group("k_proj", width, self.hidden),
            group("v_proj", width, self.hidden),
            group("b_proj", self.num_v_heads, self.hidden),
            group("f_a_proj", self.gate_rank, self.hidden),
            group("f_b_proj", width, self.gate_rank),
            group("g_a_proj", self.gate_rank, self.hidden),
            group("g_b_proj", width, self.gate_rank),
            group("q_conv1d", width, self.conv_kernel),
            group("k_conv1d", width, self.conv_kernel),
            group("v_conv1d", width, self.conv_kernel),
            group("o_proj", self.hidden, width),
        ]

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return [
            LearnedWeightGroup("a_log", self.num_v_heads, 1, "attn"),
            LearnedWeightGroup("dt_bias", self.projection, 1, "attn"),
            LearnedWeightGroup("o_norm", self.head_v_dim, 1, "norm"),
        ]

    @property
    def convolution_state_bytes(self) -> float:
        return self.conv_dim * (self.conv_kernel - 1) * self.activation_dtype_bytes
