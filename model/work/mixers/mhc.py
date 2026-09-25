"""Manifold-constrained hyper-connections (mHC), GLM-5.3-Flash's residual stream.

The residual is ``n = hc_mult`` parallel streams ``X in R^{n x d}`` instead of one
vector. Around every sublayer (attention, then FFN) of every layer the vLLM fork
(``glm5next/nvidia/model.py`` ``Glm5NextDecoderLayer``) runs:

- **pre**: ``mix = RMS(vec X) @ hc_fn^T`` with the learned ``hc_fn`` of shape
  ``[(2 + n) n, n d]`` gives ``n`` pre weights, ``n`` post weights, and an ``n x n``
  residual matrix (the latter Sinkhorn-normalized toward doubly stochastic); the
  sublayer input is ``sum_i pre_i X_i``, then the sublayer's RMSNorm;
- **post**: ``X'_j = sum_i comb_{j,i} X_i + post_j * out``.

The fork fuses layer ``L``'s FFN post with layer ``L+1``'s attention pre. This module
states the same structure semantically: a *boundary* between two consecutive
sublayers performs one post (merging the finished sublayer) and one pre (reading for
the next one). A layer therefore has an attention boundary (post of the previous
layer's FFN + attention pre; pre only in the first layer, which follows the
``hc_expand`` of the embedding) and an FFN boundary (attention post + FFN pre). The
last layer's FFN post has no following pre and is :class:`MhcFinalPost`.

Necessary FLOPs per token:

- ``hc_fn`` projection: ``2 (2 + n) n (n d)`` — a weight matmul (``MatmulGroup``);
- pre contraction ``sum_i pre_i X_i``: ``2 n d``;
- post residual mix ``comb @ X``: ``2 n^2 d``; post-weighted add of the output: ``2 n d``.

Sinkhorn iterations, the sigmoid gates, scaling, and the RMS normalization are
elementwise normalization work, outside the pinned denominator (model.work
convention 4) — about 1.1 kFLOP per token per boundary for 20 iterations of a 4x4
matrix, <0.2% of a boundary's pinned work. The ``n``-stream activations are fusible
intermediates. Learned ``hc_*_fn`` / ``hc_*_base`` / ``hc_*_scale`` and the two RMSNorm
scales are compulsory weight reads.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..attention.base import AttentionSemantic
from ..core import LearnedWeightGroup, MatmulGroup, Workload


def _post_flops(streams: int, hidden: int) -> float:
    return 2.0 * streams * streams * hidden + 2.0 * streams * hidden


def _pre_contraction_flops(streams: int, hidden: int) -> float:
    return 2.0 * streams * hidden


@dataclass
class ManifoldHyperConnections:
    """The two mHC boundaries of one decoder layer."""

    hidden: int
    streams: int
    #: The model's first layer: its attention boundary has no incoming post.
    first_layer: bool = False

    @property
    def mix_width(self) -> int:
        return (2 + self.streams) * self.streams

    def matmul_groups(self) -> list[MatmulGroup]:
        return [
            MatmulGroup(
                f"mhc.{side}_fn",
                n=self.mix_width,
                k=self.streams * self.hidden,
                bucket="residual_mix",
                module=f"hc_{side}_fn",
            )
            for side in ("attn", "ffn")
        ]

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return [
            LearnedWeightGroup("mhc.attn_base", self.mix_width, 1, "residual_mix"),
            LearnedWeightGroup("mhc.attn_scale", 3, 1, "residual_mix"),
            LearnedWeightGroup("mhc.ffn_base", self.mix_width, 1, "residual_mix"),
            LearnedWeightGroup("mhc.ffn_scale", 3, 1, "residual_mix"),
            # The sublayer RMSNorms run inside the pre kernels.
            LearnedWeightGroup("input_norm", self.hidden, 1, "norm"),
            LearnedWeightGroup("post_norm", self.hidden, 1, "norm"),
        ]

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        tokens = wl.matmul_tokens
        pre = _pre_contraction_flops(self.streams, self.hidden)
        post = _post_flops(self.streams, self.hidden)
        attn_boundary = pre if self.first_layer else pre + post
        return [
            AttentionSemantic(
                "mhc.attn_boundary", bucket="residual_mix", flops=tokens * attn_boundary
            ),
            AttentionSemantic(
                "mhc.ffn_boundary", bucket="residual_mix", flops=tokens * (pre + post)
            ),
        ]


@dataclass
class MhcFinalPost:
    """The last layer's FFN post, which has no following pre to fuse with."""

    hidden: int
    streams: int

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        return [
            AttentionSemantic(
                "mhc.final_post",
                bucket="residual_mix",
                flops=wl.matmul_tokens * _post_flops(self.streams, self.hidden),
            )
        ]
