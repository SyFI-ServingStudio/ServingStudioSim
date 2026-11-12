"""Gated low-rank hyper-connections (``GatedResidual``), a multi-stream residual.

The residual is ``S = hc_count`` streams of ``hidden`` each. Around every sublayer
(attention, then FFN) one module runs (vLLM fork
``models/qwen4_exp/{common,nvidia}/hyperconnection.py``):

- **mix**: ``xn = RMSNorm_per_stream(X)`` (one learned ``S * hidden`` scale), then
  ``gate = sigmoid(up(silu(down(xn) / S)))`` through a rank-``lowrank`` bottleneck
  (``down: S*hidden -> lowrank``, ``up: lowrank -> S*hidden``), and the sublayer input
  is the gated stream mean ``sum_s gate_s * xn_s / S``;
- **inject**: the same ``xn`` also feeds ``block_inject_weight`` (``S*hidden -> S``),
  whose ``2 * sigmoid(. / S)`` weights the sublayer output into each stream at the
  next **combine**: ``X'_s = X_s + w_s * out``.

The fork fuses each combine with the following module's norm; a combine consumes
the injection of the module that fed the finished sublayer. The final mixer has no
inject projection: it combines the last FFN output and mixes the streams into the
single sampled stream. That mixed stream feeds only ``lm_head`` at the sampled
positions; the MTP drafter reads the combined streams (the fork's
``_mtp_hidden_buffer``, and ``multi_hidden`` between draft steps). So a final
mixer's combine is necessary at every token and its down/up/mix only at
``head_positions``. This is not GLM's Sinkhorn mHC (:mod:`.mhc`).

Necessary FLOPs per token, in the same convention as :mod:`.mhc`:

- ``down`` / ``inject`` / ``up``: weight matmuls (``MatmulGroup``);
- the gated stream mean: a contraction over the stream axis, ``2 * S * hidden``;
- an injected combine (outer product of ``S`` weights with the output, added to the
  streams): ``2 * S * hidden``. A unit-weight combine (no injection) is a plain
  residual add, elementwise, so it carries no pinned FLOPs.

The norms, SiLU, sigmoids and scalings are elementwise (convention 4); the stream
activations are fusible intermediates; the learned norm scales are compulsory reads.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..attention.base import AttentionSemantic
from ..core import LearnedWeightGroup, MatmulGroup, Workload

#: How a module's combine merges the pending sublayer output into the streams.
COMBINES = ("none", "unit", "injected")


@dataclass(frozen=True)
class GatedHcModule:
    """One ``GatedResidual`` module: its combine (if any) and its mix."""

    #: Checkpoint path of the module, layer-relative (``attn_hyper_connection``).
    path: str
    #: Semantic row prefix (``hc.attn``).
    tag: str
    hidden: int
    streams: int
    lowrank: int
    #: Holds ``block_inject_weight`` (every decoder-layer module; not a final mixer).
    inject: bool
    #: The pending combine this module consumes before mixing (:data:`COMBINES`).
    combine: str
    #: The mix output feeds only the sampled positions (a final mixer), so its
    #: down/up/mix run at ``head_positions`` rather than every token.
    mix_sampled_only: bool = False

    def __post_init__(self) -> None:
        if self.combine not in COMBINES:
            raise ValueError(f"unknown hyper-connection combine {self.combine!r}")

    @property
    def width(self) -> int:
        return self.streams * self.hidden

    def mix_rows(self, wl: Workload) -> float:
        return wl.head_positions if self.mix_sampled_only else wl.matmul_tokens

    def matmul_groups(self) -> list[MatmulGroup]:
        def group(name: str, n: int, k: int, weight: str) -> MatmulGroup:
            return MatmulGroup(
                f"{self.tag}.{name}",
                n=n,
                k=k,
                bucket="residual_mix",
                module=f"{self.path}.{weight}",
                rows=self.mix_rows if self.mix_sampled_only else None,
            )

        groups = [group("down", self.lowrank, self.width, "input_mix_weight_down")]
        if self.inject:
            groups.append(group("inject", self.streams, self.width, "block_inject_weight"))
        groups.append(group("up", self.width, self.lowrank, "input_mix_weight_up"))
        return groups

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return [LearnedWeightGroup(f"{self.tag}.norm", self.width, 1)]

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        per_token = 2.0 * self.width
        rows = []
        if self.combine == "injected":
            rows.append(
                AttentionSemantic(
                    f"{self.tag}.combine", bucket="residual_mix", flops=wl.matmul_tokens * per_token
                )
            )
        rows.append(
            AttentionSemantic(
                f"{self.tag}.mix", bucket="residual_mix", flops=self.mix_rows(wl) * per_token
            )
        )
        return rows


@dataclass(frozen=True)
class GatedHyperConnections:
    """The modules around one decoder layer's sublayers, as a ``LayerStack.mixer``."""

    modules: tuple[GatedHcModule, ...]

    def matmul_groups(self) -> list[MatmulGroup]:
        return [group for module in self.modules for group in module.matmul_groups()]

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return [weight for module in self.modules for weight in module.learned_weight_groups()]

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        return [row for module in self.modules for row in module.semantic_segments(wl)]


def decoder_layer_modules(
    hidden: int, streams: int, lowrank: int, *, attn_combine: str, prefix: str = ""
) -> tuple[GatedHcModule, GatedHcModule]:
    """A decoder layer's attention and FFN modules.

    ``attn_combine`` is what the attention module merges first: ``injected`` (the
    previous layer's FFN output), ``none`` (the body's first layer, after the
    embedding is repeated into the streams) or ``unit`` (a block output with no
    injection, as the MTP layer's projected embedding). The FFN module always
    combines the attention output under the attention module's injection.
    """
    path = f"{prefix}." if prefix else ""
    return (
        GatedHcModule(
            f"{path}attn_hyper_connection", "hc.attn", hidden, streams, lowrank, True, attn_combine
        ),
        GatedHcModule(
            f"{path}mlp_hyper_connection", "hc.mlp", hidden, streams, lowrank, True, "injected"
        ),
    )


def final_mixer_module(hidden: int, streams: int, lowrank: int, prefix: str = "") -> GatedHcModule:
    """The mixer after the last layer: combine the last FFN output at every token,
    then mix (no inject) at the sampled positions only."""
    path = f"{prefix}." if prefix else ""
    return GatedHcModule(
        f"{path}hyper_connection_mixer",
        "hc.final",
        hidden,
        streams,
        lowrank,
        False,
        "injected",
        mix_sampled_only=True,
    )


@dataclass(frozen=True)
class GatedHcFinalMixer:
    """The body's final mixer, once per forward (a ``Model.epilogue`` item).

    An epilogue carries semantic rows only, so each projection is one row with its
    matmul FLOPs and its weight read (``weight_dtype_bytes`` per element, read once),
    and the builder adds :meth:`weight_groups` to ``Model.norm_weights``: the
    projections' parameters (read by their rows, so zero bytes here) and the norm.
    """

    module: GatedHcModule
    weight_dtype_bytes: float
    name: str = "final_mixer"

    def _projections(self) -> list[tuple[str, MatmulGroup]]:
        return [(group.name.split(".")[-1], group) for group in self.module.matmul_groups()]

    def weight_groups(self) -> list[LearnedWeightGroup]:
        groups = [
            LearnedWeightGroup(
                f"{self.name}.{short}.weight",
                group.n * group.k,
                1,
                "residual_mix",
                read_elements=lambda _wl: 0,
            )
            for short, group in self._projections()
        ]
        return [*groups, LearnedWeightGroup(f"{self.name}.norm", self.module.width, 1)]

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        rows = [
            AttentionSemantic(
                f"{self.name}.{short}",
                bucket="residual_mix",
                byte_kind="weights",
                flops=2.0 * group.matmul_rows(wl) * group.n * group.k,
                bytes=group.n * group.k * self.weight_dtype_bytes,
            )
            for short, group in self._projections()
        ]
        rows.extend(
            AttentionSemantic(
                f"{self.name}.{row.name.split('.')[-1]}", bucket=row.bucket, flops=row.flops
            )
            for row in self.module.semantic_segments(wl)
        )
        return rows
