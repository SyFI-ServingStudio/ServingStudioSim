"""Inkling's sublayer-output short convolutions and the norms beside them.

Each Inkling decoder layer (vLLM fork ``vllm/models/inkling/nvidia/model.py``)
closes both sublayers the same way: ``h = h + sconv(delta)``, then the next
sublayer's RMSNorm, where ``delta`` is the attention (``attn_sconv``) or FFN
(``mlp_sconv``) output and ``sconv`` is a depthwise causal convolution of ``W`` taps
over the full hidden width (plus its input as a residual). The fork fuses each
boundary into one Lamport kernel with the tensor-parallel reduce-scatter and
all-gather; the convolution and the norm are model work all the same.

Necessary work per layer:

- each convolution: ``2 * W * hidden`` FLOPs per token, as a depthwise
  ``MatmulGroup`` (``n=hidden``, ``k=W``), and its ``hidden x W`` weight;
- each convolution's state: the conv at position ``p`` reads taps
  ``p-W+1 .. p``, so a request reads ``min(prefix, W - 1)`` cached input rows and
  leaves ``min(append, W - 1)`` new ones (BF16, ``hidden`` wide);
- the ``attn_norm`` / ``mlp_norm`` scales (compulsory weight reads; the norm math is
  elementwise, convention 4). The residual add is elementwise too.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..attention.base import AttentionSemantic
from ..attention.inkling_rel import inkling_requests, state_rows
from ..core import LearnedWeightGroup, MatmulGroup, Workload

_SUBLAYERS = ("attn", "mlp")


@dataclass
class InklingSublayerShortConv:
    """The two sublayer-output convolutions and the two norms of one layer."""

    hidden: int
    kernel: int
    state_dtype_bytes: float

    def matmul_groups(self) -> list[MatmulGroup]:
        return [
            MatmulGroup(
                f"{sublayer}_sconv",
                n=self.hidden,
                k=self.kernel,
                bucket="residual_mix",
                module=f"{sublayer}_sconv",
            )
            for sublayer in _SUBLAYERS
        ]

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        return [LearnedWeightGroup(f"{sublayer}_norm", self.hidden, 1) for sublayer in _SUBLAYERS]

    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]:
        requests = [request for _phase, phase in inkling_requests(wl) for request in phase]
        state_bytes = state_rows(requests, self.kernel - 1) * self.hidden * self.state_dtype_bytes
        return [
            AttentionSemantic(f"{sublayer}_sconv_state", bucket="embedding", bytes=state_bytes)
            for sublayer in _SUBLAYERS
        ]
