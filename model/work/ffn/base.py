"""The FFNSpec contract.

An FFN spec models ONE layer's feed-forward block. It contributes only weight
matmuls, so its single method returns the per-layer :class:`MatmulGroup` list. A dense
block returns gate/up + down; a MoE block returns router + routed experts
(``activated_mult=top_k``, ``total_count=num_experts``) + any always-on shared expert.

New FFN mechanisms (shared experts, fine-grained experts) implement this same protocol
in a new file; every attention combination then works unchanged.
"""

from __future__ import annotations

from typing import Protocol

from ..core import MatmulGroup


class FFNSpec(Protocol):
    def matmul_groups(self) -> list[MatmulGroup]: ...
