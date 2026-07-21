"""Mixture-of-experts FFN: a router + top_k routed experts (+ optional shared expert).

Each token flows through ``top_k`` of ``num_experts`` routed experts (``activated_mult
= top_k``, ``total_count = num_experts`` — the routed groups are ``routed=True`` so only
the hit subset is loaded from HBM, per the balls-in-bins model in
``MatmulGroup.loaded_instances``). An always-on shared expert (Qwen2-MoE / DeepSeek
style; ``shared_intermediate > 0``) adds two dense groups every token.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core import MatmulGroup


@dataclass
class MoE:
    hidden: int
    moe_intermediate: int
    num_experts: int
    top_k: int
    shared_intermediate: int = 0  # 0 = no shared expert

    def matmul_groups(self) -> list[MatmulGroup]:
        groups = [
            MatmulGroup("router", n=self.num_experts, k=self.hidden, bucket="router"),
            MatmulGroup(
                "expert_gate_up",
                n=2 * self.moe_intermediate,
                k=self.hidden,
                activated_mult=self.top_k,
                total_count=self.num_experts,
                bucket="expert",
                routed=True,
            ),
            MatmulGroup(
                "expert_down",
                n=self.hidden,
                k=self.moe_intermediate,
                activated_mult=self.top_k,
                total_count=self.num_experts,
                bucket="expert",
                routed=True,
            ),
        ]
        if self.shared_intermediate > 0:
            groups.append(
                MatmulGroup(
                    "shared_gate_up",
                    n=2 * self.shared_intermediate,
                    k=self.hidden,
                    bucket="shared_expert",
                )
            )
            groups.append(
                MatmulGroup(
                    "shared_down", n=self.hidden, k=self.shared_intermediate, bucket="shared_expert"
                )
            )
        return groups
