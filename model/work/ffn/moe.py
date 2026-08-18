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
    # Checkpoint name of the shared-expert submodule. DeepSeek/GLM use the plural
    # `shared_experts`; Qwen2-MoE uses `shared_expert`. Only matters for matching a
    # quantized config's modules_to_not_convert.
    shared_module: str = "shared_experts"
    shared_gate: bool = False

    def matmul_groups(self) -> list[MatmulGroup]:
        if self.shared_gate and self.shared_intermediate <= 0:
            raise ValueError("shared_gate requires shared_intermediate > 0")
        groups = [
            # `mlp.gate` is the router matrix, and both GLM-5.2-FP8 and
            # Qwen3-235B-FP8 leave it at the master dtype.
            MatmulGroup(
                "router",
                n=self.num_experts,
                k=self.hidden,
                bucket="router",
                module="mlp.gate",
            ),
            MatmulGroup(
                "expert_gate_up",
                n=2 * self.moe_intermediate,
                k=self.hidden,
                activated_mult=self.top_k,
                total_count=self.num_experts,
                bucket="expert",
                routed=True,
                module="mlp.experts.gate_up_proj",
            ),
            MatmulGroup(
                "expert_down",
                n=self.hidden,
                k=self.moe_intermediate,
                activated_mult=self.top_k,
                total_count=self.num_experts,
                bucket="expert",
                routed=True,
                module="mlp.experts.down_proj",
            ),
        ]
        if self.shared_intermediate > 0:
            groups.append(
                MatmulGroup(
                    "shared_gate_up",
                    n=2 * self.shared_intermediate,
                    k=self.hidden,
                    bucket="shared_expert",
                    module=f"mlp.{self.shared_module}.gate_up_proj",
                )
            )
            groups.append(
                MatmulGroup(
                    "shared_down",
                    n=self.hidden,
                    k=self.shared_intermediate,
                    bucket="shared_expert",
                    module=f"mlp.{self.shared_module}.down_proj",
                )
            )
            if self.shared_gate:
                groups.append(
                    MatmulGroup(
                        "shared_gate",
                        n=1,
                        k=self.hidden,
                        bucket="shared_expert",
                        module="mlp.shared_expert_gate",
                    )
                )
        return groups
