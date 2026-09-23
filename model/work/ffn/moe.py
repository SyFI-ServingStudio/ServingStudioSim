"""Mixture-of-experts FFN: a router + top_k routed experts (+ optional shared expert).

Each token flows through ``top_k`` of ``num_experts`` routed experts (``activated_mult
= top_k``, ``total_count = num_experts`` — the routed groups are ``routed=True`` so only
the hit subset is loaded from HBM, per the balls-in-bins model in
``MatmulGroup.loaded_instances``). An always-on shared expert (Qwen2-MoE / DeepSeek
style; ``shared_intermediate > 0``) adds two dense groups every token.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core import LearnedWeightGroup, MatmulGroup


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
    # Prepended to every module path this FFN reports. A quantization config's
    # inclusion/exclusion lists are matched layer-relative, so two layers whose
    # submodules have the same names cannot be told apart by path alone. Naming a
    # stack that is quantized differently from the body (GLM-5.2's MTP layer
    # keeps BF16 experts under an NVFP4 checkpoint) is what makes them distinct.
    module_prefix: str = ""
    #: Optional latent expert input width. When set, routed experts run in this
    #: width while the two projections around them remain full-hidden matrices.
    expert_input_dim: int | None = None
    #: Learned router correction bias used by the noaux_tc routing method.
    router_bias_elements: int = 0
    #: Learned RMSNorm scale between the routed expert down/up projections.
    routed_norm_elements: int = 0
    #: Number of experts resident on this rank.  The router still projects to
    #: ``num_experts``; only routed weight loading and expert parameter identity
    #: use this local count.
    local_experts: int | None = None
    #: Fraction of global top-k selections handled by this rank's expert shard.
    routing_scale: float = 1.0

    def _module(self, path: str) -> str:
        return f"{self.module_prefix}.{path}" if self.module_prefix else path

    def matmul_groups(self) -> list[MatmulGroup]:
        if self.shared_gate and self.shared_intermediate <= 0:
            raise ValueError("shared_gate requires shared_intermediate > 0")
        expert_input = self.expert_input_dim or self.hidden
        local_experts = self.local_experts or self.num_experts
        if local_experts <= 0 or self.num_experts % local_experts != 0:
            raise ValueError("local_experts must be a positive divisor of num_experts")
        if self.routing_scale <= 0.0 or self.routing_scale > 1.0:
            raise ValueError("routing_scale must be in (0, 1]")
        groups = [
            # `mlp.gate` is the router matrix, and both GLM-5.2-FP8 and
            # Qwen3-235B-FP8 leave it at the master dtype.
            MatmulGroup(
                "router",
                n=self.num_experts,
                k=self.hidden,
                bucket="router",
                module=self._module("mlp.gate"),
            ),
            MatmulGroup(
                "expert_gate_up",
                n=2 * self.moe_intermediate,
                k=expert_input,
                activated_mult=self.top_k,
                total_count=local_experts,
                bucket="expert",
                routed=True,
                module=self._module("mlp.experts.gate_up_proj"),
                activated_scale=self.routing_scale,
            ),
            MatmulGroup(
                "expert_down",
                n=expert_input,
                k=self.moe_intermediate,
                activated_mult=self.top_k,
                total_count=local_experts,
                bucket="expert",
                routed=True,
                module=self._module("mlp.experts.down_proj"),
                activated_scale=self.routing_scale,
            ),
        ]
        if self.expert_input_dim is not None:
            groups.insert(
                1,
                MatmulGroup(
                    "routed_expert_down_proj",
                    n=expert_input,
                    k=self.hidden,
                    bucket="dense_ffn",
                    module=self._module("mlp.routed_expert_down_proj"),
                ),
            )
            groups.insert(
                4,
                MatmulGroup(
                    "routed_expert_up_proj",
                    n=self.hidden,
                    k=expert_input,
                    bucket="dense_ffn",
                    module=self._module("mlp.routed_expert_up_proj"),
                ),
            )
        if self.shared_intermediate > 0:
            groups.append(
                MatmulGroup(
                    "shared_gate_up",
                    n=2 * self.shared_intermediate,
                    k=self.hidden,
                    bucket="shared_expert",
                    module=self._module(f"mlp.{self.shared_module}.gate_up_proj"),
                )
            )
            groups.append(
                MatmulGroup(
                    "shared_down",
                    n=self.hidden,
                    k=self.shared_intermediate,
                    bucket="shared_expert",
                    module=self._module(f"mlp.{self.shared_module}.down_proj"),
                )
            )
            if self.shared_gate:
                groups.append(
                    MatmulGroup(
                        "shared_gate",
                        n=1,
                        k=self.hidden,
                        bucket="shared_expert",
                        module=self._module("mlp.shared_expert_gate"),
                    )
                )
        return groups

    def learned_weight_groups(self) -> list[LearnedWeightGroup]:
        groups: list[LearnedWeightGroup] = []
        if self.router_bias_elements:
            groups.append(
                LearnedWeightGroup(
                    "router_correction_bias",
                    self.router_bias_elements,
                    1,
                    "router",
                )
            )
        if self.routed_norm_elements:
            groups.append(
                LearnedWeightGroup(
                    "routed_expert_norm",
                    self.routed_norm_elements,
                    1,
                    "norm",
                )
            )
        return groups
