"""Dense SwiGLU feed-forward: fused gate+up (hidden -> 2·intermediate) then down."""

from __future__ import annotations

from dataclasses import dataclass

from ..core import MatmulGroup


@dataclass
class DenseSwiGLU:
    hidden: int
    intermediate: int
    #: Checkpoint path prefix for a block that lives outside the body's layers
    #: (a separate draft checkpoint), so a quantized target's rules cannot
    #: match it.
    module_prefix: str = ""

    def _module(self, path: str) -> str:
        return f"{self.module_prefix}.{path}" if self.module_prefix else path

    def matmul_groups(self) -> list[MatmulGroup]:
        return [
            MatmulGroup(
                "gate_up",
                n=2 * self.intermediate,
                k=self.hidden,
                bucket="dense_ffn",
                module=self._module("mlp.gate_up_proj"),
            ),
            MatmulGroup(
                "down",
                n=self.hidden,
                k=self.intermediate,
                bucket="dense_ffn",
                module=self._module("mlp.down_proj"),
            ),
        ]
