"""Dense SwiGLU feed-forward: fused gate+up (hidden -> 2·intermediate) then down."""

from __future__ import annotations

from dataclasses import dataclass

from ..core import MatmulGroup


@dataclass
class DenseSwiGLU:
    hidden: int
    intermediate: int

    def matmul_groups(self) -> list[MatmulGroup]:
        return [
            MatmulGroup("gate_up", n=2 * self.intermediate, k=self.hidden, bucket="dense_ffn"),
            MatmulGroup("down", n=self.hidden, k=self.intermediate, bucket="dense_ffn"),
        ]
