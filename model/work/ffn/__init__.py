"""FFN specs — one file per mechanism (dense today; MoE / shared-expert future)."""

from .dense import DenseSwiGLU
from .moe import MoE

__all__ = ["DenseSwiGLU", "MoE"]
