"""The AttentionSpec contract.

An attention spec models ONE layer's attention. It contributes to the minimum work
in three ways:

- ``matmul_groups()``  — the attention weight projections (QKV, O; MLA adds latent
  up/down). These count toward ``attn_proj`` FLOPs and ``attn`` params, folded ×layers.
- ``internal_flops(wl)`` — the non-weight attention compute (QK^T + softmax·V), a
  function of the workload's per-interaction (query, key) pair counts.
- ``kv_bytes(wl)`` — the KV/state bytes that must be read from HBM (the cached keys;
  a cache-less full-attention interaction reads nothing — its KV is fused activation).
- ``cache_write_bytes(wl)`` — persistent cache/state writes not removable by fusion.

New attention mechanisms (MLA, sliding-window, SSM/linear) implement this same
protocol in a new file; every FFN combination then works unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..core import MatmulGroup, Workload


@dataclass(frozen=True)
class AttentionSemantic:
    """One non-matmul attention row with an independent CostTree identity.

    Most attention families have one fused ``attn`` row per phase. MLA/DSA has
    two persistent caches and a separately scheduled indexer, so it needs a few
    more stable semantic rows while still sharing the generic Model fold.
    """

    name: str
    bucket: str = "attn_internal"
    byte_kind: str = "kv"
    flops: float = 0.0
    bytes: float = 0.0
    #: Precision this row's math runs at, when the mechanism fixes it independently
    #: of the checkpoint's weight quantization (DSA computes index logits in FP8 but
    #: the sparse MLA kernel in BF16). ``None`` inherits the caller's default.
    compute_dtype: str | None = None


class AttentionSpec(Protocol):
    split_attention_phases: bool

    def matmul_groups(self) -> list[MatmulGroup]: ...

    def internal_flops(self, wl: Workload) -> float: ...

    def kv_bytes(self, wl: Workload) -> float: ...

    def cache_write_bytes(self, wl: Workload) -> float: ...

    # Optional: a spec may expose more than the historical one fused attention
    # row. Model.label uses getattr so existing specs remain source-compatible.
    def semantic_segments(self, wl: Workload) -> list[AttentionSemantic]: ...
