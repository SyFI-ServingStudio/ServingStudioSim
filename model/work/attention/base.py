"""The AttentionSpec contract.

An attention spec models ONE layer's attention. It contributes to the minimum work
in three ways:

- ``matmul_groups()``  — the attention weight projections (QKV, O; MLA adds latent
  up/down). These count toward ``attn_proj`` FLOPs and ``attn`` params, folded ×layers.
- ``internal_flops(wl)`` — the non-weight attention compute (QK^T + softmax·V), a
  function of the workload's per-interaction (query, key) pair counts.
- ``kv_bytes(wl)`` — the KV/state bytes that must be read from HBM (the cached keys;
  a cache-less full-attention interaction reads nothing — its KV is fused activation).

New attention mechanisms (MLA, sliding-window, SSM/linear) implement this same
protocol in a new file; every FFN combination then works unchanged.
"""

from __future__ import annotations

from typing import Protocol

from ..core import MatmulGroup, Workload


class AttentionSpec(Protocol):
    def matmul_groups(self) -> list[MatmulGroup]: ...

    def internal_flops(self, wl: Workload) -> float: ...

    def kv_bytes(self, wl: Workload) -> float: ...
