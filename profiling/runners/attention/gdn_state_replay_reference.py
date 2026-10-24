"""Pure-Torch reference for ``gdn_state_replay``.

Independent of TensorFold: for one stream and one layer, walk the accepted rows
of a saved verify window in order and apply the gated delta rule to the fp32
``(value heads, value_head_dim, key_head_dim)`` state. Value head ``h`` reads key
head ``h // (num_value_heads // num_qk_heads)``. Each row ``t`` does

    S <- g_t S;  S <- S + beta_t (v_t - S k_t) k_t^T

with ``g`` the already exponentiated decay and ``beta`` already through the
sigmoid. No query is read and nothing but the state is produced.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def gdn_state_replay_reference(
    torch: Any, k: Any, v: Any, g: Any, beta: Any, state: Any, path: Sequence[int]
) -> Any:
    """The fp32 state after replaying window rows ``path`` (in order) from ``state``.

    ``k`` is ``(W, num_qk_heads, dk)``, ``v`` ``(W, num_value_heads, dv)``, ``g``
    and ``beta`` ``(W, num_value_heads)``; ``state`` is not modified.
    """

    group = v.shape[1] // k.shape[1]
    s = state.float().clone()
    for row in path:
        kt = k[row].float().repeat_interleave(group, dim=0)
        s = s * g[row].float()[:, None, None]
        memory = torch.einsum("hvk,hk->hv", s, kt)
        delta = (v[row].float() - memory) * beta[row].float()[:, None]
        s = s + delta[:, :, None] * kt[:, None, :]
    return s
