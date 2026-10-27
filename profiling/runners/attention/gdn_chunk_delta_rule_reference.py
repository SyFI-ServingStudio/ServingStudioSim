"""Naive per-token Torch oracle for ``gdn_chunk_delta_rule``.

Recomputes what one Gated DeltaNet prefill call computes, token by token, with
no chunking and no FLA or FlashInfer kernel. Inputs are what vLLM hands the
call on its prefill path (``fused_post_conv_prep(..., apply_l2norm=True)``):

- ``q``/``k`` are ``[T, Hk, K]`` and already L2-normalized; the call does not
  normalize them again (``use_qk_l2norm_in_kernel=False``);
- ``v`` is ``[T, H, V]`` with ``H`` a multiple of ``Hk``; value head ``h``
  reads key head ``h // (H // Hk)`` (grouped heads);
- ``g`` is the per-head decay ``[T, H]`` in FP32. ``g_is_log=True`` takes it
  in log space, as FLA's ``chunk_gated_delta_rule`` does; ``False`` takes it
  already exponentiated, as FlashInfer's ``chunk_gated_delta_rule`` does;
- ``beta`` is ``[T, H]`` in FP32, already passed through the sigmoid;
- ``initial_state`` is FP32 ``[N, H, V, K]`` (vLLM's SSM-state layout, the
  transpose of the math layout ``S[k, v]``).

Per sequence and value head, starting from the initial state::

    S *= decay_t
    S += k_t (beta_t (v_t - k_t^T S))^T
    o_t = (K^-1/2 q_t)^T S

Only torch is imported, so the unit tests run this module on CPU.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def gdn_chunk_delta_rule_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    boundaries: Sequence[int],
    *,
    g_is_log: bool,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(o [T,H,V] in v.dtype, final_state [N,H,V,K] fp32)``."""
    tokens, key_heads, key_dim = q.shape
    heads = v.shape[1]
    if heads % key_heads != 0:
        raise ValueError("value heads must be a multiple of key heads")
    if scale is None:
        scale = key_dim**-0.5
    group = heads // key_heads
    qf = q.float().repeat_interleave(group, dim=1) * scale  # [T, H, K]
    kf = k.float().repeat_interleave(group, dim=1)
    vf = v.float()
    decay = g.float().exp() if g_is_log else g.float()
    betaf = beta.float()

    output = torch.empty(vf.shape, dtype=torch.float32, device=v.device)
    final_states = []
    bounds = [int(b) for b in boundaries]
    if bounds[0] != 0 or bounds[-1] != tokens:
        raise ValueError("boundaries must start at 0 and end at the token count")
    for sequence, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        state = initial_state[sequence].float().transpose(-1, -2).clone()  # [H, K, V]
        for t in range(start, end):
            state = state * decay[t].view(heads, 1, 1)
            predicted = torch.einsum("hk,hkv->hv", kf[t], state)
            correction = betaf[t].unsqueeze(-1) * (vf[t] - predicted)
            state = state + kf[t].unsqueeze(-1) * correction.unsqueeze(-2)
            output[t] = torch.einsum("hk,hkv->hv", qf[t], state)
        final_states.append(state.transpose(-1, -2))
    return output.to(v.dtype), torch.stack(final_states)


def rmse_ratio(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """RMSE of the difference over RMS of the reference (FLA's ``get_err_ratio``)."""
    ref = reference.detach().float()
    diff = ref - candidate.detach().float()
    return float(diff.square().mean().sqrt() / (ref.square().mean().sqrt() + 1e-8))


__all__ = ["gdn_chunk_delta_rule_reference", "rmse_ratio"]
