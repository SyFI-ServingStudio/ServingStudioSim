"""Naive per-token Torch oracle for ``kda_chunk_prefill``.

Recomputes what ``chunk_kda_with_fused_gate`` computes on the GLM-5.3-Flash
prefill path. It does not use the chunked algorithm or any FLA kernel:

1. l2-normalize q and k per head (fp32 math, ``eps=1e-6``). The result is
   rounded back to the input dtype because ``l2norm_fwd`` returns the input
   dtype and the chunk kernels read that rounded tensor.
2. Compute the per-channel safe gate in natural-log space:
   ``g = lower_bound * sigmoid(exp(A_log[h]) * (raw_g + dt_bias[h, :]))``.
3. Run the gated delta rule once per token, one sequence at a time, on an fp32
   state ``S[h, k, v]``. This is the recurrence in FLA's ``naive_recurrent_kda``:
   ``S *= exp(g_t)``, ``S += beta_t * k_t (v_t - k_t^T S)``,
   ``o_t = (scale * q_t)^T S``.

Production stores recurrent states as ``[N, H, V, K]``, the transpose of the
math layout, so the oracle takes and returns that layout.

Only torch is imported, so the unit tests run this module on CPU.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

L2NORM_EPS = 1e-6


def l2norm_like_fla(x: torch.Tensor, eps: float = L2NORM_EPS) -> torch.Tensor:
    """``x / sqrt(sum(x^2) + eps)`` over the last dim, rounded to ``x.dtype``."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().sum(-1, keepdim=True) + eps)).to(x.dtype)


def kda_safe_gate(
    raw_g: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
) -> torch.Tensor:
    """Per-channel log-decay ``[T, H, K]`` in ``(lower_bound, 0)``."""
    _, heads, head_dim = raw_g.shape
    decay_rate = a_log.reshape(-1).float().exp().view(1, heads, 1)
    bias = dt_bias.reshape(heads, head_dim).float()
    return lower_bound * torch.sigmoid(decay_rate * (raw_g.float() + bias))


def kda_chunk_prefill_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    boundaries: Sequence[int],
    *,
    lower_bound: float,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(o [T,H,V] in v.dtype, final_state [N,H,V,K] fp32)``.

    ``q``/``k``/``raw_g`` are ``[T,H,K]``, ``v`` is ``[T,H,V]``, ``beta`` is the
    already-sigmoided ``[T,H]``, ``initial_state`` is ``[N,H,V,K]`` and
    ``boundaries`` holds the N+1 cumulative sequence offsets.
    """
    head_dim = q.shape[-1]
    if scale is None:
        scale = head_dim**-0.5
    qn = l2norm_like_fla(q).float() * scale
    kn = l2norm_like_fla(k).float()
    vf = v.float()
    decay = kda_safe_gate(raw_g, a_log, dt_bias, lower_bound).exp()
    betaf = beta.float()

    output = torch.empty(vf.shape, dtype=torch.float32, device=v.device)
    final_states = []
    bounds = [int(b) for b in boundaries]
    for sequence, (start, end) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        state = initial_state[sequence].float().transpose(-1, -2).clone()  # [H, K, V]
        for t in range(start, end):
            state = state * decay[t].unsqueeze(-1)
            predicted = torch.einsum("hk,hkv->hv", kn[t], state)
            correction = betaf[t].unsqueeze(-1) * (vf[t] - predicted)
            state = state + kn[t].unsqueeze(-1) * correction.unsqueeze(-2)
            output[t] = torch.einsum("hk,hkv->hv", qn[t], state)
        final_states.append(state.transpose(-1, -2))
    return output.to(v.dtype), torch.stack(final_states)


def rmse_ratio(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """RMSE of the difference over RMS of the reference (the fork's KDA metric)."""
    ref = reference.detach().float()
    diff = ref - candidate.detach().float()
    return float(diff.square().mean().sqrt() / (ref.square().mean().sqrt() + 1e-8))
