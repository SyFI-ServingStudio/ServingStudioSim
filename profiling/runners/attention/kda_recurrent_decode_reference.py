"""Naive single-step Torch oracle for ``kda_recurrent_decode``.

Recomputes one decode token per sequence the way ``fused_recurrent_kda``
computes it on the GLM-5.3-Flash plain-decode path. It uses no FLA kernel:

1. l2-normalize q and k per head in fp32 (``eps=1e-6``). The recurrent kernel
   normalizes in registers, so unlike the chunk oracle nothing is rounded back
   to bf16.
2. Compute the per-channel safe gate
   ``g = lower_bound * sigmoid(exp(A_log[h]) * (raw_g + dt_bias[h, :]))`` and
   ``beta = sigmoid(raw_beta)`` in fp32.
3. Apply one gated-delta-rule step to each sequence's fp32 state ``S[h, k, v]``:
   ``S *= exp(g)``, ``S += beta * k (v - k^T S)``, ``o = (scale * q)^T S``.

States are read from and written to the ``[slots, H, V, K]`` pool through
``state_indices``, the production layout. Only torch is imported, so the unit
tests run this module on CPU.
"""

from __future__ import annotations

import torch

from profiling.runners.attention.kda_chunk_prefill_reference import kda_safe_gate

L2NORM_EPS = 1e-6


def kda_recurrent_decode_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    lower_bound: float,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(o [B,H,V] in v.dtype, updated copy of state_pool)``.

    ``q``/``k``/``raw_g`` are ``[B,H,K]``, ``v`` is ``[B,H,V]``, ``raw_beta``
    is the un-sigmoided ``[B,H]``, ``state_pool`` is ``[slots,H,V,K]`` and
    ``state_indices`` holds each sequence's slot. The input pool is not mutated.
    """
    head_dim = q.shape[-1]
    if scale is None:
        scale = head_dim**-0.5
    qf, kf = q.float(), k.float()
    qn = qf * torch.rsqrt(qf.square().sum(-1, keepdim=True) + L2NORM_EPS) * scale
    kn = kf * torch.rsqrt(kf.square().sum(-1, keepdim=True) + L2NORM_EPS)
    decay = kda_safe_gate(raw_g, a_log, dt_bias, lower_bound).exp()  # [B, H, K]
    beta = raw_beta.float().sigmoid()

    slots = state_indices.long()
    state = state_pool[slots].float().transpose(-1, -2)  # [B, H, K, V]
    state = state * decay.unsqueeze(-1)
    predicted = torch.einsum("bhk,bhkv->bhv", kn, state)
    correction = beta.unsqueeze(-1) * (v.float() - predicted)
    state = state + kn.unsqueeze(-1) * correction.unsqueeze(-2)
    output = torch.einsum("bhk,bhkv->bhv", qn, state)

    updated = state_pool.clone()
    updated[slots] = state.transpose(-1, -2).to(state_pool.dtype)
    return output.to(v.dtype), updated
