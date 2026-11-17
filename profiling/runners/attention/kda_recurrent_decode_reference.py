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


def kda_recurrent_verify_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_pool: torch.Tensor,
    state_indices: torch.Tensor,
    num_accepted: torch.Tensor,
    *,
    lower_bound: float,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Speculative-verify form: ``q`` tokens per request, one state slot per token.

    Tokens are request-major ``[B·q, ...]`` (the ``cu_seqlens = arange(B+1)·q``
    layout). ``state_indices`` is vLLM's ``[B, q]`` spec-state table. Request
    ``n`` starts from slot ``state_indices[n, num_accepted[n] − 1]`` (the state
    after its last accepted token of the previous step) and writes the state
    after its token ``t`` to ``state_indices[n, t]``. Returns ``(o [B·q,H,V],
    updated copy of state_pool)``; the input pool is not mutated.
    """
    batch, per_request = state_indices.shape
    indices = state_indices.long()
    accepted = num_accepted.long().to(indices.device)
    requests = torch.arange(batch, device=indices.device)
    # Each request's running state, held as its own pool row (fp32, like the
    # kernel's registers between tokens).
    state = state_pool[indices[requests, accepted - 1]].clone()
    rows_of_request = requests * per_request
    own = requests.to(torch.int32)
    updated = state_pool.clone()
    outputs = torch.empty(batch * per_request, *v.shape[1:], dtype=v.dtype, device=v.device)
    for token in range(per_request):
        rows = rows_of_request + token
        o, state = kda_recurrent_decode_reference(
            q[rows],
            k[rows],
            v[rows],
            raw_g[rows],
            raw_beta[rows],
            a_log,
            dt_bias,
            state,
            own,
            lower_bound=lower_bound,
            scale=scale,
        )
        outputs[rows] = o
        updated[indices[:, token]] = state.to(state_pool.dtype)
    return outputs, updated
