"""Naive Torch oracles for the KDA RecoverSSM verify and commit kinds.

They recompute, without any Triton or FLA kernel, what the vLLM fork's
Kimi-K3 ``vllm.models.kimi_k3.nvidia.ops.recoverssm`` computes:

``kda_recoverssm_verify_reference``
    For each request, start from its fp32 checkpoint ``S[H, V, K]`` and run
    the window's tokens through one gated-delta-rule step each:
    ``kn = k * rsqrt(|k|^2 + 1e-6)``, ``S *= exp(gate)`` along K,
    ``correction = sigmoid(raw_beta) * (v - S kn)``, ``S += correction kn^T``,
    ``o = S (q * rsqrt(|q|^2 + 1e-6) * K^-0.5)``. The checkpoint is not
    written. Each step's ``correction`` (fp32 ``[V]``) and its raw ``k`` and
    ``raw_g`` (activation dtype, ``[2K]``) go to the request's slot of the
    record buffers.

``kda_recoverssm_commit_reference``
    Fold the first ``commit_len`` records of each request into its checkpoint:
    ``S <- S * exp(gate_t) + correction_t kn_t^T`` for t in order, the same
    state the verify window reached after ``commit_len`` tokens.

``compact_conv_state_reference``
    Keep the last ``history`` conv inputs ending at the accepted token:
    ``conv[:, h] <- conv[:, commit_len - 1 + h]`` on the dim-first view.

The gate is ``lower_bound * sigmoid(exp(A_log[h]) * (raw_g + dt_bias[h, :]))``
(``kda_safe_gate``). Only torch is imported, so the unit tests run this module
on CPU.
"""

from __future__ import annotations

import torch

from profiling.runners.attention.kda_chunk_prefill_reference import kda_safe_gate

L2NORM_EPS = 1e-6


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().sum(-1, keepdim=True) + L2NORM_EPS)


def kda_recoverssm_verify_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    checkpoint: torch.Tensor,
    state_indices: torch.Tensor,
    spec_query_len: int,
    *,
    lower_bound: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(o [T,H,V] fp32, correction [B,H,S,V] fp32, kg [B,H,S,2K])``.

    ``q``/``k``/``raw_g`` are ``[T,H,K]`` and ``v`` is ``[T,H,V]`` with
    ``T = B * spec_query_len`` (every request a full window). ``raw_beta`` is
    the un-sigmoided ``[T,H]``. ``checkpoint`` is the ``[slots,H,V,K]`` pool;
    it is only read. The records are indexed by request, not by slot.
    """
    batch = state_indices.shape[0]
    tokens, heads, head_dim = q.shape
    if tokens != batch * spec_query_len:
        raise ValueError("the reference covers full windows only")
    view = (batch, spec_query_len, heads)
    qn = (_l2norm(q) * head_dim**-0.5).view(*view, head_dim)
    kn = _l2norm(k).view(*view, head_dim)
    decay = kda_safe_gate(raw_g, a_log, dt_bias, lower_bound).exp().view(*view, head_dim)
    beta = raw_beta.float().sigmoid().view(*view)
    values = v.float().view(*view, v.shape[-1])

    state = checkpoint[state_indices.long()].float()  # [B, H, V, K]
    outputs, corrections = [], []
    for t in range(spec_query_len):
        state = state * decay[:, t, :, None, :]
        predicted = torch.einsum("bhvk,bhk->bhv", state, kn[:, t])
        correction = beta[:, t, :, None] * (values[:, t] - predicted)
        state = state + correction[..., None] * kn[:, t, :, None, :]
        outputs.append(torch.einsum("bhvk,bhk->bhv", state, qn[:, t]))
        corrections.append(correction)
    out = torch.stack(outputs, dim=1).reshape(tokens, heads, -1)
    correction_records = torch.stack(corrections, dim=2)  # [B, H, S, V]
    kg = torch.cat((k.view(*view, head_dim), raw_g.view(*view, head_dim)), dim=-1).transpose(
        1, 2
    )  # [B, H, S, 2K]
    return out, correction_records, kg.to(k.dtype)


def kda_recoverssm_commit_reference(
    checkpoint: torch.Tensor,
    correction: torch.Tensor,
    kg: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    commit_lens: torch.Tensor,
    *,
    lower_bound: float,
) -> torch.Tensor:
    """Return the committed ``[B,H,V,K]`` fp32 states.

    ``checkpoint`` is each request's ``[B,H,V,K]`` state, ``correction`` its
    ``[B,H,S,V]`` records and ``kg`` its ``[B,H,S,2K]`` records. ``a_log`` is
    ``[H]`` and ``dt_bias`` ``[H*K]``.
    """
    head_dim = checkpoint.shape[-1]
    states = []
    for b, commit_len in enumerate(commit_lens.tolist()):
        state = checkpoint[b].float()
        for t in range(int(commit_len)):
            k = kg[b, :, t, :head_dim]
            raw_g = kg[b, :, t, head_dim:]
            decay = kda_safe_gate(raw_g[None], a_log, dt_bias, lower_bound).exp()[0]
            kn = _l2norm(k)
            state = (
                state * decay[:, None, :] + correction[b, :, t, :, None].float() * kn[:, None, :]
            )
        states.append(state)
    return torch.stack(states)


def compact_conv_state_reference(
    conv_state: torch.Tensor, commit_lens: torch.Tensor, history: int
) -> torch.Tensor:
    """Return each request's compacted ``[B, dim, state_len]`` conv window.

    ``conv_state`` is the dim-first ``[B, dim, state_len]`` view. Positions
    ``history`` onward are left as they were.
    """
    out = conv_state.clone()
    for b, commit_len in enumerate(commit_lens.tolist()):
        start = int(commit_len) - 1
        out[b, :, :history] = conv_state[b, :, start : start + history]
    return out


__all__ = [
    "compact_conv_state_reference",
    "kda_recoverssm_commit_reference",
    "kda_recoverssm_verify_reference",
]
