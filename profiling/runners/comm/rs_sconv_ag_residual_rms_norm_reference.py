"""Torch semantics for Inkling's fused RS + short-conv + AG + residual + RMSNorm.

The production call (vLLM Inkling ``LamportRSConv.rs_sconv_ag_add_norm``) is a
tensor-parallel collective. This reference computes the same result on one
device from every rank's inputs at once, with no communication and no paging:

    v_r    = bf16(partial_r + shared_r)                  # per-rank sum before the RS
    x      = bf16(Σ_r v_r)                               # reduce-scatter result (all shards)
    y      = bf16(Σ_i w_i · x[pos-K+1+i] + x[pos])       # causal depthwise conv + skip
    h      = bf16(residual + y)                          # residual add (gathered row)
    normed = bf16(h · rsqrt(mean(h²) + eps) · gamma)     # RMSNorm

Additions run in FP32 and round to BF16 at the same points as the production
kernels. The conv taps before the current token come from ``prior_taps`` (the
conv-state cache content held before the call) unless an earlier token of the
same request is in this batch: the production path inserts every token's ``x``
into the cache before any conv reads, so same-batch taps see this call's ``x``.
"""

from __future__ import annotations

import torch

__all__ = ["rs_sconv_ag_residual_rms_norm_reference"]


def rs_sconv_ag_residual_rms_norm_reference(
    partials: torch.Tensor,
    shared: torch.Tensor | None,
    residual: torch.Tensor,
    conv_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    prior_taps: torch.Tensor,
    positions: torch.Tensor,
    seq_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(normed, residual_out, conv_input)``, each BF16 ``[T, H]``.

    ``partials`` and ``shared`` are BF16 ``[W, T, H]`` (one row block per TP
    rank); ``shared`` is the optional MoE sink partial. ``residual`` is BF16
    ``[T, H]``. ``conv_weight`` is the full BF16 depthwise filter ``[H, K]``
    (rank r applies rows ``r·H/W .. (r+1)·H/W``). ``norm_weight`` is BF16 ``[H]``.
    ``prior_taps`` is BF16 ``[T, K-1, H]``: the cached conv input at absolute
    positions ``pos-K+1 .. pos-1`` before this call; taps at negative positions
    are zero whatever it holds. ``positions`` is ``[T]`` absolute positions and
    ``seq_idx`` ``[T]`` the owning request of each token. ``conv_input`` is the
    reduced ``x`` that the call writes into the conv-state cache.

    Padded tokens (production ``slot_mapping == -1``) are not modeled.
    """
    if partials.dtype != torch.bfloat16 or partials.ndim != 3:
        raise ValueError("partials must be bf16 [W, T, H]")
    _, tokens, hidden = partials.shape
    if shared is not None and shared.shape != partials.shape:
        raise ValueError("shared must match partials")
    if residual.shape != (tokens, hidden):
        raise ValueError("residual must be [T, H]")
    kernel_size = conv_weight.shape[1]
    if conv_weight.shape != (hidden, kernel_size):
        raise ValueError("conv_weight must be [H, K]")
    if prior_taps.shape != (tokens, kernel_size - 1, hidden):
        raise ValueError("prior_taps must be [T, K-1, H]")

    published = partials.float()
    if shared is not None:
        published = published + shared.float()
    published = published.to(torch.bfloat16)
    conv_input = published.float().sum(dim=0).to(torch.bfloat16)

    taps = _resolve_taps(conv_input, prior_taps, positions, seq_idx)
    weight = conv_weight.float()
    conv = torch.zeros(tokens, hidden, dtype=torch.float32, device=partials.device)
    for tap in range(kernel_size):
        conv = conv + taps[:, tap].float() * weight[:, tap]
    short_conv = (conv + conv_input.float()).to(torch.bfloat16)
    residual_out = (residual.float() + short_conv.float()).to(torch.bfloat16)

    row = residual_out.float()
    inv_rms = torch.rsqrt(row.square().mean(dim=-1, keepdim=True) + eps)
    normed = (row * inv_rms * norm_weight.float()).to(torch.bfloat16)
    return normed, residual_out, conv_input


def _resolve_taps(
    conv_input: torch.Tensor,
    prior_taps: torch.Tensor,
    positions: torch.Tensor,
    seq_idx: torch.Tensor,
) -> torch.Tensor:
    """``[T, K, H]`` conv window per token, oldest tap first, current tap last."""
    tokens, history, _ = prior_taps.shape
    kernel_size = history + 1
    positions = positions.to(torch.int64)
    seq_idx = seq_idx.to(torch.int64)
    # One integer key per (request, position) of the tokens in this batch.
    span = int(positions.max().item()) + kernel_size + 1
    batch_keys = seq_idx * span + positions
    order = torch.argsort(batch_keys)
    sorted_keys = batch_keys[order]
    if tokens > 1 and bool((sorted_keys[1:] == sorted_keys[:-1]).any()):
        raise ValueError("two tokens share one (request, position)")

    window = []
    for tap in range(history):
        source_position = positions - (kernel_size - 1) + tap
        value = prior_taps[:, tap].clone()
        key = seq_idx * span + source_position
        found = torch.searchsorted(sorted_keys, key).clamp(max=tokens - 1)
        in_batch = (sorted_keys[found] == key) & (source_position >= 0)
        value[in_batch] = conv_input[order[found[in_batch]]]
        value[source_position < 0] = 0
        window.append(value)
    window.append(conv_input)
    return torch.stack(window, dim=1)
