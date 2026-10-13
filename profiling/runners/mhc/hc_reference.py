"""Independent value references for the gated hyper-connection (HC) kernels.

The residual holds ``hc_mult`` streams of ``hidden_size`` features per token,
stream-outer (``[tokens, hc_mult * hidden_size]``). The norms are Gemma-style:
``x / rms(x) * (1 + w)``, one RMS per stream, with a ``[hc_mult * hidden_size]``
weight.
"""

from typing import Any

from profiling.runners.norm.gemma_rms_norm_reference import gemma_rms_norm_reference


def hc_combine_reference(
    torch: Any, residual: Any, block: Any, injection: Any, hc_mult: int
) -> Any:
    """``residual + coeff * block`` per stream.

    coeff = 2*sigmoid(logit/hc_mult), or 1 without logits.
    """
    streams = residual.float().unflatten(-1, (hc_mult, -1))
    if injection is None:
        coeff = torch.ones(1, hc_mult, 1, device=residual.device)
    else:
        coeff = (2.0 * torch.sigmoid(injection.float() / hc_mult))[:, :, None]
    return (streams + coeff * block.float()[:, None, :]).flatten(-2).to(residual.dtype)


def hc_combine_norm_reference(
    torch: Any, residual: Any, block: Any, injection: Any, weight: Any, eps: float, hc_mult: int
) -> tuple[Any, Any]:
    combined = hc_combine_reference(torch, residual, block, injection, hc_mult)
    return combined, gemma_rms_norm_reference(torch, combined, weight, eps, hc_mult)


def hc_gate_mix_reference(torch: Any, x: Any, gate: Any, hc_mult: int) -> Any:
    """Mean over the streams of ``sigmoid(gate) * x``."""
    mixed = torch.sigmoid(gate.float()) * x.float()
    return mixed.unflatten(-1, (hc_mult, -1)).mean(-2).to(x.dtype)


def scaled_silu_reference(torch: Any, x: Any, divisor: float) -> Any:
    return torch.nn.functional.silu(x.float() / divisor).to(x.dtype)


__all__ = [
    "hc_combine_norm_reference",
    "hc_combine_reference",
    "hc_gate_mix_reference",
    "scaled_silu_reference",
]
