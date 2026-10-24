"""Independent value reference for the DFlash2 two-tap dynamic causal convolution."""

from typing import Any


def dynamic_causal_conv_reference(
    torch: Any,
    x: Any,
    dyn: Any,
    base: Any,
    branch: int,
    group_size: int,
    segment: int,
    residual: Any = None,
) -> Any:
    """``x`` (R, D) bf16, ``dyn`` (R, 2, 2, D / g), ``base`` (2, 2, D); returns (R, D) bf16.

    Row ``r`` mixes rows ``r`` and ``r - 1`` of its ``segment``-row block.
    """
    rows, _ = x.shape
    xf = x.float()
    prev = torch.zeros_like(xf)
    prev[1:] = xf[:-1]
    first = torch.arange(rows, device=x.device) % segment == 0
    prev[first] = 0.0
    k0 = (base[branch, 0].float() + dyn[:, branch, 0].float().repeat_interleave(group_size, -1)).to(
        torch.bfloat16
    )
    k1 = (base[branch, 1].float() + dyn[:, branch, 1].float().repeat_interleave(group_size, -1)).to(
        torch.bfloat16
    )
    y = (xf * k0.float() + prev * k1.float()).to(torch.bfloat16)
    if residual is not None:
        y = (residual.float() + y.float()).to(torch.bfloat16)
    return y
