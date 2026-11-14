"""Independent value reference and acceptance check for the fp32 row-split LayerNorm epilogue."""

from typing import Any

# Absolute slack for normed values that cancel to near zero (see check_row_split).
NEAR_ZERO_ATOL = 1e-6


def fp32_row_split_layer_norm_reference(
    torch: Any,
    y: Any,
    weight: Any,
    bias: Any,
    eps: float,
    bf16_lead_cols: int,
    norm_cols: int,
    bf16_trail_cols: int,
) -> tuple[Any, Any, Any]:
    """Return (lead, normed, trail) in bf16 from an fp32 row block ``y``.

    The LayerNorm reads the bf16-rounded span in fp32, as a bf16 GEMM output
    feeding ``F.layer_norm(x.float(), ...)`` would.
    """
    lead_end = bf16_lead_cols
    norm_end = lead_end + norm_cols
    trail_end = norm_end + bf16_trail_cols
    lead = y[:, :lead_end].to(torch.bfloat16)
    span = y[:, lead_end:norm_end].to(torch.bfloat16).float()
    normed = torch.nn.functional.layer_norm(
        span, (norm_cols,), weight.float(), bias.float(), eps
    ).to(torch.bfloat16)
    trail = y[:, norm_end:trail_end].to(torch.bfloat16)
    return lead, normed, trail


def bf16_ulp_distance(torch: Any, actual: Any, expected: Any) -> Any:
    """Per-element distance in bf16 units in the last place (+0 and -0 are 0 apart)."""

    def ordered(x: Any) -> Any:
        bits = x.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        magnitude = bits & 0x7FFF
        return torch.where(bits >= 0x8000, -magnitude, magnitude)

    return (ordered(actual) - ordered(expected)).abs()


def check_row_split(
    torch: Any,
    y_before: Any,
    y_after: Any,
    outputs: tuple[Any, Any, Any],
    expected: tuple[Any, Any, Any],
) -> None:
    """The acceptance rule: lead and trail bit-identical, each normed value
    within 1 bf16 ulp (reduction order) or within ``NEAR_ZERO_ATOL`` of the
    reference, and the fp32 row (its tail is read in place as the fp32 output)
    untouched.

    The absolute allowance is for values where ``x_hat * weight + bias``
    cancels to near zero: there the fp32 rounding of the two terms (~1e-7 for
    O(1) terms) is many bf16 ulps of the tiny result (measured: 2-15 ulp at
    |y| ~ 3e-7, absolute error <= 4e-8, a few per 10^5 rows)."""
    lead, normed, trail = outputs
    lead_ref, normed_ref, trail_ref = expected
    if not torch.equal(lead, lead_ref):
        raise AssertionError("bf16 lead span is not bit-identical to the reference")
    if not torch.equal(trail, trail_ref):
        raise AssertionError("bf16 trail span is not bit-identical to the reference")
    ulp = bf16_ulp_distance(torch, normed, normed_ref)
    off = (ulp > 1) & ((normed.float() - normed_ref.float()).abs() > NEAR_ZERO_ATOL)
    if bool(off.any()):
        raise AssertionError(
            f"LayerNorm span: {int(off.sum())} values more than 1 bf16 ulp and "
            f"{NEAR_ZERO_ATOL:g} from the reference (worst {int(ulp.max())} ulp)"
        )
    if not torch.equal(y_before, y_after):
        raise AssertionError("the fp32 input row (and its fp32 tail) was modified")


__all__ = ["bf16_ulp_distance", "check_row_split", "fp32_row_split_layer_norm_reference"]
