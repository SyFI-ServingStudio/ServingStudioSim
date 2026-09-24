"""Behavioral tests for the DeepSeek-FP8 backend of `nvfp4_fused_moe` (no GPU)."""

from __future__ import annotations

import pytest
import torch

from profiling.runners.moe.exact_topk import exact_topk_ids
from profiling.runners.moe.fp8_block_fused_moe import (
    _logical_bytes,
    _validate_args,
    forced_routing_logits,
)
from profiling.runners.moe.fp8_block_fused_moe_reference import (
    deepseek_v3_routing,
    fp8_block_fused_moe_reference,
)


def _spec(**overrides: object) -> dict:
    # GLM-5.3-Flash on one EP4 rank, 16 tokens, half the local experts idle.
    batches = [0] * 288
    for expert in range(0, 64, 2):
        batches[expert] = 4
    spec = {
        "num_tokens": 16,
        "hidden_size": 4096,
        "intermediate_size": 2048,
        "num_experts": 288,
        "num_local_experts": 72,
        "top_k": 8,
        "input_dtype": "bf16",
        "weight_format": "fp8_e4m3_block",
        "group_size": 128,
        "routing_method": "deepseek_v3",
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_numerator": 5,
        "routed_scaling_denominator": 2,
        "per_expert_batches": tuple(batches),
    }
    spec.update(overrides)
    return spec


def test_forced_logits_make_deepseek_v3_routing_realize_the_histogram() -> None:
    """The timed kernel routes from logits, not from the histogram.

    If sigmoid saturation or ties let DeepSeekV3 top-k pick other experts, the
    measured call would process a different per-expert load than the row key.
    """
    spec = _spec()
    ids = exact_topk_ids(
        num_tokens=spec["num_tokens"], top_k=8, per_expert_batches=spec["per_expert_batches"]
    )
    logits = forced_routing_logits(torch, ids, 288, "cpu")

    chosen, weights = deepseek_v3_routing(
        torch, logits, torch.zeros(288), top_k=8, routed_scaling_factor=2.5
    )

    counts = torch.bincount(chosen.flatten(), minlength=288).tolist()
    assert counts == list(spec["per_expert_batches"])
    torch.testing.assert_close(weights.sum(dim=-1), torch.full((16,), 2.5))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"n_group": 8, "topk_group": 4}, "ungrouped"),
        ({"weight_format": "nvfp4_e2m1", "group_size": 16}, "fp8_e4m3_block"),
        ({"routing_method": "minimax2"}, "routing method"),
        ({"input_dtype": "fp16"}, "BF16"),
    ],
)
def test_validation_rejects_specs_the_measured_call_would_not_honor(
    override: dict, message: str
) -> None:
    """Grouped routing would silently ignore the forced histogram, and an NVFP4
    or MiniMax2 key would store an FP8/DeepSeekV3 timing under the wrong row."""
    with pytest.raises(ValueError, match=message):
        _validate_args(**_spec(**override))


def test_logical_bytes_charge_fp8_weights_only_for_active_local_experts() -> None:
    """Moving one local row onto an idle expert adds exactly one expert's FP8
    weights plus its FP32 128x128 block scales."""
    spec = _validate_args(**_spec())
    moved = list(spec["per_expert_batches"])
    moved[0] -= 1
    moved[1] += 1  # expert 1 was idle; token count per expert stays <= 16
    moved_spec = dict(spec, per_expert_batches=tuple(moved))

    hidden, intermediate = 4096, 2048
    per_expert = 3 * intermediate * hidden + 4 * 3 * (intermediate // 128) * (hidden // 128)
    assert _logical_bytes(moved_spec) - _logical_bytes(spec) == per_expert


def test_reference_applies_clamp_routed_scale_and_ep_slice() -> None:
    """One token, one hidden block: checks gate/up order, the asymmetric clamp,
    the routed scale, and that remote experts contribute nothing."""
    fp8 = torch.float8_e4m3fn
    hidden_size = intermediate = 128
    hidden = torch.ones((1, hidden_size)).to(fp8)
    hidden_scale = torch.ones((1, 1))
    # Local expert 0: gate = 128 * 0.25 = 32 (clamped to 10), up = -128 * 0.25
    # = -32 (clamped to -10). Down projection: identity-like sum over 128 lanes.
    w13 = torch.cat(
        [torch.ones((1, intermediate, hidden_size)), -torch.ones((1, intermediate, hidden_size))],
        dim=1,
    ).to(fp8)
    w13_scale = torch.full((1, 2, 1), 0.25)
    w2 = torch.ones((1, hidden_size, intermediate)).to(fp8)
    w2_scale = torch.full((1, 1, 1), 1.0 / 128)
    # Two global experts, top-2: both chosen with equal weight; expert 1 is remote.
    logits = torch.zeros((1, 2))

    out = fp8_block_fused_moe_reference(
        torch,
        hidden=hidden,
        hidden_scale=hidden_scale,
        w13=w13,
        w13_scale=w13_scale,
        w2=w2,
        w2_scale=w2_scale,
        routing_logits=logits,
        routing_bias=torch.zeros(2),
        top_k=2,
        routed_scaling_factor=2.5,
        clamp_limit=10.0,
        local_offset=0,
        num_local=1,
    )

    act = torch.nn.functional.silu(torch.tensor(10.0)) * -10.0
    expected = torch.full((1, hidden_size), float(act) * 0.5 * 2.5)
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=0.0)
