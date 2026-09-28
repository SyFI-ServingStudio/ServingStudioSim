"""Behavioral tests for the MXFP4 x MXFP8 backend of `nvfp4_fused_moe` (no GPU)."""

from __future__ import annotations

import pytest
import torch

from profiling.runners.moe.mxfp4_fused_moe import _logical_bytes, _validate_args
from profiling.runners.moe.mxfp4_fused_moe_reference import (
    dequantize_mxfp4,
    mxfp4_fused_moe_reference,
)


def _spec(**overrides: object) -> dict:
    # DeepSeek-V4.1-Flash on one EP4 rank, 16 tokens, half the local experts idle.
    batches = [0] * 384
    for expert in range(0, 96, 2):
        batches[expert] = 2
    spec = {
        "num_tokens": 16,
        "hidden_size": 5120,
        "intermediate_size": 2304,
        "num_experts": 384,
        "num_local_experts": 96,
        "top_k": 6,
        "input_dtype": "bf16",
        "weight_format": "mxfp4_e2m1",
        "group_size": 32,
        "routing_method": "precomputed_dsv4",
        "n_group": 1,
        "topk_group": 1,
        "routed_scaling_numerator": 1,
        "routed_scaling_denominator": 1,
        "per_expert_batches": tuple(batches),
    }
    spec.update(overrides)
    return spec


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"n_group": 8, "topk_group": 4}, "n_group=topk_group=1"),
        ({"routed_scaling_numerator": 5, "routed_scaling_denominator": 2}, "routed scale"),
        ({"weight_format": "nvfp4_e2m1", "group_size": 16}, "mxfp4_e2m1"),
        ({"routing_method": "deepseek_v3"}, "routing method"),
        ({"intermediate_size": 2240}, "multiple of 128"),
        ({"input_dtype": "fp16"}, "BF16"),
    ],
)
def test_validation_rejects_specs_the_measured_call_would_not_honor(
    override: dict, message: str
) -> None:
    """The call takes finished top-k ids: a grouped or scaled routing key, an
    NVFP4 key, or a non-TRT-LLM-aligned size would store this timing under a
    row describing a different call."""
    with pytest.raises(ValueError, match=message):
        _validate_args(**_spec(**override))


def test_validation_rejects_token_counts_vllm_would_chunk() -> None:
    """Past the batched-GEMM grid limit vLLM splits the call; one row must stay
    one launch sequence."""
    tokens = 90_000
    batches = [0] * 384
    for expert in range(6):
        batches[expert] = tokens
    with pytest.raises(ValueError, match="chunked"):
        _validate_args(**_spec(num_tokens=tokens, per_expert_batches=tuple(batches)))


def test_logical_bytes_charge_mxfp4_weights_only_for_active_local_experts() -> None:
    """Moving one local row onto an idle expert adds exactly one expert's packed
    E2M1 weights plus its UE8M0 group-32 scales."""
    spec = _validate_args(**_spec())
    moved = list(spec["per_expert_batches"])
    moved[0] -= 1
    moved[1] += 1
    moved_spec = dict(spec, per_expert_batches=tuple(moved))

    elements = 3 * 2304 * 5120
    assert _logical_bytes(moved_spec) - _logical_bytes(spec) == elements // 2 + elements // 32


def test_mxfp4_dequant_reads_low_nibble_first_with_ue8m0_scale() -> None:
    """A swapped nibble order or a biased exponent would silently corrupt the
    reference the kernel is checked against."""
    packed = torch.zeros((1, 16), dtype=torch.uint8)
    packed[0, 0] = 0x2F  # low nibble 0xF = -6.0, high nibble 0x2 = 1.0
    scale = torch.tensor([[126]], dtype=torch.uint8)  # 2^-1

    values = dequantize_mxfp4(torch, packed, scale)

    assert values[0, :2].tolist() == [-3.0, 0.5]
    assert values[0, 2:].abs().sum() == 0


def test_reference_applies_clamp_gate_up_order_weights_and_ep_slice() -> None:
    """One token, one block: gate = 32 (clamped to 10 from above), up = -32
    (clamped to -10); expert 1 is remote and must contribute nothing."""
    hidden_size = intermediate = 32
    hidden = torch.ones((1, hidden_size)).to(torch.float8_e4m3fn)
    hidden_scale = torch.full((1, 1), 127, dtype=torch.uint8)
    one, minus_one = 0x22, 0xAA  # two E2M1 codes of 1.0 / -1.0 per byte
    w13 = torch.cat(
        [
            torch.full((1, intermediate, hidden_size // 2), one, dtype=torch.uint8),
            torch.full((1, intermediate, hidden_size // 2), minus_one, dtype=torch.uint8),
        ],
        dim=1,
    )
    w13_scale = torch.full((1, 2 * intermediate, 1), 127, dtype=torch.uint8)
    w2 = torch.full((1, hidden_size, intermediate // 2), one, dtype=torch.uint8)
    w2_scale = torch.full((1, hidden_size, 1), 127 - 5, dtype=torch.uint8)  # 1/32

    out = mxfp4_fused_moe_reference(
        torch,
        hidden=hidden,
        hidden_scale=hidden_scale,
        w13=w13,
        w13_scale=w13_scale,
        w2=w2,
        w2_scale=w2_scale,
        topk_ids=torch.tensor([[0, 1]], dtype=torch.int32),
        topk_weights=torch.tensor([[0.25, 0.75]]),
        local_offset=0,
        num_local=1,
        clamp_limit=10.0,
        requantize_intermediate=False,
    )

    act = float(torch.nn.functional.silu(torch.tensor(10.0))) * -10.0
    expected = torch.full((1, hidden_size), act * 0.25)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=0.0)
