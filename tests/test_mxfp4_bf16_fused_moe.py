"""Behavioral tests for the MXFP4-weight / BF16-activation backend of
`nvfp4_fused_moe` (no GPU)."""

from __future__ import annotations

import pytest
import torch

from launcher.backends import dedup_roles, validate_backend_map
from profiling.kernels.nvfp4_fused_moe import KIND
from profiling.runners.moe.exact_topk import exact_topk_ids
from profiling.runners.moe.mxfp4_bf16_fused_moe import _logical_bytes, _validate_args
from profiling.runners.moe.mxfp4_bf16_fused_moe_reference import mxfp4_bf16_fused_moe_reference
from profiling.runners.moe.nvfp4_fused_moe import forced_routing

BACKEND = "flashinfer_trtllm_sm100_mxfp4_bf16"


def _spec(**overrides: object) -> dict:
    # One EP4 rank of a 256-expert top-8 model, 16 tokens, half its experts idle.
    batches = [0] * 256
    for expert in range(0, 64, 2):
        batches[expert] = 2
    for expert in range(64, 128):
        batches[expert] = 1
    spec = {
        "num_tokens": 16,
        "hidden_size": 4096,
        "intermediate_size": 2048,
        "num_experts": 256,
        "num_local_experts": 64,
        "top_k": 8,
        "input_dtype": "bf16",
        "weight_format": "mxfp4_e2m1",
        "group_size": 32,
        "routing_method": "deepseek_v3",
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
        ({"n_group": 8, "topk_group": 4}, "ungrouped"),
        ({"routing_method": "precomputed_dsv4"}, "routing method"),
        ({"routing_method": "minimax2"}, "routing method"),
        ({"weight_format": "nvfp4_e2m1", "group_size": 16}, "mxfp4_e2m1"),
        ({"intermediate_size": 2240}, "multiple of 128"),
        ({"input_dtype": "fp16"}, "BF16"),
    ],
)
def test_validation_rejects_specs_the_measured_call_would_not_honor(
    override: dict, message: str
) -> None:
    """Grouped routing ignores a forced histogram, the precomputed and MiniMax2
    methods are other calls, and an NVFP4 key would describe another weight
    recipe: each would store this timing under a row for a different call."""
    with pytest.raises(ValueError, match=message):
        _validate_args(**_spec(**override))


def test_logical_bytes_charge_bf16_activations_and_fp32_routing() -> None:
    """One more local row costs exactly one BF16 hidden row (no scale); one
    more token adds its FP32 logits row and one BF16 output row."""
    spec = _validate_args(**_spec())
    moved = list(spec["per_expert_batches"])
    moved[64] -= 1  # a remote row becomes a local row on an active expert
    moved[0] += 1
    local = dict(spec, per_expert_batches=tuple(moved))
    assert _logical_bytes(local) - _logical_bytes(spec) == 2 * 4096

    idle = list(spec["per_expert_batches"])
    idle[1] += 1  # a remote row lands on an idle local expert
    idle[64] -= 1
    woken = dict(spec, per_expert_batches=tuple(idle))
    elements = 3 * 2048 * 4096
    assert _logical_bytes(woken) - _logical_bytes(spec) == (
        2 * 4096 + elements // 2 + elements // 32
    )

    assert _logical_bytes(dict(spec, num_tokens=17)) - _logical_bytes(spec) == (4 * 256 + 2 * 4096)


def test_bf16_compute_dtype_selects_only_the_weight_only_backend() -> None:
    """A config whose activation_format makes the compute dtype bf16 must not
    be offered the MXFP8 backend (it needs a quantization launch the config does
    not model), and an MXFP4-quantized config must not get the BF16 one."""
    record = {
        "pool": "main",
        "name": "unified.moe.experts",
        "kind": KIND,
        "gpu": "NVIDIA B200",
        "compute_dtype": "bf16",
        "kv_dtype": None,
        "config": {},
        "backends": [BACKEND],
    }
    (role,) = dedup_roles([record])
    assert role.options == [BACKEND]
    assert validate_backend_map({"main": {"unified.moe.experts": [BACKEND]}}, [role]) == []
    (mxfp8_role,) = dedup_roles([{**record, "compute_dtype": "mxfp4_e2m1"}])
    errors = validate_backend_map({"main": {"unified.moe.experts": [BACKEND]}}, [mxfp8_role])
    assert any("unsupported for nvfp4_fused_moe at dtype=mxfp4_e2m1" in e for e in errors)


def test_forced_logits_route_the_exact_histogram_through_the_reference() -> None:
    """The reference routes from the same FP32 logits the runner feeds the
    kernel, so a histogram it does not reproduce would mean the oracle checks
    a different expert assignment than the one timed."""
    args = _validate_args(**_spec())
    ids = exact_topk_ids(num_tokens=16, top_k=8, per_expert_batches=args["per_expert_batches"])
    logits, bias = forced_routing(torch, ids, args, device="cpu")
    assert logits.dtype is torch.float32 and bias.dtype is torch.float32

    hidden_size = intermediate = 32
    hidden = torch.zeros((16, hidden_size), dtype=torch.bfloat16)
    zeros13 = torch.zeros((64, 2 * intermediate, hidden_size // 2), dtype=torch.uint8)
    zeros2 = torch.zeros((64, hidden_size, intermediate // 2), dtype=torch.uint8)
    scale13 = torch.full((64, 2 * intermediate, 1), 127, dtype=torch.uint8)
    scale2 = torch.full((64, hidden_size, 1), 127, dtype=torch.uint8)
    out = mxfp4_bf16_fused_moe_reference(
        torch,
        hidden=hidden,
        logits=logits,
        bias=bias,
        w13=zeros13,
        w13_scale=scale13,
        w2=zeros2,
        w2_scale=scale2,
        top_k=8,
        routed_scale=1.0,
        local_offset=0,
        num_local=64,
        bf16_boundaries=True,
    )
    assert out.shape == (16, hidden_size) and not out.any()

    from profiling.runners.moe.fp8_block_fused_moe_reference import deepseek_v3_routing

    chosen, _ = deepseek_v3_routing(torch, logits, bias, top_k=8, routed_scaling_factor=1.0)
    counts = torch.bincount(chosen.flatten(), minlength=256).tolist()
    assert counts == list(args["per_expert_batches"])


def test_reference_keeps_bf16_input_unclamped_gate_up_order_and_ep_slice() -> None:
    """One token in BF16, one 32-wide block: gate = 32 and up = -32 pass
    unclamped (no swiglu limit), the route weight is the unbiased sigmoid
    share, and a remote expert contributes nothing."""
    hidden_size = intermediate = 32
    hidden = torch.ones((1, hidden_size), dtype=torch.bfloat16)
    one, minus_one = 0x22, 0xAA  # two E2M1 codes of 1.0 / -1.0 per byte
    w13 = torch.cat(
        [
            torch.full((2, intermediate, hidden_size // 2), one, dtype=torch.uint8),
            torch.full((2, intermediate, hidden_size // 2), minus_one, dtype=torch.uint8),
        ],
        dim=1,
    )
    w13_scale = torch.full((2, 2 * intermediate, 1), 127, dtype=torch.uint8)
    w2 = torch.full((2, hidden_size, intermediate // 2), one, dtype=torch.uint8)
    w2_scale = torch.full((2, hidden_size, 1), 127 - 5, dtype=torch.uint8)  # 1/32
    # Expert 0 is local (offset 0, one local expert), expert 1 remote; the bias
    # promotes expert 1 into the top-2 but does not enter the weights.
    logits = torch.tensor([[0.0, -2.0, -8.0, -8.0]])
    bias = torch.tensor([0.0, 1.0, 0.0, 0.0])

    out = mxfp4_bf16_fused_moe_reference(
        torch,
        hidden=hidden,
        logits=logits,
        bias=bias,
        w13=w13[:1],
        w13_scale=w13_scale[:1],
        w2=w2[:1],
        w2_scale=w2_scale[:1],
        top_k=2,
        routed_scale=2.0,
        local_offset=0,
        num_local=1,
        bf16_boundaries=False,
    )

    s0, s1 = torch.sigmoid(torch.tensor(0.0)), torch.sigmoid(torch.tensor(-2.0))
    weight = float(2.0 * s0 / (s0 + s1))
    act = float(torch.nn.functional.silu(torch.tensor(32.0))) * -32.0
    expected = torch.full((1, hidden_size), act * weight)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=0.0)
