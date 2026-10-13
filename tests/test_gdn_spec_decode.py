"""CPU tests for the Gated DeltaNet speculative (MTP verify) decode references and specs."""

from __future__ import annotations

import pytest
import torch

from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.db.registry import find_kernel_profiler_spec, known_backends
from profiling.kernels.gdn_causal_conv_decode import GdnCausalConvDecodeArgs
from profiling.kernels.gdn_recurrent_decode_gated_norm import KIND as GATED_NORM_KIND
from profiling.runners.attention import _gdn_conv_verify as conv_verify
from profiling.runners.attention._gdn_common import tokens_per_request
from profiling.runners.attention._gdn_verify import build_verify_recurrence_operands
from profiling.runners.attention.gdn_causal_conv_decode_reference import (
    gdn_causal_conv_decode_reference,
)
from profiling.runners.attention.gdn_recurrent_decode_reference import (
    gdn_recurrent_decode_reference,
)
from profiling.runners.attention.gdn_spec_decode_reference import (
    gdn_causal_conv_spec_decode_reference,
    gdn_recurrent_spec_decode_gated_norm_reference,
    gdn_recurrent_spec_decode_reference,
)

_GEOMETRY = dict(num_qk_heads=2, num_value_heads=4, key_head_dim=8, value_head_dim=8)


def _recurrence(batch: int, per_request: int):
    return build_verify_recurrence_operands(
        torch,
        batch_size=batch,
        per_request=per_request,
        **_GEOMETRY,
        dtype=torch.bfloat16,
        state_dtype=torch.float32,
        device="cpu",
    )


def test_one_token_spec_is_absent_and_keys_the_old_rows() -> None:
    # Catches a verify arg leaking into one-token keys (it would orphan every row).
    spec = {"batch_size": 8, "channels": 64, "kernel_size": 4, "dtype": "bf16"}
    args = coerce_args(GdnCausalConvDecodeArgs, spec | {"state_dtype": "bf16", "num_tokens": 8})
    assert args.num_tokens is None
    verify = coerce_args(GdnCausalConvDecodeArgs, spec | {"state_dtype": "bf16", "num_tokens": 32})
    assert verify.num_tokens == 32
    assert tokens_per_request(8, None) == 1 and tokens_per_request(8, 32) == 4
    with pytest.raises(ValueError):
        tokens_per_request(8, 30)


def test_conv_verify_matches_sequential_one_token_decode_from_the_accepted_history() -> None:
    # Catches a wrong history offset (num_accepted - 1) or a wrong rewrite of the row.
    batch, per_request, channels, width = 3, 4, 16, 4
    ops = conv_verify.build_operands(
        torch,
        batch_size=batch,
        per_request=per_request,
        channels=channels,
        kernel_size=width,
        dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        device="cpu",
        state_dim_first=True,
    )
    state = ops.state.clone()
    output, state = gdn_causal_conv_spec_decode_reference(
        ops.x, ops.weight, state, ops.state_indices, ops.num_accepted
    )
    for request in range(batch):
        slot = int(ops.state_indices[request, 0])
        accepted = int(ops.num_accepted[request])
        history = ops.state[slot, :, accepted - 1 : accepted + width - 2].clone()
        one_state = history.unsqueeze(0).clone()
        for token in range(per_request):
            row = request * per_request + token
            expected, one_state = _one_token(ops.x[row], ops.weight, one_state)
            assert torch.equal(output[row], expected)
        assert torch.equal(state[slot, :, -(width - 1) :], one_state[0])
        assert torch.equal(
            state[slot, :, : width - 2], ops.state[slot, :, accepted : accepted + width - 2]
        )


def _one_token(x: torch.Tensor, weight: torch.Tensor, history: torch.Tensor):
    """Slot-free one-token decode through the existing reference (slot 1 of a 2-slot pool)."""
    pool = torch.cat((torch.zeros_like(history), history), dim=0)
    output, pool = gdn_causal_conv_decode_reference(
        x.unsqueeze(0).clone(), weight, pool, torch.tensor([1], dtype=torch.int32)
    )
    return output[0], pool[1:]


def test_recurrent_verify_matches_sequential_decode_and_writes_every_checkpoint() -> None:
    # Catches reading the wrong source checkpoint or skipping per-token writes.
    batch, per_request = 3, 4
    ops = _recurrence(batch, per_request)
    query, key, value = ops.semantic_qkv()
    a, b = ops.a.contiguous(), ops.b.contiguous()
    pool = ops.state.clone()
    output, pool = gdn_recurrent_spec_decode_reference(
        query, key, value, a, b, ops.A_log, ops.dt_bias, pool, ops.state_indices, ops.num_accepted
    )
    for request in range(batch):
        source = int(ops.state_indices[request, int(ops.num_accepted[request]) - 1])
        running = ops.state[source].transpose(-1, -2).unsqueeze(0).contiguous()
        for token in range(per_request):
            row = slice(request * per_request + token, request * per_request + token + 1)
            expected, running = gdn_recurrent_decode_reference(
                query[row], key[row], value[row], a[row], b[row], ops.A_log, ops.dt_bias, running
            )
            assert torch.equal(output[row], expected)
            slot = int(ops.state_indices[request, token])
            assert torch.equal(pool[slot], running[0].transpose(-1, -2))
    assert torch.equal(pool[0], ops.state[0])


def test_gated_norm_reference_is_rmsnorm_times_sigmoid_gate_of_the_bf16_core() -> None:
    # Catches normalizing the FP32 core instead of its BF16 rounding, or a SiLU gate.
    ops = _recurrence(2, 3)
    query, key, value = ops.semantic_qkv()
    inputs = (query, key, value, ops.a.contiguous(), ops.b.contiguous(), ops.A_log, ops.dt_bias)
    core, _ = gdn_recurrent_spec_decode_reference(
        *inputs, ops.state.clone(), ops.state_indices, ops.num_accepted
    )
    gated, _ = gdn_recurrent_spec_decode_gated_norm_reference(
        *inputs,
        ops.state.clone(),
        ops.state_indices,
        ops.num_accepted,
        ops.output_gate,
        ops.norm_weight,
        1e-6,
    )
    x = core.float()
    expected = (
        x
        / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        * ops.norm_weight.float()
        * torch.sigmoid(ops.output_gate.float())
    ).to(torch.bfloat16)
    assert torch.equal(gated, expected)


def test_production_views_are_strided_like_vllm() -> None:
    # The fused kernel must see qkvz/ba row strides, not compacted copies.
    ops = _recurrence(2, 4)
    width = 2 * 2 * 8 + 2 * 4 * 8
    assert ops.mixed_qkv.stride() == (width, 1)
    assert ops.output_gate.stride() == (width, 8, 1)
    assert ops.a.stride() == (8, 1) and ops.b.stride() == (8, 1)
    assert ops.state_indices.tolist() == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert ops.num_accepted.tolist() == [1, 2]


@pytest.mark.parametrize(
    ("module", "function", "spec"),
    [
        (
            "profiling.runners.attention.gdn_causal_conv_decode_torch",
            "profile_gdn_causal_conv_decode",
            dict(batch_size=4, channels=64, kernel_size=4, dtype="bf16", state_dtype="bf16"),
        ),
        (
            "profiling.runners.attention.gdn_recurrent_decode_torch",
            "profile_gdn_recurrent_decode",
            dict(batch_size=4, **_GEOMETRY, dtype="bf16", state_dtype="fp32"),
        ),
        (
            "profiling.runners.attention.gdn_recurrent_decode_vllm_triton",
            "profile_gdn_recurrent_decode_vllm_triton",
            dict(batch_size=4, **_GEOMETRY, dtype="bf16", state_dtype="fp32"),
        ),
    ],
)
def test_one_token_backends_refuse_verify_specs_before_cuda(module, function, spec) -> None:
    # A verify spec must never be silently priced as one-token decode.
    import importlib

    runner = getattr(importlib.import_module(module), function)
    with pytest.raises(ValueError, match="verify step"):
        runner(**spec, num_tokens=16)


@pytest.mark.parametrize(
    "override",
    [
        {"num_tokens": 36},  # 9 tokens per request > kernel's 8
        {"key_head_dim": 64},
        {"num_value_heads": 96, "num_qk_heads": 16},  # ratio 6 is not instantiated
    ],
)
def test_gated_norm_runner_rejects_shapes_the_kernel_cannot_run(override) -> None:
    from profiling.runners.attention.gdn_recurrent_decode_gated_norm_vllm_cuda import (
        profile_gdn_recurrent_decode_gated_norm_vllm_cuda,
    )

    spec = dict(
        batch_size=4,
        num_tokens=16,
        num_qk_heads=16,
        num_value_heads=48,
        key_head_dim=128,
        value_head_dim=128,
        dtype="bf16",
        state_dtype="fp32",
    )
    with pytest.raises(ValueError):
        profile_gdn_recurrent_decode_gated_norm_vllm_cuda(**(spec | override))


def test_gated_norm_kind_is_the_vllm_cuda_op_on_sm80_plus() -> None:
    spec = find_kernel_profiler_spec(GATED_NORM_KIND, "vllm_cuda")
    assert known_backends(GATED_NORM_KIND) == ["vllm_cuda"]
    assert spec.subprocess_env == "vllm_env"
    assert spec.supports.min_compute_capability == (8, 0)
    assert spec.supports.compute == frozenset({DType.BF16})
