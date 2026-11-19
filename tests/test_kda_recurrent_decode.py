"""Focused CPU tests for ``kda_recurrent_decode`` (no GPU, no DB).

Each test names the defect it guards against. The GPU comparison of the
vendored Triton callable against the oracle runs inside the runner, and its
evidence comes from the Slurm smoke.
"""

from __future__ import annotations

import inspect
from dataclasses import fields

import pytest
import torch

from profiling.db.args import DType
from profiling.db.batch import coerce_args
from profiling.kernels.kda_recurrent_decode import KdaRecurrentDecodeArgs
from profiling.runners.attention import kda_recurrent_decode_vllm_triton as runner
from profiling.runners.attention.kda_chunk_prefill_reference import (
    kda_chunk_prefill_reference,
    kda_safe_gate,
)
from profiling.runners.attention.kda_recurrent_decode_reference import (
    kda_recurrent_decode_reference,
    kda_spec_verify_reference,
)

# The in-situ capture: decode iteration 727, 32 requests, TP4.
_CAPTURE = dict(batch_size=32, num_heads=16, head_dim=128, dtype="bf16")


def test_args_fields_are_the_runner_kwargs() -> None:
    # Defect: schema/DB key order drifting from what the worker passes the runner.
    names = [f.name for f in fields(KdaRecurrentDecodeArgs)]
    assert names == ["batch_size", "num_heads", "head_dim", "dtype", "num_tokens", "state_dtype"]
    signature = inspect.signature(runner.profile_kda_recurrent_decode_vllm_triton)
    assert list(signature.parameters) == names


@pytest.mark.parametrize(
    "override",
    [
        {"head_dim": 0},
        {"dtype": "fp16"},
        {"batch_size": 0},
        {"num_heads": 0},
        {"batch_size": True},
    ],
)
def test_rejects_shapes_outside_the_verified_path(override: dict) -> None:
    # Defect: silently profiling a path production never takes.
    with pytest.raises(ValueError):
        runner.validate_args(**{**_CAPTURE, **override})


@pytest.mark.parametrize("head_dim", [64, 96, 128, 256])
def test_accepts_any_positive_head_dim(head_dim: int) -> None:
    # Defect: refusing an unmeasured head dim the kernel launches (BK is
    # next_power_of_2(K), so NK == 1 for every K).
    assert runner.validate_args(**{**_CAPTURE, "head_dim": head_dim}).head_dim == head_dim


@pytest.mark.parametrize("batch", [1, 7, 8, 32, 128])
def test_state_slots_are_distinct_and_never_null(batch: int) -> None:
    # Defect: two sequences sharing a slot (a write race) or slot 0 (the kernel
    # skips NULL_BLOCK_ID and would time an early return).
    slots = runner.state_slots(batch)
    assert sorted(slots) == list(range(1, batch + 1))


def test_invoke_reproduces_the_production_call() -> None:
    # Defect: timing contiguous q/k/v/beta (skips the callable's four copy
    # launches), a pre-sigmoided beta, a precomputed gate, or a
    # non-in-place state and output.
    shape = runner.validate_args(batch_size=3, num_heads=2, head_dim=128, dtype="bf16")
    operands = runner.build_operands(torch, shape, device=torch.device("cpu"))
    seen: dict = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return kwargs["out"], kwargs["initial_state"]

    runner.invoke(fake, operands)
    width = 3 * 256 + 2 + 2 * 128
    assert operands.projected.shape == (3, width)
    storage = operands.projected.untyped_storage().data_ptr()
    for name in ("q", "k", "v"):
        view = seen[name]
        assert view.shape == (1, 3, 2, 128) and view.dtype is torch.bfloat16
        assert not view.is_contiguous() and view.stride(1) == width
        assert view.untyped_storage().data_ptr() == storage
    beta = seen["beta"]
    assert beta.shape == (1, 3, 2) and beta.dtype is torch.bfloat16
    assert not beta.is_contiguous() and beta.untyped_storage().data_ptr() == storage
    assert seen["g"].is_contiguous() and seen["g"].dtype is torch.bfloat16
    assert seen["a_log"].shape == (1, 1, 2, 1) and seen["g_bias"].shape == (256,)
    assert seen["initial_state"].shape == (4, 2, 128, 128)
    assert seen["initial_state"].dtype is torch.float32
    assert seen["ssm_state_indices"].dtype is torch.int32
    assert seen["cu_seqlens"].tolist() == [0, 1, 2, 3]
    assert seen["out"].shape == (1, 3, 2, 128) and seen["out"].is_contiguous()
    assert seen["compute_gate"] is True and seen["sigmoid_beta"] is True
    assert seen["lower_bound"] == -5.0 and seen["use_qk_l2norm_in_kernel"] is True
    assert "inplace_final_state" not in seen  # production relies on the True default


def _inputs(batch: int, heads: int = 2, dim: int = 4, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(batch, heads, dim, generator=generator) for _ in range(4))


def test_oracle_matches_one_step_of_the_chunk_oracle_through_the_slot_pool() -> None:
    # Defect: gathering/scattering the wrong slot, [H,V,K] vs [H,K,V] layout,
    # or touching slots no sequence owns. One decode step is a length-1 sequence
    # of the independent per-token chunk oracle (fp32 inputs: no l2norm rounding).
    q, k, v, raw_g = _inputs(3)
    raw_beta = torch.randn(3, 2, generator=torch.Generator().manual_seed(1))
    a_log = torch.tensor([0.5, 1.5]).log().view(1, 1, 2, 1)
    dt_bias = torch.randn(8, generator=torch.Generator().manual_seed(2)) * 0.1
    pool = torch.randn(6, 2, 4, 4, generator=torch.Generator().manual_seed(3))
    slots = torch.tensor([4, 1, 3], dtype=torch.int32)
    o, updated = kda_recurrent_decode_reference(
        q, k, v, raw_g, raw_beta, a_log, dt_bias, pool, slots, lower_bound=-5.0
    )
    expected_o, expected_state = kda_chunk_prefill_reference(
        q,
        k,
        v,
        raw_g,
        raw_beta.sigmoid(),
        a_log,
        dt_bias,
        pool[slots.long()],
        [0, 1, 2, 3],
        lower_bound=-5.0,
    )
    torch.testing.assert_close(o, expected_o)
    torch.testing.assert_close(updated[slots.long()], expected_state)
    for untouched in (0, 2, 5):
        assert torch.equal(updated[untouched], pool[untouched])


def test_oracle_decays_per_key_channel_and_sigmoids_raw_beta() -> None:
    # Defect: per-head (scalar) decay, decay on the value axis, or treating
    # raw_beta as already sigmoided. With raw_beta -> -inf, sigmoid is 0 and
    # nothing is written, so the state is only decayed along K.
    q, k, v, raw_g = _inputs(2)
    a_log, dt_bias = torch.zeros(1, 1, 2, 1), torch.zeros(8)
    pool = torch.randn(3, 2, 4, 4, generator=torch.Generator().manual_seed(4))
    slots = torch.tensor([2, 1], dtype=torch.int32)
    _, updated = kda_recurrent_decode_reference(
        q, k, v, raw_g, torch.full((2, 2), -1e4), a_log, dt_bias, pool, slots, lower_bound=-5.0
    )
    decay = kda_safe_gate(raw_g, a_log, dt_bias, -5.0).exp()  # [B, H, K]
    for seq, slot in enumerate(slots.tolist()):
        torch.testing.assert_close(updated[slot], pool[slot] * decay[seq][:, None, :])


def test_fp32_state_and_one_token_keep_the_plain_decode_key() -> None:
    # Defect: the verify fields re-keying the measured plain-decode rows (fp32
    # pool, one token per request), or a bf16 / verify spec collapsing onto them.
    plain = coerce_args(
        KdaRecurrentDecodeArgs, {**_CAPTURE, "num_tokens": 32, "state_dtype": "fp32"}
    )
    assert plain == coerce_args(KdaRecurrentDecodeArgs, _CAPTURE)
    assert plain.num_tokens is None and plain.state_dtype is None
    verify = coerce_args(
        KdaRecurrentDecodeArgs, {**_CAPTURE, "num_tokens": 256, "state_dtype": "bf16"}
    )
    assert (verify.num_tokens, verify.state_dtype) == (256, DType.BF16)


def test_verify_invoke_passes_the_spec_branch_operands() -> None:
    # Defect: timing a verify as B*q one-token decodes (1-D indices, no
    # num_accepted_tokens), or a pool at the wrong dtype.
    shape = runner.validate_args(
        batch_size=3, num_heads=2, head_dim=128, dtype="bf16", num_tokens=12, state_dtype="bf16"
    )
    operands = runner.build_operands(torch, shape, device=torch.device("cpu"))
    seen: dict = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return kwargs["out"], kwargs["initial_state"]

    runner.invoke(fake, operands)
    assert seen["q"].shape == (1, 12, 2, 128)
    assert seen["ssm_state_indices"].tolist() == [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    assert seen["num_accepted_tokens"].tolist() == [1, 2, 3]
    assert seen["cu_seqlens"].tolist() == [0, 4, 8, 12]
    assert seen["initial_state"].shape == (13, 2, 128, 128)
    assert seen["initial_state"].dtype is torch.bfloat16
    assert seen["out"].shape == (1, 12, 2, 128)


def test_verify_oracle_starts_at_the_accepted_slot_and_stores_every_position() -> None:
    # Defect: starting from slot 0 of the row instead of the last accepted
    # token's slot, or storing only the final state. Token t of a request
    # equals the plain-decode oracle chained from the accepted state.
    requests, per_request = 2, 3
    q, k, v, raw_g = _inputs(requests * per_request)
    raw_beta = torch.randn(6, 2, generator=torch.Generator().manual_seed(1))
    a_log = torch.tensor([0.5, 1.5]).log().view(1, 1, 2, 1)
    dt_bias = torch.randn(8, generator=torch.Generator().manual_seed(2)) * 0.1
    pool = torch.randn(7, 2, 4, 4, generator=torch.Generator().manual_seed(3))
    slots = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    accepted = torch.tensor([2, 3], dtype=torch.int32)
    o, updated = kda_spec_verify_reference(
        q, k, v, raw_g, raw_beta, a_log, dt_bias, pool, slots, accepted, lower_bound=-5.0
    )
    for n in range(requests):
        chain = pool.clone()
        chain[0] = pool[int(slots[n, accepted[n] - 1])]
        for t in range(per_request):
            row = n * per_request + t
            step_o, chain = kda_recurrent_decode_reference(
                q[row : row + 1],
                k[row : row + 1],
                v[row : row + 1],
                raw_g[row : row + 1],
                raw_beta[row : row + 1],
                a_log,
                dt_bias,
                chain,
                torch.tensor([0], dtype=torch.int32),
                lower_bound=-5.0,
            )
            torch.testing.assert_close(o[row], step_o[0])
            torch.testing.assert_close(updated[int(slots[n, t])], chain[0])
    assert torch.equal(updated[0], pool[0])
