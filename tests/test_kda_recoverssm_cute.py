"""CPU checks for the fork's CuTe RecoverSSM runners (fused conv + verify, cute commit).

The kernels need an SM10x GPU; these tests pin the argument contract, the
operand layout the fork's ``_recoverssm_verify`` passes, and the conv oracle.
"""

from __future__ import annotations

import inspect
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch

from profiling.db.args import DType
from profiling.kernels.kda_recoverssm_conv_verify import KdaRecoverSSMConvVerifyArgs
from profiling.kernels.kda_recoverssm_commit import KdaRecoverSSMCommitArgs
from profiling.runners.attention import kda_recoverssm_vllm_cute as runner
from profiling.runners.attention.kda_recoverssm_vllm_triton import validate_commit_args

_ARGS = dict(
    batch_size=4,
    num_heads=2,
    head_dim=128,
    spec_query_len=4,
    dtype="bf16",
    conv_weight_dtype="fp32",
)


class _FakeRecords:
    @classmethod
    def allocate(cls, batch, heads, spec, dim, dtype, device):
        return SimpleNamespace(
            correction=torch.zeros(batch, heads, spec, dim, dtype=torch.float32, device=device),
            key_gate=torch.zeros(batch, heads, spec, 2 * dim, dtype=dtype, device=device),
        )


def test_args_fields_are_the_runner_kwargs() -> None:
    # Defect: schema/DB key order drifting from what the worker passes the runner.
    names = [f.name for f in fields(KdaRecoverSSMConvVerifyArgs)]
    assert names == list(_ARGS)
    shape = runner.validate_conv_verify_args(**_ARGS, dim_first=False)
    assert shape.conv_weight_dtype is DType.FP32
    commit = [f.name for f in fields(KdaRecoverSSMCommitArgs)]
    signature = inspect.signature(runner.profile_kda_recoverssm_commit_vllm_cute)
    assert list(signature.parameters) == commit


@pytest.mark.parametrize(
    "override",
    [
        {"head_dim": 64},
        {"spec_query_len": 9},
        {"spec_query_len": 1},
        {"dtype": "fp16"},
        {"conv_weight_dtype": "fp16"},
        {"batch_size": 0},
    ],
)
def test_conv_verify_rejects_what_can_use_cute_recoverssm_refuses(override: dict) -> None:
    # Defect: a row for a shape the fork would route to the Triton fallback.
    with pytest.raises(ValueError):
        runner.validate_conv_verify_args(**{**_ARGS, **override}, dim_first=False)


@pytest.mark.parametrize("state_dtype", ["fp32"])
def test_cute_commit_rejects_an_fp32_checkpoint(state_dtype: str) -> None:
    # Defect: timing the fold on an fp32 pool, whose results it gets wrong.
    with pytest.raises(ValueError):
        runner.profile_kda_recoverssm_commit_vllm_cute(
            batch_size=4,
            commit_len=4,
            num_heads=2,
            head_dim=128,
            spec_query_len=8,
            num_layers=2,
            dtype="bf16",
            state_dtype=state_dtype,
        )


@pytest.mark.parametrize("dim_first", [False, True])
def test_operands_follow_the_fork_layout(dim_first: bool) -> None:
    # Defect: timing contiguous q/k/v copies, a conv page in the wrong layout,
    # or a checkpoint outside the page the conv shares.
    shape = runner.validate_conv_verify_args(**_ARGS, dim_first=dim_first)
    ops = runner.build_conv_verify_operands(torch, _FakeRecords, shape, device=torch.device("cpu"))
    seen: dict = {}

    def fake(**kwargs):
        seen.update(kwargs)
        return True

    runner.invoke_conv_verify(fake, ops)
    tokens, channels = 16, 3 * 256
    assert seen["mixed_qkv"].shape == (tokens, channels)
    assert seen["mixed_qkv"].stride(0) == shape.projected_width
    assert seen["raw_beta"].data_ptr() == ops.projected.data_ptr() + 2 * channels
    assert seen["conv_weight"].dtype is torch.float32 and seen["conv_weight"].shape == (channels, 4)
    assert seen["state"].dtype is torch.bfloat16 and seen["state"].shape == (5, 2, 128, 128)
    rows = shape.conv_state_len
    expected = (5, channels, rows) if dim_first else (5, rows, channels)
    assert seen["conv_state"].shape == expected and seen["conv_state"].is_contiguous() is False
    assert seen["conv_state"].untyped_storage().data_ptr() == seen["state"].untyped_storage().data_ptr()
    assert seen["query_start_loc"].tolist() == [0, 4, 8, 12, 16]
    assert seen["num_accepted_tokens"].tolist() == [1, 2, 3, 4]
    assert sorted(seen["state_indices"].tolist()) == [1, 2, 3, 4]
    assert seen["lower_bound"] == -5.0

    with pytest.raises(Exception):
        runner.invoke_conv_verify(lambda **_: False, ops)


def test_conv_reference_rolls_the_history_ending_at_the_accepted_token() -> None:
    # Defect: an off-by-one in the history window the conv and the rolled page use.
    shape = runner.validate_conv_verify_args(**{**_ARGS, "batch_size": 2}, dim_first=False)
    ops = runner.build_conv_verify_operands(torch, _FakeRecords, shape, device=torch.device("cpu"))
    out, windows = runner.conv_window_reference(torch, ops, shape, 2)
    spec, rows = shape.spec_query_len, shape.conv_state_len
    for b in range(2):
        slot, accepted = int(ops.state_indices[b]), int(ops.accepted[b])
        history = ops.conv[slot, accepted - 1 : accepted + 2]
        window = ops.mixed_qkv[b * spec : (b + 1) * spec]
        assert torch.equal(windows[b], torch.cat([history, window])[-rows:])
        full = torch.cat([history.float(), window.float()])
        last = (full[-4:] * ops.conv_weight.float().T).sum(0)
        assert torch.allclose(out[(b + 1) * spec - 1], torch.nn.functional.silu(last), atol=1e-5)


def test_cute_commit_context_gets_request_row_records_and_the_cute_backend() -> None:
    # Defect: handing the fold slot-indexed records (the fork keeps them by
    # request row) or letting the context default to the Triton fold.
    shape = validate_commit_args(4, 3, 2, 128, 8, 2, "bf16", "bf16")
    created: dict = {}

    class FakeContext:
        @classmethod
        def create(cls, layers, *, spec_query_len, max_num_reqs, backend):
            created.update(layers=layers, spec=spec_query_len, reqs=max_num_reqs, backend=backend)
            return cls()

    ops = runner.build_cute_commit_operands(torch, FakeContext, shape, device=torch.device("cpu"))
    assert created["backend"] == "cute" and created["reqs"] == 4 and len(created["layers"]) == 2
    layer, page = created["layers"][0], ops.pages[0]
    assert len(layer.kv_cache) == 2
    slots = ops.state_indices.long()
    for row, slot in enumerate(slots.tolist()):
        assert torch.equal(layer.recoverssm_records.correction[row], page.correction[slot])
        assert torch.equal(layer.recoverssm_records.key_gate[row], page.kg[slot])
