"""CPU tests for the ``gdn_causal_conv_prefill`` ``dao_channellast`` runner.

A torch stand-in for ``causal_conv1d_fn`` exercises the runner's operand views,
correctness guards and timed call without CUDA or the causal-conv1d package.
"""

from __future__ import annotations

import pytest

from profiling.db.args import DType
from profiling.db.registry import find_kernel_profiler_spec
from profiling.kernels.gdn_causal_conv_prefill import KIND, GdnCausalConvPrefillArgs

_SPEC = {
    "batch_size": 3,
    "sequence_length": 5,
    "channels": 16,
    "kernel_size": 4,
    "dtype": "bf16",
    "state_dtype": "bf16",
}


def _runner():
    import profiling.runners.attention.gdn_causal_conv_prefill_dao_channellast as runner

    return runner


def _fake_causal_conv1d_fn(*, ignore_initial_states: bool = False):
    """causal_conv1d_fn's contract in torch: x (1, C, L), optional (1, C, W-1)
    history, SiLU output, last W-1 samples of history ++ x into final_states_out."""
    import torch
    import torch.nn.functional as F

    def fn(x, weight, bias, *, initial_states, return_final_states, final_states_out, activation):
        assert bias is None and return_final_states and activation == "silu"
        assert x.stride(1) == 1 and final_states_out.stride(1) == 1
        _, channels, length = x.shape
        state_length = weight.shape[1] - 1
        history = (
            torch.zeros(1, channels, state_length, dtype=x.dtype)
            if initial_states is None or ignore_initial_states
            else initial_states
        )
        extended = torch.cat([history, x], dim=2).float()
        acc = sum(
            extended[..., k : k + length] * weight[:, k].float().view(1, channels, 1)
            for k in range(weight.shape[1])
        )
        final_states_out.copy_(extended[..., -state_length:].to(final_states_out.dtype))
        return F.silu(acc).to(x.dtype), final_states_out

    return fn


def _cpu_operands(runner, **overrides):
    import torch

    args = runner._validate_args(**(_SPEC | overrides))
    return args, runner._build_operands(torch, args, device=torch.device("cpu"))


def test_registration_shares_the_kind_schema_and_runs_on_the_project_interpreter() -> None:
    spec = find_kernel_profiler_spec(KIND, "dao_channellast")

    assert spec.kernel_kind == spec.table_name == KIND
    assert spec.args_schema is GdnCausalConvPrefillArgs
    assert spec.subprocess_env == "causal_conv1d_env"
    assert spec.supports.compute == frozenset({DType.BF16})
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA B200")
    assert spec.supports.allows(DType.BF16, gpu="NVIDIA H200")
    assert not spec.supports.allows(DType.FP16, gpu="NVIDIA B200")
    # The rows carry the package version, which the backend name cannot supply.
    assert spec.load_row_provenance() is not None


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"channels": 12}, "divisible by 8"),
        ({"kernel_size": 5}, "kernel_size"),
        ({"dtype": "fp16", "state_dtype": "fp16"}, "bf16"),
        ({"sequence_length": 0}, "> 0"),
    ],
)
def test_runner_rejects_shapes_the_channel_last_kernel_cannot_run(override, match) -> None:
    with pytest.raises(ValueError, match=match):
        _runner()._validate_args(**(_SPEC | override))


def test_guard_channels_stay_a_multiple_of_eight() -> None:
    runner = _runner()
    args = runner._validate_args(
        **(_SPEC | {"batch_size": 7, "sequence_length": 1000, "channels": 24576})
    )
    guard = runner._guard_args(args)
    assert guard.channels % 8 == 0
    assert guard.batch_size * guard.sequence_length * guard.channels <= 1_048_576
    assert (guard.batch_size, guard.sequence_length) == (7, 1000)


def test_sequence_views_are_channel_last_and_write_their_own_slot() -> None:
    import torch

    runner = _runner()
    args, operands = _cpu_operands(runner)
    views = runner._sequence_views(operands, args)

    assert len(views) == args.batch_size
    for index, (x, final_state) in enumerate(views):
        assert tuple(x.shape) == (1, args.channels, args.sequence_length)
        assert x.stride(1) == 1
        assert torch.equal(x[0].t(), operands.semantic_x[index])
        assert tuple(final_state.shape) == (1, args.channels, args.kernel_size - 1)
        final_state.fill_(index + 10)
    # Sequence b owns slot b + 1 of the store; slot zero stays reserved.
    for slot in range(1, args.batch_size + 1):
        assert torch.all(operands.state_store[slot] == slot + 9)
    assert not torch.any(operands.state_store[0] >= 10)


def test_guards_accept_the_contract_and_restore_state(monkeypatch) -> None:
    import torch

    runner = _runner()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    # sequence_length 2 < kernel_size - 1 covers the zero-padded final state.
    for length in (2, 5):
        args, operands = _cpu_operands(runner, sequence_length=length)
        store = operands.state_store.clone()
        runner._check_fresh(torch, _fake_causal_conv1d_fn(), operands, args)
        assert torch.equal(operands.state_store, store)
        runner._check_initial_state(torch, _fake_causal_conv1d_fn(), operands, args)


def test_initial_state_guard_catches_a_kernel_that_drops_the_history(monkeypatch) -> None:
    import torch

    runner = _runner()
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    args, operands = _cpu_operands(runner)
    with pytest.raises(AssertionError):
        runner._check_initial_state(
            torch, _fake_causal_conv1d_fn(ignore_initial_states=True), operands, args
        )


def test_timed_call_is_one_fresh_launch_per_sequence(monkeypatch) -> None:
    import torch

    runner = _runner()
    args, operands = _cpu_operands(runner)
    calls = []

    def recording_fn(x, weight, bias, **keyword):
        calls.append((x, keyword))
        return torch.empty_like(x), keyword["final_states_out"]

    def fake_timer(fn, *, kernel_name):
        assert kernel_name == "causal_conv1d_channellast_fwd_kernel"
        calls.clear()
        fn()
        return 0.5

    monkeypatch.setattr(runner, "_load_callable", lambda: recording_fn)
    monkeypatch.setattr(runner, "_build_operands", lambda *_a, **_k: operands)
    monkeypatch.setattr(runner, "_check_fresh", lambda *_a, **_k: None)
    monkeypatch.setattr(runner, "_check_initial_state", lambda *_a, **_k: None)
    monkeypatch.setattr(runner.Timer, "cupti", staticmethod(fake_timer))
    monkeypatch.setattr(runner.Energy, "perf", staticmethod(lambda fn, **_k: 0.25))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    metrics = runner.profile_gdn_causal_conv_prefill_dao_channellast(**_SPEC)

    assert metrics.time_ms == 0.5
    assert len(calls) == args.batch_size
    for index, (x, keyword) in enumerate(calls):
        assert torch.equal(x[0].t(), operands.semantic_x[index])
        assert keyword["initial_states"] is None
        assert keyword["final_states_out"].data_ptr() == operands.state[index + 1].data_ptr()
