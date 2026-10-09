"""Stock fused MLP semantics, fixed numerical gates and native timing coverage."""

import math

import pytest
import torch

from profiling.profilers.neuron_lite_timer import execution_unions, verify_no_event_drops
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.neuron.vllm_mlp import check_output, mlp_reference, validate_shape


def test_stock_mlp_oracle_keeps_full_fp32_math():
    x, gate, up, down = [
        torch.tensor([[value]], dtype=torch.bfloat16) for value in (2, 0.5, -1, 0.25)
    ]
    actual = mlp_reference(torch, x, gate, up, down)
    assert actual.dtype == torch.float32
    assert actual.item() == pytest.approx(-0.5 / (1 + math.exp(-1)), abs=1e-7)


def test_stock_mlp_requires_both_prespecified_error_limits():
    expected = torch.ones(10_000)
    peak_failure = expected.to(torch.bfloat16)
    peak_failure[0] = 1.125
    peak = check_output(torch, peak_failure, expected)
    assert peak["relative_l2"] < 0.02
    assert peak["normalized_peak"] > 0.05
    assert not peak["passed"]
    norm = check_output(torch, torch.full_like(peak_failure, 1.03125), expected)
    assert norm["normalized_peak"] < 0.05
    assert norm["relative_l2"] > 0.02
    assert not norm["passed"]
    assert check_output(torch, expected.to(torch.bfloat16), expected)["passed"]


def test_stock_mlp_nonfinite_and_wrong_abi_cannot_produce_timing():
    expected = torch.ones(2)
    assert not check_output(torch, torch.tensor([1, float("nan")], dtype=torch.bfloat16), expected)[
        "passed"
    ]
    with pytest.raises(ValueError, match="shape or dtype"):
        check_output(torch, expected, expected)
    with pytest.raises(ValueError, match="shape or dtype"):
        check_output(torch, expected[:1].to(torch.bfloat16), expected)


@pytest.mark.parametrize("m", [1, 16, 512])
def test_verified_stock_mlp_shapes(m):
    validate_shape(m, 4096, 3584, "bfloat16")


@pytest.mark.parametrize(
    "spec",
    [
        (2, 4096, 3584, "bfloat16"),
        (128, 4096, 3584, "bfloat16"),
        (1, 4096, 14336, "bfloat16"),
        (1, 2048, 3584, "bfloat16"),
        (1, 4096, 3584, "float16"),
    ],
)
def test_unverified_stock_mlp_shapes_reject_before_loading_neuron(spec):
    with pytest.raises(ProfilerNotImplemented):
        validate_shape(*spec)


def _event(start, duration, *, core=4, execution=1):
    return {
        "name": "nc_exec_running",
        "timestamp": start,
        "duration": duration,
        "timestamp_unit": "ns",
        "exec_id": execution,
        "device_core_idx": core,
        "lnc_idx": 0,
        "nc_idx": 0,
        "process_id": "2",
        "model_name": "",
    }


def _brackets():
    return [
        {"iteration": 0, "start_epoch_ns": 0, "stop_epoch_ns": 1_000_000},
        {"iteration": 1, "start_epoch_ns": 2_000_000, "stop_epoch_ns": 3_000_000},
    ]


def _trace():
    return {
        "trace_event": [
            _event(100_000, 200_000),
            _event(200_000, 200_000, core=5),
            _event(2_100_000, 100_000, execution=2),
            _event(2_400_000, 100_000, core=5, execution=2),
        ]
    }


def test_native_lite_timing_unions_concurrent_cores_and_excludes_gaps_and_host_time():
    records = execution_unions(_trace(), _brackets())
    assert [row["time_ms"] for row in records] == pytest.approx([0.3, 0.2])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "outside", "core", "exec", "unit"])
def test_native_lite_timing_rejects_partial_or_ambiguous_executions(mutation):
    trace = _trace()
    if mutation == "missing":
        trace["trace_event"].pop()
    elif mutation == "duplicate":
        trace["trace_event"].append(trace["trace_event"][0].copy())
    elif mutation == "outside":
        trace["trace_event"][0]["timestamp"] = 1_500_000
    elif mutation == "core":
        trace["trace_event"][-1]["device_core_idx"] = 7
    elif mutation == "exec":
        trace["trace_event"][-1]["exec_id"] = 1
    else:
        trace["trace_event"][-1]["timestamp_unit"] = "us"
    with pytest.raises(ValueError):
        execution_unions(trace, _brackets())


def test_native_lite_trace_drop_warning_invalidates_complete_counts():
    with pytest.raises(ValueError, match="dropped events"):
        verify_no_event_drops("System profile events were dropped due to full ring buffer")
    trace = _trace()
    trace["warnings"] = ["incomplete trace"]
    with pytest.raises(ValueError, match="dropped events"):
        execution_unions(trace, _brackets())


@pytest.mark.parametrize(
    "warning",
    [
        "Notification messages were dropped",
        "System profile lost 5 notifications",
        "ring buffer overflow",
        "System trace truncated",
        "lost events in profiler",
    ],
)
def test_native_lite_notification_loss_also_invalidates_timing(warning):
    with pytest.raises(ValueError, match="dropped events"):
        verify_no_event_drops(warning)
