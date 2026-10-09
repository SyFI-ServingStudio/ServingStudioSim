"""Behavioral checks for Neuron logical-invocation timing and dense math."""

from types import SimpleNamespace

import pytest
import torch

from profiling.profilers.neuron_timer import _execution_unions_ms, _validate_neff_contract
from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.neuron.dense import (
    _validate_mlp_shape,
    _validate_qkv_rows,
    _validate_shape,
    dense_mlp_reference,
)


def _event(phase, tracking_id, timestamp, *, exec_id=1, nc_idx=0, core=4):
    return {
        "event_type": "nc_exec_running",
        "phase": phase,
        "tracking_id": tracking_id,
        "nc_idx": nc_idx,
        "timestamp_ns": timestamp,
        "data": {"exec_id": exec_id, "device_core_idx": core},
    }


def test_lnc2_overlapping_physical_cores_are_one_invocation():
    trace = {"events": [
        _event("start", 1, 1_000_000), _event("start", 2, 1_100_000, core=5),
        _event("stop", 1, 1_400_000), _event("stop", 2, 1_600_000),
        _event("start", 3, 2_000_000, exec_id=2),
        _event("stop", 3, 2_300_000),
        _event("start", 4, 2_100_000, exec_id=2, core=5),
        _event("stop", 4, 2_200_000),
    ]}
    assert _execution_unions_ms(trace, 2) == pytest.approx([0.6, 0.3])


def test_device_gaps_in_one_invocation_do_not_count_as_kernel_work():
    trace = {"events": [
        _event("start", 1, 1_000_000), _event("stop", 1, 1_100_000),
        _event("start", 2, 2_000_000, core=5), _event("stop", 2, 2_200_000),
    ]}
    assert _execution_unions_ms(trace, 1) == pytest.approx([0.3])


def test_unmatched_or_wrong_count_device_trace_is_rejected():
    with pytest.raises(KernelLaunchFailed, match="unmatched starts"):
        _execution_unions_ms({"events": [_event("start", 1, 1)]}, 1)
    with pytest.raises(KernelLaunchFailed, match="expected 2"):
        _execution_unions_ms({"events": [_event("start", 1, 1), _event("stop", 1, 2)]}, 2)


def test_half_of_lnc2_trace_is_rejected_even_when_invocation_count_matches():
    with pytest.raises(KernelLaunchFailed, match="both physical cores"):
        _execution_unions_ms({"events": [_event("start", 1, 1), _event("stop", 1, 2)]}, 1)


def test_decoder_output_roles_survive_unordered_native_tensor_info():
    model = SimpleNamespace(
        output_tensors_info={"output2": None, "output0": None, "output1": None},
        alias_info={"output1": "input3", "output2": "input4"},
    )
    names = ("output0", "output1", "output2")
    assert _validate_neff_contract(model, names, model.alias_info) == names


def test_output_contract_rejects_missing_outputs_and_duplicate_roles():
    model = SimpleNamespace(output_tensors_info={"output0": None, "output1": None}, alias_info={})
    for names in (("output0",), ("output0", "output1", "output1")):
        with pytest.raises(KernelLaunchFailed, match="output ABI changed"):
            _validate_neff_contract(model, names, None)


def test_decoder_cache_alias_change_is_rejected_before_execution():
    model = SimpleNamespace(output_tensors_info={"output0": None}, alias_info={"output0": "input4"})
    with pytest.raises(KernelLaunchFailed, match="alias ABI changed"):
        _validate_neff_contract(model, ("output0",), {"output0": "input3"})


def test_default_output_order_uses_numeric_suffixes():
    model = SimpleNamespace(
        output_tensors_info={"output10": None, "output2": None, "output0": None}, alias_info={}
    )
    assert _validate_neff_contract(model, None, None) == ("output0", "output2", "output10")


def test_dense_mlp_rounds_product_before_down_projection():
    x = torch.tensor([[1.125, -0.8125]], dtype=torch.bfloat16)
    gate = torch.tensor([[0.7265625, 1.3125], [-0.96484375, 0.82421875]], dtype=torch.bfloat16)
    up = torch.tensor([[1.09375, 0.671875], [0.32421875, -0.84765625]], dtype=torch.bfloat16)
    down = torch.tensor([[0.53515625, 1.078125], [-0.91796875, 0.6640625]], dtype=torch.bfloat16)
    result = dense_mlp_reference(torch, x, gate, up, down)
    # Fixed scalar golden: rounding the product gives -0.048828125 here.
    # Leaving it FP32 gives -0.05029296875; separately rounding projections
    # changes it again. This protects the fusion's actual rounding boundary.
    expected = torch.tensor([[-0.048828125, 1.921875]], dtype=torch.bfloat16)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(0, 4096, 6144), (1, 128, 256), (1, 256, 257)])
def test_dense_runner_rejects_empty_or_unaligned_lnc2_shapes(shape):
    with pytest.raises(ValueError):
        _validate_shape(*shape, "bf16")


def test_dense_runner_refuses_unvalidated_fp16():
    with pytest.raises(ProfilerNotImplemented, match="BF16"):
        _validate_shape(1, 4096, 6144, "fp16")


def test_unverified_qkv_cte_path_is_checked_before_loading_sdk():
    _validate_qkv_rows(96)
    with pytest.raises(ProfilerNotImplemented, match="TKG"):
        _validate_qkv_rows(97)


@pytest.mark.parametrize("shape", [(128, 4096, 14336), (16, 4096, 14336), (1, 4096, 4096)])
def test_mlp_refuses_unverified_or_vendor_memory_failure_shapes(shape):
    with pytest.raises(ProfilerNotImplemented, match="validated only"):
        _validate_mlp_shape(*shape, "bf16")
