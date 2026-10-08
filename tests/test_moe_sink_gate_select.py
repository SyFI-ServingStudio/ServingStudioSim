import pytest
import torch

from profiling.runners.exceptions import KernelLaunchFailed, ProfilerNotImplemented
from profiling.runners.moe import moe_sink_gate_select_vllm_triton as runner
from profiling.runners.moe.moe_sink_gate_select_reference import (
    GateShape,
    compare,
    reference_select,
)

_SHAPE = GateShape(num_routed_experts=8, num_shared_experts=2, top_k=2, logits_row_stride=16)
_ARGS = dict(
    num_tokens=32,
    num_routed_experts=256,
    num_shared_experts=2,
    top_k=6,
    logits_row_stride=264,
    has_bias=True,
    has_global_scale=True,
    logits_dtype="fp32",
)


def _logits(rows):
    out = torch.full((len(rows), _SHAPE.logits_row_stride), 1.0e4)
    out[:, : _SHAPE.num_gate_experts] = torch.tensor(rows, dtype=torch.float32)
    return out


def test_reference_selects_by_sigmoid_plus_bias_and_renorms_with_the_sinks():
    # Defect caught: a reference that ranks by raw logit + bias (bias flips the
    # order here), weights the bias into the renorm, drops the sink logits from
    # the denominator, or reads the poisoned padding columns.
    logits = _logits([[2.0, 1.0, 0.0, -1.0, -2.0, -3.0, 0.5, 0.2, 0.3, -0.4]])
    bias = torch.tensor([0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0])
    gs = torch.tensor([1.25])
    ids, weights = reference_select(torch, _SHAPE, logits, bias, 8.0, gs)
    # sigmoid(0) + 0.3 = 0.8 > sigmoid(1) = 0.731 > sigmoid(0.5) = 0.622
    assert ids.tolist() == [[0, 2, 8, 9]]
    raw = torch.tensor([2.0, 0.0, 0.3, -0.4], dtype=torch.float64)
    expected = torch.sigmoid(raw) / torch.sigmoid(raw).sum() * 8.0 * 1.25
    torch.testing.assert_close(weights[0], expected)


def test_reference_breaks_exact_ties_to_the_lowest_id():
    # Defect caught: torch.topk's unspecified tie order standing in for the
    # kernel's lowest-index argmax rule.
    logits = _logits([[0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    ids, _ = reference_select(torch, _SHAPE, logits, None, 1.0, None)
    assert ids[0, :2].tolist() == [1, 3]


def test_compare_rejects_a_wrong_exact_tie_break_but_excuses_a_near_tie():
    # Defect caught: an oracle that excuses every close score (hiding a kernel
    # that breaks exact ties the wrong way) or none (failing on ulp-level
    # sigmoid differences between Triton and Torch).
    tied = _logits([[0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    ref_ids, ref_w = reference_select(torch, _SHAPE, tied, None, 1.0, None)
    assert compare(torch, _SHAPE, tied, None, 1.0, None, ref_ids, ref_w.float())["ok"]
    swapped = ref_ids.clone()
    swapped[0, 1] = 4
    w = runner_weights(tied, swapped)
    report = compare(torch, _SHAPE, tied, None, 1.0, None, swapped, w)
    assert not report["ok"] and report["unexplained_mismatch_tokens"] == 1

    near = _logits([[0.0, 1.0, 0.0, 1.0, 1.0 + 2.5e-6, 0.0, 0.0, 0.0, 0.0, 0.0]])
    # The reference takes expert 4 (its score is ~5e-7 higher); a kernel that
    # takes 1 and 3 instead broke a near-tie.
    near_ids = torch.tensor([[1, 3, 8, 9]])
    report = compare(torch, _SHAPE, near, None, 1.0, None, near_ids, runner_weights(near, near_ids))
    assert report["ok"] and report["near_tie_tokens"] == 1

    far_ids = torch.tensor([[1, 0, 8, 9]])
    report = compare(torch, _SHAPE, near, None, 1.0, None, far_ids, runner_weights(near, far_ids))
    assert not report["ok"]


def runner_weights(logits, ids):
    raw = torch.sigmoid(logits.double().gather(1, ids))
    return (raw / raw.sum(dim=1, keepdim=True)).float()


def test_inputs_poison_the_padding_and_keep_logits_finite():
    # Defect caught: padding columns a kernel could read unnoticed (they would
    # never win the top-k if they held ordinary values).
    x = runner.build_inputs(
        torch,
        torch.device("cpu"),
        GateShape(256, 2, 6, 264),
        4,
        has_bias=True,
        has_global_scale=True,
    )
    assert torch.all(x.logits[:, 258:] == 1.0e4)
    assert torch.all(x.logits[:, :258].abs() < 100)
    assert x.bias.shape == (256,) and x.global_scale.tolist() == [1.25]


def test_structure_check_catches_a_misplaced_sink():
    # Defect caught: a call whose sink columns or weight sums are wrong timing
    # silently as if it had worked.
    x = runner.build_inputs(
        torch, torch.device("cpu"), _SHAPE, 3, has_bias=False, has_global_scale=True
    )
    ids, w = reference_select(torch, _SHAPE, x.logits, None, x.route_scale, x.global_scale)
    runner.check_structure(torch, x, w.float(), ids.int())
    bad = ids.clone()
    bad[1, 2] = 9
    with pytest.raises(KernelLaunchFailed):
        runner.check_structure(torch, x, w.float(), bad.int())


def test_args_the_production_call_cannot_take_are_rejected():
    # Defect caught: profiling a shape the kernel cannot compile (K + S not a
    # power of two) or a dtype the wrapper asserts away.
    assert runner._validate_args(**_ARGS).active == 8
    with pytest.raises(ProfilerNotImplemented, match="power of two"):
        runner._validate_args(**dict(_ARGS, top_k=5))
    with pytest.raises(ProfilerNotImplemented, match="fp32"):
        runner._validate_args(**dict(_ARGS, logits_dtype="bf16"))
    with pytest.raises(ValueError, match="row_stride"):
        runner._validate_args(**dict(_ARGS, logits_row_stride=257))


def test_logical_bytes_match_inkling_small():
    # Defect caught: GB/s from the wrong byte count. 258 logits read plus 8
    # ids and 8 weights written, 4 bytes each: 1096 B per token.
    assert runner.logical_bytes(GateShape(256, 2, 6, 264), 10) == 10_960
