"""Logits runner contracts without CUDA initialization or profile DB writes."""

import inspect
from dataclasses import fields

import pytest

from profiling.db.args import DType
from profiling.db.batch import args_to_spec, coerce_args
from profiling.db.registry import find_kernel_profiler_spec
from profiling.runners.comm import vocab_parallel_all_gather as gather
from profiling.runners.exceptions import ProfilerNotImplemented
from profiling.runners.logits import torch as runner
from profiling.runners.logits.reference import logits_argmax_reference, logits_copy_reference

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("input_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [1, 3])
def test_production_copy_matches_oracle_and_never_aliases(rows, input_dtype, output_dtype):
    logits = runner.make_logits(torch, rows, 11, 15, input_dtype, device="cpu")
    output = runner.copy_logits(torch, logits, output_dtype)
    expected = logits_copy_reference(torch, logits, output_dtype)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    assert output.is_contiguous()
    assert output.data_ptr() != logits.data_ptr()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_production_argmax_matches_first_tie_oracle_with_row_padding(dtype):
    storage = torch.full((3, 8), float("nan"), dtype=dtype)
    logits = storage[:, :4]
    logits.copy_(torch.tensor([[1, 2, 2, 0], [float("inf"), 2, float("inf"), 0], [-3] * 4]))
    torch.testing.assert_close(runner.argmax_logits(logits), logits_argmax_reference(torch, logits))


@pytest.mark.parametrize("key", ["num_rows", "vocab_size", "row_stride"])
@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_compute_sizes_fail_before_gpu_access(key, bad):
    values = dict(num_rows=3, vocab_size=7, row_stride=9, dtype="bf16")
    values[key] = bad
    with pytest.raises(ValueError, match=key):
        runner.profile_logits_argmax(**values)
    values.pop("dtype")
    with pytest.raises(ValueError, match=key):
        runner.profile_logits_copy(**values, input_dtype="bf16", output_dtype="fp32")


def test_invalid_layout_and_dtype_fail_before_gpu_access():
    with pytest.raises(ValueError, match="row_stride"):
        runner.profile_logits_argmax(3, 7, 6, "bf16")
    with pytest.raises(ProfilerNotImplemented, match="bf16 and fp32"):
        runner.profile_logits_copy(3, 7, 7, "fp16", "fp32")
    with pytest.raises(ProfilerNotImplemented, match="bf16 and fp32"):
        runner.profile_logits_copy(3, 7, 7, "bf16", "fp16")


@pytest.mark.parametrize(
    "field,bad",
    [
        ("num_gpus", True),
        ("num_gpus", 1),
        ("num_rows", 0),
        ("vocab_size_per_rank", 1.5),
        ("fabric", "pcie"),
        ("dtype", "fp16"),
    ],
)
def test_invalid_collective_specs_do_not_launch_ranks(monkeypatch, field, bad):
    monkeypatch.setattr(
        gather, "TorchMpLauncher", lambda *a, **k: pytest.fail("unexpected rank spawn")
    )
    spec = dict(num_gpus=4, num_rows=3, vocab_size_per_rank=11, dtype="bf16", fabric="nvlink")
    spec[field] = bad
    result = gather.profile_vocab_parallel_all_gather_batch([spec])
    assert len(result) == 1
    assert result[0].error


def test_collective_batch_rejects_mixed_world_sizes_before_spawn(monkeypatch):
    monkeypatch.setattr(
        gather, "TorchMpLauncher", lambda *a, **k: pytest.fail("unexpected rank spawn")
    )
    spec = dict(num_gpus=4, num_rows=3, vocab_size_per_rank=11, dtype="bf16", fabric="nvlink")
    results = gather.profile_vocab_parallel_all_gather_batch([spec, {**spec, "num_gpus": 8}])
    assert len(results) == 2
    assert all("same num_gpus" in result.error for result in results)


def test_copy_schema_preserves_distinct_input_and_output_dtype_axes():
    spec = find_kernel_profiler_spec("logits_copy", "torch")
    raw = dict(num_rows=3, vocab_size=17, row_stride=19, input_dtype="bf16", output_dtype="fp32")
    args = coerce_args(spec.args_schema, raw)
    assert args.input_dtype is DType.BF16
    assert args.output_dtype is DType.FP32
    assert args_to_spec(args) == raw
    assert list(inspect.signature(runner.profile_logits_copy).parameters) == [
        field.name for field in fields(spec.args_schema)
    ]


def test_compute_profiles_full_callable_and_excludes_oracle(monkeypatch):
    original_make = runner.make_logits
    monkeypatch.setattr(runner, "require_b200", lambda torch: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        runner, "make_logits", lambda *args, **kwargs: original_make(*args, device="cpu")
    )
    invocations = []

    def time_callable(launch, **kwargs):
        assert kwargs == {"warmup": 5, "kernel_name": None}
        output = launch()
        invocations.append(output)
        return 2.0

    def energy_callable(launch, **kwargs):
        assert kwargs["per_iter_time_ms"] == 2.0
        launch()
        return 0.125

    monkeypatch.setattr(runner.Timer, "cupti", time_callable)
    monkeypatch.setattr(runner.Energy, "perf", energy_callable)
    copy_metrics = runner.profile_logits_copy(3, 7, 9, "bf16", "fp32")
    argmax_metrics = runner.profile_logits_argmax(3, 7, 9, "bf16")
    assert copy_metrics.time_ms == argmax_metrics.time_ms == 2.0
    assert copy_metrics.energy_j == argmax_metrics.energy_j == 0.125
    assert [output.dtype for output in invocations] == [torch.float32, torch.int64]
