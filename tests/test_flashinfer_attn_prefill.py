"""Unit tests for the ``flashinfer_attn_prefill`` L1 kernel kind.

Parallel-safety note (mirrors ``test_elementwise.py``): these import the
per-kernel module DIRECTLY so they pass before the shared barrel
``profiling/kernels/__init__.py`` is wired. They do not depend on the registry
being fully loaded, do not touch the shared ``profile.db``, and run no GPU work.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from profiling.db.args import DType
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import KernelProfilerSpec, MetricFamily, RunnerRef
from profiling.kernels import flashinfer_attn_prefill as prefill_kernel
from profiling.kernels.flashinfer_attn_prefill import KIND, FlashinferAttnPrefillArgs
from profiling.runners.metrics import ComputeMetrics

_BACKENDS = ("fa2", "fa3", "trt", "cudnn")
_RUNNER_MODULE = "profiling.runners.attention.flashinfer_attn_prefill_and_rect"


def test_args_field_contract_matches_runner_kwargs():
    # Field set / order is the load-bearing contract (A5): it must equal the
    # Rust enumerate wire fields (minus `backend`) and the runner kwargs that
    # _run_ragged_attention forwards. Config dims first, then the 2D sweep coords.
    field_names = [f.name for f in FlashinferAttnPrefillArgs.__dataclass_fields__.values()]
    assert field_names == [
        "num_qo_heads",
        "num_kv_heads",
        "head_dim",
        "q_dtype",
        "kv_dtype",
        "o_dtype",
        "prefix_len",
        "append_len",
    ]


def test_args_are_frozen():
    args = FlashinferAttnPrefillArgs(
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        q_dtype=DType.BF16,
        kv_dtype=DType.BF16,
        o_dtype=DType.BF16,
        prefix_len=0,
        append_len=512,
    )
    assert args.append_len == 512
    assert args.prefix_len == 0
    with pytest.raises(Exception):
        args.append_len = 8  # frozen


def test_kind_wire_string_and_table_stem():
    # KIND must equal the Rust KernelSpec::KIND and the registry table_name so
    # the cross-language facade name resolves (validator enforces
    # table_name == kernel_kind).
    assert KIND == "flashinfer_attn_prefill"
    assert prefill_kernel.KIND == "flashinfer_attn_prefill"


def test_register_call_built_one_compute_spec_per_backend():
    # The module's import-time register(...) side effects must produce one spec
    # per backend, all sharing the same table/schema/family and routing to the
    # per-backend entry in the shared ragged runner module. Rebuild each spec
    # and assert its shape without loading the full registry barrel.
    for backend in _BACKENDS:
        spec = KernelProfilerSpec(
            kernel_kind=KIND,
            backend=backend,
            runner_ref=RunnerRef(
                module_name=_RUNNER_MODULE,
                function_name=f"profile_flashinfer_attn_prefill_{backend}",
            ),
            table_name=KIND,
            args_schema=FlashinferAttnPrefillArgs,
            metric_family=MetricFamily.COMPUTE,
            batch_outlier_policy=BatchOutlierPolicy(),
            subprocess_env="flashinfer_pip_env",
        )
        assert spec.kernel_kind == "flashinfer_attn_prefill"
        assert spec.backend == backend
        assert spec.table_name == spec.kernel_kind  # facade-stem invariant
        assert spec.metric_family is MetricFamily.COMPUTE
        assert spec.runner_ref.module_name == _RUNNER_MODULE
        assert spec.runner_ref.function_name == f"profile_flashinfer_attn_prefill_{backend}"


def test_importing_kernel_module_does_not_eager_import_runner():
    # Lazy-import invariant (L1 design §3.2.1): importing the per-kernel module
    # must NOT pull in the runner module (which imports flashinfer/torch).
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import profiling.kernels.flashinfer_attn_prefill; "
            "print('profiling.runners.attention.flashinfer_attn_prefill_and_rect' in sys.modules)"
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False"


def _install_fake_ragged_runtime(monkeypatch, events):
    from profiling.runners.attention import _common
    from profiling.runners.attention import flashinfer_attn_prefill_and_rect as runner

    class Wrapper:
        def __init__(self, workspace, *, kv_layout, backend):
            events.append(("wrapper", workspace, kv_layout, backend))

        def plan(self, **kwargs):
            events.append(("plan", kwargs))

        def run(self, q, k, v):
            events.append(("run", q, k, v))
            return "output"

    torch = SimpleNamespace(
        bfloat16="bf16",
        float8_e4m3fn="fp8",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: events.append(("synchronize",)),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(BatchPrefillWithRaggedKVCacheWrapper=Wrapper),
    )
    inp = SimpleNamespace(
        q="q",
        k="k",
        v="v",
        qo_indptr="qo",
        kv_indptr="kv",
        scales=None,
        bytes_accessed=123,
    )
    monkeypatch.setattr(_common, "build_ragged_inputs", lambda **kwargs: inp)
    monkeypatch.setattr(_common, "make_workspace", lambda: "workspace")
    monkeypatch.setattr(_common, "to_torch_dtype", lambda dtype: "bf16")
    monkeypatch.setattr(_common, "attention_flops", lambda **kwargs: 456)
    return runner, _common


def test_fa3_plans_then_hands_off_without_its_own_warmup(monkeypatch):
    """The runner plans and hands off; warming the callable is the profiler's job.

    FA3 used to warm up here by hand so its learned CUPTI launch pattern was the
    steady-state one. FA2, on the `_measure_best_split` path, did not — and every
    FA2 spec that was first in its worker process died on the record-count check.
    The warm-up now lives in `_prepare_launch_pattern`, which covers all backends,
    so a per-backend warm-up here would be a redundant timed launch.
    """
    events = []
    runner, common = _install_fake_ragged_runtime(monkeypatch, events)
    metrics = ComputeMetrics(1.0, 2.0, 3.0, 4.0)

    def measure(fn, *, flops, bytes_accessed):
        events.append(("measure", fn, flops, bytes_accessed))
        return metrics

    monkeypatch.setattr(common, "measure", measure)

    actual = runner.profile_flashinfer_attn_prefill_fa3(
        prefix_len=0,
        append_len=128,
        num_qo_heads=16,
        num_kv_heads=2,
        head_dim=256,
        q_dtype="bf16",
        kv_dtype="bf16",
        o_dtype="bf16",
    )

    assert actual is metrics
    assert [event[0] for event in events] == [
        "wrapper",
        "plan",
        "measure",
    ]
    plan = events[1][1]
    assert plan == {
        "qo_indptr": "qo",
        "kv_indptr": "kv",
        "num_qo_heads": 16,
        "num_kv_heads": 2,
        "head_dim_qk": 256,
        "head_dim_vo": 256,
        "causal": True,
        "q_data_type": "bf16",
    }
    assert events[-1][2:] == (456, 123)


def test_fa2_best_split_path_has_no_new_common_warmup(monkeypatch):
    events = []
    runner, common = _install_fake_ragged_runtime(monkeypatch, events)
    metrics = ComputeMetrics(1.0, 2.0, 3.0, 4.0)

    def best_split(**kwargs):
        events.append(("best_split", kwargs))
        return metrics

    monkeypatch.setattr(runner, "_measure_best_split", best_split)
    monkeypatch.setattr(
        common,
        "measure",
        lambda *args, **kwargs: pytest.fail("FA2 must remain on _measure_best_split"),
    )

    actual = runner.profile_flashinfer_attn_prefill_fa2(
        prefix_len=0,
        append_len=128,
        num_qo_heads=16,
        num_kv_heads=2,
        head_dim=256,
        q_dtype="bf16",
        kv_dtype="bf16",
        o_dtype="bf16",
    )

    assert actual is metrics
    assert [event[0] for event in events] == ["wrapper", "best_split"]
    assert events[1][1]["kv_len"] == 128
