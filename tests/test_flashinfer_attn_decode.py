"""Unit tests for the ``flashinfer_attn_decode`` L1 kernel kind.

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
from profiling.db.registry import (
    KernelProfilerSpec,
    MetricFamily,
    RunnerRef,
    find_kernel_profiler_spec,
)
from profiling.kernels import flashinfer_attn_decode as decode_kernel
from profiling.kernels.flashinfer_attn_decode import KIND, FlashinferAttnDecodeArgs
from profiling.runners.metrics import ComputeMetrics

_BACKENDS = ("fa2", "fa2_cudagraph", "fa3", "trt", "cudnn")
_RUNNER_MODULE = "profiling.runners.attention.flashinfer_decode"


def test_args_field_contract_matches_runner_kwargs():
    # Field set / order is the load-bearing contract (A5): Config dims first,
    # then the 2D decode sweep coords (batch_size, total_tokens). batch is a real
    # axis; total_tokens is total kv (runner derives avg_len = total // batch).
    field_names = [f.name for f in FlashinferAttnDecodeArgs.__dataclass_fields__.values()]
    assert field_names == [
        "num_qo_heads",
        "num_kv_heads",
        "head_dim",
        "q_dtype",
        "kv_dtype",
        "o_dtype",
        "batch_size",
        "total_tokens",
    ]


def test_args_are_frozen():
    args = FlashinferAttnDecodeArgs(
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        q_dtype=DType.BF16,
        kv_dtype=DType.BF16,
        o_dtype=DType.BF16,
        batch_size=64,
        total_tokens=262144,
    )
    assert args.batch_size == 64
    assert args.total_tokens == 262144
    with pytest.raises(Exception):
        args.batch_size = 8  # frozen


def test_kind_wire_string_and_table_stem():
    assert KIND == "flashinfer_attn_decode"
    assert decode_kernel.KIND == "flashinfer_attn_decode"


def test_register_call_built_one_compute_spec_per_backend():
    for backend in _BACKENDS:
        spec = KernelProfilerSpec(
            kernel_kind=KIND,
            backend=backend,
            runner_ref=RunnerRef(
                module_name=_RUNNER_MODULE,
                function_name=f"profile_flashinfer_attn_decode_{backend}",
            ),
            table_name=KIND,
            args_schema=FlashinferAttnDecodeArgs,
            metric_family=MetricFamily.COMPUTE,
            batch_outlier_policy=BatchOutlierPolicy(),
            subprocess_env="flashinfer_pip_env",
        )
        assert spec.kernel_kind == "flashinfer_attn_decode"
        assert spec.backend == backend
        assert spec.table_name == spec.kernel_kind  # facade-stem invariant
        assert spec.metric_family is MetricFamily.COMPUTE
        assert spec.runner_ref.module_name == _RUNNER_MODULE
        assert spec.runner_ref.function_name == f"profile_flashinfer_attn_decode_{backend}"


def test_fa2_cudagraph_backend_is_registered_lazily_on_the_existing_table():
    spec = find_kernel_profiler_spec(KIND, "fa2_cudagraph")
    assert spec.args_schema is FlashinferAttnDecodeArgs
    assert spec.table_name == KIND
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.runner_ref == RunnerRef(
        module_name=_RUNNER_MODULE,
        function_name="profile_flashinfer_attn_decode_fa2_cudagraph",
    )


def test_importing_kernel_module_does_not_eager_import_runner():
    # Lazy-import invariant (L1 design §3.2.1): importing the per-kernel module
    # must NOT pull in the runner module (which imports flashinfer/torch).
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import profiling.kernels.flashinfer_attn_decode; "
            "print('profiling.runners.attention.flashinfer_decode' in sys.modules)"
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False"


def _install_fake_decode_runtime(monkeypatch, events):
    from profiling.runners.attention import _common
    from profiling.runners.attention import flashinfer_decode as runner

    class Wrapper:
        def __init__(self, workspace, **kwargs):
            events.append(("wrapper", workspace, kwargs))

        def plan(self, **kwargs):
            events.append(("plan", kwargs))

        def run(self, q, paged_kv_cache, **kwargs):
            events.append(("run", q, paged_kv_cache, kwargs))
            return "output"

    torch = SimpleNamespace(
        float8_e4m3fn="fp8",
        empty_like=lambda value: f"empty:{value}",
        cuda=SimpleNamespace(
            is_available=lambda: True,
            synchronize=lambda: events.append(("synchronize",)),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "flashinfer",
        SimpleNamespace(BatchDecodeWithPagedKVCacheWrapper=Wrapper),
    )
    inp = SimpleNamespace(
        q="q",
        k_cache="k",
        v_cache="v",
        kv_indptr="indptr",
        kv_indices="indices",
        kv_last_page_len="last_page_len",
        scales=None,
        bytes_accessed=123,
    )

    def build_inputs(**kwargs):
        events.append(("build_inputs", kwargs))
        return inp

    monkeypatch.setattr(_common, "build_paged_decode_inputs", build_inputs)
    monkeypatch.setattr(_common, "make_workspace", lambda: "workspace")
    monkeypatch.setattr(_common, "to_torch_dtype", lambda dtype: "bf16")
    monkeypatch.setattr(_common, "flashinfer_backend_name", lambda backend: backend)
    monkeypatch.setattr(_common, "attention_flops", lambda **kwargs: 456)
    return runner, _common


def _decode_args():
    return {
        "batch_size": 1,
        "total_tokens": 32,
        "num_qo_heads": 16,
        "num_kv_heads": 2,
        "head_dim": 256,
        "q_dtype": "bf16",
        "kv_dtype": "bf16",
        "o_dtype": "bf16",
    }


def test_fa3_plans_warms_once_then_synchronizes_before_measurement(monkeypatch):
    events = []
    runner, common = _install_fake_decode_runtime(monkeypatch, events)
    metrics = ComputeMetrics(1.0, 2.0, 3.0, 4.0)

    def measure(fn, *, flops, bytes_accessed):
        # Deliberately do not invoke fn: the sole run before this seam is the
        # explicit untimed warmup, outside CUPTI and energy measurement.
        events.append(("measure", fn, flops, bytes_accessed))
        return metrics

    monkeypatch.setattr(common, "measure", measure)

    actual = runner.profile_flashinfer_attn_decode_fa3(**_decode_args())

    assert actual is metrics
    assert [event[0] for event in events] == [
        "build_inputs",
        "wrapper",
        "plan",
        "run",
        "synchronize",
        "measure",
    ]
    assert events[0][1] == {
        "batch_size": 1,
        "seq_len": 32,
        "num_qo_heads": 16,
        "num_kv_heads": 2,
        "head_dim": 256,
        "q_dtype": "bf16",
        "kv_dtype": "bf16",
        "o_dtype": "bf16",
        "page_size": 16,
    }
    assert events[2][1] == {
        "indptr": "indptr",
        "indices": "indices",
        "last_page_len": "last_page_len",
        "num_qo_heads": 16,
        "num_kv_heads": 2,
        "head_dim": 256,
        "page_size": 16,
        "q_data_type": "bf16",
    }
    assert events[-1][2:] == (456, 123)


def test_eager_fa2_has_no_new_warmup(monkeypatch):
    events = []
    runner, common = _install_fake_decode_runtime(monkeypatch, events)
    metrics = ComputeMetrics(1.0, 2.0, 3.0, 4.0)
    monkeypatch.setattr(
        common,
        "measure",
        lambda fn, *, flops, bytes_accessed: (
            events.append(("measure", fn, flops, bytes_accessed)) or metrics
        ),
    )

    assert runner.profile_flashinfer_attn_decode_fa2(**_decode_args()) is metrics
    assert [event[0] for event in events] == [
        "build_inputs",
        "wrapper",
        "plan",
        "measure",
    ]


def test_fa2_cudagraph_keeps_exactly_its_existing_warmup(monkeypatch):
    events = []
    runner, common = _install_fake_decode_runtime(monkeypatch, events)
    metrics = ComputeMetrics(1.0, 2.0, 3.0, 4.0)
    monkeypatch.setattr(
        common,
        "measure",
        lambda fn, *, flops, bytes_accessed: (
            events.append(("measure", fn, flops, bytes_accessed)) or metrics
        ),
    )

    assert runner.profile_flashinfer_attn_decode_fa2_cudagraph(**_decode_args()) is metrics
    assert [event[0] for event in events] == [
        "build_inputs",
        "wrapper",
        "plan",
        "run",
        "synchronize",
        "measure",
    ]
