"""Unit tests for the ``flashinfer_attn_rect`` L1 kernel kind.

Parallel-safety note (mirrors ``test_elementwise.py``): these import the
per-kernel module DIRECTLY so they pass before the shared barrel
``profiling/kernels/__init__.py`` is wired. They do not depend on the registry
being fully loaded, do not touch the shared ``profile.db``, and run no GPU work.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from profiling.db.args import DType
from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import KernelProfilerSpec, MetricFamily, RunnerRef
from profiling.kernels import flashinfer_attn_rect as rect_kernel
from profiling.kernels.flashinfer_attn_rect import KIND, FlashinferAttnRectArgs

_BACKENDS = ("fa2", "fa3", "cudnn")
_RUNNER_MODULE = "profiling.runners.attention.flashinfer_attn_prefill_and_rect"


def test_args_field_contract_matches_runner_kwargs():
    # Field set / order is the load-bearing contract (A5): rect uses (q_len,
    # kv_len) directly (NOT prefill's prefix/append), so it can express
    # q_len > kv_len. Config dims first, then the sweep coords; `num_requests`
    # (the batch, one K/V per request) is optional so pre-batch rows keep keys.
    field_names = [f.name for f in FlashinferAttnRectArgs.__dataclass_fields__.values()]
    assert field_names == [
        "num_qo_heads",
        "num_kv_heads",
        "head_dim",
        "q_dtype",
        "kv_dtype",
        "o_dtype",
        "q_len",
        "kv_len",
        "num_requests",
    ]


def test_args_are_frozen():
    args = FlashinferAttnRectArgs(
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        q_dtype=DType.BF16,
        kv_dtype=DType.BF16,
        o_dtype=DType.BF16,
        q_len=1024,
        kv_len=256,
    )
    assert args.q_len == 1024
    assert args.kv_len == 256  # q_len > kv_len is representable
    with pytest.raises(Exception):
        args.q_len = 8  # frozen


def test_kind_wire_string_and_table_stem():
    assert KIND == "flashinfer_attn_rect"
    assert rect_kernel.KIND == "flashinfer_attn_rect"


def test_register_call_built_one_compute_spec_per_backend():
    for backend in _BACKENDS:
        spec = KernelProfilerSpec(
            kernel_kind=KIND,
            backend=backend,
            runner_ref=RunnerRef(
                module_name=_RUNNER_MODULE,
                function_name=f"profile_flashinfer_attn_rect_{backend}",
            ),
            table_name=KIND,
            args_schema=FlashinferAttnRectArgs,
            metric_family=MetricFamily.COMPUTE,
            batch_outlier_policy=BatchOutlierPolicy(),
            subprocess_env="flashinfer_pip_env",
        )
        assert spec.kernel_kind == "flashinfer_attn_rect"
        assert spec.backend == backend
        assert spec.table_name == spec.kernel_kind  # facade-stem invariant
        assert spec.metric_family is MetricFamily.COMPUTE
        assert spec.runner_ref.module_name == _RUNNER_MODULE
        assert spec.runner_ref.function_name == f"profile_flashinfer_attn_rect_{backend}"


def test_importing_kernel_module_does_not_eager_import_runner():
    # Lazy-import invariant (L1 design §3.2.1): importing the per-kernel module
    # must NOT pull in the runner module (which imports flashinfer/torch).
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import profiling.kernels.flashinfer_attn_rect; "
            "print('profiling.runners.attention.flashinfer_attn_prefill_and_rect' in sys.modules)"
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False"


def _rect_spec(**extra) -> dict:
    return {
        "num_qo_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "q_dtype": "bf16",
        "kv_dtype": "bf16",
        "o_dtype": "bf16",
        "q_len": 64,
        "kv_len": 2048,
        **extra,
    }


def test_one_request_keeps_the_key_rows_had_before_the_batch_axis():
    # Defect: a spec that spells out num_requests=1 (the Rust grid does) hashing
    # differently from the rows measured before the axis existed, which would
    # turn every one of them into a miss.
    from pathlib import Path

    from profiling.db import storage
    from profiling.db.batch import args_to_spec, coerce_args
    from profiling.db.registry import find_kernel_profiler_spec
    from profiling.db.table import Table

    table = Table(find_kernel_profiler_spec(KIND, "fa2"), Path("unused.db"))
    types = {name: "TEXT" if name.endswith("dtype") else "INTEGER" for name in _rect_spec()}
    before = storage.args_hash(_rect_spec(), types)
    for spec in (_rect_spec(), _rect_spec(num_requests=1)):
        args = coerce_args(FlashinferAttnRectArgs, spec)
        assert args.num_requests is None
        assert "num_requests" not in args_to_spec(args)
        assert table.args_hash(table.db_key(args)) == before
    batched = coerce_args(FlashinferAttnRectArgs, _rect_spec(num_requests=8))
    assert args_to_spec(batched)["num_requests"] == 8
    assert table.args_hash(table.db_key(batched)) != before


@pytest.mark.parametrize(
    "overrides",
    [
        {"q_dtype": "bf16", "kv_dtype": "bf16"},  # FA4 rows are fp8 q and KV
        {"o_dtype": "fp8_e4m3"},
        {"head_dim": 120},
    ],
)
def test_fa4_backends_refuse_what_the_fp8_paged_call_does_not_take(overrides):
    from profiling.runners.attention import fa4_block_rect
    from profiling.runners.exceptions import ProfilerNotImplemented

    spec = _rect_spec(q_dtype="fp8_e4m3", kv_dtype="fp8_e4m3", num_requests=4) | overrides
    with pytest.raises(ProfilerNotImplemented):
        fa4_block_rect._validate(**spec)
    assert fa4_block_rect._validate(**_rect_spec(q_dtype="fp8_e4m3", kv_dtype="fp8_e4m3")) == 1
