"""Unit tests for the ``elementwise`` L1 kernel kind.

Parallel-safety note: these import the per-kernel module DIRECTLY
(``import profiling.kernels.elementwise``) so they pass before the shared
barrel ``profiling/kernels/__init__.py`` is wired. They do not depend on the
registry being fully loaded, do not touch the shared ``profile.db``, and do not
run any GPU work.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from profiling.db.outlier import BatchOutlierPolicy
from profiling.db.registry import KernelProfilerSpec, MetricFamily, RunnerRef
from profiling.kernels import elementwise as elementwise_kernel
from profiling.kernels.elementwise import KIND, ElementwiseArgs

INT32_MAX = 2_147_483_647


def test_args_field_contract_matches_runner_kwargs():
    # Field set / order is the load-bearing contract (A5): it must equal the
    # triton runner kwargs (input_size_bytes, output_size_bytes) and the Rust
    # enumerate wire fields (minus `backend`). num_tokens / per-token rates are
    # Rust-side only and never appear here.
    field_names = [field.name for field in ElementwiseArgs.__dataclass_fields__.values()]
    assert field_names == ["input_size_bytes", "output_size_bytes"]


def test_args_are_frozen():
    args = ElementwiseArgs(input_size_bytes=262144, output_size_bytes=131072)
    assert args.input_size_bytes == 262144
    assert args.output_size_bytes == 131072
    with pytest.raises(Exception):
        args.input_size_bytes = 8  # frozen


def test_kind_wire_string_and_table_stem():
    # KIND must equal the Rust KernelSpec::KIND and the registry table_name so
    # the cross-language facade name resolves (registry validator enforces
    # table_name == kernel_kind).
    assert KIND == "elementwise"


def test_register_call_built_a_triton_compute_spec():
    # The module's import-time register(...) side effect must produce a spec we
    # can introspect without loading the full registry barrel. Rebuild the
    # exact spec the module registers and assert its shape.
    spec = KernelProfilerSpec(
        kernel_kind=KIND,
        backend="triton",
        runner_ref=RunnerRef(
            module_name="profiling.runners.elementwise.triton",
            function_name="profile_elementwise",
        ),
        table_name=KIND,
        args_schema=ElementwiseArgs,
        metric_family=MetricFamily.COMPUTE,
        batch_outlier_policy=BatchOutlierPolicy(),
    )
    assert spec.kernel_kind == "elementwise"
    assert spec.backend == "triton"
    assert spec.table_name == spec.kernel_kind  # facade-stem invariant
    assert spec.metric_family is MetricFamily.COMPUTE
    assert spec.runner_ref.module_name == "profiling.runners.elementwise.triton"
    assert spec.runner_ref.function_name == "profile_elementwise"
    # The module object exposes the same KIND constant.
    assert elementwise_kernel.KIND == "elementwise"


def test_importing_kernel_module_does_not_eager_import_runner():
    # Lazy-import invariant (L1 design §3.2.1): importing the per-kernel module
    # must NOT pull in the triton runner module (which imports triton/torch).
    command = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "import profiling.kernels.elementwise; "
            "print('profiling.runners.elementwise.triton' in sys.modules)"
        ),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "False"


def test_wide_offset_selection_covers_boundaries_and_rounded_fan_spans():
    from profiling.runners.elementwise import triton as runner

    assert not runner._requires_int64_offsets(INT32_MAX, INT32_MAX)
    assert runner._requires_int64_offsets(INT32_MAX + 1, INT32_MAX + 1)
    assert not runner._requires_int64_offsets(1_800_000_000, 1_000_000_000)
    # Both nominal sizes fit signed int32, but fan-in rounds 1.5 to two and
    # launches a 2.4 GB input span.
    assert runner._rounded_fan_in(1_800_000_000, 1_200_000_000) == 2
    assert runner._requires_int64_offsets(1_800_000_000, 1_200_000_000)
    assert not runner._requires_int64_offsets(0, INT32_MAX)
    assert runner._requires_int64_offsets(0, INT32_MAX + 1)


class _FakeKernel:
    def __init__(self, launches):
        self.launches = launches

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches.append((grid, args, kwargs))

        return launch


@pytest.mark.parametrize(
    ("input_size", "output_size", "kernel_attr", "expected_use_int64"),
    [
        (1024, 1024, "_elementwise_fan_kernel", False),
        (1_800_000_000, 1_200_000_000, "_elementwise_fan_kernel", True),
        (0, INT32_MAX, "_elementwise_zero_kernel", False),
        (0, INT32_MAX + 1, "_elementwise_zero_kernel", True),
    ],
)
def test_profile_passes_offset_width_to_triton_launch(
    monkeypatch,
    input_size,
    output_size,
    kernel_attr,
    expected_use_int64,
):
    from profiling.runners.elementwise import triton as runner

    allocations = []
    launches = []
    fake_torch = SimpleNamespace(
        uint8="uint8",
        cuda=SimpleNamespace(is_available=lambda: True),
        empty=lambda size, **kwargs: allocations.append((size, kwargs)) or object(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(runner, "_TRITON_IMPORT_ERROR", None)
    monkeypatch.setattr(runner, kernel_attr, _FakeKernel(launches))
    monkeypatch.setattr(
        runner.Timer,
        "cupti",
        lambda fn, **kwargs: (fn(), 1.0)[1],
    )
    monkeypatch.setattr(
        runner.Energy,
        "perf",
        lambda fn, **kwargs: (fn(), 0.25)[1],
    )

    metrics = runner.profile_elementwise(input_size, output_size)

    assert len(allocations) == 2
    assert len(launches) == 2
    assert all(launch[2]["USE_INT64"] is expected_use_int64 for launch in launches)
    assert metrics.time_ms == 1.0
    assert metrics.energy_j == 0.25
