"""Contracts for the fixed-pattern MNNVL exchange rows (runners/comm/moe_ep_mnnvl_lamport.py)."""

from __future__ import annotations

import pytest

from profiling.db.registry import find_kernel_profiler_spec
from profiling.runners.comm import moe_ep_mnnvl_lamport as runner
from profiling.runners.comm.mnnvl_fixed_exchange import DispatchRow, ExchangeLayout

_GATHER = {
    "num_gpus": 8,
    "per_rank_tokens": [128] * 8,
    "hidden_size": 4096,
    "top_k": 8,
    "activation_dtype": "nvfp4_e2m1",
    "fabric": "nvlink",
}
_SCATTER = {
    "num_gpus": 8,
    "per_rank_tokens": [128] * 8,
    "hidden_size": 4096,
    "dtype": "bf16",
    "fabric": "nvlink",
}


@pytest.mark.parametrize(
    ("kind", "backend", "function"),
    [
        (
            "moe_ep_quantized_all_gather",
            "mnnvl_lamport",
            "profile_moe_ep_quantized_all_gather_batch",
        ),
        ("moe_ep_reduce_scatter", "mnnvl_lamport", "profile_moe_ep_reduce_scatter_batch"),
        ("moe_ep_reduce_scatter", "mnnvl_pull", "profile_moe_ep_reduce_scatter_pull_batch"),
        (
            "moe_ep_quantize_and_all_gather",
            "mnnvl_lamport",
            "profile_moe_ep_quantize_and_all_gather_batch",
        ),
        ("row_all_gather", "mnnvl_lamport", "profile_all_gather_batch"),
        ("row_all_gather", "mnnvl_pull", "profile_all_gather_pull_batch"),
    ],
)
def test_each_backend_runs_its_leg_in_fabric_memory(kind, backend, function):
    spec = find_kernel_profiler_spec(kind, backend)
    assert spec.runner_ref.module_name == runner.__name__
    assert spec.runner_ref.function_name == function
    assert callable(getattr(runner, function))
    assert spec.imex_channel and spec.list_native
    assert spec.gpu_count_fn({"num_gpus": 8}) == 8


def test_dispatch_row_is_vllms_naive_prepare_payload():
    # GLM-5.3-Flash: 2048 packed NVFP4 + 256 e4m3 scales + 32 fp32 weights + 32 int32 ids.
    row = DispatchRow.nvfp4(4096, 8)
    assert (row.payload, row.scales, row.weights, row.ids, row.total) == (2048, 256, 32, 32, 2368)
    layout = ExchangeLayout.build(8, 1024, row, 4096)
    assert layout.dispatch_set >= 8 * 1024 * 2368
    assert layout.partials_set >= 8 * 1024 * 4096 * 2
    assert layout.total == layout.partials + 2 * layout.partials_set


def test_spec_error_names_what_the_kernels_cannot_run():
    assert runner.spec_error(_GATHER, "dispatch") is None
    assert runner.spec_error(_SCATTER, "pull") is None
    assert (
        "nvfp4"
        in runner.spec_error({**_GATHER, "activation_dtype": "mxfp8_e4m3"}, "dispatch").lower()
    )
    assert "multiple of 256" in runner.spec_error(
        {**_GATHER, "hidden_size": 4000}, "quant_dispatch"
    )
    assert "bf16" in runner.spec_error({**_SCATTER, "dtype": "fp32"}, "combine")
    assert "2..8" in runner.spec_error(
        {**_SCATTER, "num_gpus": 16, "per_rank_tokens": [1] * 16}, "combine"
    )
    assert "no tokens" in runner.spec_error({**_SCATTER, "per_rank_tokens": [0] * 8}, "combine")


def test_bf16_row_layout_sends_one_plane():
    row = DispatchRow.bf16(4096)
    assert (row.payload, row.scales, row.weights, row.ids, row.total) == (0, 8192, 0, 0, 8192)
    layout = ExchangeLayout.build(4, 656, row, 4096)
    assert layout.dispatch_set >= 4 * 656 * 8192
    assert (
        runner.spec_error({**_SCATTER, "num_gpus": 2, "per_rank_tokens": [8, 0]}, "gather") is None
    )


@pytest.mark.parametrize(
    ("kind", "backend", "num_gpus"),
    [
        ("row_all_gather", "mnnvl_lamport", 2),
        ("row_all_gather", "mnnvl_pull", 4),
        ("moe_ep_reduce_scatter", "mnnvl_pull", 2),
    ],
)
def test_group_rows_store_beside_the_legacy_all_gather_table(tmp_path, kind, backend, num_gpus):
    """A b < 8 group row lands in its own table, also in a DB that still carries
    the unregistered legacy `all_gather` table (num_rows/num_cols schema)."""
    import sqlite3

    from profiling.db.batch import coerce_args
    from profiling.db.table import ProfileRow, Table
    from profiling.runners.metrics import CommMetrics

    database_path = tmp_path / "profile.db"
    with sqlite3.connect(database_path) as conn:
        conn.execute(
            "CREATE TABLE all_gather (id INTEGER PRIMARY KEY, num_rows INTEGER, num_cols INTEGER)"
        )
    spec = find_kernel_profiler_spec(kind, backend)
    tokens = [8] + [0] * (num_gpus - 1)
    args = coerce_args(
        spec.args_schema,
        {**_SCATTER, "num_gpus": num_gpus, "per_rank_tokens": tokens},
    )
    table = Table(spec, database_path)
    metrics = CommMetrics(time_ms=0.005, algbw_gbps=10.0, busbw_gbps=10.0)
    table.insert([ProfileRow(args=args, metrics=metrics, gpu_name="TestGPU", backend=backend)])
    (got,) = table.query([args], gpu_name="TestGPU")
    assert isinstance(got, CommMetrics) and got.time_ms == 0.005
