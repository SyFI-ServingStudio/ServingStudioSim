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
        ("all_gather", "mnnvl_lamport", "profile_all_gather_batch"),
        ("all_gather", "mnnvl_pull", "profile_all_gather_pull_batch"),
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
