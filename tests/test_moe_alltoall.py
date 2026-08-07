"""Focused tests for the ``moe_alltoall`` / ``moe_alltoall_prepare`` L1 contracts.

Both kinds are profiled by the same 8-rank runner, so what is worth pinning here
is the wire schema, the registry row, and the pure index-plan construction the
runner uses to realise a ``(max_send_rows, max_recv_rows)`` cell exactly. The
transfer itself needs a real NVLink domain and lives in the GPU tier.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import fields
from types import SimpleNamespace

import pytest

from profiling.db.batch import coerce_args
from profiling.db.registry import MetricFamily, find_kernel_profiler_spec
from profiling.kernels.moe_alltoall import KIND as ALLTOALL_KIND
from profiling.kernels.moe_alltoall import MoeAlltoallArgs
from profiling.kernels.moe_alltoall_prepare import KIND as PREPARE_KIND
from profiling.kernels.moe_alltoall_prepare import MoeAlltoallPrepareArgs

_BACKEND = "flashinfer_mnnvl"


def test_alltoall_args_field_order_matches_the_rust_payload():
    assert [field.name for field in fields(MoeAlltoallArgs)] == [
        "ep_size",
        "max_send_rows",
        "max_recv_rows",
        "top_k",
        "slot_count",
        "hidden_bytes",
        "direction",
        "fabric",
    ]
    arguments = coerce_args(
        MoeAlltoallArgs,
        {
            "ep_size": 8,
            "max_send_rows": 8190,
            "max_recv_rows": 4100,
            "top_k": 8,
            "slot_count": 160,
            "hidden_bytes": 10240,
            "direction": "dispatch",
            "fabric": "nvlink",
        },
    )
    assert arguments == MoeAlltoallArgs(8, 8190, 4100, 8, 160, 10240, "dispatch", "nvlink")


def test_prepare_args_stay_keyed_by_tokens_not_rows():
    # The metadata pass reads the `tokens x top_k` routing table and bins it over
    # `ep_size x slot_count`. It never touches the activations, so neither a
    # payload width nor the transfer's row counts belong in its key.
    assert [field.name for field in fields(MoeAlltoallPrepareArgs)] == [
        "ep_size",
        "tokens_per_rank",
        "top_k",
        "slot_count",
        "fabric",
    ]


@pytest.mark.parametrize(
    ("kind", "args_schema", "function_name"),
    [
        (ALLTOALL_KIND, MoeAlltoallArgs, "profile_moe_alltoall_batch"),
        (
            PREPARE_KIND,
            MoeAlltoallPrepareArgs,
            "profile_moe_alltoall_prepare_batch",
        ),
    ],
)
def test_registry_contract_reserves_the_whole_ep_group(kind, args_schema, function_name):
    profiler_spec = find_kernel_profiler_spec(kind, _BACKEND)
    assert profiler_spec.table_name == kind
    assert profiler_spec.args_schema is args_schema
    assert profiler_spec.metric_family is MetricFamily.COMM
    assert profiler_spec.runner_ref.module_name == (
        "profiling.runners.moe.flashinfer_mnnvl_alltoall"
    )
    assert profiler_spec.runner_ref.function_name == function_name
    # One spawn per chunk: the fabric workspace is the expensive part, not the shape.
    assert profiler_spec.list_native
    assert profiler_spec.gpu_count_fn({"ep_size": 8}) == 8
    assert profiler_spec.gpu_count_fn({"ep_size": 4}) == 4


def test_import_is_lazy_for_runner_torch_and_flashinfer():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import profiling.kernels.moe_alltoall; "
                "import profiling.kernels.moe_alltoall_prepare; "
                "print('profiling.runners.moe.flashinfer_mnnvl_alltoall' in sys.modules, "
                "'torch' in sys.modules, 'flashinfer' in sys.modules)"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "False False False"


def test_a_single_rank_group_is_rejected_without_spawning():
    from profiling.runners.moe import flashinfer_mnnvl_alltoall as runner

    specs = [{"ep_size": 1, "max_send_rows": 64}, {"ep_size": 1, "max_send_rows": 128}]
    results = runner.profile_moe_alltoall_batch(specs)

    # One error per spec so the caller's 1:1 zip still holds.
    assert len(results) == len(specs)
    assert all("at least 2 ranks" in result.error for result in results)


def _margins(plan):
    ep_size = len(plan)
    rows = [sum(row) for row in plan]
    columns = [sum(plan[r][d] for r in range(ep_size)) for d in range(ep_size)]
    return rows, columns


@pytest.mark.parametrize(
    ("max_send_rows", "max_recv_rows"),
    [(1, 1), (16, 8), (1024, 1024), (8192, 1024), (1024, 8192), (43000, 43000)],
)
def test_the_plan_hits_both_requested_maxima_exactly(max_send_rows, max_recv_rows):
    from profiling.runners.moe.flashinfer_mnnvl_alltoall import _transfer_matrix

    plan = _transfer_matrix(max_send_rows, max_recv_rows, 8)
    rows, columns = _margins(plan)

    # The cell's key must be what the hardware actually sees, not a target the
    # rounding drifted away from.
    assert max(rows) == max_send_rows
    assert max(columns) == max_recv_rows
    # Every row sent is a row received.
    assert sum(rows) == sum(columns)


def test_the_smaller_side_is_flat_and_the_larger_side_carries_the_skew():
    from profiling.runners.moe.flashinfer_mnnvl_alltoall import _transfer_matrix

    send_skewed = _transfer_matrix(8192, 2048, 8)
    rows, columns = _margins(send_skewed)
    assert columns == [2048] * 8, "conservation leaves the receive side no slack"
    assert rows[0] == 8192 and max(rows[1:]) < 8192

    # And the transpose case, which is the other half of the axis probe.
    recv_skewed = _transfer_matrix(2048, 8192, 8)
    rows, columns = _margins(recv_skewed)
    assert rows == [2048] * 8
    assert columns[0] == 8192 and max(columns[1:]) < 8192


def test_a_balanced_cell_spreads_over_every_peer_link():
    from profiling.runners.moe.flashinfer_mnnvl_alltoall import _transfer_matrix

    plan = _transfer_matrix(4096, 4096, 8)
    # The product plan, not the northwest-corner one: a banded matrix would hit
    # the same margins while leaving most peer links idle, which is a different
    # collective.
    assert all(count > 0 for row in plan for count in row)


def test_a_ratio_wider_than_the_group_is_refused_rather_than_approximated():
    from profiling.runners.exceptions import KernelLaunchFailed
    from profiling.runners.moe.flashinfer_mnnvl_alltoall import _transfer_matrix

    # 8192 rows out of one rank means at least 8192 rows into the group, so the
    # busiest receiver takes at least 1024 of them. 512 cannot happen, and the
    # Rust `infeasible_mask` strips exactly these cells before profiling.
    with pytest.raises(KernelLaunchFailed, match="within a factor of ep_size"):
        _transfer_matrix(8192, 512, 8)
    with pytest.raises(KernelLaunchFailed, match="within a factor of ep_size"):
        _transfer_matrix(512, 8192, 8)
    # The boundary itself is feasible.
    _transfer_matrix(8192, 1024, 8)


def test_fan_out_is_the_distinct_destination_count_not_top_k():
    from profiling.runners.moe.flashinfer_mnnvl_alltoall import _destinations_per_token

    # GLM-5.2: 8 picks out of 256 slots over 8 ranks collide often enough that a
    # token reaches ~5.3 ranks, not 8. Sizing the dispatch send buffer by top_k
    # would make it 1.5x too small and change the read reuse the kernel sees.
    assert _destinations_per_token(8, 8, 256) == pytest.approx(5.2947, abs=1e-3)
    # One pick can only ever reach one rank...
    assert _destinations_per_token(8, 1, 256) == 1.0
    # ...and enough picks reach every rank.
    assert _destinations_per_token(8, 256, 256) == 8.0


def test_send_indices_reuse_token_rows_across_destinations_but_not_within_one():
    from profiling.runners.moe.flashinfer_mnnvl_alltoall import _rank_plan, _transfer_matrix

    fake_torch = SimpleNamespace(tensor=lambda values, **_: list(values), int32="int32")
    ep_size = 8
    plan = _transfer_matrix(4096, 4096, ep_size)
    input_rows = 800  # fewer rows than the 4096 sent: dispatch's read reuse
    send_cumsum, send_indices, recv_cumsum, recv_indices = _rank_plan(
        fake_torch, plan, rank=0, ep_size=ep_size, input_rows=input_rows
    )

    assert send_cumsum[-1] == sum(plan[0]) == len(send_indices)
    assert recv_cumsum[-1] == sum(plan[r][0] for r in range(ep_size)) == len(recv_indices)
    assert all(0 <= index < input_rows for index in send_indices)

    # A router selects without replacement, so no token row may be sent to the
    # same destination twice — but it is read once per destination it reaches.
    start = 0
    for destination, end in enumerate(send_cumsum):
        segment = send_indices[start:end]
        assert len(set(segment)) == len(segment), f"destination {destination} repeats a row"
        start = end
    assert len(set(send_indices)) < len(send_indices)

    # Each arriving row lands in its own slot, in both directions.
    assert recv_indices == list(range(len(recv_indices)))


def test_graph_iteration_count_is_sized_from_the_shape_not_fixed():
    from profiling.runners.moe import flashinfer_mnnvl_alltoall as runner

    # A fixed 100 iterations per graph is exactly what the row axis cannot
    # afford: at the top of the axis one call moves ~800 MB, so 100 of them
    # would be 80 GB of traffic inside one capture.
    assert runner._iters_per_graph(32 * 12_288) == 100  # decode
    assert runner._iters_per_graph(65_536 * 12_288) == 10  # top of the row axis
    assert runner._MIN_ITERS_PER_GRAPH >= 2, "one iteration would measure the replay launch"
    # Monotone: a bigger shape never captures more.
    counts = [runner._iters_per_graph(rows * 12_288) for rows in (32, 1_024, 8_192, 65_536)]
    assert counts == sorted(counts, reverse=True)

    # Budgeting by bytes rather than by count is also what bounds the sweep's
    # wall clock: above the point where the budget bites, every cell replays the
    # same total traffic, so a cell costs the same whether it is 8k rows or 64k.
    for rows in (8_192, 16_384, 65_536):
        traffic = runner._iters_per_graph(rows * 12_288) * rows * 12_288
        assert traffic == pytest.approx(runner._GRAPH_BYTE_BUDGET, rel=0.15)


def test_the_capture_length_depends_on_nothing_a_rank_could_disagree_about():
    """The deadlock guard.

    An earlier version timed the call first and sized the graph from that. Every
    rank measured its own eager loop, two of them landed on either side of a
    rounding boundary, captured different iteration counts, and replayed
    different numbers of collectives — the ranks that finished early spun inside
    the transfer waiting for peers that never came, and the 8-rank group hung
    (GPUs 2 and 4 at 0% util, 3/5/6/7 at 100%).

    So the capture length must be a pure function of the spec, which every rank
    holds identically, and never of a measurement, which is per-rank by nature.
    """
    import inspect

    from profiling.runners.moe import flashinfer_mnnvl_alltoall as runner

    assert runner._iters_per_graph(4_096 * 12_288) == runner._iters_per_graph(4_096 * 12_288)

    # Its only argument is the byte count, and its body reads nothing else.
    signature = inspect.signature(runner._iters_per_graph)
    assert list(signature.parameters) == ["bytes_per_call"]
    source = inspect.getsource(runner._iters_per_graph)
    for rank_local in ("rank", "elapsed", "Event", "time", "all_reduce"):
        assert rank_local not in source.split('"""')[-1], (
            f"{rank_local!r} in the capture-length decision would make it rank-dependent"
        )


def test_the_capture_budget_is_the_cells_maxima_not_this_ranks_share():
    """The second half of the deadlock guard, which the first half did not cover.

    Making `_iters_per_graph` pure was not enough: the ARGUMENT stayed
    rank-dependent. `_time_one_transfer` sized the graph from
    `max(send_total, recv_total)`, and those are this rank's own row and column
    of the plan — which `_transfer_matrix` skews on purpose, handing rank 0
    `max_send_rows` and its peers less. Same deadlock, one level up: rank 0
    captured fewer iterations, ran out of collectives first, and waited in the
    barrier while ranks 1-7 spun in a transfer whose last peer never came.

    So the budget must be built from the spec's own maxima, which every rank
    holds identically, and never from `plan[rank]`.
    """
    import inspect

    from profiling.runners.moe import flashinfer_mnnvl_alltoall as runner

    source = inspect.getsource(runner._time_one_transfer)
    body = source.split('"""')[-1]
    budget_lines = [
        line
        for line in body.splitlines()
        if "_graph_time_ms(" in line or "graph_budget_bytes =" in line
    ]
    assert len(budget_lines) == 2, "expected one budget assignment and one timing call"
    budget = " ".join(budget_lines)
    for rank_dependent in ("send_total", "recv_total", "plan[", "rank"):
        assert rank_dependent not in budget, (
            f"{rank_dependent!r} in the capture budget makes it rank-dependent again"
        )
    # And it is the cell's key, so two ranks of the same cell cannot disagree.
    assert "max_send_rows" in budget and "max_recv_rows" in budget

    # The prepare side is keyed the same way: tokens/world_size/top_k are spec
    # fields, so its budget was never rank-dependent and must stay that way.
    prepare_body = inspect.getsource(runner._time_one_prepare).split('"""')[-1]
    prepare_budget = [line for line in prepare_body.splitlines() if "_graph_time_ms(" in line]
    assert len(prepare_budget) == 1
    assert "rank" not in prepare_budget[0]


def test_generated_facade_symbols_exist():
    from profiling import perf_api

    assert hasattr(perf_api, "get_moe_alltoall_times")
    assert hasattr(perf_api, "count_missing_moe_alltoall")
    assert hasattr(perf_api, "get_moe_alltoall_prepare_times")
    assert hasattr(perf_api, "count_missing_moe_alltoall_prepare")
