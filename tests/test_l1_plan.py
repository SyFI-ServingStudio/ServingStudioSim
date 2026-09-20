from __future__ import annotations

import pytest

from profiling.db.batch import ProfileBatchOutcome, ProfileProvenance
from profiling.plan import (
    MIN_PIECE_SPECS,
    IssueReport,
    WorkCollector,
    WorkUnit,
    _take_next,
    collecting,
    issue,
)
from profiling.runners.metrics import ComputeMetrics


def _specs(count: int, offset: int = 0) -> list[dict]:
    return [{"m": index + offset, "n": 8, "k": 8, "dtype": "fp16"} for index in range(count)]


def _all_measured(specs: list[dict]) -> ProfileBatchOutcome:
    """An outcome whose every spec came back measured.

    `issue` reads the outcome, not just the absence of an exception:
    `execute_profile_batch` leaves `None` for a spec that failed rather than
    raising, so a stub returning `None` would look like total failure.
    """

    return ProfileBatchOutcome(
        results=[
            ComputeMetrics(time_ms=1.0, tflops=1.0, memory_bandwidth_gbps=1.0, energy_j=0.0)
            for _ in specs
        ],
        provenance=ProfileProvenance(source="measurement", requested_gpu_name="NVIDIA B200"),
    )


def _unit(size: int, name: str = "single_gemm", gpu_count: int = 1) -> WorkUnit:
    return WorkUnit(name, "torch_linear", tuple(_specs(size)), gpu_count)


def test_collector_drops_a_shape_two_cost_tree_nodes_both_asked_for() -> None:
    """Two nodes can miss on the same shape before either is measured; carrying
    it twice would spend a real measurement on the duplicate."""

    collector = WorkCollector()
    collector.record("single_gemm", "torch_linear", _specs(10))
    collector.record("single_gemm", "torch_linear", _specs(4))

    (unit,) = collector.units()
    assert unit.size == 10


def test_collector_keys_on_kind_and_backend_not_kind_alone() -> None:
    """Two backends of one kind are different runners and different envs."""

    collector = WorkCollector()
    collector.record("single_gemm", "torch_linear", _specs(3))
    collector.record("single_gemm", "torch_linear_vllm", _specs(3))

    assert {(unit.kernel_kind, unit.backend) for unit in collector.units()} == {
        ("single_gemm", "torch_linear"),
        ("single_gemm", "torch_linear_vllm"),
    }


def test_collector_orders_the_largest_unit_first() -> None:
    collector = WorkCollector()
    collector.record("rms_norm", "flashinfer", _specs(5))
    collector.record("single_gemm", "torch_linear", _specs(50))

    assert [unit.size for unit in collector.units()] == [50, 5]


def test_slicing_strides_so_an_odd_count_leaves_no_tail_piece() -> None:
    """Contiguous slicing of 257 into 2 leaves 128+128+1 and hands the largest
    shapes to the last piece; striding gives 129+128 with a mixed shape range."""

    pieces = WorkUnit("k", "b", tuple(_specs(257)), 1).sliced(2)

    assert [len(piece.specs) for piece in pieces] == [129, 128]
    assert {spec["m"] for piece in pieces for spec in piece.specs} == set(range(257))


def test_tail_split_does_not_fire_while_the_queue_is_deeper_than_the_free_cards() -> None:
    pending = [_unit(400), _unit(300), _unit(200)]
    report = IssueReport()

    taken = _take_next(pending, free_count=2, report=report)

    assert taken.size == 400
    assert report.split_units == []
    assert [unit.size for unit in pending] == [300, 200]


def test_tail_split_fires_when_cards_would_otherwise_idle() -> None:
    """The observation, not a forecast: fewer units left than free cards, so a
    card idles no matter what unless the last unit is spread."""

    pending = [_unit(400)]
    report = IssueReport()

    taken = _take_next(pending, free_count=4, report=report)

    assert taken.size == 100
    assert [unit.size for unit in pending] == [100, 100, 100]
    assert report.split_units == ["single_gemm:torch_linear x4"]


def test_tail_split_refuses_to_make_pieces_too_small_to_pay_their_own_setup() -> None:
    pending = [_unit(MIN_PIECE_SPECS * 2 - 1)]
    report = IssueReport()

    taken = _take_next(pending, free_count=4, report=report)

    assert taken.size == MIN_PIECE_SPECS * 2 - 1
    assert report.split_units == []


def test_tail_split_caps_pieces_at_the_minimum_size_not_the_card_count() -> None:
    pending = [_unit(MIN_PIECE_SPECS * 2)]
    report = IssueReport()

    _take_next(pending, free_count=8, report=report)

    assert report.split_units == ["single_gemm:torch_linear x2"]


def test_issue_runs_each_unit_whole_on_one_gpu(monkeypatch) -> None:
    """The point of the whole exercise: a unit is never fanned across cards, so
    its JIT/autotune is paid once rather than once per chunk."""

    import profiling.db.batch as batch

    seen = []

    def fake_execute(kernel_kind, specs, *, pool, db_path, gpu_name):
        seen.append((kernel_kind, len(specs), list(pool.gpus)))
        return _all_measured(specs)

    monkeypatch.setattr(batch, "execute_profile_batch", fake_execute)

    collector = WorkCollector()
    # One even share per card, so neither the oversize cut nor the tail split
    # fires and the property under test is the dispatch shape alone.
    for (kind, backend), size in zip(
        (
            ("single_gemm", "torch_linear"),
            ("rms_norm", "flashinfer"),
            ("nvfp4_quant", "vllm_cuda"),
            ("residual_rms_norm", "vllm_cuda"),
        ),
        (25, 25, 25, 25),
    ):
        collector.record(kind, backend, _specs(size))

    report = issue(collector, db_path=None, gpu_name="NVIDIA B200", gpus=[0, 1, 2, 3])

    assert report.failures == []
    assert all(len(gpus) == 1 for _, _, gpus in seen)
    assert len({tuple(gpus) for _, _, gpus in seen}) == 4  # four cards, no sharing
    assert sorted(count for _, count, _ in seen) == [25, 25, 25, 25]
    assert report.split_units == []


def test_issue_gives_a_collective_unit_every_card_it_asks_for(monkeypatch) -> None:
    import profiling.db.batch as batch

    seen = []
    monkeypatch.setattr(
        batch,
        "execute_profile_batch",
        lambda kind, specs, *, pool, db_path, gpu_name: (
            seen.append((kind, list(pool.gpus))) or _all_measured(specs)
        ),
    )

    collector = WorkCollector()
    collector.record("all_reduce", "nccl", _specs(8))
    monkeypatch.setattr("profiling.plan._gpu_count", lambda kind, backend, spec: 4)

    report = issue(collector, db_path=None, gpu_name=None, gpus=[0, 1, 2, 3])

    assert report.failures == []
    assert seen == [("all_reduce", [0, 1, 2, 3])]


def test_issue_reports_a_collective_that_cannot_fit_rather_than_hanging(monkeypatch) -> None:
    import profiling.db.batch as batch

    monkeypatch.setattr(
        batch, "execute_profile_batch", lambda *args, **kwargs: pytest.fail("must not run")
    )
    collector = WorkCollector()
    collector.record("all_reduce", "nccl", _specs(8))
    monkeypatch.setattr("profiling.plan._gpu_count", lambda kind, backend, spec: 4)

    report = issue(collector, db_path=None, gpu_name=None, gpus=[0])

    assert len(report.failures) == 1
    assert "needs 4 GPUs, have 1" in report.failures[0]


def test_collecting_context_restores_the_previous_collector() -> None:
    from profiling.plan import active_collector

    assert active_collector() is None
    with collecting() as outer:
        assert active_collector() is outer
        with collecting() as inner:
            assert active_collector() is inner
        assert active_collector() is outer
    assert active_collector() is None


def _count_missing(specs: list[dict], db) -> int:
    from profiling.facade import _coerce_input_specs
    from profiling.facade import _count_missing as count

    return count(
        "single_gemm",
        _coerce_input_specs("single_gemm", "torch_linear", specs),
        backend="torch_linear",
        gpu_name="NVIDIA B200",
        db_path=db,
    )


def test_count_missing_feeds_the_collector_during_a_dry_walk(tmp_path) -> None:
    """The simulator's dry-run mode calls only `count_missing_{kind}`, so that is
    where collection hooks in: the walk already visits every cost-tree node."""

    specs = [{"m": m, "n": 8, "k": 8, "dtype": "fp16"} for m in (16, 32, 64)]

    with collecting() as collector:
        missing = _count_missing(specs, tmp_path / "profile.db")

    assert missing == 3
    (unit,) = collector.units()
    assert unit.kernel_kind == "single_gemm"
    assert unit.backend == "torch_linear"
    assert {spec["m"] for spec in unit.specs} == {16, 32, 64}


def test_count_missing_is_unchanged_when_nothing_is_collecting(tmp_path) -> None:
    from profiling.plan import active_collector

    assert active_collector() is None
    assert _count_missing([{"m": 16, "n": 8, "k": 8, "dtype": "fp16"}], tmp_path / "db") == 1


def test_issue_collected_needs_no_gpu_when_the_cache_already_covers_the_run(monkeypatch) -> None:
    """A fill with nothing missing must not fail for want of an idle card."""

    import profiling.exec.local as local
    import profiling.perf_api as perf_api

    monkeypatch.setattr(local, "find_idle_gpus", lambda *a, **k: pytest.fail("must not ask"))
    perf_api.begin_collect()
    perf_api.issue_collected(0)


def test_begin_collect_resets_a_collector_a_failed_build_left_behind() -> None:
    """A build that raised mid-collect must not make the next build fail."""

    import profiling.perf_api as perf_api
    from profiling.plan import active_collector

    perf_api.begin_collect()
    active_collector().record("single_gemm", "torch_linear", _specs(3))
    perf_api.begin_collect()  # the abandoned collector, not an error
    try:
        assert len(active_collector()) == 0
    finally:
        perf_api.issue_collected(0)


def test_issue_collected_rejects_a_walk_that_counted_misses_but_recorded_none() -> None:
    """The failure this guards: the collector silently inactive while the counts
    still look right, so the cache build measures nothing and the next real run
    rediscovers every kernel one at a time."""

    import profiling.perf_api as perf_api

    perf_api.begin_collect()
    with pytest.raises(RuntimeError, match="would measure nothing"):
        perf_api.issue_collected(6407)


def test_issue_collected_accepts_recording_fewer_than_were_counted(monkeypatch) -> None:
    """Counts are summed per cost-tree node and two nodes can miss the same
    shape; the collector folds those into one, so fewer is normal."""

    import profiling.db.batch as batch
    import profiling.perf_api as perf_api
    from profiling.plan import active_collector

    monkeypatch.setattr(
        batch, "execute_profile_batch", lambda kind, specs, **k: _all_measured(specs)
    )
    monkeypatch.setattr("profiling.exec.local.find_idle_gpus", lambda *a, **k: [0])
    perf_api.begin_collect()
    active_collector().record("single_gemm", "torch_linear", _specs(20))
    perf_api.issue_collected(35)  # 35 node-level misses folded into 20 shapes


def test_issue_collected_rejects_recording_more_than_were_counted() -> None:
    import profiling.perf_api as perf_api
    from profiling.plan import active_collector

    perf_api.begin_collect()
    active_collector().record("single_gemm", "torch_linear", _specs(30))
    with pytest.raises(RuntimeError, match="cannot happen"):
        perf_api.issue_collected(9)


def test_issue_collected_without_begin_collect_is_an_error() -> None:
    import profiling.perf_api as perf_api

    with pytest.raises(RuntimeError, match="without begin_collect"):
        perf_api.issue_collected()


def test_collector_separates_specs_that_need_different_gpu_counts(monkeypatch) -> None:
    """`gpu_count` comes from the spec, not the kernel: the collective kinds read
    it out of `num_gpus`. One reservation cannot serve both, and sizing it from
    an arbitrary spec would hand a 4-GPU spec a 2-GPU pool."""

    monkeypatch.setattr("profiling.plan._gpu_count", lambda kind, backend, spec: spec["num_gpus"])

    collector = WorkCollector()
    collector.record(
        "all_reduce",
        "nccl",
        [{"num_gpus": 2, "bytes": 1}, {"num_gpus": 4, "bytes": 2}, {"num_gpus": 4, "bytes": 3}],
    )

    units = collector.units()
    assert sorted((unit.gpu_count, unit.size) for unit in units) == [(2, 1), (4, 2)]


def test_oversized_unit_is_cut_before_dispatch_so_it_cannot_hold_a_card_alone() -> None:
    """Without this the makespan has no bound beyond the single largest unit."""

    from profiling.plan import _slice_oversized

    report = IssueReport()
    units = [_unit(800), _unit(100, "rms_norm"), _unit(100, "nvfp4_quant")]

    out = _slice_oversized(units, gpu_count=4, report=report)

    # 1000 specs over 4 cards is a 250 share; 800 is more than three of them.
    assert report.split_units == ["single_gemm:torch_linear x4"]
    assert sorted(unit.size for unit in out) == [100, 100, 200, 200, 200, 200]


def test_an_already_balanced_set_is_left_alone() -> None:
    """Four equal units on four cards need no cutting; a rule that keyed on size
    alone would shred each into four and pay twelve extra fixed costs for
    nothing."""

    from profiling.plan import _slice_oversized

    report = IssueReport()
    units = [
        _unit(100),
        _unit(100, "rms_norm"),
        _unit(100, "nvfp4_quant"),
        _unit(100, "vllm_mla_rope"),
    ]

    out = _slice_oversized(units, gpu_count=4, report=report)

    assert report.split_units == []
    assert [unit.size for unit in out] == [100, 100, 100, 100]


def test_a_unit_just_under_the_even_share_is_left_whole() -> None:
    """The measured table's largest unit sits here: 1426 specs against a 1602
    share, so nothing is cut and nothing is paid."""

    from profiling.plan import _slice_oversized

    report = IssueReport()
    # 6407 specs over 4 cards is a 1601 share; the largest unit is 1426.
    units = [_unit(1426), _unit(4981, "rms_norm")]

    out = _slice_oversized(units, gpu_count=4, report=report)

    assert any(unit.size == 1426 for unit in out)
    assert report.split_units == ["rms_norm:torch_linear x4"]


def test_oversize_cut_never_makes_a_piece_too_small_to_pay_its_own_setup() -> None:
    from profiling.plan import _slice_oversized

    report = IssueReport()
    out = _slice_oversized([_unit(MIN_PIECE_SPECS * 2 + 1)], gpu_count=8, report=report)

    assert all(unit.size >= MIN_PIECE_SPECS for unit in out)


def test_oversize_cut_is_a_no_op_on_a_single_card() -> None:
    from profiling.plan import _slice_oversized

    report = IssueReport()
    out = _slice_oversized([_unit(800)], gpu_count=1, report=report)

    assert [unit.size for unit in out] == [800]
    assert report.split_units == []


def test_collector_carries_the_cache_key_the_miss_was_found_under() -> None:
    """The check and the fill must name the same key, or the build never converges.

    `gpu/spec.json` accepts aliases, so a run whose `gpu:` is `B200` checks key
    "B200"; letting the fill fall back to the worker's observed device name would
    insert under "NVIDIA B200" and leave the same specs missing on every build.
    """

    collector = WorkCollector()
    collector.record("single_gemm", "torch_linear", _specs(4), gpu_name="B200")

    assert [unit.gpu_name for unit in collector.units()] == ["B200"]


def test_collector_keeps_two_cache_keys_apart() -> None:
    """A PD deployment names a prefill GPU and a decode GPU in one walk.

    The dedupe is by spec identity, so without the key in the bucket these would
    fold into one unit and one of the two GPUs would never be measured.
    """

    collector = WorkCollector()
    collector.record("single_gemm", "torch_linear", _specs(4), gpu_name="NVIDIA B200")
    collector.record("single_gemm", "torch_linear", _specs(4), gpu_name="NVIDIA H200")

    units = collector.units()
    assert sorted(unit.gpu_name for unit in units) == ["NVIDIA B200", "NVIDIA H200"]
    assert len(collector) == 8


def test_issue_passes_each_unit_its_own_cache_key(monkeypatch) -> None:
    import profiling.db.batch as batch

    seen = []

    def fake_execute(kernel_kind, specs, *, pool, db_path, gpu_name):
        seen.append(gpu_name)
        return _all_measured(specs)

    monkeypatch.setattr(batch, "execute_profile_batch", fake_execute)

    collector = WorkCollector()
    collector.record("single_gemm", "torch_linear", _specs(4), gpu_name="B200")

    report = issue(collector, db_path=None, gpu_name="ignored fallback", gpus=[0])

    assert report.failures == []
    assert seen == ["B200"]


def test_issue_fails_a_unit_whose_specs_came_back_unmeasured(monkeypatch) -> None:
    """`execute_profile_batch` logs a failed spec and returns; it does not raise.

    The JIT path could rely on that, because the caller's next `table.query`
    reported the row as still missing and the command failed there. A cache build
    has no second look, so an OOM-killed worker would otherwise end with
    "cache build complete" over an empty table.
    """

    import profiling.db.batch as batch

    def fake_execute(kernel_kind, specs, *, pool, db_path, gpu_name):
        return ProfileBatchOutcome(results=[None for _ in specs], provenance=None)

    monkeypatch.setattr(batch, "execute_profile_batch", fake_execute)

    collector = WorkCollector()
    collector.record("single_gemm", "torch_linear", _specs(4))

    report = issue(collector, db_path=None, gpu_name=None, gpus=[0])

    assert len(report.failures) == 1
    assert "4 of 4 spec(s) were not measured" in report.failures[0]
    assert "single_gemm:torch_linear" in report.failures[0]
