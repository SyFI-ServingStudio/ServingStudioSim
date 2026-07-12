from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from profiling.profilers.timer import Timer


def test_timer_do_bench_uses_triton_testing(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[int, int]] = []
    timings = [2.0, 2.1, 1.95]

    def fake_do_bench(fn, *, warmup: int, rep: int) -> float:
        calls.append((warmup, rep))
        fn()
        return timings.pop(0)

    fake_triton = SimpleNamespace(testing=SimpleNamespace(do_bench=fake_do_bench))
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    # do_bench converts a rep COUNT to a ms budget via the estimate; pin it to
    # 1.0 ms/iter so rep=3 -> 3 ms and the call stays CUDA-free.
    monkeypatch.setattr("profiling.profilers.timer._estimate_per_iter_ms", lambda fn: 1.0)

    fn_calls = 0

    def fn() -> None:
        nonlocal fn_calls
        fn_calls += 1

    assert Timer.do_bench(fn, warmup=2, rep=3) == 2.0
    assert calls == [(2, 3), (2, 3), (2, 3)]
    assert fn_calls == 3


def test_timer_warns_when_aggregate_runs_diverge(monkeypatch: pytest.MonkeyPatch):
    timings = [1.0, 1.1, 1.05]

    def fake_do_bench(fn, *, warmup: int, rep: int) -> float:
        del warmup, rep
        fn()
        return timings.pop(0)

    fake_triton = SimpleNamespace(testing=SimpleNamespace(do_bench=fake_do_bench))
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setattr("profiling.profilers.timer._estimate_per_iter_ms", lambda fn: 1.0)

    with pytest.warns(RuntimeWarning, match="Timer aggregate runs differ by 1.10x"):
        assert Timer.do_bench(lambda: None, warmup=0, rep=1) == 1.05


def test_timer_cuda_event_uses_real_cuda_events(monkeypatch: pytest.MonkeyPatch):
    event_pairs: list[FakeEvent] = []

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            assert enable_timing is True
            self.recorded = False
            self.synchronized = False
            event_pairs.append(self)

        def record(self) -> None:
            self.recorded = True

        def synchronize(self) -> None:
            self.synchronized = True

        def elapsed_time(self, end: FakeEvent) -> float:
            assert self.recorded
            assert end.recorded
            return 12.0

    class FakeCuda:
        synchronize_calls = 0

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def synchronize() -> None:
            FakeCuda.synchronize_calls += 1

        Event = FakeEvent

    fake_torch = SimpleNamespace(cuda=FakeCuda)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fn_calls = 0

    def fn() -> None:
        nonlocal fn_calls
        fn_calls += 1

    assert Timer.cuda_event(fn, warmup=2, rep=4) == 3.0
    assert fn_calls == 14
    assert FakeCuda.synchronize_calls == 1
    assert len(event_pairs) == 6
    assert event_pairs[1].synchronized


def test_timer_cupti_uses_reusable_profiler(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, object]] = []
    timings = [3.0, 3.1, 2.95]

    def fake_profile_kernel(fn, **kwargs):
        fn()
        calls.append(kwargs)
        return SimpleNamespace(mean_ms=timings.pop(0))

    fake_cupti = SimpleNamespace(profile_kernel=fake_profile_kernel)
    monkeypatch.setitem(sys.modules, "profiling.profilers.cupti_kernel_profiler", fake_cupti)

    fn_calls = 0

    def fn() -> None:
        nonlocal fn_calls
        fn_calls += 1

    assert Timer.cupti(fn, warmup=3, rep=5, kernel_name="gemm") == 3.0
    assert fn_calls == 3
    assert calls == [
        {
            "num_warmup": 3,
            "num_iter": 5,
            "clear_l2_before_run": True,
            "clear_l2_between_launches": True,
            "kernel_name_contains": "gemm",
        },
        {
            "num_warmup": 3,
            "num_iter": 5,
            "clear_l2_before_run": True,
            "clear_l2_between_launches": True,
            "kernel_name_contains": "gemm",
        },
        {
            "num_warmup": 3,
            "num_iter": 5,
            "clear_l2_before_run": True,
            "clear_l2_between_launches": True,
            "kernel_name_contains": "gemm",
        },
    ]


def _fake_triton_recording(calls: list[tuple[int, int]], timings: list[float]):
    def fake_do_bench(fn, *, warmup: int, rep: int) -> float:
        calls.append((warmup, rep))
        fn()
        return timings.pop(0)

    return SimpleNamespace(testing=SimpleNamespace(do_bench=fake_do_bench))


def _fake_cupti_module(calls: list[dict[str, object]], *, mean_ms: float = 3.0):
    def fake_duration(fn, **kwargs):
        kwargs["_kind"] = "duration"
        calls.append(kwargs)
        return SimpleNamespace(mean_ms=mean_ms)

    def fake_profile_kernel(fn, **kwargs):
        kwargs["_kind"] = "fixed"
        fn()
        calls.append(kwargs)
        return SimpleNamespace(mean_ms=mean_ms)

    return SimpleNamespace(
        profile_kernel_for_duration=fake_duration,
        profile_kernel=fake_profile_kernel,
    )


def test_timer_cupti_default_estimates_then_captures_one_duration_window(
    monkeypatch: pytest.MonkeyPatch,
):
    # No rep => ten-sample estimate followed by one uninterrupted formal capture.
    calls: list[dict[str, object]] = []
    monkeypatch.setitem(
        sys.modules, "profiling.profilers.cupti_kernel_profiler", _fake_cupti_module(calls)
    )

    assert Timer.cupti(lambda: None, kernel_name="rmsnorm") == 3.0
    assert len(calls) == 1
    c = calls[0]
    assert c["_kind"] == "duration"
    assert (
        c["num_warmup"],
        c["estimate_iter"],
        c["clear_l2_before_run"],
        c["clear_l2_between_launches"],
        c["min_duration_ms"],
        c["min_iter"],
        c["max_iter"],
    ) == (
        0,
        10,
        True,
        True,
        2_000,
        20,
        5_000_000,
    )
    assert c["kernel_name_contains"] == "rmsnorm"


def test_timer_cupti_duration_count_bounds_pass_through(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, object]] = []
    monkeypatch.setitem(
        sys.modules, "profiling.profilers.cupti_kernel_profiler", _fake_cupti_module(calls)
    )

    assert (
        Timer.cupti(
            lambda: None,
            min_duration_ms=750,
            min_rep=50,
            max_rep=200,
        )
        == 3.0
    )
    c = calls[0]
    assert c["min_duration_ms"] == 750
    assert (c["min_iter"], c["max_iter"]) == (50, 200)


def test_timer_cupti_can_explicitly_disable_l2_displacement(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[dict[str, object]] = []
    monkeypatch.setitem(
        sys.modules, "profiling.profilers.cupti_kernel_profiler", _fake_cupti_module(calls)
    )

    assert Timer.cupti(lambda: None, clear_l2=False) == 3.0
    assert calls[0]["clear_l2_before_run"] is False
    assert calls[0]["clear_l2_between_launches"] is False


def test_timer_do_bench_default_is_time_centric(monkeypatch: pytest.MonkeyPatch):
    # No knobs => default 500 ms budget. do_bench is duration-native, so the
    # resolved iter count (max(ceil(500/2.0)=250, min_rep 3)=250) is converted
    # back to a ms budget: 250 * 2.0 = 500 ms handed to Triton.
    calls: list[tuple[int, int]] = []
    monkeypatch.setitem(sys.modules, "triton", _fake_triton_recording(calls, [2.0, 2.0, 2.0]))
    monkeypatch.setattr("profiling.profilers.timer._estimate_per_iter_ms", lambda fn: 2.0)

    assert Timer.do_bench(lambda: None, warmup=7) == 2.0
    assert calls == [(7, 500), (7, 500), (7, 500)]


def test_timer_do_bench_rep_count_converts_to_ms(monkeypatch: pytest.MonkeyPatch):
    # do_bench's Triton rep is a ms budget, so a rep COUNT is converted via the
    # estimate: 300 iters * 2.0 ms/iter = 600 ms.
    calls: list[tuple[int, int]] = []
    monkeypatch.setitem(sys.modules, "triton", _fake_triton_recording(calls, [1.0, 1.0, 1.0]))
    monkeypatch.setattr("profiling.profilers.timer._estimate_per_iter_ms", lambda fn: 2.0)

    assert Timer.do_bench(lambda: None, warmup=4, rep=300) == 1.0
    assert calls == [(4, 600), (4, 600), (4, 600)]


def test_timer_rep_is_mutually_exclusive_with_time_knobs():
    with pytest.raises(ValueError, match="mutually exclusive"):
        Timer.do_bench(lambda: None, warmup=0, rep=300, min_duration_ms=1000)
    with pytest.raises(ValueError, match="mutually exclusive"):
        Timer.cupti(lambda: None, rep=300, min_rep=5)
    with pytest.raises(ValueError, match="mutually exclusive"):
        Timer.cupti(lambda: None, rep=300, min_duration_ms=2_000)


def test_cupti_active_time_floor_and_convergence_are_both_required(
    monkeypatch: pytest.MonkeyPatch,
):
    from profiling.profilers import cupti_kernel_profiler as cupti

    monkeypatch.setattr(cupti, "_require_torch", lambda: SimpleNamespace())
    profiler = SimpleNamespace(clear_l2_bytes=0)

    def capture_once(fn, **kwargs):
        del fn, kwargs
        return [
            cupti.KernelRecord(
                name="gemm",
                device_id=0,
                stream_id=0,
                correlation_id=1,
                start_ns=0,
                end_ns=100_000_000,
                duration_ns=100_000_000,
            )
        ]

    profiler.capture_once = capture_once
    summary = cupti.CuptiKernelProfiler.profile_until_converged(
        profiler,
        lambda: None,
        batch=2,
        min_duration_ms=250,
        min_iter=2,
        max_iter=10,
        tol=0.01,
    )

    # Two identical samples have already converged, but contain only 200 ms.
    # The next complete batch takes active time to 400 ms before stopping.
    assert summary.num_iter == 4
    assert sum(summary.per_iter_ms) == pytest.approx(400.0)


def test_cupti_fails_instead_of_returning_a_short_duration(
    monkeypatch: pytest.MonkeyPatch,
):
    from profiling.profilers import cupti_kernel_profiler as cupti

    monkeypatch.setattr(cupti, "_require_torch", lambda: SimpleNamespace())
    profiler = SimpleNamespace(clear_l2_bytes=0)

    def capture_once(fn, **kwargs):
        del fn, kwargs
        return [
            cupti.KernelRecord(
                name="gemm",
                device_id=0,
                stream_id=0,
                correlation_id=1,
                start_ns=0,
                end_ns=100_000_000,
                duration_ns=100_000_000,
            )
        ]

    profiler.capture_once = capture_once
    with pytest.raises(RuntimeError, match="max_iter"):
        cupti.CuptiKernelProfiler.profile_until_converged(
            profiler,
            lambda: None,
            batch=1,
            min_duration_ms=500,
            min_iter=2,
            max_iter=3,
            tol=0.01,
        )


def test_cupti_multi_launch_unit_splits_target_and_l2_clear_records(
    monkeypatch: pytest.MonkeyPatch,
):
    from profiling.profilers import cupti_kernel_profiler as cupti

    monkeypatch.setattr(cupti, "_require_torch", lambda: SimpleNamespace())

    def target_fn(): ...

    clear_buffer = SimpleNamespace(sum=lambda: None)
    profiler = SimpleNamespace(clear_l2_bytes=0, _clear_buffer=clear_buffer)

    def record(name: str, ordinal: int, duration_ns: int) -> cupti.KernelRecord:
        return cupti.KernelRecord(
            name=name,
            device_id=0,
            stream_id=0,
            correlation_id=ordinal,
            start_ns=ordinal * 10,
            end_ns=ordinal * 10 + duration_ns,
            duration_ns=duration_ns,
        )

    def capture_once(fn, *, launches_per_run: int, **kwargs):
        del kwargs
        if launches_per_run == 1 and fn is target_fn:
            return [record("target_a", 0, 1_000_000), record("target_b", 1, 2_000_000)]
        if launches_per_run == 1:
            return [record("l2_clear", 0, 500_000)]
        assert launches_per_run == 3
        names = [
            ("target_a", 1_000_000),
            ("target_b", 2_000_000),
            ("l2_clear", 500_000),
            ("target_a", 1_000_000),
            ("target_b", 2_000_000),
            ("l2_clear", 500_000),
            ("target_a", 1_000_000),
            ("target_b", 2_000_000),
        ]
        return [record(name, ordinal, duration) for ordinal, (name, duration) in enumerate(names)]

    profiler.capture_once = capture_once
    summary = cupti.CuptiKernelProfiler.profile_until_converged(
        profiler,
        target_fn,
        batch=1,
        min_duration_ms=9,
        min_iter=2,
        max_iter=3,
        tol=0.01,
        launches_per_run=3,
        clear_l2_between_launches=True,
    )

    assert summary.num_iter == 3
    assert summary.per_iter_ms == pytest.approx([3.0, 3.0, 3.0])
    assert summary.matched_kernel_count_per_run == [2, 2, 2]
    assert summary.matched_kernel_names == ["target_a", "target_b"]


def test_cupti_duration_path_estimates_count_then_uses_one_formal_window(
    monkeypatch: pytest.MonkeyPatch,
):
    from profiling.profilers import cupti_kernel_profiler as cupti

    monkeypatch.setattr(cupti, "_require_torch", lambda: SimpleNamespace())
    pattern = cupti._LaunchPattern(callable_kernel_names=("gemm",), clear_kernel_names=())
    monkeypatch.setattr(cupti, "_prepare_launch_pattern", lambda *args, **kwargs: pattern)
    launches: list[tuple[int, bool, bool]] = []

    def capture_unit(
        profiler,
        fn,
        *,
        launches_per_run,
        clear_l2_before_run,
        clear_l2_between_launches,
        **kwargs,
    ):
        del profiler, fn, kwargs
        launches.append((launches_per_run, clear_l2_before_run, clear_l2_between_launches))
        value_ms = 100.0 if launches_per_run == 10 else 110.0
        return [value_ms] * launches_per_run, {"gemm"}, [1] * launches_per_run

    monkeypatch.setattr(cupti, "_capture_launch_unit", capture_unit)
    profiler = SimpleNamespace(clear_l2_bytes=0)
    summary = cupti.CuptiKernelProfiler.profile_for_duration(
        profiler,
        lambda: None,
        estimate_iter=10,
        min_duration_ms=250,
        min_iter=2,
        max_iter=10,
    )

    # ceil(250 / 100) = 3; the formal measurement is one three-launch window.
    assert launches == [(10, True, True), (3, True, True)]
    assert summary.num_iter == 3
    assert summary.launches_per_run == 3
    assert summary.per_iter_ms == [110.0, 110.0, 110.0]
    assert summary.mean_ms == 110.0


def test_cupti_duration_path_rejects_estimated_count_above_cap(
    monkeypatch: pytest.MonkeyPatch,
):
    from profiling.profilers import cupti_kernel_profiler as cupti

    monkeypatch.setattr(cupti, "_require_torch", lambda: SimpleNamespace())
    pattern = cupti._LaunchPattern(callable_kernel_names=("gemm",), clear_kernel_names=())
    monkeypatch.setattr(cupti, "_prepare_launch_pattern", lambda *args, **kwargs: pattern)
    monkeypatch.setattr(
        cupti,
        "_capture_launch_unit",
        lambda *args, launches_per_run, **kwargs: (
            [0.1] * launches_per_run,
            {"gemm"},
            [1] * launches_per_run,
        ),
    )
    profiler = SimpleNamespace(clear_l2_bytes=0)
    with pytest.raises(RuntimeError, match="exceeds max_iter"):
        cupti.CuptiKernelProfiler.profile_for_duration(
            profiler,
            lambda: None,
            estimate_iter=10,
            min_duration_ms=2_000,
            min_iter=20,
            max_iter=100,
        )


def test_timer_cupti_rep_uses_fixed_path(monkeypatch: pytest.MonkeyPatch):
    # rep => deterministic fixed-count path: profile_kernel with median-of-3.
    calls: list[dict[str, object]] = []
    monkeypatch.setitem(
        sys.modules, "profiling.profilers.cupti_kernel_profiler", _fake_cupti_module(calls)
    )

    assert Timer.cupti(lambda: None, warmup=2, rep=7, kernel_name="g") == 3.0
    assert len(calls) == 3  # median of 3 aggregate runs
    assert all(c["_kind"] == "fixed" and c["num_iter"] == 7 for c in calls)
    assert all(c["clear_l2_before_run"] is True for c in calls)
    assert all(c["clear_l2_between_launches"] is True for c in calls)


def test_cupti_l2_displacement_reads_buffer_then_synchronizes(
    monkeypatch: pytest.MonkeyPatch,
):
    from profiling.profilers import cupti_kernel_profiler as cupti

    calls: list[object] = []
    buffer = SimpleNamespace(device="cuda:0", sum=lambda: calls.append("sum"))
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda device: calls.append(("sync", device)))
    )
    monkeypatch.setattr(cupti, "_require_torch", lambda: fake_torch)

    cupti.clear_l2_cache(buffer)

    assert calls == ["sum", ("sync", "cuda:0")]


def test_timer_cuda_event_default_runs_time_centric(monkeypatch: pytest.MonkeyPatch):
    # No knobs => default 500 ms; iteration timer keeps the count directly.
    # ceil(500 / 0.25) = 2000 iters, above the min_rep floor.
    monkeypatch.setattr("profiling.profilers.timer._estimate_per_iter_ms", lambda fn: 0.25)
    loop_iters: list[int] = []

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            self.n = 0

        def record(self) -> None: ...
        def synchronize(self) -> None: ...

        def elapsed_time(self, end: FakeEvent) -> float:
            return 5000.0

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def synchronize() -> None: ...

        Event = FakeEvent

    def counting_fn() -> None:
        loop_iters.append(1)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCuda))

    # 5000 ms elapsed / 2000 iters = 2.5 ms; aggregate median of 3 == 2.5.
    assert Timer.cuda_event(counting_fn, warmup=0) == 2.5
    # 3 aggregate runs * 2000 iters each (warmup=0, estimate is monkeypatched out).
    assert len(loop_iters) == 6000


def test_timer_warns_multi_gpu_in_min_duration_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "triton", _fake_triton_recording([], [1.0, 1.0, 1.0]))
    monkeypatch.setattr("profiling.profilers.timer._estimate_per_iter_ms", lambda fn: 1.0)
    fake_torch = SimpleNamespace(
        distributed=SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            get_world_size=lambda: 4,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.warns(RuntimeWarning, match="non-deterministic across the 4 ranks"):
        Timer.do_bench(lambda: None, warmup=0, min_duration_ms=1000)


def test_local_cupti_profiler_points_at_csrc_extension():
    from profiling.profilers import cupti_kernel_profiler

    assert cupti_kernel_profiler.EXT_SOURCE.name == "cupti_activity_profiler.cpp"
    assert cupti_kernel_profiler.EXT_SOURCE.exists()
