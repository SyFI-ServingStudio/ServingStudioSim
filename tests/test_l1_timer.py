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
    monkeypatch.setattr(
        "profiling.profilers.timer._estimate_per_iter_ms", lambda fn: 1.0
    )

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
    monkeypatch.setattr(
        "profiling.profilers.timer._estimate_per_iter_ms", lambda fn: 1.0
    )

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
            "kernel_name_contains": "gemm",
        },
        {
            "num_warmup": 3,
            "num_iter": 5,
            "kernel_name_contains": "gemm",
        },
        {
            "num_warmup": 3,
            "num_iter": 5,
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
    def fake_converged(fn, **kwargs):
        kwargs["_kind"] = "converged"
        calls.append(kwargs)
        return SimpleNamespace(mean_ms=mean_ms)

    def fake_profile_kernel(fn, **kwargs):
        kwargs["_kind"] = "fixed"
        fn()
        calls.append(kwargs)
        return SimpleNamespace(mean_ms=mean_ms)

    return SimpleNamespace(
        profile_kernel_until_converged=fake_converged,
        profile_kernel=fake_profile_kernel,
    )


def test_timer_cupti_default_converges_adaptively(monkeypatch: pytest.MonkeyPatch):
    # No rep => adaptive convergence path with the default knobs, no warmup,
    # no median-of-3 (the convergence loop already aggregates).
    calls: list[dict[str, object]] = []
    monkeypatch.setitem(
        sys.modules, "profiling.profilers.cupti_kernel_profiler", _fake_cupti_module(calls)
    )

    assert Timer.cupti(lambda: None, kernel_name="rmsnorm") == 3.0
    assert len(calls) == 1
    c = calls[0]
    assert c["_kind"] == "converged"
    assert (c["num_warmup"], c["batch"], c["min_iter"], c["max_iter"], c["tol"]) == (
        0,
        10,
        20,
        500,
        0.01,
    )
    assert c["kernel_name_contains"] == "rmsnorm"


def test_timer_cupti_convergence_knobs_pass_through(monkeypatch: pytest.MonkeyPatch):
    calls: list[dict[str, object]] = []
    monkeypatch.setitem(
        sys.modules, "profiling.profilers.cupti_kernel_profiler", _fake_cupti_module(calls)
    )

    assert Timer.cupti(lambda: None, min_rep=50, max_rep=200, tol=0.02) == 3.0
    c = calls[0]
    assert (c["min_iter"], c["max_iter"], c["tol"]) == (50, 200, 0.02)


def test_timer_do_bench_default_is_time_centric(monkeypatch: pytest.MonkeyPatch):
    # No knobs => default 500 ms budget. do_bench is duration-native, so the
    # resolved iter count (max(ceil(500/2.0)=250, min_rep 3)=250) is converted
    # back to a ms budget: 250 * 2.0 = 500 ms handed to Triton.
    calls: list[tuple[int, int]] = []
    monkeypatch.setitem(
        sys.modules, "triton", _fake_triton_recording(calls, [2.0, 2.0, 2.0])
    )
    monkeypatch.setattr(
        "profiling.profilers.timer._estimate_per_iter_ms", lambda fn: 2.0
    )

    assert Timer.do_bench(lambda: None, warmup=7) == 2.0
    assert calls == [(7, 500), (7, 500), (7, 500)]


def test_timer_do_bench_rep_count_converts_to_ms(monkeypatch: pytest.MonkeyPatch):
    # do_bench's Triton rep is a ms budget, so a rep COUNT is converted via the
    # estimate: 300 iters * 2.0 ms/iter = 600 ms.
    calls: list[tuple[int, int]] = []
    monkeypatch.setitem(
        sys.modules, "triton", _fake_triton_recording(calls, [1.0, 1.0, 1.0])
    )
    monkeypatch.setattr(
        "profiling.profilers.timer._estimate_per_iter_ms", lambda fn: 2.0
    )

    assert Timer.do_bench(lambda: None, warmup=4, rep=300) == 1.0
    assert calls == [(4, 600), (4, 600), (4, 600)]


def test_timer_rep_is_mutually_exclusive_with_time_knobs():
    with pytest.raises(ValueError, match="mutually exclusive"):
        Timer.do_bench(lambda: None, warmup=0, rep=300, min_duration_ms=1000)
    with pytest.raises(ValueError, match="mutually exclusive"):
        Timer.cupti(lambda: None, rep=300, min_rep=5)


def test_timer_cupti_rep_uses_fixed_path(monkeypatch: pytest.MonkeyPatch):
    # rep => deterministic fixed-count path: profile_kernel with median-of-3.
    calls: list[dict[str, object]] = []
    monkeypatch.setitem(
        sys.modules, "profiling.profilers.cupti_kernel_profiler", _fake_cupti_module(calls)
    )

    assert Timer.cupti(lambda: None, warmup=2, rep=7, kernel_name="g") == 3.0
    assert len(calls) == 3  # median of 3 aggregate runs
    assert all(c["_kind"] == "fixed" and c["num_iter"] == 7 for c in calls)


def test_timer_cuda_event_default_runs_time_centric(monkeypatch: pytest.MonkeyPatch):
    # No knobs => default 500 ms; iteration timer keeps the count directly.
    # ceil(500 / 0.25) = 2000 iters, above the min_rep floor.
    monkeypatch.setattr(
        "profiling.profilers.timer._estimate_per_iter_ms", lambda fn: 0.25
    )
    loop_iters: list[int] = []

    class FakeEvent:
        def __init__(self, *, enable_timing: bool) -> None:
            self.n = 0

        def record(self) -> None: ...
        def synchronize(self) -> None: ...

        def elapsed_time(self, end: "FakeEvent") -> float:
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
    monkeypatch.setitem(
        sys.modules, "triton", _fake_triton_recording([], [1.0, 1.0, 1.0])
    )
    monkeypatch.setattr(
        "profiling.profilers.timer._estimate_per_iter_ms", lambda fn: 1.0
    )
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
