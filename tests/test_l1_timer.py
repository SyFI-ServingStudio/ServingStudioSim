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


def test_local_cupti_profiler_points_at_csrc_extension():
    from profiling.profilers import cupti_kernel_profiler

    assert cupti_kernel_profiler.EXT_SOURCE.name == "cupti_activity_profiler.cpp"
    assert cupti_kernel_profiler.EXT_SOURCE.exists()
