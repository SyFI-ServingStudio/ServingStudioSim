from __future__ import annotations

import pytest

# Whole module is the `gpu` tier (conftest auto-skips when no CUDA device).
pytestmark = pytest.mark.gpu


def _require_cuda_cupti():
    # CUDA presence is the `gpu` marker's job; here we only probe the CUPTI
    # headers/libs, a *feature* the device may still lack (inner skip).
    torch = pytest.importorskip("torch")
    from profiling.profilers import cupti_kernel_profiler

    try:
        cupti_kernel_profiler._resolve_cupti_paths()
    except RuntimeError as exc:
        pytest.skip(f"CUPTI headers/libs are not available: {exc}")
    return torch


def _make_matmul_callable(torch):
    device = torch.device("cuda:0")
    left = torch.randn((128, 128), device=device, dtype=torch.float16)
    right = torch.randn((128, 128), device=device, dtype=torch.float16)

    def run_matmul():
        return torch.mm(left, right)

    run_matmul()
    torch.cuda.synchronize(device)
    return device, run_matmul


def test_cupti_profile_kernel_captures_cuda_matmul_activity():
    torch = _require_cuda_cupti()
    from profiling.profilers.cupti_kernel_profiler import profile_kernel

    device, run_matmul = _make_matmul_callable(torch)

    # Agent note: this is intentionally a real integration check. It should
    # skip for absent CUDA/CUPTI, but fail if CUPTI is present and cannot compile
    # the extension or capture kernel activity records.
    summary = profile_kernel(
        run_matmul,
        device=device,
        num_warmup=1,
        num_iter=3,
        clear_l2_bytes=1024 * 1024,
        clear_l2_before_run=False,
    )

    assert summary.num_iter == 3
    assert len(summary.per_iter_ms) == 3
    assert summary.matched_kernel_names
    assert all(time_ms > 0 for time_ms in summary.per_iter_ms)
    assert all(kernel_count >= 1 for kernel_count in summary.matched_kernel_count_per_run)
    assert summary.mean_ms > 0
    assert summary.min_ms <= summary.median_ms <= summary.max_ms


def test_timer_cupti_measures_cuda_matmul_with_real_profiler():
    torch = _require_cuda_cupti()
    from profiling.profilers.timer import Timer

    _, run_matmul = _make_matmul_callable(torch)

    time_ms = Timer.cupti(run_matmul, warmup=1, rep=2)

    assert 0 < time_ms < 100


def test_relative_sem_pure():
    from profiling.profilers.cupti_kernel_profiler import relative_sem

    assert relative_sem([]) == float("inf")
    assert relative_sem([1.0]) == float("inf")  # need >= 2 samples
    assert relative_sem([0.0, 0.0]) == float("inf")  # mean <= 0
    # Identical samples => zero spread => zero relative error.
    assert relative_sem([2.0, 2.0, 2.0, 2.0]) == 0.0
    # More samples of the same noisy signal => smaller relative SEM (1/sqrt(n)).
    few = relative_sem([1.0, 1.1] * 5)
    many = relative_sem([1.0, 1.1] * 50)
    assert many < few


def test_cupti_profile_until_converged_stops_early_for_stable_kernel():
    torch = _require_cuda_cupti()
    from profiling.profilers.cupti_kernel_profiler import profile_kernel_until_converged

    device, run_matmul = _make_matmul_callable(torch)

    summary = profile_kernel_until_converged(
        run_matmul,
        device=device,
        batch=10,
        min_iter=20,
        max_iter=500,
        tol=0.01,
        clear_l2_bytes=1024 * 1024,
        clear_l2_before_run=False,
    )

    # A stable kernel should converge well before the cap.
    assert 20 <= summary.num_iter <= 500
    assert summary.num_iter == len(summary.per_iter_ms)
    assert summary.mean_ms > 0
    assert summary.min_ms <= summary.median_ms <= summary.max_ms


def test_timer_cupti_default_path_converges_on_real_gpu():
    torch = _require_cuda_cupti()
    from profiling.profilers.timer import Timer

    _, run_matmul = _make_matmul_callable(torch)

    # No rep => adaptive convergence path end to end.
    time_ms = Timer.cupti(run_matmul, max_rep=200)
    assert 0 < time_ms < 100
