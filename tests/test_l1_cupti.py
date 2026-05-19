from __future__ import annotations

import pytest


def _require_cuda_cupti():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

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
