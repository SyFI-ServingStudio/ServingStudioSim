"""Single-spec → list runner adapter.

The L1 runner contract is ``run(kwargs_list: list[dict]) -> list[RunnerResult]``
(1:1, in order). Most runners are single-GPU compute kernels whose only expensive
setup — the worker subprocess + CUDA context — is already amortized once per chunk
by the execution backend, so their per-spec call is a cheap in-process launch.
``batched`` wraps such a single-spec ``profile_X(**kwargs) -> Metrics`` function
into the list contract without touching its body: it loops the specs, capturing
per-item errors so one bad shape doesn't sink the rest (the same resilience the
worker loop used to provide). Multi-GPU comm runners, whose rank-group launcher is
expensive per call, instead implement the list contract natively to spawn once.

Loaded only inside the worker subprocess (via ``KernelProfilerSpec.load_list_runner``),
so torch is imported lazily to preserve the no-eager-torch-in-main-process invariant.
"""

from __future__ import annotations

from collections.abc import Callable

from profiling.runners.metrics import Metrics, RunnerResult

SingleSpecFn = Callable[..., Metrics]
ListRunner = Callable[[list[dict]], list[RunnerResult]]


def batched(single_fn: SingleSpecFn) -> ListRunner:
    """Adapt a single-spec runner to the ``list[dict] -> list[RunnerResult]`` contract.

    Each item is run independently; a failure is captured as a ``RunnerResult``
    error (with the CUDA cache emptied) rather than aborting the batch. Output is
    exactly one result per input, in order.
    """

    def run(kwargs_list: list[dict]) -> list[RunnerResult]:
        results: list[RunnerResult] = []
        for kwargs in kwargs_list:
            try:
                results.append(RunnerResult(metrics=single_fn(**kwargs)))
            except Exception as exc:  # noqa: BLE001 — capture per-item, keep the batch alive
                _empty_cuda_cache()
                results.append(RunnerResult(error=str(exc)))
        return results

    return run


def _empty_cuda_cache() -> None:
    """Best-effort CUDA cache flush after a failed launch (import torch lazily)."""
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
