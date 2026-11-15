"""TorchMpLauncher hands back results of any size and surfaces a rank's error.

CPU only: two gloo ranks, no GPU. The result queue is an OS pipe; a result
larger than its buffer (64 KiB on Linux) used to deadlock a parent that joined
the ranks before reading it.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

import pytest

from profiling.runners.comm._launcher import TorchMpLauncher
from profiling.runners.exceptions import KernelLaunchFailed

pytest.importorskip("torch")

# 2720 metrics dicts pickle to about 130 KB, twice the pipe buffer; the
# 2720-spec chunk that hung a profiling fill had that many results.
_LARGE = 2720


def _rows(rank: int, world_size: int, n: int) -> list[dict] | None:
    del world_size
    if rank != 0:
        return None
    return [
        {"time_ms": 0.001 * i, "tflops": 1.0 + i, "memory_bandwidth_gbps": 2.0 + i, "energy_j": 0.0}
        for i in range(n)
    ]


def _within(seconds: float, call: Callable[[], Any]) -> Any:
    """``call()``, failing the test instead of hanging the suite if it blocks."""
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = call()
        except BaseException as exc:  # re-raised on the test thread
            outcome["error"] = exc

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), f"the launcher did not return within {seconds:.0f} s"
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def _fail_on(rank: int, world_size: int, bad_rank: int) -> None:
    del world_size
    if rank == bad_rank:
        raise RuntimeError(f"rank {rank} failed on purpose")


def test_a_result_larger_than_the_pipe_buffer_comes_back() -> None:
    rows = _within(90, lambda: TorchMpLauncher(2, backend="gloo").run(_rows, n=_LARGE))
    assert len(rows) == _LARGE
    assert rows[-1]["tflops"] == 1.0 + _LARGE - 1


@pytest.mark.parametrize("bad_rank", [0, 1])
def test_a_failing_rank_raises_instead_of_hanging(bad_rank: int) -> None:
    with pytest.raises(KernelLaunchFailed, match="failed on purpose"):
        _within(90, lambda: TorchMpLauncher(2, backend="gloo").run(_fail_on, bad_rank=bad_rank))
