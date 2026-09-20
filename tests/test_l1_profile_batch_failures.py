"""A backend that fails every spec must print why, in full."""

from __future__ import annotations

import logging
from collections import Counter

from profiling.db.batch import _log_batch_failures

LONG = (
    "Process 0 terminated with the following error: Traceback "
    + "x" * 600
    + " RuntimeError: the real cause"
)


def test_total_failure_reports_the_whole_reason(caplog) -> None:
    """Every spec failed, so this backend is a hard stop for its caller. The
    240-char cut lands mid-traceback and hides the exception that ended it."""

    with caplog.at_level(logging.ERROR):
        _log_batch_failures("all_reduce_fusion", "flashinfer_trtllm", 30, Counter({LONG: 30}))

    assert "RuntimeError: the real cause" in caplog.text
    assert "..." not in caplog.text.split(LONG[:40])[-1][:700]


def test_partial_failure_still_truncates(caplog) -> None:
    """A few bad specs inside a large batch must not bury the log."""

    with caplog.at_level(logging.WARNING):
        _log_batch_failures("single_gemm", "torch", 100, Counter({LONG: 3}))

    assert "RuntimeError: the real cause" not in caplog.text
    assert "..." in caplog.text
