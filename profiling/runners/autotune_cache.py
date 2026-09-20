"""A FlashInfer tactic cache that outlives the worker process.

``flashinfer.autotuner.AutoTuner`` keeps its results in a per-process dict, so
every worker re-runs the whole tactic search from scratch. That is not a small
tax: profiling eight nvfp4 fused-MoE shapes spent 166 s tuning and 5.7 s
measuring, so 97% of the submission went into re-deriving tactics an earlier
process had already derived. A full table re-profile pays it per shape, per
process, forever.

FlashInfer >= 0.6.12 can persist that dict: ``autotune(cache=path)`` loads the
file on entry and writes it back on exit, and the write is already atomic
(temp file plus ``os.replace``) and already merges what it loaded. It also
stamps the file with the environment it was tuned in and refuses to use a
mismatched one, so a stale cache degrades to a miss rather than to a wrong
tactic.

The path lives under ``$HOME``, which is the one directory that persists on
both sides of the container boundary: a host worker gets the real home, and a
container worker gets ``HOME=/cache/home``, bind-mounted from the host cache
dir by ``_container_worker_command``. Nothing new has to be mounted.
"""

from __future__ import annotations

import contextlib
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

CACHE_DIR_ENV = "VIBESIM_AUTOTUNE_CACHE_DIR"


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower() or "unknown"


def autotune_cache_path(kernel: str) -> Path | None:
    """File holding the persisted tactics for ``kernel``, or ``None``.

    Keyed by device and FlashInfer version on top of the kernel name. The
    library's own metadata check would catch a mismatch anyway, but it reacts
    by ignoring the whole file and declining to save -- so a host serving two
    card types through one path would tune forever and cache nothing. Separate
    files let each environment keep its own.

    ``None`` disables persistence, which is what an unwritable or unset home
    should do: a tactic cache is an optimisation, and failing to place one is
    not a reason to fail a measurement.
    """

    configured = os.environ.get(CACHE_DIR_ENV)
    if configured is not None and not configured.strip():
        return None
    base = Path(configured) if configured else Path.home() / ".cache" / "vibesim-autotune"

    device, version = "unknown", "unknown"
    with contextlib.suppress(Exception):
        import torch

        device = torch.cuda.get_device_name()
    with contextlib.suppress(Exception):
        import flashinfer

        version = flashinfer.__version__
    try:
        base.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    return base / f"{_slug(kernel)}.{_slug(device)}.fi{_slug(version)}.json"


@contextlib.contextmanager
def autotune_cached(autotune: Any, kernel: str) -> Iterator[None]:
    """``autotune()`` that starts from, and writes back to, the shared cache.

    ``autotune`` is passed in rather than imported here so the import stays
    inside the runner's own try block, where a missing FlashInfer is already
    translated into ``ProfilerNotImplemented``.

    Concurrent workers can still race -- one that loaded before another saved
    will overwrite it -- but the loser is a cache miss on the next run, not a
    wrong tactic, and the file converges as runs accumulate.
    """

    path = autotune_cache_path(kernel) if _supports_cache(autotune) else None
    with autotune() if path is None else autotune(cache=str(path)):
        yield


def _supports_cache(autotune: Any) -> bool:
    """Does this FlashInfer's ``autotune`` take a ``cache=`` path?

    Checked by signature rather than by catching ``TypeError`` around the
    context: the body runs inside that context, so a ``TypeError`` from the
    kernel under test would be indistinguishable from an unsupported argument
    and would silently re-run the whole measurement.
    """

    import inspect

    with contextlib.suppress(TypeError, ValueError):
        return "cache" in inspect.signature(autotune).parameters
    return False
