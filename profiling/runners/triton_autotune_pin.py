"""Pin Triton ``@autotune`` selections at a documented anchor shape.

A Triton autotuner benchmarks its configs the first time it sees a tuning key
and reuses the winner for every later call with that key. When the key leaves
out the axes a profiling grid sweeps (the FLA/KDA kernels key on H/K/BT, not on
the token count), the winner is chosen by whichever row happens to run first,
so rows depend on grid order. With ``TRITON_CACHE_AUTOTUNING=1`` (vLLM sets it)
the winner is also persisted in ``TRITON_CACHE_DIR`` and leaks into every later
process. A row at T=1 is launch-bound, so its winner is close to arbitrary and
was measured 29% slower at the production shape.

``AutotunePin`` removes both dependencies:

- the registry row sets ``worker_env=(("TRITON_CACHE_AUTOTUNING", "0"),)`` so
  no selection is read from or written to disk; ``ensure`` refuses to run if an
  autotuner still persists results;
- before a worker's first row of a given state key (the tuning-relevant config:
  heads, head dim, dtype), ``ensure`` clears the in-memory selections and runs
  the runner's anchor call once, so the anchor shape picks every config;
- the selections of each state key are snapshotted, so a worker that alternates
  state keys restores them rather than re-tuning;
- ``note`` returns ``anchor=<label> configs=<n>:<sha256 prefix>`` for row
  provenance, a digest of every autotuner's selected configs, plus
  ``retuned=...`` if a later call tuned a key the anchor did not reach.

Worker-only: Triton is imported lazily.
"""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Callable, Hashable, Iterable
from typing import Any

_MAX_WRAPPER_DEPTH = 4


def iter_autotuners(module_prefixes: Iterable[str]) -> list[tuple[str, Any]]:
    """Every distinct Triton ``Autotuner`` reachable from loaded modules under
    ``module_prefixes`` (through ``heuristics`` wrappers), sorted by name."""
    from triton.runtime.autotuner import Autotuner

    prefixes = tuple(module_prefixes)
    found: dict[int, tuple[str, Any]] = {}
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith(prefixes):
            continue
        for value in list(vars(module).values()):
            for _ in range(_MAX_WRAPPER_DEPTH):
                if isinstance(value, Autotuner):
                    base = value.base_fn
                    found[id(value)] = (f"{base.__module__}.{base.__qualname__}", value)
                    break
                value = getattr(value, "fn", None)
                if value is None:
                    break
    return sorted(found.values(), key=lambda item: item[0])


def selection_digest(autotuners: list[tuple[str, Any]]) -> tuple[int, str]:
    """(number of selected configs, sha256 prefix over name, key and config)."""
    lines = sorted(
        f"{name}|{key!r}|{config}"
        for name, tuner in autotuners
        for key, config in tuner.cache.items()
    )
    digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()[:12]
    return len(lines), digest


class AutotunePin:
    """Per-worker-process pin of the autotune selections for one runner."""

    def __init__(self, module_prefixes: tuple[str, ...]) -> None:
        self._module_prefixes = module_prefixes
        self._active: Hashable | None = None
        self._snapshots: dict[Hashable, dict[int, dict]] = {}
        self._notes: dict[Hashable, str] = {}

    def ensure(self, state_key: Hashable, anchor_label: str, tune: Callable[[], None]) -> str:
        """Make ``state_key``'s anchor-tuned selections the live ones; return its note."""
        if state_key == self._active:
            return self._notes[state_key]
        autotuners = iter_autotuners(self._module_prefixes)
        persisting = [name for name, tuner in autotuners if tuner.cache_results]
        if persisting:
            raise RuntimeError(
                "Triton autotune persistence is on for "
                f"{persisting[:3]}; the registry row must set worker_env "
                "TRITON_CACHE_AUTOTUNING=0 so selections cannot leak between processes"
            )
        if self._active is not None:
            self._snapshots[self._active] = {
                id(tuner): dict(tuner.cache) for _name, tuner in autotuners
            }
        saved = self._snapshots.get(state_key)
        for _name, tuner in autotuners:
            tuner.cache.clear()
            if saved is not None:
                tuner.cache.update(saved.get(id(tuner), {}))
        if saved is None:
            tune()
            # The anchor call imports nothing new in practice, but re-scan so a
            # lazily imported kernel module is still covered by the digest.
            count, digest = selection_digest(iter_autotuners(self._module_prefixes))
            self._notes[state_key] = f"anchor={anchor_label} configs={count}:{digest}"
        self._active = state_key
        return self._notes[state_key]

    def note(self, state_key: Hashable) -> str | None:
        """The anchor note of ``state_key``. For the live key, a selection made
        after the anchor (a tuning key the anchor did not reach, so some row
        tuned it at its own shape) is flagged with ``retuned=<n>:<digest>``."""
        note = self._notes.get(state_key)
        if note is None or state_key != self._active:
            return note
        live = "configs=%d:%s" % selection_digest(iter_autotuners(self._module_prefixes))
        return note if note.endswith(live) else f"{note} retuned={live.removeprefix('configs=')}"


__all__ = ["AutotunePin", "iter_autotuners", "selection_digest"]
