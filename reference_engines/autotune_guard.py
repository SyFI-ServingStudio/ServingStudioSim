"""Make kernel autotuning a hard error unless the parent allowed it.

A measurement must run the tactics an earlier, deliberate warm-up chose, not
re-derive them: a tactic search costs minutes per process (FlashInfer's NVFP4
MoE search took 8.5 min of a TP4 vLLM start), and a search that runs inside a
measurement picks winners from whatever the measurement happens to run first.

`VIBESIM_AUTOTUNE=allow|forbid` carries the decision. Only the parent sets it,
in two places: `profiling/exec/local.py:autotune_policy` for L1 workers and
`alignment/warm_caches.py` for captures (the policy table is there). Unset
means forbid, so a process nobody decided for cannot tune either.

This module is the in-process backstop. The parents first check, before any
container starts, that the tactic caches a run needs exist (the preflight);
when they say so they also set `VIBESIM_AUTOTUNE_PREFLIGHT` to what they found,
and a measurement that still starts here is reported as a manifest/actual
mismatch: the preflight key is missing something that changes the tuning.

The engine images load this at interpreter start (a `.pth` in site-packages,
so every engine process and its workers have it); L1 workers also call
`install()` themselves, on any image or the host. It hooks, when each module is
first imported:

- FlashInfer `AutoTuner._profile_single_kernel`: the one per-tactic measurement
  of both the legacy autotuner and autotune_v2 (`choose_one`, `rank_tactics`).
  vLLM's `kernel_warmup.flashinfer_autotune` (cache_hit False), SGLang's
  `_flashinfer_autotune`, and the L1 `autotune_cached` runners reach it.
- Triton `Autotuner._bench`: a config benchmark, which Triton runs only when
  neither its in-process nor its disk cache (`TRITON_CACHE_AUTOTUNING=1`, set
  here unless the caller set it) has the key. Configs with a `pre_hook` are
  never disk-cached, so they always need an allowed run.
- Inductor `AlgorithmSelectorCache.autotune` / `benchmark_choices`
  (max-autotune template choice), `CachingAutotuner.benchmark_all_configs`
  (multi-config runtime kernels) and `_coordinate_descent_tuning`; each runs
  only when inductor's own autotune cache misses.

Under `allow` the same hooks time each outermost measurement instead
(`tuned()`), so an L1 jit-fill can log what it tuned and for how long.

The error is a `BaseException`, so engine code that catches `Exception` around a
tactic (FlashInfer skips a failing tactic that way) cannot swallow it, and it is
printed with `MARKER` first so a launcher reading the process's log can tell
it from any other crash.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import sys
import time
from collections.abc import Callable
from types import ModuleType
from typing import Any

POLICY_ENV = "VIBESIM_AUTOTUNE"
ALLOW, FORBID = "allow", "forbid"
#: Set by a parent whose preflight found the caches warm; its value says which.
PREFLIGHT_ENV = "VIBESIM_AUTOTUNE_PREFLIGHT"
MARKER = "VIBESIM_AUTOTUNE_FORBIDDEN"
WARM_HINT = (
    "autotuning is off for this run: warm the caches with "
    "`just ref warm-caches <engine> <profile.yaml>` for a capture, or pass "
    "--allow-autotune to `launcher alignment profile` / `python -m profiling run` "
    "to tune deliberately"
)


class AutotuneForbidden(BaseException):  # noqa: N818 - the name is the contract
    """A tactic measurement ran without VIBESIM_AUTOTUNE=allow."""


class AutotuneForbiddenError(RuntimeError):
    """A parent's view: a run needs tuning its policy forbids (a preflight miss,
    or a worker or engine that stopped with `MARKER`)."""


def forbidden_line(log_text: str | None) -> str | None:
    """The guard's `MARKER` line in a process's output, if it stopped there."""

    for line in (log_text or "").splitlines():
        if MARKER in line:
            return line.strip()
    return None


def allowed() -> bool:
    return os.environ.get(POLICY_ENV) == ALLOW


def forbid(source: str, **detail: Any) -> None:
    """Raise `AutotuneForbidden` for a tactic measurement, unless allowed."""

    if allowed():
        return
    record = json.dumps({"source": source, **detail}, default=str, sort_keys=True)
    preflight = os.environ.get(PREFLIGHT_ENV)
    cause = (
        f"manifest/actual mismatch: the preflight found warm caches ({preflight}) but "
        "this run still tunes, so the preflight key misses an input of this tuning; "
        "add it to the key"
        if preflight
        else WARM_HINT
    )
    message = f"{MARKER} {record} -- {cause}"
    print(message, file=sys.stderr, flush=True)
    raise AutotuneForbidden(message)


# Under `allow`: (source, what) -> [measurements, seconds], outermost calls only.
_TUNED: dict[tuple[str, str], list[float]] = {}
_depth = 0


def tuned(reset: bool = True) -> list[dict[str, Any]]:
    """What this process tuned since the last call, slowest first."""

    rows = [
        {"source": source, "what": what, "measurements": int(n), "seconds": round(t, 3)}
        for (source, what), (n, t) in sorted(_TUNED.items(), key=lambda item: -item[1][1])
    ]
    if reset:
        _TUNED.clear()
    return rows


def _shapes(values: Any, limit: int = 8) -> list[Any]:
    shapes = []
    for value in values or ():
        shape = getattr(value, "shape", None)
        if shape is not None:
            shapes.append(tuple(shape))
        if len(shapes) >= limit:
            break
    return shapes


def _wrap(owner: type, name: str, describe: Callable[..., dict[str, Any]], source: str) -> bool:
    raw = owner.__dict__.get(name)
    if raw is None:
        return False
    is_classmethod = isinstance(raw, classmethod)
    function = raw.__func__ if is_classmethod else raw
    if getattr(function, "__vibesim_autotune_guard__", False):
        return True

    def guarded(*args: Any, **kwargs: Any) -> Any:
        global _depth
        if not allowed():
            forbid(source, **describe(*args, **kwargs))
        if _depth:
            return function(*args, **kwargs)
        _depth += 1
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            _depth -= 1
            try:
                # One row per tuned op and shape, not per candidate tactic.
                detail = {
                    key: value
                    for key, value in describe(*args, **kwargs).items()
                    if key not in ("tactic", "config")
                }
                what = json.dumps(detail, default=str, sort_keys=True)
            except Exception:  # noqa: BLE001 - accounting must not fail a tune
                what = "?"
            entry = _TUNED.setdefault((source, what), [0, 0.0])
            entry[0] += 1
            entry[1] += time.perf_counter() - started

    guarded.__vibesim_autotune_guard__ = True  # type: ignore[attr-defined]
    guarded.__wrapped__ = function  # type: ignore[attr-defined]
    guarded.__name__ = function.__name__
    guarded.__qualname__ = function.__qualname__
    setattr(owner, name, classmethod(guarded) if is_classmethod else guarded)
    return True


def _flashinfer(module: ModuleType) -> list[str]:
    def describe(_self, runner, inputs, tactic, *_args, **_kwargs):
        return {"runner": type(runner).__name__, "tactic": tactic, "shapes": _shapes(inputs)}

    tuner = getattr(module, "AutoTuner", None)
    if tuner is None or not _wrap(
        tuner, "_profile_single_kernel", describe, "flashinfer.AutoTuner"
    ):
        return ["flashinfer AutoTuner._profile_single_kernel"]
    return []


def _triton(module: ModuleType) -> list[str]:
    def describe(self, *_args, config=None, **_kwargs):
        base = getattr(self, "base_fn", None)
        return {
            "kernel": getattr(base, "__qualname__", repr(base)),
            "config": str(config),
            "configs": len(getattr(self, "configs", ()) or ()),
        }

    autotuner = getattr(module, "Autotuner", None)
    if autotuner is None or not _wrap(autotuner, "_bench", describe, "triton.autotune"):
        return ["triton Autotuner._bench"]
    return []


def _inductor_select(module: ModuleType) -> list[str]:
    cache = getattr(module, "AlgorithmSelectorCache", None)
    missing = []

    def describe_autotune(_self, name, *_args, **_kwargs):
        return {"op": name}

    def describe_choices(_cls, choices, *_args, **_kwargs):
        return {"choices": len(choices)}

    if cache is None or not _wrap(cache, "autotune", describe_autotune, "inductor.max_autotune"):
        missing.append("inductor AlgorithmSelectorCache.autotune")
    if cache is None or not _wrap(
        cache, "benchmark_choices", describe_choices, "inductor.max_autotune"
    ):
        missing.append("inductor AlgorithmSelectorCache.benchmark_choices")
    return missing


def _inductor_runtime(module: ModuleType) -> list[str]:
    autotuner = getattr(module, "CachingAutotuner", None)
    missing = []

    def describe(self, *_args, **_kwargs):
        meta = getattr(self, "inductor_meta", None) or {}
        return {"kernel": meta.get("kernel_name"), "configs": len(getattr(self, "configs", ()))}

    for name, source in (
        ("benchmark_all_configs", "inductor.runtime_autotune"),
        ("_coordinate_descent_tuning", "inductor.coordinate_descent"),
    ):
        if autotuner is None or not _wrap(autotuner, name, describe, source):
            missing.append(f"inductor CachingAutotuner.{name}")
    return missing


# Module -> its patch, applied right after the module first executes.
_PATCHES: dict[str, Callable[[ModuleType], list[str]]] = {
    "flashinfer.autotuner.autotuner": _flashinfer,
    "triton.runtime.autotuner": _triton,
    "torch._inductor.select_algorithm": _inductor_select,
    "torch._inductor.runtime.triton_heuristics": _inductor_runtime,
}

# Entry points a patch could not find (the library moved them): a hole in the
# guard, reported on stderr once per process rather than failing an import.
UNHOOKED: list[str] = []


def _patch(name: str, module: ModuleType) -> None:
    try:
        missing = _PATCHES[name](module)
    except Exception as exc:  # a guard bug must not break the engine's import
        missing = [f"{name} ({type(exc).__name__}: {exc})"]
    if missing:
        UNHOOKED.extend(missing)
        print(
            f"[autotune_guard] not hooked, autotuning there is NOT blocked: {missing}",
            file=sys.stderr,
            flush=True,
        )


class _PatchingLoader(importlib.abc.Loader):
    def __init__(self, loader: importlib.abc.Loader, name: str) -> None:
        self._loader = loader
        self._name = name

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return self._loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self._loader.exec_module(module)
        _patch(self._name, module)

    def __getattr__(self, attribute: str) -> Any:
        return getattr(self._loader, attribute)


class _Finder(importlib.abc.MetaPathFinder):
    # The images load a copy of this file from site-packages at interpreter
    # start; an L1 worker then imports the checkout's. One finder is enough.
    vibesim_autotune_guard = True

    def find_spec(self, name: str, path: Any, target: Any = None) -> Any:
        if name not in _PATCHES:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        if spec.loader is not None and not isinstance(spec.loader, _PatchingLoader):
            spec.loader = _PatchingLoader(spec.loader, name)
        return spec


def install() -> None:
    """Hook the autotuners: those already imported now, the rest on import."""

    if any(getattr(finder, "vibesim_autotune_guard", False) for finder in sys.meta_path):
        return
    # Persist Triton's selections, so a warmed key is a disk-cache hit in every
    # later process instead of a benchmark (vLLM sets the same default).
    os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")
    sys.meta_path.insert(0, _Finder())
    for name in _PATCHES:
        module = sys.modules.get(name)
        if module is not None:
            _patch(name, module)
