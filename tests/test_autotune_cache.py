"""The FlashInfer tactic cache has to survive the worker process."""

from __future__ import annotations

import contextlib
from pathlib import Path

from profiling.runners.autotune_cache import CACHE_DIR_ENV, autotune_cache_path, autotune_cached


def _recording_autotune(seen: list[dict], *, accepts_cache: bool):
    if accepts_cache:

        @contextlib.contextmanager
        def autotune(tune_mode: bool = True, cache: str | None = None):
            seen.append({"cache": cache})
            yield

    else:

        @contextlib.contextmanager
        def autotune(tune_mode: bool = True):
            seen.append({"cache": None})
            yield

    return autotune


def test_cache_path_separates_devices_and_library_versions(tmp_path: Path, monkeypatch) -> None:
    """One file per environment, because FlashInfer's own mismatch check is
    all-or-nothing: it ignores a foreign cache AND refuses to write to it, so a
    host serving two card types through one path would tune forever."""

    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))
    path = autotune_cache_path("nvfp4_fused_moe.vllm")

    assert path is not None
    assert path.parent == tmp_path
    assert path.name.startswith("nvfp4-fused-moe-vllm.")
    assert path.name.endswith(".json")
    assert ".fi" in path.name


def test_cache_path_is_optional(tmp_path: Path, monkeypatch) -> None:
    """An empty setting disables persistence. A tactic cache is an optimisation,
    and failing to place one must never fail a measurement."""

    monkeypatch.setenv(CACHE_DIR_ENV, "")
    assert autotune_cache_path("nvfp4_fused_moe.vllm") is None


def test_cached_autotune_hands_the_path_to_flashinfer(tmp_path: Path, monkeypatch) -> None:
    seen: list[dict] = []
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))

    with autotune_cached(_recording_autotune(seen, accepts_cache=True), "nvfp4_fused_moe.vllm"):
        pass

    assert len(seen) == 1
    assert seen[0]["cache"] is not None
    assert seen[0]["cache"].startswith(str(tmp_path))


def test_cached_autotune_falls_back_when_flashinfer_has_no_cache_parameter(
    tmp_path: Path, monkeypatch
) -> None:
    """Older FlashInfer tunes in memory, as before -- not an error."""

    seen: list[dict] = []
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))

    with autotune_cached(_recording_autotune(seen, accepts_cache=False), "nvfp4_fused_moe.vllm"):
        pass

    assert seen == [{"cache": None}]


def test_cached_autotune_does_not_swallow_the_body(tmp_path: Path, monkeypatch) -> None:
    """A TypeError from the kernel under test must propagate.

    The version probe is a signature check for exactly this reason: catching
    TypeError around the context would make an unsupported argument and a
    broken kernel call look identical, and silently re-run the measurement.
    """

    seen: list[dict] = []
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))

    try:
        with autotune_cached(_recording_autotune(seen, accepts_cache=True), "k"):
            raise TypeError("kernel call is wrong")
    except TypeError as error:
        assert str(error) == "kernel call is wrong"
    else:  # pragma: no cover - the assertion is the point
        raise AssertionError("TypeError was swallowed")

    assert len(seen) == 1


def test_each_file_receives_only_its_own_configs_tactics(tmp_path: Path, monkeypatch) -> None:
    """One worker profiles several configurations. FlashInfer saves the whole
    process-wide tactic dict under hash-free keys into whichever file is open, so
    the second configuration's file must not inherit the first one's entries."""

    import json
    import sys
    import threading
    import types

    class AutoTuner:
        instance = None

        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.profiling_cache: dict[str, str] = {}

        @classmethod
        def get(cls) -> AutoTuner:
            cls.instance = cls.instance or cls()
            return cls.instance

    module = types.ModuleType("fake_flashinfer_autotuner")
    module.AutoTuner = AutoTuner
    monkeypatch.setitem(sys.modules, module.__name__, module)

    @contextlib.contextmanager
    def autotune(tune_mode: bool = True, cache: str | None = None):
        yield
        Path(cache).write_text(json.dumps(AutoTuner.get().profiling_cache))

    autotune.__module__ = module.__name__
    monkeypatch.setenv(CACHE_DIR_ENV, str(tmp_path))

    for label, tactic in (("cfg-a", "a"), ("cfg-b", "b")):
        with autotune_cached(autotune, label):
            AutoTuner.get().profiling_cache[f"shape-{label}"] = tactic
    saved = {
        label: json.loads(autotune_cache_path(label).read_text()) for label in ("cfg-a", "cfg-b")
    }
    assert saved == {"cfg-a": {"shape-cfg-a": "a"}, "cfg-b": {"shape-cfg-b": "b"}}
