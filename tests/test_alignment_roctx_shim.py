"""CPU test for the ROCm roctx iteration shim (``alignment/profiler/roctx_shim``).

The shim is the AMD analog of the vLLM NVTX fork: it emits
``vllm_iteration(N): forward`` roctx ranges and the ``VibeSimAlignmentIteration``
boundary marker. This test proves two things without a GPU or a real roctx stack:

1. the labels it emits match the parser's ``ITER_RE`` (contract parity with the
   nsys/rocpd side) and the counter/record behaviour is correct;
2. the regions it would record, written into a synthetic rocpd and run through
   the *real* ``alignment/rocpd`` ownership join, are recognized as iteration
   markers and own the dispatches launched inside them — i.e. a capture driven by
   this shim feeds the existing offline producer unchanged.

Only the roctx backend is replaced (by the in-module recording backend); the join
and parse code are the real thing.
"""

from __future__ import annotations

import importlib
import json
import sqlite3
import tomllib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from alignment.nsys.parse import ITER_RE
from alignment.profiler import roctx_shim
from alignment.profiler.roctx_shim import (
    ITERATION_RECORD_TAG,
    ROCTX_PLUGIN_ENTRY_POINT_NAME,
    IterationAnnotator,
    RecordingBackend,
    install_vllm_roctx_shim,
    iteration_label,
    iteration_record_marker,
    roctx_scopes_enabled,
    select_backend,
)
from alignment.rocpd.evidence import build_ranges_from_rocpd

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

_GUID = "0000b1c7_c35b_735b_96a7_f0a02ff013cc"
_PID = 4242
_TID = 55
_AGENT = 1


def test_emitted_label_matches_parser_regex():
    label = iteration_label(3, "forward")
    assert label == "vllm_iteration(3): forward"
    match = ITER_RE.match(label)
    assert match is not None
    assert (int(match.group(1)), match.group(2)) == (3, "forward")
    # Every contract phase produces an ITER_RE-matching label.
    for phase in roctx_shim.PHASES:
        assert ITER_RE.match(iteration_label(0, phase)) is not None


def test_sglang_prefix_also_matches():
    assert ITER_RE.match(iteration_label(7, "forward", prefix="sglang")) is not None


def test_iteration_record_marker_is_tagged_json():
    marker = iteration_record_marker({"total_num_scheduled_tokens": 12}, iteration=5)
    tag, _, payload = marker.partition(" ")
    assert tag == ITERATION_RECORD_TAG
    body = json.loads(payload)
    assert body["iteration"] == 5
    assert body["total_num_scheduled_tokens"] == 12
    assert body["schema_version"] == roctx_shim.ITERATION_RECORD_SCHEMA_VERSION


def test_annotator_counts_and_emits_record_before_range():
    backend = RecordingBackend()
    annotator = IterationAnnotator(backend, enabled=True)
    with annotator.forward(record={"total_num_scheduled_tokens": 4}) as n0:
        pass
    with annotator.forward(record={"total_num_scheduled_tokens": 1}) as n1:
        pass
    assert (n0, n1) == (0, 1)
    ops = [(e.op, e.message) for e in backend.events]
    # record mark precedes the forward push, pop closes it (NVIDIA ordering).
    assert ops[0][0] == "mark" and ops[0][1].startswith(ITERATION_RECORD_TAG)
    assert ops[1] == ("push", "vllm_iteration(0): forward")
    assert ops[2][0] == "pop"
    assert ops[4] == ("push", "vllm_iteration(1): forward")


def test_registered_plugin_entry_point_resolves_to_installer():
    """The ``vllm.general_plugins`` entry point points at a real importable callable.

    This is the mechanism vLLM actually uses to run the shim inside a serving
    worker (the AMD analog of baking the NVTX scopes into the fork). The test
    reads the declared ``module:attr`` from ``pyproject.toml``, imports it, and
    asserts it is exactly :func:`install_vllm_roctx_shim` — so the registration
    cannot drift from the callable or name a dead path.
    """
    data = tomllib.loads(_PYPROJECT.read_text())
    group = data["project"]["entry-points"]["vllm.general_plugins"]
    assert ROCTX_PLUGIN_ENTRY_POINT_NAME in group
    module_name, _, attr = group[ROCTX_PLUGIN_ENTRY_POINT_NAME].partition(":")
    resolved = getattr(importlib.import_module(module_name), attr)
    assert resolved is install_vllm_roctx_shim


def test_plugin_is_a_noop_when_flag_unset():
    """Called as vLLM would call it, the plugin patches nothing unless gated on.

    vLLM invokes the loaded entry point with no arguments. With the env gate off
    the shim must return ``False`` and touch neither vLLM (not importable here)
    nor a GPU, so a non-timing server launch is never silently instrumented.
    """
    assert install_vllm_roctx_shim(environ={}) is False


def test_disabled_annotator_is_a_noop():
    backend = RecordingBackend()
    sentinel = roctx_shim.RecordingSentinelLauncher()
    annotator = IterationAnnotator(backend, enabled=False, sentinel=sentinel)
    with annotator.forward(record={"x": 1}):
        pass
    annotator.emit_iteration_record({"x": 1}, iteration=0)
    assert backend.events == []
    # A disabled annotator launches no sentinel either.
    assert sentinel.iterations == []


def test_annotator_launches_one_sentinel_per_forward_in_order():
    backend = RecordingBackend()
    sentinel = roctx_shim.RecordingSentinelLauncher()
    annotator = IterationAnnotator(backend, enabled=True, sentinel=sentinel)
    for _ in range(3):
        with annotator.forward(record={"total_num_scheduled_tokens": 2}):
            pass
    # One sentinel per forward, carrying the forward's iteration index in order.
    assert sentinel.iterations == [0, 1, 2]


def test_select_sentinel_launcher_record_and_null():
    name, launcher = roctx_shim.select_sentinel_launcher("record")
    assert name == "record" and isinstance(launcher, roctx_shim.RecordingSentinelLauncher)
    name, launcher = roctx_shim.select_sentinel_launcher("null")
    assert name == "null" and isinstance(launcher, roctx_shim.NullSentinelLauncher)
    # Unknown names are rejected like the roctx backend selector.
    with pytest.raises(ValueError, match="unknown sentinel launcher"):
        roctx_shim.select_sentinel_launcher("bogus")


def test_recorded_sentinels_feed_the_real_sentinel_reconstruction(tmp_path):
    """End-to-end: the ordinals the shim asks the launcher to mark, written as
    sentinel ``kernel_dispatch`` rows (grid-y encoding), reconstruct iteration
    ranges through the real ``alignment/rocpd`` sentinel path — the roctx-free
    Option-B flow a real ``--kernel-trace`` capture produces.
    """
    sentinel = roctx_shim.RecordingSentinelLauncher()
    annotator = IterationAnnotator(RecordingBackend(), enabled=True, sentinel=sentinel)
    for _ in range(2):
        with annotator.forward(record={"total_num_scheduled_tokens": 2}):
            pass

    db = tmp_path / "sentinel.db"
    conn = sqlite3.connect(db)
    conn.execute(f"CREATE TABLE rocpd_info_kernel_symbol_{_GUID} "
                 "(id INTEGER PRIMARY KEY, kernel_name TEXT, display_name TEXT)")
    conn.execute(
        f"CREATE TABLE rocpd_kernel_dispatch_{_GUID} "
        "(id INTEGER PRIMARY KEY, pid INTEGER, agent_id INTEGER, kernel_id INTEGER, "
        "dispatch_id INTEGER, stream_id INTEGER, start BIGINT, end BIGINT, "
        "grid_size_x INTEGER, grid_size_y INTEGER, grid_size_z INTEGER)"
    )
    rows = []
    symbols: dict[str, int] = {}

    def add(start, end, name, grid_y):
        if name not in symbols:
            symbols[name] = len(symbols) + 1
            conn.execute(
                f"INSERT INTO rocpd_info_kernel_symbol_{_GUID} (id, kernel_name, display_name) "
                "VALUES (?, ?, ?)", (symbols[name], f"_m_{name}", name))
        rows.append((start, end, symbols[name], grid_y))

    # Two sentinels (one per recorded forward) with a model kernel inside each.
    base = 1_000
    for ordinal, iteration in enumerate(sentinel.iterations):
        s = base + ordinal * 10_000
        grid_y = iteration + roctx_shim.SENTINEL_GRID_Y_OFFSET
        add(s, s + 20, roctx_shim.SENTINEL_KERNEL_NAME, grid_y)
        add(s + 100, s + 500, "hipblaslt_gemm_f16", 1)
    for i, (start, end, kid, grid_y) in enumerate(rows, start=1):
        conn.execute(
            f"INSERT INTO rocpd_kernel_dispatch_{_GUID} "
            "(id, pid, agent_id, kernel_id, dispatch_id, stream_id, start, end, "
            "grid_size_x, grid_size_y, grid_size_z) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (i, _PID, _AGENT, kid, i, 0, start, end, 64, grid_y, 1))
    conn.commit()
    conn.close()

    ranges = build_ranges_from_rocpd(str(db))
    by_iter = {item.iteration: item for item in ranges}
    assert set(by_iter) == {0, 1}
    assert all(item.kernel_count == 1 for item in ranges)
    assert {e.name for item in ranges for e in item.kernel_events} == {"hipblaslt_gemm_f16"}


def test_env_gate_and_select_backend_record():
    assert roctx_scopes_enabled({}) is False
    assert roctx_scopes_enabled({roctx_shim.ROCTX_SCOPES_ENV: "1"}) is True
    name, backend = select_backend("record")
    assert name == "record"
    assert isinstance(backend, RecordingBackend)


def _fake_cdll(present, attempts):
    """A ``ctypes.CDLL`` stand-in that loads only sonames in ``present``.

    Records every attempted soname in ``attempts`` (so the try-order is
    observable) and returns a ``MagicMock`` lib for a present one — the backend
    sets ``argtypes``/``restype`` on its roctx symbols, which a MagicMock accepts.
    """

    def fake(name):
        attempts.append(name)
        if name in present:
            return MagicMock()
        raise OSError(f"no such shared object: {name}")

    return fake


def test_roctracer_backend_prefers_sdk_over_legacy(monkeypatch):
    """With both present, the rocprofiler-sdk roctx library is chosen first.

    On rocprofv3-1.3.2 ``--marker-trace`` records roctx ranges only from
    ``librocprofiler-sdk-roctx``; legacy ``libroctx64`` is the fallback. The
    selection must therefore try the SDK soname first and stop there.
    """
    from alignment.profiler.roctx_shim import RoctracerBackend

    attempts: list[str] = []
    monkeypatch.setattr(
        roctx_shim.ctypes,
        "CDLL",
        _fake_cdll({"librocprofiler-sdk-roctx.so.1", "libroctx64.so"}, attempts),
    )
    backend = RoctracerBackend()
    assert backend.soname == "librocprofiler-sdk-roctx.so.1"
    # SDK soname is first in the order, so the legacy one is never even tried.
    assert attempts == ["librocprofiler-sdk-roctx.so.1"]


def test_roctracer_backend_falls_back_to_legacy_libroctx64(monkeypatch):
    """When only legacy ``libroctx64`` is present it is used (old roctracer stack).

    The SDK sonames are tried and fail first, proving they are preferred, then the
    first resolvable legacy soname wins.
    """
    from alignment.profiler.roctx_shim import RoctracerBackend

    attempts: list[str] = []
    monkeypatch.setattr(
        roctx_shim.ctypes, "CDLL", _fake_cdll({"libroctx64.so"}, attempts)
    )
    backend = RoctracerBackend()
    assert backend.soname == "libroctx64.so"
    # The SDK sonames were attempted (and failed) before the legacy fallback.
    assert attempts.index("librocprofiler-sdk-roctx.so.1") < attempts.index("libroctx64.so")


def test_roctracer_backend_raises_when_no_roctx_lib(monkeypatch):
    from alignment.profiler.roctx_shim import RoctracerBackend

    attempts: list[str] = []
    monkeypatch.setattr(roctx_shim.ctypes, "CDLL", _fake_cdll(set(), attempts))
    with pytest.raises(OSError, match="roctx shared object"):
        RoctracerBackend()
    # Every candidate was tried before giving up.
    assert "librocprofiler-sdk-roctx.so.1" in attempts and "libroctx64.so" in attempts


def test_roctracer_backend_explicit_soname_forces_one(monkeypatch):
    """An explicit soname bypasses the preference order and loads exactly that."""
    from alignment.profiler.roctx_shim import RoctracerBackend

    attempts: list[str] = []
    monkeypatch.setattr(
        roctx_shim.ctypes, "CDLL", _fake_cdll({"libroctx64.so.1"}, attempts)
    )
    backend = RoctracerBackend(soname="libroctx64.so.1")
    assert backend.soname == "libroctx64.so.1"
    assert attempts == ["libroctx64.so.1"]


def _fake_cdll_with_mode(present, calls):
    """A ``ctypes.CDLL`` stand-in that records ``(name, mode)`` of each attempt.

    Loads only sonames in ``present`` (returns a ``MagicMock``), else raises
    ``OSError`` — so the preloader's gating, try-order, and RTLD_GLOBAL mode are
    observable without a real roctx stack.
    """

    def fake(name, mode=0):
        calls.append((name, mode))
        if name in present:
            return MagicMock()
        raise OSError(f"no such shared object: {name}")

    return fake


@pytest.fixture
def _reset_preload():
    """Clear the preload module globals before and after a test uses them."""
    from alignment.profiler import _roctx_preload

    saved = (_roctx_preload._PRELOADED_HANDLE, _roctx_preload._PRELOADED_SONAME)
    _roctx_preload._PRELOADED_HANDLE = None
    _roctx_preload._PRELOADED_SONAME = None
    yield _roctx_preload
    _roctx_preload._PRELOADED_HANDLE, _roctx_preload._PRELOADED_SONAME = saved


def test_preload_is_a_noop_when_flag_unset(monkeypatch, _reset_preload):
    """Gated off: the preloader loads nothing and records nothing."""
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(_reset_preload.ctypes, "CDLL", _fake_cdll_with_mode(set(), calls))
    env: dict[str, str] = {}
    assert _reset_preload.preload_sdk_roctx(environ=env) is None
    assert calls == []  # CDLL never touched when the flag is off
    assert _reset_preload.preloaded_handle() is None
    assert _reset_preload.ROCTX_SONAME_RESOLVED_ENV not in env


def test_preload_loads_sdk_roctx_rtld_global_when_flag_set(monkeypatch, _reset_preload):
    """Gated on: the SDK soname is loaded RTLD_GLOBAL and recorded for the shim."""
    import ctypes as _ctypes

    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        _reset_preload.ctypes,
        "CDLL",
        _fake_cdll_with_mode({"librocprofiler-sdk-roctx.so.1"}, calls),
    )
    env = {roctx_shim.ROCTX_SCOPES_ENV: "1"}
    resolved = _reset_preload.preload_sdk_roctx(environ=env)
    assert resolved == "librocprofiler-sdk-roctx.so.1"
    # Loaded with RTLD_GLOBAL so its roctx* symbols win global resolution.
    assert calls == [("librocprofiler-sdk-roctx.so.1", _ctypes.RTLD_GLOBAL)]
    assert _reset_preload.preloaded_handle() is not None
    assert env[_reset_preload.ROCTX_SONAME_RESOLVED_ENV] == "librocprofiler-sdk-roctx.so.1"


def test_preload_tolerates_absent_library(monkeypatch, _reset_preload):
    """Gated on but no SDK lib present: logs + returns None, never raises."""
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(
        _reset_preload.ctypes, "CDLL", _fake_cdll_with_mode(set(), calls)
    )
    env = {roctx_shim.ROCTX_SCOPES_ENV: "1"}
    assert _reset_preload.preload_sdk_roctx(environ=env) is None
    # Every SDK candidate was tried before giving up, and no crash.
    assert [name for name, _ in calls] == list(_reset_preload.SDK_SONAMES)
    assert _reset_preload.preloaded_handle() is None


def test_roctracer_backend_prefers_preloaded_sdk_handle(monkeypatch, _reset_preload):
    """The shim reuses the preloaded RTLD_GLOBAL SDK handle instead of dlopening again.

    This is what makes the shim's ``roctx*`` calls resolve to the SDK library
    rocprofv3's MARKER service registered against, not torch's legacy libroctx64.
    """
    from alignment.profiler.roctx_shim import RoctracerBackend

    sentinel = MagicMock()
    _reset_preload._PRELOADED_HANDLE = sentinel
    _reset_preload._PRELOADED_SONAME = "librocprofiler-sdk-roctx.so.1"

    # If the backend tried to dlopen itself, this would record a call and fail.
    attempts: list[str] = []
    monkeypatch.setattr(roctx_shim.ctypes, "CDLL", _fake_cdll(set(), attempts))

    backend = RoctracerBackend()
    assert backend.soname == "librocprofiler-sdk-roctx.so.1"
    assert backend._lib is sentinel
    assert attempts == []  # the resident handle was reused, no fresh dlopen


def _write_rocpd_from_regions(path, regions, dispatches):
    """Write a synthetic rocpd db from ``(label, start, end)`` regions + dispatches.

    ``dispatches`` is ``(start, end, name)``. Mirrors the fixture schema used by
    ``tests/test_alignment_rocpd_parse.py`` so the real readers resolve it.
    """
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE rocpd_string_{_GUID} (id INTEGER PRIMARY KEY, string TEXT)")
    conn.execute(
        f"CREATE TABLE rocpd_info_kernel_symbol_{_GUID} "
        "(id INTEGER PRIMARY KEY, kernel_name TEXT, display_name TEXT)"
    )
    conn.execute(
        f"CREATE TABLE rocpd_kernel_dispatch_{_GUID} "
        "(id INTEGER PRIMARY KEY, pid INTEGER, tid INTEGER, agent_id INTEGER, "
        "kernel_id INTEGER, dispatch_id INTEGER, queue_id INTEGER, stream_id INTEGER, "
        "start BIGINT, end BIGINT, region_name_id INTEGER)"
    )
    conn.execute(
        f"CREATE TABLE rocpd_region_{_GUID} "
        "(id INTEGER PRIMARY KEY, pid INTEGER, tid INTEGER, start BIGINT, end BIGINT, name_id INTEGER)"
    )
    strings: dict[str, int] = {}

    def intern(text):
        if text not in strings:
            strings[text] = len(strings) + 1
            conn.execute(
                f"INSERT INTO rocpd_string_{_GUID} (id, string) VALUES (?, ?)",
                (strings[text], text),
            )
        return strings[text]

    symbols: dict[str, int] = {}
    for _s, _e, name in dispatches:
        if name not in symbols:
            symbols[name] = len(symbols) + 1
            conn.execute(
                f"INSERT INTO rocpd_info_kernel_symbol_{_GUID} (id, kernel_name, display_name) "
                "VALUES (?, ?, ?)",
                (symbols[name], f"_mangled_{name}", name),
            )
    for i, (start, end, name) in enumerate(dispatches, start=1):
        conn.execute(
            f"INSERT INTO rocpd_kernel_dispatch_{_GUID} "
            "(id, pid, tid, agent_id, kernel_id, dispatch_id, queue_id, stream_id, "
            "start, end, region_name_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (i, _PID, _TID, _AGENT, symbols[name], i, 0, 0, start, end, 0),
        )
    for i, (start, end, text) in enumerate(regions, start=1):
        conn.execute(
            f"INSERT INTO rocpd_region_{_GUID} (id, pid, tid, start, end, name_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (i, _PID, _TID, start, end, intern(text)),
        )
    conn.commit()
    conn.close()


def test_shim_regions_feed_the_real_rocpd_join(tmp_path):
    """End-to-end: shim-emitted ranges own dispatches via the real C1 join.

    The annotator records two forward ranges. We map its recorded call ordinals
    to a nanosecond timeline, drop one dispatch inside each range (plus one before
    the first, which must be dropped as warm-up), and run the actual
    ``build_ranges_from_rocpd`` ownership join — the same code the offline producer
    uses. The shim's labels must be recognized and the containment attribution must
    hold.
    """
    backend = RecordingBackend()
    annotator = IterationAnnotator(backend, enabled=True)
    for _ in range(2):
        with annotator.forward(record={"total_num_scheduled_tokens": 2}):
            pass

    # Space recorded ranges onto a timeline with a gap between iterations.
    regions = []
    for ordinal, (label, start_order, end_order) in enumerate(backend.regions()):
        base = 1_000 + ordinal * 10_000
        regions.append((base, base + 5_000, label))

    db = tmp_path / "shim.db"
    _write_rocpd_from_regions(
        db,
        regions=regions,
        dispatches=[
            (500, 900, "warmup_fill_kernel"),  # before iter 0 -> dropped
            (1_500, 2_000, "hipblaslt_gemm_f16"),  # inside iter 0
            (11_500, 12_000, "flash_fwd_attn_kernel"),  # inside iter 1
        ],
    )

    ranges = build_ranges_from_rocpd(str(db))
    by_iter = {item.iteration: item for item in ranges}
    assert set(by_iter) == {0, 1}
    assert by_iter[0].phase == "forward" and by_iter[1].phase == "forward"
    assert by_iter[0].kernel_count == 1
    assert by_iter[1].kernel_count == 1
    assert {e.name for item in ranges for e in item.kernel_events} == {
        "hipblaslt_gemm_f16",
        "flash_fwd_attn_kernel",
    }
