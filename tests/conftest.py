"""Shared pytest gating + fixtures for the ServingStudioSim test tiers.

Tests declare a **capability tier** with a marker; this conftest auto-skips the
ones the host can't run, so bare ``uv run pytest`` "just works" anywhere:

  - (unmarked)      cpu — deterministic, no GPU/binary. Always collected.
  - ``needs_binary`` requires the built ``simulator`` binary (release).
  - ``gpu``         requires a CUDA device.
  - ``needs_db``    requires warm ``profile.db`` rows for the config under test.
  - ``agent``       Codex runner+judge runtime — opt-in only (``--run-agent``).
  - ``bench``       perf/timing, non-deterministic — opt-in only (``-m bench``).

Numeric goldens that depend on hardware (throughput, kernel time) are stored
**per GPU type** under ``tests/golden/<metric>/<gpu_name>.json`` and selected by
the detected device (see the ``golden`` fixture). ``--update-golden`` records.

Run under ``uv`` so torch / the venv interpreter resolve (see
``vibesim_pyo3_build_python_pin``); ``just test-<tier>`` wraps the right flags.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from launcher.exec import binary_path
from launcher.golden import GOLDEN_ROOT, Golden

#: Intra-op threads the CPU reference kernels are allowed. Torch otherwise sizes
#: its pool from the core count, and on a many-core host the fan-out/join cost
#: dwarfs these small reference GEMMs: on a 256-core box the same three files
#: took 18 s and 63 s on two back-to-back runs, against a flat 13 s once capped.
#: Any small cap behaves the same (4, 8 and 32 all measured within noise), so
#: this only bounds the pathology — it is not a tuned value.
_TEST_INTRA_OP_THREADS = 8


def _cap_cpu_thread_pools() -> None:
    """Bound the thread pools before any test module imports torch.

    Set as a default only: an explicit ``OMP_NUM_THREADS`` in the environment
    still wins, so a run can be widened by hand.
    """
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(variable, str(_TEST_INTRA_OP_THREADS))
    # A plugin may have imported torch already, in which case the OpenMP pool is
    # sized and only the runtime setter still bites.
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(_TEST_INTRA_OP_THREADS)


_cap_cpu_thread_pools()

_TIER_MARKERS = {
    "gpu": "requires a CUDA device",
    "needs_binary": "requires the built simulator binary (release)",
    "needs_db": "requires warm profile.db rows for the config under test",
    "agent": "requires the Codex runner+judge runtime (opt-in: --run-agent)",
    "bench": "perf/timing, non-deterministic (opt-in: -m bench)",
}

# `GOLDEN_ROOT` / `Golden` are imported from `launcher.golden` (re-exported here
# for tests that read the store directly): `alignment-campaign compare --record`
# writes the same files from a production command, so the storage implementation
# is shared rather than duplicated on either side.
assert GOLDEN_ROOT == Path(__file__).resolve().parent / "golden"


# ── registration + CLI options ───────────────────────────────────────────────

def pytest_configure(config: pytest.Config) -> None:
    for name, desc in _TIER_MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {desc}")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-agent", action="store_true", default=False,
                     help="collect the `agent` tier (Codex skill tests)")
    parser.addoption("--update-golden", action="store_true", default=False,
                     help="record measured values into the per-GPU golden store")


# ── capability detection (cached once per session) ───────────────────────────

def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _detect_gpu_name() -> str | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)  # exact profile.db key, e.g. "NVIDIA H200"
    except Exception:
        pass
    return None


# ── auto-skip hook ───────────────────────────────────────────────────────────

def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tiers the host can't (or wasn't asked to) run. Capability tiers
    (gpu/needs_binary/needs_db) auto-skip on absence; expensive tiers
    (agent/bench) are skipped unless explicitly opted in."""
    has_cuda = _cuda_available()
    has_binary = binary_path("release").is_file()
    markexpr = config.getoption("markexpr") or ""
    run_agent = config.getoption("--run-agent")
    bench_requested = "bench" in markexpr

    skips = {
        "gpu": None if has_cuda else pytest.mark.skip(reason="no CUDA device"),
        "needs_binary": None if has_binary else pytest.mark.skip(
            reason="release simulator not built (uv run cargo build --release -p simulator)"),
        "needs_db": None,  # presence is config-specific; the test asserts/skips itself
        "agent": None if run_agent else pytest.mark.skip(reason="agent tier opt-in (--run-agent)"),
        "bench": None if bench_requested else pytest.mark.skip(reason="bench tier opt-in (-m bench)"),
    }
    for item in items:
        for tier, marker in skips.items():
            if marker is not None and tier in item.keywords:
                item.add_marker(marker)


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def gpu_name() -> str:
    """Exact CUDA device name = the `profile.db` / golden key (e.g. "NVIDIA H200").
    Skips if no GPU — pair with @pytest.mark.gpu so the skip is consistent."""
    name = _detect_gpu_name()
    if name is None:
        pytest.skip("no CUDA device to identify")
    return name


@pytest.fixture(scope="session")
def sim_bin() -> Path:
    """Path to the release simulator binary; skip if unbuilt (pair with
    @pytest.mark.needs_binary)."""
    path = binary_path("release")
    if not path.is_file():
        pytest.skip("release simulator not built")
    return path


@pytest.fixture
def golden(gpu_name: str, request: pytest.FixtureRequest):
    """Factory: `golden("throughput")` → a `Golden` for the detected GPU."""
    update = request.config.getoption("--update-golden")
    return lambda metric: Golden(metric, gpu_name, update)
