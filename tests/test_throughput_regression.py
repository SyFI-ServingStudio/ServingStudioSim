"""GPU throughput + sim-speed regression — the worked example of the
capability-tiered test infra (see tests/conftest.py + skill `run-tests`).

Runs a fixed deterministically-generated trace (tests/fixtures/gen_throughput_trace.py)
through the **launcher** (the canonical run interface — it also prebuilds the
kernel cache, warming `profile.db`) at a fixed config, then reads `summary.json`.
All checks **warn** on drift from a per-GPU golden rather than hard-failing, so an
intended cost-model change surfaces a signal without breaking the run; re-record
with `--update-golden` once the change is confirmed.

  - **throughput / completion-rate** (`gpu`): modeled tok-s / req-s of *simulated*
    time. Bit-identical across runs (a pure function of trace + config + cost
    model), so the warn bar is tight (±1%) — it trips on any cost-model change.
  - **sim-speed** (`bench`, opt-in): tick-loop `realtime_x` (sim ms per wall s,
    timed inside `run_sim` so startup is excluded). Host-load dependent, so it runs
    the sim several times and compares the **median** to the golden (±10% warn).

The golden key encodes the trace size `n`; changing the generator/size cleanly
invalidates it.

    just test-gpu                                        # throughput (warn ±1%)
    just test-bench                                      # + sim-speed median (warn ±10%)
    uv run pytest -m "gpu or bench" --update-golden      # (re)record goldens
"""

from __future__ import annotations

import json
import statistics
import warnings
from pathlib import Path

import pytest

from fixtures.gen_throughput_trace import DEFAULT_N, DEFAULT_SEED, write_trace

# Fixed config — any change here invalidates the goldens (re-record).
_MODEL_CONFIG = "model/config/llama3_8b.json"
_TP_SIZE = 1
_REQUEST_RATE = 150.0
_TRACE_N = DEFAULT_N
_THROUGHPUT_WARN = 0.01   # ±1%: throughput is deterministic → tight tripwire.
_SPEED_WARN = 0.10        # ±10% on the median of N (host-load dependent).
_SPEED_SAMPLES = 5        # sim runs to median over for the sim-speed comparison.

# Only `gpu` is a hard gate: the launcher (invoked below) builds the release
# binary and prewarms profile.db itself, so we must NOT gate on those being
# pre-provisioned — that would skip the test exactly when the launcher could
# bootstrap them. A CUDA device is the one thing it can't manufacture.
pytestmark = [pytest.mark.gpu]


def _golden_key() -> str:
    return f"llama3_8b@tp{_TP_SIZE},r{_REQUEST_RATE:g},n{_TRACE_N}"


def _run_once(log_dir: Path) -> dict:
    """Generate the trace + run it once through the launcher; parse summary.json."""
    trace = write_trace(log_dir / "trace.csv", n=_TRACE_N, seed=DEFAULT_SEED)
    preset = {
        "deployment": "unified",
        "model_config": _MODEL_CONFIG,
        "trace_files": [str(trace)],
        "tp_size": _TP_SIZE,
        "request_rate": _REQUEST_RATE,
        "run_to_end": True,
        "log_dir": str(log_dir),
    }
    (log_dir / "preset.json").write_text(json.dumps(preset))

    from launcher.__main__ import main
    rc = main([str(log_dir / "preset.json")])
    if rc != 0:
        stdout_log = log_dir / "stdout.log"
        tail = stdout_log.read_text()[-2000:] if stdout_log.is_file() else ""
        pytest.fail(f"launcher run failed (rc={rc}):\n{tail}")

    summary = json.loads((log_dir / "summary.json").read_text())
    assert summary["cause"] == "DrainComplete", summary["cause"]
    return summary


@pytest.fixture(scope="module")
def run_summary(tmp_path_factory) -> dict:
    """One launcher run — enough for the deterministic throughput metrics."""
    return _run_once(tmp_path_factory.mktemp("throughput_run"))


@pytest.fixture(scope="module")
def speed_samples(tmp_path_factory) -> list[float]:
    """`realtime_x` from N runs, for a median sim-speed comparison (jitter-robust)."""
    out = []
    for i in range(_SPEED_SAMPLES):
        out.append(_run_once(tmp_path_factory.mktemp(f"speed_run{i}"))["realtime_x"])
    return out


def _warn_vs_golden(metric: str, measured: float, golden, threshold: float) -> None:
    """Compare to the per-GPU golden: record+skip if unset (or --update-golden),
    else WARN on >threshold drift. Never fails — these are monitors, not gates."""
    g, key = golden(metric), _golden_key()
    if g.update_enabled or key not in g:
        g.record(key, measured)
        pytest.skip(f"recorded {metric}[{key}]={measured:.2f} -> {g.path.name}")
    ref = g.get(key)
    rel = abs(measured - ref) / ref if ref else float("inf")
    if rel > threshold:
        warnings.warn(
            f"{metric}[{key}] drifted {rel * 100:.1f}% on {g.gpu_name}: "
            f"measured={measured:.2f} vs golden={ref:.2f} (bar ±{threshold * 100:.0f}%; "
            f"re-record with --update-golden if intended)",
            stacklevel=2,
        )


def test_total_throughput_matches_golden(run_summary, golden):
    """Modeled total tok/s vs the per-GPU golden (warn ±1%)."""
    _warn_vs_golden("throughput", run_summary["total_tok_s"], golden, _THROUGHPUT_WARN)


def test_completion_rate_matches_golden(run_summary, golden):
    """Modeled completed req/s vs the per-GPU golden (warn ±1%)."""
    _warn_vs_golden("completion_rate", run_summary["completed_req_s"], golden, _THROUGHPUT_WARN)


@pytest.mark.bench
def test_sim_speed_median_trend(speed_samples, golden):
    """Median tick-loop sim-speed (realtime_x) over N runs vs the per-GPU golden
    (warn ±10%). Median tames host-load jitter; opt-in via -m bench."""
    _warn_vs_golden("sim_speed", statistics.median(speed_samples), golden, _SPEED_WARN)
