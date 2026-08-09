"""GPU throughput + sim-speed + PD functional regression — the worked example
of the capability-tiered test infra (see tests/conftest.py + skill `dev-run-tests`).

Each test runs a fixed deterministically-generated trace
(tests/fixtures/gen_throughput_trace.py) through the **launcher** (the canonical
run interface — it also prebuilds the kernel cache, warming `profile.db`) at a
fixed YAML preset, then reads `summary.json`. All checks **warn** on drift from
a per-GPU golden rather than hard-failing, so an intended cost-model change
surfaces a signal without breaking the run; re-record with `--update-golden`
once the change is confirmed.

  - **unified throughput / completion-rate** (`gpu`): modeled tok-s / req-s of
    *simulated* time on the single-pool unified deployment. Bit-identical across
    runs (a pure function of trace + config + cost model), so the warn bar is
    tight (±1%) — it trips on any cost-model change.
  - **unified sim-speed** (`bench`, opt-in): tick-loop `realtime_x` (sim ms per
    wall s, timed inside `run_sim` so startup is excluded). Host-load dependent,
    so it runs the sim several times and compares the **median** to the golden
    (±10% warn).
  - **PD 1p32d throughput / completion-rate** (`gpu`): same shape as unified but
    on a prefill/decode-disaggregated deployment (1 prefill replica, 32 decode
    replicas) — the canonical PD config. Bit-identical, ±1% warn.

The golden key encodes trace size `n` + deployment shape; changing either
cleanly invalidates it.

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
import yaml

from fixtures.gen_throughput_trace import DEFAULT_N, DEFAULT_SEED, write_trace

# Fixed config — any change to the preset templates below invalidates the
# goldens (re-record with --update-golden).
_REQUEST_RATE = 150.0
_TRACE_N = DEFAULT_N
_THROUGHPUT_WARN = 0.01   # ±1%: throughput is deterministic → tight tripwire.
_SPEED_WARN = 0.10        # ±10% on the median of N (host-load dependent).
_SPEED_SAMPLES = 5        # sim runs to median over for the sim-speed comparison.

# YAML preset templates — the launcher accepts YAML directly (matches the
# `presets/*.yaml` shape). Trace path + log_dir are filled in per test run.
_UNIFIED_PRESET_TEMPLATE = """
deployment: unified
workload:
  request_rate: 150.0
  replay_pacing: open_loop
  session_dependency: independent
  run_to_end: true
pools:
  main:
    groups:
      - gpu: "NVIDIA H200"
        replicas: 1
        arch:
          type: llama3_dense_tp
          model_config: model/config/llama3_8b.json
          tp_size: 1
        worker:
          type: barebone
"""

_PD_1P32D_PRESET_TEMPLATE = """
deployment: pd
workload:
  request_rate: 150.0
  replay_pacing: open_loop
  session_dependency: independent
  run_to_end: true
pools:
  prefill:
    groups:
      - gpu: "NVIDIA H200"
        replicas: 1
        arch:
          type: llama3_dense_tp
          model_config: model/config/llama3_8b.json
          tp_size: 1
        worker:
          type: pd_prefill
  decode:
    groups:
      - gpu: "NVIDIA H200"
        replicas: 32
        arch:
          type: llama3_dense_tp
          model_config: model/config/llama3_8b.json
          tp_size: 1
        worker:
          type: pd_decode
"""

# Only `gpu` is a hard gate: the launcher (invoked below) builds the release
# binary and prewarms profile.db itself, so we must NOT gate on those being
# pre-provisioned — that would skip the test exactly when the launcher could
# bootstrap them. A CUDA device is the one thing it can't manufacture.
pytestmark = [pytest.mark.gpu]


def _golden_key() -> str:
    """Trace + config slot inside each per-metric golden file. The metric name
    itself disambiguates deployment shape (`throughput` vs `pd_1p32d_throughput`),
    so the in-file key just needs to encode trace size + model + rate."""
    return f"llama3_8b@tp1,r{_REQUEST_RATE:g},n{_TRACE_N}"


def _run_preset(log_dir: Path, template: str) -> dict:
    """Render a preset template (YAML) with this run's trace path + log_dir,
    run it through the launcher, parse `summary.json`."""
    trace = write_trace(log_dir / "trace.csv", n=_TRACE_N, seed=DEFAULT_SEED)
    preset = yaml.safe_load(template)
    preset["workload"]["trace_files"] = [str(trace)]
    preset.setdefault("io", {})["log_dir"] = str(log_dir)
    preset_path = log_dir / "preset.yaml"
    preset_path.write_text(yaml.safe_dump(preset, sort_keys=False))

    from launcher.__main__ import main
    rc = main([str(preset_path)])
    if rc != 0:
        stdout_log = log_dir / "stdout.log"
        tail = stdout_log.read_text()[-2000:] if stdout_log.is_file() else ""
        pytest.fail(f"launcher run failed (rc={rc}):\n{tail}")

    summary = json.loads((log_dir / "summary.json").read_text())
    assert summary["cause"] == "DrainComplete", summary["cause"]
    return summary


@pytest.fixture(scope="module")
def unified_summary(tmp_path_factory) -> dict:
    """One unified launcher run — enough for the deterministic throughput metrics."""
    return _run_preset(tmp_path_factory.mktemp("unified_run"), _UNIFIED_PRESET_TEMPLATE)


@pytest.fixture(scope="module")
def pd_1p32d_summary(tmp_path_factory) -> dict:
    """One PD 1p32d launcher run — verifies the PD path completes cleanly and
    surfaces a comparable throughput / completion-rate number."""
    return _run_preset(tmp_path_factory.mktemp("pd_1p32d_run"), _PD_1P32D_PRESET_TEMPLATE)


@pytest.fixture(scope="module")
def speed_samples(tmp_path_factory) -> list[float]:
    """`realtime_x` from N unified runs, for a median sim-speed comparison."""
    out = []
    for i in range(_SPEED_SAMPLES):
        out.append(_run_preset(tmp_path_factory.mktemp(f"speed_run{i}"), _UNIFIED_PRESET_TEMPLATE)["realtime_x"])
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


def test_total_throughput_matches_golden(unified_summary, golden):
    """Unified modeled total tok/s vs the per-GPU golden (warn ±1%)."""
    _warn_vs_golden("throughput", unified_summary["total_tok_s"], golden, _THROUGHPUT_WARN)


def test_completion_rate_matches_golden(unified_summary, golden):
    """Unified modeled completed req/s vs the per-GPU golden (warn ±1%)."""
    _warn_vs_golden("completion_rate", unified_summary["completed_req_s"], golden, _THROUGHPUT_WARN)


def test_pd_1p32d_throughput_matches_golden(pd_1p32d_summary, golden):
    """PD 1p32d modeled total tok/s vs the per-GPU golden (warn ±1%). Doubles as
    a functional smoke for the PD prefill→decode handoff path."""
    _warn_vs_golden("pd_1p32d_throughput", pd_1p32d_summary["total_tok_s"], golden, _THROUGHPUT_WARN)


def test_pd_1p32d_completion_rate_matches_golden(pd_1p32d_summary, golden):
    """PD 1p32d modeled completed req/s vs the per-GPU golden (warn ±1%)."""
    _warn_vs_golden("pd_1p32d_completion_rate", pd_1p32d_summary["completed_req_s"], golden, _THROUGHPUT_WARN)


@pytest.mark.bench
def test_sim_speed_median_trend(speed_samples, golden):
    """Median tick-loop sim-speed (realtime_x) over N runs vs the per-GPU golden
    (warn ±10%). Median tames host-load jitter; opt-in via -m bench."""
    _warn_vs_golden("sim_speed", statistics.median(speed_samples), golden, _SPEED_WARN)
