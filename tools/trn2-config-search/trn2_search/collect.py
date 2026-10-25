"""Read finished runs into one row each.

A run directory is `<log_root>/<config label>/<workload label>/` (see
`presets.render`). Per run this reads:

- `summary.json`: token throughputs and the simulated span;
- `raw/request_slo.parquet`: per-request TTFT and mean TPOT, for the
  percentiles and for goodput under each SLO;
- `reports/optimality_report.json`: the cluster R0-R7 ladder and its gap
  buckets, when the analyzer produced it.

Goodput under an SLO `(ttft_ms, tpot_ms)` counts the completed requests whose
TTFT and mean TPOT both meet it, per second of simulated span; `goodput_tok_s`
is their output tokens per second, `slo_attain` their fraction of all requests.
A request with one output token has no TPOT and is judged on TTFT alone.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

LADDER_KEYS = (
    "real",
    "busy",
    "balanced",
    "per_config_best",
    "ignore_network",
    "sustained_limit",
    "achievable_limit",
    "hardware_limit",
    "segmented_necessary",
)
BUCKET_KEYS = (
    "idle",
    "imbalance",
    "batching",
    "communication",
    "kernel_gap",
    "power_gap",
    "hardware_gap",
    "excess_over_necessary",
    "fusion",
    "hardware_necessary",
)


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (numpy's default), NaN when empty."""
    xs = sorted(v for v in values if v is not None and not math.isnan(v))
    if not xs:
        return float("nan")
    pos = (len(xs) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def goodput(requests: list[dict], span_ms: float, ttft_ms: float, tpot_ms: float) -> dict:
    """Goodput of `requests` (`completed, ttft_ms, tpot_mean_ms, num_output_tokens`)."""
    ok = [
        r
        for r in requests
        if r["completed"]
        and r["ttft_ms"] is not None
        and r["ttft_ms"] <= ttft_ms
        and (r["tpot_mean_ms"] is None or r["tpot_mean_ms"] <= tpot_ms)
    ]
    span_s = span_ms / 1000.0 if span_ms > 0 else float("nan")
    return {
        "goodput_req_s": len(ok) / span_s,
        "goodput_tok_s": sum(r["num_output_tokens"] for r in ok) / span_s,
        "slo_attain": len(ok) / len(requests) if requests else float("nan"),
    }


def read_requests(run_dir: Path) -> list[dict]:
    import pyarrow.parquet as pq

    cols = ["completed", "ttft_ms", "tpot_mean_ms", "num_output_tokens", "arrival_time_ms"]
    return pq.read_table(run_dir / "raw" / "request_slo.parquet", columns=cols).to_pylist()


def run_row(run_dir: Path, slos: dict[str, tuple[float, float]]) -> dict | None:
    """One finished run as a flat dict; None when it has no summary."""
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        return None
    s = json.loads(summary_path.read_text())
    reqs = read_requests(run_dir)
    ttft = [r["ttft_ms"] for r in reqs if r["completed"]]
    tpot = [r["tpot_mean_ms"] for r in reqs if r["completed"] and r["tpot_mean_ms"] is not None]
    span = float(s["sim_ms"])
    row = {
        "requests": s["requests_total"],
        "finished": s["requests_finished"],
        "total_tok_s": s["total_tok_s"],
        "prefill_tok_s": s["prefill_tok_s"],
        "decode_tok_s": s["decode_tok_s"],
        "cores": s["num_gpus"],
        "tok_s_per_chip": s["total_tok_s"] / (s["num_gpus"] / 4),
        "req_s": s["completed_req_s"],
        "sim_ms": span,
    }
    for q in (50, 90, 99):
        row[f"ttft_p{q}"] = percentile(ttft, q)
        row[f"tpot_p{q}"] = percentile(tpot, q)
    for name, (t_ms, p_ms) in slos.items():
        for k, v in goodput(reqs, span, t_ms, p_ms).items():
            row[f"{k}@{name}"] = v
    opt = run_dir / "reports" / "optimality_report.json"
    if opt.is_file():
        cl = json.loads(opt.read_text()).get("cluster") or {}
        for k in LADDER_KEYS:
            row[f"ladder_{k}"] = cl.get(k)
        for k in BUCKET_KEYS:
            b = (cl.get("buckets") or {}).get(k)
            row[f"gap_{k}"] = b.get("frac") if b else None
    return row


def collect(
    log_root: Path, configs: dict[str, dict], slos: dict[str, tuple[float, float]]
) -> list[dict]:
    """Every finished run under `log_root` (any depth: `.../<config label>/<workload
    label>/summary.json`) whose config label is in `configs`."""
    rows = []
    for summary in sorted(log_root.rglob("summary.json")):
        run_dir = summary.parent
        label, workload = run_dir.parent.name, run_dir.name
        cfg = configs.get(label)
        if cfg is None:
            continue
        r = run_row(run_dir, slos)
        if r is None:
            continue
        base, _, rate = workload.rpartition("-a")
        if not base:
            base, rate = workload, None
        rows.append(
            {
                "label": label,
                "workload": workload,
                # The trace without its MTP acceptance suffix, and that rate.
                "workload_base": base,
                "cfg.accept_rate": float(rate) if rate else None,
                **{f"cfg.{k}": v for k, v in cfg.items() if k != "label"},
                **r,
            }
        )
    return rows
