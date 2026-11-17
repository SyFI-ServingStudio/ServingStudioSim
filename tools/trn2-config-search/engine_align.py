"""Align glm53_flash_neuron_fp8 with a glm5-trn2 serving timeline (``engine/serve.py`` JSONL).

1. Parse the engine's steps: per step the decode forward (rows per DP group, ms) and the prefill
   forward (rows and starts per group, ms); the first use of every shape compiles or loads a NEFF
   and is dropped (a forward slower than ``--stall-x`` times its shape's median). Each decode
   step's context is rebuilt from the request rows (position = isl + tokens generated so far).
2. Replay every distinct shape through ``launcher timing-predict`` with the host constants set to
   zero, so each case's cost log holds pure device leaves; group them into the forward's regions
   (embed, one per layer, head, combine) from the cost manifest.
3. Fit the host constants of the pipelined launch model (``region_host_ms``, ``region_gap_ms``,
   ``forward_overhead_ms``) and the per-forward device factors (decode / prefill f_g) by weighted
   least squares in relative error over the measured forwards; ``step_overhead_ms`` is the
   median host time between forwards (step wall minus its forwards, plus the gap to the next
   step).

    uv run --no-sync python tools/trn2-config-search/engine_align.py \\
        /home/ec2-user/glm5-trn2/results/seed/serve_c16_replica0.jsonl --out <dir>

A service measurement (glm5-trn2 ``glmflash.service``) has no step timeline: pass its evaluator
``metrics.json`` (``measure-*/measure/metrics.json``) instead. Its cases are the warm-up's
steady-state forwards (every compiled shape once per replica, pass 1, one request per DP group),
and the in-service decode / prefill means are reported beside the fit. ``--host-ms H,F`` fixes
``region_host_ms`` / ``forward_overhead_ms`` to directly measured values (a device-bound
timeline cannot identify them; ``campaigns/glm8k1k/round2/tick2/host_probe.py`` measures the
whole-forward dispatch with the engine's throttle lifted).
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from inferencex import _multiplicity  # noqa: E402


def bucket(n: int, buckets: list[int]) -> int:
    return next((b for b in buckets if b >= n), n)


def parse(path: Path, stall_x: float = 3.0) -> dict:
    rows = [json.loads(line) for line in path.open() if line.strip()]
    header = next(r for r in rows if r["kind"] == "header")
    eng = header["engine"]
    steps = [r for r in rows if r["kind"] == "step"]
    reqs = [r for r in rows if r["kind"] == "request"]
    # Decode context: every request decoding at step time t sits at isl + generated tokens.
    active = []
    for r in reqs:
        first = r["arrival"] + r["ttft"]
        active.append((first, r["arrival"] + r["e2e"], r["isl"], r["tpot"]))

    def max_kv(t: float) -> int:
        kv = [isl + (t - f) / tp for f, e, isl, tp in active if f <= t < e and tp > 0]
        return int(max(kv)) + 1 if kv else 1

    fwd = defaultdict(list)
    gaps = []
    for i, s in enumerate(steps):
        if "decode" in s:
            b = max(s["decode"]["rows"])
            key = (
                "decode",
                bucket(b, eng["decode_buckets"]),
                bucket(min(max_kv(s["t"]), eng["cmax"]), eng["context_buckets"]),
            )
            fwd[key].append(s["decode"]["ms"])
        if "prefill" in s:
            p = s["prefill"]
            t = bucket(max(p["rows"]), eng["prefill_buckets"])
            end = max(st + t for st, r in zip(p["starts"], p["rows"]) if r)
            key = (
                "prefill",
                t,
                bucket(min(end, eng["cmax"]), eng["context_buckets"]),
                max(p["starts"]),
            )
            fwd[key].append(p["ms"])
        if i + 1 < len(steps):
            own = s.get("decode", {}).get("ms", 0) + s.get("prefill", {}).get("ms", 0)
            gaps.append((steps[i + 1]["t"] - s["t"]) * 1e3 - own)
    cases = []
    for key, ms in sorted(fwd.items()):
        med = statistics.median(ms)
        kept = [m for m in ms if m <= stall_x * med]
        cases.append(
            {
                "key": list(key),
                "n": len(kept),
                "dropped": len(ms) - len(kept),
                "measured_ms": statistics.median(kept),
            }
        )
    summary = next((r for r in rows if r["kind"] == "summary"), None)
    return {
        "engine": eng,
        "cases": cases,
        "step_gap_ms": statistics.median(gaps),
        "summary": summary,
    }


def parse_service(path: Path) -> dict:
    """Cases from a service evaluator's metrics.json: warm-up forwards per shape (both replicas)."""
    m = json.loads(path.read_text())
    cfg = json.loads((path.parent / "config.json").read_text())
    reps = m["final_state"]["replicas"]
    fwd = defaultdict(list)
    for rep in reps:
        for key, ms in rep["warm_ms"].items():
            kind, shape = key.split("_")
            x, ctx = (int(v) for v in shape.split("x"))
            fwd[(kind, x, ctx) if kind == "decode" else (kind, x, ctx, ctx - x)].append(ms)
    cases = [
        {"key": list(k), "n": len(v), "dropped": 0, "measured_ms": statistics.median(v)}
        for k, v in sorted(fwd.items())
    ]
    eng = {
        k: cfg[k]
        for k in (
            "tp",
            "dp",
            "prefill_buckets",
            "decode_buckets",
            "context_buckets",
            "slots",
            "cmax",
        )
    }
    eng["world"] = cfg["world_per_replica"]
    service = {
        "decode_step_ms_mean": [r["decode_step_ms_mean"] for r in reps],
        "prefill_step_ms_mean": [r["prefill_step_ms_mean"] for r in reps],
        "host_gap_ms_mean": [r["host_gap_ms_mean"] for r in reps],
        "decode_ctx_hist": [r["decode_ctx_hist"] for r in reps],
    }
    gap = statistics.mean(service["host_gap_ms_mean"])
    return {
        "engine": eng,
        "cases": cases,
        "step_gap_ms": gap,
        "summary": {
            k: m[k]
            for k in (
                "conc",
                "completed",
                "tput_per_chip",
                "median_tpot_ms",
                "median_intvty",
                "median_ttft_ms",
            )
        },
        "service": service,
    }


def predict_cases(parsed: dict, out: Path) -> list[dict]:
    """Per case: device ms per region of its forward (host constants zeroed)."""
    eng = parsed["engine"]
    dp, tp = eng["dp"], eng["tp"]
    groups = []
    for c in parsed["cases"]:
        kind, t, ctx = c["key"][:3]
        if kind == "decode":
            # A context just inside the bucket: the bucket is what the regions read.
            kv = max(ctx - 64, 1)
            groups.append({"groups": [{"decode_count": t, "average_decode_length": kv}] * dp})
        else:
            start = max(ctx - t, 0)
            groups.append({"groups": [{"prefill_chunk_pairs": [[start, t]]}] * dp})
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps(groups, indent=1))
    arch = {
        "type": "glm53_flash_neuron_fp8",
        "model_config": "model/config/glm53_flash.json",
        "fp8": False,
        "world": eng["world"],
        "attn_tp": tp,
        "prefill_buckets": eng["prefill_buckets"],
        "decode_buckets": eng["decode_buckets"],
        "context_buckets": eng["context_buckets"],
        "slots": eng["slots"],
        "cmax": eng["cmax"],
        "max_model_len": eng["cmax"],
        "region_host_ms": 0.0,
        "region_gap_ms": 0.0,
        "forward_overhead_ms": 0.0,
        "mtp_forward_overhead_ms": 0.0,
        "step_overhead_ms": 0.0,
        "region_device_factor": 1.0,
        "prefill_device_factor": 1.0,
    }
    cfg = {
        "arch": {"iter": arch},
        "gpu": "AWS Trainium2 LNC2",
        "log_dir": str(out / "predict"),
        "cases_file": "cases.json",
    }
    (out / "predict.json").write_text(json.dumps(cfg, indent=1))
    subprocess.run(
        [sys.executable, "-m", "launcher", "timing-predict", str(out / "predict.json"), "--no-gpu"],
        cwd=ROOT,
        check=True,
    )
    return regions_of(out / "predict")


def regions_of(run: Path) -> list[dict]:
    import pyarrow.parquet as pq

    mf = sorted((run / "raw" / "cost_manifest").glob("*.json"))[0]
    manifest = json.loads(mf.read_text())["sections"][0]
    mult = _multiplicity(manifest)
    names = [s["name"] for s in manifest["slots"]]
    table = pq.read_table(
        sorted((run / "raw" / "cost_log").glob("*.parquet"))[0],
        columns=["slot_time_ms", "total_time_ms"],
    ).to_pylist()
    out = []
    for row in table:
        fw: dict = {"decode": defaultdict(float), "prefill": defaultdict(float)}
        for name, ms, m in zip(names, row["slot_time_ms"], mult):
            parts = name.split(".")
            side = parts[1]
            if side not in fw or ".mtp" in name or parts[-1] in ("launch", "step"):
                continue
            if len(parts) > 4:  # <name>.<side>.<group>.<attn|ffn>.<op>: a layer region
                region = parts[2]
                fw[side][(region, m)] += ms * m
            else:
                region = (
                    "combine"
                    if parts[-1] == "combine_all_gather"
                    else ("head" if parts[-1] in ("head_all_gather_tp", "lm_head") else "embed")
                )
                fw[side][(region, 1)] += ms
        out.append(
            {side: [(dev / n, n) for (_, n), dev in regs.items()] for side, regs in fw.items()}
        )
    return out


def forward_ms(
    regions: list[tuple[float, int]], h: float, gap: float, ovh: float, f: float
) -> float:
    """The arch's pipelined forward: max(sum_r max(f d_r + gap, h), R h + overhead)."""
    floored = sum(n * max(f * d + gap, h) for d, n in regions)
    n_regions = sum(n for _, n in regions)
    return max(floored, n_regions * h + ovh)


def fit(parsed: dict, regions: list[dict], host: tuple[float, float] | None = None) -> dict:
    meas = []
    for c, reg in zip(parsed["cases"], regions):
        side = c["key"][0]
        meas.append((side, reg[side], c["measured_ms"], c["n"]))
    grid_h = [0.5 + 0.1 * i for i in range(26)] if host is None else [host[0]]
    grid_gap = [0.0, 0.05, 0.1, 0.15, 0.17, 0.2, 0.25, 0.35, 0.5]
    grid_ovh = [0.0, 5.0, 10.0, 15.0, 19.0, 25.0, 30.0, 40.0] if host is None else [host[1]]
    grid_f = [0.8 + 0.02 * i for i in range(91)]
    best = None
    for h, gap, ovh in itertools.product(grid_h, grid_gap, grid_ovh):
        # Device factors per side: the best f for each side at these host constants.
        fs, err = {}, 0.0
        for side in ("decode", "prefill"):
            cs = [m for m in meas if m[0] == side]
            if not cs:
                continue
            e_best = None
            for f in grid_f:
                e = sum(n * ((forward_ms(r, h, gap, ovh, f) - ms) / ms) ** 2 for _, r, ms, n in cs)
                if e_best is None or e < e_best[0]:
                    e_best = (e, f)
            err += e_best[0]
            fs[side] = e_best[1]
        # Prefer the trn2-flashnext priors on ties (identical error): smallest change.
        prior = abs(h - 1.3) + abs(gap - 0.17) + abs(ovh - 19.0) / 10
        key = (round(err, 9), prior)
        if best is None or key < best[0]:
            best = (
                key,
                dict(
                    region_host_ms=h,
                    region_gap_ms=gap,
                    forward_overhead_ms=ovh,
                    decode_device_factor=fs.get("decode", 1.0),
                    prefill_device_factor=fs.get("prefill", 1.0),
                ),
            )
    p = best[1]
    report = []
    for (side, r, ms, n), c in zip(meas, parsed["cases"]):
        f = p[f"{side}_device_factor"]
        report.append(
            {
                "case": c["key"],
                "n": n,
                "measured_ms": ms,
                "device_ms": sum(d * k for d, k in r),
                "predicted_default_ms": forward_ms(r, 1.3, 0.17, 19.0, 1.0),
                "predicted_fit_ms": forward_ms(
                    r, p["region_host_ms"], p["region_gap_ms"], p["forward_overhead_ms"], f
                ),
            }
        )
    p["step_overhead_ms"] = max(0.0, parsed["step_gap_ms"])
    return {"fit": p, "cases": report}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("timeline", type=Path, help="serve.py JSONL timeline or a service metrics.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stall-x", type=float, default=3.0)
    ap.add_argument("--host-ms", default=None, help="fix region_host_ms,forward_overhead_ms")
    a = ap.parse_args()
    service = a.timeline.name == "metrics.json"
    parsed = parse_service(a.timeline) if service else parse(a.timeline, a.stall_x)
    host = tuple(float(v) for v in a.host_ms.split(",")) if a.host_ms else None
    regions = predict_cases(parsed, a.out)
    result = fit(parsed, regions, host)
    result["summary"] = parsed["summary"]
    result["step_gap_ms"] = parsed["step_gap_ms"]
    if service:
        result["service"] = parsed["service"]
    (a.out / "alignment.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["fit"], indent=1))
    for c in result["cases"]:
        print(
            f"{str(c['case']):32s} n={c['n']:5d} measured {c['measured_ms']:8.1f}  device "
            f"{c['device_ms']:8.1f}  default {c['predicted_default_ms']:8.1f}  fit "
            f"{c['predicted_fit_ms']:8.1f}"
        )


if __name__ == "__main__":
    main()
