"""CLI: replay the flashnext engine's serving steps and runs through the Sim.

    align.py replay <out-dir> [--arch JSON] [--runs SUBSTR ...]
    align.py e2e    <out-dir> [--arch JSON] [--worker JSON] [--runs SUBSTR ...]

Both print a markdown table and write it (and the raw rows) under <out-dir>.
`--arch` / `--worker` override knobs on top of each run's own (what-ifs, fits)."""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import engine_align as ea  # noqa: E402

# A step this much slower than predicted, by at least this much, is a first-use
# stall (the engine's first execution of a shape after warm-up).
STALL_RATIO, STALL_MS = 1.3, 200.0


def replay(out: Path, arch: dict, runs: list[str]) -> str:
    names = sorted({f for _, files in ea.RUNS for f in files})
    if runs:
        names = [n for n in names if any(x in n for x in runs)]
    cols = [
        "run",
        "kind",
        "steps",
        "step wall meas / pred (ms)",
        "err",
        "decode fwd+MTP meas / pred",
        "prefill fwd+MTP meas / pred",
        "device meas / pred",
        "total err",
        "total err w/o stalls",
    ]
    lines = ["| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
    for name in names:
        steps = ea.replay_run(name, arch, out / "tp")
        (out / f"{name}.json").write_text(json.dumps(steps))
        stalls = sum(
            r["wall"] - r["p_wall"]
            for r in steps
            if r["wall"] > STALL_RATIO * r["p_wall"] and r["wall"] - r["p_wall"] > STALL_MS
        )
        tot_m = sum(r["wall"] for r in steps)
        tot_p = sum(r["p_wall"] for r in steps)
        by = defaultdict(list)
        for r in steps[1:]:
            by[r["kind"]].append(r)
        for i, (kind, rs) in enumerate(sorted(by.items())):

            def med(key, rows=rs):
                v = [r[key] for r in rows if r.get(key)]
                return st.median(v) if v else None

            pairs = {
                "dec": (med("dec_wall"), med("p_dec", [r for r in rs if r.get("dec_wall")])),
                "pre": (med("pre_wall"), med("p_pre", [r for r in rs if r.get("pre_wall")])),
                "dev": (med("device"), med("p_device", [r for r in rs if r.get("device")])),
            }
            cell = lambda p: f"{p[0]:.1f} / {p[1]:.1f}" if p[0] else ""  # noqa: E731
            mw, pw = med("wall"), med("p_wall")
            total = f"{ea.error_pct(tot_p, tot_m):+.1f}%" if i == 0 else ""
            clean = (
                f"{ea.error_pct(tot_p, tot_m - stalls):+.1f}% ({stalls / 1e3:.1f} s stalls)"
                if i == 0
                else ""
            )
            row = [
                name,
                kind,
                str(len(rs)),
                f"{mw:.1f} / {pw:.1f}",
                f"{ea.error_pct(pw, mw):+.1f}%",
            ]
            row += [cell(pairs["dec"]), cell(pairs["pre"]), cell(pairs["dev"]), total, clean]
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def e2e(out: Path, arch: dict, worker: dict, runs: list[str]) -> str:
    results = {}
    for label, files in ea.RUNS:
        if runs and not any(x in label for x in runs):
            continue
        results[label] = ea.e2e_run(label, files, arch, worker, out)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    head = " | ".join(f"{m} meas / sim (err)" for m in ea.METRICS)
    lines = [f"| run | {head} |", "|---" * (len(ea.METRICS) + 1) + "|"]
    errs = defaultdict(list)
    for label, r in results.items():
        cells = []
        for m in ea.METRICS:
            mv, sv = r["measured"][m], r["simulated"][m]
            e = ea.error_pct(sv, mv)
            errs[m].append(e)
            cells.append(f"{mv:.1f} / {sv:.1f} ({e:+.1f}%)")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Median |error|: "
        + ", ".join(f"{m} {ea.median_abs(v):.1f}%" for m, v in errs.items())
        + f"; all {ea.median_abs([x for v in errs.values() for x in v]):.1f}%."
    )
    w64 = [(label, r) for label, r in results.items() if label.startswith("w64")]
    if w64:

        def rank(key):
            return [label for label, _ in sorted(w64, key=lambda x: x[1][key]["duration_s"])]

        lines.append("")
        lines.append("W64 by duration, measured: " + ", ".join(rank("measured")))
        lines.append("")
        lines.append("W64 by duration, simulated: " + ", ".join(rank("simulated")))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mode", choices=["replay", "e2e"])
    ap.add_argument("out", type=Path)
    ap.add_argument("--arch", default="{}", help="arch knob overrides (JSON)")
    ap.add_argument("--worker", default="{}", help="worker knob overrides (JSON, e2e)")
    ap.add_argument(
        "--runs", nargs="*", default=[], help="only runs whose name contains one of these"
    )
    a = ap.parse_args()
    out = a.out if a.out.is_absolute() else ea.SIM / a.out
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "replay":
        table = replay(out, json.loads(a.arch), a.runs)
    else:
        table = e2e(out, json.loads(a.arch), json.loads(a.worker), a.runs)
    (out / f"{a.mode}.md").write_text(table + "\n")
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
