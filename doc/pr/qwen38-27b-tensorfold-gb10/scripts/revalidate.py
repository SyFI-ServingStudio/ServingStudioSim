"""Quiet-machine re-validation of merged TensorFold GB10 rows (read-only on the kept DB).

For every (table, backend) whose backend starts with `tensorfold_`, sample N stored, non-outlier GB10 rows spread
over the table, re-measure them with `kernel-profile run --fresh` (nothing is written) against a scratch copy of the
DB, and report fresh/stored time ratios. Run under the GPU lock with no other heavy CPU work on the box.
Usage: revalidate.py KEPT_DB OUT_DIR [--per-backend 8]"""
import argparse, json, shutil, sqlite3, statistics, subprocess, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("db"); ap.add_argument("out")
ap.add_argument("--per-backend", type=int, default=8)
ap.add_argument("--only", nargs="*", help="TABLE__BACKEND names to re-run")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
scratch = out / "scratch_profile.db"
if not scratch.exists():
    shutil.copy(a.db, scratch)

listing = json.loads(subprocess.run(
    ["uv", "run", "python", "-m", "launcher", "kernel-profile", "list", "--json"],
    capture_output=True, text=True, check=True).stdout)["profilers"]
pairs = sorted({(p["table"], p["backend"], p["args"]) for p in listing if p["backend"].startswith("tensorfold_")})

BOOL_FIELDS = {"residual", "causal"}  # stored as 0/1, the runner wants bool
con = sqlite3.connect(f"file:{a.db}?mode=ro", uri=True)
summary = []
for table, backend, args in pairs:
    if a.only and f"{table}__{backend}" not in a.only:
        continue
    fields = args.split(",")
    rows = con.execute(
        f"select {', '.join(chr(34) + f + chr(34) for f in fields)}, time_ms from \"{table}\" "
        f"where gpu_name = 'NVIDIA GB10' and backend = ? and is_outlier = 0 order by id", (backend,)).fetchall()
    if not rows:
        continue
    step = max(1, len(rows) // a.per_backend)
    sample = rows[::step][:a.per_backend]
    specs = []
    for r in sample:
        spec = {}
        for f, v in zip(fields, r[:-1]):
            spec[f] = json.loads(v) if isinstance(v, str) and v[:1] in "[{" else v
            if f in BOOL_FIELDS and v in (0, 1, "0", "1"):
                spec[f] = bool(int(v))
        specs.append(spec)
    spec_file = out / f"{table}__{backend}.specs.json"
    spec_file.write_text(json.dumps(specs))
    for attempt in range(6):
        # the previous submission's profiler container can still be exiting
        while subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip():
            time.sleep(5)
        proc = subprocess.run(
            ["uv", "run", "python", "-m", "launcher", "kernel-profile", "run", table, "--backend", backend, "--fresh",
             "--gpu-name", "NVIDIA GB10", "--db", str(scratch), "--specs", str(spec_file), "--json"],
            capture_output=True, text=True)
        if "idle GPU" not in proc.stdout:
            break
        time.sleep(15)
    (out / f"{table}__{backend}.out.json").write_text(proc.stdout)
    (out / f"{table}__{backend}.err").write_text(proc.stderr[-20000:])
    try:
        raw = proc.stdout; res = json.loads(raw[raw.find("{"):])["results"]
    except Exception as exc:  # noqa: BLE001
        summary.append((table, backend, len(sample), None, f"parse error: {exc}"))
        continue
    ratios = []
    for r, x in zip(sample, res):
        m = x.get("metrics", x)
        if x.get("status") == "ok" and m.get("time_ms"):
            ratios.append(m["time_ms"] / r[-1])
    note = f"min {min(ratios):.3f} max {max(ratios):.3f}" if ratios else "no ok results"
    summary.append((table, backend, len(sample), statistics.median(ratios) if ratios else None, note))
    print(f"{table:28s} {backend:18s} n={len(ratios)}/{len(sample)} median={summary[-1][3]} {note}", flush=True)

(out / ("summary_rerun.json" if a.only else "summary.json")).write_text(json.dumps(summary, indent=1))
