"""Export the profiled kernel rows the design runs on: kernel_specs.jsonl beside this file.

The design's kernels are the slots of the PP4 [12,11,11,11] stage graphs (each run's raw/cost_manifest) that take
time in prefill (nonzero in raw/cost_log). For each (kind, backend, static config) this writes every B200 row of
profile.db that matches the static config: the full spec, the measured time, and the positions that use it.
Elementwise glue ops are left out: their rows are a byte-moving stand-in, not the op itself (see 2_kernels.md).

usage (ServingStudioSim root, at the frozen commit, after the bring-up one-request runs):
  PYTHONPATH=$PWD uv run python doc/handoff/glm53_flash_pp4_b200_v1/data/kernel_specs.py <one-request run dir>
"""
import collections, json, sqlite3, sys
from pathlib import Path

import pyarrow.parquet as pq

from profiling.db.registry import find_args_schema
from dataclasses import fields

run = Path(sys.argv[1])
here = Path(__file__).parent
cases = sorted(p for p in run.iterdir() if (p / "raw/cost_manifest").is_dir())

used = collections.OrderedDict()  # (kind, backend, static json) -> set of slot names
for case in cases:
    for manifest in sorted((case / "raw/cost_manifest").glob("worker_stage_*.json")):
        stage = manifest.stem.removeprefix("worker_stage_")
        slots = [s for sec in json.loads(manifest.read_text())["sections"] for s in sec["slots"]]
        log = pq.read_table(case / f"raw/cost_log/{manifest.stem}.parquet", columns=["slot_time_ms", "slot_backend"]).to_pylist()
        for row in log:
            for s, t, b in zip(slots, row["slot_time_ms"], row["slot_backend"]):
                if t <= 0 or b == 255 or s["kind"] == "elementwise":
                    continue
                cfg = s["kernel_config"]
                static = {k: (v["value"] if isinstance(v, dict) and "value" in v else v)
                          for k, v in cfg.items() if k not in ("backends", "gpu_name")}
                for backend in cfg["backends"]:  # every candidate: the design takes the faster per shape
                    key = (s["kind"], backend, json.dumps(static, sort_keys=True))
                    used.setdefault(key, set()).add(f"stage {stage}: {s['name']}")

db = sqlite3.connect(f"file:{Path('profiling/profile.db').resolve()}?mode=ro", uri=True)
out = (here / "kernel_specs.jsonl").open("w")
summary = []
for (kind, backend, static_json), positions in used.items():
    static = json.loads(static_json)
    args = [f.name for f in fields(find_args_schema(kind, backend))]
    match = {k: v for k, v in static.items() if k in args and not isinstance(v, (list, dict))}
    sql = f'select {", ".join(chr(34) + a + chr(34) for a in args)}, time_ms from "{kind}" ' \
          f'where gpu_name = ? and backend = ? and coalesce(is_outlier, 0) = 0' + "".join(f' and "{k}" = ?' for k in match)
    rows = db.execute(sql, ["NVIDIA B200", backend, *[int(v) if isinstance(v, bool) else v for v in match.values()]]).fetchall()
    for r in rows:
        spec = dict(zip(args, r[:-1]))
        out.write(json.dumps({"kind": kind, "backend": backend, "spec": spec, "db_time_ms": r[-1],
                              "positions": sorted(positions)}) + "\n")
    summary.append((kind, backend, static, len(rows), sorted(positions)))
out.close()
for kind, backend, static, n, positions in summary:
    print(f"{kind:30} {backend:28} rows {n:5}  {static}  [{len(positions)} positions]")
print(sum(s[3] for s in summary), "rows in", here / "kernel_specs.jsonl")
