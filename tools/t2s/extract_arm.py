"""Runs on coriander (stdlib, python3.9): per-rollout ground truth of one slime arm.

    python3 extract_arm.py <arm_dir> <out_dir>

Writes into out_dir:
  report.csv      report.json rollouts (streaming arms) or rollout_timing.jsonl (colocate)
  tuner.csv       train_metrics tuner_decision records (B before/effective, busy/idle GPU-s)
  chunks.csv      train_metrics train_chunk records
  windows.csv     trace.json/perfetto.json `inference` slices: rollout, engine, start, dur (s, trace clock)
  spans.csv       every other per-rollout span (training, chunk_*, switch_to_training ...) pid/name/ts/dur
  migrations.csv  run.log [MIGRATION] aborting lines + the next summary's rollout id
  wall_epoch.txt  trace wall_clock_epoch if present
"""
import csv, glob, gzip, json, os, re, sys

arm, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

def write(name, header, rows):
    with open(os.path.join(out, name), "w") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

rep = os.path.join(arm, "report.json")
if os.path.exists(rep):
    rolls = json.load(open(rep))["rollouts"]
    keys = [k for k in rolls[0] if not isinstance(rolls[0][k], (list, dict))]
    write("report.csv", keys, [[r.get(k) for k in keys] for r in rolls])
elif os.path.exists(os.path.join(arm, "rollout_timing.jsonl")):
    rows = []
    begin = {}
    for line in open(os.path.join(arm, "rollout_timing.jsonl")):
        d = json.loads(line)
        if d["event"] == "begin":
            begin[d["rollout"]] = d["epoch"]
        else:
            rows.append([d["rollout"], begin.get(d["rollout"]), d["epoch"], d["duration_s"],
                         d["inference_s"], d["train_s"], d["weight_update_s"]])
    write("report.csv", ["rollout_id", "begin_epoch", "end_epoch", "duration_s", "inference_s",
                         "train_s", "weight_update_s"], rows)

tuner, chunks = [], []
TK = ["rollout_id", "tuner", "applied", "b_before", "b_proposed", "b_effective", "idle_ratio",
      "training_span_gpu_s", "busy_gpu_s", "interior_idle_gpu_s", "trailing_idle_gpu_s",
      "interior_idle_ratio", "wall_s", "timestamp", "reason"]
CK = ["rollout_id", "train_group", "chunk_id", "total_tokens", "num_microbatches", "fwd_bwd_s",
      "chunk_total_s", "timestamp"]
for path in glob.glob(os.path.join(arm, "train_metrics", "*.jsonl")):
    for line in open(path):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("phase") == "tuner_decision":
            tuner.append([d.get(k) for k in TK])
        elif d.get("phase") == "train_chunk":
            chunks.append([d.get(k) for k in CK])
write("tuner.csv", TK, sorted(tuner, key=lambda r: r[0]))
write("chunks.csv", CK, sorted(chunks, key=lambda r: (r[0], r[-1])))

for tname in ("trace.json", "perfetto.json"):
    tp = os.path.join(arm, tname)
    if os.path.exists(tp):
        ev = json.load(open(tp))
        ev = ev["traceEvents"] if isinstance(ev, dict) else ev
        win, spans, epoch = [], [], None
        for e in ev:
            n, a = e.get("name"), e.get("args") or {}
            if n == "wall_clock_epoch":
                epoch = a
            if e.get("ph") != "X":
                continue
            if n == "inference" and "engine_idx" in a:
                win.append([a.get("rollout_id"), a["engine_idx"], e["ts"] / 1e6, e["dur"] / 1e6])
            elif n and not n.startswith("ws_"):
                spans.append([a.get("rollout_id"), e.get("pid"), e.get("tid"), n, e["ts"] / 1e6,
                              e["dur"] / 1e6, a.get("train_group", a.get("engine_idx", ""))])
        write("windows.csv", ["rollout", "engine", "start_s", "dur_s"], win)
        write("spans.csv", ["rollout", "pid", "tid", "name", "start_s", "dur_s", "group"], spans)
        with open(os.path.join(out, "wall_epoch.txt"), "w") as f:
            f.write(json.dumps({"source": tname, "epoch": epoch}) + "\n")
        break

ABORT = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\].*\[MIGRATION\] aborting (\d+) rid\(s\) on engine (\d+)"
                   r" -> dst engine (\d+) \((.*)\)")
PRES = re.compile(r"\[MIGRATION\] preserved buffered decode: mean=(\d+) tokens, min=(\d+), max=(\d+) \(across (\d+)")
SUMMARY = re.compile(r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\].*Migration summary for rollout (\d+): (\d+) group")
log = os.path.join(arm, "run.log.gz")
rows, pending = [], []
if os.path.exists(log):
    with gzip.open(log, "rt", errors="replace") as f:
        for line in f:
            h = ABORT.search(line)
            if h:
                pending.append([h.group(1), int(h.group(2)), int(h.group(3)), int(h.group(4)), h.group(5), "", "", ""])
                continue
            p = PRES.search(line)
            if p and pending:
                pending[-1][5:8] = [p.group(1), p.group(2), p.group(3)]
                continue
            s = SUMMARY.search(line)
            if s:
                for r in pending:
                    rows.append([int(s.group(2))] + r)
                rows.append([int(s.group(2)), s.group(1), -1, -1, -1, "summary groups=" + s.group(3), "", "", ""])
                pending = []
write("migrations.csv", ["rollout", "ts_local", "n_rids", "src", "dst", "detail", "pres_mean", "pres_min", "pres_max"], rows)
print("done", out)
