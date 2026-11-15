"""Read a run's per-request tier reads from `raw/prefix_cache_event` (the tier log of these scripts).

`reads(run)` yields one row per tier read that its request was admitted after: the DP rank (partition), the
session, the request's declared prefix tokens, the read's start (`read_ms`) and the admission that took it
(`admit_ms`, the request's next HBM `activate` event). `admissions(run)` yields every session request's HBM
admission: (partition, admit_ms, session, declared prefix tokens).
"""
import glob

import pyarrow.parquet as pq

COLUMNS = ["partition_id", "time_ms", "sequence", "request_id", "session_id", "operation", "reason", "tier"]


def _events(run):
    rows = []
    for f in sorted(glob.glob(f"{run}/raw/prefix_cache_event/*.parquet")):
        rows += pq.read_table(f, columns=COLUMNS).to_pylist()
    rows.sort(key=lambda r: (r["time_ms"], r["sequence"]))
    return rows


def _declared(run):
    t = pq.read_table(f"{run}/raw/request_slo.parquet", columns=["request_id", "declared_prefix_tokens"]).to_pydict()
    return dict(zip(t["request_id"], t["declared_prefix_tokens"]))


def reads(run):
    declared = _declared(run)
    started = {}
    for r in _events(run):
        if r["tier"] != "hbm" and r["reason"] == "read":
            started[r["request_id"]] = r["time_ms"]
        elif r["tier"] == "hbm" and r["operation"] == "activate" and r["request_id"] in started:
            yield dict(partition=r["partition_id"], session_id=r["session_id"],
                       declared_tokens=declared[r["request_id"]], read_ms=started.pop(r["request_id"]),
                       admit_ms=r["time_ms"])


def admissions(run):
    declared = _declared(run)
    for r in _events(run):
        if r["tier"] == "hbm" and r["operation"] == "activate" and r["session_id"] is not None:
            yield dict(partition=r["partition_id"], admit_ms=r["time_ms"], session_id=r["session_id"],
                       declared_tokens=declared[r["request_id"]])
