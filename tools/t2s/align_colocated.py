"""Colocated Text2SQL arm: simulated inference phase vs measured, per rollout and per iteration.

    uv run python tools/t2s/align_colocated.py <sweep_dir> --per-turn <handoff>/per_turn.csv.gz \
        --trace-dir trace/t2s/colocated

<sweep_dir> holds one simulated run per rollout (`r{N}/`) replaying
trace/t2s/colocated/r{N}.csv. Ground truth:
  per_turn.csv.gz         the handoff's per-turn export: turn_t_send / turn_t_recv /
                          turn_server_s per turn
  data/arms/*colocate*/report.csv   rollout_timing.jsonl (inference_s)
  data/coloc_iters.csv.gz every SGLang scheduler iteration, all 8 engines
                          (extract_engine_iters.py)

Prints (1) per-rollout makespan, (2) the rollout-0..34 totals, (3) decode
iteration time by running-batch bucket, sim iteration time vs measured
consecutive-timestamp gaps (DECODE between DECODEs), (4) request latency
p50/p90/p99 split into the t=0 burst (each step's turn 0) and the rest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent


def decode_gaps(iters_csv: Path) -> pd.DataFrame:
    """Measured decode iteration time: the gap to the engine's next scheduler record.

    SGLang overlaps scheduling with the forward, so a record's timestamp is taken
    just after its launch and the gap to the next one is the iteration. Only DECODE
    records between two DECODEs count (a prefill neighbour changes the gap), and
    gaps of 0.2 s or more are the engine idling, not an iteration.
    """
    it = pd.read_csv(iters_csv).sort_values(["rank", "ts"])
    by = it.groupby("rank")
    it["gap"] = by.ts.shift(-1) - it.ts
    it["next_mode"] = by["mode"].shift(-1)
    it["prev_mode"] = by["mode"].shift(1)
    return it[(it["mode"] == "D") & (it.next_mode == "D") & (it.prev_mode == "D") & (it.gap < 0.2)]


def sim_rollout(run: Path) -> dict:
    s = json.loads((run / "summary.json").read_text())
    slo = pq.read_table(run / "raw" / "request_slo.parquet",
                        columns=["request_id", "arrival_time_ms", "ttft_ms", "finish_decode_time_ms",
                                 "num_output_tokens", "retraction_count"]).to_pandas()
    iters = []
    for p in sorted((run / "raw" / "cost_log").glob("worker_main_*.parquet")):
        t = pq.read_table(p, columns=["worker_id", "wall_start_ms", "total_time_ms", "groups"]).to_pandas()
        g = t.pop("groups").map(lambda gs: gs[0])
        t["prefill_tokens"] = g.map(lambda x: x["prefill_tokens"])
        t["decode_n"] = g.map(lambda x: x["decode_request_count"])
        t["decode_kv"] = g.map(lambda x: x["decode_kv_total"])
        iters.append(t)
    return {"summary": s, "slo": slo, "iters": pd.concat(iters, ignore_index=True)}


def q3(x) -> str:
    x = np.asarray(x, dtype=float)
    return "/".join(f"{np.quantile(x, q):.2f}" for q in (0.5, 0.9, 0.99))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep", type=Path)
    ap.add_argument("--measured-dir", type=Path, default=HERE / "data")
    ap.add_argument("--rollouts", type=int, default=35)
    ap.add_argument("--per-turn", type=Path, required=True, help="the handoff's per_turn.csv.gz")
    ap.add_argument("--trace-dir", type=Path, required=True, help="the replayed traces (r{N}.csv)")
    args = ap.parse_args()

    turns = pd.read_csv(args.per_turn)
    rep = pd.read_csv(args.measured_dir / "arms" /
                      "StreamTrainer_RollPackerFaithfulQueue_35rollout_colocate_idle_measurement" / "report.csv")
    rep = rep.set_index("rollout_id")

    rows, lat_rows, sim_iters = [], [], []
    for r in range(args.rollouts):
        run = args.sweep / f"r{r}"
        if not (run / "summary.json").exists():
            continue
        sim = sim_rollout(run)
        mt = turns[turns.rollout_id == r]
        t0 = mt.turn_t_send.min()
        meas_span = mt.turn_t_recv.max() - t0
        slo = sim["slo"]
        sim_span = (slo.finish_decode_time_ms.max()) / 1e3
        rows.append({"rollout": r, "meas_span_s": meas_span, "meas_inference_s": rep.loc[r, "inference_s"],
                     "sim_span_s": sim_span, "sim/meas": sim_span / meas_span,
                     "sim_retractions": int(slo.retraction_count.sum())})
        # request latency: measured client e2e vs simulated arrival->finish, burst (turn 0) apart
        mt = mt.assign(e2e=mt.turn_client_s, burst=mt.turn_idx == 0)
        ids = slo.sort_values("request_id")
        lat_rows.append(pd.DataFrame({"src": "meas", "rollout": r, "burst": mt.burst.values,
                                      "e2e": mt.turn_server_s.values, "ttft": np.nan}))
        # the trace writes rows sorted by (arrival, sample, turn), so dense id order is not turn order;
        # recover turn 0 from the arrival: a turn-0 request arrives at its session's trace time
        trace = pd.read_csv(args.trace_dir / f"r{r}.csv")
        kind = trace.id.str.rsplit("_", n=1).str[1]
        ids = ids.assign(kind=kind.values[ids.request_id.values])
        ids = ids[ids.kind != "d"]  # dispatch placeholders
        lat_rows.append(pd.DataFrame({"src": "sim", "rollout": r, "burst": (ids.kind == "t0").values,
                                      "e2e": ((ids.finish_decode_time_ms - ids.arrival_time_ms) / 1e3).values,
                                      "ttft": (ids.ttft_ms / 1e3).values}))
        it = sim["iters"]
        sim_iters.append(it[(it.prefill_tokens == 0) & (it.decode_n > 0)].assign(rollout=r))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("== per-rollout inference makespan (s): measured first send -> last receive, sim first arrival -> last finish")
    print(df.round(3).to_string(index=False))
    print(f"\nsum meas_span {df.meas_span_s.sum():.1f}  sum sim_span {df.sim_span_s.sum():.1f}  "
          f"ratio {df.sim_span_s.sum() / df.meas_span_s.sum():.4f}  mean|err| {(df['sim/meas'] - 1).abs().mean():.4f}")
    long = df.meas_span_s > 100
    for name, m in (("long (6-turn chain)", long), ("short", ~long)):
        print(f"  {name:20s} n={m.sum():2d} meas {df.meas_span_s[m].mean():6.1f}  sim {df.sim_span_s[m].mean():6.1f}  "
              f"ratio {df.sim_span_s[m].sum() / df.meas_span_s[m].sum():.4f}")

    print("\n== decode iteration time by running batch (ms): sim cost_log vs measured timestamp gaps")
    gaps = decode_gaps(args.measured_dir / "coloc_iters.csv.gz")
    si = pd.concat(sim_iters, ignore_index=True)
    edges = [0, 8, 16, 32, 64, 96, 128, 160, 192, 256]
    si["bucket"] = pd.cut(si.decode_n, edges, right=False)
    gaps["bucket"] = pd.cut(gaps.rbs, edges, right=False)
    a = si.groupby("bucket", observed=True).agg(sim_n=("total_time_ms", "size"), sim_ms=("total_time_ms", "median"),
                                                sim_ctx=("decode_kv", lambda x: 0))
    si["ctx"] = si.decode_kv / si.decode_n
    a["sim_ctx_per_req"] = si.groupby("bucket", observed=True).ctx.median()
    b = gaps.groupby("bucket", observed=True).agg(meas_n=("gap", "size"), meas_ms=("gap", lambda x: 1e3 * x.median()))
    tab = a.drop(columns="sim_ctx").join(b)
    tab["sim/meas"] = tab.sim_ms / tab.meas_ms
    tab["sim_time_share"] = (si.groupby("bucket", observed=True).total_time_ms.sum() / si.total_time_ms.sum())
    print(tab.round(3).to_string())

    lat = pd.concat(lat_rows, ignore_index=True)
    print("\n== engine-side request latency (s) p50/p90/p99: measured turn_server_s vs sim arrival->finish;"
          " measured has no TTFT, sim TTFT shown")
    for burst, name in ((True, "turn 0 (t=0 burst)"), (False, "turns 1+")):
        m = lat[(lat.src == "meas") & (lat.burst == burst)]
        s = lat[(lat.src == "sim") & (lat.burst == burst)]
        print(f"  {name:20s} e2e meas {q3(m.e2e)}  sim {q3(s.e2e)}   sim TTFT {q3(s.ttft)}  (n={len(m)}/{len(s)})")
    df.to_csv(args.sweep / "align_colocated.csv", index=False)
    tab.to_csv(args.sweep / "align_colocated_decode.csv")


if __name__ == "__main__":
    main()
