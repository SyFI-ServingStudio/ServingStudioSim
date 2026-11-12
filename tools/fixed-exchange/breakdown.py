"""Where the dispatch's small-row fixed time goes (8 ranks, graph replay like dev_check).

1. launch floor: an empty kernel at the exchange's grid shapes, N per graph;
2. dispatch / quant_dispatch time vs. grid size (blocks) at small row counts;
3. per-phase timeline from the -DFIXED_EXCHANGE_TIMELINE build: block entry spread, push, wait for
   the peers' rows (receive), finish (epoch atomics), and the gap between consecutive kernels.

    python tools/fixed-exchange/breakdown.py --out breakdown.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dev_check import (  # noqa: E402
    HIDDEN,
    TOP_K,
    graph_time_ms,
    make_inputs,
    max_over_ranks,
    out_planes,
    quant_inputs,
)

ROWS = (0, 8, 32, 64, 128)  # 0: markers only (the cross-GPU signal floor of this kernel shape)
BLOCKS = (None, 64, 32, 16, 8, 4, 2, 1)
PHASES = ("entry_min", "entry_max", "push_end", "receive_end", "exit")


def timeline_phases(torch, ex, run_once, iters: int):
    """Run `iters` calls in one graph; return per-round phase deltas (us) of this rank."""
    import torch.distributed as dist

    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(10):
            run_once()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        for _ in range(iters):
            run_once()
    graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    stream = torch.cuda.current_stream().cuda_stream
    ex.module.timeline_reset(stream)
    dist.barrier()
    graph.replay()
    torch.cuda.synchronize()
    buf = torch.empty(1024 * 8, dtype=torch.int64, device="cuda")
    ex.module.timeline_copy(buf.data_ptr(), stream)
    torch.cuda.synchronize()
    t = buf.view(1024, 8)[:, :5].cpu().tolist()
    rows = [r for r in t if r[4] > 0 and r[0] != -1]
    rows.sort(key=lambda r: r[0])
    out = {"spread": [], "push": [], "receive": [], "finish": [], "gap": [], "total": []}
    for i, (e0, e1, p, rcv, x) in enumerate(rows):
        out["spread"].append((e1 - e0) / 1e3)
        out["push"].append((p - e1) / 1e3)
        out["receive"].append((rcv - p) / 1e3)
        out["finish"].append((x - rcv) / 1e3)
        out["total"].append((x - e0) / 1e3)
        if i:
            out["gap"].append((e0 - rows[i - 1][4]) / 1e3)
    del graph
    return {k: statistics.median(v[2:]) if len(v) > 2 else None for k, v in out.items()}


def worker(rank: int, world: int, args) -> None:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world,
                            device_id=torch.device("cuda", rank))
    from profiling.runners.comm.mnnvl_fixed_exchange import FixedExchange

    max_rows = max(ROWS)
    ex = FixedExchange(rank, world, max_rows, HIDDEN, TOP_K)
    tl = FixedExchange(rank, world, max_rows, HIDDEN, TOP_K, timeline=True)
    gscale = torch.tensor([448.0 * 6.0 / 20.0], dtype=torch.float32, device="cuda")
    res = {"noop": {}, "dispatch": {}, "quant_dispatch": {}, "timeline": {}}
    try:
        stream = lambda: torch.cuda.current_stream().cuda_stream  # noqa: E731
        for blocks in (148, 296, 32, 8, 1):
            res["noop"][blocks] = 1e3 * max_over_ranks(torch, graph_time_ms(
                torch, lambda: ex.module.noop(blocks, 512, stream()), 1))
        for rows in ROWS:
            counts = [rows] * world
            for e in (ex, tl):
                e.set_counts(counts)
            inp = make_inputs(torch, rank, rows, 1)
            out = out_planes(torch, ex, rows * world)
            x, w, ids = quant_inputs(torch, rank, rows, 1)
            nbytes = rows * world * ex.row.total
            for blocks in BLOCKS:
                key = f"{rows}/{blocks or 'default'}"
                res["dispatch"][key] = 1e3 * max_over_ranks(torch, graph_time_ms(
                    torch, lambda: ex.dispatch(*inp, out, blocks=blocks), nbytes))
                res["quant_dispatch"][key] = 1e3 * max_over_ranks(torch, graph_time_ms(
                    torch, lambda: ex.quant_dispatch(x, gscale, w, ids, out, blocks=blocks),
                    nbytes))
            for blocks in (None, 8):
                for leg, run in (
                    ("dispatch", lambda: tl.dispatch(*inp, out, blocks=blocks)),
                    ("quant_dispatch",
                     lambda: tl.quant_dispatch(x, gscale, w, ids, out, blocks=blocks)),
                ):
                    phases = timeline_phases(torch, tl, run, 64)
                    gathered = [None] * world
                    dist.all_gather_object(gathered, phases)
                    res["timeline"][f"{leg} {rows}/{blocks or 'default'}"] = {
                        k: {"rank0": gathered[0][k],
                            "max": max(g[k] for g in gathered if g[k] is not None)}
                        for k in phases if phases[k] is not None
                    }
            if rank == 0:
                print(json.dumps({"rows": rows,
                                  "dispatch": {k: round(v, 2) for k, v in res["dispatch"].items()
                                               if k.startswith(f"{rows}/")},
                                  "quant_dispatch": {k: round(v, 2) for k, v in
                                                     res["quant_dispatch"].items()
                                                     if k.startswith(f"{rows}/")}}), flush=True)
    finally:
        ex.close()
        tl.close()
    if rank == 0:
        print("noop us", json.dumps(res["noop"]), flush=True)
        for k, v in res["timeline"].items():
            print("timeline", k, json.dumps({p: [round(x["rank0"], 2), round(x["max"], 2)]
                                             for p, x in v.items()}), flush=True)
        if args.out:
            Path(args.out).write_text(json.dumps(res, indent=1))
    dist.destroy_process_group()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import torch
    import torch.multiprocessing as mp

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(29500 + os.getpid() % 1000)
    mp.spawn(worker, args=(torch.cuda.device_count(), args), nprocs=torch.cuda.device_count(),
             join=True)


if __name__ == "__main__":
    main()
