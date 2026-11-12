"""bf16 group all-gather / reduce-scatter vs FlashInfer MNNVL two-shot all-reduce, b = 2/4/8.

An attention-TP group of b ranks gathers its members' bf16 rows (T rows per rank, hidden 4096,
8192 B per row) and returns each owner the sum of the group's partial outputs. The 8 GPUs run
8/b groups at once (as in serving). Per group size, row count and leg:

  ag_push   fixed-pattern push all-gather (dispatch kernel, one bf16 plane, Lamport arrival)
  ag_pull   reader-side all-gather (rows already in each rank's mapped buffer; ready flags)
  rs_push   fixed-pattern push reduce-scatter (combine kernel)
  rs_pull   reader-side reduce-scatter (partials already in the mapped buffer)
  ar        FlashInfer allreduce_fusion kAllReduce, MNNVL, forced two-shot, workspace sized to
            the request ("unbudgeted"), over the group's b*T rows (the same total bytes)
  push0     the push kernel with 0 rows (markers only: the signal floor of the push legs)
  pull0     the pull kernel with 0 rows (ready-flag handshake only)
  noop      empty kernel, 148 x 512 threads

Two timing methods, both reduced to the slowest rank per sample:
  graph   back-to-back CUDA-graph replay (20 calls per graph), one sample per replay;
  cupti   per call: flush L2 (256 MiB memset), a 0-row push exchange as group barrier, the
          call; CUPTI kernel intervals via torch.profiler, the union of the call's kernels.
Medians and IQR over samples. Correctness (bit-exact for the exchange, fp32-sum tolerance for
the all-reduce) against torch, on equal, uneven and zero-row cases, including back-to-back
bursts without a host sync.

    python tools/fixed-exchange/kda_group.py --out kda_group.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HIDDEN = 4096
ROWS = (8, 16, 32, 64, 128, 256, 384, 656)
MAX_ROWS = 656
GROUPS = (2, 4, 8)
CALLS_PER_GRAPH = 20
REPLAYS = 50
CUPTI_ITERS = 40
OPS = ("ag_push", "ag_pull", "rs_push", "rs_pull", "ar", "push0", "pull0", "noop")


def rows_of(torch, rank: int, count: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed * 1009 + rank)
    return (torch.randn((count, HIDDEN), generator=g, device="cuda") * 2).to(torch.bfloat16)


def partials_of(torch, rank: int, total: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed * 7919 + rank)
    return (torch.randn((total, HIDDEN), generator=g, device="cuda") * 4).to(torch.bfloat16)


def quantiles(values):
    q = statistics.quantiles(values, n=4)
    return {"median": statistics.median(values), "p25": q[0], "p75": q[2], "n": len(values)}


class Group:
    """One rank's view of its attention-TP group: exchange, barrier exchange, AR workspace."""

    def __init__(self, torch, dist, rank: int, world: int, b: int):
        import flashinfer.comm as fcomm
        from flashinfer.comm.mnnvl import TorchDistBackend

        from profiling.runners.comm.mnnvl_fixed_exchange import DispatchRow, FixedExchange

        self.torch, self.dist, self.b = torch, dist, b
        self.fcomm = fcomm
        cpu_groups = [
            dist.new_group(list(range(i, i + b)), backend="gloo") for i in range(0, world, b)
        ]
        self.cpu = cpu_groups[rank // b]
        self.grank = rank % b
        row = DispatchRow.bf16(HIDDEN)
        self.ex = FixedExchange(self.grank, b, MAX_ROWS, HIDDEN, 8, group=self.cpu, row=row)
        self.bar = FixedExchange(self.grank, b, 1, HIDDEN, 8, group=self.cpu, row=row)
        self.bar.set_counts([0] * b)
        self.empty = torch.empty((1, HIDDEN), dtype=torch.bfloat16, device="cuda")
        self.ws = fcomm.create_allreduce_fusion_workspace(
            backend="mnnvl",
            world_size=b,
            rank=self.grank,
            max_token_num=b * MAX_ROWS,
            hidden_dim=HIDDEN,
            dtype=torch.bfloat16,
            comm_backend=TorchDistBackend(group=self.cpu),
            force_oneshot_support=False,
            group=self.cpu,
        )

    def barrier(self):
        self.bar.all_gather(self.empty, self.empty)

    def ar(self, x):
        return self.fcomm.allreduce_fusion(
            input=x,
            workspace=self.ws,
            pattern=self.fcomm.AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=True,
            trigger_completion_at_end=x.shape[0] > 16,
            use_oneshot=False,
        )

    def close(self):
        self.ws.destroy()
        self.bar.close()
        self.ex.close()


def setup(gr: Group, counts, seed):
    """Inputs of one case; the pull legs' sources written into the mapped set 0 (gathered rows
    for the gather, partials for the scatter, which run at different times: see `stage`)."""
    torch = gr.torch
    b, r = gr.b, gr.grank
    total = sum(counts)
    first = sum(counts[:r])
    s = {
        "counts": counts,
        "total": total,
        "first": first,
        "x": rows_of(torch, r, counts[r], seed),
        "partials": partials_of(torch, r, total, seed),
        "gathered": torch.empty((total, HIDDEN), dtype=torch.bfloat16, device="cuda"),
        "owned": torch.empty((counts[r], HIDDEN), dtype=torch.bfloat16, device="cuda"),
        "ar_in": partials_of(torch, r, max(total, 1), seed + 17),
    }
    s["want_gather"] = torch.cat([rows_of(torch, q, counts[q], seed) for q in range(b)])
    acc = torch.zeros((counts[r], HIDDEN), dtype=torch.float32, device="cuda")
    for q in range(b):
        acc += partials_of(torch, q, total, seed)[first : first + counts[r]].float()
    s["want_owned"] = acc.to(torch.bfloat16)
    ar_acc = torch.zeros_like(s["ar_in"], dtype=torch.float32)
    for q in range(b):
        ar_acc += partials_of(torch, q, max(total, 1), seed + 17).float()
    s["want_ar"] = ar_acc
    return s


def stage(gr: Group, s, leg: str):
    """Put the pull legs' source rows into this rank's mapped set 0, then a group barrier."""
    torch = gr.torch
    if leg == "ag_pull" and s["total"]:
        src = gr.ex.partials(0, s["total"])
        src[s["first"] : s["first"] + s["counts"][gr.grank]].copy_(s["x"])
    elif leg == "rs_pull" and s["total"]:
        gr.ex.partials(0, s["total"]).copy_(s["partials"])
    torch.cuda.synchronize()
    gr.dist.barrier(group=gr.cpu)


def runner(gr: Group, s, op: str):
    ex = gr.ex
    if op == "ag_push":
        return lambda: ex.all_gather(s["x"], s["gathered"])
    if op == "ag_pull":
        return lambda: ex.pull_gather(s["gathered"], source_set=0)
    if op == "rs_push":
        return lambda: ex.combine(s["partials"], s["owned"])
    if op == "rs_pull":
        return lambda: ex.pull_combine(s["owned"], partials_set=0)
    if op == "ar":
        x = s["ar_in"]
        return lambda: gr.ar(x)
    if op == "push0":
        return gr.barrier
    if op == "pull0":
        return lambda: gr.bar.pull_combine(gr.empty, partials_set=0)
    if op == "noop":
        stream = gr.torch.cuda.current_stream
        return lambda: ex.module.noop(148, 512, stream().cuda_stream)
    raise ValueError(op)


def check(gr: Group, counts, seed):
    """Failures of one case (every leg, single call and a back-to-back burst)."""
    torch = gr.torch
    s = setup(gr, counts, seed)
    gr.ex.set_counts(counts)
    bad = []
    for leg in ("ag_push", "ag_pull", "rs_push", "rs_pull"):
        stage(gr, s, leg)
        for calls in (1, 6):
            for key in ("gathered", "owned"):
                s[key].zero_()
            run = runner(gr, s, leg)
            for _ in range(calls):
                run()
            torch.cuda.synchronize()
            got, want = (
                (s["gathered"], s["want_gather"])
                if leg.startswith("ag")
                else (s["owned"], s["want_owned"])
            )
            if not torch.equal(got, want):
                bad.append(f"{leg} {counts} x{calls}")
        gr.dist.barrier(group=gr.cpu)
    if s["total"]:
        out = gr.ar(s["ar_in"])
        torch.cuda.synchronize()
        err = (out.float() - s["want_ar"]).abs().max().item()
        tol = 0.02 * s["want_ar"].abs().max().item() + 0.1
        if err > tol:
            bad.append(f"ar {counts} max err {err:.3g} > {tol:.3g}")
    gr.dist.barrier(group=gr.cpu)
    return bad


def graph_samples(gr: Group, run):
    torch = gr.torch
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(10):
            run()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    gr.dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(CALLS_PER_GRAPH):
            run()
    graph.replay()
    torch.cuda.synchronize()
    gr.dist.barrier()
    events = [torch.cuda.Event(enable_timing=True) for _ in range(REPLAYS + 1)]
    events[0].record()
    for i in range(REPLAYS):
        graph.replay()
        events[i + 1].record()
    torch.cuda.synchronize()
    del graph
    return [1e3 * events[i].elapsed_time(events[i + 1]) / CALLS_PER_GRAPH for i in range(REPLAYS)]


def cupti_samples(gr: Group, runs):
    """{op: [us per call]} from one profiler session: per call flush, barrier, call."""
    torch = gr.torch
    from torch.profiler import ProfilerActivity, profile

    flush = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    for _, run in runs:
        for _ in range(3):
            flush.zero_()
            gr.barrier()
            run()
    torch.cuda.synchronize()
    gr.dist.barrier()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _, run in runs:
            for _ in range(CUPTI_ITERS):
                flush.zero_()
                gr.barrier()
                run()
        torch.cuda.synchronize()
    kernels = sorted(
        (
            e
            for e in prof.events()
            if e.device_type.name == "CUDA"
            and e.time_range.end > 0
            and "memcpy" not in e.name.lower()
            and "memset" not in e.name.lower()
        ),
        key=lambda e: e.time_range.start,
    )
    iters, cur, state = [], None, "idle"
    for e in kernels:
        if "FillFunctor" in e.name:
            if cur:
                iters.append(cur)
            cur, state = [], "fill"
        elif state == "fill":
            state = "call"  # the barrier kernel
        elif state == "call":
            cur.append((e.time_range.start, e.time_range.end))
    if cur:
        iters.append(cur)
    out = {}
    if len(iters) != CUPTI_ITERS * len(runs):
        return {op: None for op, _ in runs} | {"_parse": f"{len(iters)} iterations"}
    for i, (op, _) in enumerate(runs):
        chunk = iters[i * CUPTI_ITERS : (i + 1) * CUPTI_ITERS]
        out[op] = [max(e for _, e in c) - min(s for s, _ in c) if c else 0.0 for c in chunk]
    return out


def worker(rank: int, world: int, args) -> None:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", rank=rank, world_size=world, device_id=torch.device("cuda", rank)
    )
    results = {"failures": [], "timing": []}
    rows_list = [int(r) for r in args.rows.split(",")]
    for b in [int(x) for x in args.groups.split(",")]:
        gr = Group(torch, dist, rank, world, b)
        try:
            cases = []
            for t in (8, 656):
                cases += [
                    [t] * b,
                    [t] + [0] * (b - 1),
                    [0] * (b - 1) + [t],
                    [max(t - 3 * q, 0) for q in range(b)],
                ]
            cases.append([0] * b)
            for i, counts in enumerate(cases):
                results["failures"] += [f"b{b} {f}" for f in check(gr, counts, 100 + i)]
            timed = [[t] * b for t in rows_list] + [
                [rows_list[0]] + [0] * (b - 1),
                [64, 64, 0, 0, 32, 32, 16, 0][:b],
            ]
            for counts in timed:
                s = setup(gr, counts, 1)
                gr.ex.set_counts(counts)
                runs = []
                rec = {"b": b, "counts": counts, "graph": {}, "cupti": {}}
                for op in OPS:
                    if op == "ar" and s["total"] == 0:
                        continue
                    stage(gr, s, op)
                    run = runner(gr, s, op)
                    rec["graph"][op] = graph_samples(gr, run)
                    runs.append((op, run))
                rec["cupti"] = cupti_samples(gr, runs)
                rec["load"] = os.getloadavg()[0]
                gathered = [None] * world
                dist.all_gather_object(gathered, rec)
                if rank == 0:
                    merged = {"b": b, "counts": counts, "load": rec["load"]}
                    for method in ("graph", "cupti"):
                        merged[method] = {}
                        for op in rec[method]:
                            series = [g[method].get(op) for g in gathered]
                            if op.startswith("_") or any(x is None for x in series):
                                merged[method][op] = None
                                continue
                            per_sample = [max(v) for v in zip(*series)]
                            merged[method][op] = quantiles(per_sample)
                    results["timing"].append(merged)
                    print(
                        json.dumps(
                            {"b": b, "T": counts[0], "counts": counts}
                            | {
                                f"{m}:{op}": round(v["median"], 2)
                                for m in ("graph", "cupti")
                                for op, v in merged[m].items()
                                if v
                            }
                        ),
                        flush=True,
                    )
        finally:
            gr.close()
    all_failures = [None] * world
    dist.all_gather_object(all_failures, results["failures"])
    if rank == 0:
        flat = [f"rank {r}: {f}" for r, fs in enumerate(all_failures) for f in fs]
        print("CHECK", "PASS" if not flat else "FAIL", flush=True)
        for f in flat[:20]:
            print("  ", f, flush=True)
        if args.out:
            Path(args.out).write_text(
                json.dumps({"timing": results["timing"], "failures": flat}, indent=1)
            )
    dist.destroy_process_group()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default=",".join(map(str, ROWS)))
    ap.add_argument("--groups", default=",".join(map(str, GROUPS)))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import torch
    import torch.multiprocessing as mp

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(29500 + os.getpid() % 1000)
    world = torch.cuda.device_count()
    mp.spawn(worker, args=(world, args), nprocs=world, join=True)


if __name__ == "__main__":
    main()
