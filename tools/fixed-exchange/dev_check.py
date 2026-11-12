"""8-rank correctness and timing check of the fixed-pattern MoE exchange kernels.

Runs inside the vLLM profiler image (FlashInfer 0.7.3, vLLM quant ops) with the IMEX channel,
one process per GPU (see run_dev.sh). Every rank regenerates every rank's inputs from per-rank
seeds, so the references need no communication.

    python tools/fixed-exchange/dev_check.py --check --time --rows 8,16,128 --out out.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

HIDDEN = 4096
TOP_K = 8
NUM_EXPERTS = 288
SENTINEL = 0xFFFFFFFF


def canon_fp4_words(w):
    """Every -0 nibble -> +0 (torch int64 view of uint32 words)."""
    mag = w & 0x77777777
    nz = (mag | (mag >> 1) | (mag >> 2)) & 0x11111111
    return w & (0x77777777 | (nz << 3))


def canon_word(w):
    return w.masked_fill(w == SENTINEL, 0x7FFFFFFF)


def words(t):
    import torch

    return t.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF


def make_inputs(torch, rank: int, rows: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed * 1000 + rank)
    payload = torch.randint(0, 256, (rows, HIDDEN // 2), generator=g, device="cuda", dtype=torch.int32).to(torch.uint8)
    if rows:
        # Words that collide with the sentinel before canonicalization: eight -0 / eight -6.
        payload.view(torch.int32)[0, 0] = int(0x88888888 - (1 << 32))
        payload.view(torch.int32)[rows - 1, 1] = -1
    scales = torch.randint(0, 0x7F, (rows, HIDDEN // 16), generator=g, device="cuda", dtype=torch.int32).to(torch.uint8)
    weights = torch.rand((rows, TOP_K), generator=g, device="cuda", dtype=torch.float32)
    if rows:
        weights.view(torch.int32)[0, 0] = -1  # a NaN word equal to the sentinel
    ids = torch.randint(0, NUM_EXPERTS, (rows, TOP_K), generator=g, device="cuda", dtype=torch.int32)
    return payload, scales, weights, ids


def canonical_planes(torch, payload, scales, weights, ids):
    p = canon_fp4_words(words(payload))
    s = canon_word(words(scales))
    w = canon_word(words(weights))
    return p, s, w, words(ids)


def make_partials(torch, rank: int, total: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed * 7919 + rank)
    return (torch.randn((total, HIDDEN), generator=g, device="cuda") * 4).to(torch.bfloat16)


# ----------------------------------------------------------------- timing (a2a runner's basis)
_GRAPH_BYTE_BUDGET = 8.0 * 1024**3
_REPLAYS = 20


def iters_per_graph(bytes_per_call: int) -> int:
    return max(4, min(100, int(_GRAPH_BYTE_BUDGET // max(1, bytes_per_call))))


def graph_time_ms(torch, run_once, bytes_per_call: int, warmup: int = 20) -> float:
    import torch.distributed as dist

    n = iters_per_graph(bytes_per_call)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            run_once()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(n):
            run_once()
    for _ in range(2):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(_REPLAYS):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / (_REPLAYS * n)
    del graph
    return ms


def max_over_ranks(torch, value: float) -> float:
    import torch.distributed as dist

    t = torch.tensor([value], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def worker(rank: int, world: int, args) -> None:
    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    torch.cuda.set_device(rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world,
                            device_id=torch.device("cuda", rank))
    from profiling.runners.comm.mnnvl_fixed_exchange import FixedExchange

    rows_list = [int(r) for r in args.rows.split(",")]
    max_rows = max(rows_list + [args.max_rows])
    t0 = time.time()
    ex = FixedExchange(rank, world, max_rows, HIDDEN, TOP_K)
    if rank == 0:
        print(f"workspace up in {time.time() - t0:.1f}s, {ex.layout.total / 2**20:.0f} MiB/rank, "
              f"max blocks {ex.max_blocks}", flush=True)
    results = []
    try:
        if args.check:
            check(torch, ex, rank, world, max_rows, args)
        if args.time:
            for rows in rows_list:
                results.append(time_shape(torch, ex, rank, world, rows, args))
                if rank == 0:
                    print(json.dumps(results[-1]), flush=True)
    finally:
        ex.close()
        dist.destroy_process_group()
    if rank == 0 and args.out:
        Path(args.out).write_text(json.dumps(results, indent=1))


def gathered(torch, world, counts, seed):
    planes = [[], [], [], []]
    for r in range(world):
        for i, t in enumerate(canonical_planes(torch, *make_inputs(torch, r, counts[r], seed))):
            planes[i].append(t)
    return [torch.cat(p) for p in planes]


def out_planes(torch, ex, total):
    row = ex.row
    return [torch.empty((total, b), dtype=torch.uint8, device="cuda") for b in
            (row.payload, row.scales, row.weights, row.ids)]


def check(torch, ex, rank, world, max_rows, args):
    import torch.distributed as dist

    cases = [[1] * world, [8] * world, [128] * world, [max_rows] * world,
             [(3 * r + 1) % 17 for r in range(world)], [0] + [5] * (world - 1),
             [max_rows - r for r in range(world)]]
    failures = []
    for case_index, counts in enumerate(cases):
        ex.set_counts(counts)
        total = sum(counts)
        for rep in range(3):  # several rounds: both landing sets, re-arming
            seed = 100 * case_index + rep
            inp = make_inputs(torch, rank, counts[rank], seed)
            out = out_planes(torch, ex, total)
            ex.dispatch(*inp, out)
            torch.cuda.synchronize()
            ref = gathered(torch, world, counts, seed)
            for name, o, r in zip(("payload", "scales", "weights", "ids"), out, ref):
                if not torch.equal(words(o), r):
                    failures.append(f"dispatch {counts} rep {rep} {name}")
            # combine
            partials = make_partials(torch, rank, total, seed)
            o = torch.empty((counts[rank], HIDDEN), dtype=torch.bfloat16, device="cuda")
            ex.combine(partials, o)
            torch.cuda.synchronize()
            acc = torch.zeros((counts[rank], HIDDEN), dtype=torch.float32, device="cuda")
            first = sum(counts[:rank])
            for r in range(world):
                acc += make_partials(torch, r, total, seed)[first:first + counts[rank]].float()
            if not torch.equal(o, acc.to(torch.bfloat16)):
                failures.append(f"combine {counts} rep {rep}: max err "
                                f"{(o.float() - acc).abs().max().item()}")
            # pull combine: partials in the mapped set the round reads
            ex.partials(ex.next_pull_set, total).copy_(partials)
            torch.cuda.synchronize()
            dist.barrier()
            o2 = torch.empty_like(o)
            ex.pull_combine(o2)
            torch.cuda.synchronize()
            if not torch.equal(o2, acc.to(torch.bfloat16)):
                diff = (o2.float() - acc.to(torch.bfloat16).float()).abs()
                failures.append(f"pull_combine {counts} rep {rep}: {(diff > 0).float().mean().item():.4f} "
                                f"of elements differ, max {diff.max().item():.3g}, set {ex.next_pull_set - 0}")
            dist.barrier()  # nobody rewrites partials while a peer still reads them
            # quant dispatch vs vLLM's own quantization
            failures += check_quant(torch, ex, rank, world, counts, seed)
    all_failures = [None] * world
    dist.all_gather_object(all_failures, failures)
    if rank == 0:
        flat = [f"rank {r}: {f}" for r, fs in enumerate(all_failures) for f in fs]
        print("CHECK", "PASS" if not flat else "FAIL", len(cases) * 3, "rounds per leg", flush=True)
        import collections
        kinds = collections.Counter(f.split(": ", 1)[1].split(" ")[0] for f in flat)
        print("  failures by leg:", dict(kinds), flush=True)
        for f in flat[:12]:
            print("  ", f, flush=True)


def quant_inputs(torch, rank, rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed * 31 + rank)
    x = (torch.randn((rows, HIDDEN), generator=g, device="cuda") * 3).to(torch.bfloat16)
    if rows:
        x[0, :32] = 0  # an all-zero block
        x[rows - 1, 16:32] = -1e-9  # tiny negatives: -0 nibbles
    w = torch.rand((rows, TOP_K), generator=g, device="cuda", dtype=torch.float32)
    ids = torch.randint(0, NUM_EXPERTS, (rows, TOP_K), generator=g, device="cuda", dtype=torch.int32)
    return x, w, ids


def vllm_quant(torch, x, gscale):
    from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input

    q, s = moe_kernel_quantize_input(x, gscale, quant_dtype="nvfp4", per_act_token_quant=False,
                                     block_shape=None, is_scale_swizzled=False)
    return q.view(torch.uint8).reshape(x.shape[0], HIDDEN // 2), s.view(torch.uint8).reshape(x.shape[0], HIDDEN // 16)


def check_quant(torch, ex, rank, world, counts, seed):
    failures = []
    gscale = torch.tensor([448.0 * 6.0 / 20.0], dtype=torch.float32, device="cuda")
    total = sum(counts)
    x, w, ids = quant_inputs(torch, rank, counts[rank], seed)
    out = out_planes(torch, ex, total)
    ex.quant_dispatch(x, gscale, w, ids, out)
    torch.cuda.synchronize()
    ref = [[], [], [], []]
    for r in range(world):
        xr, wr, idr = quant_inputs(torch, r, counts[r], seed)
        if counts[r]:
            q, s = vllm_quant(torch, xr, gscale)
        else:
            q = torch.empty((0, HIDDEN // 2), dtype=torch.uint8, device="cuda")
            s = torch.empty((0, HIDDEN // 16), dtype=torch.uint8, device="cuda")
        for i, t in enumerate(canonical_planes(torch, q, s, wr, idr)):
            ref[i].append(t)
    for name, o, r in zip(("payload", "scales", "weights", "ids"), out, ref):
        r = torch.cat(r)
        if not torch.equal(words(o), r):
            bad = (words(o) != r).sum().item()
            failures.append(f"quant_dispatch {counts} {name}: {bad} words differ")
    return failures


def time_shape(torch, ex, rank, world, rows, args):
    counts = [rows] * world
    ex.set_counts(counts)
    total = rows * world
    inp = make_inputs(torch, rank, rows, 1)
    out = out_planes(torch, ex, total)
    row_bytes = ex.row.total
    res = {"rows_per_rank": rows, "dispatch_row_bytes": row_bytes, "combine_row_bytes": HIDDEN * 2}
    blocks = args.blocks
    res["dispatch_ms"] = max_over_ranks(torch, graph_time_ms(
        torch, lambda: ex.dispatch(*inp, out, blocks=blocks), rows * world * row_bytes))
    x, w, ids = quant_inputs(torch, rank, rows, 1)
    gscale = torch.tensor([448.0 * 6.0 / 20.0], dtype=torch.float32, device="cuda")
    res["quant_dispatch_ms"] = max_over_ranks(torch, graph_time_ms(
        torch, lambda: ex.quant_dispatch(x, gscale, w, ids, out, blocks=blocks),
        rows * world * row_bytes))
    try:
        res["vllm_quant_ms"] = max_over_ranks(torch, graph_time_ms(
            torch, lambda: vllm_quant(torch, x, gscale), rows * HIDDEN * 2))
    except Exception as exc:  # noqa: BLE001
        res["vllm_quant_error"] = repr(exc)[:200]
    partials = make_partials(torch, rank, total, 1)
    o = torch.empty((rows, HIDDEN), dtype=torch.bfloat16, device="cuda")
    res["combine_ms"] = max_over_ranks(torch, graph_time_ms(
        torch, lambda: ex.combine(partials, o, blocks=blocks), total * HIDDEN * 2))
    for s in (0, 1):
        ex.partials(s, total).copy_(partials)
    torch.cuda.synchronize()
    import torch.distributed as dist

    dist.barrier()
    res["pull_combine_ms"] = max_over_ranks(torch, graph_time_ms(
        torch, lambda: ex.pull_combine(o, blocks=blocks), total * HIDDEN * 2))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--rows", default="8,16,32,64,128,256,384,512,640,1024")
    ap.add_argument("--max-rows", type=int, default=1024)
    ap.add_argument("--blocks", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import torch
    import torch.multiprocessing as mp

    world = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(29500 + os.getpid() % 1000)
    mp.spawn(worker, args=(world, args), nprocs=world, join=True)


if __name__ == "__main__":
    main()
