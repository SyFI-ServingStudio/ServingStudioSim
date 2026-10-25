"""MoE rows: logs/bench/moe.jsonl (bench_moe.py; device time).

Per-rank shapes from the bench row's (ep, tp): local_experts = 512 / ep, intermediate_size =
640 / tp. bench_moe shards every kind with `shard(ep, tp, rank)` and `shared="split"`, so the
shared expert's slice is split over all ep * tp ranks: shared intermediate = 640 / (ep * tp)
(`shared_expert` rows have ep 1, i.e. 640 / tp). Routed and block rows are kept only for
`routing == "real"` (real layer-0 routing on the critical-path rank); the router and the shared
expert do not depend on routing (bench `routing == "none"`). `moe_block` impl
"torch_topk_first/<routed>/torch" maps to backend neuron_<routed>.
"""

from __future__ import annotations

from pathlib import Path

from import_bench import ImportRow, device_ms, jsonl

from families.costs import cost

H, E, K, INTER, SHARED = 2560, 512, 10, 640, 640
ROUTED_IMPLS = {
    "auto", "nki", "nki_tkg", "nki_tkg_all", "nki_cte", "nki_cte_cf2",
    "torch_gather_w", "torch_dense", "torch_gather", "torch_onehot",
}  # fmt: skip
ROUTER_IMPLS = {"torch", "torch_topk_first", "torch_bf16", "nki_topk"}
SHARED_IMPLS = {"torch", "nki"}
BLOCK_ROUTED = {"auto", "nki"}


def map_row(r: dict) -> tuple[str, str, dict] | None:
    """(table, backend, args) for one bench row, or None when it maps to no registered row."""
    kind, impl, T = r.get("kind"), r.get("impl"), r.get("tokens")
    ep, tp = int(r.get("ep") or 1), int(r.get("tp") or 1)
    if T is None or E % ep or INTER % tp:
        return None
    T = int(T)
    if kind == "moe_router":
        if impl not in ROUTER_IMPLS:
            return None
        return (
            "moe_router",
            f"neuron_{impl}",
            dict(num_tokens=T, hidden_size=H, num_experts=E, top_k=K, dtype="bf16"),
        )
    if kind == "shared_expert":
        if impl not in SHARED_IMPLS or SHARED % (ep * tp):
            return None
        args = dict(
            num_tokens=T, hidden_size=H, intermediate_size=SHARED // (ep * tp), dtype="bf16"
        )
        return "moe_shared_expert", f"neuron_{impl}", args
    if r.get("routing") != "real":
        return None
    routed = dict(local_experts=E // ep, intermediate_size=INTER // tp)
    if kind == "routed_experts":
        if impl not in ROUTED_IMPLS:
            return None
        args = dict(num_tokens=T, hidden_size=H, num_experts=E, top_k=K, **routed, dtype="bf16")
        return "moe_routed_experts", f"neuron_{impl}", args
    if kind == "moe_block":
        parts = str(impl).split("/")
        if len(parts) != 3 or parts[0] != "torch_topk_first" or parts[2] != "torch":
            return None
        if parts[1] not in BLOCK_ROUTED or SHARED % (ep * tp):
            return None
        args = dict(
            num_tokens=T,
            hidden_size=H,
            num_experts=E,
            top_k=K,
            **routed,
            shared_intermediate_size=SHARED // (ep * tp),
            dtype="bf16",
        )
        return "moe_block", f"neuron_{parts[1]}", args
    return None


def rows(bench_dir: Path):
    name = "moe.jsonl"
    for line, r in jsonl(bench_dir / name):
        ms = device_ms(r)
        mapped = map_row(r) if ms is not None else None
        if mapped is None:
            continue
        table, backend, args = mapped
        yield ImportRow(table, backend, args, ms, f"{name}:{line}", *cost(table, args))
