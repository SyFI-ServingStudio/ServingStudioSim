"""Balance a pipeline-parallel layer split from finished PP runs.

A pipeline's steady-state throughput is set by its busiest stage, so the split
(`layer_partition`, vLLM's `VLLM_PP_LAYER_PARTITION`) should even out stage
time. This tool measures what one layer of each kind costs on the run's own
workload, then searches contiguous splits for the smallest bottleneck.

Inputs are PP run directories (a launcher `log_dir` with `raw/cost_manifest/`
and the analyzer's `reports/kernel_time_share_report.json`). Use runs whose
load keeps every stage busy (the saturated point of a sweep), since a split
balanced for an idle pipeline balances nothing.

## How it measures

- Each stage's cost manifest folds its layers into `Scale` groups, one per layer
  kind, and labels each group with the layers it holds
  (`dsa_moe x3 (Scale 3) layers [3, 7, 11]`). Leaves outside every group are
  fixed per stage (embedding on the first stage, lm_head on the last).
- The kernel-time report gives each worker's time per leaf position, so a
  group's time on a stage divided by its layer count is that kind's per-layer
  cost. Every stage runs the same microbatches over the same contexts, so this
  cost does not depend on which stage runs the layer. The tool checks that
  and prints the spread.
- Predicted stage time is then the sum over its layers plus its fixed leaves.
  The tool reproduces each run's own stage times as a check before it searches.

It models compute only: no activation transfer and no pipeline bubble, so read
its gain as the change in the busiest stage's kernel time, then confirm with a
real run of the chosen split. On GLM-5.3-Flash NVFP4 TraceLab saturated runs
it scored vLLM's PP4 split 3.4% behind [12, 11, 11, 11]; the runs measured
2.8-2.9%, and a split it scored as tied measured within 0.2%.

## Usage

    uv run python tools/pp-layer-balance/pp_layer_balance.py <run_dir>... \\
        [--pp-size 8] [--each-stage-needs 'dsa_*'] [--top 5] [--json out.json]

- Several runs (e.g. the saturated point of two trace policies) are weighed
  equally: each run's costs are normalized to its own total before summing.
- `--pp-size` targets another pipeline depth from the same runs (default: the
  runs' own); per-layer costs carry over.
- `--each-stage-needs GLOB` rejects splits with a stage holding no layer whose
  kind matches GLOB (repeatable). GLM-5.3-Flash needs `'dsa_*'`: vLLM's hybrid
  cache grouping cannot place KDA state on a stage without a DSA layer.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

STAGE_RE = re.compile(r"pipeline stage (\d+) of (\d+)")
GROUP_RE = re.compile(r"^(?P<kind>\S+) x(?P<n>\d+) \(Scale \d+\) layers \[(?P<layers>[\d, ]*)\]$")


@dataclass
class Stage:
    index: int
    layers: dict[str, list[int]] = field(default_factory=dict)  # kind -> layers this stage runs
    group_of_leaf: dict[str, str] = field(default_factory=dict)  # leaf name -> kind
    fixed_leaves: set[str] = field(default_factory=set)


@dataclass
class Run:
    path: Path
    num_stages: int
    stages: list[Stage]
    layer_cost: dict[str, float]  # kind -> ms per layer
    fixed_cost: dict[str, float]  # "first" / "last" / "every" -> ms
    stage_ms: list[float]  # each worker's measured kernel time
    spread: dict[str, float]  # kind -> max relative deviation of per-layer cost across stages
    kind_of_layer: dict[int, str]


def parse_manifest(path: Path) -> Stage:
    section = json.loads(path.read_text())["sections"]
    if len(section) != 1:
        sys.exit(f"{path}: expected one manifest section, found {len(section)}")
    nodes, labels, slots = section[0]["nodes"], section[0]["node_labels"], section[0]["slots"]
    root = STAGE_RE.search(labels[0] or "")
    if not root:
        sys.exit(f"{path}: root label is not a pipeline stage: {labels[0]!r}")
    stage = Stage(index=int(root.group(1)))

    def children(i):
        node = nodes[i]
        ((tag, body),) = node.items()
        if tag == "Leaf":
            return []
        span = body["children"] if isinstance(body, dict) else body
        return range(span["start"], span["end"])

    def leaves(i):
        node = nodes[i]
        if "Leaf" in node:
            yield slots[node["Leaf"]]["name"]
        for c in children(i):
            yield from leaves(c)

    def walk(i, kind):
        label = labels[i] or ""
        group = GROUP_RE.match(label)
        if "Scale" in nodes[i] and label and not group:
            sys.exit(
                f"{path}: Scale group label carries no layer list: {label!r} "
                "(rerun with a simulator whose PP stage labels name their layers)"
            )
        if group:
            kind = group["kind"]
            layers = [int(x) for x in group["layers"].replace(" ", "").split(",") if x]
            if len(layers) != int(group["n"]):
                sys.exit(f"{path}: {label!r}: layer list disagrees with its count")
            stage.layers.setdefault(kind, []).extend(layers)
            for leaf in leaves(i):
                if stage.group_of_leaf.setdefault(leaf, kind) != kind:
                    sys.exit(
                        f"{path}: leaf {leaf} sits in groups {stage.group_of_leaf[leaf]} "
                        f"and {kind}; the report pools same-name positions, so their costs "
                        "cannot be split"
                    )
            return
        node = nodes[i]
        if "Leaf" in node:
            stage.fixed_leaves.add(slots[node["Leaf"]]["name"])
        for c in children(i):
            walk(c, kind)

    walk(0, None)
    clash = stage.fixed_leaves & set(stage.group_of_leaf)
    if clash:
        sys.exit(f"{path}: leaves both fixed and in a layer group: {sorted(clash)[:3]}")
    return stage


def load_run(path: Path) -> Run:
    manifests = sorted((path / "raw/cost_manifest").glob("*.json"))
    if not manifests:
        sys.exit(f"{path}: no raw/cost_manifest/*.json")
    stages = sorted((parse_manifest(m) for m in manifests), key=lambda s: s.index)
    n = len(stages)
    if [s.index for s in stages] != list(range(n)):
        sys.exit(
            f"{path}: stage manifests are not one pipeline (stages {[s.index for s in stages]}); "
            "runs with several pipeline replicas are not supported"
        )
    kind_of_layer = {}
    for s in stages:
        for kind, layers in s.layers.items():
            for layer in layers:
                if layer in kind_of_layer:
                    sys.exit(f"{path}: layer {layer} appears on two stages")
                kind_of_layer[layer] = kind
    if sorted(kind_of_layer) != list(range(len(kind_of_layer))):
        sys.exit(f"{path}: layers do not cover 0..{len(kind_of_layer) - 1}")

    report = json.loads((path / "reports/kernel_time_share_report.json").read_text())
    workers = sorted(report["totals"]["workers"], key=lambda w: w["worker_id"])
    if [w["worker_id"] for w in workers] != list(range(n)):
        sys.exit(
            f"{path}: report workers {[w['worker_id'] for w in workers]} are not stages 0..{n - 1}"
        )

    per_stage_cost: dict[str, list[float]] = {}
    fixed_by_stage = [0.0] * n
    for s, w in zip(stages, workers):
        group_ms: dict[str, float] = {}
        for seg in w["segments"]:
            name, ms = seg["position"], seg["kernel_time_ms"]
            if name in s.group_of_leaf:
                kind = s.group_of_leaf[name]
                group_ms[kind] = group_ms.get(kind, 0.0) + ms
            elif name in s.fixed_leaves:
                fixed_by_stage[s.index] += ms
            else:
                sys.exit(f"{path}: stage {s.index} reports position {name} that its manifest lacks")
        for kind, layers in s.layers.items():
            per_stage_cost.setdefault(kind, []).append(group_ms.get(kind, 0.0) / len(layers))

    layer_cost = {k: sum(v) / len(v) for k, v in per_stage_cost.items()}
    spread = {
        k: max(abs(x - layer_cost[k]) for x in v) / layer_cost[k] if layer_cost[k] else 0.0
        for k, v in per_stage_cost.items()
    }
    # Fixed leaves: by which stages carry them (the first, the last, or all).
    first = fixed_by_stage[0]
    last = fixed_by_stage[-1] if n > 1 else 0.0
    middle = fixed_by_stage[1:-1]
    every = sum(middle) / len(middle) if middle else 0.0
    fixed_cost = {"first": first - every, "last": last - every, "every": every}
    return Run(
        path,
        n,
        stages,
        layer_cost,
        fixed_cost,
        [w["kernel_time_ms"] for w in workers],
        spread,
        kind_of_layer,
    )


def stage_cost(run: Run, lo: int, hi: int, index: int, num_stages: int) -> float:
    cost = (
        sum(run.layer_cost[run.kind_of_layer[layer]] for layer in range(lo, hi))
        + run.fixed_cost["every"]
    )
    if index == 0:
        cost += run.fixed_cost["first"]
    if index == num_stages - 1:
        cost += run.fixed_cost["last"]
    return cost


def vllm_default(num_layers: int, pp: int) -> list[int]:
    """vLLM `get_pp_indices`: the remainder goes to the stages before the last, nearest it first."""
    sizes = [num_layers // pp] * pp
    for i in range(2, num_layers % pp + 2):
        sizes[pp - i] += 1
    return sizes


def ranges(sizes):
    out, start = [], 0
    for n in sizes:
        out.append((start, start + n))
        start += n
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--pp-size", type=int)
    ap.add_argument("--each-stage-needs", action="append", default=[], metavar="GLOB")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="list splits whose bottleneck is within this fraction of the best (default 0.02)",
    )
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    runs = [load_run(p) for p in args.runs]
    kinds = runs[0].kind_of_layer
    for r in runs[1:]:
        if r.kind_of_layer != kinds:
            sys.exit(
                f"{r.path}: layer kinds differ from {runs[0].path}; mix runs of one model only"
            )
    num_layers = len(kinds)
    pp = args.pp_size or runs[0].num_stages

    print(f"{num_layers} layers; per-layer cost by kind (ms; max spread across stages):")
    for r in runs:
        print(f"  {r.path}")
        for kind in sorted(r.layer_cost):
            n = sum(1 for k in kinds.values() if k == kind)
            print(f"    {kind:24} x{n:<3} {r.layer_cost[kind]:12,.1f}  spread {r.spread[kind]:.2%}")
        print(
            f"    fixed: first {r.fixed_cost['first']:,.1f}  last {r.fixed_cost['last']:,.1f}  "
            f"every {r.fixed_cost['every']:,.1f}"
        )
        own = [stage_cost(r, lo, hi, i, r.num_stages) for i, (lo, hi) in enumerate(stage_ranges(r))]
        err = max(abs(a - b) / b for a, b in zip(own, r.stage_ms))
        print(f"    check: rebuilt stage times vs report, max error {err:.3%}")
        if err > 0.01:
            sys.exit(
                "    rebuilt stage times disagree with the report by >1%; the attribution is wrong"
            )

    # Normalize each run to its own total so every run weighs the same.
    totals = [sum(r.stage_ms) for r in runs]

    def cost(lo, hi, i):
        return sum(stage_cost(r, lo, hi, i, pp) / t for r, t in zip(runs, totals))

    def feasible(lo, hi):
        present = {kinds[layer] for layer in range(lo, hi)}
        return all(any(fnmatch.fnmatch(k, g) for k in present) for g in args.each_stage_needs)

    # Min-max contiguous split by DP, then every split within tolerance of it.
    INF = float("inf")
    best = [[INF] * (num_layers + 1) for _ in range(pp + 1)]
    best[0][0] = 0.0
    seg = {}
    for i in range(1, pp + 1):
        for hi in range(i, num_layers + 1):
            for lo in range(i - 1, hi):
                if best[i - 1][lo] == INF:
                    continue
                c = seg.setdefault(
                    (lo, hi, i - 1), cost(lo, hi, i - 1) if feasible(lo, hi) else INF
                )
                best[i][hi] = min(best[i][hi], max(best[i - 1][lo], c))
    optimum = best[pp][num_layers]
    if optimum == INF:
        sys.exit(f"no PP{pp} split satisfies --each-stage-needs {args.each_stage_needs}")
    bound = optimum * (1 + args.tolerance)
    found = []

    def dfs(i, lo, sizes, worst, sq):
        if i == pp:
            if lo == num_layers:
                found.append((worst, sq, sizes))
            return
        for hi in range(lo + 1, num_layers - (pp - i - 1) + 1):
            c = seg.get((lo, hi, i))
            if c is None:
                c = seg[(lo, hi, i)] = cost(lo, hi, i) if feasible(lo, hi) else INF
            if c <= bound:
                dfs(i + 1, hi, sizes + [hi - lo], max(worst, c), sq + c * c)

    dfs(0, 0, [], 0.0, 0.0)
    found.sort(key=lambda x: (x[0], x[1]))

    def describe(sizes):
        cs = [cost(lo, hi, i) for i, (lo, hi) in enumerate(ranges(sizes))]
        return max(cs), max(cs) / (sum(cs) / len(cs)) - 1, cs

    rows = []
    reference = [("vLLM default", vllm_default(num_layers, pp))]
    if pp == runs[0].num_stages:
        own = [hi - lo for lo, hi in stage_ranges(runs[0])]
        if own != reference[0][1]:
            reference.append(("run's split", own))
    base = describe(reference[-1][1])[0]
    print(
        f"\nPP{pp} splits (stage cost = share of each run's total, summed over runs; "
        f"gain = bottleneck vs {reference[-1][0]}):"
    )
    print(f"  {'split':42} {'bottleneck':>10} {'imbalance':>9} {'gain':>7}  stage costs")
    for name, sizes in reference + [
        (f"best #{k + 1}", s) for k, (_, _, s) in enumerate(found[: args.top])
    ]:
        worst, imb, cs = describe(sizes)
        rows.append(
            {
                "name": name,
                "layer_partition": sizes,
                "bottleneck": worst,
                "imbalance": imb,
                "gain": base / worst - 1,
                "stage_costs": cs,
            }
        )
        print(
            f"  {name + ' ' + str(sizes):42} {worst:10.4f} {imb:9.2%} {base / worst - 1:+7.2%}  "
            + " ".join(f"{c:.3f}" for c in cs)
        )
    keep = reference[-1]
    if base <= found[0][0] * (1 + 1e-9):
        # A tie is no reason to move layers: keep the split that already runs.
        print(f"\n{keep[0]} is already optimal; the remaining imbalance is layer granularity.")
        print(f"layer_partition: {keep[1]}")
    else:
        print(f"\nlayer_partition: {found[0][2]}")
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "runs": [str(r.path) for r in runs],
                    "pp_size": pp,
                    "each_stage_needs": args.each_stage_needs,
                    "splits": rows,
                },
                indent=2,
            )
        )


def stage_ranges(run: Run):
    out = []
    for s in run.stages:
        layers = sorted(x for group in s.layers.values() for x in group)
        out.append((layers[0], layers[-1] + 1))
    return out


if __name__ == "__main__":
    main()
