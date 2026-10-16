"""Export the per-layer targets of the one-request bring-up: layer_targets.jsonl beside this file.

For each prompt length and stage, the stage graph (raw/cost_manifest) is evaluated on that run's kernel times
(raw/cost_log), the same way the stage time is: Sum adds, Scale{n} multiplies, Max/Parallel{overlap} take the
slowest child over the overlap. One line per (prompt length, stage, layer type), plus the stage's head (stage 0) and
output head (stage 3):
  layer_ms      the time of one layer of that type
  kernels       every kernel with time in it: position, kind, backend, call sizes, launches per layer, time per
                launch and in total, and beyond_rows (the call is larger than the largest profiled row, so its
                time is extrapolated). The totals sum to layer_ms.
  moe_per_expert_batches   tokens per routed expert to force in the layer test, where the layer has routed experts
Each stage's time is also checked against the run's own total. It also draws each layer type's cost tree, with
every node's time per prompt length, into layers/<layer type>.txt.

usage (ServingStudioSim root, at the frozen commit, after the bring-up one-request runs and kernel_specs.py):
  PYTHONPATH=$PWD uv run python doc/handoff/glm53_flash_pp4_b200_v2/data/layer_targets.py <one-request run dir>
"""
import collections, json, re, sys
from pathlib import Path

import pyarrow.parquet as pq

run = Path(sys.argv[1])
here = Path(__file__).parent

# Largest profiled call size per (kind, backend, position, axis), and the MoE histograms by token count.
largest = collections.defaultdict(int)
moe_rows = {}
for line in (here / "kernel_specs.jsonl").read_text().splitlines():
    r = json.loads(line)
    for pos in r["positions"]:
        name = pos.split(": ", 1)[1]
        for axis, v in r["spec"].items():
            if isinstance(v, int) and not isinstance(v, bool):
                key = (r["kind"], r["backend"], name, axis)
                largest[key] = max(largest[key], v)
    if r["kind"] == "nvfp4_fused_moe":
        moe_rows[r["spec"]["num_tokens"]] = json.loads(r["spec"]["per_expert_batches"])


def evaluate(nodes, i, t):
    (op, v), = nodes[i].items()
    if op == "Leaf":
        return t[v]
    children = [evaluate(nodes, c, t) for c in range(v["children"]["start"], v["children"]["end"])]
    if op == "Sum":
        return sum(children)
    if op == "Scale":
        return v["n"] * sum(children)
    if op in ("Max", "Parallel"):
        return max(children) / v["overlap"]
    raise ValueError(op)


def attribute(nodes, i, t, weight=1.0):
    """(slot, weight) pairs whose weight * slot time sums to the node's time: Sum and Scale pass the multiplier
    down, Max and Parallel charge their slowest child over the overlap."""
    (op, v), = nodes[i].items()
    if op == "Leaf":
        return [(v, weight)]
    children = range(v["children"]["start"], v["children"]["end"])
    if op == "Scale":
        weight *= v["n"]
    if op in ("Max", "Parallel"):
        children = [max(children, key=lambda c: evaluate(nodes, c, t))]
        weight /= v["overlap"]
    return [p for c in children for p in attribute(nodes, c, t, weight)]


PAIR_AXES = {"prefill_query_cache_pairs": ("num_queries", "num_cache_tokens"),
             "prefill_query_key_pairs": ("num_queries", "num_keys")}


def beyond_rows(kind, backend, name, call):
    sizes = []
    for axis, v in call.items():
        if axis in PAIR_AXES:  # one launch per pair
            sizes += [(a, x) for pair in v for a, x in zip(PAIR_AXES[axis], pair)]
        elif isinstance(v, list):  # e.g. sequence_lengths -> the per-sequence axis
            sizes.append((axis.removesuffix("s"), max(v, default=0)))
        else:
            sizes.append((axis, v))
    for axis, v in sizes:
        limit = largest.get((kind, backend, name, axis))
        if isinstance(v, int) and limit is not None and v > limit:
            return True
    return False


def moe_histogram(tokens):
    if tokens in moe_rows:
        return moe_rows[tokens]
    edge = max(n for n in moe_rows if n <= tokens)  # past the largest row: scale its proportions
    assert tokens % edge == 0, (tokens, edge)
    return [b * (tokens // edge) for b in moe_rows[edge]]


def render(nodes, labels, slots, i, times, depth=0):
    """Text lines of the subtree at i: each node with its time per prompt length (times: length -> slot times)."""
    (op, v), = nodes[i].items()
    pad = "  " * depth
    if op == "Leaf":
        slot = slots[v]
        cfg = slot["kernel_config"]
        what = slot["kind"] if slot["kind"] == "elementwise" else f"{slot['kind']} / {'|'.join(cfg['backends'])}"
        head = f"{pad}- {slot['name']}  [{what}]"
        vals = [t[v] for t in times.values()]
    else:
        label = re.sub(r" \(\w+(Worklet|Model)\)", "", labels[i] or "")
        args = f"{{n={v['n']}}}" if op == "Scale" else f"{{overlap={v['overlap']}}}" if "overlap" in v else ""
        if op == "Scale" and v["n"] == 0:
            label = "not launched with these backends"
        head = f"{pad}{op}{args}  {label}".rstrip()
        vals = [evaluate(nodes, i, t) for t in times.values()]
    lines = [(head, vals)]
    if op != "Leaf":
        for c in range(v["children"]["start"], v["children"]["end"]):
            lines += render(nodes, labels, slots, c, times, depth + 1)
    return lines


cases = sorted((p for p in run.iterdir() if (p / "raw/cost_manifest").is_dir()),
               key=lambda p: int(re.search(r"(\d+)k", p.name).group(1)))
# The stage each layer type's tree is drawn from.
TREE_STAGE = {"input_head": 0, "first_kda_dense": 0, "kda_dense": 0, "dsa_moe": 1, "kda_moe": 1, "output_head": 3}
trees = {}  # layer type -> (nodes, labels, slots, node ids, layers, {length label: slot times})
out = (here / "layer_targets.jsonl").open("w")
for case in cases:
    tokens = int(re.search(r"(\d+)k", case.name).group(1)) * 1024
    for manifest in sorted((case / "raw/cost_manifest").glob("worker_stage_*.json")):
        stage = int(manifest.stem.removeprefix("worker_stage_"))
        section = json.loads(manifest.read_text())["sections"][0]
        nodes, labels, slots = section["nodes"], section["node_labels"], section["slots"]
        row, = pq.read_table(case / f"raw/cost_log/{manifest.stem}.parquet",
                             columns=["total_time_ms", "slot_time_ms", "slot_input", "slot_backend"]).to_pylist()
        t = row["slot_time_ms"]
        total = evaluate(nodes, 0, t)
        assert abs(total - row["total_time_ms"]) < 1e-3 * row["total_time_ms"], (case.name, stage)

        (_, root), = nodes[0].items()
        groups = collections.OrderedDict()  # layer type -> (layers, node ids timing one layer, layer count)
        for c in range(root["children"]["start"], root["children"]["end"]):
            (op, v), = nodes[c].items()
            label = labels[c] or ""
            if op == "Scale":
                layers = json.loads(re.search(r"layers (\[[^\]]*\])", label).group(1))
                groups[label.split(" ")[0]] = (layers, [v["children"]["start"]], v["n"])
            else:
                name = "output_head" if stage > 0 else "input_head"
                layers, node_ids, _ = groups.get(name, ([], [], 1))
                groups[name] = (layers, node_ids + [c], 1)

        for layer_type, (layers, node_ids, count) in groups.items():
            layer_ms = sum(evaluate(nodes, n, t) for n in node_ids)
            kernels = []
            for s, weight in (p for n in node_ids for p in attribute(nodes, n, t)):
                if t[s] <= 0 or weight == 0:
                    continue
                slot = slots[s]
                backend = slot["kernel_config"]["backends"][row["slot_backend"][s]] if row["slot_backend"][s] != 255 else None
                call = json.loads(row["slot_input"][s])
                kernels.append(dict(position=slot["name"], kind=slot["kind"], backend=backend, call=call,
                                    launches=weight, time_ms=t[s], total_ms=weight * t[s],
                                    beyond_rows=slot["kind"] != "elementwise"
                                    and beyond_rows(slot["kind"], backend, slot["name"], call)))
            assert abs(sum(k["total_ms"] for k in kernels) - layer_ms) < 1e-6 * max(layer_ms, 1), layer_type
            line = dict(prompt_tokens=tokens, stage=stage, layer_type=layer_type, layers=layers, count=count,
                        layer_ms=layer_ms, kernels=kernels)
            if any(k["kind"] == "nvfp4_fused_moe" for k in kernels):
                line["moe_per_expert_batches"] = moe_histogram(tokens)
            out.write(json.dumps(line) + "\n")
            if TREE_STAGE[layer_type] == stage:
                entry = trees.setdefault(layer_type, [nodes, labels, slots, node_ids, set(), {}])
                entry[4].update(layers)
                entry[5][case.name.removeprefix("one_")] = t
        print(f"{case.name:9} stage {stage}  {total:9.3f} ms  " +
              "  ".join(f"{k}={sum(evaluate(nodes, n, t) for n in v[1]):.3f}" for k, v in groups.items()))
out.close()

# One tree per layer type: data/layers/<layer type>.txt, every node with its time per prompt length.
tree_dir = here / "layers"
tree_dir.mkdir(exist_ok=True)
all_layers = collections.defaultdict(set)
for line in (here / "layer_targets.jsonl").read_text().splitlines():
    r = json.loads(line)
    all_layers[r["layer_type"]].update(r["layers"])
for layer_type, (nodes, labels, slots, node_ids, _, times) in trees.items():
    lines = [l for n in node_ids for l in render(nodes, labels, slots, n, times)]
    width = max(len(h) for h, _ in lines) + 2
    layers = sorted(all_layers[layer_type])
    text = [f"# {layer_type}: one layer" + (f" (layers {', '.join(map(str, layers))})" if layers else ""),
            "# Cost tree of one-request bring-up prefill, ms per prompt length. Sum adds its children, Scale{n} runs",
            "# its child n times, Max/Parallel{overlap} take the slowest child over the overlap. A zero row is a",
            "# branch this workload never takes (decode, or the shared-expert overlap at <= 256 tokens).",
            "# Scale{n=0} is an op the chosen backends do not need: the KDA beta sigmoid, which FlashInfer's KDA",
            "# prefill applies itself because it takes beta logits.",
            "",
            " " * width + "".join(f"{k:>9}" for k in times)]
    text += [h.ljust(width) + "".join(f"{x:9.3f}" if x else f"{'-':>9}" for x in vals) for h, vals in lines]
    (tree_dir / f"{layer_type}.txt").write_text("\n".join(text) + "\n")
print(here / "layer_targets.jsonl", tree_dir)
