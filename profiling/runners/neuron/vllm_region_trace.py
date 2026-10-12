"""Bind compiled region graphs and attribute native time to model/head per forward.

A split forward is one ordered pair of whole-chip executions inside a public
request: the ``model`` region NEFF, then the ``head`` region NEFF of the same
compiled shape. Each region's time is the union of its eight physical-core
intervals. A pair must not overlap in time, every execution must belong to a
pair, and the pairs must cover every scheduler forward the stock check expects.
Nothing is fitted or scaled.
"""

from __future__ import annotations

import collections
import json
import re
import shlex
from pathlib import Path

from profiling.runners.neuron.vllm_forward_trace import (
    check_forward_coverage,
    export_events,
    group_executions,
    reserved_physical_cores,
    union_duration,
)

REGIONS = ("model", "head")
# The stock vllm_neuron backend's trn2 flags (see any stock compile_cache/*/command.txt).
ORIGINAL_COMPILER_ARGS = (
    "--auto-cast=none",
    "--verbose=35",
    "-O1",
    "--internal-hlo2tensorizer-options=--modular-flow-mac-threshold=10 "
    "--experimental-unsafe-fp8e4m3fn-as-fp8e4m3",
    "--internal-backend-options=--enable-verifier=false --enable-nested-dynamic-loop",
)


def canonical_graph(text: str) -> str:
    """Drop only FX diagnostic user counts; op/target/args/kwargs text is retained."""
    return re.sub(r"\[num_users=\d+\]", "", text).strip()


def bind_region_graphs(compile_cache: Path, partition_dir: Path, log: str, expected) -> dict:
    """Map each compiled region key loaded in ``log`` to (region, phase, token_bucket).

    A key binds only when its compiled FX graph equals a partition receipt's
    region graph on all four ranks, it was compiled with the original flags, and
    it has exactly one NEFF. ``expected`` is the set of (phase, token_bucket).
    """
    receipts = []
    for path in sorted(partition_dir.glob("rank*-*/receipt.json")):
        receipt = json.loads(path.read_text())
        if receipt.get("passed") is not True:
            raise ValueError(f"structural partition check failed: {path}")
        rank = int(re.fullmatch(r"rank(\d+)-[0-9a-f]+", path.parent.name)[1])
        for region in REGIONS:
            text = canonical_graph((path.parent / f"{region}.fx.txt").read_text())
            receipts.append((text, region, receipt["phase"], receipt["token_bucket"], rank))
    keys = set(re.findall(r"Compilation cache key: ([0-9a-f]{32})", log))
    bound = {}
    for key in sorted(keys):
        graph = compile_cache / key / "fxgraph.txt"
        if not graph.exists():
            continue
        text = canonical_graph(graph.read_text())
        matches = [item for item in receipts if item[0] == text]
        if not matches:
            continue
        identities = {item[1:4] for item in matches}
        if len(identities) != 1:
            raise ValueError(f"compiled graph {key} matches several region identities")
        if {item[4] for item in matches} != {0, 1, 2, 3}:
            raise ValueError(f"compiled region graph {key} lacks receipts from all four ranks")
        command = shlex.split((compile_cache / key / "command.txt").read_text())
        if tuple(command[-len(ORIGINAL_COMPILER_ARGS) :]) != ORIGINAL_COMPILER_ARGS:
            raise ValueError(f"region graph {key} was not compiled with the original flags")
        if len(list((compile_cache / key).glob("*.neff"))) != 1:
            raise ValueError(f"region graph {key} must have exactly one NEFF")
        region, phase, bucket = identities.pop()
        bound[key] = {"region": region, "phase": phase, "token_bucket": bucket}
    observed = collections.Counter(
        (row["region"], row["phase"], row["token_bucket"]) for row in bound.values()
    )
    wanted = {(region, *shape) for region in REGIONS for shape in expected}
    if set(observed) != wanted or any(count != 1 for count in observed.values()):
        raise ValueError(f"region graph binding incomplete or ambiguous: {dict(observed)}")
    return bound


def measure_region_trace(events, requests, region_info, physical_cores):
    """Return ({(region, phase, bucket): [ms per forward]}, per-forward records)."""
    by_request = collections.defaultdict(list)
    for execution in group_executions(events, requests, physical_cores):
        identity = region_info.get(execution["model_hash"])
        if identity is None:
            raise ValueError("native execution ran a graph not bound to a validated region")
        by_request[execution["request_index"]].append({**execution, **identity})
    records, measured = [], collections.defaultdict(list)
    for request_index, executions in sorted(by_request.items()):
        executions.sort(key=lambda row: row["spans"][0][0])
        if len(executions) % 2:
            raise ValueError("a region execution has no model/head partner")
        for model, head in zip(executions[::2], executions[1::2]):
            if (model["region"], head["region"]) != REGIONS:
                raise ValueError("region executions are not ordered model then head")
            shape = (model["phase"], model["token_bucket"])
            if (head["phase"], head["token_bucket"]) != shape:
                raise ValueError("model and head executions belong to different shapes")
            model_ns, head_ns = union_duration(model["spans"]), union_duration(head["spans"])
            whole_ns = union_duration(model["spans"] + head["spans"])
            if model_ns + head_ns != whole_ns:
                raise ValueError("model and head native executions overlap in time")
            spans = model["spans"] + head["spans"]
            record = {
                "request_index": request_index,
                "phase": shape[0],
                "token_bucket": shape[1],
                "model_hash": model["model_hash"],
                "head_hash": head["model_hash"],
                "model_ms": model_ns / 1e6,
                "head_ms": head_ns / 1e6,
                "whole_union_ms": whole_ns / 1e6,
                "whole_span_ms": (max(b for _, b in spans) - min(a for a, _ in spans)) / 1e6,
            }
            records.append(record)
            measured[("model", *shape)].append(record["model_ms"])
            measured[("head", *shape)].append(record["head_ms"])
    check_forward_coverage(records, requests)
    return dict(measured), records


def export_and_measure_regions(root: Path, compile_cache: Path, expected) -> dict:
    """Bind the split accuracy run's region graphs, then measure its profile trace."""
    log = (root / "accuracy.log").read_text()
    region_info = bind_region_graphs(compile_cache, root / "partition-accuracy", log, expected)
    (root / "region-graph-binding.json").write_text(json.dumps(region_info, indent=2))
    requests = json.loads((root / "profiled-outputs.json").read_text())
    measured, records = measure_region_trace(
        export_events(root), requests, region_info, reserved_physical_cores()
    )
    (root / "region-execution-records.json").write_text(json.dumps(records, indent=2))
    (root / "region-timing.json").write_text(
        json.dumps({":".join(map(str, key)): values for key, values in measured.items()}, indent=2)
    )
    return measured
