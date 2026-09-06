"""Prepare an explicitly approximate E2E replay against existing full workloads.

Request lengths and arrivals must match exactly. NSYS supplies acceptance only,
never request latency. Short-capture cases use their full workload counters.
"""

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml


def read_trace(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def full_rates(metrics):
    counts = [metrics["accepted_tokens_per_position"][str(i)] for i in range(5)]
    denominators = [metrics["num_drafts"], *counts[:-1]]
    if any(n < 0 or n > d for n, d in zip(counts, denominators)):
        raise ValueError("invalid full-workload acceptance chain")
    return [n / d if d else 0.0 for n, d in zip(counts, denominators)]


def request_rates(path, allowed, fallback):
    totals = defaultdict(lambda: [[0] * 5, [0] * 5])
    for line in path.open():
        record = json.loads(line)
        for step in record.get("decode_request_progress", []):
            request_id = step["request_id"]
            request_id = request_id.removeprefix("independent_")
            if request_id not in allowed:
                raise ValueError(f"NSYS request outside full trace: {request_id}")
            if step.get("request_finished_before", False):
                continue
            drafted, accepted = step["drafted_tokens"], step["accepted_draft_tokens"]
            if not (0 <= accepted <= drafted <= 5):
                raise ValueError("invalid request acceptance chain")
            numerator, denominator = totals[request_id]
            for position in range(drafted):
                denominator[position] += accepted >= position
                numerator[position] += accepted > position
    rates = {}
    fallback_positions = 0
    for request_id in allowed:
        numerator, denominator = totals[request_id]
        rates[request_id] = [
            n / d if d else fallback[i]
            for i, (n, d) in enumerate(zip(numerator, denominator))
        ]
        fallback_positions += sum(d == 0 for d in denominator)
    return rates, fallback_positions


def prepare(case, reference):
    old_case = reference / case.name
    old_profile = old_case / "profile_workload"
    if not (old_profile / "profile_result.json").is_file():
        return {"case": case.name, "status": "missing historical full workload"}
    rows = read_trace(case / "trace.csv")
    if rows != read_trace(old_case / "trace.csv"):
        raise ValueError(f"{case.name}: workload trace differs")
    configs = [yaml.safe_load((p / "profile_workload.yaml").read_text())
               for p in (case, old_case)]
    for config in configs:
        config.pop("cuda_visible_devices", None)
        config["server"].pop("port", None)
    if configs[0] != configs[1]:
        raise ValueError(f"{case.name}: serving/workload config differs beyond device IDs and port")
    result = json.loads((old_profile / "profile_result.json").read_text())
    if result["profile_kind"] != "workload_metrics":
        raise ValueError("E2E requires independent workload metrics")
    if result["request_timing_count"] != len(rows) or not result["drive_summary"]["reached_idle"]:
        raise ValueError(f"{case.name}: full workload did not complete the requested population")
    metrics_path = old_profile / "spec_decode_metrics.json"
    fallback = full_rates(json.loads(metrics_path.read_text()))
    server_log = next((old_profile / "vllm").glob("*_server.log"))
    capacities = set(re.findall(r"GPU KV cache size: ([\d,]+) tokens", server_log.read_text()))
    if len(capacities) != 1:
        raise ValueError(f"{case.name}: ambiguous measured KV capacity")
    capacity = int(capacities.pop().replace(",", ""))
    short_capture = case.name[:2] in {"13", "14", "15"}
    if short_capture:
        rates = {row["id"]: fallback for row in rows}
        fallback_positions = len(rows) * 5
        acceptance_source = metrics_path
        method = "same-case full-workload aggregate conditional acceptance"
    else:
        acceptance_source = next((case / "profile_nsys" / "vllm").glob("*_metrics.jsonl"))
        rates, fallback_positions = request_rates(
            acceptance_source, {row["id"] for row in rows}, fallback
        )
        method = "cross-pass per-request raw acceptance; no output-progress clipping"
    evidence = case / "e2e_reuse"
    evidence.mkdir(exist_ok=True)
    simulation_path = case / "simulation.yaml"
    simulation = yaml.safe_load(simulation_path.read_text())
    before = evidence / "simulation_before.yaml"
    if not before.exists():
        before.write_text(simulation_path.read_text())
    trace_path = evidence / "trace_observed.csv"
    with trace_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*rows[0], "accept_rate"])
        writer.writeheader()
        writer.writerows({**row, "accept_rate": json.dumps(rates[row["id"]])} for row in rows)
    simulation["workload"]["trace_files"] = [str(trace_path.resolve())]
    simulation["pools"]["main"]["groups"][0]["worker"]["attn_gpu_memory_gb"] = (
        capacity * 55224 / 1e9
    )
    simulation_path.write_text(yaml.safe_dump(simulation, sort_keys=False))
    analysis_path = case / "analyze_e2e.yaml"
    analysis = yaml.safe_load(analysis_path.read_text())
    analysis["workload_profile_log_dir"] = str(old_profile.resolve())
    analysis_path.write_text(yaml.safe_dump(analysis, sort_keys=False))
    manifest = {
        "case": case.name, "status": "prepared", "approximate": True,
        "requests": len(rows), "workload_profile": str(old_profile.resolve()),
        "acceptance_source": str(acceptance_source.resolve()), "acceptance_method": method,
        "fallback_positions": fallback_positions, "fallback_rates": fallback,
        "measured_kv_tokens": capacity, "kv_bytes_per_token": 55224,
        "config_comparison": "equal except physical device IDs and port",
        "trace_sha256": hashlib.sha256((case / "trace.csv").read_bytes()).hexdigest(),
        "acceptance_source_sha256": hashlib.sha256(acceptance_source.read_bytes()).hexdigest(),
        "request_lengths_and_arrivals_unchanged": True,
        "limitations": "Cross-run acceptance estimate; no exact scheduler or acceptance replay.",
    }
    (evidence / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    for case in sorted(args.root.iterdir()):
        if (case / "simulation.yaml").is_file():
            print(json.dumps(prepare(case, args.reference)), flush=True)
