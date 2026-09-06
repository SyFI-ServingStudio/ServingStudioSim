"""Calibrate one rendered Spec5 case from its complete warm workload pass."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import yaml

from scripts.prepare_matrix_e2e_reuse import full_rates, read_trace, request_rates


def prepare(case):
    profile = case / "profile_workload"
    result = json.loads((profile / "profile_result.json").read_text())
    config = yaml.safe_load((case / "profile_workload.yaml").read_text())
    rows = read_trace(case / "trace.csv")
    if not config["workload"].get("warmup"):
        raise ValueError("warmup must be enabled")
    if (result["profile_kind"] != "workload_metrics"
            or result["request_timing_count"] != len(rows)
            or not result["drive_summary"]["reached_idle"]):
        raise ValueError("complete independent workload required")
    replay = [json.loads(line) for line in Path(result["replay_result"]).read_text().splitlines()
              if line.strip()]
    measured = {entry["source"]["data"]["id"]: entry for entry in replay}
    if len(measured) != len(replay) or set(measured) != {row["id"] for row in rows}:
        raise ValueError("measured request population differs from trace")
    for row in rows:
        entry = measured[row["id"]]
        source, outcome = entry["source"]["data"], entry["outcome"]
        if not (outcome["status"] == "SUCCESS"
                and int(row["input_len"]) == source["input_len"]
                and int(row["output_len"]) == source["output_len_target"]
                == outcome["output_len_actual"]
                and float(row["arrival_time"]) == source["arrival_time_ms"]):
            raise ValueError(f"request mismatch: {row['id']}")
    metrics = Path(result["metrics_jsonl"])
    fallback = full_rates(json.loads((profile / "spec_decode_metrics.json").read_text()))
    rates, fallback_positions = request_rates(metrics, set(measured), fallback)
    server_log = next((profile / "vllm").glob("*_server.log"))
    capacities = set(re.findall(r"GPU KV cache size: ([\d,]+) tokens", server_log.read_text()))
    if len(capacities) != 1:
        raise ValueError("ambiguous measured KV capacity")
    capacity = int(capacities.pop().replace(",", ""))
    evidence = case / "e2e_reuse"
    evidence.mkdir(exist_ok=True)
    trace = evidence / "trace_observed.csv"
    with trace.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*rows[0], "accept_rate"])
        writer.writeheader()
        writer.writerows({**row, "accept_rate": json.dumps(rates[row["id"]])} for row in rows)
    path = case / "simulation.yaml"
    original = path.read_text()
    backup = evidence / "simulation_before.yaml"
    if not backup.exists():
        backup.write_text(original)
    simulation = yaml.safe_load(original)
    simulation["workload"]["trace_files"] = [str(trace.resolve())]
    simulation["pools"]["main"]["groups"][0]["worker"]["attn_gpu_memory_gb"] = capacity * 55224 / 1e9
    path.write_text(yaml.safe_dump(simulation, sort_keys=False))
    manifest = {
        "case": case.name, "requests": len(rows), "workload_profile": str(profile.resolve()),
        "acceptance_source": str(metrics),
        "acceptance_method": "same-pass per-request raw conditional acceptance",
        "fallback_positions": fallback_positions, "fallback_rates": fallback,
        "measured_kv_tokens": capacity, "kv_bytes_per_token": 55224,
        "request_lengths_and_arrivals_unchanged": True,
        "trace_sha256": hashlib.sha256((case / "trace.csv").read_bytes()).hexdigest(),
        "acceptance_source_sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
        "limitations": "Observed-conditioned stochastic prediction, not exact acceptance replay. "
                        "Missing conditional samples use this measurement's aggregate rates.",
    }
    (evidence / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.case_dir.resolve()), indent=2))
