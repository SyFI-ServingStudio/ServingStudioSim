"""Snapshot matrix results from the owned Analyzer API, preserving missing cases."""

import argparse
import csv
import json
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=120) as response:
        return json.load(response)


def error(predicted, measured):
    return 100 * (predicted / measured - 1) if measured else None


def audit_request_population(case, provenance, *, trace_path=None):
    """Resolve dense simulator IDs using the independent CSV loader's row order."""
    with (trace_path or case / "e2e_reuse/trace_observed.csv").open(newline="") as stream:
        trace = list(csv.DictReader(stream))
    profile = Path(provenance["workload_profile"])
    result = json.loads((profile / "profile_result.json").read_text())
    replay = [json.loads(line) for line in Path(result["replay_result"]).read_text().splitlines()
              if line.strip()]
    measured = {row["source"]["data"]["id"]: row for row in replay}
    simulated_rows = pq.read_table(case / "simulation/raw/request_slo.parquet", columns=[
        "request_id", "completed", "fresh_prompt_tokens", "num_output_tokens",
    ]).to_pylist()
    simulated = {row["request_id"]: row for row in simulated_rows}
    checks = {
        "unique_trace_ids": len({row["id"] for row in trace}) == len(trace),
        "unique_measured_ids": len(measured) == len(replay),
        "unique_simulated_ids": len(simulated) == len(simulated_rows),
        "measured_population": set(measured) == {row["id"] for row in trace},
        "simulated_population": set(simulated) == set(range(len(trace))),
    }
    mismatches = []
    identities = []
    for index, row in enumerate(trace):
        identities.append({"request_id": index, "source_id": row["id"]})
        real, sim = measured.get(row["id"]), simulated.get(index)
        if real is None or sim is None:
            mismatches.append(row["id"])
            continue
        source, outcome = real["source"]["data"], real["outcome"]
        if not (outcome["status"] == "SUCCESS" and sim["completed"]
                and int(row["input_len"]) == source["input_len"] == sim["fresh_prompt_tokens"]
                and int(row["output_len"]) == source["output_len_target"]
                == outcome["output_len_actual"] == sim["num_output_tokens"]
                and float(row["arrival_time"]) == source["arrival_time_ms"]):
            mismatches.append(row["id"])
    checks["per_request_lengths_arrivals_and_completion"] = not mismatches
    return {
        "all_ok": all(checks.values()), "requests": len(trace), "checks": checks,
        "mismatched_source_ids": mismatches, "identities": identities,
        "mapping_basis": "sim/frontend/schema.rs: load independent rows in order; "
                         "SourceIdentities::intern_request assigns dense IDs before replay",
        "limitations": "Declared trace arrivals are checked; actual release times may "
                       "differ under concurrency limits. Latencies remain unpaired distributions.",
    }


def summarize(root, base):
    catalog = get(base, "/api/v1/alignments")["alignments"]
    resources = {r["display_name"]: r for r in catalog}
    records = []
    for case in sorted(root.iterdir()):
        if not (case / "simulation.yaml").exists():
            continue
        record = {"case": case.name, "kernel": None, "e2e": None}
        if root.name == "20260905_2_spec5_matrix" and case.name[:2] in {"11", "12"}:
            record["unavailable_reason"] = "real KV-capacity failure"
        elif root.name == "20260905_2_spec5_matrix" and case.name[:2] == "04":
            record["e2e_unavailable_reason"] = "no historical independent workload"
        resource = resources.get(f"{root.name}/{case.name}")
        if resource:
            resource_id = resource["alignment_id"]
            record["alignment_id"] = resource_id
            prefix = f"/api/v1/alignments/{resource_id}/subjects/"
            if (case / "analysis_kernel/reports/alignment_iteration_report.json").exists():
                report = get(base, prefix + "iteration/report")
                if report.get("available"):
                    record["kernel"] = {
                        "comparison": report["comparison"],
                        "coverage": report["mapping"]["coverage"],
                        "meta": report["meta"],
                        "operations": report["operations"],
                        "unmapped_measured_kernels": report["mapping"]["unmapped_measured_kernels"],
                        "unmapped_simulated_slots": report["mapping"]["unmapped_simulated_slots"],
                        "multiplier": report["meta"]["recommended_gpu_time_multiplier"],
                        "largest_operation_errors": [
                            {k: op[k] for k in ("operation", "comparison", "missing_measured",
                                               "missing_simulated")}
                            for op in report["operations"][:8]
                        ],
                        "largest_unmapped_simulated": sorted(
                            report["mapping"]["unmapped_simulated_slots"],
                            key=lambda row: -row["total_ms"],
                        )[:8],
                    }
            if (case / "analysis_e2e/reports/alignment_e2e_report.json").exists():
                report = get(base, prefix + "e2e/report")
                if report.get("available"):
                    throughput = report["throughput"]
                    record["e2e"] = {
                        "meta": report["meta"],
                        "latency": report["latency"], "throughput": throughput,
                        "errors_pct": {
                            "throughput": error(throughput["simulated_completion_tps"],
                                                throughput["measured_client_completion_tps"]),
                            **{metric: error(values["simulated_ms"]["mean"],
                                             values["measured_ms"]["mean"])
                               for metric, values in report["latency"].items()},
                        },
                    }
                    workload = get(base, prefix + "workload/report")
                    if workload.get("available"):
                        record["workload"] = {
                            "meta": workload["meta"], "metrics": workload["metrics"],
                        }
        provenance = case / "e2e_reuse/provenance.json"
        if provenance.exists():
            record["e2e_provenance"] = json.loads(provenance.read_text())
            if record["e2e"]:
                record["request_population_audit"] = audit_request_population(
                    case, record["e2e_provenance"],
                )
        stage_path = case / ".launcher/stages/simulation.json"
        if record["e2e"] and stage_path.is_file():
            command = json.loads(stage_path.read_text()).get("argv", [])
            flag = "--gpu-time-multiplier-from"
            if flag in command:
                source = Path(command[command.index(flag) + 1]).resolve()
                record["gpu_calibration"] = {
                    "source": str(source),
                    "borrowed": source != (case / "analysis_kernel").resolve(),
                }
        conservation = case / "simulation/reports/workload_conservation_report.json"
        if conservation.exists():
            record["conservation"] = json.loads(conservation.read_text())
        records.append(record)
        print(case.name, "kernel=" + str(record["kernel"] is not None),
              "e2e=" + str(record["e2e"] is not None), flush=True)
    snapshot = {
        "updated_at": datetime.now(UTC).isoformat(), "analyzer": base,
        "kernel_reports": sum(row["kernel"] is not None for row in records),
        "e2e_reports": sum(row["e2e"] is not None for row in records),
        "cases": records,
    }
    (root / "CURRENT_RESULTS.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    lines = ["# Matrix Alignment Results", "", f"Updated: {snapshot['updated_at']}", "",
             f"Completed reports: {snapshot['kernel_reports']} kernel, "
             f"{snapshot['e2e_reports']} independent E2E.", "",
             "NSYS kernel windows and independent full-workload E2E are separate evidence.",
             "E2E uses explicitly approximate observed acceptance; provenance is in the JSON.", "",
             "| Case | Kernel signed % | Critical mapping % | Sim mapping % | "
             "Server TTFT P50 % | P90 % | P99 % | Server TPOT P50 % | P90 % | P99 % |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "pending" if value is None else f"{value:.2f}"
    for row in records:
        kernel, e2e = row["kernel"], row["e2e"]
        values = [None] * 9
        if kernel:
            values[:3] = [kernel["comparison"]["all"]["signed_error_pct"],
                          100 * kernel["coverage"]["measured_critical_path_fraction"],
                          100 * kernel["coverage"]["simulated_workload_fraction"]]
        if e2e:
            values[3:] = [
                error(e2e["latency"][metric]["simulated_ms"][quantile],
                      e2e["latency"][metric]["measured_ms"][quantile])
                for metric in ("server_ttft", "server_tpot")
                for quantile in ("p50", "p90", "p99")
            ]
        cells = list(map(fmt, values))
        if "unavailable_reason" in row:
            cells = ["n/a"] * 9
        elif "e2e_unavailable_reason" in row:
            cells[3:] = ["n/a"] * 6
        lines.append("| " + row["case"] + " | " + " | ".join(cells) + " |")
    lines.append("")
    for row in records:
        reason = row.get("unavailable_reason") or row.get("e2e_unavailable_reason")
        if reason:
            lines.append(f"{row['case']}: {reason}.")
        calibration = row.get("gpu_calibration", {})
        if calibration.get("borrowed"):
            lines.append(f"{row['case']}: approximate GPU calibration reused from "
                         f"{calibration['source']}.")
    lines.append("Raw residency mapping coverage, per-operation gaps and conservation "
                 "remain in CURRENT_RESULTS.json.")
    (root / "CURRENT_RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--analyzer", default="http://127.0.0.1:8787")
    args = parser.parse_args()
    summarize(args.root, args.analyzer)
