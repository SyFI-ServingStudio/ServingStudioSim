"""Explicit observed-conditioned inputs; never rewrite simulation settings."""

import csv
import hashlib
import json
from pathlib import Path

from .request_population import read_trace, validate_replay


def acceptance_counts(metrics_path: Path, request_ids: set[str], *, draft_tokens: int,
                      request_id_prefix: str = "") -> dict:
    """Count conditional trials, including rejection at the first draft position."""
    if type(draft_tokens) is not int or draft_tokens <= 0:
        raise ValueError("draft_tokens must be a positive integer")
    encoded_ids = {request_id_prefix + key: key for key in request_ids}
    counts = {key: [[0] * draft_tokens, [0] * draft_tokens] for key in request_ids}
    with metrics_path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            for step in record.get("decode_request_progress", []):
                if step["request_id"] not in encoded_ids:
                    raise ValueError(f"observed request outside full trace: {step['request_id']}")
                # These rounds do not advance output; raw records retain their executed work.
                if step.get("request_finished_before", False):
                    continue
                drafted, accepted = step["drafted_tokens"], step["accepted_draft_tokens"]
                if (type(drafted) is not int or type(accepted) is not int
                        or not 0 <= accepted <= drafted <= draft_tokens):
                    raise ValueError("invalid request acceptance chain")
                numerator, denominator = counts[encoded_ids[step["request_id"]]]
                for position in range(drafted):
                    denominator[position] += accepted >= position
                    numerator[position] += accepted > position
    return counts


def prepare_workload(*, source_trace: Path, profile_dir: Path, output_trace: Path,
                     draft_tokens: int, missing_acceptance: str,
                     request_id_prefix: str = "") -> dict:
    """Prepare per-request probabilities from a complete workload-metrics pass."""
    if missing_acceptance not in {"error", "run-aggregate"}:
        raise ValueError("missing_acceptance must be error or run-aggregate")
    result_path = profile_dir / "profile_result.json"
    result = json.loads(result_path.read_text())
    if (result.get("profile_kind") != "workload_metrics"
            or result.get("drive_summary", {}).get("reached_idle") is not True):
        raise ValueError("a complete workload_metrics pass is required")
    fields, rows = read_trace(source_trace)
    replay_path = Path(result["replay_result"])
    metrics_path = Path(result["metrics_jsonl"])
    validate_replay(rows, replay_path)
    counts = acceptance_counts(
        metrics_path, {row["id"] for row in rows}, draft_tokens=draft_tokens,
        request_id_prefix=request_id_prefix,
    )
    numerators = [sum(value[0][i] for value in counts.values()) for i in range(draft_tokens)]
    denominators = [sum(value[1][i] for value in counts.values()) for i in range(draft_tokens)]
    aggregate = [n / d if d else None for n, d in zip(numerators, denominators)]
    fallback_positions = []
    for row in rows:
        rates = []
        for position, (n, d) in enumerate(zip(*counts[row["id"]])):
            if d:
                rate = n / d
            elif missing_acceptance == "run-aggregate" and aggregate[position] is not None:
                rate = aggregate[position]
                fallback_positions.append({"request_id": row["id"], "position": position})
            else:
                raise ValueError(f"missing acceptance evidence: {row['id']} position {position}")
            rates.append(rate)
        row["accept_rate"] = json.dumps(rates, separators=(",", ":"))
    manifest = {
        "schema_version": 1, "kind": "observed_conditioned_per_request_acceptance",
        "predictive_alignment": False, "draft_tokens": draft_tokens,
        "missing_acceptance": missing_acceptance, "request_id_prefix": request_id_prefix,
        "requests": len(rows), "fallback_positions": fallback_positions,
        "aggregate_rates": aggregate,
        "sources": {key: {"path": str(path.resolve()),
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                    for key, path in {"trace": source_trace, "profile_result": result_path,
                                      "replay": replay_path, "metrics": metrics_path}.items()},
        "preserved_fields": [field for field in fields if field != "accept_rate"],
        "limitations": "Observed-conditioned probabilities, not independent prediction or "
        "per-round acceptance replay. No scheduler decisions are copied.",
    }
    manifest_path = output_trace.with_suffix(output_trace.suffix + ".manifest.json")
    if output_trace.exists() or manifest_path.exists():
        raise FileExistsError("output trace or manifest already exists")
    output_trace.parent.mkdir(parents=True, exist_ok=True)
    with output_trace.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields if "accept_rate" in fields
                                else [*fields, "accept_rate"])
        writer.writeheader()
        writer.writerows(rows)
    manifest["output_trace_sha256"] = hashlib.sha256(output_trace.read_bytes()).hexdigest()
    with manifest_path.open("x") as stream:
        stream.write(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest
