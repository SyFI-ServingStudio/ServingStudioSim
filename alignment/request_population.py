"""Identity and request-contract checks for independent CSV workloads."""

import csv
import hashlib
import json
import math
from pathlib import Path


def read_trace(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not {"id", "input_len", "output_len", "arrival_time"}.issubset(fields):
            raise ValueError("independent trace requires id/input_len/output_len/arrival_time")
        rows = list(reader)
    ids = [row["id"] for row in rows]
    if not rows or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("trace requires nonempty, distinct request IDs")
    for row in rows:
        arrival = float(row["arrival_time"])
        if (int(row["input_len"]) <= 0 or int(row["output_len"]) <= 0
                or not math.isfinite(arrival) or arrival < 0):
            raise ValueError(f"invalid request geometry: {row['id']}")
    return fields, rows


def arrival_time_scale(rows: list[dict], measured: dict) -> float:
    """The factor req-frontend's `--rate` applied to the trace's arrival offsets.

    `apply_arrival_rate` multiplies every offset by one `trace_rate / target_rate`
    (1 when no rate is given), so the replay records scaled arrivals. Read at the
    latest arrival, where it is best conditioned; every row must then agree.
    """
    anchor = max(rows, key=lambda row: float(row["arrival_time"]))
    base = float(anchor["arrival_time"])
    if base <= 0:
        return 1.0
    scale = measured[anchor["id"]]["source"]["data"]["arrival_time_ms"] / base
    if not math.isfinite(scale) or scale < 0:
        raise ValueError(f"invalid replay arrival-time scale {scale}")
    return scale


# A simulator replaying the trace at another arrival scale than the measured run
# (e.g. `request_rate` left at its default) compares different loads.
ARRIVAL_TIME_SCALE_TOLERANCE = 0.10


def simulated_arrival_time_scale(rows: list[dict], simulated: dict) -> float | None:
    """The factor the simulator applied to the trace's arrival offsets.

    Dense simulator IDs follow CSV row order. Read at the latest arrival, like
    `arrival_time_scale`; `None` when that request is missing from the run.
    """
    index, anchor = max(enumerate(rows), key=lambda item: float(item[1]["arrival_time"]))
    base = float(anchor["arrival_time"])
    if base <= 0:
        return 1.0
    sim = simulated.get(index)
    if sim is None or sim.get("arrival_time_ms") is None:
        return None
    return float(sim["arrival_time_ms"]) / base


def arrival_time_scales_agree(measured_scale: float, simulated_scale: float | None) -> bool:
    if simulated_scale is None or not math.isfinite(simulated_scale):
        return False
    if measured_scale == 0:
        return simulated_scale == 0
    return (
        abs(simulated_scale - measured_scale) / measured_scale <= ARRIVAL_TIME_SCALE_TOLERANCE
    )


def validate_replay(rows: list[dict], replay_path: Path) -> dict:
    with replay_path.open() as stream:
        replay = [json.loads(line) for line in stream if line.strip()]
    measured = {row["source"]["data"]["id"]: row for row in replay}
    if len(measured) != len(replay) or set(measured) != {row["id"] for row in rows}:
        raise ValueError("measured request population differs from trace")
    scale = arrival_time_scale(rows, measured)
    for row in rows:
        entry = measured[row["id"]]
        source, outcome = entry["source"]["data"], entry["outcome"]
        if not (
            outcome["status"] == "SUCCESS"
            and int(row["input_len"]) == source["input_len"]
            and int(row["output_len"]) == source["output_len_target"]
            == outcome["output_len_actual"]
            and math.isclose(
                source["arrival_time_ms"], float(row["arrival_time"]) * scale,
                rel_tol=1e-9, abs_tol=1e-6,
            )
        ):
            raise ValueError(f"request lengths, arrival or completion differ: {row['id']}")
    return measured


def audit_request_population(
    *, trace_path: Path, replay_path: Path, slo_path: Path,
    simulated_arrival_mode: str | None = None,
) -> dict:
    """Dense simulator IDs follow independent CSV row order, before scheduling.

    A `trace_timed` simulation must also replay the trace at the measured
    arrival scale, within `ARRIVAL_TIME_SCALE_TOLERANCE`.
    """
    import pyarrow.parquet as pq

    _, rows = read_trace(trace_path)
    measured = validate_replay(rows, replay_path)
    measured_scale = arrival_time_scale(rows, measured)
    check_arrival_scale = simulated_arrival_mode == "trace_timed"
    columns = ["request_id", "completed", "fresh_prompt_tokens", "num_output_tokens"]
    if check_arrival_scale:
        columns.append("arrival_time_ms")
    simulated_rows = pq.read_table(slo_path, columns=columns).to_pylist()
    simulated = {row["request_id"]: row for row in simulated_rows}
    checks = {
        "unique_simulated_ids": len(simulated) == len(simulated_rows),
        "simulated_population": set(simulated) == set(range(len(rows))),
    }
    mismatches = []
    for index, row in enumerate(rows):
        sim = simulated.get(index)
        if sim is None or not (
            sim["completed"] and sim["fresh_prompt_tokens"] == int(row["input_len"])
            and sim["num_output_tokens"] == int(row["output_len"])
        ):
            mismatches.append(row["id"])
    checks["per_request_lengths_arrivals_and_completion"] = not mismatches
    simulated_scale = None
    if check_arrival_scale:
        simulated_scale = simulated_arrival_time_scale(rows, simulated)
        checks["arrival_time_scale_matches"] = arrival_time_scales_agree(
            measured_scale, simulated_scale
        )
    return {
        "available": True, "all_ok": all(checks.values()), "requests": len(rows),
        "arrival_time_scale": measured_scale,
        "simulated_arrival_time_scale": simulated_scale,
        "checks": checks, "mismatched_source_ids": mismatches,
        "identities": [{"request_id": i, "source_id": row["id"]} for i, row in enumerate(rows)],
        "mapping_basis": "independent CSV row order before scheduling",
        "limitations": "Declared arrivals are checked; concurrency may delay actual release. "
        "Latency samples remain unpaired distributions.",
    }


def audit_alignment_population(manifest_path: Path, *, repo_root: Path) -> dict:
    """Audit supported independent runs using the analysis manifest's actual inputs."""
    if not manifest_path.is_file():
        return {"available": False, "reason": "analysis manifest absent; counts-only legacy check"}
    manifest = json.loads(manifest_path.read_text())
    simulation = Path(manifest["simulation_log_dir"])
    params = json.loads((simulation / "raw/params.json").read_text())
    workload = params.get("workload", {})
    traces = workload.get("trace_files", [])
    if (workload.get("input_file_format") != "text-generation-independent"
            or "session" in workload.get("input_file_tags", []) or len(traces) != 1
            or Path(traces[0]).suffix.lower() != ".csv"):
        return {"available": False, "reason": "identity audit requires one independent CSV trace"}
    trace = Path(traces[0])
    if not trace.is_absolute():
        trace = repo_root / trace
    audit = audit_request_population(
        trace_path=trace, replay_path=Path(manifest["replay_result"]),
        slo_path=simulation / "raw/request_slo.parquet",
        simulated_arrival_mode=workload.get("arrival_mode"),
    )
    sidecar = trace.with_suffix(trace.suffix + ".manifest.json")
    if sidecar.is_file():
        preparation = json.loads(sidecar.read_text())
        if preparation.get("output_trace_sha256") != hashlib.sha256(trace.read_bytes()).hexdigest():
            raise ValueError("input preparation manifest does not match the simulation trace")
        audit["input_preparation"] = preparation
    return audit
