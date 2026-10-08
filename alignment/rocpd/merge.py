"""Merge N per-rank rocpd parsed documents into one multi-device document.

A tensor-parallel (TP>1) rocpd capture writes one SQLite database per worker
process, and ``alignment.rocpd.parse`` turns each into its own single-device
``parsed.json`` (``tp_size: 1``, ``device_ids: [one]``) plus its
``kernel_sequences.json``. The Rust Check-1 reader, however, consumes ONE
``parsed.json`` that carries every device: it reduces ranks per-kernel-position
keyed by ``MeasuredRange.device_id`` (independent kernels take the rank-max,
collectives ``max(end) - max(start)``). This module is the one missing producer
step between the two — it merges the per-rank normalized documents into that
single multi-device document, additively and offline (no GPU, no rocprofiler).

The merge happens at the NORMALIZED parsed-document layer, not by concatenating
raw databases: the Option-B sentinel reconstruction builds one global timeline
and cannot separate concurrent worker processes, so a raw-DB merge collapses
every rank onto one device. Here each input is already a clean single-device
document, so assembling the N of them is unambiguous.

Device identity is taken from the FILENAME RANK, never the in-DB ``agent_id``.
Every worker process sees its own GPU as agent 0, so trusting the in-DB value
would collapse all ranks onto device 0. The rank parsed from each input's
filename (or an explicit ``--rank`` ordering) assigns ``device_id`` ``0..N-1``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from ..nsys.parse import write_kernel_sequences
from ..nsys.parsed_io import kernel_rows_path, read_parsed, write_parsed

#: Rank embedded in a per-rank artifact name, e.g. ``parsed.rank0.json`` or
#: ``glm_tp4_rank3_results.db``. Mirrors ``rocprof_capture.rank_from_rocpd_path``
#: but generalized to the parsed/sequence filenames the merge actually receives.
_RANK_FROM_NAME_RE = re.compile(r"rank(\d+)")


def rank_from_artifact_path(path: Path) -> int | None:
    """Parse the worker rank out of a per-rank artifact filename, or ``None``."""
    match = _RANK_FROM_NAME_RE.search(Path(path).name)
    return int(match.group(1)) if match else None


def _single_device(parsed: dict[str, Any], source: Path) -> int:
    """The one device a per-rank parsed document describes (its in-DB agent)."""
    device_ids = parsed.get("device_ids", [])
    if len(device_ids) != 1:
        raise ValueError(
            f"{source} is not a single-device per-rank document "
            f"(device_ids={device_ids}); merge takes per-rank parse outputs"
        )
    return int(device_ids[0])


def _rekey_single(block: dict[str, Any], rank: int) -> dict[str, Any]:
    """Re-key a single-device summary block onto the device the rank now owns.

    Each per-rank summary (``all`` / a ``by_stage`` stage / a ``by_phase`` entry)
    is keyed by the input's in-DB device string and holds exactly that one
    device. Re-keyed onto ``rank`` the N blocks merge without collision, so the
    merged summary honestly describes all N devices.
    """
    if len(block) > 1:
        raise ValueError(f"per-rank summary block is not single-device: {sorted(block)}")
    return {str(rank): value for value in block.values()}


def _merge_summaries(parsed_by_rank: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Merge the per-device ``all`` / ``by_stage`` / ``by_phase`` summary blocks."""
    merged_all: dict[str, Any] = {}
    merged_by_stage: dict[str, dict[str, Any]] = {}
    merged_by_phase: dict[str, dict[str, Any]] = {}
    for rank in sorted(parsed_by_rank):
        doc = parsed_by_rank[rank]
        merged_all.update(_rekey_single(doc.get("all", {}), rank))
        for stage, block in doc.get("by_stage", {}).items():
            merged_by_stage.setdefault(stage, {}).update(_rekey_single(block, rank))
        for phase, phase_doc in doc.get("by_phase", {}).items():
            entry = merged_by_phase.setdefault(phase, {"by_stage": {}, "all": {}})
            entry["all"].update(_rekey_single(phase_doc.get("all", {}), rank))
            for stage, block in phase_doc.get("by_stage", {}).items():
                entry["by_stage"].setdefault(stage, {}).update(_rekey_single(block, rank))
    return {"all": merged_all, "by_stage": merged_by_stage, "by_phase": merged_by_phase}


def merge_parsed(parsed_by_rank: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """Merge single-device per-rank parsed docs into one multi-device document.

    ``parsed_by_rank`` maps the worker rank (the authoritative device id) to a
    parsed document with its kernels inline (as ``read_parsed`` returns them).
    Each input is single-device; the output carries ``device_ids = [0..N-1]``,
    ``tp_size = N``, and one unified kernel-name id space.
    """
    ranks = sorted(parsed_by_rank)
    if not ranks:
        raise ValueError("merge needs at least one per-rank parsed document")

    for rank, doc in parsed_by_rank.items():
        _single_device(doc, Path(str(doc.get("sqlite", f"rank {rank}"))))
    sources = {rank: str(doc.get("sqlite", "")) for rank, doc in parsed_by_rank.items()}
    kinds = {doc.get("source") for doc in parsed_by_rank.values()}
    # Backend-agnostic: the per-rank docs may all be "rocpd" or all "torch" (the
    # Kineto producer), but they must agree -- merging ranks captured by different
    # backends into one document would mix incomparable timelines. The single
    # uniform kind is propagated to the merged document's `source`.
    if len(kinds) != 1:
        raise ValueError(f"inputs disagree on capture source: {sorted(kinds)}")
    source = next(iter(kinds))
    range_modes = {doc.get("range_mode") for doc in parsed_by_rank.values()}
    if len(range_modes) != 1:
        raise ValueError(f"inputs disagree on range_mode: {sorted(range_modes)}")

    # One unified name-id space. Names are interned in rank order, then by each
    # rank's own id order, so the assignment is deterministic; every rank keeps a
    # remap from its local id to the merged id.
    name_to_id: dict[str, int] = {}
    remap_by_rank: dict[int, dict[int, int]] = {}
    for rank in ranks:
        names = parsed_by_rank[rank]["kernel_names"]
        remap: dict[int, int] = {}
        for local_id in sorted(names, key=int):
            name = names[local_id]
            merged_id = name_to_id.setdefault(name, len(name_to_id) + 1)
            remap[int(local_id)] = merged_id
        remap_by_rank[rank] = remap
    merged_kernel_names = {str(merged_id): name for name, merged_id in name_to_id.items()}

    # Group ranges by iteration index, concatenating each rank's ranges under its
    # own device id. Every range's device_id is OVERWRITTEN with the filename
    # rank — the in-DB agent_id is untrustworthy (0 on every worker).
    by_iteration: dict[int, dict[str, Any]] = {}
    for rank in ranks:
        doc = parsed_by_rank[rank]
        remap = remap_by_rank[rank]
        for detail in doc["iteration_details"]:
            iteration = int(detail["iteration"])
            merged_detail = by_iteration.setdefault(
                iteration,
                {
                    "iteration": iteration,
                    "iteration_type": detail.get("iteration_type"),
                    "stage": detail.get("stage"),
                    "reference_device_id": ranks[0],
                    "iteration_index_by_device": {},
                    "devices_absent": [],
                    "metrics": detail.get("metrics"),
                    # The offline rocpd database carries no host module-load
                    # calls, so these stay 0 (the honest value on this path).
                    "jit_module_loads": 0,
                    "jit_stall_ns": 0,
                    "metrics_by_dp_rank": {},
                    "ranges": [],
                },
            )
            for range_row in detail["ranges"]:
                merged_range = dict(range_row)
                merged_range["device_id"] = rank
                merged_range["dp_rank"] = 0
                merged_range["kernels"] = [
                    {**kernel, "name_id": remap[int(kernel["name_id"])]}
                    for kernel in range_row.get("kernels", [])
                ]
                merged_detail["ranges"].append(merged_range)
                merged_detail["iteration_index_by_device"][str(rank)] = int(
                    range_row.get("iteration_index", iteration)
                )

    # Finalize each merged iteration: canonical range order (device, start,
    # phase), present/absent device bookkeeping across all N devices.
    for iteration, detail in by_iteration.items():
        detail["ranges"].sort(
            key=lambda row: (row["device_id"], row["start_ns"], row["phase"])
        )
        present = {int(row["device_id"]) for row in detail["ranges"]}
        detail["devices_absent"] = [rank for rank in ranks if rank not in present]
        detail["iteration_index_by_device"] = {
            device: detail["iteration_index_by_device"][device]
            for device in sorted(
                detail["iteration_index_by_device"], key=lambda key: int(key)
            )
        }
    merged_details = [by_iteration[iteration] for iteration in sorted(by_iteration)]
    iterations = sorted(by_iteration)

    base = parsed_by_rank[ranks[0]]
    stages = sorted({stage for doc in parsed_by_rank.values() for stage in doc.get("stages", [])})
    phases = sorted({phase for doc in parsed_by_rank.values() for phase in doc.get("phases", [])})
    summaries = _merge_summaries(parsed_by_rank)

    merged = {
        "schema_version": base.get("schema_version"),
        "step_alignment": {
            "rule": "iteration_index",
            "reference_device_id": ranks[0],
            "steps": len(merged_details),
            "unpaired_steps_by_device": {},
        },
        "sqlite": ", ".join(sources[rank] for rank in ranks),
        "source": source,
        "iteration_start": base.get("iteration_start"),
        "iteration_end": base.get("iteration_end"),
        "range_mode": base.get("range_mode"),
        "iterations": iterations,
        "scanned_kernel_rows": sum(
            int(doc.get("scanned_kernel_rows", 0)) for doc in parsed_by_rank.values()
        ),
        "jit_module_loads": 0,
        "stages": stages,
        "phases": phases,
        "kernel_names": merged_kernel_names,
        "iteration_details": merged_details,
        "device_ids": list(ranks),
        "dp_rank_by_device": {str(rank): 0 for rank in ranks},
        "tp_size": len(ranks),
        "kernel_sequences": {},  # filled by merge_documents from the sequence docs
        "by_phase": summaries["by_phase"],
        "by_stage": summaries["by_stage"],
        "all": summaries["all"],
    }
    return merged


def merge_sequences(
    sequences_by_rank: dict[int, dict[str, Any]], device_ids: list[int]
) -> dict[str, dict[str, Any]]:
    """Merge the per-rank folded kernel-sequence catalogs into one union catalog.

    Each per-rank catalog folds one device, so a sequence's occurrences all
    belong to that rank; the occurrence is re-stamped with the filename rank.
    Tensor-parallel ranks are symmetric and share a ``sequence_id`` (a hash of
    the ordered kernel NAMES, so it is invariant to the id remap), which collapses
    them onto one entry whose ``occurrences`` then list every rank. A collision
    with divergent folded ``tracks`` is a contradiction and raises rather than
    silently mislabel a device.
    """
    merged: dict[str, dict[str, dict[str, Any]]] = {}
    for rank in sorted(sequences_by_rank):
        doc = sequences_by_rank[rank]
        for phase, phase_doc in doc["phases"].items():
            phase_merged = merged.setdefault(phase, {})
            for sequence in phase_doc["unique_sequences"]:
                sequence_id = sequence["sequence_id"]
                iterations = sorted(
                    {
                        iteration
                        for occurrence in sequence["occurrences"]
                        for iteration in occurrence["iterations"]
                    }
                )
                occurrence = {"device_id": rank, "iterations": iterations}
                existing = phase_merged.get(sequence_id)
                if existing is None:
                    phase_merged[sequence_id] = {
                        "sequence_id": sequence_id,
                        "occurrences": [occurrence],
                        "expanded_kernel_count": sequence["expanded_kernel_count"],
                        "tracks": sequence["tracks"],
                    }
                    continue
                if existing["tracks"] != sequence["tracks"]:
                    raise ValueError(
                        f"sequence {sequence_id!r} folds differently on device {rank}"
                    )
                existing["occurrences"].append(occurrence)

    catalogs: dict[str, dict[str, Any]] = {}
    for phase, phase_merged in merged.items():
        sequences = []
        for entry in phase_merged.values():
            entry["occurrences"].sort(key=lambda occurrence: occurrence["device_id"])
            sequences.append(entry)
        catalogs[phase] = {
            "unique_sequences": sorted(
                sequences,
                key=lambda sequence: (
                    min(
                        iteration
                        for occurrence in sequence["occurrences"]
                        for iteration in occurrence["iterations"]
                    ),
                    sequence["sequence_id"],
                ),
            )
        }
    return catalogs


def merge_documents(
    parsed_by_rank: dict[int, dict[str, Any]],
    sequences_by_rank: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Merge both the parsed docs and the folded sequence catalogs by rank.

    Returns the merged parsed document, with its ``kernel_sequences`` field set to
    the merged union catalog so the standalone ``kernel_sequences.json`` written
    by ``write_kernel_sequences`` and the inline copy stay consistent.
    """
    if sorted(parsed_by_rank) != sorted(sequences_by_rank):
        raise ValueError(
            "parsed and sequence inputs cover different ranks: "
            f"{sorted(parsed_by_rank)} vs {sorted(sequences_by_rank)}"
        )
    merged = merge_parsed(parsed_by_rank)
    merged["kernel_sequences"] = merge_sequences(sequences_by_rank, merged["device_ids"])
    return merged


def _resolve_ranks(paths: list[Path], explicit: list[int] | None) -> list[int]:
    """One rank per input path, from ``--rank`` ordering or the filename."""
    if explicit:
        if len(explicit) != len(paths):
            raise ValueError(
                f"--rank gives {len(explicit)} values for {len(paths)} parsed inputs"
            )
        return list(explicit)
    ranks = []
    for path in paths:
        rank = rank_from_artifact_path(path)
        if rank is None:
            raise ValueError(
                f"cannot parse a rank from {path.name!r}; pass --rank to order inputs explicitly"
            )
        ranks.append(rank)
    return ranks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge N per-rank rocpd parsed docs into one multi-device document"
    )
    parser.add_argument(
        "--parsed",
        type=Path,
        nargs="+",
        required=True,
        help="per-rank parsed.json inputs (one per worker rank)",
    )
    parser.add_argument(
        "--sequences",
        type=Path,
        nargs="+",
        required=True,
        help="per-rank kernel_sequences.json inputs, in the same order as --parsed",
    )
    parser.add_argument(
        "--rank",
        type=int,
        nargs="*",
        default=None,
        help="explicit device-id order for --parsed inputs (default: 'rankN' from each filename)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="write the merged parsed.json here (and its .kernels.parquet sibling)",
    )
    parser.add_argument(
        "--sequences-output",
        type=Path,
        required=True,
        help="write the merged kernel_sequences.json here",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if len(args.parsed) != len(args.sequences):
        raise ValueError(
            f"{len(args.parsed)} parsed inputs but {len(args.sequences)} sequence inputs; "
            "they pair by position"
        )
    ranks = _resolve_ranks(args.parsed, args.rank)
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"ranks are not distinct: {ranks}")

    parsed_by_rank: dict[int, dict[str, Any]] = {}
    sequences_by_rank: dict[int, dict[str, Any]] = {}
    for rank, parsed_path, sequences_path in zip(ranks, args.parsed, args.sequences, strict=True):
        parsed = read_parsed(Path(parsed_path))
        _single_device(parsed, Path(parsed_path))
        parsed_by_rank[rank] = parsed
        sequences_by_rank[rank] = json.loads(Path(sequences_path).read_text())

    merged = merge_documents(parsed_by_rank, sequences_by_rank)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.sequences_output).parent.mkdir(parents=True, exist_ok=True)
    write_parsed(Path(args.output), merged)
    print(f"wrote {args.output} and {kernel_rows_path(Path(args.output)).name}")
    write_kernel_sequences(args.sequences_output, merged, args.output)
    print(f"wrote {args.sequences_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
