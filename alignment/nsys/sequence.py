"""Build a lossless folded inventory of per-iteration kernel sequences.

The measured timeline remains the ground truth.  This module removes only exact
contiguous repetition from the separate human-labeling surface; expanding a
program must reproduce the original ordered ``(name, category)`` sequence.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

KernelOccurrence = dict[str, str]


def build_kernel_sequences(
    iteration_details: list[dict[str, Any]], kernel_names: dict[int, str]
) -> dict[str, dict[str, Any]]:
    """Deduplicate exact phase sequences and fold their contiguous repeats."""
    grouped: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    for detail in iteration_details:
        ranges_by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for range_row in detail.get("ranges", []):
            ranges_by_phase[str(range_row["phase"])].extend(range_row.get("kernels", []))
        for phase, kernels in ranges_by_phase.items():
            if not kernels:
                continue
            name_ids = tuple(int(kernel["name_id"]) for kernel in kernels)
            key = (phase, name_ids)
            categories = tuple(str(kernel["category"]) for kernel in kernels)
            entry = grouped.get(key)
            if entry is None:
                occurrences = [
                    {
                        "name": kernel_names[name_id],
                        "suggested_category": category,
                    }
                    for name_id, category in zip(name_ids, categories, strict=True)
                ]
                sequence_id = _sequence_id(phase, name_ids, kernel_names)
                grouped[key] = {
                    "sequence_id": sequence_id,
                    "iterations": [int(detail["iteration"])],
                    "expanded_kernel_count": len(occurrences),
                    "program": _fold_program(occurrences),
                }
            else:
                existing_categories = tuple(
                    kernel["suggested_category"] for kernel in expand_program(entry["program"])
                )
                if existing_categories != categories:
                    raise ValueError(f"phase {phase!r} sequence has inconsistent kernel categories")
                entry["iterations"].append(int(detail["iteration"]))

    catalogs: dict[str, dict[str, Any]] = {}
    for (phase, _), sequence in sorted(
        grouped.items(), key=lambda item: min(item[1]["iterations"])
    ):
        catalogs.setdefault(phase, {"unique_sequences": []})["unique_sequences"].append(sequence)
    return catalogs


def expand_program(program: list[dict[str, Any]]) -> list[KernelOccurrence]:
    """Expand folded nodes; shared by validation tests and artifact consumers."""
    expanded: list[KernelOccurrence] = []
    for node in program:
        if "kernels" in node:
            expanded.extend(node["kernels"])
            continue
        repeat = node["repeat"]
        for _ in range(int(repeat["count"])):
            expanded.extend(repeat["body"]["kernels"])
    return expanded


def _fold_program(kernels: list[KernelOccurrence]) -> list[dict[str, Any]]:
    """Recursively choose profitable exact repeats and keep literals ordered."""
    program = _fold_segment(kernels, absolute_start=0)
    expanded = expand_program(program)
    if expanded != kernels:
        raise AssertionError("folded kernel program is not lossless")
    return program


def _fold_segment(kernels: list[KernelOccurrence], *, absolute_start: int) -> list[dict[str, Any]]:
    best: tuple[int, int, int, int, int, int, int] | None = None
    tokens = [(kernel["name"], kernel["suggested_category"]) for kernel in kernels]
    for start in range(len(tokens)):
        for width in range(1, (len(tokens) - start) // 2 + 1):
            count = 1
            while (
                start + (count + 1) * width <= len(tokens)
                and tokens[start : start + width]
                == tokens[start + count * width : start + (count + 1) * width]
            ):
                count += 1
            saved_occurrences = (count - 1) * width
            if saved_occurrences < 2:
                continue
            suffix = len(tokens) - start - count * width
            stored_occurrences = start + width + suffix
            aligned = 0 if (absolute_start + start) % width == 0 else 1
            candidate = (
                stored_occurrences,
                aligned,
                suffix,
                -count,
                start,
                width,
                count,
            )
            if best is None or candidate < best:
                best = candidate

    if best is None:
        return [{"kernels": kernels}] if kernels else []

    _, _, suffix, _, start, width, count = best
    repeat_end = start + width * count
    program = _fold_segment(kernels[:start], absolute_start=absolute_start)
    program.append(
        {
            "repeat": {
                "count": count,
                "body": {"kernels": kernels[start : start + width]},
            }
        }
    )
    program.extend(_fold_segment(kernels[repeat_end:], absolute_start=absolute_start + repeat_end))
    return _merge_literal_nodes(program)


def _merge_literal_nodes(program: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for node in program:
        if merged and "kernels" in merged[-1] and "kernels" in node:
            merged[-1]["kernels"].extend(node["kernels"])
        else:
            merged.append(node)
    return merged


def _sequence_id(
    phase: str,
    name_ids: tuple[int, ...],
    kernel_names: dict[int, str],
) -> str:
    identity = {
        "phase": phase,
        "kernel_names": [kernel_names[name_id] for name_id in name_ids],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    return f"sequence_{digest}"
