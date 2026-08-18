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

import numpy as np

KernelOccurrence = dict[str, str]


def build_kernel_sequences(
    iteration_details: list[dict[str, Any]], kernel_names: dict[int, str]
) -> dict[str, dict[str, Any]]:
    """Build the union catalog of every device's phase sequences."""
    catalogs, _ = build_device_kernel_sequences(iteration_details, kernel_names)
    return catalogs


def build_device_kernel_sequences(
    iteration_details: list[dict[str, Any]], kernel_names: dict[int, str]
) -> tuple[dict[str, dict[str, Any]], list[int]]:
    """Build one catalog holding the union of every device's phase sequences.

    Each device is folded independently — merging several devices' ranges into
    one sequence would serialize concurrent ranks and multiply the apparent model
    work. The per-device catalogs are then unioned: `sequence_id` is a hash of
    the ordered kernel names, so devices that ran the identical sequence collapse
    onto one entry and one labeling decision, while devices that genuinely
    diverge each keep their own entry.

    Divergence is normal, not a defect. Tensor-parallel ranks are symmetric and
    collapse to a single entry; data-parallel ranks each schedule their own batch
    and routinely differ on the steps where one rank has a prefill chunk and
    another does not.

    Every entry records `occurrences` — which (device, iteration) pairs executed
    it — so a label applies to exactly the positions it was derived from.
    """
    device_ids = sorted(
        {
            int(range_row["device_id"])
            for detail in iteration_details
            for range_row in detail.get("ranges", [])
            if range_row.get("device_id") is not None and range_row.get("kernels")
        }
    )
    # Small unit fixtures written before the normalized schema carried an
    # explicit device id represent the original single-rank contract.
    if not device_ids and any(
        range_row.get("kernels")
        for detail in iteration_details
        for range_row in detail.get("ranges", [])
    ):
        device_ids = [0]
    if not device_ids:
        raise ValueError("kernel sequence inventory has no kernel-bearing device")

    merged: dict[str, dict[str, dict[str, Any]]] = {}
    for device_id in device_ids:
        catalog = _build_device_kernel_sequences(iteration_details, kernel_names, device_id)
        for phase, phase_catalog in catalog.items():
            phase_merged = merged.setdefault(phase, {})
            for sequence in phase_catalog["unique_sequences"]:
                sequence_id = sequence["sequence_id"]
                existing = phase_merged.get(sequence_id)
                occurrence = {
                    "device_id": device_id,
                    "iterations": sequence["iterations"],
                }
                if existing is None:
                    phase_merged[sequence_id] = {
                        "sequence_id": sequence_id,
                        "occurrences": [occurrence],
                        "expanded_kernel_count": sequence["expanded_kernel_count"],
                        "program": sequence["program"],
                    }
                    continue
                # Same id means the same ordered (name, category) sequence, so the
                # folded program is identical by construction; assert rather than
                # trust, because a mismatch would silently mislabel a device.
                if existing["program"] != sequence["program"]:
                    raise ValueError(
                        f"sequence {sequence_id!r} folds differently on device {device_id}"
                    )
                existing["occurrences"].append(occurrence)

    catalogs: dict[str, dict[str, Any]] = {}
    for phase, phase_merged in merged.items():
        catalogs[phase] = {
            "unique_sequences": sorted(
                phase_merged.values(),
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
    return catalogs, device_ids


def _build_device_kernel_sequences(
    iteration_details: list[dict[str, Any]], kernel_names: dict[int, str], device_id: int
) -> dict[str, dict[str, Any]]:
    """Deduplicate and fold exact phase sequences for one physical rank."""
    grouped: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    for detail in iteration_details:
        ranges_by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for range_row in detail.get("ranges", []):
            if int(range_row.get("device_id", 0)) != device_id:
                continue
            ranges_by_phase[str(range_row["phase"])].extend(range_row.get("kernels", []))
        for phase, kernels in ranges_by_phase.items():
            if not kernels:
                continue
            kernels = _stream_major(kernels)
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


def _stream_major(kernels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group each CUDA stream's kernels contiguously, keeping their launch order.

    The rows arrive sorted by start time across every stream, which reads like an
    execution order but is not one when streams run concurrently: two independent
    kernels that overlap can land either way round, and nanoseconds of jitter flip
    them between otherwise identical iterations. Folding is exact-match, so each
    flip both breaks a layer repeat and forks a new "unique" sequence. On a
    Qwen3.6 capture that turned one recurring decode shape into 909 sequences
    folding to 92% of their expanded size — a labeling surface roughly five times
    larger than the work it describes, none of the excess meaningful.

    This is the same rule the per-device split above already follows, applied one
    level down: concurrent tracks are kept apart rather than interleaved, because
    a single ordered list of concurrent work asserts a serialization that did not
    happen. Sorting is stable and keyed on the stream alone, so within a stream —
    where execution really is ordered — the original start order survives intact.

    Position in the folded sequence is a labeling coordinate, not evidence;
    durations and overlap are read from each kernel's own timestamps, which this
    does not touch.
    """
    return sorted(kernels, key=lambda kernel: int(kernel.get("stream_id", 0)))


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


def _best_repeat(tokens: list[int], absolute_start: int) -> tuple[int, ...] | None:
    """Lowest-ranking `(stored, aligned, suffix, -count, start, width, count)`.

    Every `(start, width)` pair with `saved_occurrences >= 2` contributes exactly
    one candidate, and `start`/`width` are themselves in the tuple, so the
    minimum is unique and independent of the order pairs are visited in. That is
    what lets this iterate width-major while the definition reads start-major.

    The definition asks, for each pair, how many consecutive `width`-blocks from
    `start` are equal. Comparing blocks costs `O(width)` each, which makes the
    obvious triple loop `O(n^3)` — minutes per iteration on a real capture. But
    "the first k blocks are equal" is exactly "`tokens[start:]` and
    `tokens[start + width:]` share a prefix of at least `(k - 1) * width`", so

        count = 1 + lcp(start, start + width) // width

    and for one fixed `width` the whole `lcp` diagonal falls out of a single
    backward pass. That is `O(n)` per width, `O(n^2)` overall, with no change to
    which repeat is chosen.
    """
    n = len(tokens)
    if n < 2:
        return None
    token_array = np.asarray(tokens, dtype=np.int64)
    positions = np.arange(n, dtype=np.int64)

    best: tuple[int, ...] | None = None
    best_saved = 1  # `saved_occurrences < 2` never qualifies, so start just below
    for width in range(1, n // 2 + 1):
        # `saved = (count - 1) * width <= n - start - width <= n - width`, and
        # `n - width` only shrinks as width grows: once the best possible saving
        # at this width cannot match what is already banked, no wider repeat can
        # either. On a layer-stack sequence the winner is a narrow body repeated
        # many times, so this lands early and retires most of the width range.
        if n - width < best_saved:
            break

        # lcp(i, i + width) for every i at once: mark each mismatch with its own
        # index and every match with `span`, then a reverse running minimum turns
        # that into "index of the next mismatch at or after i".
        span = n - width
        matches = token_array[:span] == token_array[width:]
        marked = np.where(matches, span, positions[:span])
        next_mismatch = np.minimum.accumulate(marked[::-1])[::-1]
        last_start = span - width  # from `width <= (n - start) // 2`
        lcp = next_mismatch[: last_start + 1] - positions[: last_start + 1]

        saved = lcp // width * width
        # Equality is kept, not dropped: a candidate that only ties the banked
        # saving can still win on alignment or suffix below.
        qualifying = np.flatnonzero(saved >= max(2, best_saved))
        if qualifying.size == 0:
            continue
        # The primary key is `n - saved`, so within one width only the starts
        # achieving its maximum saving can ever win; the rest are dominated.
        width_saved = int(saved[qualifying].max())
        starts = qualifying[saved[qualifying] == width_saved]
        # Remaining tie-break at fixed (width, saved): `aligned` first, then the
        # smallest suffix — and suffix shrinks as start grows, so that is the
        # largest start. `start` itself never decides, being determined by suffix.
        aligned_starts = starts[(absolute_start + starts) % width == 0]
        start = int((aligned_starts if aligned_starts.size else starts).max())

        count = width_saved // width + 1
        candidate = (
            n - width_saved,  # == start + width + suffix
            0 if (absolute_start + start) % width == 0 else 1,
            n - start - count * width,
            -count,
            start,
            width,
            count,
        )
        if best is None or candidate < best:
            best = candidate
            best_saved = width_saved
    return best


def _fold_segment(kernels: list[KernelOccurrence], *, absolute_start: int) -> list[dict[str, Any]]:
    # Intern to ints so the hot equality test is a machine compare rather than a
    # two-string tuple compare.
    token_ids: dict[tuple[str, str], int] = {}
    tokens = [
        token_ids.setdefault(
            (kernel["name"], kernel["suggested_category"]), len(token_ids)
        )
        for kernel in kernels
    ]
    best = _best_repeat(tokens, absolute_start)

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
