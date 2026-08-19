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
                        "tracks": sequence["tracks"],
                    }
                    continue
                # Same id means the same ordered (name, category) sequence, so the
                # folded program is identical by construction; assert rather than
                # trust, because a mismatch would silently mislabel a device.
                if existing["tracks"] != sequence["tracks"]:
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
    """Deduplicate and fold exact phase sequences for one physical rank.

    Folding runs per concurrent track, never across one. `parse` already
    serialized each range track-major, so the tracks are contiguous runs of the
    phase's kernel list and this only has to cut on the boundary — which keeps
    the flat expansion, and therefore `sequence_id:expanded_ordinal`, exactly the
    order the analyzer validates against.
    """
    grouped: dict[tuple[str, tuple[tuple[int, ...], ...]], dict[str, Any]] = {}
    for detail in iteration_details:
        ranges_by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for range_row in detail.get("ranges", []):
            if int(range_row.get("device_id", 0)) != device_id:
                continue
            ranges_by_phase[str(range_row["phase"])].extend(range_row.get("kernels", []))
        for phase, kernels in ranges_by_phase.items():
            if not kernels:
                continue
            runs = _track_runs(kernels)
            track_name_ids = tuple(
                tuple(int(kernel["name_id"]) for kernel in run) for _, run in runs
            )
            key = (phase, track_name_ids)
            track_categories = tuple(
                tuple(str(kernel["category"]) for kernel in run) for _, run in runs
            )
            entry = grouped.get(key)
            if entry is None:
                tracks = []
                for track_index, ((range_track, _), name_ids, categories) in enumerate(
                    zip(runs, track_name_ids, track_categories, strict=True)
                ):
                    occurrences = [
                        {
                            "name": kernel_names[name_id],
                            "suggested_category": category,
                        }
                        for name_id, category in zip(name_ids, categories, strict=True)
                    ]
                    tracks.append(
                        {
                            "track_index": track_index,
                            "stream_role": "primary" if range_track == 0 else "concurrent",
                            "kernel_count": len(occurrences),
                            "program": _fold_program(occurrences),
                        }
                    )
                grouped[key] = {
                    "sequence_id": _sequence_id(phase, track_name_ids, kernel_names),
                    "iterations": [int(detail["iteration"])],
                    "expanded_kernel_count": sum(len(run) for _, run in runs),
                    "tracks": tracks,
                }
            else:
                existing_categories = tuple(
                    tuple(
                        kernel["suggested_category"] for kernel in expand_program(track["program"])
                    )
                    for track in entry["tracks"]
                )
                if existing_categories != track_categories:
                    raise ValueError(f"phase {phase!r} sequence has inconsistent kernel categories")
                entry["iterations"].append(int(detail["iteration"]))

    catalogs: dict[str, dict[str, Any]] = {}
    for (phase, _), sequence in sorted(
        grouped.items(), key=lambda item: min(item[1]["iterations"])
    ):
        catalogs.setdefault(phase, {"unique_sequences": []})["unique_sequences"].append(sequence)
    return catalogs


def _track_runs(kernels: list[dict[str, Any]]) -> list[tuple[int, list[dict[str, Any]]]]:
    """Cut the phase's serialized kernels into maximal single-track runs.

    Returns `(range track index, kernels)` per run, in order. `parse` emits each
    range track-major, so a track is already a contiguous run and the cut is just
    where `track_index` changes. Two ranges of the same phase that both end and
    begin on the primary track therefore stay one run — which is what folding
    across a range boundary did before tracks existed, so a single-stream capture
    is one run and folds exactly as it always has.

    This is the same rule the per-device split already follows, one level down:
    concurrent work is kept apart rather than interleaved, because a single
    ordered list of concurrent kernels asserts a serialization that did not
    happen. Folding is exact-match, so an interleaving that jitter can flip both
    breaks a layer repeat and forks a new "unique" sequence — on a Qwen3.6
    capture, one recurring decode shape became 909 sequences that still folded to
    92% of their expanded size.

    A run is a labeling coordinate, not evidence: durations and overlap are read
    from each kernel's own timestamps, which this does not touch.
    """
    runs: list[tuple[int, list[dict[str, Any]]]] = []
    for kernel in kernels:
        track_index = int(kernel.get("track_index", 0))
        if not runs or runs[-1][0] != track_index:
            runs.append((track_index, []))
        runs[-1][1].append(kernel)
    return runs


def expand_sequence(sequence: dict[str, Any]) -> list[KernelOccurrence]:
    """Expand a whole sequence: its tracks concatenated, in canonical order.

    This is the flat order `parse` serialized and the analyzer validates against,
    so the position of a kernel here is its `expanded_ordinal`.
    """
    return [
        kernel
        for track in sequence["tracks"]
        for kernel in expand_program(track["program"])
    ]


def expand_program(program: list[dict[str, Any]]) -> list[KernelOccurrence]:
    """Expand one track's folded nodes; shared by tests and artifact consumers."""
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
    track_name_ids: tuple[tuple[int, ...], ...],
    kernel_names: dict[int, str],
) -> str:
    # The track split is part of the identity: the same flat kernel names cut
    # into different concurrent tracks are different programs, and collapsing
    # them onto one id would hand both the same labeling decision.
    identity = {
        "phase": phase,
        "kernel_names": [
            [kernel_names[name_id] for name_id in name_ids] for name_ids in track_name_ids
        ],
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:12]
    return f"sequence_{digest}"
