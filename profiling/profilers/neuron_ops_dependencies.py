"""Observed Neuron DMA completions and engine waits, without timing prediction.

DMA packets lack transfer IDs when DGE notifications are disabled. Instruction
semaphore increments and observed AXI updates still identify completed transfers.
All intervals here are measured queueing-inclusive intervals, never standalone
service costs. Engine waits overlap work on other engines.
"""

from __future__ import annotations

import bisect
import re
from collections import Counter, defaultdict

_TRIGGER = re.compile(r"semaphore=(\d+) sem_increment=(\d+)")
_WAIT = re.compile(r"\$?S\[(\d+)\](?: \((\w*)\))?(>=|==)(\d+)")


def observed_dependencies(instructions: list[dict], updates: list[dict]) -> dict:
    """Join DMA triggers to reset/counter completions and explain engine gaps.

    Instruction dictionaries use the export column names plus ``operation``.
    Every match keeps its evidence and IDs, so a downstream view can distinguish
    an observed counter completion from a nearby engine-end inference.
    """
    sems = defaultdict(list)
    for row in updates:
        key = (row["pcore_idx"], int(re.search(r"\d+", str(row["id"])).group()))
        sems[key].append(row)
    for values in sems.values():
        values.sort(key=lambda row: row["timestamp"])
    times = {key: [row["timestamp"] for row in values] for key, values in sems.items()}
    resets = {
        key: [row for row in values if row["update_operation_type"] == "sem_wr_imm"]
        for key, values in sems.items()
    }
    reset_times = {key: [row["timestamp"] for row in values] for key, values in resets.items()}
    completions = {}
    epoch_quality = {}
    for key, values in sems.items():
        for row in values:
            epoch = bisect.bisect_right(reset_times[key], row["timestamp"]) - 1
            if row["update_operation_type"] != "sem_wr_imm":
                counter_key = (*key, epoch)
                unexpected, maximum = epoch_quality.get(counter_key, (False, 0))
                epoch_quality[counter_key] = (
                    unexpected
                    or row["update_source"] != "AXI"
                    or row["update_operation_type"] != "sem_inc",
                    max(maximum, row["value"]),
                )
            if row["update_source"] == "AXI":
                completions.setdefault((*key, epoch, row["value"]), row)

    counters = Counter()
    transfers, by_completion = [], {}
    transfer_epochs = defaultdict(list)
    ordered = sorted(instructions, key=lambda row: (row["start_ts"], row["id"]))
    for inst in ordered:
        match = _TRIGGER.search(inst.get("operands") or "")
        if not match:
            continue
        semaphore, increment = map(int, match.groups())
        key = inst["pcore_idx"], semaphore
        epoch = bisect.bisect_right(reset_times.get(key, []), inst["start_ts"]) - 1
        counter_key = (*key, epoch)
        counters[counter_key] += increment
        threshold = resets[key][epoch]["value"] + counters[counter_key] if epoch >= 0 else None
        hit = completions.get((*counter_key, threshold)) if epoch >= 0 else None
        if hit and hit["timestamp"] < inst["end_ts"]:
            hit = None
        transfer = {
            "trigger_id": inst["id"],
            "pcore_idx": inst["pcore_idx"],
            "timeline_track": inst["timeline_track"],
            "operation": inst["operation"],
            "semaphore": semaphore,
            "reset_ts": resets[key][epoch]["timestamp"] if epoch >= 0 else None,
            "completion_counter": threshold,
            "trigger_start_ts": inst["start_ts"],
            "trigger_end_ts": inst["end_ts"],
            "completion_ts": hit["timestamp"] if hit else None,
            "counter_crossing_ts": hit["timestamp"] if hit else None,
            "latency_ns": hit["timestamp"] - inst["end_ts"] if hit else None,
            "evidence": "semaphore_reset_and_counter" if hit else "unresolved_completion",
        }
        transfers.append(transfer)
        transfer_epochs[counter_key].append(transfer)

    # Counter thresholds alone identify a completed prefix. Individual ownership
    # additionally requires only one outstanding trigger using this semaphore.
    # Reject the whole epoch if that proof fails: no assumed hardware FIFO.
    for (core, semaphore, epoch_id), epoch in transfer_epochs.items():
        ambiguous = any(
            left["completion_ts"] is None or right["trigger_start_ts"] < left["completion_ts"]
            for left, right in zip(epoch, epoch[1:])
        )
        unexpected_writer, maximum = epoch_quality.get((core, semaphore, epoch_id), (False, 0))
        if unexpected_writer:
            ambiguous = True
        if (
            epoch[-1]["completion_counter"] is not None
            and maximum > epoch[-1]["completion_counter"]
        ):
            ambiguous = True  # An unobserved writer contributes to this counter.
        if ambiguous:
            for transfer in epoch:
                transfer["completion_ts"] = transfer["latency_ns"] = None
                transfer["evidence"] = "ambiguous_counter_prefix"
        else:
            for transfer in epoch:
                if transfer["completion_ts"] is not None:
                    transfer["evidence"] = "counter_inferred_single_outstanding"
                    by_completion[
                        (
                            (core, semaphore),
                            transfer["completion_ts"],
                            transfer["completion_counter"],
                        )
                    ] = transfer

    endpoints, tracks = defaultdict(list), defaultdict(list)
    for inst in ordered:
        endpoints[inst["pcore_idx"], inst["engine"]].append(
            (inst["end_ts"], inst["id"], inst["operation"], inst["timeline_track"])
        )
        tracks[inst["pcore_idx"], inst["timeline_track"]].append(inst)
    for values in endpoints.values():
        values.sort()

    def crossing(core, semaphore, threshold, timestamp, equal):
        key = core, semaphore
        values = sems.get(key, [])
        index = bisect.bisect_right(times.get(key, []), timestamp) - 1

        def satisfied(value):
            return value == threshold if equal else value >= threshold

        if index < 0 or not satisfied(values[index]["value"]):
            return None
        while index > 0 and satisfied(values[index - 1]["value"]):
            index -= 1
        return values[index]

    gaps = []
    for (core, track), values in tracks.items():
        previous_end = values[0]["end_ts"]
        for index, inst in enumerate(values[1:], 1):
            start, end = previous_end, inst["start_ts"]
            previous_end = max(previous_end, inst["end_ts"])
            if end <= start:
                continue
            candidates = [(inst, "instruction_wait")]
            if track == "GpSimd" and inst["opcode"] == "EVENT_SEMAPHORE":
                for other in values[index + 1 : index + 4]:
                    if (
                        0 <= other["start_ts"] - end <= 100
                        and other.get("pc") is not None
                        and inst.get("pc") is not None
                        and other["pc"] == inst["pc"] - 1
                        and other["opcode"].startswith("DMA_")
                    ):
                        candidates.append((other, "adjacent_dma_timestamp_inversion"))
            ready = []
            for candidate, match_kind in candidates:
                for match in _WAIT.finditer(candidate.get("operands") or ""):
                    semaphore, _, relation, threshold = match.groups()
                    semaphore, threshold = int(semaphore), int(threshold)
                    hit = crossing(
                        core, semaphore, threshold, candidate["start_ts"], relation == "=="
                    )
                    if hit and hit["timestamp"] >= start:
                        ready.append((hit["timestamp"], candidate, match_kind, semaphore, hit))
            best = max(ready, key=lambda item: item[0]) if ready else None
            source, source_id, source_track, source_engine, evidence = (
                "unresolved",
                None,
                None,
                None,
                "unresolved_gap",
            )
            if best:
                timestamp, candidate, match_kind, semaphore, hit = best
                source_engine = hit["update_source"]
                dma = by_completion.get(((core, semaphore), timestamp, hit["value"]))
                if dma:
                    source, source_id, evidence = (
                        dma["operation"],
                        dma["trigger_id"],
                        dma["evidence"],
                    )
                    source_track = dma["timeline_track"]
                elif hit["update_source"] == "AXI":
                    source, evidence = "unresolved_dma_or_collective", "observed_axi_update"
                else:
                    source, evidence = "unresolved_engine_source", "observed_semaphore_update"
                    ends = endpoints.get((core, source_engine), [])
                    pos = bisect.bisect_right(ends, (timestamp, 2**63 - 1, "\uffff", "\uffff")) - 1
                    if pos >= 0 and 0 <= timestamp - ends[pos][0] <= 100:
                        _, source_id, source, source_track = ends[pos]
                        evidence = "engine_end_within_100ns"
                explained = max(0, min(end, timestamp) - start)
            else:
                candidate, match_kind, semaphore, explained = inst, None, None, 0
            gaps.append(
                {
                    "pcore_idx": core,
                    "timeline_track": track,
                    "start_ts": start,
                    "end_ts": end,
                    "gap_ns": end - start,
                    "explained_ns": explained,
                    "residual_ns": end - start - explained,
                    "consumer_operation": candidate["operation"],
                    "waiter_id": inst["id"],
                    "blocking_instruction_id": candidate["id"],
                    "source_operation": source,
                    "source_id": source_id,
                    "semaphore": semaphore,
                    "match": match_kind,
                    "evidence": evidence,
                    "source_timeline_track": source_track,
                    "source_engine": source_engine,
                }
            )
    return {"transfers": transfers, "gaps": gaps}
