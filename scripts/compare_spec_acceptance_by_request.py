"""Compare measured request progress with the production keyed acceptance model.

The result describes observed-conditioned acceptance, not predictive alignment.
KV lengths here mean computed tokens before verification, as in vLLM records.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import struct
from collections import defaultdict
from pathlib import Path

_MASK = (1 << 64) - 1
_MIX = (0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB)


def splitmix64(value: int) -> int:
    value = (value + _MIX[0]) & _MASK
    value = ((value ^ (value >> 30)) * _MIX[1]) & _MASK
    value = ((value ^ (value >> 27)) * _MIX[2]) & _MASK
    return value ^ (value >> 31)


def bernoulli(seed, request_id, output_tokens_emitted, draft_position, accept_rate):
    rate = struct.unpack("f", struct.pack("f", accept_rate))[0]
    key = seed
    for value, multiplier in zip((request_id, output_tokens_emitted, draft_position), _MIX):
        key ^= (value * multiplier) & _MASK
    return (splitmix64(key) >> 11) / (1 << 53) < rate


def _probabilities(value, depth):
    rates = value if isinstance(value, list) else [value] * depth
    if len(rates) != depth or any(
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        or not 0 <= rate <= 1
        for rate in rates
    ):
        raise ValueError(f"acceptance must contain {depth} finite probabilities")
    return rates


def simulate_request(*, dense_request_id, input_len, output_len, acceptance, seed):
    if input_len <= 0 or output_len <= 0 or not acceptance:
        raise ValueError("request lengths and draft depth must be positive")
    rates = _probabilities(acceptance, len(acceptance))
    emitted = 1
    progress = []
    while emitted < output_len:
        accepted = 1
        for position, rate in enumerate(rates):
            if not bernoulli(seed, dense_request_id, emitted, position, rate):
                break
            accepted += 1
        accepted = min(accepted, output_len - emitted)
        progress.append(
            {
                "kv_len": input_len + emitted - 1,
                "output_tokens_before": emitted,
                "drafted_tokens": len(rates),
                "emitted_tokens": accepted,
                "accepted_draft_tokens": accepted - 1,
            }
        )
        emitted += accepted
    return {
        "decode_rounds": len(progress),
        "checked_rows": len(progress) * (len(rates) + 1),
        "accepted_draft_tokens": sum(row["accepted_draft_tokens"] for row in progress),
        "decode_kv_sum": sum(row["kv_len"] for row in progress),
        "progress": progress,
    }


def _bind_trace_request_ids(trace, measured_ids):
    if len({row["request_id"] for row in trace}) != len(trace):
        raise ValueError("duplicate trace request ID")
    decoded = [row["request_id"] for row in trace if row["output_len"] > 1]
    for prefix in ("", "independent_"):
        if {prefix + request_id for request_id in decoded} == measured_ids:
            for row in trace:
                row["request_id"] = prefix + row["request_id"]
            return
    raise ValueError("measured request population mismatch")


def _acceptance_summary(steps, *, draft_tokens):
    denominators = [0] * draft_tokens
    accepted = [0] * draft_tokens
    for step in steps:
        for position in range(min(step["drafted_tokens"], draft_tokens)):
            denominators[position] += step["accepted_draft_tokens"] >= position
            accepted[position] += step["accepted_draft_tokens"] > position
    drafted = sum(step["drafted_tokens"] for step in steps)
    accepted_total = sum(step["accepted_draft_tokens"] for step in steps)
    return {
        "rounds": len(steps),
        "drafted_tokens": drafted,
        "accepted_draft_tokens": accepted_total,
        "accepted_fraction_of_drafts": accepted_total / drafted if drafted else None,
        "position_denominators": denominators,
        "accepted_per_position": accepted,
        "conditional_acceptance_rates": [
            a / d if d else None for a, d in zip(accepted, denominators)
        ],
    }


def _output_progress_quartiles(steps, *, draft_tokens):
    groups = [[] for _ in range(4)]
    for step in steps:
        index = min(3, 4 * step["output_tokens_before"] // max(step["request_output_len"] - 1, 1))
        groups[index].append(step)
    return [
        {
            "quartile": index + 1,
            "normalized_progress_start": index / 4,
            "normalized_progress_end": (index + 1) / 4,
            "checked_rows": sum(row.get("query_len", row["drafted_tokens"] + 1) for row in group),
            "decode_kv_sum": sum(row["kv_len"] for row in group),
            **_acceptance_summary(group, draft_tokens=draft_tokens),
        }
        for index, group in enumerate(groups)
    ]


def _delta_pct(simulated, measured):
    return 100 * (simulated - measured) / measured if measured else None


def _pearson(xs, ys):
    if len(xs) < 2:
        return None
    dx = [x - sum(xs) / len(xs) for x in xs]
    dy = [y - sum(ys) / len(ys) for y in ys]
    scale = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / scale if scale else None


def _quartile_groups(requests):
    ordered = sorted(requests, key=lambda row: (row["input_len"], row["request_id"]))
    result = []
    for index in range(4):
        group = ordered[len(ordered) * index // 4 : len(ordered) * (index + 1) // 4]
        row = {
            "quartile": index + 1,
            "n": len(group),
            "input_len_min": min((r["input_len"] for r in group), default=None),
            "input_len_max": max((r["input_len"] for r in group), default=None),
        }
        for side in ("measured", "simulated"):
            for field in ("decode_rounds", "decode_kv_sum"):
                key = f"{side}_{field}"
                row[key] = sum(r[key] for r in group)
        row["decode_kv_delta_pct"] = _delta_pct(
            row["simulated_decode_kv_sum"], row["measured_decode_kv_sum"]
        )
        result.append(row)
    return result


def compare(*, metrics_jsonl: Path, trace_csv: Path, draft_tokens: int, seed: int):
    if draft_tokens <= 0 or not 0 <= seed <= _MASK:
        raise ValueError("draft_tokens must be positive and seed must fit u64")
    with trace_csv.open(newline="") as handle:
        trace = [
            {
                "request_id": row["id"],
                "dense_request_id": index,
                "input_len": int(row["input_len"]),
                "output_len": int(row["output_len"]),
                "acceptance": _probabilities(json.loads(row["accept_rate"]), draft_tokens),
            }
            for index, row in enumerate(csv.DictReader(handle))
        ]
    if not trace:
        raise ValueError("trace must not be empty")
    measured = defaultdict(list)
    with metrics_jsonl.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("schema_version") != 4:
                raise ValueError("per-request progress requires schema_version 4")
            seen = set()
            for step in record["decode_request_progress"]:
                request_id = step["request_id"]
                if request_id in seen:
                    raise ValueError("duplicate request within measured iteration")
                seen.add(request_id)
                finished = step.get("request_finished_before", False)
                if type(finished) is not bool:
                    raise ValueError("invalid measured request_finished_before")
                for key in (
                    "kv_len",
                    "query_len",
                    "output_tokens_before",
                    "drafted_tokens",
                    "emitted_tokens",
                    "accepted_draft_tokens",
                ):
                    if key == "output_tokens_before" and step[key] is None and finished:
                        continue
                    if type(step[key]) is not int or step[key] < 0:
                        raise ValueError(f"invalid measured progress: {key}")
                if (
                    step["query_len"] == 0
                    or step["drafted_tokens"] > draft_tokens
                    or step["accepted_draft_tokens"] > step["drafted_tokens"]
                    or step["accepted_draft_tokens"] != max(step["emitted_tokens"] - 1, 0)
                ):
                    raise ValueError("inconsistent measured acceptance")
                measured[request_id].append(step)
    _bind_trace_request_ids(trace, set(measured))
    requests = []
    all_measured = []
    all_simulated = []
    for request in trace:
        steps = measured[request["request_id"]]
        verify = [s for s in steps if s["query_len"] == s["drafted_tokens"] + 1]
        recompute = [s for s in steps if s["query_len"] != s["drafted_tokens"] + 1]
        effective = []
        for step in verify:
            # Queued work can complete after the scheduler retired a request.
            # Keep it in raw work totals, but it cannot advance client output.
            if step.get("request_finished_before", False):
                continue
            remaining = request["output_len"] - step["output_tokens_before"]
            if remaining <= 0:
                raise ValueError("measured decode starts after request completion")
            emitted = min(step["emitted_tokens"], remaining)
            effective.append(
                {
                    **step,
                    "emitted_tokens": emitted,
                    "accepted_draft_tokens": max(emitted - 1, 0),
                    "request_output_len": request["output_len"],
                }
            )
        simulated = simulate_request(
            dense_request_id=request["dense_request_id"],
            input_len=request["input_len"],
            output_len=request["output_len"],
            acceptance=request["acceptance"],
            seed=seed,
        )
        simulated_steps = [
            {**s, "request_output_len": request["output_len"]} for s in simulated["progress"]
        ]
        all_measured.extend(effective)
        all_simulated.extend(simulated_steps)
        row = {key: value for key, value in request.items() if key != "acceptance"}
        row.update(
            {
                "measured_recompute_steps": len(recompute),
                "measured_recompute_query_rows": sum(s["query_len"] for s in recompute),
                "measured_decode_rounds": len(verify),
                "measured_finished_request_rounds": sum(
                    s.get("request_finished_before", False) for s in verify
                ),
                "measured_checked_rows": sum(s["query_len"] for s in verify),
                "measured_decode_kv_sum": sum(s["kv_len"] for s in verify),
                "measured_accepted_draft_tokens_raw": sum(
                    s["accepted_draft_tokens"] for s in verify
                ),
                "measured_emitted_tokens_raw": sum(s["emitted_tokens"] for s in verify),
                "measured_acceptance": _acceptance_summary(effective, draft_tokens=draft_tokens),
                "simulated_acceptance": _acceptance_summary(
                    simulated_steps, draft_tokens=draft_tokens
                ),
            }
        )
        row.update(
            {f"simulated_{key}": value for key, value in simulated.items() if key != "progress"}
        )
        row["decode_round_delta"] = row["simulated_decode_rounds"] - row["measured_decode_rounds"]
        row["decode_kv_delta"] = row["simulated_decode_kv_sum"] - row["measured_decode_kv_sum"]
        requests.append(row)
    aggregate = {"requests": len(requests)}
    aggregate["measured_finished_request_rounds"] = sum(
        row["measured_finished_request_rounds"] for row in requests
    )
    for field, delta in (
        ("decode_rounds", "decode_round"),
        ("checked_rows", "checked_row"),
        ("decode_kv_sum", "decode_kv"),
    ):
        for side in ("measured", "simulated"):
            aggregate[f"{side}_{field}"] = sum(row[f"{side}_{field}"] for row in requests)
        aggregate[f"{delta}_delta_pct"] = _delta_pct(
            aggregate[f"simulated_{field}"], aggregate[f"measured_{field}"]
        )
    for field in ("steps", "query_rows"):
        aggregate[f"measured_recompute_{field}_excluded"] = sum(
            row[f"measured_recompute_{field}"] for row in requests
        )
    aggregate["measured_acceptance_effective"] = _acceptance_summary(
        all_measured, draft_tokens=draft_tokens
    )
    aggregate["simulated_acceptance_effective"] = _acceptance_summary(
        all_simulated, draft_tokens=draft_tokens
    )
    lengths = [row["input_len"] for row in requests]
    for field, key in (("decode_round_delta", "round"), ("decode_kv_delta", "decode_kv")):
        aggregate[f"input_len_vs_{key}_delta_pearson"] = _pearson(
            lengths, [row[field] for row in requests]
        )
    return {
        "schema_version": 1,
        "predictive_alignment": False,
        "inputs": {
            "metrics_jsonl": str(metrics_jsonl),
            "trace_csv": str(trace_csv),
            "draft_tokens": draft_tokens,
            "acceptance_seed": seed,
        },
        "aggregate": aggregate,
        "requests_by_trace_order": requests,
        "input_length_quartiles": _quartile_groups(requests),
        "output_progress_quartiles": {
            "measured": _output_progress_quartiles(all_measured, draft_tokens=draft_tokens),
            "simulated": _output_progress_quartiles(all_simulated, draft_tokens=draft_tokens),
        },
        "largest_positive_decode_kv_deltas": sorted(
            requests, key=lambda r: r["decode_kv_delta"], reverse=True
        )[:20],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-jsonl", type=Path, required=True)
    parser.add_argument("--trace-csv", type=Path, required=True)
    parser.add_argument("--draft-tokens", type=int, required=True)
    parser.add_argument("--acceptance-seed", dest="seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = vars(parser.parse_args())
    output = args.pop("out")
    result = compare(**args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
