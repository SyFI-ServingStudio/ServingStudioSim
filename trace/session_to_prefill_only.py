"""Turn a session trace into an independent, prefill-only request trace.

Each round of a `session-execution-v2` trace (for example
`trace/tracelab_preserving.csv`) becomes one independent request:

- `input_len` is the round's fresh tokens;
- `prefix_len` is the round's planned prefix, which the simulator treats as a
  pinned prefix hit (see `trace/README.md`);
- `output_len` is 1, so the request is prefill only.

A round whose prefix covers its whole prompt (zero fresh tokens) still computes
its last token, as an engine does to produce logits after a full prefix hit, so
it becomes one fresh token after a prefix one token shorter.

Session chaining, tool waits, and the source timeline are dropped. Rounds are
taken in a seeded random order, so a `--requests` subset is a uniform sample of
all rounds and the length mix does not drift over the run. Arrivals are a
seeded Poisson process at 1 request/s; a preset's `workload.request_rate`
scales that timeline to the rate under test.

A manifest beside the output records the source, the selection, and the token
totals.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SOURCE_COLUMNS = ("request_id", "prefix_len", "input_len")
OUTPUT_HEADER = "id,input_len,output_len,arrival_time,prefix_len"
ARRIVAL_RATE_PER_SECOND = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_rounds(source: Path) -> tuple[list[tuple[str, int, int]], int]:
    """The rounds as `(id, prefix, fresh)`, and how many had zero fresh tokens."""
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [column for column in SOURCE_COLUMNS if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{source} is not a session trace: missing columns {missing}")
        rounds = []
        full_prefix_rounds = 0
        for row in reader:
            prefix, fresh = int(row["prefix_len"]), int(row["input_len"])
            if prefix + fresh == 0:
                raise ValueError(f"{source}: round {row['request_id']} has an empty prompt")
            if fresh == 0:
                prefix, fresh = prefix - 1, 1
                full_prefix_rounds += 1
            rounds.append((row["request_id"], prefix, fresh))
        return rounds, full_prefix_rounds


def write_prefill_only_trace(
    source: Path,
    output_path: Path,
    *,
    requests: int | None,
    seed: int,
    max_context: int,
) -> dict:
    rounds, full_prefix_rounds = _read_rounds(source)
    if not rounds:
        raise ValueError(f"{source} has no rounds")
    if requests is not None and not 0 < requests <= len(rounds):
        raise ValueError(f"requests must be in 1..{len(rounds)}")
    # The request's context after its one output token.
    too_long = [rid for rid, prefix, fresh in rounds if prefix + fresh + 1 > max_context]
    if too_long:
        raise ValueError(
            f"{len(too_long)} rounds exceed max_context {max_context}, first {too_long[0]}"
        )

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rounds))[: requests or len(rounds)]
    gaps_ms = rng.exponential(1000.0 / ARRIVAL_RATE_PER_SECOND, size=len(order))
    gaps_ms[0] = 0.0
    arrivals_ms = np.cumsum(gaps_ms)

    selected = [rounds[int(index)] for index in order]
    lines = [OUTPUT_HEADER]
    lines.extend(
        f"{rid},{fresh},1,{arrival:.6f},{prefix}"
        for (rid, prefix, fresh), arrival in zip(selected, arrivals_ms)
    )
    output_path.write_text("\n".join(lines) + "\n")

    fresh_tokens = sum(fresh for _, _, fresh in selected)
    prefix_tokens = sum(prefix for _, prefix, _ in selected)
    manifest = {
        "schema": "text-generation-independent",
        "generator": "trace/session_to_prefill_only.py",
        "source_name": source.name,
        "source_sha256": _sha256(source),
        "source_rounds": len(rounds),
        "source_full_prefix_rounds_given_one_fresh_token": full_prefix_rounds,
        "requests": len(selected),
        "selection": "seeded_uniform_sample_in_random_order",
        "seed": seed,
        "arrival_pattern": "poisson",
        "arrival_rate_per_second": ARRIVAL_RATE_PER_SECOND,
        "output_len": 1,
        "max_context": max_context,
        "max_request_context": max(prefix + fresh for _, prefix, fresh in selected) + 1,
        "total_fresh_tokens": fresh_tokens,
        "total_prefix_tokens": prefix_tokens,
        "prefix_hit_rate": prefix_tokens / (prefix_tokens + fresh_tokens),
    }
    output_path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="session-execution-v2 CSV")
    parser.add_argument("output", type=Path)
    parser.add_argument("--requests", type=int, help="sample this many rounds (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-context",
        type=int,
        default=1_048_576,
        help="reject the trace if any request's context exceeds this (max_model_len)",
    )
    arguments = parser.parse_args()
    manifest = write_prefill_only_trace(
        arguments.source,
        arguments.output,
        requests=arguments.requests,
        seed=arguments.seed,
        max_context=arguments.max_context,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
