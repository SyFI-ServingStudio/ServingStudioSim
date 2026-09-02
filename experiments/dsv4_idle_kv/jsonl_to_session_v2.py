#!/usr/bin/env python3
"""Convert agent replay JSONL workloads to VibeSim session_execution_v2 CSV traces."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = (
    "request_id",
    "session_id",
    "round_idx",
    "arrival_time_ms",
    "prefix_len",
    "input_len",
    "output_len",
    "tool_wait_after_ms",
)


def jsonl_to_session_v2(in_path: Path, out_path: Path) -> int:
    rows = [json.loads(line) for line in in_path.read_text().splitlines() if line.strip()]
    by: dict[str, list[dict]] = {}
    for row in rows:
        by.setdefault(row["session_id"], []).append(row)

    out: list[dict] = []
    for sid in sorted(by):
        rounds = sorted(by[sid], key=lambda x: x["round"])
        prev = 0
        for rr in rounds:
            total = len(rr["token_ids"])
            out.append(
                {
                    "request_id": f"{sid}_r{rr['round']:03d}",
                    "session_id": sid,
                    "round_idx": rr["round"],
                    "arrival_time_ms": 0.0,
                    "prefix_len": prev,
                    "input_len": max(1, total - prev),
                    "output_len": rr["output_tokens"],
                    "tool_wait_after_ms": rr.get("wait_ms", 0),
                }
            )
            prev = total

    if not out:
        raise SystemExit(f"no rows in {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out)
    print(f"wrote {out_path} ({len(out)} rounds from {in_path})")
    return len(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("inputs", nargs="+", type=Path, help="input .jsonl paths")
    ap.add_argument(
        "--trace-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "trace",
        help="output directory (default: repo trace/)",
    )
    ap.add_argument(
        "--prefix",
        default="dsv4_",
        help="output filename prefix (default: dsv4_)",
    )
    args = ap.parse_args()

    for in_path in args.inputs:
        stem = in_path.stem.replace(".jsonl", "")
        out_path = args.trace_dir / f"{args.prefix}{stem}.csv"
        jsonl_to_session_v2(in_path, out_path)


if __name__ == "__main__":
    main()
