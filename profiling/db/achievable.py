"""Achievable hardware peaks: the best rates a real kernel reaches on one GPU.

The optimality analyzer's achievable-limit rung sits between the profiled
kernels (R4) and the spec sheet (R5). It prices a compute-bound leaf at the
fastest measured GEMM of the same compute dtype and a bandwidth-bound leaf at
the fastest measured large copy. This module reads those two rates from
``profile.db``:

- **compute**: the highest ``single_gemm`` TFLOP/s per ``dtype`` string, over
  every backend and shape. The analyzer groups dtype strings into classes
  (``fp8_e4m3`` and ``mxfp8_e4m3`` are both fp8), so this reports raw strings.
- **bandwidth**: the highest ``elementwise`` copy (``input == output`` bytes)
  GB/s among rows whose input is at least ``MIN_COPY_INPUT_BYTES``. Smaller
  copies overstate HBM bandwidth even with a cold L2: L2 is write-back, so a
  copy whose output fits in L2 finishes before its writes reach HBM. A B200
  copy of 80 MB reads 8.4 TB/s against an 8 TB/s HBM; at 1 GiB the absorbed
  writes are a few percent of the traffic.

Outlier rows are excluded. Reads only; no torch, no profiling.

Run as ``python -m profiling.db.achievable --gpu-name "NVIDIA B200"``; it
prints one JSON object on stdout.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

from profiling.db import layout

# 1 GiB of input is at least 8x the L2 of every GPU in gpu/spec.json.
MIN_COPY_INPUT_BYTES = 1 << 30


def achievable_peaks(db_path: Path | str, gpu_name: str) -> dict[str, Any]:
    """Best measured GEMM TFLOP/s per dtype and large-copy GB/s for ``gpu_name``."""
    root = layout.require_directory(db_path)
    gemm_rows = _query(
        layout.kind_file(root, "single_gemm"),
        """
            SELECT dtype, backend, m, n, k, tflops
            FROM single_gemm AS row
            WHERE gpu_name = ? AND is_outlier = 0
              AND tflops = (
                SELECT MAX(tflops) FROM single_gemm
                WHERE gpu_name = row.gpu_name AND dtype = row.dtype AND is_outlier = 0
              )
            ORDER BY dtype, backend
            """,
        (gpu_name,),
    )
    copy_rows = _query(
        layout.kind_file(root, "elementwise"),
        """
            SELECT backend, input_size_bytes, memory_bandwidth_gbps
            FROM elementwise
            WHERE gpu_name = ? AND is_outlier = 0
              AND input_size_bytes = output_size_bytes
              AND input_size_bytes >= ?
            ORDER BY memory_bandwidth_gbps DESC
            LIMIT 1
            """,
        (gpu_name, MIN_COPY_INPUT_BYTES),
    )
    copy_row = copy_rows[0] if copy_rows else None

    gemm: dict[str, dict[str, Any]] = {}
    for row in gemm_rows:
        # Ties keep the first backend in sort order.
        gemm.setdefault(
            row["dtype"],
            {
                "tflops": row["tflops"],
                "backend": row["backend"],
                "shape": {"m": row["m"], "n": row["n"], "k": row["k"]},
            },
        )
    copy = None
    if copy_row is not None:
        copy = {
            "gbps": copy_row["memory_bandwidth_gbps"],
            "backend": copy_row["backend"],
            "input_size_bytes": copy_row["input_size_bytes"],
        }
    return {
        "gpu_name": gpu_name,
        "gemm_by_dtype": gemm,
        "copy": copy,
        "min_copy_input_bytes": MIN_COPY_INPUT_BYTES,
    }


def _query(path: Path, sql: str, params: tuple[Any, ...]) -> list[sqlite3.Row]:
    """``sql``'s rows from one kind file; none when the kind has no file."""
    if not path.is_file():
        return []
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(sql, params).fetchall()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m profiling.db.achievable",
        description="Print the best measured GEMM TFLOP/s and large-copy GB/s for one GPU.",
    )
    parser.add_argument(
        "--gpu-name", required=True, help='profile.db gpu_name, e.g. "NVIDIA B200".'
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Profile DB directory (default VIBESIM_PROFILE_DB or profiling/profile_db).",
    )
    args = parser.parse_args(argv)
    db = args.db if args.db is not None else layout.configured_root()
    if not db.is_dir():
        print(f"profile DB directory not found: {db}", file=sys.stderr)
        return 2
    print(json.dumps(achievable_peaks(db, args.gpu_name), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
