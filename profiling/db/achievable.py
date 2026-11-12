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

# 1 GiB of input is at least 8x the L2 of every GPU in gpu/spec.json.
MIN_COPY_INPUT_BYTES = 1 << 30

_DEFAULT_DB = Path(__file__).resolve().parents[1] / "profile.db"


def achievable_peaks(db_path: Path | str, gpu_name: str) -> dict[str, Any]:
    """Best measured GEMM TFLOP/s per dtype and large-copy GB/s for ``gpu_name``."""
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        gemm_rows = conn.execute(
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
        ).fetchall()
        copy_row = conn.execute(
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
        ).fetchone()

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m profiling.db.achievable",
        description="Print the best measured GEMM TFLOP/s and large-copy GB/s for one GPU.",
    )
    parser.add_argument(
        "--gpu-name", required=True, help='profile.db gpu_name, e.g. "NVIDIA B200".'
    )
    parser.add_argument("--db", type=Path, default=_DEFAULT_DB, help="profile.db path.")
    args = parser.parse_args(argv)
    if not args.db.is_file():
        print(f"profile.db not found: {args.db}", file=sys.stderr)
        return 2
    print(json.dumps(achievable_peaks(args.db, args.gpu_name), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
