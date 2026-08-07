"""Drop the CUDA-event-timed `moe_alltoall` / `moe_alltoall_prepare` rows.

Both tables were first recorded with the sibling NCCL runner's timing shape —
CUDA events around a rep loop. For a single fat all-reduce the launch gaps that
also charges are noise; for `prepare`, which launches five kilobyte-sized index
kernels, they ARE the measurement: the recorded curve is flat at ~74 us from 1
token to 4096, against ~16 us of real kernel time in a vLLM trace.

The runner now measures with CUPTI, which sums each launch's true kernel
duration — the same basis `analyze kernel-align` scores against. The old rows
answer a different question and a re-profile would not replace them (the cache
is keyed by shape, not by how it was measured), so they have to be removed
explicitly. They are written out first: they are the evidence for the change.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

TABLES = ("moe_alltoall", "moe_alltoall_prepare")
BACKEND = "flashinfer_mnnvl"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=Path("profiling/profile.db"))
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be removed and write the backup, but keep the rows.",
    )
    arguments = parser.parse_args()

    connection = sqlite3.connect(arguments.database)
    connection.row_factory = sqlite3.Row
    backup: dict[str, list[dict]] = {}
    try:
        for table in TABLES:
            rows = connection.execute(
                f"SELECT * FROM {table} WHERE backend = ?", (BACKEND,)
            ).fetchall()
            backup[table] = [dict(row) for row in rows]
            print(f"{table}: {len(rows)} row(s) recorded on the event basis")

        arguments.backup.parent.mkdir(parents=True, exist_ok=True)
        arguments.backup.write_text(json.dumps(backup, indent=2))
        print(f"backup written to {arguments.backup}")

        if arguments.dry_run:
            print("dry run: rows kept")
            return

        for table in TABLES:
            connection.execute(f"DELETE FROM {table} WHERE backend = ?", (BACKEND,))
        connection.commit()
        print("rows deleted; the next profiling pass will re-record them with CUPTI")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
