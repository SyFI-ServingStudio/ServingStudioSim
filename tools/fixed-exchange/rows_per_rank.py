"""Per-rank EP exchange rows at each load point of a stacked sweep (read-only).

The fast exchange's dispatch leg logs the busiest sender's rows, own*(N-1), and the busiest
receiver's, total - own. Prints the iteration-weighted distribution of (own rows, total rows).
"""

import collections
import json
import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

LEAVES = ("dispatch_fast_all_gather", "dispatch_fused_all_gather")


def main(base: Path, strategy: str, ranks: int = 8) -> None:
    for run in sorted(base.glob(f"*/{strategy}/c*")):
        manifest = json.loads((run / "raw/cost_manifest/worker_main_0.json").read_text())
        slots = manifest["sections"][0]["slots"]
        index = next(i for i, s in enumerate(slots) if s["name"].endswith(LEAVES))
        column = pq.read_table(run / "raw/cost_log/worker_main_0.parquet", columns=["slot_input"])
        values = pc.list_element(column["slot_input"], index).to_pylist()
        hist = collections.Counter()
        for raw in values:
            leg = json.loads(raw)
            own = round(leg["max_send_rows"] / (ranks - 1))
            hist[(own, round(own + leg["max_recv_rows"]))] += 1
        total = sum(hist.values())
        top = [(k, round(v / total, 3)) for k, v in hist.most_common(5)]
        print(run.relative_to(base), total, top, flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2])
