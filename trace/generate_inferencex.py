"""Generate an InferenceX fixed-sequence ("8k1k") closed-loop request trace.

InferenceX `infx.bench fixed-seq point --isl ISL --osl OSL --random-range-ratio R --conc C
--num-prompts 10C` (2C warm-up requests first): every prompt length is uniform in
[floor(R * ISL), ISL], every completion exactly OSL tokens (EOS ignored), and a fixed number of
requests C is in flight — a new request starts when one finishes.

The trace lists the warm-up and measured requests (`(warmup + measure) * C` rows) with every
`arrival_time` 0; the closed loop comes from the workload, not the trace:

    workload:
      arrival_mode: saturated     # every request eligible at once
      max_concurrency: C          # at most C in flight; the next starts when one completes

`tools/trn2-config-search` drops the first `warmup * C` completions before computing the
metrics, as the InferenceX client does. `--accept-rate` adds the `accept_rate` column a
`speculative`-tagged trace needs (per-position conditional draft acceptance).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path


def write_inferencex_trace(
    output_path: Path,
    *,
    concurrency: int,
    isl: int = 8192,
    osl: int = 1024,
    ratio: float = 0.8,
    warmup: int = 2,
    measure: int = 10,
    seed: int = 42,
    accept_rate: float | None = None,
) -> int:
    if concurrency <= 0 or isl <= 0 or osl <= 0:
        raise ValueError("concurrency, isl and osl must be positive")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    if accept_rate is not None and not 0.0 <= accept_rate <= 1.0:
        raise ValueError("accept_rate must be in [0, 1]")
    rnd = random.Random(seed)
    n = (warmup + measure) * concurrency
    lo = int(isl * ratio)
    header = "id,input_len,output_len,arrival_time" + (",accept_rate" if accept_rate is not None else "")
    lines = [header]
    for i in range(n):
        row = f"{i},{rnd.randint(lo, isl)},{osl},0.0"
        if accept_rate is not None:
            row += f",{accept_rate}"
        lines.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output", type=Path)
    ap.add_argument("--conc", type=int, required=True)
    ap.add_argument("--isl", type=int, default=8192)
    ap.add_argument("--osl", type=int, default=1024)
    ap.add_argument("--ratio", type=float, default=0.8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--measure", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--accept-rate", type=float, default=None)
    a = ap.parse_args()
    write_inferencex_trace(
        a.output, concurrency=a.conc, isl=a.isl, osl=a.osl, ratio=a.ratio, warmup=a.warmup,
        measure=a.measure, seed=a.seed, accept_rate=a.accept_rate,
    )


if __name__ == "__main__":
    main()
