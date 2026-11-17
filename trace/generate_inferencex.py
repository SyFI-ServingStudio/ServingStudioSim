"""Generate an InferenceX fixed-sequence ("8k1k") closed-loop request trace.

InferenceX `infx.bench fixed-seq point --isl ISL --osl OSL --random-range-ratio R --conc C
--num-prompts 10C` (`infx/bench_serving/benchmark_serving.py::sample_random_requests`): every
prompt length is uniform in [floor(R * ISL), ISL] and every completion length uniform in
[floor(R * OSL), OSL] (EOS ignored, so a request generates exactly its drawn length), and a
fixed number of requests C is in flight — a new request starts when one finishes.

The client first runs 2C warm-up requests (copies of the first prompt) to completion, then
starts the measured run on an idle server: the first C requests at once, the rest as slots
free. The trace therefore lists the measured requests only (`measure * C` rows; `--warmup W`
prepends `W * C` more for a steady-state slice) with every `arrival_time` 0; the closed loop
comes from the workload, not the trace:

    workload:
      arrival_mode: saturated     # every request eligible at once
      max_concurrency: C          # at most C in flight; the next starts when one completes

`tools/trn2-config-search/inferencex.py --warmup W` drops the first `W * C` completions (0 by
default). `--fixed-output` restores the old exact-OSL completions.

`--sampler inferencex` (the default) reproduces the client's own draw exactly: it seeds numpy with
`--seed` (the client's default 0) and draws `randint(lo, hi + 1, size=N)` for the inputs, then the
outputs, as `sample_random_requests` does after `np.random.seed(args.seed)`. The lists then equal
the official run's `input_lens` / `output_lens` at the same C (checked against the round-1 and
round-2 C = 8 measures). `--sampler python` keeps the previous `random.Random(seed)` draw.
`--accept-rate` adds the `accept_rate` column a `speculative`-tagged trace needs (per-position
conditional draft acceptance).
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
    warmup: int = 0,
    measure: int = 10,
    seed: int = 0,
    accept_rate: float | None = None,
    fixed_output: bool = False,
    sampler: str = "inferencex",
) -> int:
    if concurrency <= 0 or isl <= 0 or osl <= 0:
        raise ValueError("concurrency, isl and osl must be positive")
    if not 0.0 < ratio <= 1.0:
        raise ValueError("ratio must be in (0, 1]")
    if accept_rate is not None and not 0.0 <= accept_rate <= 1.0:
        raise ValueError("accept_rate must be in [0, 1]")
    n = (warmup + measure) * concurrency
    lo = int(isl * ratio)
    lo_out = osl if fixed_output else int(osl * ratio)
    if sampler == "inferencex":
        import numpy as np

        np.random.seed(seed)  # noqa: NPY002 - the client's global numpy stream
        np.random.randint(0, 2, size=0)  # noqa: NPY002 - the client's empty prefix draw
        inputs = np.random.randint(lo, isl + 1, size=n).tolist()  # noqa: NPY002
        outputs = np.random.randint(lo_out, osl + 1, size=n).tolist()  # noqa: NPY002
    elif sampler == "python":
        rnd = random.Random(seed)
        inputs, outputs = [], []
        for _ in range(n):
            inputs.append(rnd.randint(lo, isl))
            outputs.append(rnd.randint(lo_out, osl))
    else:
        raise ValueError(f"unknown sampler {sampler!r}")
    header = "id,input_len,output_len,arrival_time" + (
        ",accept_rate" if accept_rate is not None else ""
    )
    lines = [header]
    for i in range(n):
        row = f"{i},{inputs[i]},{outputs[i]},0.0"
        if accept_rate is not None:
            row += f",{accept_rate}"
        lines.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("output", type=Path)
    ap.add_argument("--conc", type=int, required=True)
    ap.add_argument("--isl", type=int, default=8192)
    ap.add_argument("--osl", type=int, default=1024)
    ap.add_argument("--ratio", type=float, default=0.8)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--measure", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--accept-rate", type=float, default=None)
    ap.add_argument("--fixed-output", action="store_true")
    ap.add_argument("--sampler", choices=["inferencex", "python"], default="inferencex")
    a = ap.parse_args()
    write_inferencex_trace(
        a.output,
        concurrency=a.conc,
        isl=a.isl,
        osl=a.osl,
        ratio=a.ratio,
        warmup=a.warmup,
        measure=a.measure,
        seed=a.seed,
        accept_rate=a.accept_rate,
        fixed_output=a.fixed_output,
        sampler=a.sampler,
    )


if __name__ == "__main__":
    main()
