"""Stage time of one microbatch as a function of its composition: microbatch_fit.json beside this file.

  stage time (ms) = a + b * tokens / 1k + c * pairs / 1e9,   pairs = sum over requests of append * (prefix + append / 2)

`tokens` is the microbatch's computed tokens; each request in it adds `append` tokens on top of `prefix` cached (or
already computed) tokens. One fit per stage, least squares over the microbatches of a session run with at least 256
tokens; smaller ones run as captured graphs and are left out.

usage (ServingStudioSim root, at the frozen commit):
  PYTHONPATH=$PWD uv run python .../data/microbatch_fit.py fit <session run dir> [<more run dirs to check>]
  uv run python .../data/microbatch_fit.py check <measured/microbatches.jsonl>
      one line per (microbatch, stage): {"stage": 0, "compute_ms": 57.4, "requests": [{"prefix": 55373, "append": 1196}, ...]}
"""
import json, sys
from pathlib import Path

import numpy as np

here = Path(__file__).parent
MIN_TOKENS = 256


def features(requests):
    tokens = sum(r["append"] for r in requests)
    pairs = sum(r["append"] * (r["prefix"] + r["append"] / 2) for r in requests)
    return tokens, np.array([1.0, tokens / 1e3, pairs / 1e9])


def sim_microbatches(run, stage):
    import pyarrow.parquet as pq
    rows = pq.read_table(Path(run) / f"raw/cost_log/worker_stage_{stage}.parquet",
                         columns=["total_time_ms", "groups"]).to_pylist()
    for r in rows:
        yield r["total_time_ms"], [dict(prefix=p, append=a) for g in r["groups"]
                                   for p, a in zip(g["prefill_prefix_lens"], g["prefill_append_lens"])]


def errors(coef, items):
    e = [(features(req)[1] @ coef - t) / t for t, req in items if features(req)[0] >= MIN_TOKENS]
    e = np.abs(e)
    return dict(microbatches=len(e), abs_err_p50=float(np.percentile(e, 50)), abs_err_p99=float(np.percentile(e, 99)))


if sys.argv[1] == "fit":
    out = []
    for stage in range(4):
        items = [(t, req) for t, req in sim_microbatches(sys.argv[2], stage) if features(req)[0] >= MIN_TOKENS]
        X = np.array([features(req)[1] for _, req in items])
        coef = np.linalg.lstsq(X, np.array([t for t, _ in items]), rcond=None)[0]
        out.append(dict(stage=stage, a_ms=coef[0], b_ms_per_1k_tokens=coef[1], c_ms_per_1e9_pairs=coef[2],
                        fit=errors(coef, items),
                        checks={"/".join(Path(run).parts[-2:]): errors(coef, list(sim_microbatches(run, stage)))
                                for run in sys.argv[3:]}))
        print(json.dumps(out[-1]))
    (here / "microbatch_fit.json").write_text(json.dumps(out, indent=1) + "\n")
else:
    fit = {f["stage"]: np.array([f["a_ms"], f["b_ms_per_1k_tokens"], f["c_ms_per_1e9_pairs"]])
           for f in json.loads((here / "microbatch_fit.json").read_text())}
    by_stage = {}
    for line in Path(sys.argv[2]).read_text().splitlines():
        m = json.loads(line)
        tokens, x = features(m["requests"])
        if tokens >= MIN_TOKENS:
            by_stage.setdefault(m["stage"], []).append(m["compute_ms"] / (x @ fit[m["stage"]]))
    for stage, ratios in sorted(by_stage.items()):
        r = np.array(ratios)
        print(f"stage {stage}: {len(r)} microbatches, measured / fit p10 {np.percentile(r, 10):.3f} "
              f"p50 {np.percentile(r, 50):.3f} p90 {np.percentile(r, 90):.3f}")
