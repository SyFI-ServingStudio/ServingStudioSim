"""InferenceX metrics and the step bottleneck breakdown of one simulation run.

``point(run_dir, conc)``: the InferenceX fixed-seq point of a closed-loop run (trace from
``trace/generate_inferencex.py``, workload ``arrival_mode: saturated`` + ``max_concurrency: C``).
The InferenceX client drains its 2C warm-up requests before the measured run starts on an idle
server, so the trace holds the measured requests only and every completion counts
(``warmup`` = 0); ``--warmup W`` drops the first ``W * C`` completions of a trace generated with
``--warmup W``:

- ``tok_s_per_chip`` = (input + output tokens of the measured requests) / measured window / chips,
  the window from the first measured request's start to the last one's finish;
- ``intvty`` = 1000 / median TPOT (ms) (tok/s/user), TPOT = (E2E - TTFT) / (output - 1);
- ``ttft_s``, ``e2e_s`` = medians over the measured requests.

``breakdown(run_dir)``: per engine step, the mean time of every L1 kind (folded by its layer
count) in decode-only steps and in steps with a prefill chunk, from the cost log.

    uv run --no-sync python tools/trn2-config-search/inferencex.py logs/trn2/<run>/c16 --conc 16
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

CORES_PER_CHIP = 4


def _requests(run: Path) -> list[dict]:
    import pyarrow.parquet as pq

    cols = [
        "request_id",
        "completed",
        "arrival_time_ms",
        "ttft_ms",
        "num_output_tokens",
        "finish_decode_time_ms",
        "prefill_processed",
        "fresh_prompt_tokens",
    ]
    return pq.read_table(run / "raw" / "request_slo.parquet", columns=cols).to_pylist()


def chips(run: Path) -> float:
    summary = json.loads((run / "summary.json").read_text())
    return summary["num_gpus"] / CORES_PER_CHIP


def point(run: Path, conc: int, warmup: int = 0) -> dict:
    reqs = [r for r in _requests(run) if r["completed"]]
    reqs.sort(key=lambda r: r["finish_decode_time_ms"])
    meas = reqs[warmup * conc :]
    if not meas:
        raise ValueError(f"{run}: no measured requests after {warmup * conc} warm-up completions")
    t0 = min(r["arrival_time_ms"] for r in meas)
    t1 = max(r["finish_decode_time_ms"] for r in meas)
    tokens = sum(r["fresh_prompt_tokens"] + r["num_output_tokens"] for r in meas)
    tpot = [
        (r["finish_decode_time_ms"] - r["arrival_time_ms"] - r["ttft_ms"])
        / max(1, r["num_output_tokens"] - 1)
        for r in meas
    ]
    window_s = (t1 - t0) / 1e3
    n_chips = chips(run)
    med_tpot = statistics.median(tpot)
    return {
        "conc": conc,
        "requests": len(meas),
        "chips": n_chips,
        "window_s": window_s,
        "total_tok_s": tokens / window_s,
        "tok_s_per_chip": tokens / window_s / n_chips,
        "median_tpot_ms": med_tpot,
        "intvty": 1000.0 / med_tpot,
        "ttft_s": statistics.median(r["ttft_ms"] for r in meas) / 1e3,
        "e2e_s": statistics.median(r["finish_decode_time_ms"] - r["arrival_time_ms"] for r in meas)
        / 1e3,
    }


def _multiplicity(manifest: dict) -> list[int]:
    """Per slot: the product of the Scale folds above it."""
    nodes = manifest["nodes"]
    mult = [1] * len(manifest["slots"])

    def walk(i: int, m: int) -> None:
        node = nodes[i]
        ((op, body),) = node.items()
        if op == "Leaf":
            mult[body] = m
            return
        n = body.get("n", 1) if op == "Scale" else 1
        ch = body["children"]
        for j in range(ch["start"], ch["end"]):
            walk(j, m * n)

    walk(0, 1)
    return mult


def _category(name: str, kind: str) -> str:
    parts = name.split(".")
    if kind == "neuron_launch":
        return "launch+host"
    if kind in ("all_gather", "reduce_scatter"):
        scope = "world" if "world" in name else "tp"
        return f"{kind}({scope})"
    if any(p in ("kda_dense", "kda_moe") for p in parts) and parts[-1] == "kda":
        return "kda"
    return {"dsa": "dsa"}.get(parts[-1], kind)


def breakdown(run: Path) -> dict:
    """{'decode_only'|'with_prefill': {'steps': n, 'step_ms': mean, kinds: {cat: mean ms}}}."""
    import pyarrow.parquet as pq

    out: dict = {}
    sums: dict = {k: defaultdict(float) for k in ("decode_only", "with_prefill")}
    counts = defaultdict(int)
    step_ms = defaultdict(float)
    for mf in sorted((run / "raw" / "cost_manifest").glob("*.json")):
        manifest = json.loads(mf.read_text())["sections"][0]
        mult = _multiplicity(manifest)
        cats = [_category(s["name"], s["kind"]) for s in manifest["slots"]]
        fwd = ["mtp" if ".mtp" in s["name"] else s["name"].split(".")[1] for s in manifest["slots"]]
        log = run / "raw" / "cost_log" / (mf.stem + ".parquet")
        t = pq.read_table(log, columns=["total_time_ms", "slot_time_ms", "groups"])
        for row in t.to_pylist():
            pre = any(g["prefill_tokens"] > 0 for g in row["groups"])
            key = "with_prefill" if pre else "decode_only"
            counts[key] += 1
            step_ms[key] += row["total_time_ms"]
            for ms, m, c, f in zip(row["slot_time_ms"], mult, cats, fwd):
                if ms:
                    sums[key][f"{f}:{c}" if f in ("decode", "prefill", "mtp") else c] += ms * m
    for key in sums:
        n = counts[key]
        if not n:
            continue
        out[key] = {
            "steps": n,
            "step_ms": step_ms[key] / n,
            "kinds": dict(
                sorted(((k, v / n) for k, v in sums[key].items()), key=lambda kv: -kv[1])
            ),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run", type=Path)
    ap.add_argument("--conc", type=int, required=True)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--breakdown", action="store_true")
    a = ap.parse_args()
    print(json.dumps(point(a.run, a.conc, a.warmup), indent=1))
    if a.breakdown:
        print(json.dumps(breakdown(a.run), indent=1))


if __name__ == "__main__":
    main()
