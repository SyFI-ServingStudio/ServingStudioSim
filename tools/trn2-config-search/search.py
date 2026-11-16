"""Simulator-only config search for GLM-5.3-Flash FP8 on one trn2.48xlarge (InferenceX 8k1k).

Every candidate is a glm5-trn2 seed-engine layout (``glm53_flash_neuron_fp8``; two replicas of
32 LNC2 ranks = 16 chips) at every concurrency of the sweep. Axes:

- ``attn_tp`` in {1, 2, 4, 8, 16} (attention DP = 32 / attn_tp groups per replica);
- ``chunk``: the prefill chunk (= the one prefill bucket) in {512, 1024, 2048, 4096};
- ``dmax``: the largest decode bucket per DP group = requests per group (slots = dmax + 1); the
  decode buckets are the powers of two from max(8, attn_tp) to dmax;
- ``regions``: ``layer`` (the engine) or ``block<k>`` (what-if: k layers per region);
- ``host``: host_time_scale (what-if: 1.0 = the trn2-flashnext fit).

A layout whose FP8 weights + preallocated pools exceed a core's 24 GiB is refused by the arch
build (it never runs). One launcher sweep runs every (layout, C); ``inferencex.point`` turns each
run into the InferenceX point, and ``report.md`` answers: the best tok/s/chip with median
interactivity >= X, and the step bottlenecks of the winners.

    uv run --no-sync python tools/trn2-config-search/search.py --stage main --out <dir>
    uv run --no-sync python tools/trn2-config-search/search.py --stage whatif --out <dir>
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from inferencex import breakdown, point  # noqa: E402

CONCS = [4, 8, 16, 32, 64, 128, 256]
TARGETS = [20, 30, 40, 50]
CONTEXT_BUCKETS = [2048, 4096, 8192, 9216]


@dataclass(frozen=True)
class Layout:
    attn_tp: int = 8
    chunk: int = 2048
    dmax: int = 16
    regions: str = "layer"
    host: float = 1.0

    @property
    def label(self) -> str:
        h = "" if self.host == 1.0 else f"_host{self.host:g}"
        return f"tp{self.attn_tp}_chunk{self.chunk}_d{self.dmax}_{self.regions}{h}"

    def decode_buckets(self) -> list[int]:
        b, out = max(8, self.attn_tp), []
        while b <= self.dmax:
            out.append(b)
            b *= 2
        return out or [self.dmax]

    def valid(self) -> bool:
        return self.dmax >= self.attn_tp and self.chunk % self.attn_tp == 0

    def members(self, conc: int, trace_dir: Path) -> dict:
        gran, block = ("layer", 4) if self.regions == "layer" else ("block", int(self.regions[5:]))
        return {
            "conc": conc,
            "traces": [str(trace_dir / f"inferencex_8k1k_c{conc}.csv")],
            "attn_tp": self.attn_tp,
            "prefill_buckets": [self.chunk],
            "decode_buckets": self.decode_buckets(),
            "slots": self.dmax + 1,
            "max_req": self.dmax,
            "chunk": self.chunk,
            "max_batch_tokens": self.chunk + self.dmax,
            "granularity": gran,
            "block_layers": block,
            "host": self.host,
        }


# Model geometry for the HBM check (mirrors arch/glm53_flash_neuron_fp8.rs::memory_report).
H, KDA_LAYERS, DSA_LAYERS, DENSE_LAYERS, MOE_LAYERS = 4096, 34, 11, 3, 42
HBM_GB, RESERVE_GB, CMAX, WORLD = 25.769803776, 2.0, 9216, 32


def memory_gb(lay: Layout) -> dict:
    """Per-rank GB: weights (FP8 FFN, BF16 attention / mHC / head) and the engine's pools."""
    tp, ep = lay.attn_tp, WORLD
    kh = 64 * 128
    kda = KDA_LAYERS * ((4 * H * kh + 2 * 128 * kh + H * 64 + 3 * kh * 4) // tp + 2 * H * 128) * 2
    dsa_sh = 1536 * 64 * 256 + 512 * 64 * 512 + 64 * 256 * H
    dsa_rep = H * 1536 + H * 512 + 1536 * 32 * 128 + H * 128 + H * 32
    dsa = DSA_LAYERS * (dsa_sh // tp + dsa_rep) * 2
    mhc = 45 * 2 * (4 * 6) * 4 * H * 2

    def fp8(e: int) -> int:
        return e + -(-e // (128 * 128)) * 4

    dense = DENSE_LAYERS * fp8(3 * H * 12288 // ep)
    moe = MOE_LAYERS * (fp8(288 * 3 * H * 2048 // ep + 3 * H * 2048 // ep) + 288 * H * 4)
    head = 2 * -(-154880 // tp) * H * 2 + H * 2
    weights = kda + dsa + mhc + dense + moe + head
    hl = 64 // tp
    kda_state = KDA_LAYERS * (hl * 128 * 128 * 4 + 3 * 3 * hl * 128 * 2)
    dsa_tok = DSA_LAYERS * (512 + 256) * 2
    pools = (lay.dmax + 1) * (kda_state + CMAX * dsa_tok)
    return {"weights_gb": weights / 1e9, "pools_gb": pools / 1e9,
            "fits": weights + pools + RESERVE_GB * 1e9 <= HBM_GB * 1e9}


def stage_layouts(stage: str) -> list[Layout]:
    if stage == "main":
        grid = itertools.product([1, 2, 4, 8, 16], [512, 1024, 2048, 4096], [8, 16, 32, 64],
                                 ["layer"], [1.0])
    elif stage == "whatif":
        grid = itertools.product([4, 8, 16], [1024, 2048], [16, 32, 64],
                                 ["block4", "block15", "block45"], [1.0, 0.25])
    elif stage == "smoke":
        grid = [(8, 2048, 16, "layer", 1.0)]
    else:
        raise SystemExit(f"unknown stage {stage}")
    return [lay for lay in (Layout(*g) for g in grid) if lay.valid() and memory_gb(lay)["fits"]]


def refused_layouts(stage: str) -> list[tuple[Layout, dict]]:
    """Valid layouts whose weights + pools do not fit a core (the arch build refuses them)."""
    stage_grid = {"main": ([1, 2, 4, 8, 16], [8, 16, 32, 64])}.get(stage, ([], []))
    out = []
    for tp, d in itertools.product(*stage_grid):
        lay = Layout(attn_tp=tp, dmax=d)
        if lay.valid() and not memory_gb(lay)["fits"]:
            out.append((lay, memory_gb(lay)))
    return out


def preset(layouts: list[Layout], concs: list[int], log_root: Path, trace_dir: Path) -> dict:
    base = yaml.safe_load((ROOT / "presets/trn2/glm53_flash_fp8_trn2_8k1k.yaml").read_text())
    base.pop("compound", None)
    base["workload"]["trace_files"] = "${traces}"
    base["workload"]["max_concurrency"] = "${conc}"
    base["io"]["log_dir"] = str(log_root / "{point}")
    arch = base["pools"]["main"]["groups"][0]["arch"]
    arch.update(attn_tp="${attn_tp}", prefill_buckets="${prefill_buckets}",
                decode_buckets="${decode_buckets}", slots="${slots}",
                region_granularity="${granularity}", block_layers="${block_layers}",
                host_time_scale="${host}", context_buckets=CONTEXT_BUCKETS)
    worker = base["pools"]["main"]["groups"][0]["worker"]
    worker.update(max_batch_tokens="${max_batch_tokens}", max_prefill_chunk_tokens="${chunk}",
                  max_requests_per_partition="${max_req}")
    base["compound"] = {
        "point": {
            f"{lay.label}__c{c}": lay.members(c, trace_dir) for lay in layouts for c in concs
        }
    }
    return base


def run(stage: str, out: Path, parallelism: int, concs: list[int]) -> list[dict]:
    layouts = stage_layouts(stage)
    out.mkdir(parents=True, exist_ok=True)
    trace_dir = ROOT / "presets" / "trn2"
    log_root = out / "runs"
    p = out / f"search_{stage}.yaml"
    p.write_text(yaml.safe_dump(preset(layouts, concs, log_root, trace_dir), sort_keys=False))
    cmd = [sys.executable, "-m", "launcher", str(p), "--no-gpu", "--parallelism", str(parallelism)]
    print("[search]", len(layouts), "layouts x", len(concs), "concurrencies:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=False)
    return collect(layouts, concs, log_root, out / f"results_{stage}.json")


def collect(layouts: list[Layout], concs: list[int], log_root: Path, path: Path) -> list[dict]:
    results = []
    for lay in layouts:
        for c in concs:
            run_dir = log_root / f"{lay.label}__c{c}"
            row = {"layout": lay.label, **asdict(lay)}
            try:
                row.update(point(run_dir, c))
            except Exception as e:  # noqa: BLE001 - an unbuildable layout (HBM) or failed run
                row.update(conc=c, error=_error(run_dir, e))
            results.append(row)
    path.write_text(json.dumps(results, indent=1) + "\n")
    return results


def _error(run_dir: Path, e: Exception) -> str:
    log = run_dir / "stdout.log"
    if log.exists():
        lines = [ln for ln in log.read_text(errors="replace").splitlines() if "exceed" in ln or "rror" in ln]
        if lines:
            return lines[-1][-300:]
    return f"{type(e).__name__}: {e}"[:300]


def best_under(results: list[dict], target: float) -> dict | None:
    ok = [r for r in results if "error" not in r and r["intvty"] >= target]
    return max(ok, key=lambda r: r["tok_s_per_chip"]) if ok else None


def pareto(results: list[dict]) -> list[dict]:
    pts = sorted((r for r in results if "error" not in r), key=lambda r: -r["intvty"])
    out, best = [], -1.0
    for r in pts:
        if r["tok_s_per_chip"] > best:
            out.append(r)
            best = r["tok_s_per_chip"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="main", choices=["smoke", "main", "whatif"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parallelism", type=int, default=64)
    ap.add_argument("--concs", default=",".join(map(str, CONCS)))
    ap.add_argument("--collect-only", action="store_true")
    a = ap.parse_args()
    concs = [int(c) for c in a.concs.split(",")]
    if a.collect_only:
        results = collect(stage_layouts(a.stage), concs, a.out / "runs", a.out / f"results_{a.stage}.json")
    else:
        results = run(a.stage, a.out, a.parallelism, concs)
    ok = [r for r in results if "error" not in r]
    print(f"[search] {len(ok)} points, {len(results) - len(ok)} failed")
    for lay, mem in refused_layouts(a.stage):
        print(f"  refused tp{lay.attn_tp} d{lay.dmax}: {mem['weights_gb']:.2f} GB weights + "
              f"{mem['pools_gb']:.2f} GB pools per rank > {HBM_GB - RESERVE_GB:.2f} GB")
    for x in TARGETS:
        b = best_under(results, x)
        print(f"intvty >= {x}: " + (f"{b['layout']} C={b['conc']} {b['tok_s_per_chip']:.1f} tok/s/chip "
                                    f"(intvty {b['intvty']:.1f})" if b else "none"))
    for r in pareto(results):
        print(f"  pareto {r['layout']:36s} C={r['conc']:4d} {r['tok_s_per_chip']:8.1f} tok/s/chip "
              f"intvty {r['intvty']:6.2f} ttft {r['ttft_s']:.2f}s")
    best = {r["layout"] + f"__c{r['conc']}": r for r in pareto(results)}
    bd = {}
    for key in list(best)[:6]:
        try:
            bd[key] = breakdown(a.out / "runs" / key)
        except Exception as e:  # noqa: BLE001
            bd[key] = {"error": str(e)}
    (a.out / f"breakdown_{a.stage}.json").write_text(json.dumps(bd, indent=1) + "\n")


if __name__ == "__main__":
    main()
