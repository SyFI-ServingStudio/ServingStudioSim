"""Cache fidelity for TensorFold's copies, casts, gathers and drafter glue (tf-copies lane).

Three stages, so only `truth` needs the GPU (run it under the shared GPU lock):

    uv run python tools/cache-fidelity-analyzer/copies_fidelity.py probes --out-dir DIR
    uv run python tools/cache-fidelity-analyzer/copies_fidelity.py truth  --out-dir DIR
    uv run python tools/cache-fidelity-analyzer/copies_fidelity.py report --out-dir DIR

`probes` takes each config's Rust grid (`kernel-query grid`) and places geometric
cell midpoints, a few grid points (re-measured in the same session as the probes,
to separate session drift from interpolation error) and 1.5x/2x past the last
point. `truth` measures every probe fresh (`launcher kernel-profile run --fresh`,
the tensorfold image, nothing written). `report` evaluates the Rust cache at each
probe (`kernel-query eval`) and prints cache/truth by config and region.

Elementwise probes are units (rows or streams) of the config's byte rates; their
truth spec is the total bytes, as the Rust `enumerate` folds them.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path

import cache_fidelity as cf

ROOT = Path(__file__).resolve().parents[2]
SIM = str(ROOT / "target/release/simulator")
G = "NVIDIA GB10"
WITHIN = 1.15
GPU_LOCK = "/home/yilegu/ServingStudio/tmp/gpu.lock"


def _ew(backend: str, i: int, o: int) -> tuple[str, dict]:
    return "elementwise", {
        "backends": [backend],
        "gpu_name": G,
        "input_bytes_per_token": i,
        "output_bytes_per_token": o,
    }


CONFIGS: dict[str, tuple[str, dict]] = {
    "cast_sampling_row": _ew("tensorfold_torch_cast", 496640, 993280),
    "cat_conv_stream": _ew("tensorfold_torch_cat", 61440, 61440),
    "cat_qkv_row": _ew("tensorfold_torch_cat", 20480, 20480),
    "fp32_add_tap_row": _ew("tensorfold_torch_fp32_add", 20480, 10240),
    "foreach_conv_stream": _ew("tensorfold_torch_foreach_copy", 2949120, 2949120),
    "foreach_kv_row": _ew("tensorfold_torch_foreach_copy", 65536, 65536),
    "cat_strided_head_row": _ew("tensorfold_torch_cat_strided", 197184, 197184),
    "irc_conv_pick": (
        "indexed_row_copy",
        {
            "backends": ["tensorfold_torch"],
            "gpu_name": G,
            "row_elements": 10240,
            "dtype": "bf16",
            "operation": "index_select",
        },
    ),
    "irc_kv_take": (
        "indexed_row_copy",
        {
            "backends": ["tensorfold_torch"],
            "gpu_name": G,
            "row_elements": 1024,
            "dtype": "bf16",
            "operation": "index_select",
        },
    ),
    "embed": (
        "embedding_row_gather",
        {
            "backends": ["tensorfold_torch"],
            "gpu_name": G,
            "num_heads": 1,
            "head_dim": 5120,
            "table_rows": 248320,
            "residency": "device",
            "weight_dtype": "bf16",
        },
    ),
    "rms_eager": (
        "rms_norm",
        {"backends": ["tensorfold_torch_eager"], "gpu_name": G, "hidden": 5120, "dtype": "bf16"},
    ),
    "gsum_5120": (
        "row_group_sums",
        {
            "backends": ["tensorfold_triton"],
            "gpu_name": G,
            "hidden": 5120,
            "group_size": 64,
            "dtype": "bf16",
        },
    ),
    "gsum_25600": (
        "row_group_sums",
        {
            "backends": ["tensorfold_triton"],
            "gpu_name": G,
            "hidden": 25600,
            "group_size": 64,
            "dtype": "bf16",
        },
    ),
    "dconv_res": (
        "dynamic_causal_conv",
        {
            "backends": ["tensorfold_triton"],
            "gpu_name": G,
            "hidden": 5120,
            "group_size": 16,
            "residual": True,
            "dtype": "bf16",
        },
    ),
    "prep_q32": (
        "head_major_qk_norm_rope",
        {
            "backends": ["tensorfold_triton"],
            "gpu_name": G,
            "num_q_heads": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "dtype": "bf16",
        },
    ),
    "prep_kv": (
        "head_major_qk_norm_rope",
        {
            "backends": ["tensorfold_triton"],
            "gpu_name": G,
            "num_q_heads": 0,
            "num_kv_heads": 8,
            "head_dim": 128,
            "dtype": "bf16",
        },
    ),
}


def truth_spec(kind: str, config: dict, field: str, value: int) -> dict:
    if kind == "elementwise":
        return {
            "input_size_bytes": config["input_bytes_per_token"] * value,
            "output_size_bytes": config["output_bytes_per_token"] * value,
        }
    dims = {k: v for k, v in config.items() if k not in ("backends", "gpu_name")}
    return {**dims, field: value}


def stage_probes(out: Path) -> None:
    probes = []
    for name, (kind, config) in CONFIGS.items():
        grid = cf.query_grid(SIM, kind, config)
        field, axis = grid["input_fields"][0], [int(v) for v in grid["grid_axes"][0]]
        pts: list[tuple[int, str]] = []
        for a, b in list(zip(axis, axis[1:]))[::2]:  # every other cell keeps the GPU hold short
            mid = round(math.sqrt(a * b))
            if a < mid < b:
                pts.append((mid, "mid"))
        for v in axis[1 :: max(1, len(axis) // 3)][:3]:
            pts.append((v, "grid"))
        pts += [(round(axis[-1] * 1.5), "extrap"), (axis[-1] * 2, "extrap")]
        for value, region in pts:
            probes.append(
                {
                    "name": name,
                    "kind": kind,
                    "config": config,
                    "field": field,
                    "value": value,
                    "region": region,
                }
            )
    out.mkdir(parents=True, exist_ok=True)
    (out / "probes.json").write_text(json.dumps(probes, indent=0))
    print(f"{len(probes)} probes")


def stage_truth(out: Path) -> None:
    probes = json.loads((out / "probes.json").read_text())
    truth = json.loads((out / "truth.json").read_text()) if (out / "truth.json").exists() else {}
    groups: dict[tuple[str, str], list[int]] = {}
    for i, p in enumerate(probes):
        if str(i) not in truth:
            groups.setdefault((p["kind"], p["config"]["backends"][0]), []).append(i)
    for (kind, backend), idx in groups.items():
        specs = [
            truth_spec(
                probes[i]["kind"], probes[i]["config"], probes[i]["field"], probes[i]["value"]
            )
            for i in idx
        ]
        spec_file = out / f"truth_specs_{kind}__{backend}.json"
        spec_file.write_text(json.dumps(specs))
        run = (
            f"uv run --no-sync python -m launcher kernel-profile run {kind} --backend {backend} "
            f"--gpu-name '{G}' --specs {spec_file} --fresh --json"
        )
        # One shared-GPU lock hold per submission, after the device is idle.
        busy = '"$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"'
        wait = f"while [ -n {busy} ]; do sleep 5; done"
        cmd = ["flock", GPU_LOCK, "bash", "-c", f"{wait}; {run}"]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
        try:
            results = json.loads(proc.stdout)["results"]
        except (json.JSONDecodeError, KeyError):
            print(f"{kind}/{backend}: truth failed\n{proc.stderr[-2000:]}")
            continue
        for i, r in zip(idx, results):
            ms = (r.get("metrics") or {}).get("time_ms")
            if ms:
                truth[str(i)] = ms
        (out / "truth.json").write_text(json.dumps(truth))
        print(f"{kind}/{backend}: {len(results)} probes measured")


def stage_report(out: Path) -> None:
    probes = json.loads((out / "probes.json").read_text())
    truth = json.loads((out / "truth.json").read_text())
    rows = []
    for name in CONFIGS:
        sel = [(i, p) for i, p in enumerate(probes) if p["name"] == name]
        kind, config = sel[0][1]["kind"], sel[0][1]["config"]
        res = cf.eval_kind(SIM, kind, config, [{p["field"]: p["value"]} for _, p in sel])
        for (i, p), r in zip(sel, res):
            t = truth.get(str(i))
            rows.append(
                {
                    "name": name,
                    "region": p["region"],
                    "value": p["value"],
                    "cache_ms": r["time_ms"],
                    "truth_ms": t,
                    "coverage": r["coverage"],
                    "ratio": r["time_ms"] / t if t else None,
                }
            )
    (out / "fidelity.json").write_text(json.dumps(rows, indent=0))
    lines = []
    for name in CONFIGS:
        for region in ("mid", "grid", "extrap", "all"):
            rs = [
                r["ratio"]
                for r in rows
                if r["name"] == name and r["ratio"] and region in ("all", r["region"])
            ]
            if not rs:
                continue
            within = sum(1 for x in rs if 1 / WITHIN <= x <= WITHIN)
            worst = max(rs, key=lambda x: abs(math.log(x)))
            lines.append(
                f"{name:22s} {region:6s} med={statistics.median(rs):.3f} "
                f"within={within}/{len(rs)} worst={worst:.3f}"
            )
    bad = [r for r in rows if r["ratio"] and not 1 / WITHIN <= r["ratio"] <= WITHIN]
    lines.append("outside +-15%:")
    lines += [
        f"  {r['name']:22s} {r['region']:6s} {r['value']:>8} cache={r['cache_ms']:.4f} "
        f"truth={r['truth_ms']:.4f} ratio={r['ratio']:.3f}"
        for r in bad
    ]
    (out / "fidelity_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("stage", choices=["probes", "truth", "report"])
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    {"probes": stage_probes, "truth": stage_truth, "report": stage_report}[args.stage](args.out_dir)


if __name__ == "__main__":
    main()
