"""FlashInfer causal-prefill attention: the worked instance of the generic
cache-fidelity harness (`cache_fidelity.py` in this folder).

Why a dedicated caller (vs the generic CLI): the prefill cache is built over the
re-axis `(A = prefix_len + append_len/2, B = append_len)`, so its `grid_axes`
are NOT the physical query space. Generic probe placement (off `grid_axes`)
would emit `(A,B)` points, not physical `(prefix_len, append_len)` shapes. So we
place probes on a PHYSICAL reference grid here and hand them to `run_fidelity`;
the kernel's `coords()` projection (Rust) and perf_api (physical specs) make the
re-axis invisible to the generic core.

This also adds attention-domain probe regions the generic sweeper can't know:
the fresh-prefill diagonal (`prefix=0`, pure `q²/2` curvature), near-fresh
(`k≪q`), and small/memory-bound shapes — where bilinear-in-(k,q) is weakest and
the `(A,B)` re-axis earns its keep (see
`agent-trace/flashinfer_attn_prefill_reaxis.md`).

Run from the repo root under uv:

    uv run python tools/cache-fidelity-analyzer/flashinfer_attn_fidelity.py \
        --model-config model/config/llama3_8b.json --gpu-name "NVIDIA H200" \
        --backends fa2,fa3 --log-dir logs/<dated>/fidelity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cache_fidelity as cf  # the generic core (sibling file)

# HuggingFace torch_dtype -> the wire dtype string the Rust DType / perf_api use.
_DTYPE_WIRE = {
    "bfloat16": "bf16", "bf16": "bf16",
    "float16": "fp16", "fp16": "fp16",
    "float32": "fp32", "fp32": "fp32",
}

# PHYSICAL reference grid for probe placement only (NOT the kernel's (A,B) cache
# grid). Probes are off-grid physical shapes placed at cell midpoints of these —
# independent of the cache's axes. append mirrors the cache's dense low-q ramp so
# probes densely sample the mid-q region (the GPU efficiency trough).
_PREFIX_REF = [0, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
_APPEND_REF = [128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048,
               2560, 3072, 3584, 4096, 8192, 16384, 32768]


def load_dims(model_config: Path) -> dict:
    """Attention dims + dtype from a HuggingFace config.json. NOTE: raw model
    dims — correct while TP/EP doesn't shard heads; once sharding lands, read the
    resolved dims from the kernel-query `describe_config` instead."""
    cfg = json.loads(model_config.read_text())
    head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
    dt = _DTYPE_WIRE[cfg["torch_dtype"]]
    return {
        "num_qo_heads": cfg["num_attention_heads"],
        "num_kv_heads": cfg["num_key_value_heads"],
        "head_dim": head_dim,
        "q_dtype": dt,
        "kv_dtype": dt,
        "o_dtype": dt,
    }


def attn_probes() -> list[tuple[tuple, str]]:
    """Physical off-grid probes `((prefix_len, append_len), region)` in
    `input_fields` order. Regions stress where causal-attention interp is
    weakest: the fresh-prefill diagonal, near-fresh, small/memory-bound, interior
    cell midpoints, and beyond-grid extrapolation."""
    g = cf.geomean
    pre, app = _PREFIX_REF, _APPEND_REF
    probes: list[tuple[tuple, str]] = []

    # fresh-prefill diagonal: prefix=0, append at each append-cell midpoint.
    for j in range(len(app) - 1):
        probes.append(((0, round(g(app[j], app[j + 1]))), "fresh"))

    # interior cell midpoints (geometric, both axes), strided to bound GPU cost.
    for i in range(1, len(pre) - 1, 2):
        for j in range(0, len(app) - 1, 2):
            probes.append(((round(g(pre[i], pre[i + 1])), round(g(app[j], app[j + 1]))),
                           "interior"))

    # near-fresh: small prefix, large append (k << q, high curvature).
    for q in (app[-3], app[-1]):
        probes.append(((64, round(g(q, app[-2]))), "near-fresh"))

    # small / memory-bound shapes off-grid.
    for p, q in ((192, 192), (300, 200), (64, 192)):
        probes.append(((p, q), "small"))

    # extrapolation beyond the grid max on either axis (where the (A,B) re-axis
    # matters most — the base raw-(k,q) cache biased low on the append axis here).
    pmax, qmax = pre[-1], app[-1]
    for p, q in ((0, round(qmax * 1.5)), (0, qmax * 2),
                 (round(pmax * 1.5), 4096), (round(pmax * 1.5), round(qmax * 1.5))):
        probes.append(((p, q), "extrap"))

    return probes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-config", required=True, type=Path)
    ap.add_argument("--gpu-name", required=True)
    ap.add_argument("--backends", default="fa2,fa3")
    ap.add_argument("--kind", default="flashinfer_attn_prefill")
    ap.add_argument("--compare", default=None,
                    help="optional second KIND to eval on the same physical probes")
    ap.add_argument("--sim-bin", default="target/release/simulator")
    ap.add_argument("--log-dir", type=Path, default=None)
    args = ap.parse_args()

    if not Path(args.sim_bin).exists():
        raise SystemExit(f"{args.sim_bin} not found — build it first:\n"
                         f"  uv run cargo build --release -p simulator")

    config = {
        "backends": args.backends.split(","),
        "gpu_name": args.gpu_name,
        **load_dims(args.model_config),
    }
    cf.run_fidelity(args.sim_bin, args.kind, config,
                    probes=attn_probes(), compare=args.compare, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
