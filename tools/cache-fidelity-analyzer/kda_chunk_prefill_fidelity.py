"""Cache fidelity for `kda_chunk_prefill` (GLM-5.3-Flash KDA chunked prefill).

Thin Recipe-B caller of `cache_fidelity.run_fidelity`. The Rust cache is built
over the re-axis `(L, R=(T-D)/L, D)`, so the generic probes (placed off
`grid_axes`) would not be physical shapes. Probes here are physical
`(num_tokens, max_sequence_length, num_decode_sequences)` points; kernel-query
projects them and perf_api profiles them as-is.

    uv run python tools/cache-fidelity-analyzer/kda_chunk_prefill_fidelity.py \
        --log-dir logs/<dated>/fidelity

`--emit-grid-specs PATH` instead writes the 432 feasible grid specs (the rows
`enumerate` asks for) as JSONL, for pre-warming the DB with `profiling run`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

KIND = "kda_chunk_prefill"
MAX_PREFILL_TOKENS = 16_384


def probes() -> list[tuple[tuple[int, int, int], str]]:
    out: list[tuple[tuple[int, int, int], str]] = [((2048, 2019, 29), "capture")]
    # Single prefill + decodes: the R = 1 grid line, off-grid in L and D.
    for length in (100, 700, 1500, 3000, 6000):
        for decodes in (0, 5, 29, 50):
            out.append(((length + decodes, length, decodes), "single_prefill"))
    # Several prefills (full sequences plus a remainder).
    for shape in ((4096, 1000, 12), (8000, 2500, 40), (3000, 300, 0), (6000, 1200, 7)):
        out.append((shape, "multi_prefill"))
    # Sub-chunk prefill sequences (L < 64).
    for shape in ((500, 40, 20), (1000, 16, 8), (300, 50, 3), (2000, 32, 0), (4000, 50, 0),
                  (3016, 24, 16), (600, 8, 0), (1300, 20, 60)):
        out.append((shape, "short_seq"))
    # Decode-heavy with a small prefill.
    for shape in ((70, 10, 60), (200, 150, 50), (45, 3, 40)):
        out.append((shape, "decode_heavy"))
    # Interior points strictly inside grid cells on all three axes.
    for shape in ((5020, 1500, 20), (2510, 350, 10), (13045, 3000, 45), (6024, 180, 24),
                  (9503, 4500, 3)):
        out.append((shape, "interior"))
    return out


def grid_specs(num_heads: int, head_dim: int, dtype: str) -> list[dict]:
    lengths = [2] + [1 << e for e in range(6, 14)]
    full_sequences = [1 << e for e in range(10)]
    decodes = [0, 1, 2, 4, 8, 16, 32, 64]
    specs = []
    for length in lengths:
        for full in full_sequences:
            if length * full > MAX_PREFILL_TOKENS:
                continue
            for d in decodes:
                specs.append({
                    "num_tokens": length * full + d,
                    "max_sequence_length": length,
                    "num_decode_sequences": d,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "dtype": dtype,
                })
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backends", default="vllm_triton")
    ap.add_argument("--gpu-name", default="NVIDIA B200")
    ap.add_argument("--num-heads", type=int, default=16)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--sim-bin", default="target/release/simulator")
    ap.add_argument("--log-dir", type=Path, default=None)
    ap.add_argument("--emit-grid-specs", type=Path, default=None)
    args = ap.parse_args()

    if args.emit_grid_specs:
        specs = grid_specs(args.num_heads, args.head_dim, args.dtype)
        args.emit_grid_specs.write_text("".join(json.dumps(s) + "\n" for s in specs))
        print(f"{len(specs)} grid specs -> {args.emit_grid_specs}")
        return

    import cache_fidelity as cf

    config = {
        "backends": args.backends.split(","),
        "gpu_name": args.gpu_name,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
    }
    cf.run_fidelity(args.sim_bin, KIND, config, probes=probes(), log_dir=args.log_dir)


if __name__ == "__main__":
    main()
