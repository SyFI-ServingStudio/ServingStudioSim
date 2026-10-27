"""Cache fidelity for `gdn_chunk_delta_rule` (Gated DeltaNet chunked prefill).

Thin Recipe-B caller of `cache_fidelity.run_fidelity`. The Rust cache is built
over the re-axis `(L, N = T/L)`, so the generic probes (placed off
`grid_axes`) would not be physical shapes. Probes here are physical
`(num_tokens, max_sequence_length)` points on a generic grid: single long
prefills up to and past the 8192-token chunk, chunk-budget batches of several
full sequences plus a remainder, many short prefills (N past the grid's 16),
sub-chunk sequences, and interior cells. kernel-query projects them and
perf_api profiles them as-is.

    uv run python tools/cache-fidelity-analyzer/gdn_chunk_delta_rule_fidelity.py \\
        --backends fla_triton --gpu-name "NVIDIA L40S" --num-key-heads 16 \\
        --num-heads 48 --log-dir logs/<dated>/fidelity

`--emit-grid-specs PATH` instead writes the feasible grid specs (the rows
`enumerate` asks for) as JSONL, for pre-warming the DB with `profiling run`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

KIND = "gdn_chunk_delta_rule"
# Mirrors simulator/src/timing/kernels/gdn_chunk_delta_rule.rs.
MAX_TOKENS = 65_536


def probes() -> list[tuple[tuple[int, int], str]]:
    out: list[tuple[tuple[int, int], str]] = []
    # One prefill sequence, off-grid in L (N = 1).
    for length in (100, 700, 1500, 3000, 6000, 7840, 10_000, 12_000):
        out.append(((length, length), "single_prefill"))
    # A token budget of full sequences plus one remainder (1 < N <= 16).
    for shape in ((8192, 3000), (8192, 5000), (8192, 6272), (12_000, 5000), (16_384, 6000),
                  (4096, 1000), (6000, 1200), (3000, 300)):
        out.append((shape, "multi_prefill"))
    # Many short prefills: N past the grid's 16.
    for shape in ((8192, 128), (8192, 200), (8192, 300), (4096, 100), (8192, 64),
                  (16_384, 256), (6000, 150)):
        out.append((shape, "many_short"))
    # Sequences shorter than one 64-token chunk (L below the grid).
    for shape in ((500, 40), (2000, 32), (600, 8)):
        out.append((shape, "sub_chunk"))
    # Interior points strictly inside grid cells on both axes.
    for shape in ((5000, 1500), (2500, 350), (13_000, 3000), (9500, 4500), (700, 90)):
        out.append((shape, "interior"))
    return out


# Mirrors `sweep_grid`: FLA configs add sub-chunk L, the 1536/3072 bend and N
# up to 256; FlashInfer configs keep L = 64..16384, N = 1..16.
FLA_LENGTHS = (8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096, 8192, 16384)
FLASHINFER_LENGTHS = tuple(1 << e for e in range(6, 15))


def grid_specs(dims: dict, backends: list[str]) -> list[dict]:
    fla = "fla_triton" in backends
    lengths = FLA_LENGTHS if fla else FLASHINFER_LENGTHS
    counts = [1 << e for e in range(9 if fla else 5)]
    specs = []
    for length in lengths:
        for full in counts:
            if length * full > MAX_TOKENS:
                continue
            specs.append({"num_tokens": length * full, "max_sequence_length": length, **dims})
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backends", default="fla_triton")
    ap.add_argument("--gpu-name", default="NVIDIA L40S")
    ap.add_argument("--num-key-heads", type=int, default=16)
    ap.add_argument("--num-heads", type=int, default=48)
    ap.add_argument("--key-head-dim", type=int, default=128)
    ap.add_argument("--value-head-dim", type=int, default=128)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--sim-bin", default="target/release/simulator")
    ap.add_argument("--log-dir", type=Path, default=None)
    ap.add_argument("--emit-grid-specs", type=Path, default=None)
    args = ap.parse_args()

    dims = {
        "num_key_heads": args.num_key_heads,
        "num_heads": args.num_heads,
        "key_head_dim": args.key_head_dim,
        "value_head_dim": args.value_head_dim,
        "dtype": args.dtype,
    }
    if args.emit_grid_specs:
        specs = grid_specs(dims, args.backends.split(","))
        args.emit_grid_specs.write_text("".join(json.dumps(s) + "\n" for s in specs))
        print(f"{len(specs)} grid specs -> {args.emit_grid_specs}")
        return

    import cache_fidelity as cf

    config = {"backends": args.backends.split(","), "gpu_name": args.gpu_name, **dims}
    cf.run_fidelity(args.sim_bin, KIND, config, probes=probes(), log_dir=args.log_dir)


if __name__ == "__main__":
    main()
