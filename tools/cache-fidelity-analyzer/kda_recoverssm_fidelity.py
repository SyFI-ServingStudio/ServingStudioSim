"""Cache fidelity for `kda_recoverssm_verify` and `kda_recoverssm_commit`.

Thin caller of `cache_fidelity.run_fidelity`. Both caches are built over their
query coordinates, so probes are physical points placed between and past the
grid:

- verify `(batch_size,)`: batch cell midpoints and batch past 512.
- commit `(batch_size, mean_commit_len)`: batch midpoints at a few lengths,
  the off-grid lengths 3, 5 and 7 at several batches, and batch past 512.

The commit query field is the batch's *mean* accepted length, while the
profiler runs every request at one integer `commit_len`. Ground truth renames
the field and probes only integer lengths, so it checks the length
interpolation, not how a mixed batch averages.

    uv run python tools/cache-fidelity-analyzer/kda_recoverssm_fidelity.py \
        --kind kda_recoverssm_commit --num-heads 64 --num-layers 9 \
        --log-dir logs/<dated>/fidelity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

VERIFY = "kda_recoverssm_verify"
COMMIT = "kda_recoverssm_commit"
MID_BATCHES = (3, 6, 12, 24, 48, 96, 192, 384)
EXTRAP_BATCHES = (768, 1024)


def verify_probes() -> list[tuple[tuple[int, ...], str]]:
    return [((b,), "batch_mid") for b in MID_BATCHES] + [((b,), "extrap") for b in EXTRAP_BATCHES]


def commit_probes(spec_query_len: int) -> list[tuple[tuple[int, ...], str]]:
    out: list[tuple[tuple[int, ...], str]] = []
    for length in (1, 4, spec_query_len):
        out += [((b, length), "batch_mid") for b in MID_BATCHES]
    for length in (3, 5, 7):
        if length < spec_query_len:
            out += [((b, length), "length_mid") for b in (1, 8, 64, 512)]
    out += [((b, length), "diag") for b, length in ((6, 3), (48, 5), (192, 7))]
    out += [((b, length), "extrap") for b in EXTRAP_BATCHES for length in (1, spec_query_len)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kind", choices=(VERIFY, COMMIT), required=True)
    ap.add_argument("--backends", default="vllm_triton")
    ap.add_argument("--gpu-name", default="NVIDIA B200")
    ap.add_argument("--num-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--spec-query-len", type=int, default=8)
    ap.add_argument("--num-layers", type=int, default=9)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--sim-bin", default="target/release/simulator")
    ap.add_argument("--log-dir", type=Path, default=None)
    args = ap.parse_args()

    import cache_fidelity as cf

    config = {
        "backends": args.backends.split(","),
        "gpu_name": args.gpu_name,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "spec_query_len": args.spec_query_len,
        "dtype": args.dtype,
    }
    if args.kind == VERIFY:
        probes = verify_probes()
    else:
        config["num_layers"] = args.num_layers
        probes = commit_probes(args.spec_query_len)
        truth = cf.ground_truth

        def commit_truth(kind, cfg, input_fields, shapes):
            fields = ["commit_len" if f == "mean_commit_len" else f for f in input_fields]
            return truth(kind, cfg, fields, [tuple(int(v) for v in s) for s in shapes])

        cf.ground_truth = commit_truth
    cf.run_fidelity(args.sim_bin, args.kind, config, probes=probes, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
