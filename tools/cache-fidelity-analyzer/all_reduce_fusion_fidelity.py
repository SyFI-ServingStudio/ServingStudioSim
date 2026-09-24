"""Cache fidelity for `all_reduce_fusion` (FlashInfer standalone TP all-reduce).

Thin caller of `cache_fidelity.run_fidelity` with physical token probes. The
generic probes are not used because their `extrap` points sit above the
workspace cap, where the runner rejects the shape and vLLM uses another
all-reduce. Probes stay within `[1, cap]` and are grouped by the MNNVL strategy
regime (one-shot iff `T * H * N * elem <= 1 MiB`).

    uv run python tools/cache-fidelity-analyzer/all_reduce_fusion_fidelity.py \
        --backends flashinfer_mnnvl --num-gpus 4 --hidden-dim 4096 \
        --log-dir logs/<dated>/fidelity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_fidelity as cf  # noqa: E402

KIND = "all_reduce_fusion"
DEFAULT_TOKENS = (8, 24, 31, 34, 40, 100, 300, 700, 1500, 3000)


def region(num_tokens: int, last_oneshot: int) -> str:
    if num_tokens <= last_oneshot:
        return "oneshot"
    if num_tokens <= 256:
        return "twoshot_small"
    if num_tokens <= 1024:
        return "twoshot_mid"
    return "twoshot_large"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backends", default="flashinfer_mnnvl")
    ap.add_argument("--gpu-name", default="NVIDIA B200")
    ap.add_argument("--num-gpus", type=int, default=4)
    ap.add_argument("--hidden-dim", type=int, default=4096)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--tokens", default=",".join(map(str, DEFAULT_TOKENS)))
    ap.add_argument("--sim-bin", default="target/release/simulator")
    ap.add_argument("--log-dir", type=Path, default=None)
    args = ap.parse_args()

    config = {
        "backends": args.backends.split(","),
        "gpu_name": args.gpu_name,
        "num_gpus": args.num_gpus,
        "hidden_dim": args.hidden_dim,
        "dtype": args.dtype,
        "fabric": "nvlink",
    }
    elem = {"bf16": 2, "fp16": 2, "fp32": 4}[args.dtype]
    last_oneshot = (1 << 20) // (args.hidden_dim * args.num_gpus * elem)
    tokens = [int(t) for t in args.tokens.split(",")]
    probes = [((t,), region(t, last_oneshot)) for t in tokens]
    cf.run_fidelity(args.sim_bin, KIND, config, probes=probes, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
