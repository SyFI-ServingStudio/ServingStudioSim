"""Cache fidelity for `rs_sconv_ag_residual_rms_norm` (Lamport TP collective).

Thin caller of `cache_fidelity.run_fidelity` with physical token probes. The
generic probes are not used because their `extrap` points sit above the
Lamport buffer capacity (16384 tokens), where the runner rejects the shape and
vLLM falls back to NCCL. Probes sit between grid points inside `[1, 16384]`,
grouped by the launch regime (phase warps 4 below 128 tokens, 16 up to 2048,
8 above).

    uv run python tools/cache-fidelity-analyzer/rs_sconv_ag_residual_rms_norm_fidelity.py \
        --has-shared false --log-dir logs/<dated>/fidelity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_fidelity as cf  # noqa: E402

KIND = "rs_sconv_ag_residual_rms_norm"
DEFAULT_TOKENS = (
    10, 14, 26, 30, 36, 44, 52, 60, 72, 88, 104, 120, 136, 176, 208, 240, 288, 352, 416, 480,
    576, 704, 832, 960, 1056, 1152, 1408, 1664, 1920, 2016, 2304, 2816, 3584, 5120, 7168,
    10240, 14336,
)


def region(num_tokens: int) -> str:
    if num_tokens <= 64:
        return "decode"
    if num_tokens < 128:
        return "small"
    if num_tokens <= 2048:
        return "mid"
    return "large"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backends", default="vllm_lamport_triton")
    ap.add_argument("--gpu-name", default="NVIDIA B200")
    ap.add_argument("--num-gpus", type=int, default=4)
    ap.add_argument("--hidden-dim", type=int, default=4096)
    ap.add_argument("--has-shared", choices=("true", "false"), default="false")
    ap.add_argument("--tokens", default=",".join(map(str, DEFAULT_TOKENS)))
    ap.add_argument("--sim-bin", default="target/release/simulator")
    ap.add_argument("--log-dir", type=Path, default=None)
    args = ap.parse_args()

    config = {
        "backends": args.backends.split(","),
        "gpu_name": args.gpu_name,
        "num_gpus": args.num_gpus,
        "hidden_dim": args.hidden_dim,
        "dtype": "bf16",
        "fabric": "nvlink",
        "conv_kernel_size": 4,
        "has_shared": args.has_shared == "true",
    }
    probes = [((t,), region(t)) for t in (int(t) for t in args.tokens.split(","))]
    cf.run_fidelity(args.sim_bin, KIND, config, probes=probes, log_dir=args.log_dir)


if __name__ == "__main__":
    main()
