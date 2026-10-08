"""Cache fidelity for `dsa_topk_prefill` at the GLM-5.3-Flash prefill indexer's shapes.

Thin caller of `cache_fidelity.run_fidelity` with physical probes: the launches
vLLM's indexer chunk plan makes (512 MiB of fp32 logits, so `2**27 // pools`
query rows per call over a long request, its short tail slice, and packs of
short requests), plus off-grid interior points. One run per backend, so each
backend's own cache is compared with its own measurement.

    uv run python tools/cache-fidelity-analyzer/dsa_topk_prefill_fidelity.py \
        --backends vllm_cuda,deep_select --log-dir logs/<dated>/fidelity

perf_api JIT-profiles each probe into the active profile DB; point
`VIBESIM_PROFILE_DB` at a scratch copy to keep off-grid rows out of the real one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

KIND = "dsa_topk_prefill"
LOGITS_ELEMS = (512 << 20) // 4


def probes() -> list[tuple[tuple[int, int], str]]:
    out: list[tuple[tuple[int, int], str]] = []
    # A long request's full query slices: rows = 2**27 // pools.
    for pools in (15_360, 30_720, 61_440, 122_880, 245_760, 491_520, 983_040):
        out.append(((LOGITS_ELEMS // pools, pools), "query_slice"))
    # The short last slice of a long request.
    for shape in ((4, 245_761), (100, 200_000), (300, 500_000), (37, 60_000)):
        out.append((shape, "tail_slice"))
    # Packs of short requests (rows x pools within the budget).
    for shape in ((3000, 6000), (2003, 30_000), (6000, 20_000), (700, 1000), (1500, 2500)):
        out.append((shape, "pack"))
    # Interior points inside grid cells on both axes.
    for shape in ((90, 3000), (700, 50_000), (3000, 12_000), (12_000, 40_000), (200, 700_000)):
        out.append((shape, "interior"))
    return out


def logits_row_stride(num_keys: int) -> int:
    """The padded DeepGEMM row the Rust `enumerate` derives from `num_keys`."""
    return -(-num_keys // 256) * 256 + 256


def ground_truth(kind: str, config: dict, input_fields: list[str], shapes: list[tuple]):
    """`cache_fidelity.ground_truth` plus the derived `logits_row_stride` arg,
    which the Rust config does not carry."""
    import math

    from profiling import perf_api

    perf_api.enable_jit_profiling()
    fn = getattr(perf_api, f"get_{kind}_times")
    dims = {k: v for k, v in config.items() if k not in ("backends", "gpu_name")}
    specs = []
    for shape in shapes:
        point = dict(zip(input_fields, shape))
        specs.append({**dims, **point, "logits_row_stride": logits_row_stride(point["num_keys"])})
    best = [math.inf] * len(specs)
    for backend in config["backends"]:
        for i, r in enumerate(fn(specs, backend=backend, gpu_name=config["gpu_name"])):
            best[i] = min(best[i], float(getattr(r, "time_ms", math.inf)))
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backends", default="vllm_cuda,deep_select")
    ap.add_argument("--gpu-name", default="NVIDIA B200")
    ap.add_argument("--top-k", type=int, default=512)
    ap.add_argument("--sim-bin", default="target/release/simulator")
    ap.add_argument("--log-dir", type=Path, default=None)
    args = ap.parse_args()

    import cache_fidelity as cf

    cf.ground_truth = ground_truth
    for backend in args.backends.split(","):
        config = {
            "backends": [backend],
            "gpu_name": args.gpu_name,
            "num_sequences": 1,
            "top_k": args.top_k,
            "logits_dtype": "fp32",
            "index_dtype": "int32",
            "span_mode": "single_causal_tail",
        }
        cf.run_fidelity(
            args.sim_bin,
            KIND,
            config,
            probes=probes(),
            log_dir=args.log_dir / backend if args.log_dir else None,
        )


if __name__ == "__main__":
    main()
