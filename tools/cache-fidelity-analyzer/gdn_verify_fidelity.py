"""Cache fidelity of the Gated DeltaNet speculative (MTP verify) decode kernels.

`gdn_causal_conv_decode`, `gdn_recurrent_decode` and
`gdn_recurrent_decode_gated_norm` carry `tokens_per_request` in their Rust
config, while the profiler args carry `num_tokens = tokens_per_request ·
batch_size`. The generic harness builds truth specs from the config fields, so
this thin caller swaps in a truth function that makes that mapping, then probes
off-grid batch sizes (interior points between grid rows and one point past the grid).

    uv run python tools/cache-fidelity-analyzer/gdn_verify_fidelity.py \
        --gpu-name "NVIDIA B200" --log-dir logs/<dated>/fidelity
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_fidelity as cf  # noqa: E402

_GEOMETRY = {
    "num_qk_heads": 16,
    "num_value_heads": 48,
    "key_head_dim": 128,
    "value_head_dim": 128,
    "dtype": "bf16",
    "state_dtype": "fp32",
}
_INTERIOR = (10, 12, 14, 24, 48, 96, 192)
_EXTRAP = (320,)


def _truth(kind: str, config: dict, input_fields: list[str], shapes: list[tuple]) -> list[float]:
    cf.perf_api.enable_jit_profiling()
    fn = getattr(cf.perf_api, f"get_{kind}_times")
    dims = {k: v for k, v in config.items() if k not in ("backends", "gpu_name")}
    per_request = int(dims.pop("tokens_per_request", 1))
    specs = []
    for shape in shapes:
        spec = {**dims, **dict(zip(input_fields, shape))}
        if per_request != 1 or kind == "gdn_recurrent_decode_gated_norm":
            spec["num_tokens"] = int(spec["batch_size"]) * per_request
        specs.append(spec)
    times = []
    for backend in config["backends"]:
        res = fn(specs, backend=backend, gpu_name=config["gpu_name"])
        times.append([float(getattr(r, "time_ms", math.inf)) for r in res])
    return [min(column) for column in zip(*times)]


def _cases(gpu: str) -> list[tuple[str, dict]]:
    conv = {
        "channels": 10240,
        "kernel_size": 4,
        "dtype": "bf16",
        "state_dtype": "bf16",
    }
    cases = []
    for q in (2, 4):
        cases.append(
            (
                "gdn_causal_conv_decode",
                {"backends": ["vllm_triton"], "gpu_name": gpu, **conv, "tokens_per_request": q},
            )
        )
        cases.append(
            (
                "gdn_recurrent_decode",
                {
                    "backends": ["vllm_triton_sigmoid_gating"],
                    "gpu_name": gpu,
                    **_GEOMETRY,
                    "tokens_per_request": q,
                },
            )
        )
    for q in (1, 4, 8):
        cases.append(
            (
                "gdn_recurrent_decode_gated_norm",
                {
                    "backends": ["vllm_cuda"],
                    "gpu_name": gpu,
                    "tokens_per_request": q,
                    **_GEOMETRY,
                },
            )
        )
    return cases


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu-name", required=True)
    ap.add_argument("--sim-bin", default="target/release/simulator")
    ap.add_argument("--log-dir", type=Path, required=True)
    args = ap.parse_args()

    cf.ground_truth = _truth
    probes = [((b,), "interior") for b in _INTERIOR] + [((b,), "extrap") for b in _EXTRAP]
    for kind, config in _cases(args.gpu_name):
        q = config["tokens_per_request"]
        cf.run_fidelity(
            args.sim_bin, kind, config, probes=probes, log_dir=args.log_dir / f"{kind}_q{q}"
        )


if __name__ == "__main__":
    main()
