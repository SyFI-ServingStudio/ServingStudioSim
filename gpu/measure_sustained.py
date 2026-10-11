"""Measure a GPU's sustained GEMM TFLOP/s and copy GB/s for ``gpu/spec.json``.

The optimality analyzer's R4a rung (sustained limit) prices compute work at the
best rate a GEMM of the same dtype holds when the GPU runs it back to back, and
memory work at a large back-to-back copy. That differs from the peak rows in
``profile.db``: those are timed with an L2 flush and host dispatch between
launches, so a short kernel runs on a partly idle board and keeps full clocks a
continuous workload cannot hold under the power limit (B200 NVFP4: 6.6 PF peak
row, ~4.0 PF sustained at ~945 MHz).

For each dtype this takes the shape and backend of that dtype's fastest
``single_gemm`` row (``profiling.db.achievable``) and replays it from a CUDA
graph with no L2 flush (``python -m profiling measure --cuda-graph
--no-clear-l2``); the rate over the window's last second is the sustained one.
The copy is the fastest >= 1 GiB ``elementwise`` copy row, measured the same
way.

Run on an idle GPU of the target model (through Slurm where available), from
the ServingStudioSim root::

    uv run python gpu/measure_sustained.py --gpu-name "NVIDIA B200" --write

``--write`` stores ``{"tflops": {dtype: ...}, "copy_gbps": ...}`` as the
``sustained`` block of the matching ``gpu/spec.json`` entry. Artifacts (trend
CSVs, telemetry, plots) go to ``--output-dir``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from profiling.db.achievable import achievable_peaks  # noqa: E402

SPEC_PATH = ROOT / "gpu" / "spec.json"
DB_PATH = ROOT / "profiling" / "profile.db"


def _measure_last_second_ms(
    table: str, backend: str, spec: dict, out: Path, duration_s: float
) -> float:
    """Mean kernel ms over the last second of a back-to-back CUDA-graph window."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "profiling",
            "measure",
            table,
            "--backend",
            backend,
            "--spec",
            json.dumps(spec),
            "--cuda-graph",
            "--no-clear-l2",
            "--duration-s",
            str(duration_s),
            "--output-dir",
            str(out),
        ],
        cwd=ROOT,
        check=True,
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    return float(summary["runtime_ms"]["last_1s_mean"])


def sustained_gemm_tflops(
    dtype: str, backend: str, shape: dict, duration_s: float, output_dir: Path
) -> float:
    """Back-to-back TFLOP/s of one ``single_gemm`` shape over its last second."""
    ms = _measure_last_second_ms(
        "single_gemm", backend, {**shape, "dtype": dtype}, output_dir / f"gemm_{dtype}", duration_s
    )
    return 2 * shape["m"] * shape["n"] * shape["k"] / (ms * 1e-3) / 1e12


def sustained_copy_gbps(
    backend: str, input_bytes: int, duration_s: float, output_dir: Path
) -> float:
    """Back-to-back elementwise copy GB/s (read + write) over its last second."""
    spec = {"input_size_bytes": input_bytes, "output_size_bytes": input_bytes}
    ms = _measure_last_second_ms("elementwise", backend, spec, output_dir / "copy", duration_s)
    return 2 * input_bytes / (ms * 1e-3) / 1e9


def write_spec(gpu_name: str, sustained: dict) -> None:
    document = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    for gpu in document["gpus"]:
        if gpu_name == gpu["name"] or gpu_name in gpu.get("aliases", []):
            gpu["sustained"] = sustained
            break
    else:
        raise SystemExit(f"{gpu_name!r} matches no gpu/spec.json entry")
    SPEC_PATH.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--gpu-name", required=True, help='profile.db gpu_name, e.g. "NVIDIA B200".'
    )
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/sustained"))
    parser.add_argument("--write", action="store_true", help="Store the result in gpu/spec.json.")
    args = parser.parse_args(argv)

    # nvidia-smi, not torch: a CUDA context here would hold memory on the card
    # and make `profiling measure` reject it as busy.
    observed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "--id=0"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if observed != args.gpu_name:
        raise SystemExit(f"visible GPU is {observed!r}, not {args.gpu_name!r}")

    peaks = achievable_peaks(DB_PATH, args.gpu_name)
    tflops = {}
    for dtype, row in sorted(peaks["gemm_by_dtype"].items()):
        try:
            tflops[dtype] = round(
                sustained_gemm_tflops(
                    dtype, row["backend"], row["shape"], args.duration_s, args.output_dir
                )
            )
        except subprocess.CalledProcessError:
            # A backend whose profiling env is not provisioned here. The analyzer
            # takes the best dtype of each class, so a sibling dtype still covers it.
            print(f"skipped {dtype} ({row['backend']}): measure failed", file=sys.stderr)
    copy = peaks["copy"]
    copy_gbps = None
    if copy is not None:
        copy_gbps = round(
            sustained_copy_gbps(
                copy["backend"], copy["input_size_bytes"], args.duration_s, args.output_dir
            )
        )
    sustained = {"tflops": tflops, "copy_gbps": copy_gbps}
    print(json.dumps({"gpu_name": args.gpu_name, "sustained": sustained}, indent=2))
    if args.write:
        write_spec(args.gpu_name, sustained)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
