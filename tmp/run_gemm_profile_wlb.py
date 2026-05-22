"""Run a forced single-GEMM perf_api sweep and print local GPU chunk usage.

This is an operator/debug script, not a library entry point. It intentionally
enters through ``profiling.perf_api`` so it exercises the same public facade
path used by build-cache callers while still exposing enough local-pool tracing
to inspect static WLB across GPUs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from profiling import perf_api  # noqa: E402
from profiling.db.batch import args_to_spec, coerce_args  # noqa: E402
from profiling.db.kind import KernelKind  # noqa: E402
from profiling.db.registry import find_kernel_profiler_spec  # noqa: E402
from profiling.db.table import MissingEntry  # noqa: E402
from profiling.exec import GpuChunk, GpuPool, set_default_pool  # noqa: E402
from profiling.exec.env import resolve_profile_env  # noqa: E402
from profiling.exec.local import find_idle_gpus  # noqa: E402
from profiling.exec.payload import (  # noqa: E402
    chunk_result_from_payload,
    metrics_to_payload,
    resolve_chunk_backend,
)
from profiling.runners.metrics import ComputeMetrics  # noqa: E402

DEFAULT_M_VALUES = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192]


@dataclass(frozen=True)
class ChunkTrace:
    gpus: tuple[int, ...]
    m_values: tuple[int, ...]
    elapsed_s: float
    controller_prepare_s: float
    subprocess_s: float
    controller_decode_s: float
    worker_timing: dict[str, Any]


@dataclass(frozen=True)
class TimingBreakdown:
    cuda_ready_s: float
    db_prepare_s: float
    force_facade_s: float
    post_cache_check_s: float
    total_s: float
    post_missing_count: int


class TracingLocalGpuPool(GpuPool):
    """LocalGpuPool-compatible wrapper that records chunk-to-spec assignment."""

    def __init__(self, gpus: list[int] | None):
        self.gpus = gpus
        self.detected_gpus: list[int] | None = None
        self.detect_elapsed_s = 0.0
        self.chunk_traces: list[ChunkTrace] = []
        self._lock = Lock()

    def acquire_chunks(self, k: int, max_concurrent: int) -> Iterator[GpuChunk]:
        # Keep this splitter aligned with LocalGpuPool; only the chunk wrapper is
        # different so the script reports WLB without changing profiler behavior.
        gpus_per_chunk = k
        if gpus_per_chunk < 1:
            raise ValueError(f"k must be >= 1, got {gpus_per_chunk}")
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")
        detect_start_s = time.perf_counter()
        available_gpus = self.gpus if self.gpus is not None else find_idle_gpus()
        self.detect_elapsed_s = time.perf_counter() - detect_start_s
        self.detected_gpus = list(available_gpus)
        if len(available_gpus) < gpus_per_chunk:
            raise RuntimeError(
                f"need {gpus_per_chunk} GPU(s), found {len(available_gpus)}"
            )

        chunk_count = min(max_concurrent, len(available_gpus) // gpus_per_chunk)
        for chunk_index in range(chunk_count):
            start_gpu_index = chunk_index * gpus_per_chunk
            chunk_gpus = available_gpus[start_gpu_index : start_gpu_index + gpus_per_chunk]
            yield TracingLocalGpuChunk(chunk_gpus, self.chunk_traces, self._lock)


class TracingLocalGpuChunk(GpuChunk):
    def __init__(self, gpus: list[int], chunk_traces: list[ChunkTrace], lock: Lock):
        self.gpus = gpus
        self.chunk_traces = chunk_traces
        self.lock = lock

    def run(self, kernel_kind: KernelKind, specs: list[dict]) -> list[Any]:
        chunk_start_s = time.perf_counter()
        if not specs:
            return []

        prepare_start_s = time.perf_counter()
        backend = resolve_chunk_backend(kernel_kind, specs)
        profiler_spec = find_kernel_profiler_spec(kernel_kind, backend)
        profiler_env = resolve_profile_env(profiler_spec.subprocess_env)
        profiler_env.validate_python_executable()

        with tempfile.TemporaryDirectory(prefix="mlsim-profile-debug-") as tmp:
            input_path = Path(tmp) / "input.json"
            output_path = Path(tmp) / "output.json"
            input_path.write_text(
                json.dumps({"kernel_kind": kernel_kind.value, "specs": specs}),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in self.gpus)
            env["PYTHONPATH"] = _with_project_pythonpath(env.get("PYTHONPATH"))
            controller_prepare_s = time.perf_counter() - prepare_start_s

            subprocess_start_s = time.perf_counter()
            cmd = [
                str(profiler_env.python_executable),
                str(Path(__file__).resolve()),
                "--worker-input",
                str(input_path),
                "--worker-output",
                str(output_path),
            ]
            completed = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            subprocess_s = time.perf_counter() - subprocess_start_s
            if completed.returncode != 0:
                error = (
                    completed.stderr.strip()
                    or completed.stdout.strip()
                    or "debug worker failed"
                )
                results = [
                    chunk_result_from_payload({"ok": False, "error": error})
                    for _ in specs
                ]
                worker_timing: dict[str, Any] = {"error": error}
                controller_decode_s = 0.0
            else:
                decode_start_s = time.perf_counter()
                worker_response = json.loads(output_path.read_text(encoding="utf-8"))
                results = [
                    chunk_result_from_payload(result_payload)
                    for result_payload in worker_response["results"]
                ]
                worker_timing = dict(worker_response.get("debug_timing", {}))
                controller_decode_s = time.perf_counter() - decode_start_s

        with self.lock:
            self.chunk_traces.append(
                ChunkTrace(
                    gpus=tuple(self.gpus),
                    m_values=tuple(int(spec["m"]) for spec in specs),
                    elapsed_s=time.perf_counter() - chunk_start_s,
                    controller_prepare_s=controller_prepare_s,
                    subprocess_s=subprocess_s,
                    controller_decode_s=controller_decode_s,
                    worker_timing=worker_timing,
                )
            )
        return results


def main() -> int:
    total_start_s = time.perf_counter()
    args = _parse_args()
    if args.worker_input is not None or args.worker_output is not None:
        if args.worker_input is None or args.worker_output is None:
            raise ValueError("--worker-input and --worker-output must be provided together")
        _worker_main(args.worker_input, args.worker_output)
        return 0

    explicit_gpus = _parse_gpu_list(args.gpus) if args.gpus else None
    m_values = _parse_int_list(args.m_values)
    profile_specs = [
        {"m": m, "n": args.n, "k": args.k, "dtype": args.dtype}
        for m in m_values
    ]

    cuda_start_s = time.perf_counter()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GEMM profiling")
    gpu_name = args.gpu_name or torch.cuda.get_device_name(0)
    cuda_ready_s = time.perf_counter() - cuda_start_s

    db_prepare_start_s = time.perf_counter()
    db_path = args.db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh_db and db_path.exists():
        db_path.unlink()
    db_prepare_s = time.perf_counter() - db_prepare_start_s

    tracing_pool = TracingLocalGpuPool(explicit_gpus)
    perf_api.DB_PATH = db_path
    set_default_pool(tracing_pool)
    perf_api.enable_jit_profiling()
    force_start_s = time.perf_counter()
    try:
        results = perf_api.get_single_gemm_times(
            profile_specs,
            backend="torch",
            gpu_name=gpu_name,
            force=True,
        )
    finally:
        force_facade_s = time.perf_counter() - force_start_s
        perf_api.disable_jit_profiling()
        set_default_pool(None)

    post_cache_start_s = time.perf_counter()
    post_missing_count = perf_api.count_missing_single_gemm(
        profile_specs,
        backend="torch",
        gpu_name=gpu_name,
    )
    post_cache_check_s = time.perf_counter() - post_cache_start_s
    total_s = time.perf_counter() - total_start_s
    timing_breakdown = TimingBreakdown(
        cuda_ready_s=cuda_ready_s,
        db_prepare_s=db_prepare_s,
        force_facade_s=force_facade_s,
        post_cache_check_s=post_cache_check_s,
        total_s=total_s,
        post_missing_count=post_missing_count,
    )

    _print_summary(
        db_path=db_path,
        explicit_gpus=explicit_gpus,
        detected_gpus=tracing_pool.detected_gpus or [],
        auto_detect_s=tracing_pool.detect_elapsed_s,
        timing_breakdown=timing_breakdown,
        profile_specs=profile_specs,
        chunk_traces=tracing_pool.chunk_traces,
        results=results,
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forced perf_api GEMM WLB validation")
    parser.add_argument("--worker-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--gpus", help="Comma-separated physical GPU ids, e.g. 0,1,2,3")
    parser.add_argument("--m-values", default=",".join(str(m) for m in DEFAULT_M_VALUES))
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("/tmp/mlsim_gemm_wlb_profile.db"),
    )
    parser.add_argument("--gpu-name", help="DB gpu_name override; defaults to torch device 0")
    parser.add_argument("--fresh-db", action="store_true", help="Delete db_path before running")
    return parser.parse_args()


def _parse_int_list(raw_values: str) -> list[int]:
    values = [int(value.strip()) for value in raw_values.split(",") if value.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    return values


def _parse_gpu_list(raw_gpus: str) -> list[int]:
    return _parse_int_list(raw_gpus)


def _worker_main(input_path: Path, output_path: Path) -> None:
    worker_start_s = time.perf_counter()

    input_start_s = time.perf_counter()
    worker_request = json.loads(input_path.read_text(encoding="utf-8"))
    input_read_s = time.perf_counter() - input_start_s

    resolve_start_s = time.perf_counter()
    kernel_kind = KernelKind(worker_request["kernel_kind"])
    chunk_specs = worker_request["specs"]
    backend = resolve_chunk_backend(kernel_kind, chunk_specs)
    profiler_spec = find_kernel_profiler_spec(kernel_kind, backend)
    resolve_s = time.perf_counter() - resolve_start_s

    runner_load_start_s = time.perf_counter()
    runner = profiler_spec.load_runner()
    runner_load_s = time.perf_counter() - runner_load_start_s

    specs_start_s = time.perf_counter()
    worker_results = []
    per_spec_timing = []
    for chunk_spec in chunk_specs:
        spec_start_s = time.perf_counter()
        runner_spec = {key: value for key, value in chunk_spec.items() if key != "backend"}
        coerce_s = 0.0
        runner_s = 0.0
        gpu_name_s = 0.0
        runner_detail: dict[str, float] = {}
        try:
            coerce_start_s = time.perf_counter()
            kernel_args = coerce_args(profiler_spec.args_schema, runner_spec)
            runner_kwargs = args_to_spec(kernel_args)
            coerce_s = time.perf_counter() - coerce_start_s

            runner_start_s = time.perf_counter()
            metrics = _run_runner_with_probe(runner, runner_kwargs, runner_detail)
            runner_s = time.perf_counter() - runner_start_s

            gpu_name_start_s = time.perf_counter()
            gpu_name = _current_gpu_name()
            gpu_name_s = time.perf_counter() - gpu_name_start_s
            worker_results.append(
                {
                    "ok": True,
                    "metrics": metrics_to_payload(metrics),
                    "gpu_name": gpu_name,
                }
            )
            per_spec_timing.append(
                {
                    "ok": True,
                    "m": runner_kwargs.get("m"),
                    "coerce_s": coerce_s,
                    "runner_s": runner_s,
                    "timer_do_bench_s": runner_detail.get("timer_do_bench_s", 0.0),
                    "energy_perf_s": runner_detail.get("energy_perf_s", 0.0),
                    "runner_unattributed_s": runner_s
                    - runner_detail.get("timer_do_bench_s", 0.0)
                    - runner_detail.get("energy_perf_s", 0.0),
                    "gpu_name_s": gpu_name_s,
                    "total_s": time.perf_counter() - spec_start_s,
                }
            )
        except Exception as exc:
            _empty_cuda_cache()
            worker_results.append({"ok": False, "error": str(exc)})
            per_spec_timing.append(
                {
                    "ok": False,
                    "m": runner_spec.get("m"),
                    "coerce_s": coerce_s,
                    "runner_s": runner_s,
                    "timer_do_bench_s": runner_detail.get("timer_do_bench_s", 0.0),
                    "energy_perf_s": runner_detail.get("energy_perf_s", 0.0),
                    "runner_unattributed_s": runner_s
                    - runner_detail.get("timer_do_bench_s", 0.0)
                    - runner_detail.get("energy_perf_s", 0.0),
                    "gpu_name_s": gpu_name_s,
                    "total_s": time.perf_counter() - spec_start_s,
                    "error": str(exc),
                }
            )

    specs_total_s = time.perf_counter() - specs_start_s
    output_payload = {
        "results": worker_results,
        "debug_timing": {
            "input_read_s": input_read_s,
            "resolve_s": resolve_s,
            "runner_load_s": runner_load_s,
            "specs_total_s": specs_total_s,
            "worker_total_before_write_s": time.perf_counter() - worker_start_s,
            "per_spec": per_spec_timing,
        },
    }
    output_path.write_text(json.dumps(output_payload), encoding="utf-8")


def _run_runner_with_probe(runner: Any, runner_kwargs: dict[str, Any], probe: dict[str, float]):
    runner_module = sys.modules.get(getattr(runner, "__module__", ""))
    timer_cls = getattr(runner_module, "Timer", None)
    energy_cls = getattr(runner_module, "Energy", None)
    original_do_bench = getattr(timer_cls, "do_bench", None)
    original_energy_perf = getattr(energy_cls, "perf", None)

    def timed_do_bench(*args: Any, **kwargs: Any):
        start_s = time.perf_counter()
        try:
            return original_do_bench(*args, **kwargs)
        finally:
            probe["timer_do_bench_s"] = probe.get("timer_do_bench_s", 0.0) + (
                time.perf_counter() - start_s
            )

    def timed_energy_perf(*args: Any, **kwargs: Any):
        start_s = time.perf_counter()
        try:
            return original_energy_perf(*args, **kwargs)
        finally:
            probe["energy_perf_s"] = probe.get("energy_perf_s", 0.0) + (
                time.perf_counter() - start_s
            )

    if timer_cls is not None and original_do_bench is not None:
        timer_cls.do_bench = staticmethod(timed_do_bench)
    if energy_cls is not None and original_energy_perf is not None:
        energy_cls.perf = staticmethod(timed_energy_perf)
    try:
        return runner(**runner_kwargs)
    finally:
        if timer_cls is not None and original_do_bench is not None:
            timer_cls.do_bench = staticmethod(original_do_bench)
        if energy_cls is not None and original_energy_perf is not None:
            energy_cls.perf = staticmethod(original_energy_perf)


def _current_gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except ImportError:
        pass
    return None


def _empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _with_project_pythonpath(existing: str | None) -> str:
    if not existing:
        return str(PROJECT_ROOT)
    return os.pathsep.join([str(PROJECT_ROOT), existing])


def _print_summary(
    *,
    db_path: Path,
    explicit_gpus: list[int] | None,
    detected_gpus: list[int],
    auto_detect_s: float,
    timing_breakdown: TimingBreakdown,
    profile_specs: list[dict],
    chunk_traces: list[ChunkTrace],
    results: list[ComputeMetrics | MissingEntry],
) -> None:
    used_gpus = sorted({gpu for trace in chunk_traces for gpu in trace.gpus})
    expected_gpus = explicit_gpus if explicit_gpus is not None else detected_gpus
    max_chunk_worker_s = max((trace.elapsed_s for trace in chunk_traces), default=0.0)
    sum_chunk_worker_s = sum(trace.elapsed_s for trace in chunk_traces)
    controller_overhead_s = timing_breakdown.force_facade_s - max_chunk_worker_s
    print(f"db_path={db_path}")
    print(f"explicit_gpus={explicit_gpus}")
    print(f"auto_detected_gpus={detected_gpus}")
    print(f"used_gpus={used_gpus}")
    print(f"all_detected_gpus_used={set(expected_gpus).issubset(used_gpus)}")
    print("timing_breakdown:")
    print(f"  total_s={timing_breakdown.total_s:.3f}")
    print(f"  cuda_ready_s={timing_breakdown.cuda_ready_s:.3f}")
    print(f"  db_prepare_s={timing_breakdown.db_prepare_s:.3f}")
    print(f"  auto_detect_s={auto_detect_s:.3f}")
    print(f"  force_facade_s={timing_breakdown.force_facade_s:.3f}")
    print(f"  max_chunk_worker_s={max_chunk_worker_s:.3f}")
    print(f"  sum_chunk_worker_s={sum_chunk_worker_s:.3f}")
    print(f"  controller_db_overhead_est_s={controller_overhead_s:.3f}")
    print(f"  post_cache_check_s={timing_breakdown.post_cache_check_s:.3f}")
    print(f"  post_missing_count={timing_breakdown.post_missing_count}")
    print("chunk_assignments:")
    for trace in sorted(chunk_traces, key=lambda item: item.gpus):
        worker_total_s = float(trace.worker_timing.get("worker_total_before_write_s", 0.0))
        subprocess_unattributed_s = trace.subprocess_s - worker_total_s
        print(
            "  "
            f"gpus={list(trace.gpus)} "
            f"m_values={list(trace.m_values)} "
            f"elapsed_s={trace.elapsed_s:.3f} "
            f"controller_prepare_s={trace.controller_prepare_s:.3f} "
            f"subprocess_s={trace.subprocess_s:.3f} "
            f"controller_decode_s={trace.controller_decode_s:.3f} "
            f"worker_total_before_write_s={worker_total_s:.3f} "
            f"subprocess_unattributed_s={subprocess_unattributed_s:.3f}"
        )
        if trace.worker_timing:
            print(
                "    worker: "
                f"input_read_s={float(trace.worker_timing.get('input_read_s', 0.0)):.3f} "
                f"resolve_s={float(trace.worker_timing.get('resolve_s', 0.0)):.3f} "
                f"runner_load_s={float(trace.worker_timing.get('runner_load_s', 0.0)):.3f} "
                f"specs_total_s={float(trace.worker_timing.get('specs_total_s', 0.0)):.3f}"
            )
            for spec_timing in trace.worker_timing.get("per_spec", []):
                status = "ok" if spec_timing.get("ok") else "error"
                print(
                    "    spec: "
                    f"m={spec_timing.get('m')} "
                    f"status={status} "
                    f"total_s={float(spec_timing.get('total_s', 0.0)):.3f} "
                    f"coerce_s={float(spec_timing.get('coerce_s', 0.0)):.3f} "
                    f"runner_s={float(spec_timing.get('runner_s', 0.0)):.3f} "
                    f"timer_do_bench_s={float(spec_timing.get('timer_do_bench_s', 0.0)):.3f} "
                    f"energy_perf_s={float(spec_timing.get('energy_perf_s', 0.0)):.3f} "
                    "runner_unattributed_s="
                    f"{float(spec_timing.get('runner_unattributed_s', 0.0)):.3f} "
                    f"gpu_name_s={float(spec_timing.get('gpu_name_s', 0.0)):.3f}"
                )

    print("results:")
    print("  m,time_ms,tflops,energy_j")
    for spec, result in zip(profile_specs, results, strict=True):
        if isinstance(result, MissingEntry):
            print(f"  {spec['m']},MISSING,MISSING,MISSING")
            continue
        print(
            "  "
            f"{spec['m']},"
            f"{result.time_ms:.6f},"
            f"{result.tflops:.6f},"
            f"{result.energy_j:.6f}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
