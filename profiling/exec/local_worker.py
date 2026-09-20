"""Subprocess worker entrypoint for local L1 profiling.

Agent note:
- This module is executed as ``python -m profiling.exec.local_worker`` inside
  the selected ``ProfileEnv``.
- Keep GPU selection, temp-file ownership, and subprocess command construction
  in ``profiling.exec.local``. This worker only consumes a JSON payload,
  lazy-loads the registered runner, executes specs, and writes JSON results.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path

from profiling.db.batch import args_to_spec, coerce_args
from profiling.db.kind import KernelKind
from profiling.db.registry import find_kernel_profiler_spec
from profiling.exec.payload import metrics_to_payload, resolve_chunk_backend
from profiling.instrument import emit, span
from profiling.profilers.energy import set_energy_enabled
from profiling.profilers.measure_context import (
    MeasureContext,
    clear_measure_context,
    set_measure_context,
)
from profiling.runners.metrics import RunnerResult

_PROCESS_STARTED = time.time()


def _worker_main(input_path: Path, output_path: Path) -> None:
    # _PROCESS_STARTED is captured at import, so this span covers the
    # interpreter's own start-up cost as seen from inside: torch and the backend
    # import, CUDA context creation, registry resolution.
    boot_started = _PROCESS_STARTED
    worker_request = json.loads(input_path.read_text(encoding="utf-8"))
    kernel_kind: KernelKind = worker_request["kernel_kind"]
    chunk_specs = worker_request["specs"]
    backend = resolve_chunk_backend(kernel_kind, chunk_specs)
    profiler_spec = find_kernel_profiler_spec(kernel_kind, backend)

    # Settled before any runner loads, because `Energy.perf` is read 96 call
    # sites down and none of those signatures carries policy. Absent key → leave
    # the process default alone, so a payload written by an older controller
    # still runs.
    if "energy" in worker_request:
        set_energy_enabled(bool(worker_request["energy"]))

    # An optional `measure` block turns this run into the trend+telemetry probe:
    # the context makes the shared Timer.cupti seam capture per-launch durations
    # instead of a single mean, without touching the runner. Absent block → the
    # ordinary cache-fill path, byte-identical to before.
    measure_request = worker_request.get("measure")
    measure_context = (
        _build_measure_context(measure_request, kernel_kind, backend, chunk_specs)
        if measure_request is not None
        else None
    )

    # The chunk is homogeneous (one backend/gpu_count — enforced upstream), so the
    # worker dispatch is a single uniform list call. Coercion is hoisted here so
    # every runner receives already-coerced kwargs (the `batched` adapter and the
    # native comm runners stay schema-free). List-native comm runners spawn their
    # rank group once for the whole list; compute runners are wrapped by `batched`.
    runner = profiler_spec.load_list_runner()
    kwargs_list = [
        args_to_spec(coerce_args(profiler_spec.args_schema, _strip_backend(chunk_spec)))
        for chunk_spec in chunk_specs
    ]
    if measure_context is not None:
        set_measure_context(measure_context)
    emit("worker.boot", boot_started, time.time(), kind=kernel_kind, specs=len(kwargs_list))
    try:
        with span("worker.runner", kind=kernel_kind, specs=len(kwargs_list)):
            results = runner(kwargs_list)
    finally:
        if measure_context is not None:
            clear_measure_context()
    gpu_name = _current_gpu_name()
    runtime_versions = _runtime_versions(backend)
    worker_results = [_to_payload(result, gpu_name, runtime_versions) for result in results]

    output: dict = {"results": worker_results}
    if measure_context is not None:
        # `consumed` stays False when the runner never reached Timer.cupti — the
        # driver reads this to reject non-CUPTI kernels for the `measure` verb.
        output["measure"] = {
            "consumed": measure_context.consumed,
            "time_ms": measure_context.time_ms,
            "artifacts": measure_context.artifacts,
        }
    output_path.write_text(json.dumps(output), encoding="utf-8")


def _build_measure_context(
    measure_request: dict,
    kernel_kind: KernelKind,
    backend: str,
    chunk_specs: list[dict],
) -> MeasureContext:
    shape = _strip_backend(chunk_specs[0])
    return MeasureContext(
        output_dir=Path(measure_request["output_dir"]),
        label=f"{kernel_kind}:{backend}",
        shape=shape,
        duration_s=float(measure_request.get("duration_s", 10.0)),
        telemetry_hz=float(measure_request.get("telemetry_hz", 20.0)),
        clear_l2=bool(measure_request.get("clear_l2", True)),
        telemetry=bool(measure_request.get("telemetry", True)),
    )


def _strip_backend(chunk_spec: dict) -> dict:
    """Drop the ``backend`` selector (already resolved to the chunk spec) so the
    remaining keys are exactly the runner's schema args."""
    return {key: value for key, value in chunk_spec.items() if key != "backend"}


def _to_payload(
    result: RunnerResult,
    gpu_name: str | None,
    runtime_versions: dict[str, str | None] | None = None,
) -> dict:
    """Render one ``RunnerResult`` into the worker's on-disk JSON shape that
    ``chunk_result_from_payload`` consumes (unchanged from the single-spec loop)."""
    if result.error is not None or result.metrics is None:
        return {"ok": False, "error": result.error}
    return {
        "ok": True,
        "metrics": metrics_to_payload(result.metrics),
        "gpu_name": gpu_name,
        **(runtime_versions or {}),
    }


def _current_gpu_name() -> str | None:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except ImportError:
        pass
    return None


def _runtime_versions(backend: str) -> dict[str, str | None]:
    try:
        import torch
    except ImportError:
        return {"cuda_version": None, "backend_version": None}

    package = "vllm" if backend.startswith("vllm") else backend
    try:
        backend_version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        backend_version = None
    return {
        "cuda_version": str(torch.version.cuda) if torch.version.cuda else None,
        "backend_version": backend_version,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-input", type=Path, required=True)
    parser.add_argument("--worker-output", type=Path, required=True)
    args = parser.parse_args(argv)
    _worker_main(args.worker_input, args.worker_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
