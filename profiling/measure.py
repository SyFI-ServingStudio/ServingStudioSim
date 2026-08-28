"""Driver for ``python -m profiling measure`` — the trend+telemetry diagnostic.

This is an L1a-side, **cache-free** verb. Given one kernel spec it picks an idle
GPU and spawns the profiling worker with a ``measure`` block, so the shared
``Timer.cupti`` seam runs a sustained per-launch duration trend + NVML telemetry
(see ``profilers/trend.py``) and writes CSV plus ``summary.json`` into
``output_dir``. PNG rendering is optional. It never writes ``profile.db``.

Only CUPTI-timed compute kernels are supported: comm kinds are rejected up front,
and a compute runner that never reaches ``Timer.cupti`` is reported unsupported
via the worker's ``consumed`` flag (no hardcoded allowlist).
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from profiling.db.kind import KernelKind
from profiling.db.registry import (
    MetricFamily,
    find_kernel_profiler_spec,
    resolve_spec_backend,
)
from profiling.exec.env import (
    ContainerProfileEnv,
    ProfileEnv,
    compose_library_path,
    compose_pythonpath,
    resolve_profile_env,
)
from profiling.exec.local import _container_worker_command, find_idle_gpus


class MeasureError(RuntimeError):
    """A ``measure`` invocation could not run (no GPU, unsupported kernel, worker fail)."""


def measure_kernel(
    kernel_kind: KernelKind,
    spec: dict[str, Any],
    *,
    backend: str | None = None,
    gpu_name: str | None = None,
    output_dir: str | Path,
    duration_s: float = 10.0,
    telemetry_hz: float = 20.0,
    clear_l2: bool = True,
    telemetry: bool = True,
) -> dict[str, Any]:
    """Run one trend+telemetry capture for ``spec`` and return the artifact summary."""

    # ``gpu_name`` is the requested cache key for the measurement metadata (may be
    # None). The physical GPU is selected by idleness, not by DB key; the
    # worker-observed physical name is carried back separately as provenance.
    spec = dict(spec)
    if backend is not None:
        spec["backend"] = backend
    resolved_backend = resolve_spec_backend(kernel_kind, spec)
    profiler_spec = find_kernel_profiler_spec(kernel_kind, resolved_backend)
    if profiler_spec.metric_family != MetricFamily.COMPUTE:
        raise MeasureError(
            f"measure supports COMPUTE (CUPTI) kernels only; "
            f"{kernel_kind}:{resolved_backend} is {profiler_spec.metric_family.value}"
        )

    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    idle_gpus = find_idle_gpus()
    if not idle_gpus:
        raise MeasureError("no idle GPU found for measure")
    gpu_index = idle_gpus[0]

    profiler_env = resolve_profile_env(profiler_spec.subprocess_env)
    profiler_env.validate()

    response = _run_worker(
        kernel_kind=kernel_kind,
        spec=spec,
        gpu_index=gpu_index,
        profiler_env=profiler_env,
        measure_block={
            "output_dir": str(resolved_output_dir),
            "duration_s": duration_s,
            "telemetry_hz": telemetry_hz,
            "clear_l2": clear_l2,
            "telemetry": telemetry,
        },
    )

    measure_result = response.get("measure") or {}
    if not measure_result.get("consumed"):
        raise MeasureError(
            f"{kernel_kind}:{resolved_backend} is not CUPTI-timed; "
            "measure supports CUPTI kernels only"
        )

    results = response.get("results") or []
    first = results[0] if results else {}
    runner_ok = bool(first.get("ok"))
    return {
        "kernel_kind": kernel_kind,
        "backend": resolved_backend,
        "gpu_index": gpu_index,
        "gpu_name": gpu_name,
        # The worker already reports the physical GPU (torch device-0 name); keep
        # it as provenance instead of dropping it, so the measurement metadata can
        # state precisely what hardware observed the capture.
        "observed_gpu_name": first.get("gpu_name") if runner_ok else None,
        "output_dir": str(resolved_output_dir),
        "time_ms": measure_result.get("time_ms"),
        "artifacts": measure_result.get("artifacts", []),
        "metrics": first.get("metrics") if runner_ok else None,
        "runner_error": None if runner_ok else first.get("error"),
    }


def _run_worker(
    *,
    kernel_kind: KernelKind,
    spec: dict[str, Any],
    gpu_index: int,
    profiler_env: ProfileEnv | ContainerProfileEnv,
    measure_block: dict[str, Any],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="vibesim-measure-") as tmp:
        input_path = Path(tmp) / "input.json"
        worker_output = Path(tmp) / "output.json"
        input_path.write_text(
            json.dumps(
                {
                    "kernel_kind": kernel_kind,
                    "specs": [spec],
                    "measure": measure_block,
                }
            ),
            encoding="utf-8",
        )
        if isinstance(profiler_env, ContainerProfileEnv):
            output_dir = Path(measure_block["output_dir"]).resolve()
            cmd, env = _container_worker_command(
                profiler_env,
                [gpu_index],
                Path(tmp),
                additional_volumes=((output_dir, output_dir),),
            )
        else:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
            env["PYTHONPATH"] = compose_pythonpath(
                profiler_env,
                env.get("PYTHONPATH"),
            )
            if profiler_env.additional_library_paths:
                env["LD_LIBRARY_PATH"] = compose_library_path(
                    profiler_env,
                    env.get("LD_LIBRARY_PATH"),
                )
            cmd = [
                str(profiler_env.python_executable),
                "-m",
                "profiling.exec.local_worker",
                "--worker-input",
                str(input_path),
                "--worker-output",
                str(worker_output),
            ]
        completed = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            error = completed.stderr.strip() or completed.stdout.strip() or "measure worker failed"
            raise MeasureError(error)
        return json.loads(worker_output.read_text(encoding="utf-8"))
