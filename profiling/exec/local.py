"""Local execution backend for L1 profiling."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

from profiling.db.kind import KernelKind
from profiling.db.registry import find_kernel_profiler_spec
from profiling.exec.env import compose_library_path, compose_pythonpath, resolve_profile_env
from profiling.exec.payload import chunk_result_from_payload, resolve_chunk_backend
from profiling.exec.pool import ChunkResult, GpuChunk, GpuPool


class LocalGpuPool(GpuPool):
    def __init__(self, gpus: list[int] | None = None):
        self.gpus = gpus

    def acquire_chunks(self, k: int, max_concurrent: int = 1) -> Iterator[GpuChunk]:
        gpus_per_chunk = k
        if gpus_per_chunk < 1:
            raise ValueError(f"k must be >= 1, got {gpus_per_chunk}")
        if max_concurrent < 1:
            raise ValueError(f"max_concurrent must be >= 1, got {max_concurrent}")
        available_gpus = self.gpus if self.gpus is not None else find_idle_gpus()
        if len(available_gpus) < gpus_per_chunk:
            raise RuntimeError(
                f"need {gpus_per_chunk} idle GPU(s), found {len(available_gpus)}"
            )
        # Yield non-overlapping chunks. run_profile_batch decides how to split
        # specs across them; this pool only owns local GPU slot selection.
        chunk_count = min(max_concurrent, len(available_gpus) // gpus_per_chunk)
        for chunk_index in range(chunk_count):
            start_gpu_index = chunk_index * gpus_per_chunk
            yield LocalGpuChunk(
                available_gpus[start_gpu_index : start_gpu_index + gpus_per_chunk]
            )


class LocalGpuChunk(GpuChunk):
    def __init__(self, gpus: list[int]):
        self.gpus = gpus

    def run(self, kernel_kind: KernelKind, specs: list[dict]) -> list[ChunkResult]:
        chunk_specs = specs
        if not chunk_specs:
            return []
        backend = resolve_chunk_backend(kernel_kind, chunk_specs)
        profiler_spec = find_kernel_profiler_spec(kernel_kind, backend)
        profiler_env = resolve_profile_env(profiler_spec.subprocess_env)
        profiler_env.validate()

        with tempfile.TemporaryDirectory(prefix="vibesim-profile-") as tmp:
            input_path = Path(tmp) / "input.json"
            output_path = Path(tmp) / "output.json"
            input_path.write_text(
                json.dumps(
                    {
                        "kernel_kind": kernel_kind,
                        "specs": chunk_specs,
                    }
                ),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in self.gpus)
            env["PYTHONPATH"] = compose_pythonpath(
                profiler_env,
                env.get("PYTHONPATH"),
            )
            if profiler_env.additional_library_paths:
                env["LD_LIBRARY_PATH"] = compose_library_path(
                    profiler_env,
                    env.get("LD_LIBRARY_PATH"),
                )

            # One worker process handles the whole chunk payload, so Python,
            # imports, CUDA context, and runner JIT setup are amortized per
            # chunk instead of paid once per spec.
            cmd = [
                str(profiler_env.python_executable),
                "-m",
                "profiling.exec.local_worker",
                "--worker-input",
                str(input_path),
                "--worker-output",
                str(output_path),
            ]
            completed = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                error = (
                    completed.stderr.strip()
                    or completed.stdout.strip()
                    or "local profile failed"
                )
                return [ChunkResult(metrics=None, error=error) for _ in chunk_specs]

            worker_response = json.loads(output_path.read_text(encoding="utf-8"))
            return [
                chunk_result_from_payload(result_payload)
                for result_payload in worker_response["results"]
            ]


def find_idle_gpus(memory_threshold_mb: int = 1000, util_threshold_pct: int = 10) -> list[int]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-x"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    root = ET.fromstring(result.stdout)
    visible_gpu_order = _cuda_visible_gpu_order(root, os.environ.get("CUDA_VISIBLE_DEVICES"))
    visible_gpu_set = set(visible_gpu_order) if visible_gpu_order is not None else None
    idle_by_index = {}
    for gpu_index, gpu in enumerate(root.findall("gpu")):
        if visible_gpu_set is not None and gpu_index not in visible_gpu_set:
            continue
        mem_used = gpu.findtext("fb_memory_usage/used", "999999 MiB")
        util = gpu.findtext("utilization/gpu_util", "100 %")
        mem_used_mb = int(mem_used.split()[0])
        util_pct = int(util.split()[0])
        if mem_used_mb < memory_threshold_mb and util_pct < util_threshold_pct:
            idle_by_index[gpu_index] = gpu_index
    if visible_gpu_order is None:
        return list(idle_by_index)
    return [gpu_index for gpu_index in visible_gpu_order if gpu_index in idle_by_index]


def _cuda_visible_gpu_order(root: ET.Element, raw_visible_devices: str | None) -> list[int] | None:
    if raw_visible_devices is None:
        return None
    visible_tokens = [
        token.strip()
        for token in raw_visible_devices.split(",")
        if token.strip()
    ]
    if not visible_tokens:
        return []

    # CUDA_VISIBLE_DEVICES is itself ordered. Preserve that order when choosing
    # chunks so a parent mask like "3,1" stays "3,1" in worker subprocesses.
    gpu_index_by_token: dict[str, int] = {}
    for gpu_index, gpu in enumerate(root.findall("gpu")):
        gpu_index_by_token[str(gpu_index)] = gpu_index
        gpu_uuid = gpu.findtext("uuid")
        if gpu_uuid:
            gpu_index_by_token[gpu_uuid.strip()] = gpu_index

    visible_gpu_order: list[int] = []
    for token in visible_tokens:
        gpu_index = gpu_index_by_token.get(token)
        if gpu_index is not None and gpu_index not in visible_gpu_order:
            visible_gpu_order.append(gpu_index)
    return visible_gpu_order
