"""Local execution backend for L1 profiling."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from pathlib import Path

from profiling.db.kind import KernelKind
from profiling.db.registry import find_kernel_profiler_spec
from profiling.exec.env import (
    ContainerProfileEnv,
    ProfileEnv,
    compose_library_path,
    compose_pythonpath,
    resolve_profile_env,
)
from profiling.exec.payload import chunk_result_from_payload, resolve_chunk_backend
from profiling.exec.pool import ChunkResult, GpuChunk, GpuPool
from profiling.instrument import TIMELINE_ENV, span
from profiling.profilers.energy import energy_enabled
from profiling.profilers.timer import CUPTI_BUDGET_ENV, CUPTI_TRACE_ENV, TIMER_COMPARE_ENV

CONTAINER_INDUCTOR_CACHE_ENV = "VIBESIM_CONTAINER_INDUCTOR_CACHE"

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONTAINER_SOURCE_MODE_ENV = "VIBESIM_PROFILE_SOURCE_MODE"
_CONTAINER_SOURCE_DIRS = ("profiling", "launcher", "gpu")


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
            raise RuntimeError(f"need {gpus_per_chunk} idle GPU(s), found {len(available_gpus)}")
        # Yield non-overlapping chunks. run_profile_batch decides how to split
        # specs across them; this pool only owns local GPU slot selection.
        chunk_count = min(max_concurrent, len(available_gpus) // gpus_per_chunk)
        for chunk_index in range(chunk_count):
            start_gpu_index = chunk_index * gpus_per_chunk
            yield LocalGpuChunk(available_gpus[start_gpu_index : start_gpu_index + gpus_per_chunk])


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
                        # Measurement policy travels in the payload, like the
                        # `measure` block, because the payload is the one thing
                        # every worker gets the same way on every path -- host
                        # subprocess or container. Forwarding it in the
                        # environment instead meant a container worker silently
                        # used a different policy whenever the forwarding list
                        # fell behind.
                        "energy": energy_enabled(),
                    }
                ),
                encoding="utf-8",
            )
            if isinstance(profiler_env, ContainerProfileEnv):
                cmd, env = _container_worker_command(profiler_env, self.gpus, Path(tmp))
            else:
                cmd, env = _host_worker_command(profiler_env, self.gpus, input_path, output_path)
            # The gap between this span and the worker.boot span inside it is
            # process/container start: fork, image, CUDA context.
            with span("worker.subprocess", kind=kernel_kind, specs=len(chunk_specs)):
                completed = subprocess.run(
                    cmd, env=env, capture_output=True, text=True, check=False
                )
            if completed.returncode != 0:
                error = _worker_failure_message(completed)
                return [ChunkResult(metrics=None, error=error) for _ in chunk_specs]

            worker_response = json.loads(output_path.read_text(encoding="utf-8"))
            return [
                chunk_result_from_payload(result_payload)
                for result_payload in worker_response["results"]
            ]


_WORKER_STREAM_TAIL_CHARS = 2000


def _worker_failure_message(completed: subprocess.CompletedProcess[str]) -> str:
    """Describe a worker that exited non-zero, without hiding half the evidence.

    The previous version was ``stderr.strip() or stdout.strip()``. Container
    workers print a wrapper banner to stderr on every single run, so stderr was
    always non-empty, the ``or`` always short-circuited, and stdout was never
    reported. A chunk of 1426 specs then failed with nothing but that banner as
    the cause.

    The exit status was dropped entirely, which erased the one distinction that
    matters first: a negative ``returncode`` is a signal, so the worker was
    killed (OOM, segfault) rather than raising -- and a killed worker leaves no
    traceback to look for.

    Both streams are tailed rather than headed: a traceback ends with the
    exception, and a crash log ends with the crash.
    """

    status = completed.returncode
    if status < 0:
        try:
            name = signal.Signals(-status).name
        except ValueError:
            name = f"signal {-status}"
        headline = f"worker killed by {name} (returncode {status})"
    else:
        headline = f"worker exited {status}"

    parts = [headline]
    for label, stream in (("stderr", completed.stderr), ("stdout", completed.stdout)):
        text = (stream or "").strip()
        if not text:
            parts.append(f"{label}: <empty>")
            continue
        if len(text) > _WORKER_STREAM_TAIL_CHARS:
            text = "...(truncated)... " + text[-_WORKER_STREAM_TAIL_CHARS:]
        parts.append(f"{label}: {text}")
    return " | ".join(parts)


def _host_worker_command(
    profiler_env: ProfileEnv,
    gpus: list[int],
    input_path: Path,
    output_path: Path,
) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu) for gpu in gpus)
    env["PYTHONPATH"] = compose_pythonpath(profiler_env, env.get("PYTHONPATH"))
    if profiler_env.additional_library_paths:
        env["LD_LIBRARY_PATH"] = compose_library_path(profiler_env, env.get("LD_LIBRARY_PATH"))
    return (
        [
            str(profiler_env.python_executable),
            "-m",
            "profiling.exec.local_worker",
            "--worker-input",
            str(input_path),
            "--worker-output",
            str(output_path),
        ],
        env,
    )


def _container_worker_command(
    profiler_env: ContainerProfileEnv,
    gpus: list[int],
    exchange_dir: Path,
    additional_volumes: tuple[tuple[Path, Path], ...] = (),
) -> tuple[list[str], dict[str, str]]:
    cache_dir = Path(
        os.environ.get(
            "VIBESIM_PROFILE_CACHE_DIR",
            Path.home() / ".cache" / "vibesim-profiler",
        )
    ).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Address the cards by UUID, never by index. See gpu_uuids_for_indices.
    gpu_request = f'"device={",".join(gpu_uuids_for_indices(gpus))}"'
    # A container does not inherit this process's environment, so measurement
    # policy has to be handed over explicitly; the host worker gets it for free
    # through os.environ.copy(). Anything here that changes what a number MEANS
    # belongs on this list, or a container row would silently be measured under
    # a different policy than the host rows it sits beside. The energy window is
    # not on the list because it rides the JSON payload, which both worker paths
    # read identically -- that is the shape the rest of these should move to.
    policy_args = [
        argument
        for name in (CUPTI_BUDGET_ENV, CUPTI_TRACE_ENV, TIMELINE_ENV, TIMER_COMPARE_ENV)
        if name in os.environ
        for argument in ("--env", f"{name}={os.environ[name]}")
    ]
    # Forwarding a path in the environment is only half the handover: the path
    # has to exist inside the container too. The trace file is the one policy
    # value that names a host path, and without this the container silently
    # writes nothing while the host workers beside it trace normally -- a
    # diagnostic that is blank for exactly the runs you most wanted to inspect.
    # Mounted at its own absolute path so one env value works on both sides.
    # Same handover as the trace: a path forwarded in the environment is only
    # half the job, the directory has to exist inside the container too.
    host_paths = [os.environ.get(CUPTI_TRACE_ENV), os.environ.get(TIMELINE_ENV)]
    mounted: list[str] = []
    trace_volume_args = []
    for host_path in host_paths:
        if not host_path:
            continue
        parent = str(Path(host_path).parent.resolve())
        if parent in mounted:
            continue
        mounted.append(parent)
        trace_volume_args += ["--volume", f"{parent}:{parent}"]
    # Set empty to leave inductor on its ephemeral /tmp default. Exists so the
    # persistent cache can be A/B'd against the old behaviour on one card: a
    # compile cache is not supposed to change a measured number, and the only
    # way to say that about this one is to measure both.
    inductor_cache = os.environ.get(
        CONTAINER_INDUCTOR_CACHE_ENV, "/cache/home/.cache/torchinductor"
    )
    inductor_args = (
        ["--env", f"TORCHINDUCTOR_CACHE_DIR={inductor_cache}"] if inductor_cache else []
    )
    source_volume_args = _container_source_volume_args()
    volume_args = [
        argument
        for host_path, container_path in additional_volumes
        for argument in ("--volume", f"{host_path.resolve()}:{container_path}")
    ]
    return (
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--ipc",
            "host",
            "--gpus",
            gpu_request,
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--env",
            "HOME=/cache/home",
            "--env",
            "USER=vibesim",
            "--env",
            "LOGNAME=vibesim",
            # Inductor defaults to /tmp/torchinductor_<user>, and /tmp inside a
            # --rm container dies with the container, so every submission
            # recompiles from nothing. /cache is the bind mount below, so this
            # points it at the same place the host workers already use.
            # Triton and FlashInfer need no equivalent: both derive their cache
            # from HOME, which is already inside /cache.
            *inductor_args,
            *policy_args,
            "--volume",
            f"{exchange_dir.resolve()}:/io",
            "--volume",
            f"{cache_dir}:/cache",
            *trace_volume_args,
            *source_volume_args,
            *volume_args,
            profiler_env.image,
            "--worker-input",
            "/io/input.json",
            "--worker-output",
            "/io/output.json",
        ],
        os.environ.copy(),
    )


def _container_source_volume_args() -> list[str]:
    """Select current-worktree code for development or the baked image snapshot.

    The image owns the dependency boundary, including the instrumented vLLM
    checkout, its virtual environment, and native extensions. Development mode
    overlays only ServingStudioSim's pure-Python worker code and GPU catalog read-only so
    newly registered profilers run without rebuilding that dependency image.
    Release measurements can request the fully frozen source snapshot with
    ``VIBESIM_PROFILE_SOURCE_MODE=image``.
    """

    source_mode = os.environ.get(_CONTAINER_SOURCE_MODE_ENV, "worktree")
    if source_mode == "image":
        return []
    if source_mode != "worktree":
        raise ValueError(
            f"{_CONTAINER_SOURCE_MODE_ENV} must be 'worktree' or 'image', got {source_mode!r}"
        )

    volume_args: list[str] = []
    for directory in _CONTAINER_SOURCE_DIRS:
        host_path = (_PROJECT_ROOT / directory).resolve()
        if not host_path.is_dir():
            raise FileNotFoundError(f"container source directory does not exist: {host_path}")
        volume_args.extend(("--volume", f"{host_path}:/opt/vibesim/{directory}:ro"))
    return volume_args


def _nvidia_smi_xml() -> ET.Element | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-q", "-x"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return ET.fromstring(result.stdout)


def gpu_uuids_for_indices(gpu_indices: Sequence[int]) -> list[str]:
    """Name the cards at ``gpu_indices`` by UUID, for a reader in another view.

    A GPU index only means something relative to the device list the process
    that read it could see, and a container runtime reads a different one: under
    a device cgroup (Slurm's ``ConstrainDevices``, say) the one allocated card
    enumerates here as index 0, while ``dockerd`` sits outside that cgroup and
    resolves ``--gpus device=0`` to the host's first card. Passing the index
    across that boundary silently profiles whatever GPU 0 happens to be —
    someone else's job, or every concurrent worker piling onto one card. A UUID
    names the same silicon in both views, so the container path uses it.

    Raises when a UUID cannot be established, rather than falling back to the
    index: the fallback is exactly the mis-targeting this exists to prevent.
    """

    root = _nvidia_smi_xml()
    if root is None:
        raise RuntimeError("nvidia-smi is required to address container GPUs by UUID")

    uuid_by_index: dict[int, str] = {}
    for gpu_index, gpu in enumerate(root.findall("gpu")):
        gpu_uuid = (gpu.findtext("uuid") or "").strip()
        if gpu_uuid:
            uuid_by_index[gpu_index] = gpu_uuid

    missing = [gpu_index for gpu_index in gpu_indices if gpu_index not in uuid_by_index]
    if missing:
        raise RuntimeError(
            f"nvidia-smi reported no UUID for GPU index(es) {missing}; "
            f"indices visible here are {sorted(uuid_by_index)}"
        )
    return [uuid_by_index[gpu_index] for gpu_index in gpu_indices]


def find_idle_gpus(memory_threshold_mb: int = 1000, util_threshold_pct: int = 10) -> list[int]:
    root = _nvidia_smi_xml()
    if root is None:
        return []

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
    visible_tokens = [token.strip() for token in raw_visible_devices.split(",") if token.strip()]
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
