"""Local Neuron allocation, isolated workers, and physical device provenance.

Reserve whole chips while a backend uses one or more logical LNC2 units.
This prevents two measurements from sharing HBM/compute on the same chip.
Single-core runners retain their visibility; stock serving backends assign
per-rank runtime cores inside their validated framework-visible span.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlparse

from profiling.db.registry import find_kernel_profiler_spec
from profiling.exec.env import ContainerProfileEnv, ProfileEnv, resolve_profile_env
from profiling.exec.local import _host_worker_command, _worker_failure_message
from profiling.exec.payload import chunk_result_from_payload, resolve_chunk_backend
from profiling.exec.pool import ChunkResult, GpuChunk, GpuPool
from profiling.gpu_policy import require_gpu
from profiling.instrument import TIMELINE_ENV

TRAINIUM2_LNC2 = "AWS Trainium2 LNC2"
# Reservations must be shared across checkouts with different workspace TMPDIRs.
NEURON_LOCK_DIR = Path("/tmp")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_NXDI_DATA_ENV = (
    "SERVINGSTUDIO_NXDI_MODEL_DIR",
    "SERVINGSTUDIO_NXDI_COMPILED_DIR",
    "SERVINGSTUDIO_NXDI_COMPILER_WORKDIR",
)
_CONTAINER_ABI_ENV = {"PATH", "LD_LIBRARY_PATH", "PYTHONHOME", "PYTHONPATH"}
_VLLM_DATA_ENV = ("SERVINGSTUDIO_VLLM_NEURON_MODEL_DIR",)
_CALLER_POLICY_ENV = (
    *_NXDI_DATA_ENV,
    *_VLLM_DATA_ENV,
    TIMELINE_ENV,
    "PJRT_DEVICE",
    "NEURON_CC_FLAGS",
    "XLA_FLAGS",
)
_IMMUTABLE_IMAGE = re.compile(r"(?:sha256:|[^\s]+@sha256:)[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class NeuronDevice:
    device_id: int
    core_ids: tuple[int, ...]
    lnc: int
    memory_bytes: int
    busy: bool
    instance_type: str

    @property
    def logical_core_memory_bytes(self) -> int:
        # neuron-ls reports logical IDs under the configured LNC mode. Its
        # nc_count/core_ids must not be divided by LNC a second time.
        return self.memory_bytes // len(self.core_ids)

    @property
    def gpu_name(self) -> str:
        if not self.instance_type.startswith("trn2.") or self.lnc != 2:
            raise RuntimeError(f"unsupported Neuron target: {self.instance_type}, LNC{self.lnc}")
        return TRAINIUM2_LNC2


def neuron_devices() -> list[NeuronDevice]:
    require_gpu("discovering Neuron devices")
    executable = shutil.which("neuron-ls") or "/opt/aws/neuron/bin/neuron-ls"
    result = subprocess.run(
        [executable, "--json-output"],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = json.loads(result.stdout)
    if not isinstance(rows, list):
        raise RuntimeError("neuron-ls did not return a device list")
    devices = []
    for row in rows:
        cores = tuple(int(core) for core in row["neuroncore_ids"])
        if not cores:
            continue
        devices.append(
            NeuronDevice(
                int(row["neuron_device"]),
                cores,
                int(row["logical_neuroncore_config"]),
                int(row["memory_size"]),
                bool(row.get("neuron_processes")),
                str(row["instance_type"]),
            )
        )
    return devices


def current_neuron_name() -> str:
    devices = neuron_devices()
    requested = os.environ.get("SERVINGSTUDIO_NEURON_DEVICE")
    if requested is not None:
        devices = [device for device in devices if device.device_id == int(requested)]
    names = {device.gpu_name for device in devices}
    if len(names) != 1:
        raise RuntimeError("cannot resolve a unique physical Neuron target")
    return names.pop()


def _reservation_payload(device: NeuronDevice, logical_cores: int = 1) -> dict:
    _reservation_environment(device, logical_cores)
    payload = {
        "device_family": "neuron",
        "neuron_device": device.device_id,
        "visible_core": device.core_ids[0],
        "lnc": device.lnc,
        "core_ids": list(device.core_ids),
        "instance_type": device.instance_type,
    }
    if logical_cores != 1:
        payload["logical_cores"] = logical_cores
    return payload


def observe_neuron_reservation(execution: dict) -> NeuronDevice:
    """Compare worker-owned hardware observations with the controller reservation."""
    if not isinstance(execution, dict) or execution.get("device_family") != "neuron":
        raise ValueError("worker reservation must name the Neuron device family")
    for name in ("neuron_device", "visible_core", "lnc"):
        if type(execution.get(name)) is not int or execution[name] < 0:
            raise ValueError(f"invalid Neuron reservation field: {name}")
    cores = execution.get("core_ids")
    if (
        not isinstance(cores, list)
        or not cores
        or any(type(core) is not int or core < 0 for core in cores)
        or len(set(cores)) != len(cores)
        or execution["visible_core"] != cores[0]
    ):
        raise ValueError("invalid Neuron reservation core IDs")
    matches = [
        device for device in neuron_devices() if device.device_id == execution["neuron_device"]
    ]
    if len(matches) != 1:
        raise RuntimeError("worker cannot observe its reserved physical Neuron device")
    device = matches[0]
    device.gpu_name  # Validate generation/LNC from actual observation.
    logical_cores = execution.get("logical_cores", 1)
    if _reservation_payload(device, logical_cores) != execution:
        raise RuntimeError("observed Neuron device/core identity differs from the reservation")
    expected = _reservation_environment(device, logical_cores)
    for name in (
        "SERVINGSTUDIO_NEURON_DEVICE",
        "NEURON_RT_VISIBLE_CORES",
        "NEURON_VISIBLE_DEVICES",
        "NEURON_LOGICAL_NC_CONFIG",
    ):
        if os.environ.get(name) != expected.get(name):
            raise RuntimeError(f"worker {name} differs from its reserved Neuron device/core")
    return device


def _reservation_environment(device: NeuronDevice, logical_cores: int = 1) -> dict[str, str]:
    device.gpu_name
    if type(logical_cores) is not int or not 1 <= logical_cores <= len(device.core_ids):
        raise ValueError("requested Neuron logical cores exceed reserved chip")
    selected = device.core_ids[:logical_cores]
    if selected != tuple(range(selected[0], selected[0] + logical_cores)):
        raise ValueError("Neuron logical-core span must be contiguous")
    visibility = str(selected[0]) if logical_cores == 1 else f"{selected[0]}-{selected[-1]}"
    return {
        "SERVINGSTUDIO_NEURON_DEVICE": str(device.device_id),
        # vLLM assigns RT visibility separately inside each spawned TP rank.
        # Its public multiprocess executor rejects pre-set RT visibility.
        ("NEURON_RT_VISIBLE_CORES" if logical_cores == 1 else "NEURON_VISIBLE_DEVICES"): visibility,
        "NEURON_LOGICAL_NC_CONFIG": str(device.lnc),
        "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
        "CUDA_VISIBLE_DEVICES": "",
    }


def _docker_cli_environment() -> dict[str, str]:
    # The declared local socket owns transport; inherited context/TLS overrides
    # cannot redirect this subprocess. Never mutate the controller environment.
    return {name: value for name, value in os.environ.items() if not name.startswith("DOCKER_")}


def _neuron_docker_prefix(profile_env: ContainerProfileEnv) -> list[str]:
    if profile_env.docker_host is None or profile_env.python_executable is None:
        raise ValueError("Neuron containers require explicit docker_host and container Python")
    if not _IMMUTABLE_IMAGE.fullmatch(profile_env.image):
        raise ValueError("Neuron containers require an immutable local image ID or digest")
    return [*profile_env.docker_command, "--host", profile_env.docker_host]


def _inspect_neuron_image(profile_env: ContainerProfileEnv) -> str:
    completed = subprocess.run(
        [*_neuron_docker_prefix(profile_env), "image", "inspect", profile_env.image],
        env=_docker_cli_environment(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if completed.returncode:
        raise RuntimeError(
            f"local Neuron image inspection failed: {_worker_failure_message(completed)}"
        )
    records = json.loads(completed.stdout)
    if not isinstance(records, list) or len(records) != 1:
        raise RuntimeError("Docker inspection did not resolve exactly one local image")
    record = records[0]
    image_id = record.get("Id", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise RuntimeError("Docker inspection returned an invalid image ID")
    if profile_env.image != image_id and profile_env.image not in (record.get("RepoDigests") or []):
        raise RuntimeError("inspected local image differs from its declared immutable identity")
    return image_id


def _host_driver_version() -> str | None:
    """Host installed driver-package provenance, distinct from image runtime libraries."""
    if shutil.which("rpm"):
        command = ["rpm", "-q", "--queryformat", "%{VERSION}-%{RELEASE}", "aws-neuronx-dkms"]
    elif shutil.which("dpkg-query"):
        command = ["dpkg-query", "--show", "--showformat=${Version}", "aws-neuronx-dkms"]
    else:
        return None
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    return result.stdout.strip() if result.returncode == 0 else None


def _absolute_directory(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute directory")
    if not path.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {path}")
    return path.resolve()


def _neuron_container_worker_command(
    profile_env: ContainerProfileEnv,
    device: NeuronDevice,
    exchange_dir: Path,
    *,
    image_id: str,
    worker_env: Mapping[str, str] | None = None,
    kernel_kind: str | None = None,
    logical_cores: int = 1,
) -> tuple[list[str], dict[str, str]]:
    """Bind request source and declared data around an image-owned Neuron ABI."""
    prefix = _neuron_docker_prefix(profile_env)
    if os.getuid() == 0:
        raise ValueError("Neuron containers require a nonroot invoking user")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("Neuron launch requires the controller-inspected image ID")
    if os.environ.get("VIBESIM_PROFILE_SOURCE_MODE", "worktree") != "worktree":
        raise ValueError("Neuron containers currently require worktree source mode")
    policy = {name: os.environ[name] for name in _CALLER_POLICY_ENV if name in os.environ}
    policy.update(worker_env or {})
    policy = {name: value for name, value in policy.items() if name not in _CONTAINER_ABI_ENV}
    if (
        kernel_kind in ("neuron_llama_forward", "neuron_llama_region")
        and not policy.get("SERVINGSTUDIO_VLLM_NEURON_MODEL_DIR")
        and any(not policy.get(name) for name in _NXDI_DATA_ENV)
    ):
        raise ValueError(
            "whole-forward containers require explicit model, compiled "
            "and compiler-work directories"
        )
    readonly = [_PROJECT_ROOT / name for name in ("profiling", "gpu", "tools")]
    for path in readonly:
        if not path.is_dir():
            raise FileNotFoundError(f"request source directory does not exist: {path}")
    for name in (*_NXDI_DATA_ENV, *_VLLM_DATA_ENV):
        if policy.get(name):
            path = _absolute_directory(policy[name], name)
            readonly.append(path)
            policy[name] = str(path)
    cache = Path(
        os.environ.get(
            "SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR",
            Path(tempfile.gettempdir()) / "servingstudio-neuron-cache",
        )
    )
    if not cache.is_absolute():
        raise ValueError("SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR must be absolute")
    cache = cache.resolve()
    writable = [exchange_dir.resolve(), cache]
    if policy.get(TIMELINE_ENV):
        path = Path(policy[TIMELINE_ENV])
        if not path.is_absolute():
            raise ValueError(f"{TIMELINE_ENV} must be absolute in a Neuron container")
        writable.append(path.parent.resolve())
    for rw in writable:
        if any(rw.is_relative_to(ro) or ro.is_relative_to(rw) for ro in readonly):
            raise ValueError("Neuron writable cache/output mounts overlap read-only source or data")
    socket = Path(urlparse(profile_env.docker_host).path)
    if any(socket.is_relative_to(path) for path in [*readonly, *writable]):
        raise ValueError("Neuron worker mounts must not expose the Docker socket")
    for path in writable:
        path.mkdir(parents=True, exist_ok=True)
    for directory in ("home", "tmp", "artifacts"):
        (cache / directory).mkdir(exist_ok=True)
    policy.update(_reservation_environment(device, logical_cores))
    policy.update(
        {
            "PYTHONPATH": str(_PROJECT_ROOT),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": str(cache / "home"),
            "TMPDIR": str(cache / "tmp"),
            "TEMP": str(cache / "tmp"),
            "TMP": str(cache / "tmp"),
            "NKI_ARTIFACTS_DIR": str(cache / "artifacts"),
            "SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR": str(cache),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    if kernel_kind == "neuron_llama_forward" or profile_env.name == "vllm_neuron_env":
        # Stock rank-local kernels use the same lite/NKI cache ABI as serving.
        # Keep its writable paths in the executor, outside production runners.
        policy.update(
            {
                "VLLM_CACHE_ROOT": str(cache / "cache"),
                "XDG_CACHE_HOME": str(cache / "cache"),
                "NKI_COMPILE_CACHE_URL": str(cache / "nki-cache"),
            }
        )
    command = [
        *prefix,
        "run",
        "--rm",
        "--pull=never",
        "--network=none",
        "--no-healthcheck",
        "--ulimit=core=0",
        "--shm-size=2g",
        "--security-opt=no-new-privileges",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        f"--device=/dev/neuron{device.device_id}:/dev/neuron{device.device_id}",
        f"--entrypoint={profile_env.python_executable}",
    ]
    for name, value in sorted(policy.items()):
        command.extend(("--env", f"{name}={value}"))
    for path in dict.fromkeys(readonly):
        command.extend(("--volume", f"{path}:{path}:ro"))
    command.extend(("--volume", f"{exchange_dir.resolve()}:/io"))
    for path in dict.fromkeys(writable[1:]):
        command.extend(("--volume", f"{path}:{path}"))
    executable = shutil.which("neuron-ls") or "/opt/aws/neuron/bin/neuron-ls"
    if not Path(executable).is_file():
        raise FileNotFoundError("Neuron containers require a readable host neuron-ls identity tool")
    command.extend(("--volume", f"{Path(executable).resolve()}:/opt/aws/neuron/bin/neuron-ls:ro"))
    command.extend(
        (
            "--workdir=/io",
            image_id,
            "-m",
            "profiling.exec.local_worker",
            "--worker-input",
            "/io/input.json",
            "--worker-output",
            "/io/output.json",
        )
    )
    return command, _docker_cli_environment()


class LocalNeuronPool(GpuPool):
    def __init__(self, devices: list[int] | None = None):
        self.devices = devices

    def idle_devices(self) -> list[NeuronDevice]:
        candidates = neuron_devices()
        # A caller's visibility restriction must be respected. It is never
        # widened to make more cores appear available to this measurement.
        if any(
            name in os.environ for name in ("NEURON_RT_VISIBLE_CORES", "NEURON_VISIBLE_DEVICES")
        ):
            raise RuntimeError(
                "unset NEURON_RT_VISIBLE_CORES and NEURON_VISIBLE_DEVICES before profiling; "
                "the Neuron pool owns allocation"
            )
        return [
            device
            for device in candidates
            if not device.busy and (self.devices is None or device.device_id in self.devices)
        ]

    def acquire_chunks(self, k: int, max_concurrent: int = 1):
        if k != 1:
            raise ValueError("the initial Neuron backend supports one chip reservation per chunk")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        acquired = 0
        for device in self.idle_devices():
            device.gpu_name  # Validate generation and LNC before acquiring.
            lock_path = NEURON_LOCK_DIR / f"servingstudio-neuron-{device.device_id}.lock"
            lock = lock_path.open("a")
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            yield LocalNeuronChunk(device, lock)
            acquired += 1
            if acquired >= max_concurrent:
                break
        if acquired == 0:
            raise RuntimeError("no idle supported Neuron chip available")


class LocalNeuronChunk(GpuChunk):
    def __init__(self, device: NeuronDevice, lock):
        self.device = device
        self.lock = lock

    def run(self, kernel_kind: str, specs: list[dict]) -> list[ChunkResult]:
        try:
            return self._run(kernel_kind, specs)
        finally:
            self.lock.close()

    def _run(self, kernel_kind: str, specs: list[dict]) -> list[ChunkResult]:
        require_gpu("executing a Neuron profiling chunk")
        from profiling.profilers.energy import energy_enabled

        if energy_enabled():
            raise RuntimeError("Neuron energy measurement is not implemented")
        if not specs:
            return []
        backend = resolve_chunk_backend(kernel_kind, specs)
        spec = find_kernel_profiler_spec(kernel_kind, backend)
        if spec.supports.device_family != "neuron":
            raise ValueError(f"{kernel_kind}/{backend} is not a Neuron backend")
        profile_env = resolve_profile_env(spec.subprocess_env)
        if not isinstance(profile_env, (ProfileEnv, ContainerProfileEnv)):
            raise ValueError("Neuron profiling requires a declared host or container environment")
        profile_env.validate()
        container = isinstance(profile_env, ContainerProfileEnv)
        image_id = _inspect_neuron_image(profile_env) if container else None
        host_driver = _host_driver_version() if container else None
        with tempfile.TemporaryDirectory(prefix="servingstudio-neuron-") as scratch:
            input_path, output_path = (
                Path(scratch) / "input.json",
                Path(scratch) / "output.json",
            )
            input_path.write_text(
                json.dumps(
                    {
                        "kernel_kind": kernel_kind,
                        "specs": specs,
                        "energy": False,
                        "execution": _reservation_payload(self.device, spec.neuron_logical_cores),
                    }
                )
            )
            if container:
                cmd, env = _neuron_container_worker_command(
                    profile_env,
                    self.device,
                    Path(scratch),
                    image_id=image_id,
                    worker_env=dict(spec.worker_env),
                    kernel_kind=kernel_kind,
                    logical_cores=spec.neuron_logical_cores,
                )
            else:
                cmd, env = _host_worker_command(
                    profile_env,
                    [],
                    input_path,
                    output_path,
                    worker_env=dict(spec.worker_env),
                )
                env.update(_reservation_environment(self.device, spec.neuron_logical_cores))
            completed = subprocess.run(cmd, cwd=scratch, env=env, capture_output=True, text=True)
            if completed.returncode:
                return [ChunkResult(None, error=_worker_failure_message(completed)) for _ in specs]
            response = json.loads(output_path.read_text())
            results = response["results"]
            if len(results) != len(specs):
                raise RuntimeError("Neuron worker returned a different number of results")
            parsed = []
            for row in results:
                result = chunk_result_from_payload(row)
                if result.metrics is not None:
                    if result.observed_gpu_name != self.device.gpu_name:
                        raise RuntimeError(
                            "Neuron worker physical observation differs from its reservation"
                        )
                    if container:
                        annotation = (
                            f"container_image_id={image_id}; "
                            f"host_driver_package={host_driver or 'unavailable'}"
                        )
                        version = result.backend_version
                        result = replace(
                            result,
                            backend_version=f"{version}; {annotation}" if version else annotation,
                        )
                parsed.append(result)
            return parsed

    def __del__(self):
        self.lock.close()
