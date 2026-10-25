"""Neuron chip reservation and container workers for L1 profiling.

A Neuron backend (``BackendSupport.device_family == "neuron"``) runs in the
vendor image its profile env names (``profiling/container/images.py``,
``VENDOR_IMAGES``), never on a host interpreter. The pool reserves a whole chip
even when a backend uses fewer logical cores, so two measurements never share
its HBM or compute. The worker compares its own ``neuron-ls`` observation with
the reservation before it loads a runner (``observe_neuron_reservation``).

The Docker transport is host configuration: ``SERVINGSTUDIO_NEURON_DOCKER_HOST``
(a local ``unix://`` socket, default ``unix:///var/run/docker.sock``) and
``SERVINGSTUDIO_NEURON_DOCKER_COMMAND`` (the CLI argv, default ``docker``; e.g.
``sudo -n docker`` on a host whose socket only root can open).

The container sees every ``/dev/neuron*`` node and shares the host IPC and
network namespaces, as the Neuron runtime's collectives bootstrap needs; the
reservation is enforced by core visibility (``NEURON_RT_VISIBLE_CORES``) and the
host-wide chip lock, not by hiding device nodes. ``profiling/README.md`` (the
Neuron section) lists every variable.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.parse import urlparse

from profiling.container.images import vendor_image_for_env
from profiling.db.registry import find_kernel_profiler_spec
from profiling.exec.env import container_image, profile_runtime
from profiling.exec.local import _worker_failure_message
from profiling.exec.payload import chunk_result_from_payload, resolve_chunk_backend
from profiling.exec.pool import ChunkResult, GpuChunk, GpuPool
from profiling.gpu_policy import require_gpu
from profiling.instrument import TIMELINE_ENV

# The DB key of one Trainium2 LNC2 logical core: the catalog alias of
# `Trainium2-LNC2` (gpu/spec.json), as every recorded Neuron row names it.
TRAINIUM2_LNC2 = "AWS Trainium2 LNC2"
# A chip lock must be seen by every checkout on the host, whatever its TMPDIR.
NEURON_LOCK_DIR = Path("/tmp")
DOCKER_HOST_ENV = "SERVINGSTUDIO_NEURON_DOCKER_HOST"
DOCKER_COMMAND_ENV = "SERVINGSTUDIO_NEURON_DOCKER_COMMAND"
DEFAULT_DOCKER_HOST = "unix:///var/run/docker.sock"
# Writable compile/runtime cache, mounted at its own host path; default
# `$TMPDIR/servingstudio-neuron-cache`.
CACHE_DIR_ENV = "SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR"
# Colon-separated host directories of op source a runner imports (e.g. an
# out-of-tree framework package), mounted read-only at their own paths and put
# on the worker's PYTHONPATH after this checkout.
EXTRA_SOURCE_DIRS_ENV = "SERVINGSTUDIO_NEURON_EXTRA_SOURCE_DIRS"
# Colon-separated host data directories (e.g. checkpoints), mounted read-only at
# their own paths and passed to the worker under the same variable.
DATA_DIRS_ENV = "SERVINGSTUDIO_NEURON_DATA_DIRS"
# Chips this host's profiling may reserve, as device IDs or ranges ("8-15" or
# "8,9,12-15"). Unset: every chip. A host shared with other Neuron jobs names
# its own chips here; the busy check and chip lock still apply inside the set.
DEVICES_ENV = "SERVINGSTUDIO_NEURON_DEVICES"
# Measurement policy a worker inherits from the caller.
_CALLER_POLICY_ENV = (TIMELINE_ENV, "PJRT_DEVICE", "NEURON_CC_FLAGS", "XLA_FLAGS")
# Runtime defaults every NEFF launch in the image needs; a row's worker_env may
# override them. The runtime queue is otherwise too short for back-to-back
# launches ("Execution Queue Full").
_RUNTIME_DEFAULT_ENV = {"NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS": "256"}
# The image owns its interpreter and native libraries.
_IMAGE_ABI_ENV = {"PATH", "LD_LIBRARY_PATH", "PYTHONHOME", "PYTHONPATH"}
# Worker source, overlaid read-only at its host path.
_SOURCE_DIRS = ("profiling", "gpu")
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_IMMUTABLE_IMAGE = re.compile(r"(?:sha256:|[^\s]+@sha256:)[0-9a-f]{64}\Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class NeuronDevice:
    device_id: int
    core_ids: tuple[int, ...]
    lnc: int
    memory_bytes: int
    busy: bool
    instance_type: str

    @property
    def gpu_name(self) -> str:
        if not self.instance_type.startswith("trn2.") or self.lnc != 2:
            raise RuntimeError(f"unsupported Neuron target: {self.instance_type}, LNC{self.lnc}")
        return TRAINIUM2_LNC2


def _neuron_ls() -> str:
    return shutil.which("neuron-ls") or "/opt/aws/neuron/bin/neuron-ls"


def neuron_devices() -> list[NeuronDevice]:
    require_gpu("discovering Neuron devices")
    result = subprocess.run(
        [_neuron_ls(), "--json-output"], check=True, capture_output=True, text=True
    )
    rows = json.loads(result.stdout)
    if not isinstance(rows, list):
        raise RuntimeError("neuron-ls did not return a device list")
    return [
        NeuronDevice(
            int(row["neuron_device"]),
            tuple(int(core) for core in row["neuroncore_ids"]),
            # neuron-ls reports logical core IDs under the configured LNC mode.
            int(row["logical_neuroncore_config"]),
            int(row["memory_size"]),
            bool(row.get("neuron_processes")),
            str(row["instance_type"]),
        )
        for row in rows
        if row["neuroncore_ids"]
    ]


def current_neuron_name() -> str:
    """The DB key of this host's (or the reserved, inside a worker) Neuron device."""
    devices = neuron_devices()
    requested = os.environ.get("SERVINGSTUDIO_NEURON_DEVICE")
    if requested is not None:
        devices = [device for device in devices if device.device_id == int(requested)]
    names = {device.gpu_name for device in devices}
    if len(names) != 1:
        raise RuntimeError("cannot resolve a unique physical Neuron target")
    return names.pop()


def _reservation_environment(device: NeuronDevice, logical_cores: int = 1) -> dict[str, str]:
    device.gpu_name  # Validate generation and LNC.
    if type(logical_cores) is not int or not 1 <= logical_cores <= len(device.core_ids):
        raise ValueError("requested Neuron logical cores exceed the reserved chip")
    selected = device.core_ids[:logical_cores]
    if selected != tuple(range(selected[0], selected[0] + logical_cores)):
        raise ValueError("Neuron logical-core span must be contiguous")
    visibility = str(selected[0]) if logical_cores == 1 else f"{selected[0]}-{selected[-1]}"
    return {
        "SERVINGSTUDIO_NEURON_DEVICE": str(device.device_id),
        # A multi-rank backend (stock vLLM) assigns runtime visibility inside
        # each rank and rejects a preset NEURON_RT_VISIBLE_CORES.
        ("NEURON_RT_VISIBLE_CORES" if logical_cores == 1 else "NEURON_VISIBLE_DEVICES"): visibility,
        "NEURON_LOGICAL_NC_CONFIG": str(device.lnc),
        "NEURON_PLATFORM_TARGET_OVERRIDE": "trn2",
        "CUDA_VISIBLE_DEVICES": "",
    }


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
    """Compare the worker's own device observation with the controller's reservation."""
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


@dataclass(frozen=True)
class NeuronContainer:
    """How a Neuron worker chunk starts: the Docker transport and the image."""

    docker_command: tuple[str, ...]
    docker_host: str
    image: str
    python: str

    def __post_init__(self) -> None:
        if not self.docker_command or not all(self.docker_command):
            raise ValueError(f"{DOCKER_COMMAND_ENV} must name a nonempty argv")
        address = urlparse(self.docker_host)
        if (
            not self.docker_host.startswith("unix://")
            or address.netloc
            or not Path(address.path).is_absolute()
            or address.query
            or address.fragment
        ):
            raise ValueError(f"{DOCKER_HOST_ENV} must be a local absolute unix:// socket URI")
        if not _IMMUTABLE_IMAGE.fullmatch(self.image):
            raise ValueError(f"Neuron image {self.image!r} is not an immutable digest")
        if not Path(self.python).is_absolute():
            raise ValueError("the image's Python must be an absolute path inside the image")

    @property
    def socket(self) -> Path:
        return Path(urlparse(self.docker_host).path)

    @property
    def prefix(self) -> list[str]:
        return [*self.docker_command, "--host", self.docker_host]

    def validate(self) -> None:
        if shutil.which(self.docker_command[0]) is None:
            raise FileNotFoundError(f"Docker CLI is unavailable: {self.docker_command[0]}")
        if not self.socket.is_socket():
            raise FileNotFoundError(f"Neuron Docker socket is unavailable: {self.docker_host}")


def neuron_container(env_name: str | None) -> NeuronContainer:
    """The container a Neuron backend declaring ``subprocess_env=env_name`` runs in."""
    if profile_runtime() != "container":
        raise ValueError("Neuron backends run only in their vendor image (runtime 'container')")
    vendor = vendor_image_for_env(env_name or "default_env")
    if vendor is None:
        raise ValueError(f"profile env {env_name!r} names no Neuron vendor image")
    host = os.environ.get(DOCKER_HOST_ENV) or DEFAULT_DOCKER_HOST
    return NeuronContainer(
        tuple(shlex.split(os.environ.get(DOCKER_COMMAND_ENV, "docker"))),
        host,
        container_image(env_name),
        vendor.python,
    )


def docker_cli_environment() -> dict[str, str]:
    # The declared socket owns the transport: inherited DOCKER_* context or TLS
    # settings must not redirect this subprocess. The caller's env is untouched.
    return {name: value for name, value in os.environ.items() if not name.startswith("DOCKER_")}


def _inspect_neuron_image(container: NeuronContainer) -> str:
    """The local image ID the declared digest resolves to; never pulled."""
    completed = subprocess.run(
        [*container.prefix, "image", "inspect", container.image],
        env=docker_cli_environment(),
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
    if not _IMAGE_ID.fullmatch(image_id):
        raise RuntimeError("Docker inspection returned an invalid image ID")
    if container.image != image_id and container.image not in (record.get("RepoDigests") or []):
        raise RuntimeError("inspected local image differs from its declared immutable identity")
    return image_id


def _host_driver_version() -> str | None:
    """The host's installed Neuron driver package, distinct from the image's runtime."""
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


def _directory_list(name: str) -> list[Path]:
    """The absolute directories a colon-separated host variable names."""
    raw = os.environ.get(name, "")
    return [_absolute_directory(entry, name) for entry in raw.split(":") if entry]


def _neuron_device_nodes() -> list[Path]:
    """Every host Neuron device node; the runtime's collectives need them all."""
    return sorted(
        (path for path in Path("/dev").glob("neuron*") if path.name[6:].isdigit()),
        key=lambda path: int(path.name[6:]),
    )


def _free_local_port() -> int:
    """A loopback TCP port free now, for one worker's collectives root."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _neuron_container_worker_command(
    container: NeuronContainer,
    device: NeuronDevice,
    exchange_dir: Path,
    *,
    image_id: str,
    worker_env: Mapping[str, str] | None = None,
    logical_cores: int = 1,
) -> tuple[list[str], dict[str, str]]:
    """Run the JSON worker in the image as the invoking user, on one reserved chip."""
    if os.getuid() == 0:
        raise ValueError("Neuron containers require a nonroot invoking user")
    if not _IMAGE_ID.fullmatch(image_id):
        raise ValueError("Neuron launch requires the controller-inspected image ID")
    policy = {name: os.environ[name] for name in _CALLER_POLICY_ENV if name in os.environ}
    policy.update(_RUNTIME_DEFAULT_ENV)
    policy.update(worker_env or {})
    policy = {name: value for name, value in policy.items() if name not in _IMAGE_ABI_ENV}
    readonly = [_PROJECT_ROOT / name for name in _SOURCE_DIRS]
    for path in readonly:
        if not path.is_dir():
            raise FileNotFoundError(f"worker source directory does not exist: {path}")
    extra_sources = _directory_list(EXTRA_SOURCE_DIRS_ENV)
    data_dirs = _directory_list(DATA_DIRS_ENV)
    readonly.extend(extra_sources)
    readonly.extend(data_dirs)
    if data_dirs:
        policy[DATA_DIRS_ENV] = ":".join(str(path) for path in data_dirs)
    cache = Path(
        os.environ.get(CACHE_DIR_ENV, Path(tempfile.gettempdir()) / "servingstudio-neuron-cache")
    )
    if not cache.is_absolute():
        raise ValueError(f"{CACHE_DIR_ENV} must be absolute")
    cache = cache.resolve()
    writable = [exchange_dir.resolve(), cache]
    if policy.get(TIMELINE_ENV):
        timeline = Path(policy[TIMELINE_ENV])
        if not timeline.is_absolute():
            raise ValueError(f"{TIMELINE_ENV} must be absolute for a Neuron container")
        writable.append(timeline.parent.resolve())
    for rw in writable:
        if any(rw.is_relative_to(ro) or ro.is_relative_to(rw) for ro in readonly):
            raise ValueError("Neuron writable mounts overlap read-only source or data")
    if any(container.socket.is_relative_to(path) for path in [*readonly, *writable]):
        raise ValueError("Neuron worker mounts must not expose the Docker socket")
    nodes = _neuron_device_nodes()
    if Path(f"/dev/neuron{device.device_id}") not in nodes:
        raise FileNotFoundError(f"the reserved chip's /dev/neuron{device.device_id} is missing")
    for path in writable:
        path.mkdir(parents=True, exist_ok=True)
    for directory in ("home", "tmp", "artifacts"):
        (cache / directory).mkdir(exist_ok=True)
    policy.update(_reservation_environment(device, logical_cores))
    policy.update(
        {
            # Every NEFF load bootstraps collectives, even on one core; without
            # a root the load fails NRT_INVALID. One port per worker: the
            # container shares the host network with other jobs.
            "NEURON_RT_ROOT_COMM_ID": f"127.0.0.1:{_free_local_port()}",
            "PYTHONPATH": ":".join(str(path) for path in [_PROJECT_ROOT, *extra_sources]),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HOME": str(cache / "home"),
            "TMPDIR": str(cache / "tmp"),
            "TEMP": str(cache / "tmp"),
            "TMP": str(cache / "tmp"),
            # neuronx-cc writes fixed file names into the NKI artifacts
            # directory and needs it clean per compilation, so concurrent
            # workers sharing the cache each get their own.
            "NKI_ARTIFACTS_DIR": str(
                Path(tempfile.mkdtemp(prefix=f"chip{device.device_id}-", dir=cache / "artifacts"))
            ),
            CACHE_DIR_ENV: str(cache),
            "NEURON_COMPILE_CACHE_URL": str(cache / "compile"),
            "XDG_CACHE_HOME": str(cache / "cache"),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    neuron_ls = Path(_neuron_ls())
    if not neuron_ls.is_file():
        raise FileNotFoundError("Neuron containers need the host neuron-ls to observe the chip")
    command = [
        *container.prefix,
        "run",
        "--rm",
        "--pull=never",
        "--ipc=host",
        "--network=host",
        "--no-healthcheck",
        "--ulimit=core=0",
        "--security-opt=no-new-privileges",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        *(f"--device={node}:{node}" for node in nodes),
        # The image's ENTRYPOINT wraps every command in a vLLM launcher.
        "--entrypoint",
        "",
    ]
    for name, value in sorted(policy.items()):
        command.extend(("--env", f"{name}={value}"))
    for path in dict.fromkeys(readonly):
        command.extend(("--volume", f"{path}:{path}:ro"))
    command.extend(("--volume", f"{exchange_dir.resolve()}:/io"))
    for path in dict.fromkeys(writable[1:]):
        command.extend(("--volume", f"{path}:{path}"))
    command.extend(("--volume", f"{neuron_ls.resolve()}:/opt/aws/neuron/bin/neuron-ls:ro"))
    command.extend(
        (
            "--workdir=/io",
            image_id,
            container.python,
            "-m",
            "profiling.exec.local_worker",
            "--worker-input",
            "/io/input.json",
            "--worker-output",
            "/io/output.json",
        )
    )
    return command, docker_cli_environment()


def allowed_neuron_devices() -> list[int] | None:
    """The chip IDs ``SERVINGSTUDIO_NEURON_DEVICES`` allows, or ``None`` (all)."""
    raw = os.environ.get(DEVICES_ENV, "").strip()
    if not raw:
        return None
    devices: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        if not lo.isdigit() or (sep and not hi.isdigit()):
            raise ValueError(f"{DEVICES_ENV} entry {part!r} is not a chip ID or ID range")
        first, last = int(lo), int(hi) if sep else int(lo)
        if last < first:
            raise ValueError(f"{DEVICES_ENV} range {part!r} is empty")
        devices.extend(range(first, last + 1))
    if not devices:
        raise ValueError(f"{DEVICES_ENV} names no chip")
    return sorted(set(devices))


class LocalNeuronPool(GpuPool):
    """Idle local Trainium chips, one whole chip per chunk."""

    def __init__(self, devices: list[int] | None = None):
        self.devices = devices

    def idle_devices(self) -> list[NeuronDevice]:
        candidates = neuron_devices()
        # The pool owns core visibility; an inherited restriction is refused
        # rather than silently widened or narrowed.
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

    def acquire_chunks(self, k: int, max_concurrent: int = 1) -> Iterator[GpuChunk]:
        if k != 1:
            raise ValueError("a Neuron chunk reserves exactly one chip")
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        acquired = 0
        for device in self.idle_devices():
            device.gpu_name  # Validate generation and LNC before locking.
            lock = (NEURON_LOCK_DIR / f"servingstudio-neuron-{device.device_id}.lock").open("a")
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
    """One reserved chip; its lock is held until the chunk's results are parsed."""

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
        container = neuron_container(spec.subprocess_env)
        container.validate()
        image_id = _inspect_neuron_image(container)
        host_driver = _host_driver_version()
        with tempfile.TemporaryDirectory(prefix="servingstudio-neuron-") as scratch:
            output_path = Path(scratch) / "output.json"
            (Path(scratch) / "input.json").write_text(
                json.dumps(
                    {
                        "kernel_kind": kernel_kind,
                        "specs": specs,
                        "energy": False,
                        "execution": _reservation_payload(self.device, spec.neuron_logical_cores),
                    }
                )
            )
            cmd, env = _neuron_container_worker_command(
                container,
                self.device,
                Path(scratch),
                image_id=image_id,
                worker_env=dict(spec.worker_env),
                logical_cores=spec.neuron_logical_cores,
            )
            completed = subprocess.run(cmd, cwd=scratch, env=env, capture_output=True, text=True)
            if completed.returncode:
                return [ChunkResult(None, error=_worker_failure_message(completed)) for _ in specs]
            results = json.loads(output_path.read_text())["results"]
            if len(results) != len(specs):
                raise RuntimeError("Neuron worker returned a different number of results")
            annotation = (
                f"container_image_id={image_id}; host_driver_package={host_driver or 'unavailable'}"
            )
            parsed = []
            for row in results:
                result = chunk_result_from_payload(row)
                if result.metrics is not None:
                    if result.observed_gpu_name != self.device.gpu_name:
                        raise RuntimeError(
                            "Neuron worker physical observation differs from its reservation"
                        )
                    version = result.backend_version
                    result = replace(
                        result,
                        backend_version=f"{version}; {annotation}" if version else annotation,
                    )
                parsed.append(result)
            return parsed
