"""What the machine a capture ran on looked like, read just before the server starts.

A capture is only as reproducible as the hardware and limits it ran under, and a
capture shipped to someone else's GPUs (an image run, later a work order) is
worth nothing if those cannot be compared. So every new capture writes
`capture_facts.json` beside its artifacts: the GPUs the process sees (model,
UUID, bus id, driver, clocks, power limit), their NVLink topology and switch
multicast / fabric-handle support, the IMEX channel, the memlock limit,
`/dev/shm`, `perf_event_paranoid`, and the image manifest when the capture runs
in one.

Only what the config itself states is enforced: every visible GPU must be the
config's `gpu`, and there must be exactly `gpu_count` (`dp_size * tp_size * pp * pcp`)
of them. Everything else is recorded, not judged here.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import resource
import subprocess
from pathlib import Path

from .config import ProfileConfig

FACTS_NAME = "capture_facts.json"
IMAGE_MANIFEST = Path("/etc/vibesim-profiler-image.json")
IMEX_CHANNEL = Path("/dev/nvidia-caps-imex-channels/channel0")

_GPU_FIELDS = ("index", "uuid", "name", "pci.bus_id", "driver_version", "memory.total")
# Queried one by one: a field a driver does not report must not hide the rest.
_GPU_OPTIONAL_FIELDS = (
    "clocks.max.sm",
    "clocks.max.mem",
    "clocks.applications.graphics",
    "power.limit",
    "persistence_mode",
    "compute_mode",
)

# cuda.h CUdevice_attribute values (stable since CUDA 12.0).
_CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 128
_CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132


def _run(argv: list[str], env: dict[str, str] | None = None) -> str | None:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, check=True, env=env)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout


def _query(fields: tuple[str, ...], env: dict[str, str]) -> list[dict[str, str]] | None:
    out = _run(
        ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"], env
    )
    if out is None:
        return None
    return [
        dict(zip(fields, (value.strip() for value in line.split(",")), strict=True))
        for line in out.splitlines()
        if line.strip()
    ]


def _gpus(env: dict[str, str]) -> list[dict[str, str]]:
    gpus = _query(_GPU_FIELDS, env) or []
    for field_name in _GPU_OPTIONAL_FIELDS:
        rows = _query(("index", field_name), env)
        values = {row["index"]: row[field_name] for row in rows or ()}
        for gpu in gpus:
            gpu[field_name] = values.get(gpu["index"])
    return gpus


def _cuda_device_attributes(env: dict[str, str]) -> dict:
    """Fabric-handle and switch-multicast support per CUDA ordinal, via the driver API.

    The process's own CUDA_VISIBLE_DEVICES is swapped for the server's while the
    driver initializes, so the ordinals are the server's.
    """
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "")
    try:
        libcuda = ctypes.CDLL("libcuda.so.1")
        if libcuda.cuInit(0) != 0:
            return {"error": "cuInit failed"}
        count = ctypes.c_int()
        if libcuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
            return {"error": "cuDeviceGetCount failed"}
        version = ctypes.c_int()
        libcuda.cuDriverGetVersion(ctypes.byref(version))
        devices = []
        for ordinal in range(count.value):
            device = ctypes.c_int()
            libcuda.cuDeviceGet(ctypes.byref(device), ordinal)
            values = {}
            for key, attribute in (
                ("fabric_handle_supported", _CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED),
                ("multicast_supported", _CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED),
            ):
                value = ctypes.c_int()
                status = libcuda.cuDeviceGetAttribute(ctypes.byref(value), attribute, device)
                values[key] = bool(value.value) if status == 0 else None
            devices.append({"ordinal": ordinal, **values})
        return {"driver_api_version": version.value, "devices": devices}
    except OSError as error:
        return {"error": f"libcuda.so.1: {error}"}
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous


def _limit(value: int) -> int | str:
    return "unlimited" if value == resource.RLIM_INFINITY else value


def collect(cfg: ProfileConfig, env: dict[str, str]) -> dict:
    """The facts, for the devices the server will be given (`env`)."""
    visible = env.get("CUDA_VISIBLE_DEVICES", "")
    # nvidia-smi ignores CUDA_VISIBLE_DEVICES and lists every device this
    # process may open; its index is the CUDA ordinal in the same view.
    all_gpus = _gpus(dict(os.environ))
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    by_token = {gpu["index"]: gpu for gpu in all_gpus} | {gpu["uuid"]: gpu for gpu in all_gpus}
    selected = [by_token[token] for token in tokens if token in by_token]
    memlock_soft, memlock_hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    shm = os.statvfs("/dev/shm")
    paranoid = Path("/proc/sys/kernel/perf_event_paranoid")
    return {
        "schema_version": 1,
        "hostname": platform.node(),
        "kernel": platform.release(),
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": visible,
        "gpus": selected,
        "gpus_seen": len(all_gpus),
        "cuda": _cuda_device_attributes(env),
        "topology": _run(["nvidia-smi", "topo", "-m"]),
        "nvlink_status": _run(["nvidia-smi", "nvlink", "--status"]),
        "imex_channel0": IMEX_CHANNEL.exists(),
        "memlock": {"soft": _limit(memlock_soft), "hard": _limit(memlock_hard)},
        "dev_shm_bytes": shm.f_frsize * shm.f_blocks,
        "perf_event_paranoid": paranoid.read_text().strip() if paranoid.exists() else None,
        "image": json.loads(IMAGE_MANIFEST.read_text()) if IMAGE_MANIFEST.is_file() else None,
    }


def check(cfg: ProfileConfig, facts: dict) -> list[str]:
    """What the config states and the machine contradicts. Empty means go."""
    problems = []
    expected = cfg.gpu_count
    gpus = facts["gpus"]
    if len(gpus) != expected:
        problems.append(
            f"server needs {expected} GPU(s) (dp_size * tp_size * pp * pcp), but "
            f"CUDA_VISIBLE_DEVICES={facts['cuda_visible_devices']!r} names "
            f"{len(gpus)} visible one(s)"
        )
    names = sorted({gpu["name"] for gpu in gpus})
    if any(name != cfg.gpu for name in names):
        problems.append(f"config gpu is {cfg.gpu!r}, but the devices are {names}")
    return problems


def record(cfg: ProfileConfig, env: dict[str, str], log_dir: Path) -> dict:
    """Write `capture_facts.json`; raise before the server starts if the config is contradicted."""
    facts = collect(cfg, env)
    (log_dir / FACTS_NAME).write_text(json.dumps(facts, indent=2))
    problems = check(cfg, facts)
    if problems:
        raise RuntimeError("capture preflight failed: " + "; ".join(problems))
    return facts
