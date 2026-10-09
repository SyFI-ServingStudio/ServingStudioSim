"""Immutable stock server launch and CPU-only Explorer export commands."""

from __future__ import annotations

import os
from pathlib import Path

from alignment.neuron.vllm_config import VllmNeuronProfileConfig

SOURCE_ROOT = Path(__file__).resolve().parents[2]
SERVED_MODEL = "llama31-8b-neuron"


def docker_prefix(config: VllmNeuronProfileConfig) -> list[str]:
    env = config.server.container_env()
    return [*env.docker_command, "--host", env.docker_host]


def docker_environment() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith("DOCKER_")}


def server_command(config: VllmNeuronProfileConfig, device, container_name: str) -> list[str]:
    config.validate()
    s = config.server
    if (
        device.device_id != s.neuron_device
        or device.lnc != 2
        or len(device.core_ids) != 4
        or device.busy
    ):
        raise ValueError("stock server requires its reserved idle whole TP4/LNC2 chip")
    if tuple(device.core_ids) != tuple(range(device.core_ids[0], device.core_ids[0] + 4)):
        raise ValueError("stock rank visibility requires contiguous logical core IDs")
    import json

    extra = {"neuron_config": {"num_batched_tokens_buckets": [512], "num_seqs_buckets": [1, 16]}}
    if config.profile_kind == "neuron":
        extra["neuron_profiler"] = {
            "output_dir": "/capture/profiles",
            "activities": ["system_profile"],
            "neuron_cores": [0, 1, 2, 3],
            "sys_trace_max_events_per_nc": 4000000,
        }
    command = [
        *docker_prefix(config),
        "run",
        "--rm",
        "--pull=never",
        "--no-healthcheck",
        "--name",
        container_name,
        "--network=host",
        "--shm-size=2g",
        "--security-opt=no-new-privileges",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--ulimit=core=0",
        "--device",
        f"/dev/neuron{device.device_id}",
    ]
    for host, target, access in (
        (s.model_path, "/model", "ro"),
        (s.cache_path, "/trial", "rw"),
        (str(Path(config.log_dir).resolve()), "/capture", "rw"),
        (str(SOURCE_ROOT / "alignment"), "/source/alignment", "ro"),
    ):
        command += ["--volume", f"{host}:{target}:{access}"]
    policy = {
        "PYTHONPATH": "/source",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "DO_NOT_TRACK": "1",
        "VLLM_SERVER_DEV_MODE": "1",
        "VLLM_CACHE_ROOT": "/trial/cache",
        "XDG_CACHE_HOME": "/trial/cache",
        "NKI_COMPILE_CACHE_URL": "/trial/nki-cache",
        "TMPDIR": "/trial/tmp",
        "NEURON_SKIP_EFA_AFFINITY": "1",
        "NEURON_LOGICAL_NC_CONFIG": "2",
        "NEURON_VISIBLE_DEVICES": f"{device.core_ids[0]}-{device.core_ids[-1]}",
        "MASTER_ADDR": "127.0.0.1",
        "VLLM_HOST_IP": "127.0.0.1",
        "GLOO_SOCKET_IFNAME": "lo",
        "OMP_NUM_THREADS": "4",
    }
    for key, value in policy.items():
        command += ["--env", f"{key}={value}"]
    command += [
        "--entrypoint",
        s.python_executable,
        s.image,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        "/model",
        "--host",
        s.host,
        "--port",
        str(s.port),
        "--served-model-name",
        SERVED_MODEL,
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "4",
        "--seed",
        "0",
        "--max-model-len",
        "512",
        "--max-num-batched-tokens",
        "512",
        "--max-num-seqs",
        "16",
        "--num-gpu-blocks-override",
        "6782",
        "--no-enable-prefix-caching",
        "--enable-prompt-tokens-details",
        "--enable-server-load-tracking",
        "--api-server-count",
        "1",
        # Drain the engine before Docker's 30s stop deadline. The public
        # default (0) force-kills the manager and can race its output handler.
        "--shutdown-timeout",
        "10",
        "--scheduler-cls",
        "alignment.neuron.vllm_observer.ObservedScheduler",
        "--worker-cls",
        "alignment.neuron.vllm_observer.ObservedWorker",
        "--additional-config",
        json.dumps(extra, separators=(",", ":")),
    ]
    if config.profile_kind == "neuron":
        command += ["--profiler-config", '{"profiler":"cuda"}']
    return command


def export_command(config: VllmNeuronProfileConfig) -> list[str]:
    """Export system records after the server has stopped; no device mapping."""
    return [
        *docker_prefix(config),
        "run",
        "--rm",
        "--pull=never",
        "--no-healthcheck",
        "--network=none",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--security-opt=no-new-privileges",
        "--volume",
        f"{config.log_dir}:/capture:rw",
        "--entrypoint",
        "/opt/aws/neuron/bin/neuron-explorer",
        config.server.image,
        "view",
        "-d",
        "/capture/profiles",
        "--output-format",
        "json",
        "--ignore-device-profile",
        "--system-trace-filter-event-type",
        "nc_exec_running,nrt_model_submit,kbl_exec_pre,kbl_exec_post",
        "--output-file",
        "/capture/system-trace.json",
        "--disable-ui",
        "--force",
    ]
