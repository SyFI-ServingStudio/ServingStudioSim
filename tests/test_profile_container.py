"""Container-specific profiling provenance and path contracts."""

from __future__ import annotations

import json
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from profiling.exec.container_user import rootless_runtime as real_rootless_runtime
from profiling.exec.env import ContainerProfileEnv
from profiling.exec.local import (
    IMEX_CHANNEL_CDI_DEVICE,
    LocalGpuChunk,
    _container_worker_command,
    gpu_uuids_for_indices,
)
from profiling.profilers.energy import ENERGY_ENV
from profiling.profilers.timer import CUPTI_TRACE_ENV


def _fake_smi_xml(count: int = 8) -> ET.Element:
    gpus = "".join(f"<gpu><uuid>GPU-fake-{index}</uuid></gpu>" for index in range(count))
    return ET.fromstring(f"<nvidia_smi_log>{gpus}</nvidia_smi_log>")


@pytest.fixture(autouse=True)
def stub_nvidia_smi(monkeypatch):
    """Give every container-command test a deterministic index-to-UUID map.

    The command builder now resolves UUIDs, so these unit tests would otherwise
    depend on the host's real cards.
    """

    monkeypatch.setattr("profiling.exec.local._nvidia_smi_xml", _fake_smi_xml)
    # Nor on the host's container runtime: pin a rootful one.
    monkeypatch.setattr("profiling.exec.container_user.rootless_runtime", lambda: False)


def test_container_worker_mounts_worktree_source_read_only(tmp_path: Path, monkeypatch) -> None:
    exchange_dir = tmp_path / "exchange"
    cache_dir = tmp_path / "cache"
    exchange_dir.mkdir()
    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(cache_dir))

    command, _env = _container_worker_command(
        ContainerProfileEnv("vllm_env", "profiler:test", "/usr/bin/python3"),
        [3, 1],
        exchange_dir,
    )

    assert '"device=GPU-fake-3,GPU-fake-1"' in command
    assert "HOME=/cache/home" in command
    assert "USER=vibesim" in command
    assert "LOGNAME=vibesim" in command
    assert f"{exchange_dir}:/io" in command
    # Each image keeps its own JIT cache: builds against one image's libraries.
    assert f"{cache_dir}/profiler_test:/cache" in command
    assert "MOESIM_CUPTI_BUILD_DIR=/cache/cupti-ext" in command
    project_root = Path(__file__).parents[1].resolve()
    for directory in ("profiling", "launcher", "gpu", "reference_engines"):
        assert f"{project_root / directory}:/opt/vibesim/{directory}:ro" in command
    assert not any("profile.db" in argument for argument in command)
    # The worker is named explicitly, never left to the image's ENTRYPOINT.
    assert command[command.index("--entrypoint") + 1] == "/usr/bin/python3"
    assert command[-7:] == [
        "profiler:test",
        "-m",
        "profiling.exec.local_worker",
        "--worker-input",
        "/io/input.json",
        "--worker-output",
        "/io/output.json",
    ]


def test_container_worker_mounts_explicit_additional_volume(tmp_path: Path, monkeypatch) -> None:
    exchange_dir = tmp_path / "exchange"
    cache_dir = tmp_path / "cache"
    measurement_dir = tmp_path / "measurement"
    exchange_dir.mkdir()
    measurement_dir.mkdir()
    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(cache_dir))

    command, _env = _container_worker_command(
        ContainerProfileEnv("vllm_env", "profiler:test", "/usr/bin/python3"),
        [4],
        exchange_dir,
        additional_volumes=((measurement_dir, measurement_dir.resolve()),),
    )

    measurement_mount = f"{measurement_dir.resolve()}:{measurement_dir.resolve()}"
    assert measurement_mount in command
    assert command.index(measurement_mount) < command.index("profiler:test")


def test_container_worker_never_passes_a_bare_gpu_index(tmp_path: Path, monkeypatch) -> None:
    """The container runtime resolves indices in its own device view, not ours.

    Under a device cgroup the allocated card enumerates locally as index 0 while
    ``dockerd``, outside the cgroup, reads ``device=0`` as the host's first card,
    so every worker silently lands on GPU 0. Only a UUID survives the crossing.
    """

    exchange_dir = tmp_path / "exchange"
    exchange_dir.mkdir()
    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(tmp_path / "cache"))

    command, _env = _container_worker_command(
        ContainerProfileEnv("vllm_env", "profiler:test", "/usr/bin/python3"),
        [0],
        exchange_dir,
    )

    gpu_request = command[command.index("--gpus") + 1]
    assert gpu_request == '"device=GPU-fake-0"'
    assert "device=0" not in gpu_request


def test_container_worker_pins_cuda_to_the_requested_uuids(tmp_path: Path, monkeypatch) -> None:
    """A rootless Podman inside a Slurm job ignores ``--gpus`` and exposes every
    job GPU, so the device request alone left concurrent workers on job GPU 0.
    CUDA_VISIBLE_DEVICES carries the same UUIDs, in reservation order."""

    exchange_dir = tmp_path / "exchange"
    exchange_dir.mkdir()
    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(tmp_path / "cache"))

    command, _env = _container_worker_command(
        ContainerProfileEnv("vllm_env", "profiler:test", "/usr/bin/python3"),
        [3, 1],
        exchange_dir,
    )

    assert "CUDA_VISIBLE_DEVICES=GPU-fake-3,GPU-fake-1" in command
    assert command[command.index("CUDA_VISIBLE_DEVICES=GPU-fake-3,GPU-fake-1") - 1] == "--env"
    assert command[command.index("--gpus") + 1] == '"device=GPU-fake-3,GPU-fake-1"'


def _run_chunk_with_worker_uuids(monkeypatch, gpus: list[int], worker_uuids) -> list:
    """Run a host-runtime chunk whose fake worker succeeds and reports ``worker_uuids``."""

    monkeypatch.setenv("VIBESIM_PROFILE_RUNTIME", "host")

    def fake_subprocess_run(cmd, **kwargs):
        del kwargs
        output_path = Path(cmd[cmd.index("--worker-output") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "ok": True,
                            "metrics": {
                                "kind": "compute",
                                "time_ms": 1.0,
                                "tflops": 0.1,
                                "memory_bandwidth_gbps": 0.2,
                                "energy_j": 0.0,
                            },
                            "gpu_name": "FakeGPU",
                        }
                    ],
                    "gpu_uuids": worker_uuids,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("profiling.exec.local.subprocess.run", fake_subprocess_run)
    return LocalGpuChunk(gpus).run(
        "single_gemm",
        [{"m": 8, "n": 8, "k": 8, "dtype": "fp16", "backend": "torch"}],
    )


def test_chunk_keeps_results_measured_on_the_reserved_card(monkeypatch) -> None:
    (result,) = _run_chunk_with_worker_uuids(monkeypatch, [2], ["GPU-fake-2"])

    assert result.error is None
    assert result.metrics is not None


@pytest.mark.parametrize(
    "worker_uuids",
    [
        ["GPU-fake-0"],  # another card: the runtime ignored the request
        ["GPU-fake-0", "GPU-fake-2"],  # every job card visible, CUDA picks the first
        None,  # a worker that could not say
    ],
)
def test_chunk_rejects_results_from_a_card_it_did_not_reserve(monkeypatch, worker_uuids) -> None:
    (result,) = _run_chunk_with_worker_uuids(monkeypatch, [2], worker_uuids)

    assert result.metrics is None
    assert "worker GPU mismatch" in (result.error or "")
    assert "GPU-fake-2" in (result.error or "")


def test_container_worker_fails_when_the_uuid_is_unknown(tmp_path: Path, monkeypatch) -> None:
    exchange_dir = tmp_path / "exchange"
    exchange_dir.mkdir()
    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr("profiling.exec.local._nvidia_smi_xml", lambda: _fake_smi_xml(2))

    with pytest.raises(RuntimeError, match="no UUID for GPU index"):
        _container_worker_command(
            ContainerProfileEnv("vllm_env", "profiler:test", "/usr/bin/python3"),
            [5],
            exchange_dir,
        )


def test_gpu_uuids_require_nvidia_smi(monkeypatch) -> None:
    monkeypatch.setattr("profiling.exec.local._nvidia_smi_xml", lambda: None)

    with pytest.raises(RuntimeError, match="nvidia-smi is required"):
        gpu_uuids_for_indices([0])


def test_container_worker_does_not_forward_the_energy_policy(tmp_path, monkeypatch) -> None:
    """The energy policy rides the JSON payload, and must ride only that.

    It used to be forwarded as ``--env`` here as well. Two channels for one
    policy is how they drift: a container run whose payload said off while its
    environment said on would measure a different thing than the host rows
    sitting beside it in the same table.
    """

    exchange_dir = tmp_path / "exchange"
    exchange_dir.mkdir()
    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv(ENERGY_ENV, "1")

    command, _env = _container_worker_command(
        ContainerProfileEnv("vllm_env", "profiler:test", "/usr/bin/python3"),
        [0],
        exchange_dir,
    )

    assert not any(argument.startswith(ENERGY_ENV) for argument in command)


def test_container_worker_mounts_the_trace_directory(tmp_path, monkeypatch) -> None:
    """Forwarding the trace path is useless unless the path also exists inside.

    Without the mount the container write fails, ``_emit_trace`` swallows it,
    and the container groups come back blank while the host groups beside them
    trace normally.
    """

    exchange_dir = tmp_path / "exchange"
    trace_dir = tmp_path / "traces"
    exchange_dir.mkdir()
    trace_dir.mkdir()
    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv(CUPTI_TRACE_ENV, str(trace_dir / "run.jsonl"))

    command, _env = _container_worker_command(
        ContainerProfileEnv("vllm_env", "profiler:test", "/usr/bin/python3"),
        [0],
        exchange_dir,
    )

    assert f"{CUPTI_TRACE_ENV}={trace_dir / 'run.jsonl'}" in command
    assert f"{trace_dir.resolve()}:{trace_dir.resolve()}" in command


def test_container_worker_redirects_the_inductor_cache_into_the_mount(
    tmp_path: Path, monkeypatch
) -> None:
    """Inductor's default cache dir is under /tmp, which dies with the container.

    Left alone, every submission recompiles from nothing while the host workers
    beside it reuse a warm cache -- a difference in cost only, but an unbounded
    one, and invisible because a recompile looks exactly like a slow import.
    """

    exchange_dir = tmp_path / "exchange"
    exchange_dir.mkdir()
    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(tmp_path / "cache"))

    command, _env = _container_worker_command(
        ContainerProfileEnv("vllm_env", "profiler:test", "/usr/bin/python3"),
        [0],
        exchange_dir,
    )

    cache_dir = command[command.index("HOME=/cache/home") - 1 :]
    assert "TORCHINDUCTOR_CACHE_DIR=/cache/home/.cache/torchinductor" in command
    # Whatever it points at has to be inside the one directory that is mounted.
    inductor = next(a for a in command if a.startswith("TORCHINDUCTOR_CACHE_DIR="))
    assert inductor.split("=", 1)[1].startswith("/cache/")
    assert any(a.endswith(":/cache") for a in cache_dir)


def test_container_worker_requests_imex_channel_only_when_asked(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(tmp_path / "cache"))
    profiler_env = ContainerProfileEnv("vllm_env", "profiler:test", "/usr/bin/python3")

    plain, _env = _container_worker_command(profiler_env, [0, 1], tmp_path)
    fabric, _env = _container_worker_command(profiler_env, [0, 1], tmp_path, imex_channel=True)

    assert IMEX_CHANNEL_CDI_DEVICE == "nvidia.com/imex-channel=0"
    assert "--device" not in plain
    device = fabric.index("--device")
    assert fabric[device + 1] == IMEX_CHANNEL_CDI_DEVICE
    # Docker options must precede the image name.
    assert device < fabric.index("profiler:test")
    assert [arg for arg in fabric if arg not in ("--device", IMEX_CHANNEL_CDI_DEVICE)] == plain


def test_imex_channel_is_declared_by_the_mnnvl_all_reduce_backends_only() -> None:
    from profiling.db.registry import iter_kernel_profiler_specs

    declared = {
        (spec.kernel_kind, spec.backend)
        for spec in iter_kernel_profiler_specs()
        if spec.imex_channel
    }
    assert declared == {
        ("all_reduce_fusion", "flashinfer_mnnvl"),
        ("all_reduce_residual_rms_norm", "flashinfer_mnnvl"),
    }


def test_container_worker_runs_as_root_under_a_rootless_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    """Catches a worker run as the caller's uid under rootless Podman, which maps
    it to a subordinate host uid that cannot read the 0700 exchange directory."""
    exchange_dir = tmp_path / "exchange"
    exchange_dir.mkdir()
    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(tmp_path / "cache"))
    profiler_env = ContainerProfileEnv("vllm_env", "profiler:test", "/usr/bin/python3")

    def user(command: list[str]) -> str:
        return command[command.index("--user") + 1]

    rootful, _env = _container_worker_command(profiler_env, [0], exchange_dir)
    assert user(rootful) == f"{os.getuid()}:{os.getgid()}"
    monkeypatch.setattr("profiling.exec.container_user.rootless_runtime", lambda: True)
    rootless, _env = _container_worker_command(profiler_env, [0], exchange_dir)
    assert user(rootless) == "0:0"


@pytest.mark.parametrize(
    ("security_options", "rootless"),
    [
        ('["name=seccomp,profile=default","name=rootless","name=cgroupns"]', True),
        ('["name=seccomp,profile=builtin","name=cgroupns"]', False),
    ],
)
def test_rootless_runtime_reads_docker_info(monkeypatch, security_options, rootless) -> None:
    import subprocess

    real_rootless_runtime.cache_clear()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_a, **_k: subprocess.CompletedProcess([], 0, stdout=security_options),
    )
    try:
        assert real_rootless_runtime() is rootless
    finally:
        real_rootless_runtime.cache_clear()
