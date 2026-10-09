"""CPU coverage of the Neuron container boundary; no SDK or physical device calls."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from profiling.exec import local_worker, neuron
from profiling.exec.env import ContainerProfileEnv, ProfileEnv
from profiling.exec.payload import metrics_to_payload
from profiling.kernels.single_gemm import SingleGemmArgs
from profiling.runners.metrics import ComputeMetrics, RunnerResult

IMAGE = "sha256:" + "a" * 64
DEVICE = neuron.NeuronDevice(3, (12, 13, 14, 15), 2, 96 * 1024**3, False, "trn2.3xlarge")


@pytest.fixture
def container_setup(tmp_path, monkeypatch):
    source = tmp_path / "request source"
    for name in ("profiling", "gpu", "tools"):
        (source / name).mkdir(parents=True)
    exchange = tmp_path / "exchange"
    exchange.mkdir()
    cache = tmp_path / "cache"
    tool = tmp_path / "neuron-ls"
    tool.write_text("identity tool fixture")
    address = tmp_path / "private.sock"
    # Socket binding is unnecessary for this CPU boundary test and is blocked
    # in restricted test environments. Mock only the OS file-type observation.
    address.touch()
    is_socket = Path.is_socket
    monkeypatch.setattr(Path, "is_socket", lambda path: path == address or is_socket(path))
    monkeypatch.setattr(neuron, "_PROJECT_ROOT", source)
    monkeypatch.setattr(neuron.shutil, "which", lambda name: str(tool))
    monkeypatch.setenv("SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR", str(cache))
    monkeypatch.delenv("VIBESIM_PROFILE_SOURCE_MODE", raising=False)
    for name in (*neuron._NXDI_DATA_ENV, neuron.TIMELINE_ENV):
        monkeypatch.delenv(name, raising=False)
    env = ContainerProfileEnv(
        "coherent",
        IMAGE,
        ("sudo", "-n", "docker"),
        f"unix://{address}",
        Path("/opt/image python/bin/python"),
    )
    return SimpleNamespace(env=env, source=source, exchange=exchange, cache=cache, tool=tool)


def worker_environment(command):
    return dict(command[i + 1].split("=", 1) for i, value in enumerate(command) if value == "--env")


def test_container_declarations_keep_cuda_defaults_and_image_python_unstatted(container_setup):
    assert ContainerProfileEnv("cuda", "tag").docker_command == ("docker",)
    assert ContainerProfileEnv("cuda", "tag").python_executable is None
    container_setup.env.validate()  # The declared /opt image Python does not exist on the host.


@pytest.mark.parametrize(
    "host",
    [
        "tcp://remote:2375",
        "unix://remote/tmp/docker.sock",
        "unix://relative",
        "unix:/tmp/docker.sock",
        "unix:///tmp/docker.sock?redirect=1",
    ],
)
def test_container_transport_rejects_nonlocal_or_ambiguous_address(host):
    with pytest.raises(ValueError, match="local absolute"):
        ContainerProfileEnv("bad", IMAGE, docker_host=host)


def test_container_declarations_reject_shell_strings_and_relative_python():
    with pytest.raises(TypeError, match="argv tuple"):
        ContainerProfileEnv("bad", IMAGE, docker_command="sudo -n docker")
    with pytest.raises(ValueError, match="nonempty"):
        ContainerProfileEnv("bad", IMAGE, docker_command=())
    with pytest.raises(ValueError, match="absolute"):
        ContainerProfileEnv("bad", IMAGE, python_executable=Path("venv/bin/python"))


def test_container_requires_real_socket_and_explicit_transport_python(tmp_path, monkeypatch):
    monkeypatch.setattr(neuron.shutil, "which", lambda name: "/usr/bin/docker")
    with pytest.raises(FileNotFoundError, match="socket"):
        ContainerProfileEnv("missing", IMAGE, docker_host=f"unix://{tmp_path}/missing").validate()
    with pytest.raises(ValueError, match="explicit"):
        neuron._neuron_docker_prefix(ContainerProfileEnv("missing", IMAGE))


def test_neuron_argv_and_mounts_preserve_identity_and_image_abi(container_setup, monkeypatch):
    setup = container_setup
    data = {}
    for name in neuron._NXDI_DATA_ENV:
        path = setup.source.parent / (name.lower() + " with spaces")
        path.mkdir()
        data[name] = str(path)
        monkeypatch.setenv(name, str(path))
    monkeypatch.setenv("PYTHONPATH", "/host/site-packages")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/host/libraries")
    monkeypatch.setenv("DOCKER_CONTEXT", "remote")
    monkeypatch.setenv("DOCKER_TLS_VERIFY", "1")
    command, cli_env = neuron._neuron_container_worker_command(
        setup.env,
        DEVICE,
        setup.exchange,
        image_id=IMAGE,
        worker_env={
            "SERVINGSTUDIO_NEURON_DEVICE": "99",
            "NEURON_RT_VISIBLE_CORES": "0-99",
            "NEURON_LOGICAL_NC_CONFIG": "1",
            "PYTHONPATH": "/other/checkout",
            "PATH": "/host/bin",
            "LD_LIBRARY_PATH": "/host/libs",
            "PJRT_DEVICE": "NEURON",
        },
        kernel_kind="neuron_llama_forward",
    )
    policy = worker_environment(command)
    assert command[:5] == ["sudo", "-n", "docker", "--host", setup.env.docker_host]
    assert "--device=/dev/neuron3:/dev/neuron3" in command
    assert sum(value.startswith("--device=") for value in command) == 1
    assert "--gpus" not in command and "--privileged" not in command
    assert {"--pull=never", "--network=none", "--no-healthcheck", "--ulimit=core=0"} <= set(command)
    assert "--entrypoint=/opt/image python/bin/python" in command
    assert policy["SERVINGSTUDIO_NEURON_DEVICE"] == "3"
    assert policy["NEURON_RT_VISIBLE_CORES"] == "12"
    assert policy["NEURON_LOGICAL_NC_CONFIG"] == "2"
    assert policy["CUDA_VISIBLE_DEVICES"] == ""
    assert policy["PJRT_DEVICE"] == "NEURON"
    assert policy["PYTHONPATH"] == str(setup.source)
    assert not {"PATH", "LD_LIBRARY_PATH", "PYTHONHOME"} & policy.keys()
    assert policy["TMPDIR"] == policy["TEMP"] == policy["TMP"] == str(setup.cache / "tmp")
    assert policy["HF_HUB_OFFLINE"] == policy["TRANSFORMERS_OFFLINE"] == "1"
    for name in ("profiling", "gpu", "tools"):
        path = setup.source / name
        assert f"{path}:{path}:ro" in command
    for path in data.values():
        assert f"{path}:{path}:ro" in command
    assert f"{setup.cache}:{setup.cache}" in command
    assert f"{setup.exchange}:/io" in command
    assert not any("private.sock:" in value for value in command)
    assert "DOCKER_CONTEXT" not in cli_env and "DOCKER_TLS_VERIFY" not in cli_env
    assert "DOCKER_CONTEXT" in neuron.os.environ  # CLI cleanup is process-local.
    assert command[-6:] == [
        "-m",
        "profiling.exec.local_worker",
        "--worker-input",
        "/io/input.json",
        "--worker-output",
        "/io/output.json",
    ]


def test_image_inspection_binds_immutable_local_content(container_setup, monkeypatch):
    calls = []

    def inspect(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0, stdout=json.dumps([{"Id": IMAGE, "RepoDigests": []}]), stderr=""
        )

    monkeypatch.setattr(neuron.subprocess, "run", inspect)
    assert neuron._inspect_neuron_image(container_setup.env) == IMAGE
    assert calls[0][-3:] == ["image", "inspect", IMAGE]
    monkeypatch.setattr(
        neuron.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=json.dumps([{"Id": "sha256:" + "b" * 64}]), stderr=""
        ),
    )
    with pytest.raises(RuntimeError, match="declared immutable"):
        neuron._inspect_neuron_image(container_setup.env)


@pytest.mark.parametrize(
    "problem", ["image-source", "missing-data", "relative-data", "cache-overlap", "socket-exposure"]
)
def test_neuron_path_and_source_failures(container_setup, monkeypatch, problem):
    setup = container_setup
    kind = None
    if problem == "image-source":
        monkeypatch.setenv("VIBESIM_PROFILE_SOURCE_MODE", "image")
    elif problem == "missing-data":
        kind = "neuron_llama_forward"
    elif problem == "relative-data":
        monkeypatch.setenv(neuron._NXDI_DATA_ENV[0], "relative-model")
    elif problem == "cache-overlap":
        monkeypatch.setenv(
            "SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR", str(setup.source / "profiling/cache")
        )
    else:
        setup.env = ContainerProfileEnv(
            "socket-in-cache",
            IMAGE,
            setup.env.docker_command,
            f"unix://{setup.cache}/private.sock",
            setup.env.python_executable,
        )
    with pytest.raises(ValueError):
        neuron._neuron_container_worker_command(
            setup.env, DEVICE, setup.exchange, image_id=IMAGE, kernel_kind=kind
        )


def test_neuron_timeline_is_writable_without_source_or_data_writes(container_setup, monkeypatch):
    trace = container_setup.source.parent / "timeline outputs" / "run.jsonl"
    monkeypatch.setenv(neuron.TIMELINE_ENV, str(trace))
    command, _ = neuron._neuron_container_worker_command(
        container_setup.env, DEVICE, container_setup.exchange, image_id=IMAGE
    )
    assert worker_environment(command)[neuron.TIMELINE_ENV] == str(trace)
    assert f"{trace.parent}:{trace.parent}" in command


def test_stock_rank_local_kernel_keeps_lite_and_nki_caches_in_declared_mount(container_setup):
    setup = container_setup
    command, _ = neuron._neuron_container_worker_command(
        replace(setup.env, name="vllm_neuron_env"), DEVICE, setup.exchange,
        image_id=IMAGE, kernel_kind="neuron_dense_mlp",
    )
    policy = worker_environment(command)
    assert policy["VLLM_CACHE_ROOT"] == policy["XDG_CACHE_HOME"] == str(setup.cache / "cache")
    assert policy["NKI_COMPILE_CACHE_URL"] == str(setup.cache / "nki-cache")
    assert policy["NEURON_RT_VISIBLE_CORES"] == "12"
    assert "NEURON_VISIBLE_DEVICES" not in policy


@pytest.mark.parametrize("mismatch", ["device", "cores", "range", "lnc", "generation"])
def test_worker_reservation_mismatch_prevents_runner_loading(tmp_path, monkeypatch, mismatch):
    events = []
    supports = SimpleNamespace(device_family="neuron")
    profiler = SimpleNamespace(supports=supports, load_list_runner=lambda: events.append("loaded"))
    monkeypatch.setattr(local_worker, "resolve_chunk_backend", lambda *a: "test")
    monkeypatch.setattr(local_worker, "find_kernel_profiler_spec", lambda *a: profiler)
    monkeypatch.setattr(local_worker, "unsupported_device", lambda *a: None)
    monkeypatch.setattr(neuron, "neuron_devices", lambda: [DEVICE])
    for name, value in neuron._reservation_environment(DEVICE).items():
        monkeypatch.setenv(name, value)
    snapshot = neuron._reservation_payload(DEVICE)
    if mismatch == "device":
        snapshot["neuron_device"] = 4
    elif mismatch == "cores":
        snapshot["core_ids"] = [12, 15, 14, 13]
    elif mismatch == "range":
        monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "12-15")
    elif mismatch == "lnc":
        snapshot["lnc"] = 1
    else:
        monkeypatch.setattr(
            neuron,
            "neuron_devices",
            lambda: [
                neuron.NeuronDevice(
                    3, DEVICE.core_ids, 2, DEVICE.memory_bytes, False, "trn3.3xlarge"
                )
            ],
        )
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "kernel_kind": "single_gemm",
                "specs": [{"backend": "test"}],
                "energy": False,
                "execution": snapshot,
            }
        )
    )
    with pytest.raises((ValueError, RuntimeError)):
        local_worker._worker_main(request, tmp_path / "output.json")
    assert events == []


def test_worker_observation_and_row_fingerprints_survive_container_boundary(tmp_path, monkeypatch):
    metrics = ComputeMetrics(1, 2, 3)
    profiler = SimpleNamespace(
        supports=SimpleNamespace(device_family="neuron"),
        args_schema=SingleGemmArgs,
        load_list_runner=lambda: lambda specs: [RunnerResult(metrics)],
        load_row_provenance=lambda: lambda **kwargs: "checkpoint=abc; hlo=def",
    )
    monkeypatch.setattr(local_worker, "resolve_chunk_backend", lambda *a: "test")
    monkeypatch.setattr(local_worker, "find_kernel_profiler_spec", lambda *a: profiler)
    monkeypatch.setattr(local_worker, "unsupported_device", lambda *a: None)
    monkeypatch.setattr(
        local_worker,
        "_neuron_versions",
        lambda: {"cuda_version": None, "backend_version": "torch=2.9"},
    )
    monkeypatch.setattr(neuron, "neuron_devices", lambda: [DEVICE])
    for name, value in neuron._reservation_environment(DEVICE).items():
        monkeypatch.setenv(name, value)
    request, output = tmp_path / "request.json", tmp_path / "output.json"
    request.write_text(
        json.dumps(
            {
                "kernel_kind": "single_gemm",
                "specs": [{"m": 2, "n": 2, "k": 2, "dtype": "bf16", "backend": "test"}],
                "energy": False,
                "execution": neuron._reservation_payload(DEVICE),
            }
        )
    )
    local_worker._worker_main(request, output)
    (row,) = json.loads(output.read_text())["results"]
    assert row["gpu_name"] == DEVICE.gpu_name and row["cuda_version"] is None
    assert row["backend_version"] == "torch=2.9; checkpoint=abc; hlo=def"


@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "nonzero",
        "malformed",
        "count",
        "missing-observation",
        "wrong-observation",
        "missing-image",
    ],
)
def test_container_chunk_holds_and_releases_lock_through_result_parsing(
    container_setup, monkeypatch, outcome
):
    from profiling.profilers import energy

    setup = container_setup
    monkeypatch.delenv("SERVINGSTUDIO_NO_GPU", raising=False)
    monkeypatch.setattr(energy, "energy_enabled", lambda: False)
    monkeypatch.setattr(neuron, "resolve_chunk_backend", lambda *a: "test")
    monkeypatch.setattr(
        neuron,
        "find_kernel_profiler_spec",
        lambda *a: SimpleNamespace(
            supports=SimpleNamespace(device_family="neuron"),
            subprocess_env="test",
            worker_env=(),
            neuron_logical_cores=1,
        ),
    )
    monkeypatch.setattr(neuron, "resolve_profile_env", lambda name: setup.env)
    if outcome == "missing-image":

        def fail(*a):
            raise RuntimeError("missing local image")

        monkeypatch.setattr(neuron, "_inspect_neuron_image", fail)
    else:
        monkeypatch.setattr(neuron, "_inspect_neuron_image", lambda env: IMAGE)
    monkeypatch.setattr(neuron, "_host_driver_version", lambda: "2.34.10")
    lock = (setup.source.parent / "reservation.lock").open("a")
    chunk = neuron.LocalNeuronChunk(DEVICE, lock)

    def run(command, **kwargs):
        assert not lock.closed
        cwd = Path(kwargs["cwd"])
        assert json.loads((cwd / "input.json").read_text())[
            "execution"
        ] == neuron._reservation_payload(DEVICE)
        row = {
            "ok": True,
            "metrics": metrics_to_payload(ComputeMetrics(1, 2, 3)),
            "gpu_name": DEVICE.gpu_name,
            "cuda_version": None,
            "backend_version": "hlo=abc",
        }
        if outcome == "missing-observation":
            row.pop("gpu_name")
        elif outcome == "wrong-observation":
            row["gpu_name"] = "NVIDIA H200"
        text = (
            "invalid JSON"
            if outcome == "malformed"
            else json.dumps({"results": [] if outcome == "count" else [row]})
        )
        (cwd / "output.json").write_text(text)
        return SimpleNamespace(
            returncode=17 if outcome == "nonzero" else 0, stderr="worker stderr", stdout=""
        )

    monkeypatch.setattr(neuron.subprocess, "run", run)
    if outcome in ("success", "nonzero"):
        (result,) = chunk.run("test_kind", [{"backend": "test"}])
        if outcome == "success":
            assert result.observed_gpu_name == DEVICE.gpu_name
            assert (
                result.backend_version
                == f"hlo=abc; container_image_id={IMAGE}; host_driver_package=2.34.10"
            )
            assert result.cuda_version is None
        else:
            assert result.metrics is None and "exited 17" in result.error
    else:
        with pytest.raises((ValueError, RuntimeError)):
            chunk.run("test_kind", [{"backend": "test"}])
    assert lock.closed


def test_host_neuron_chunk_keeps_host_interpreter_and_reservation(tmp_path, monkeypatch):
    from profiling.profilers import energy

    monkeypatch.delenv("SERVINGSTUDIO_NO_GPU", raising=False)
    monkeypatch.setattr(energy, "energy_enabled", lambda: False)
    monkeypatch.setattr(neuron, "resolve_chunk_backend", lambda *a: "test")
    monkeypatch.setattr(
        neuron,
        "find_kernel_profiler_spec",
        lambda *a: SimpleNamespace(
            supports=SimpleNamespace(device_family="neuron"),
            subprocess_env="host",
            worker_env=(),
            neuron_logical_cores=1,
        ),
    )
    env = ProfileEnv("host", Path(sys.executable))
    monkeypatch.setattr(neuron, "resolve_profile_env", lambda name: env)
    lock = (tmp_path / "reservation").open("a")

    def run(command, **kwargs):
        assert command[0] == str(env.python_executable) and "docker" not in command
        assert kwargs["env"]["NEURON_RT_VISIBLE_CORES"] == "12"
        assert not lock.closed
        (Path(kwargs["cwd"]) / "output.json").write_text(
            json.dumps({"results": [{"ok": False, "error": "shape unsupported"}]})
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(neuron.subprocess, "run", run)
    (result,) = neuron.LocalNeuronChunk(DEVICE, lock).run("test", [{"backend": "test"}])
    assert result.error == "shape unsupported" and lock.closed


def test_ubuntu_runtime_provenance_is_worker_owned(monkeypatch):
    monkeypatch.setattr(local_worker.importlib.metadata, "version", lambda name: "fixture-version")
    monkeypatch.setattr(
        local_worker.shutil,
        "which",
        lambda name: "/usr/bin/dpkg-query" if name == "dpkg-query" else None,
    )
    calls = []

    def query(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout="aws-neuronx-runtime-lib=2.34.10\naws-neuronx-collectives=2.34.10\n",
        )

    monkeypatch.setattr(local_worker.subprocess, "run", query)
    versions = local_worker._neuron_versions()
    assert versions["cuda_version"] is None
    assert "torch=fixture-version" in versions["backend_version"]
    assert "numpy=fixture-version" in versions["backend_version"]
    assert "ml-dtypes=fixture-version" in versions["backend_version"]
    assert "aws-neuronx-runtime-lib=2.34.10" in versions["backend_version"]
    assert "aws-neuronx-dkms" not in versions["backend_version"]
    assert calls[0][0] == "dpkg-query"


def test_digest_reference_inspection_resolves_exact_id_and_rejects_tags(
    container_setup, monkeypatch
):
    digest = "local/repository@sha256:" + "b" * 64
    env = replace(container_setup.env, image=digest)
    monkeypatch.setattr(
        neuron.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0,
            stdout=json.dumps([{"Id": IMAGE, "RepoDigests": [digest]}]),
            stderr="",
        ),
    )
    assert neuron._inspect_neuron_image(env) == IMAGE
    with pytest.raises(ValueError, match="immutable"):
        neuron._neuron_docker_prefix(replace(env, image="mutable:latest"))


@pytest.mark.parametrize("policy", [{"energy": True}, {"measure": {"duration_s": 1}}])
def test_neuron_worker_rejects_measure_and_energy_before_observation_or_runner(
    tmp_path, monkeypatch, policy
):
    profiler = SimpleNamespace(supports=SimpleNamespace(device_family="neuron"))
    monkeypatch.setattr(local_worker, "resolve_chunk_backend", lambda *a: "test")
    monkeypatch.setattr(local_worker, "find_kernel_profiler_spec", lambda *a: profiler)
    monkeypatch.setattr(local_worker, "set_energy_enabled", lambda enabled: None)
    monkeypatch.setattr(neuron, "neuron_devices", lambda: pytest.fail("device observation"))
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({"kernel_kind": "single_gemm", "specs": [{"backend": "test"}], **policy})
    )
    with pytest.raises(ValueError, match="energy or CUPTI"):
        local_worker._worker_main(request, tmp_path / "output.json")


def test_cuda_worker_protocol_does_not_observe_neuron(tmp_path, monkeypatch):
    profiler = SimpleNamespace(
        supports=SimpleNamespace(device_family="cuda"),
        args_schema=SingleGemmArgs,
        load_list_runner=lambda: lambda specs: [RunnerResult(ComputeMetrics(1, 2, 3))],
        load_row_provenance=lambda: None,
    )
    monkeypatch.setattr(local_worker, "resolve_chunk_backend", lambda *a: "test")
    monkeypatch.setattr(local_worker, "find_kernel_profiler_spec", lambda *a: profiler)
    monkeypatch.setattr(local_worker, "unsupported_device", lambda *a: None)
    monkeypatch.setattr(local_worker, "_current_gpu_name", lambda: "NVIDIA H200")
    monkeypatch.setattr(
        local_worker,
        "_runtime_versions",
        lambda backend: {"cuda_version": "13.0", "backend_version": "torch"},
    )
    monkeypatch.setattr(neuron, "neuron_devices", lambda: pytest.fail("Neuron observation"))
    request, output = tmp_path / "request.json", tmp_path / "output.json"
    request.write_text(
        json.dumps(
            {
                "kernel_kind": "single_gemm",
                "specs": [{"m": 2, "n": 2, "k": 2, "dtype": "bf16", "backend": "test"}],
                "energy": False,
            }
        )
    )
    local_worker._worker_main(request, output)
    (row,) = json.loads(output.read_text())["results"]
    assert row["gpu_name"] == "NVIDIA H200" and row["cuda_version"] == "13.0"


@pytest.mark.parametrize("manager", ["rpm", "dpkg-query"])
def test_host_driver_package_provenance_and_missing_package(manager, monkeypatch):
    monkeypatch.setattr(
        neuron.shutil, "which", lambda name: f"/usr/bin/{manager}" if name == manager else None
    )
    calls = []

    def query(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="2.34.10\n")

    monkeypatch.setattr(neuron.subprocess, "run", query)
    assert neuron._host_driver_version() == "2.34.10"
    assert calls[0][0] == manager and calls[0][-1] == "aws-neuronx-dkms"
    monkeypatch.setattr(
        neuron.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1, stdout="")
    )
    assert neuron._host_driver_version() is None
