"""Neuron profiling execution on the CPU: device routing, chip reservation and the
container boundary. No Neuron SDK, device or Docker daemon is touched."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from profiling.container import images
from profiling.db.args import DType
from profiling.db.registry import BackendSupport
from profiling.exec import local_worker, neuron
from profiling.exec.payload import metrics_to_payload
from profiling.gpu_policy import GpuDisabledError
from profiling.kernels.single_gemm import SingleGemmArgs
from profiling.runners.device import unsupported_device
from profiling.runners.metrics import ComputeMetrics, RunnerResult

IMAGE = "sha256:" + "a" * 64
ENV = "neuron_sdk233_env"
DEVICE = neuron.NeuronDevice(3, (12, 13, 14, 15), 2, 96 * 1024**3, False, "trn2.3xlarge")
NEURON_SUPPORT = BackendSupport(
    compute=frozenset({DType.BF16}),
    device_family="neuron",
    architectures=frozenset({"Trainium2"}),
)


def _neuron_ls_row(**overrides) -> dict:
    return {
        "instance_type": "trn2.3xlarge",
        "neuron_device": 0,
        "neuroncore_ids": [0, 1, 2, 3],
        "logical_neuroncore_config": 2,
        "memory_size": 96 * 1024**3,
        "neuron_processes": [],
        **overrides,
    }


# --- device family routing ---------------------------------------------------


def test_backend_family_prevents_cuda_neuron_cross_selection():
    cuda = BackendSupport(compute=frozenset({DType.BF16}))
    assert not cuda.allows(DType.BF16, gpu=neuron.TRAINIUM2_LNC2)
    assert cuda.allows(DType.BF16, gpu="NVIDIA H200")
    # Both catalog spellings of the part resolve to the Neuron family.
    assert NEURON_SUPPORT.allows(DType.BF16, gpu=neuron.TRAINIUM2_LNC2)
    assert NEURON_SUPPORT.allows(DType.BF16, gpu="Trainium2-LNC2")
    assert not NEURON_SUPPORT.allows(DType.BF16, gpu="NVIDIA H200")
    # A CUDA backend tolerates an unknown device; a Neuron backend never does.
    assert cuda.allows(DType.BF16, gpu="unrecognized accelerator")
    assert not NEURON_SUPPORT.allows(DType.BF16, gpu="unrecognized accelerator")
    assert not NEURON_SUPPORT.allows(DType.FP16, gpu=neuron.TRAINIUM2_LNC2)
    other_generation = BackendSupport(
        compute=None, device_family="neuron", architectures=frozenset({"Trainium3"})
    )
    assert not other_generation.allows(DType.BF16, gpu=neuron.TRAINIUM2_LNC2)


@pytest.mark.parametrize(
    ("supports", "cores", "message"),
    [
        (BackendSupport(compute=None, device_family="tpu"), 1, "unknown device family"),
        (
            BackendSupport(compute=None, device_family="neuron", sm_targets=frozenset({"sm_90a"})),
            1,
            "CUDA SM rules",
        ),
        (
            BackendSupport(compute=None, architectures=frozenset({"Trainium2"})),
            1,
            "not architectures",
        ),
        (BackendSupport(compute=None), 4, "cannot reserve NeuronCores"),
        (NEURON_SUPPORT, 0, "positive integer"),
    ],
)
def test_registry_rejects_inconsistent_device_declarations(supports, cores, message):
    from dataclasses import replace

    from profiling.db.registry import _validate_registry, find_kernel_profiler_spec

    spec = replace(
        find_kernel_profiler_spec("single_gemm", "torch"),
        backend="device_rule_fixture",
        supports=supports,
        neuron_logical_cores=cores,
    )
    with pytest.raises(ValueError, match=message):
        _validate_registry([spec])


def test_neuron_identity_comes_from_neuron_ls_not_the_requested_label(monkeypatch):
    monkeypatch.delenv("SERVINGSTUDIO_NO_GPU", raising=False)
    row = _neuron_ls_row()
    monkeypatch.setattr(
        neuron.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps([row]))
    )
    assert neuron.current_neuron_name() == neuron.TRAINIUM2_LNC2
    assert unsupported_device(NEURON_SUPPORT, "fixture") is None
    row["instance_type"] = "trn3.3xlarge"
    with pytest.raises(RuntimeError, match="unsupported Neuron target"):
        neuron.current_neuron_name()
    assert "Neuron is required" in unsupported_device(NEURON_SUPPORT, "fixture")


def test_no_gpu_blocks_neuron_discovery_before_subprocess(monkeypatch):
    monkeypatch.setenv("SERVINGSTUDIO_NO_GPU", "1")
    monkeypatch.setattr(neuron.subprocess, "run", lambda *a, **k: pytest.fail("device access"))
    with pytest.raises(GpuDisabledError):
        neuron.neuron_devices()


def test_default_pool_follows_the_backend_family():
    from profiling.exec import LocalGpuPool, get_default_pool

    assert isinstance(get_default_pool(), LocalGpuPool)
    assert isinstance(get_default_pool("neuron"), neuron.LocalNeuronPool)
    with pytest.raises(ValueError, match="unknown"):
        get_default_pool("tpu")


def test_the_sdk_env_names_a_pinned_vendor_image_that_is_never_built():
    image = images.image_for_env(ENV)
    assert image is not None and neuron._IMMUTABLE_IMAGE.fullmatch(image)
    assert image.startswith("public.ecr.aws/neuron/pytorch-inference-vllm-neuronx@sha256:")
    assert images.vendor_image_for_env(ENV).python == "/opt/conda/bin/python"
    assert not any(ENV in image.envs for image in images.IMAGES.values())
    assert images.vendor_image_for_env("vllm_env") is None


# --- chip reservation --------------------------------------------------------


def test_host_chip_allowlist_limits_the_default_pool(monkeypatch):
    """A shared host names its own chips; the pool must never reserve one outside them."""
    from profiling.exec import get_default_pool

    monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
    monkeypatch.delenv("NEURON_VISIBLE_DEVICES", raising=False)
    chips = [
        neuron.NeuronDevice(
            i, tuple(range(4 * i, 4 * i + 4)), 2, 96 * 1024**3, False, "trn2.48xlarge"
        )
        for i in range(16)
    ]
    monkeypatch.setattr(neuron, "neuron_devices", lambda: chips)
    monkeypatch.setenv(neuron.DEVICES_ENV, "8-10,15")
    pool = get_default_pool("neuron")
    assert [d.device_id for d in pool.idle_devices()] == [8, 9, 10, 15]
    monkeypatch.setenv(neuron.DEVICES_ENV, "9-8")
    with pytest.raises(ValueError, match="empty"):
        neuron.allowed_neuron_devices()
    monkeypatch.setenv(neuron.DEVICES_ENV, "x")
    with pytest.raises(ValueError, match="chip ID"):
        neuron.allowed_neuron_devices()


def test_busy_chip_is_never_reserved(monkeypatch):
    monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
    busy = neuron.NeuronDevice(0, (0, 1, 2, 3), 2, 96 * 1024**3, True, "trn2.3xlarge")
    monkeypatch.setattr(neuron, "neuron_devices", lambda: [busy])
    with pytest.raises(RuntimeError, match="no idle"):
        list(neuron.LocalNeuronPool().acquire_chunks(1))


@pytest.mark.parametrize("visibility", ["NEURON_RT_VISIBLE_CORES", "NEURON_VISIBLE_DEVICES"])
def test_pool_cannot_override_inherited_core_visibility(monkeypatch, visibility):
    monkeypatch.setenv(visibility, "2")
    monkeypatch.setattr(neuron, "neuron_devices", lambda: [])
    with pytest.raises(RuntimeError, match="owns allocation"):
        neuron.LocalNeuronPool().idle_devices()


def test_chip_reservation_is_shared_across_workspace_tmpdirs(monkeypatch, tmp_path):
    monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
    device = neuron.NeuronDevice(0, (0, 1, 2, 3), 2, 96 * 1024**3, False, "trn2.3xlarge")
    monkeypatch.setattr(neuron, "neuron_devices", lambda: [device])
    monkeypatch.setattr(neuron, "NEURON_LOCK_DIR", tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "workspace-a"))
    first = list(neuron.LocalNeuronPool().acquire_chunks(1))[0]
    try:
        monkeypatch.setenv("TMPDIR", str(tmp_path / "workspace-b"))
        with pytest.raises(RuntimeError, match="no idle"):
            list(neuron.LocalNeuronPool().acquire_chunks(1))
    finally:
        first.lock.close()
    second = list(neuron.LocalNeuronPool().acquire_chunks(1))[0]
    second.lock.close()


def test_tensor_parallel_reservation_exposes_a_contiguous_device_span():
    environment = neuron._reservation_environment(DEVICE, 4)
    assert environment["NEURON_VISIBLE_DEVICES"] == "12-15"
    assert "NEURON_RT_VISIBLE_CORES" not in environment
    assert neuron._reservation_environment(DEVICE)["NEURON_RT_VISIBLE_CORES"] == "12"
    with pytest.raises(ValueError, match="exceed"):
        neuron._reservation_environment(DEVICE, 5)


# --- container launch ----------------------------------------------------------


@pytest.fixture
def container_setup(tmp_path, monkeypatch):
    source = tmp_path / "request source"
    for name in ("profiling", "gpu"):
        (source / name).mkdir(parents=True)
    model = tmp_path / "model with spaces"
    model.mkdir()
    framework = tmp_path / "framework source"
    framework.mkdir()
    exchange = tmp_path / "exchange"
    exchange.mkdir()
    cache = tmp_path / "cache"
    tool = tmp_path / "neuron-ls"
    tool.write_text("identity tool fixture")
    address = tmp_path / "private.sock"
    # Binding a real socket is blocked in restricted test environments; only
    # the file-type observation is replaced.
    address.touch()
    is_socket = Path.is_socket
    monkeypatch.setattr(Path, "is_socket", lambda path: path == address or is_socket(path))
    monkeypatch.setattr(neuron, "_PROJECT_ROOT", source)
    monkeypatch.setattr(neuron.shutil, "which", lambda name: str(tool))
    monkeypatch.setenv(neuron.CACHE_DIR_ENV, str(cache))
    monkeypatch.setenv(neuron.DATA_DIRS_ENV, str(model))
    monkeypatch.setenv(neuron.EXTRA_SOURCE_DIRS_ENV, str(framework))
    nodes = [Path(f"/dev/neuron{index}") for index in range(4)]
    monkeypatch.setattr(neuron, "_neuron_device_nodes", lambda: nodes)
    monkeypatch.setattr(neuron, "_free_local_port", lambda: 63182)
    monkeypatch.setenv(neuron.DOCKER_HOST_ENV, f"unix://{address}")
    monkeypatch.setenv(neuron.DOCKER_COMMAND_ENV, "sudo -n docker")
    monkeypatch.delenv(neuron.TIMELINE_ENV, raising=False)
    monkeypatch.delenv("VIBESIM_PROFILE_RUNTIME", raising=False)
    monkeypatch.delenv("VIBESIM_PROFILE_IMAGES", raising=False)
    container = neuron.NeuronContainer(
        ("sudo", "-n", "docker"), f"unix://{address}", IMAGE, "/opt/image python/bin/python"
    )
    return SimpleNamespace(
        container=container,
        source=source,
        model=model,
        framework=framework,
        exchange=exchange,
        cache=cache,
        nodes=nodes,
    )


def worker_environment(command):
    return dict(command[i + 1].split("=", 1) for i, value in enumerate(command) if value == "--env")


def test_container_comes_from_the_vendor_image_and_host_transport(container_setup, monkeypatch):
    container = neuron.neuron_container(ENV)
    assert container.image == images.image_for_env(ENV)
    assert container.docker_command == ("sudo", "-n", "docker")
    assert container.python == images.vendor_image_for_env(ENV).python
    container.validate()  # The image's Python is never looked up on the host.
    monkeypatch.delenv(neuron.DOCKER_HOST_ENV)
    assert neuron.neuron_container(ENV).docker_host == neuron.DEFAULT_DOCKER_HOST


@pytest.mark.parametrize("runtime", ["host", "in-image"])
def test_neuron_backends_refuse_non_container_runtimes(container_setup, monkeypatch, runtime):
    monkeypatch.setenv("VIBESIM_PROFILE_RUNTIME", runtime)
    with pytest.raises(ValueError, match="vendor image"):
        neuron.neuron_container(ENV)


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
        neuron.NeuronContainer(("docker",), host, IMAGE, "/python")


def test_container_rejects_mutable_images_and_a_missing_socket(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="immutable"):
        neuron.NeuronContainer(("docker",), "unix:///run/docker.sock", "mutable:latest", "/py")
    monkeypatch.setattr(neuron.shutil, "which", lambda name: "/usr/bin/docker")
    missing = neuron.NeuronContainer(("docker",), f"unix://{tmp_path}/missing", IMAGE, "/py")
    with pytest.raises(FileNotFoundError, match="socket"):
        missing.validate()


def test_concurrent_workers_get_their_own_nki_artifacts_dir(container_setup):
    """neuronx-cc writes fixed names there and needs it clean per compile:
    a shared directory made parallel NKI compiles fail (NCC_EVRF059)."""
    setup = container_setup
    dirs = []
    for _ in range(2):
        command, _ = neuron._neuron_container_worker_command(
            setup.container, DEVICE, setup.exchange, image_id=IMAGE
        )
        dirs.append(worker_environment(command)["NKI_ARTIFACTS_DIR"])
    assert dirs[0] != dirs[1]
    cache = Path(worker_environment(command)[neuron.CACHE_DIR_ENV])
    assert all(Path(d).parent == cache / "artifacts" for d in dirs)


def test_neuron_argv_and_mounts_preserve_identity_and_image_abi(container_setup, monkeypatch):
    setup = container_setup
    monkeypatch.setenv("PJRT_DEVICE", "NEURON")
    monkeypatch.setenv("PYTHONPATH", "/host/site-packages")
    monkeypatch.setenv("DOCKER_CONTEXT", "remote")
    monkeypatch.setenv("DOCKER_TLS_VERIFY", "1")
    command, cli_env = neuron._neuron_container_worker_command(
        setup.container,
        DEVICE,
        setup.exchange,
        image_id=IMAGE,
        worker_env={
            # The reservation and the image ABI win over a row's worker_env.
            "SERVINGSTUDIO_NEURON_DEVICE": "99",
            "NEURON_RT_VISIBLE_CORES": "0-99",
            "PYTHONPATH": "/other/checkout",
            "LD_LIBRARY_PATH": "/host/libs",
        },
    )
    policy = worker_environment(command)
    assert command[:5] == ["sudo", "-n", "docker", "--host", setup.container.docker_host]
    # Every chip's node is visible (collectives need them); cores are the boundary.
    assert [value for value in command if value.startswith("--device=")] == [
        f"--device={node}:{node}" for node in setup.nodes
    ]
    assert "--gpus" not in command and "--privileged" not in command
    assert {"--pull=never", "--ipc=host", "--network=host", "--ulimit=core=0"} <= set(command)
    assert "--network=none" not in command
    # The image's vLLM ENTRYPOINT is cleared; its Python runs the worker.
    entry = command.index("--entrypoint")
    assert command[entry + 1] == ""
    assert command[-8:-6] == [IMAGE, "/opt/image python/bin/python"]
    assert policy["NEURON_RT_ROOT_COMM_ID"] == "127.0.0.1:63182"
    assert policy["NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS"] == "256"
    assert policy["SERVINGSTUDIO_NEURON_DEVICE"] == "3"
    assert policy["NEURON_RT_VISIBLE_CORES"] == "12"
    assert policy["NEURON_LOGICAL_NC_CONFIG"] == "2"
    assert policy["CUDA_VISIBLE_DEVICES"] == ""
    assert policy["PJRT_DEVICE"] == "NEURON"
    assert policy["PYTHONPATH"] == f"{setup.source}:{setup.framework}"
    assert policy[neuron.DATA_DIRS_ENV] == str(setup.model)
    assert "LD_LIBRARY_PATH" not in policy
    assert policy["TMPDIR"] == str(setup.cache / "tmp")
    assert policy["NEURON_COMPILE_CACHE_URL"] == str(setup.cache / "compile")
    assert policy["HF_HUB_OFFLINE"] == policy["TRANSFORMERS_OFFLINE"] == "1"
    for path in (setup.source / "profiling", setup.source / "gpu", setup.model, setup.framework):
        assert f"{path}:{path}:ro" in command
    assert f"{setup.cache}:{setup.cache}" in command
    assert f"{setup.exchange}:/io" in command
    assert not any("private.sock:" in value for value in command)
    assert "DOCKER_CONTEXT" not in cli_env and "DOCKER_TLS_VERIFY" not in cli_env
    assert "DOCKER_CONTEXT" in neuron.os.environ  # Only the subprocess is cleaned.
    assert command[-6:] == [
        "-m",
        "profiling.exec.local_worker",
        "--worker-input",
        "/io/input.json",
        "--worker-output",
        "/io/output.json",
    ]


@pytest.mark.parametrize(
    "problem",
    [
        "missing-source",
        "relative-source",
        "relative-data",
        "cache-overlap",
        "cache-under-source",
        "socket-exposure",
        "missing-device-node",
    ],
)
def test_neuron_path_failures(container_setup, monkeypatch, problem):
    setup = container_setup
    if problem == "missing-source":
        monkeypatch.setenv(neuron.EXTRA_SOURCE_DIRS_ENV, str(setup.source.parent / "absent"))
    elif problem == "relative-source":
        monkeypatch.setenv(neuron.EXTRA_SOURCE_DIRS_ENV, f"{setup.framework}:relative-src")
    elif problem == "relative-data":
        monkeypatch.setenv(neuron.DATA_DIRS_ENV, "relative-model")
    elif problem == "cache-under-source":
        monkeypatch.setenv(neuron.CACHE_DIR_ENV, str(setup.framework / "cache"))
    elif problem == "missing-device-node":
        monkeypatch.setattr(neuron, "_neuron_device_nodes", lambda: setup.nodes[:3])
    elif problem == "cache-overlap":
        monkeypatch.setenv(neuron.CACHE_DIR_ENV, str(setup.source / "profiling/cache"))
    else:
        setup.container = neuron.NeuronContainer(
            setup.container.docker_command,
            f"unix://{setup.cache}/private.sock",
            IMAGE,
            setup.container.python,
        )
    with pytest.raises((ValueError, FileNotFoundError)):
        neuron._neuron_container_worker_command(
            setup.container, DEVICE, setup.exchange, image_id=IMAGE
        )


def test_optional_mounts_and_row_policy(container_setup, monkeypatch):
    setup = container_setup
    monkeypatch.delenv(neuron.DATA_DIRS_ENV)
    monkeypatch.delenv(neuron.EXTRA_SOURCE_DIRS_ENV)
    command, _ = neuron._neuron_container_worker_command(
        setup.container,
        DEVICE,
        setup.exchange,
        image_id=IMAGE,
        worker_env={
            # A row may retune runtime defaults, never its collectives root.
            "NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS": "64",
            "NEURON_RT_ROOT_COMM_ID": "10.0.0.1:1",
        },
    )
    policy = worker_environment(command)
    assert policy["PYTHONPATH"] == str(setup.source)
    assert neuron.DATA_DIRS_ENV not in policy
    assert policy["NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS"] == "64"
    assert policy["NEURON_RT_ROOT_COMM_ID"] == "127.0.0.1:63182"
    assert not any(str(setup.model) in value for value in command)


def test_device_nodes_are_the_numbered_neuron_nodes(monkeypatch, tmp_path):
    names = ["neuron10", "neuron2", "neuron_monitor", "neuron0"]
    monkeypatch.setattr(
        neuron.Path, "glob", lambda self, pattern: iter(Path("/dev") / name for name in names)
    )
    assert neuron._neuron_device_nodes() == [
        Path("/dev/neuron0"),
        Path("/dev/neuron2"),
        Path("/dev/neuron10"),
    ]


def test_neuron_timeline_is_writable_without_source_or_data_writes(container_setup, monkeypatch):
    trace = container_setup.source.parent / "timeline outputs" / "run.jsonl"
    monkeypatch.setenv(neuron.TIMELINE_ENV, str(trace))
    command, _ = neuron._neuron_container_worker_command(
        container_setup.container, DEVICE, container_setup.exchange, image_id=IMAGE
    )
    assert worker_environment(command)[neuron.TIMELINE_ENV] == str(trace)
    assert f"{trace.parent}:{trace.parent}" in command


def test_image_inspection_binds_immutable_local_content(container_setup, monkeypatch):
    calls = []

    def inspect(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(
            returncode=0, stdout=json.dumps([{"Id": IMAGE, "RepoDigests": []}]), stderr=""
        )

    monkeypatch.setattr(neuron.subprocess, "run", inspect)
    assert neuron._inspect_neuron_image(container_setup.container) == IMAGE
    assert calls[0][-3:] == ["image", "inspect", IMAGE]
    other = "sha256:" + "b" * 64
    monkeypatch.setattr(
        neuron.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout=json.dumps([{"Id": other}])),
    )
    with pytest.raises(RuntimeError, match="declared immutable"):
        neuron._inspect_neuron_image(container_setup.container)


def test_digest_reference_inspection_resolves_the_exact_id(container_setup, monkeypatch):
    digest = "local/repository@sha256:" + "b" * 64
    container = neuron.NeuronContainer(
        ("docker",), container_setup.container.docker_host, digest, "/py"
    )
    monkeypatch.setattr(
        neuron.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=json.dumps([{"Id": IMAGE, "RepoDigests": [digest]}])
        ),
    )
    assert neuron._inspect_neuron_image(container) == IMAGE


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


# --- chunk lifecycle -----------------------------------------------------------


def test_rejected_chunk_releases_chip_reservation(monkeypatch, tmp_path):
    monkeypatch.delenv("SERVINGSTUDIO_NO_GPU", raising=False)
    from profiling.profilers import energy

    monkeypatch.setattr(energy, "energy_enabled", lambda: False)
    lock = (tmp_path / "reservation").open("a")
    chunk = neuron.LocalNeuronChunk(DEVICE, lock)
    with pytest.raises(ValueError, match="not a Neuron backend"):
        chunk.run("single_gemm", [{"backend": "torch"}])
    assert lock.closed


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
            subprocess_env=ENV,
            worker_env=(),
            neuron_logical_cores=1,
        ),
    )
    if outcome == "missing-image":

        def fail(container):
            raise RuntimeError("missing local image")

        monkeypatch.setattr(neuron, "_inspect_neuron_image", fail)
    else:
        monkeypatch.setattr(neuron, "_inspect_neuron_image", lambda container: IMAGE)
    monkeypatch.setattr(neuron, "_host_driver_version", lambda: "2.34.10")
    lock = (setup.source.parent / "reservation.lock").open("a")
    chunk = neuron.LocalNeuronChunk(DEVICE, lock)

    def run(command, **kwargs):
        assert not lock.closed
        cwd = Path(kwargs["cwd"])
        request = json.loads((cwd / "input.json").read_text())
        assert request["execution"] == neuron._reservation_payload(DEVICE)
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


# --- worker side ---------------------------------------------------------------


def _write_request(path: Path, **request) -> Path:
    path.write_text(json.dumps({"kernel_kind": "single_gemm", "energy": False, **request}))
    return path


@pytest.mark.parametrize("mismatch", ["device", "cores", "range", "lnc", "generation"])
def test_worker_reservation_mismatch_prevents_runner_loading(tmp_path, monkeypatch, mismatch):
    events = []
    profiler = SimpleNamespace(
        supports=SimpleNamespace(device_family="neuron"),
        load_list_runner=lambda: events.append("loaded"),
    )
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
        trn3 = neuron.NeuronDevice(3, DEVICE.core_ids, 2, DEVICE.memory_bytes, False, "trn3.48xl")
        monkeypatch.setattr(neuron, "neuron_devices", lambda: [trn3])
    request = _write_request(
        tmp_path / "request.json", specs=[{"backend": "test"}], execution=snapshot
    )
    with pytest.raises((ValueError, RuntimeError)):
        local_worker._worker_main(request, tmp_path / "output.json")
    assert events == []


def test_worker_observation_and_row_fingerprints_survive_container_boundary(tmp_path, monkeypatch):
    profiler = SimpleNamespace(
        supports=SimpleNamespace(device_family="neuron"),
        args_schema=SingleGemmArgs,
        load_list_runner=lambda: lambda specs: [RunnerResult(ComputeMetrics(1, 2, 3))],
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
    request = _write_request(
        tmp_path / "request.json",
        specs=[{"m": 2, "n": 2, "k": 2, "dtype": "bf16", "backend": "test"}],
        execution=neuron._reservation_payload(DEVICE),
    )
    local_worker._worker_main(request, tmp_path / "output.json")
    (row,) = json.loads((tmp_path / "output.json").read_text())["results"]
    assert row["gpu_name"] == DEVICE.gpu_name and row["cuda_version"] is None
    assert row["backend_version"] == "torch=2.9; checkpoint=abc; hlo=def"


@pytest.mark.parametrize("policy", [{"energy": True}, {"measure": {"duration_s": 1}}])
def test_neuron_worker_rejects_measure_and_energy_before_observation_or_runner(
    tmp_path, monkeypatch, policy
):
    profiler = SimpleNamespace(supports=SimpleNamespace(device_family="neuron"))
    monkeypatch.setattr(local_worker, "resolve_chunk_backend", lambda *a: "test")
    monkeypatch.setattr(local_worker, "find_kernel_profiler_spec", lambda *a: profiler)
    monkeypatch.setattr(local_worker, "set_energy_enabled", lambda enabled: None)
    monkeypatch.setattr(neuron, "neuron_devices", lambda: pytest.fail("device observation"))
    request = _write_request(tmp_path / "request.json", specs=[{"backend": "test"}], **policy)
    with pytest.raises(ValueError, match="energy or CUPTI"):
        local_worker._worker_main(request, tmp_path / "output.json")


def test_cuda_worker_does_not_observe_neuron_or_accept_a_reservation(tmp_path, monkeypatch):
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
    specs = [{"m": 2, "n": 2, "k": 2, "dtype": "bf16", "backend": "test"}]
    request = _write_request(tmp_path / "request.json", specs=specs)
    local_worker._worker_main(request, tmp_path / "output.json")
    (row,) = json.loads((tmp_path / "output.json").read_text())["results"]
    assert row["gpu_name"] == "NVIDIA H200" and row["cuda_version"] == "13.0"
    _write_request(request, specs=specs, execution=neuron._reservation_payload(DEVICE))
    with pytest.raises(ValueError, match="CUDA backend"):
        local_worker._worker_main(request, tmp_path / "output.json")


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
    assert "torch-neuronx=fixture-version" in versions["backend_version"]
    assert "aws-neuronx-runtime-lib=2.34.10" in versions["backend_version"]
    # The host driver is the controller's observation, never the image's.
    assert "aws-neuronx-dkms" not in versions["backend_version"]
    assert calls[0][0] == "dpkg-query"
