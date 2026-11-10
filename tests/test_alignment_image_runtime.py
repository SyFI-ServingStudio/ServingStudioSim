"""The in-image alignment capture: its image tag, container paths, and rebinding."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from alignment import image_runtime
from launcher.alignment_config import load_profile_config
from profiling.container import images

REPO_ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path, **overrides) -> Path:
    hub = tmp_path / "hub" / "models--org--model"
    snapshot = hub / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (tmp_path / "trace.csv").write_text("x\n")
    corpus = tmp_path / "corpus" / "enwik9"
    corpus.parent.mkdir()
    corpus.write_text("text")
    fork = tmp_path / "venv" / "bin" / "python"
    fork.parent.mkdir(parents=True)
    fork.write_text("")
    document = {
        "schema_version": 1,
        "name": "case",
        "log_dir": "./profile_nsys",
        "gpu": "NVIDIA B200",
        "cuda_visible_devices": "2,3",
        "profile_kind": "nsys",
        "engine": "vllm",
        "fork_python": str(fork),
        "server": {
            "model_path": str(snapshot),
            "port": 21000,
            "tp_size": 2,
            "extra_args": ["--speculative-config", json.dumps({"model": str(snapshot)})],
        },
        "nsys": {"executable": "/opt/nvidia/nsight-systems/2025.5.2/bin/nsys"},
        "workload": {
            "frontend": {"type": "independent", "path": "./trace.csv"},
            "backend": {"type": "openai"},
            "text_file": str(corpus),
            "tokenizer": str(snapshot),
        },
        **overrides,
    }
    path = tmp_path / "profile_nsys.yaml"
    path.write_text(yaml.safe_dump(document))
    return path


@pytest.mark.parametrize("engine", ["vllm", "sglang"])
def test_engine_image_serves_l1_and_captures_from_its_pinned_base(engine, tmp_path, monkeypatch):
    image = images.IMAGES[image_runtime.ENGINE_IMAGES[engine].image]
    # One image per engine: the L1 env and the captures share it.
    assert image.envs == (f"{engine}_env",)
    assert image.capture_python == str(image_runtime.ENGINE_IMAGES[engine].capture_python)
    assert "alignment/load_generator/req-frontend" in image.submodules
    assert image.base_image(REPO_ROOT).split("@")[1].startswith("sha256:")
    assert image_runtime.image_tag(engine, REPO_ROOT) == image.tag(REPO_ROOT)
    assert image.tag(REPO_ROOT).startswith(f"vibesim-profiler-{engine}:")

    # The tag follows BASE and every patch: a copy of the inputs with one patch
    # line changed names another image.
    for relative in (image.dockerfile, *image.inputs):
        source, target = REPO_ROOT / relative, tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            target.write_bytes(source.read_bytes())
    monkeypatch.setattr(images, "gitlink", lambda root, path: "0" * 40)
    before = image.tag(tmp_path)
    patch = sorted((tmp_path / f"reference_engines/{engine}/patches").glob("*.patch"))[-1]
    patch.write_text(patch.read_text() + "\n")
    assert image.tag(tmp_path) != before


def test_container_config_uses_fixed_paths(tmp_path):
    config = _config(tmp_path)
    log_dir = tmp_path / "profile_nsys"
    document, mounts, info = image_runtime.container_config(config, log_dir)

    snapshot = "/models/models--org--model/snapshots/abc"
    assert document["log_dir"] == "/out"
    assert document["cuda_visible_devices"] == "0,1"
    assert info["host_devices"] == ["2", "3"]
    assert document["fork_python"] == str(image_runtime.ENGINE_IMAGES["vllm"].fork_python)
    assert document["nsys"]["executable"] == str(image_runtime.NSYS)
    assert document["server"]["model_path"] == snapshot
    assert json.loads(document["server"]["extra_args"][1]) == {"model": snapshot}
    assert document["workload"]["frontend"]["path"] == "/work/trace.csv"
    assert document["workload"]["text_file"] == "/work/text_corpus"
    assert document["workload"]["tokenizer"] == snapshot
    # The whole repo directory (its blobs/ too), read-only; outputs writable.
    entries = mounts.entries
    assert entries[str(tmp_path / "hub")] == "/models"
    assert entries[str(log_dir)] == "/out" and mounts.writable == {str(log_dir)}
    assert f"{log_dir}:/out:rw" in mounts.args()


def test_model_mount_follows_relative_links_into_a_shared_blob_store(tmp_path):
    # /scratch/hf layout: snapshot -> repo blobs/ -> <hub>/blobs/xx/<sha>.
    hub = tmp_path / "hub"
    snapshot = hub / "models--org--big" / "snapshots" / "s1"
    snapshot.mkdir(parents=True)
    (hub / "blobs" / "b3").mkdir(parents=True)
    (hub / "blobs" / "b3" / "b3ff").write_text("weights")
    (hub / "models--org--big" / "blobs").mkdir()
    (hub / "models--org--big" / "blobs" / "13bd").symlink_to("../../blobs/b3/b3ff")
    (snapshot / "model.safetensors").symlink_to("../../blobs/13bd")
    mounts = image_runtime.Mounts()
    assert image_runtime._model_mount(snapshot, mounts) == "/models/models--org--big/snapshots/s1"
    assert mounts.entries == {str(hub): "/models"}

    # An absolute link names a host path the container lacks; a dangling one fails later.
    (snapshot / "abs.safetensors").symlink_to(hub / "blobs" / "b3" / "b3ff")
    with pytest.raises(image_runtime.ImageRunError, match="absolute link"):
        image_runtime._model_mount(snapshot, image_runtime.Mounts())
    (snapshot / "abs.safetensors").unlink()
    (snapshot / "gone.safetensors").symlink_to("../../blobs/missing")
    with pytest.raises(image_runtime.ImageRunError, match="broken link|outside"):
        image_runtime._model_mount(snapshot, image_runtime.Mounts())


def test_container_config_serves_sglang_from_its_image_venv(tmp_path):
    runtime = {
        "packages": [{"name": "nvtx", "version": "0.2.16", "index_url": "https://pypi.org/simple"}],
        "environment": {"FLASHINFER_DISABLE_JIT": "1"},
    }
    config = _config(tmp_path, engine="sglang", python_runtime=runtime)
    document, _, _ = image_runtime.container_config(config, tmp_path / "out")
    assert document["fork_python"] == "/opt/sglang/bin/python"
    # The prebuilt-artifact contract travels unchanged; the image must satisfy it.
    assert document["python_runtime"] == runtime


def test_container_config_refuses_what_the_image_cannot_honour(tmp_path):
    (tmp_path / "libcuda.so.1").write_text("")
    with pytest.raises(image_runtime.ImageRunError, match="driver_compat_lib_dir"):
        image_runtime.container_config(
            _config(tmp_path, driver_compat_lib_dir=str(tmp_path)), tmp_path / "out"
        )


def test_overlay_keeps_the_images_own_submodules():
    targets = set(image_runtime.overlay_mounts(REPO_ROOT).entries.values())
    assert "/opt/vibesim/alignment/runner.py" in targets
    assert "/opt/vibesim/alignment/profiler" in targets
    assert "/opt/vibesim/launcher" in targets
    for submodule in ("alignment/load_generator/req-frontend",):
        path = f"/opt/vibesim/{submodule}"
        assert not any(target == path or path.startswith(target + "/") for target in targets)


def test_rebind_maps_container_paths_back_and_keeps_the_original(tmp_path):
    log_dir = tmp_path / "out"
    log_dir.mkdir()
    result = {
        "log_dir": "/out",
        "parsed_nsys": "/out/parsed.json",
        "drive_summary": {"source_trace": "/work/trace.csv", "other": "/outside"},
        "cuda_visible_devices": "0,1",
    }
    (log_dir / "profile_result.json").write_text(json.dumps(result))
    paths = {"/out": str(log_dir), "/work/trace.csv": str(tmp_path / "trace.csv")}
    rebound = image_runtime.rebind_result(log_dir, paths, ["2", "3"])
    assert rebound["log_dir"] == str(log_dir)
    assert rebound["parsed_nsys"] == f"{log_dir}/parsed.json"
    assert rebound["drive_summary"] == {
        "source_trace": str(tmp_path / "trace.csv"),
        "other": "/outside",
    }
    assert rebound["cuda_visible_devices"] == "2,3"
    assert json.loads((log_dir / "profile_result.container.json").read_text()) == result


def test_prewarm_block_is_validated(tmp_path):
    config = _config(tmp_path, prewarm={"prompt_lengths": [28, 60], "max_tokens": 4})
    assert load_profile_config(config).prewarm.prompt_lengths == [28, 60]
    with pytest.raises(ValueError, match="prompt_lengths"):
        load_profile_config(_config(tmp_path / "b", prewarm={"prompt_lengths": []}))


def test_session_runner_is_never_built_in_the_image(monkeypatch, tmp_path):
    from alignment.load_generator import runner

    monkeypatch.setenv(image_runtime.RUNTIME_ENV, image_runtime.IN_IMAGE)
    monkeypatch.setattr(runner, "SESSION_RUNNER", tmp_path / "missing")
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **k: pytest.fail("cargo ran"))
    with pytest.raises(FileNotFoundError, match="no req-frontend binary"):
        runner.build_session_runner()


def test_capture_facts_refuse_only_what_the_config_states(tmp_path):
    from alignment.profiler import capture_facts

    cfg = load_profile_config(_config(tmp_path))  # NVIDIA B200, tp 2
    gpus = [{"index": "2", "name": "NVIDIA B200"}, {"index": "3", "name": "NVIDIA B200"}]
    facts = {"gpus": gpus, "cuda_visible_devices": "2,3", "memlock": {"soft": 64}}
    assert capture_facts.check(cfg, facts) == []
    assert "names 1 visible" in capture_facts.check(cfg, facts | {"gpus": gpus[:1]})[0]
    other = [gpus[0], {"index": "3", "name": "NVIDIA H200"}]
    assert "NVIDIA H200" in capture_facts.check(cfg, facts | {"gpus": other})[0]


def test_capture_runs_the_capture_interpreter_explicitly(tmp_path):
    engine = image_runtime.ENGINE_IMAGES["vllm"]
    argv = image_runtime.docker_argv(
        "vibesim-profiler-vllm:test",
        image_runtime.Mounts(),
        capture_python=engine.capture_python,
        gpu_uuids=["GPU-a"],
        cache_dir=tmp_path,
        name="capture",
        rootless=True,
    )
    assert argv[argv.index("--entrypoint") + 1] == "/opt/vibesim-capture/.venv/bin/python"
    image = argv.index("vibesim-profiler-vllm:test")
    assert argv[image + 1 :] == ["-m", "launcher", "alignment", "profile", "/work/profile.yaml"]


def _with_server(path: Path, engine: str, devices: str, **server) -> Path:
    document = yaml.safe_load(path.read_text())
    document["engine"] = engine
    document["cuda_visible_devices"] = devices
    document["server"] = {**document["server"], "extra_args": [], **server}
    path.write_text(yaml.safe_dump(document))
    return path


@pytest.mark.parametrize(
    ("engine", "server", "gpus"),
    [
        ("vllm", {"tp_size": 2, "extra_args": ["--pipeline-parallel-size", "2"]}, 4),
        ("vllm", {"tp_size": 1, "dp_size": 2, "extra_args": ["-pp=2"]}, 4),
        ("vllm", {"tp_size": 2, "extra_args": ["--pipeline_parallel_size", "2"]}, 4),
        ("vllm", {"tp_size": 2, "extra_args": ["-pcp", "2"]}, 4),
        # Decode context parallelism reuses the TP GPUs.
        ("vllm", {"tp_size": 4, "extra_args": ["--decode-context-parallel-size", "2"]}, 4),
        ("sglang", {"tp_size": 2, "extra_args": ["--pp-size", "2"]}, 4),
        # Attention DP splits the TP group: no extra GPUs.
        ("sglang", {"tp_size": 4, "extra_args": ["--attn-dp-size", "4"]}, 4),
        ("sglang", {"tp_size": 1, "dp_size": 4}, 4),  # replicas
    ],
)
def test_the_gpu_count_is_tp_dp_and_pp(tmp_path, engine, server, gpus):
    from alignment.profiler import capture_facts

    devices = ",".join(str(index) for index in range(gpus))
    cfg = load_profile_config(
        _with_server(_config(tmp_path), engine, devices, **server), require_python_runtime=False
    )
    assert cfg.gpu_count == gpus
    assert cfg.replica_ranks == gpus // cfg.server.dp_size

    facts = {
        "gpus": [{"index": str(index), "name": "NVIDIA B200"} for index in range(gpus)],
        "cuda_visible_devices": devices,
    }
    assert capture_facts.check(cfg, facts) == []

    fewer = ",".join(str(index) for index in range(gpus - 1))
    with pytest.raises(ValueError, match=f"={gpus} devices"):
        load_profile_config(
            _with_server(_config(tmp_path / "x"), engine, fewer, **server),
            require_python_runtime=False,
        )


def test_an_expert_group_spans_one_pipeline_stage_including_prefill_cp(tmp_path):
    # vLLM's EP group is dp * pcp * tp: tp2 pcp2 holds an EP group of 4 ...
    path = _with_server(
        _config(tmp_path), "vllm", "0,1,2,3",
        tp_size=2, expert_parallel_size=4, expert_count_reduction_group_size=4,
        extra_args=["-pcp", "2"],
    )
    assert load_profile_config(path, require_python_runtime=False).expert_parallel_world == 4
    # ... while PP stages do not share one: tp2 pp2 does not.
    path = _with_server(
        _config(tmp_path / "pp"), "vllm", "0,1,2,3",
        tp_size=2, expert_parallel_size=4, expert_count_reduction_group_size=4,
        extra_args=["-pp", "2"],
    )
    with pytest.raises(ValueError, match="expert group can span"):
        load_profile_config(path, require_python_runtime=False)
    # SGLang's expert group lies inside one TP group: tp4 dp2 cannot hold ep8.
    path = _with_server(
        _config(tmp_path / "sgl"), "sglang", ",".join(map(str, range(8))),
        tp_size=4, dp_size=2, expert_parallel_size=8, expert_count_reduction_group_size=4,
    )
    with pytest.raises(ValueError, match="expert group can span"):
        load_profile_config(path, require_python_runtime=False)


def test_sglangs_deprecated_dp_attention_spelling_is_refused_with_the_new_one(tmp_path):
    path = _with_server(
        _config(tmp_path), "sglang", "0,1,2,3",
        tp_size=4, dp_size=4, extra_args=["--enable-dp-attention"],
    )
    with pytest.raises(ValueError, match="--attn-dp-size 4"):
        load_profile_config(path, require_python_runtime=False)
