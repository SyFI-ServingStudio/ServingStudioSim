"""The in-image alignment capture: its image tag, container paths, and rebinding."""

from __future__ import annotations

import json
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
def test_align_image_tag_names_its_base_and_load_generator(engine):
    name = image_runtime.ENGINE_IMAGES[engine].image
    align = images.IMAGES[name]
    assert align.base == engine
    assert align.envs == ()
    assert "alignment/load_generator/req-frontend" in align.submodules
    # A new engine image (another fork gitlink) is a new alignment image.
    other = images.Image(
        name,
        align.dockerfile,
        align.envs,
        align.inputs,
        align.submodules,
        align.build_args,
        base="default",
    )
    assert other.tag(REPO_ROOT) != align.tag(REPO_ROOT)
    assert image_runtime.image_tag(engine, REPO_ROOT) == align.tag(REPO_ROOT)
    assert align.tag(REPO_ROOT).startswith(f"vibesim-profiler-{name}:")


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
    assert document["fork_python"] == "/opt/sglang-venv/bin/python"
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
    assert "/opt/vibesim/alignment/profiler/vllm_server.py" in targets
    assert "/opt/vibesim/launcher" in targets
    for submodule in (
        "alignment/profiler/vllm",
        "alignment/profiler/sglang",
        "alignment/load_generator/req-frontend",
    ):
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
