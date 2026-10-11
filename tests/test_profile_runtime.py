"""Which environment runs a profile env's worker: the image
`profiling/container/images.py` assigns it, or (`VIBESIM_PROFILE_RUNTIME=host`)
its host interpreter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from profiling.container import images
from profiling.exec.env import (
    ENV_REGISTRY,
    IMAGE_MANIFEST_ENV,
    PROFILE_RUNTIME_ENV,
    ContainerProfileEnv,
    ProfileEnv,
    container_image,
    resolve_profile_env,
)


def _inputs(root: Path) -> None:
    image = images.IMAGES["default"]
    for relative in (image.dockerfile, *image.inputs):
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_text(relative)


def test_an_image_tag_follows_its_dependency_inputs_only(tmp_path):
    _inputs(tmp_path)
    image = images.IMAGES["default"]
    tag = image.tag(tmp_path)
    assert tag.startswith("vibesim-profiler-default:")
    (tmp_path / "profiling" / "cli.py").parent.mkdir(exist_ok=True)
    (tmp_path / "profiling" / "cli.py").write_text("changed source")
    assert image.tag(tmp_path) == tag
    (tmp_path / "uv.lock").write_text("a new lock")
    assert image.tag(tmp_path) != tag


def test_every_imaged_env_is_a_registered_host_env():
    for image in images.IMAGES.values():
        for name in image.envs:
            assert isinstance(ENV_REGISTRY[name], ProfileEnv), name


def test_every_registered_env_has_an_image():
    for name in ENV_REGISTRY:
        assert images.image_for_env(name) is not None, name


def test_the_container_runtime_is_the_default(monkeypatch):
    monkeypatch.delenv(PROFILE_RUNTIME_ENV, raising=False)
    env = resolve_profile_env(None)
    assert isinstance(env, ContainerProfileEnv)
    assert env.name == "default_env"
    assert env.image == images.IMAGES["default"].tag()
    assert container_image("vllm_env") == images.IMAGES["vllm"].tag()


def test_the_host_runtime_runs_the_registered_interpreter(monkeypatch):
    monkeypatch.setenv(PROFILE_RUNTIME_ENV, "host")
    assert resolve_profile_env("vllm_env") is ENV_REGISTRY["vllm_env"]


def test_an_override_names_the_image(monkeypatch):
    monkeypatch.delenv(PROFILE_RUNTIME_ENV, raising=False)
    monkeypatch.setenv(images.IMAGES_ENV, '{"sglang_env": "vibesim-profiler-sglang:sent"}')
    assert resolve_profile_env("sglang_env").image == "vibesim-profiler-sglang:sent"
    monkeypatch.setenv(images.IMAGES_ENV, '["not", "a", "map"]')
    with pytest.raises(images.ImageError, match=images.IMAGES_ENV):
        resolve_profile_env("sglang_env")


def test_an_env_without_an_image_runs_only_on_the_host(monkeypatch):
    monkeypatch.setitem(ENV_REGISTRY, "local_env", ProfileEnv("local_env", Path("/bin/true")))
    monkeypatch.delenv(PROFILE_RUNTIME_ENV, raising=False)
    with pytest.raises(ValueError, match="local_env.*has no image"):
        resolve_profile_env("local_env")
    monkeypatch.setenv(PROFILE_RUNTIME_ENV, "host")
    assert resolve_profile_env("local_env").name == "local_env"


def test_a_missing_image_names_its_build_command(monkeypatch):
    monkeypatch.setattr(
        "profiling.exec.env.subprocess.run",
        lambda *a, **k: __import__("subprocess").CompletedProcess(a, 1, "", "No such image"),
    )
    monkeypatch.setattr("profiling.exec.env.shutil.which", lambda _: "/usr/bin/docker")
    with pytest.raises(FileNotFoundError, match="images build flashinfer-kda"):
        ContainerProfileEnv("flashinfer_kda_env", "vibesim-profiler-flashinfer-kda:0123").validate()


def test_an_unknown_runtime_is_rejected(monkeypatch):
    monkeypatch.setenv(PROFILE_RUNTIME_ENV, "docker")
    with pytest.raises(ValueError, match=PROFILE_RUNTIME_ENV):
        resolve_profile_env("default_env")


def test_the_selfcheck_needs_an_env_with_backends(capsys):
    from profiling.container import selfcheck

    assert selfcheck.main([]) == 2
    assert selfcheck.main(["no_such_env"]) == 1
    assert "no backend runs in no_such_env" in capsys.readouterr().err


def _manifest(tmp_path, monkeypatch, **fields):
    manifest = {
        "inputs": "abc123",
        "envs": ["vllm_env"],
        "python": "/opt/v/.venv/bin/python",
        "python_paths": ["/opt/extra"],
        **fields,
    }
    path = tmp_path / "vibesim-profiler-image.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setenv(IMAGE_MANIFEST_ENV, str(path))
    monkeypatch.setenv(PROFILE_RUNTIME_ENV, "in-image")
    monkeypatch.setenv(images.IMAGES_ENV, '{"vllm_env": "vibesim-profiler-vllm:abc123"}')


def test_in_image_runs_the_images_env_on_its_own_interpreter(tmp_path, monkeypatch):
    _manifest(tmp_path, monkeypatch)
    env = resolve_profile_env("vllm_env")
    assert env == ProfileEnv(
        "vllm_env",
        Path("/opt/v/.venv/bin/python"),
        additional_python_paths=(Path("/opt/extra"),),
        isolated_site_packages=True,
    )


def test_in_image_refuses_another_env_and_another_build(tmp_path, monkeypatch):
    _manifest(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="serves only"):
        resolve_profile_env(None)
    _manifest(tmp_path, monkeypatch, inputs="other")
    with pytest.raises(ValueError, match="built from inputs other"):
        resolve_profile_env("vllm_env")


def test_in_image_outside_an_image_says_so(tmp_path, monkeypatch):
    monkeypatch.setenv(PROFILE_RUNTIME_ENV, "in-image")
    monkeypatch.setenv(IMAGE_MANIFEST_ENV, str(tmp_path / "absent.json"))
    with pytest.raises(FileNotFoundError, match="does not run in a profiler image"):
        resolve_profile_env("vllm_env")
