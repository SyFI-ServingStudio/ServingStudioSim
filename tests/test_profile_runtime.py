"""Which environment runs a profile env's worker: its host interpreter, or the
image `profiling/container/images.py` assigns it (`VIBESIM_PROFILE_RUNTIME`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from profiling.container import images
from profiling.exec.env import (
    ENV_REGISTRY,
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


def test_the_host_runtime_is_the_default(monkeypatch):
    monkeypatch.delenv(PROFILE_RUNTIME_ENV, raising=False)
    assert isinstance(resolve_profile_env("default_env"), ProfileEnv)


def test_the_container_runtime_runs_host_envs_in_their_image(monkeypatch):
    monkeypatch.setenv(PROFILE_RUNTIME_ENV, "container")
    env = resolve_profile_env(None)
    assert isinstance(env, ContainerProfileEnv)
    assert env.name == "default_env"
    assert env.image == images.IMAGES["default"].tag()
    assert resolve_profile_env("vllm_env") == ENV_REGISTRY["vllm_env"]
    assert container_image("vllm_env") == ENV_REGISTRY["vllm_env"].image


def test_the_container_runtime_refuses_an_env_without_an_image(monkeypatch):
    monkeypatch.setenv(PROFILE_RUNTIME_ENV, "container")
    with pytest.raises(ValueError, match="sglang_env.*has no image"):
        resolve_profile_env("sglang_env")


def test_an_unknown_runtime_is_rejected(monkeypatch):
    monkeypatch.setenv(PROFILE_RUNTIME_ENV, "docker")
    with pytest.raises(ValueError, match=PROFILE_RUNTIME_ENV):
        resolve_profile_env("default_env")
