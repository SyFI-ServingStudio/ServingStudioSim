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


def test_an_image_tag_names_a_non_x86_machine(tmp_path, monkeypatch):
    _inputs(tmp_path)
    image = images.IMAGES["default"]
    monkeypatch.setattr(images.platform, "machine", lambda: "x86_64")
    x86 = image.tag(tmp_path)
    monkeypatch.setattr(images.platform, "machine", lambda: "AMD64")
    assert image.tag(tmp_path) == x86
    monkeypatch.setattr(images.platform, "machine", lambda: "aarch64")
    assert image.tag(tmp_path) != x86


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


def test_an_image_builds_only_from_a_checkout_that_is_exactly_its_gitlink(tmp_path):
    import subprocess

    def git(where, *args):
        subprocess.run(
            ["git", "-C", str(where), "-c", "user.name=t", "-c", "user.email=t@t", *args],
            check=True,
            capture_output=True,
        )

    root, checkout = tmp_path / "root", tmp_path / "root" / "sub"
    checkout.mkdir(parents=True)
    git(checkout, "init", "-q")
    (checkout / ".gitignore").write_text("build/\n")
    git(checkout, "add", ".gitignore")
    git(checkout, "commit", "-qm", "c")
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    git(root, "init", "-q")
    git(root, "update-index", "--add", "--cacheinfo", f"160000,{head},sub")
    image = images.Image(name="x", dockerfile="x", envs=("e",), submodules=("sub",))

    (checkout / "__pycache__").mkdir()
    (checkout / "__pycache__" / "m.pyc").write_bytes(b"")
    images._require_pinned_checkouts(image, root)  # never copied into an image

    # An ignored build output and an untracked file are copied in all the same.
    for stray in ("build/_C.so", "notes.txt"):
        (checkout / stray).parent.mkdir(exist_ok=True)
        (checkout / stray).write_text("x")
        with pytest.raises(images.ImageError, match="untracked or ignored"):
            images._require_pinned_checkouts(image, root)
        (checkout / stray).unlink()


def test_a_superseded_image_tags_jit_cache_is_removed_once_stale(tmp_path, monkeypatch):
    import os
    import time

    from profiling.exec import local

    monkeypatch.setenv("VIBESIM_PROFILE_CACHE_DIR", str(tmp_path))
    old = time.time() - local.STALE_CACHE_SECONDS - 60
    for name in ("vibesim-profiler-vllm_old", "vibesim-profiler-vllm_recent", "other_old"):
        (tmp_path / name).mkdir()
    for name in ("vibesim-profiler-vllm_old", "other_old"):
        os.utime(tmp_path / name, (old, old))

    cache = local._image_cache_dir("vibesim-profiler-vllm:new")

    assert cache == tmp_path / "vibesim-profiler-vllm_new"
    # Only the same image's stale tags go; another image's cache is its own.
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "other_old",
        "vibesim-profiler-vllm_new",
        "vibesim-profiler-vllm_recent",
    ]
