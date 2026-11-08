"""Profiler images: which image runs each profile env, and its tag.

A built image's tag is a hash of what decides its contents apart from the
worker source: the envs it serves, the Dockerfile, the files it copies before
the source, the commit of each submodule it builds (the superproject's gitlink,
so the tag is known without that checkout) and the image it starts `FROM`. Any
checkout with the same inputs names the same image, and a host that has the tag
has the image. Workers overlay the checkout's ``profiling/``, ``launcher/`` and
``gpu/`` (`exec/local.py`).

The `vllm` and `sglang` images are the reference engines (`reference_engines/`):
the official image pinned by digest, with the engine's patches, serving both
L1 kernels and alignment captures.

``VIBESIM_PROFILE_IMAGES`` (a JSON object, env name -> tag) overrides the
computed tags; `--remote` uses it to hand a job the tags its sender verified,
since a shipped tree has no git metadata to compute them from.

    python -m profiling.container.images list
    python -m profiling.container.images build [NAME ...]
    python -m profiling.container.images env ENV
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGES_ENV = "VIBESIM_PROFILE_IMAGES"


class ImageError(RuntimeError):
    """An image cannot be named or built from this checkout."""


def gitlink(root: Path, path: str) -> str:
    """The commit the superproject pins ``path`` at."""

    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-s", "--", path],
        capture_output=True,
        text=True,
        check=False,
    )
    fields = result.stdout.split()
    if result.returncode != 0 or len(fields) < 2 or fields[0] != "160000":
        raise ImageError(f"{path} is not a submodule of {root}")
    return fields[1]


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise ImageError(f"git {' '.join(args)} in {root}: {result.stderr.strip()}")
    return result.stdout.strip()


def _engine_base(engine: str) -> Callable[[Path], str]:
    """The official image an engine image starts `FROM`, pinned for this host's platform."""

    def base(root: Path) -> str:
        from reference_engines import BaseError, host_platform, parse_base

        path = root / "reference_engines" / engine / "BASE"
        try:
            return parse_base(path.read_text(encoding="utf-8"), str(path)).image_for(
                host_platform()
            )
        except (OSError, BaseError) as exc:
            raise ImageError(str(exc)) from None

    return base


def _engine_args(engine: str) -> Callable[[Path], dict[str, str]]:
    def args(root: Path) -> dict[str, str]:
        from reference_engines import host_platform, parse_base

        base = parse_base((root / "reference_engines" / engine / "BASE").read_text())
        platform = host_platform()
        digest = base.digests[platform].removeprefix("sha256:")
        return {
            "VIBESIM_UPSTREAM_COMMIT": base.commit_for(platform),
            "VIBESIM_REQ_FRONTEND_REVISION": gitlink(root, _REQ_FRONTEND),
            # The engine half of the persistent-cache key (the official base),
            # read at run time by profiling/exec/local.py:image_cache_key.
            "VIBESIM_CACHE_KEY": f"{engine}-{digest[:12]}",
        }

    return args


_REQ_FRONTEND = "alignment/load_generator/req-frontend"


def _engine_image(
    engine: str, envs: tuple[str, ...], python: str, inputs: tuple[str, ...] = ()
) -> Image:
    directory = f"reference_engines/{engine}"
    return Image(
        engine,
        f"{directory}/Dockerfile",
        envs,
        inputs=(
            f"{directory}/BASE",
            f"{directory}/patches",
            "reference_engines/capture-requirements.txt",
            "reference_engines/autotune_guard.py",
            *inputs,
        ),
        submodules=(_REQ_FRONTEND,),
        build_args=_engine_args(engine),
        base_image=_engine_base(engine),
        python=python,
        capture_python=CAPTURE_PYTHON,
    )


CAPTURE_PYTHON = "/opt/vibesim-capture/.venv/bin/python"


@dataclass(frozen=True)
class Image:
    name: str
    dockerfile: str
    envs: tuple[str, ...]
    # The interpreter that runs a worker (`python -m profiling.exec.local_worker`).
    python: str = "/opt/venv/bin/python"
    # Files (or directories, every file under them) the Dockerfile copies into
    # its layers, relative to the checkout root (the build context).
    inputs: tuple[str, ...] = ()
    # Submodules the image builds; named by their gitlink.
    submodules: tuple[str, ...] = ()
    build_args: Callable[[Path], dict[str, str]] = field(default=lambda root: {})
    # A pinned `repository@sha256:...` the image is built `FROM`, passed as the
    # `BASE_IMAGE` build argument. It is an input of the tag.
    base_image: Callable[[Path], str] | None = None
    # The alignment capture runner's interpreter, in an image that runs captures.
    capture_python: str | None = None

    def tag(self, root: Path = REPO_ROOT) -> str:
        digest = hashlib.sha256(f"{' '.join(self.envs)}\0".encode())
        if self.base_image is not None:
            digest.update(f"FROM {self.base_image(root)}\0".encode())
        for relative in (self.dockerfile, *self.inputs):
            path = root / relative
            files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
            for file in files:
                digest.update(file.relative_to(root).as_posix().encode() + b"\0")
                digest.update(file.read_bytes() + b"\0")
        for path in self.submodules:
            digest.update(f"{path}\0{gitlink(root, path)}\0".encode())
        return f"vibesim-profiler-{self.name}:{digest.hexdigest()[:16]}"


IMAGES = {
    image.name: image
    for image in (
        Image(
            "default",
            "profiling/container/default.Dockerfile",
            ("default_env", "flashinfer_pip_env", "causal_conv1d_env"),
            inputs=("pyproject.toml", "uv.lock", ".python-version"),
        ),
        _engine_image("vllm", ("vllm_env",), "/usr/bin/python3"),
        _engine_image(
            "sglang",
            ("sglang_env",),
            "/opt/sglang/bin/python",
            inputs=("reference_engines/complete_flashinfer_aot.py",),
        ),
        Image(
            "flashinfer-kda",
            "profiling/container/flashinfer_kda.Dockerfile",
            ("flashinfer_kda_env",),
            inputs=("profiling/container/requirements/flashinfer_kda.txt",),
        ),
    )
}


def _overrides() -> dict[str, str]:
    raw = os.environ.get(IMAGES_ENV, "").strip()
    if not raw:
        return {}
    try:
        overrides = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ImageError(f"{IMAGES_ENV} is not JSON: {exc}") from None
    if not isinstance(overrides, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in overrides.items()
    ):
        raise ImageError(f"{IMAGES_ENV} must map profile env names to image tags")
    return overrides


def python_for_env(env_name: str) -> str | None:
    """The interpreter a worker for ``env_name`` runs on in its image, or None."""

    for image in IMAGES.values():
        if env_name in image.envs:
            return image.python
    return None


def image_for_env(env_name: str, root: Path = REPO_ROOT) -> str | None:
    """The image tag that runs ``env_name``, or None when no image does."""

    override = _overrides().get(env_name)
    if override:
        return override
    for image in IMAGES.values():
        if env_name in image.envs:
            return image.tag(root)
    return None


# Directory names `.dockerignore` keeps out of every build context.
_DOCKER_IGNORED = frozenset({".git", ".venv", "__pycache__", "target"})


def _require_pinned_checkouts(image: Image, root: Path) -> None:
    for path in image.submodules:
        checkout = root / path
        if not (checkout / ".git").exists():
            raise ImageError(f"{path} is not checked out: git submodule update --init {path}")
        head, pinned = _git(checkout, "rev-parse", "HEAD"), gitlink(root, path)
        if head != pinned:
            raise ImageError(f"{path} is at {head[:12]}, but the gitlink pins {pinned[:12]}")
        # Untracked and ignored files are copied in too (`.dockerignore` drops
        # only the entries below), and would change the image under one tag.
        listed = _git(checkout, "status", "--porcelain", "--ignored").splitlines()
        stray = [
            name
            for name in (line.split(maxsplit=1)[-1] for line in listed)
            if not _DOCKER_IGNORED.intersection(PurePosixPath(name.rstrip("/")).parts)
            and not name.endswith(".pyc")
        ]
        if stray:
            raise ImageError(
                f"{path} differs from its gitlink (changed, untracked or ignored files: "
                f"{', '.join(stray[:5])}); the image would not be it"
            )


def _image_exists(tag: str) -> bool:
    return (
        subprocess.run(
            ["docker", "image", "inspect", tag], capture_output=True, check=False
        ).returncode
        == 0
    )


def build(name: str, root: Path = REPO_ROOT) -> str:
    image = IMAGES[name]
    _require_pinned_checkouts(image, root)
    tag = image.tag(root)
    base_args = {}
    if image.base_image is not None:
        base = image.base_image(root)
        base_args = {"BASE_IMAGE": base, "VIBESIM_BASE_IMAGE": base}
    args = {
        **base_args,
        **image.build_args(root),
        "VIBESIM_PROFILE_ENVS": " ".join(image.envs),
        "VIBESIM_REVISION": _git(root, "rev-parse", "HEAD"),
        "VIBESIM_IMAGE_INPUTS": tag.rpartition(":")[2],
    }
    command = ["docker", "build", "--file", str(root / image.dockerfile), "--tag", tag]
    for key, value in args.items():
        command += ["--build-arg", f"{key}={value}"]
    subprocess.run([*command, str(root)], check=True)
    return tag


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m profiling.container.images")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list")
    commands.add_parser("tag").add_argument("image", choices=sorted(IMAGES))
    commands.add_parser("build", help="build the named images, or all of them").add_argument(
        "images", nargs="*", choices=sorted(IMAGES)
    )
    commands.add_parser("env").add_argument("env")
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            for image in IMAGES.values():
                print(f"{image.tag()}  {' '.join(image.envs)}")
        elif args.command == "tag":
            print(IMAGES[args.image].tag())
        elif args.command == "build":
            for name in args.images or IMAGES:
                print(build(name))
        else:
            tag = image_for_env(args.env)
            if tag is None:
                print(f"no image runs profile env {args.env!r}", file=sys.stderr)
                return 1
            print(tag)
    except (ImageError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
