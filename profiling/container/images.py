"""Profiler images: which image runs each profile env, and its tag.

A built image's tag is a hash of what decides its contents apart from the
worker source: the envs it serves, the Dockerfile, the files it copies before
the source, and the commit of each alignment checkout it installs (the
superproject's gitlink, so the tag is known without that checkout). Any checkout
with the same inputs names the same image, and a host that has the tag has the
image. Workers overlay the checkout's ``profiling/``, ``launcher/`` and ``gpu/``
(`exec/local.py`).

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


def _scm_version(checkout: Path) -> str:
    """The version setuptools-scm gives an untagged checkout, as a host install sees it.

    An image is built without the checkout's .git, where setuptools-scm cannot
    read it; neither fork carries upstream's tags.
    """

    count = _git(checkout, "rev-list", "--count", "HEAD")
    return f"0.1.dev{count}+g{_git(checkout, 'rev-parse', '--short=9', 'HEAD')}"


def _vllm_args(root: Path) -> dict[str, str]:
    """The precompiled-wheel base of the alignment fork (alignment/profiler/README.md)."""

    checkout = root / "alignment" / "profiler" / "vllm"
    first = _git(
        checkout,
        "log",
        "--reverse",
        "--format=%H",
        "--fixed-strings",
        "--grep=feat(alignment):",
        "HEAD",
    ).splitlines()
    if not first:
        raise ImageError("cannot resolve the vLLM fork's first alignment commit")
    base = _git(checkout, "rev-parse", f"{first[0]}^")
    native = [
        name
        for name in _git(checkout, "diff", "--name-only", f"{base}..HEAD").splitlines()
        if name.endswith((".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".rs"))
    ]
    if native:
        raise ImageError(f"the vLLM fork changes native code ({native[:3]}); build it from source")
    return {
        "VLLM_REVISION": _git(checkout, "rev-parse", "HEAD"),
        "VLLM_BASE_REVISION": base,
        "VLLM_VERSION": _scm_version(checkout),
    }


def _sglang_args(root: Path) -> dict[str, str]:
    checkout = root / "alignment" / "profiler" / "sglang"
    return {
        "SGLANG_REVISION": _git(checkout, "rev-parse", "HEAD"),
        "SGLANG_VERSION": _scm_version(checkout),
    }


@dataclass(frozen=True)
class Image:
    name: str
    dockerfile: str
    envs: tuple[str, ...]
    # Files the Dockerfile copies into its dependency layers, relative to the
    # checkout root (the build context).
    inputs: tuple[str, ...] = ()
    # Alignment checkouts the image installs; named by their gitlink.
    submodules: tuple[str, ...] = ()
    build_args: Callable[[Path], dict[str, str]] = field(default=lambda root: {})

    def tag(self, root: Path = REPO_ROOT) -> str:
        digest = hashlib.sha256(f"{' '.join(self.envs)}\0".encode())
        for relative in (self.dockerfile, *self.inputs):
            digest.update(relative.encode() + b"\0")
            digest.update((root / relative).read_bytes() + b"\0")
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
        Image(
            "vllm",
            "profiling/container/vllm.Dockerfile",
            ("vllm_env",),
            submodules=("alignment/profiler/vllm",),
            build_args=_vllm_args,
        ),
        Image(
            "sglang",
            "profiling/container/sglang.Dockerfile",
            ("sglang_env",),
            inputs=("profiling/container/sglang.Dockerfile.dockerignore",),
            submodules=("alignment/profiler/sglang",),
            build_args=_sglang_args,
        ),
        Image(
            "flashinfer-kda",
            "profiling/container/flashinfer_kda.Dockerfile",
            ("flashinfer_kda_env",),
            inputs=("profiling/container/requirements/flashinfer_kda.txt",),
        ),
    )
}


@dataclass(frozen=True)
class VendorImage:
    """An image this checkout runs but never builds: a vendor release, named by
    its immutable content digest, whose own interpreter runs the worker."""

    name: str
    image: str
    envs: tuple[str, ...]
    python: str


# A Neuron backend selects its SDK by naming one of these envs as its
# ``subprocess_env``. A later SDK is added beside the existing one, under a new
# env name, so rows measured on the old SDK keep their own image identity.
VENDOR_IMAGES = {
    image.name: image
    for image in (
        # Neuron SDK 2.33 with vLLM Neuron 0.29 (Python 3.14, torch 2.13,
        # `libtorch_neuronx_lite`, neuronx-cc 2.28, nki 0.7). Tag
        # 0.29.0.1.2.0-neuronx-py314-sdk2.33.0-ubuntu26.04. Its ENTRYPOINT is a
        # vLLM wrapper; `exec/neuron.py` overrides it.
        VendorImage(
            "neuron-sdk233",
            "public.ecr.aws/neuron/pytorch-inference-vllm-neuronx@sha256:"
            "b9a9015b75a0fe6a2e46cce4c355f032979f507054ba33716e10b945d971e49b",
            ("neuron_sdk233_env",),
            "/opt/conda/bin/python",
        ),
    )
}


def vendor_image_for_env(env_name: str) -> VendorImage | None:
    return next((image for image in VENDOR_IMAGES.values() if env_name in image.envs), None)


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


def image_for_env(env_name: str, root: Path = REPO_ROOT) -> str | None:
    """The image tag that runs ``env_name``, or None when no image does."""

    override = _overrides().get(env_name)
    if override:
        return override
    for image in IMAGES.values():
        if env_name in image.envs:
            return image.tag(root)
    vendor = vendor_image_for_env(env_name)
    return vendor.image if vendor is not None else None


# Directory names `.dockerignore` keeps out of every build context.
_DOCKER_IGNORED = frozenset({".git", ".venv", "__pycache__"})


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


def build(name: str, root: Path = REPO_ROOT) -> str:
    image = IMAGES[name]
    _require_pinned_checkouts(image, root)
    tag = image.tag(root)
    args = {
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
            for vendor in VENDOR_IMAGES.values():
                print(f"{vendor.image}  {' '.join(vendor.envs)}  (vendor, not built)")
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
