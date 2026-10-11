"""Profiler images: which image runs each profile env, and its tag.

A built image's tag is a hash of what decides its dependency layers -- the
Dockerfile and the files it copies before the source -- so any checkout that
has the same inputs names the same image, and a host that has the tag has the
image. The source the worker runs is not part of it: workers overlay the
checkout's ``profiling/``, ``launcher/`` and ``gpu/`` (`exec/local.py`).

``vllm_env``'s image is built by ``build.sh`` from the alignment vLLM checkout
and keeps the tag that script gives it.

    python -m profiling.container.images tag default
    python -m profiling.container.images build default
    python -m profiling.container.images env default_env
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Image:
    name: str
    dockerfile: str
    # Files the Dockerfile copies into its dependency layers, relative to the
    # checkout root (the build context).
    inputs: tuple[str, ...]
    envs: tuple[str, ...]

    def tag(self, root: Path = REPO_ROOT) -> str:
        digest = hashlib.sha256()
        for relative in (self.dockerfile, *self.inputs):
            digest.update(relative.encode() + b"\0")
            digest.update((root / relative).read_bytes() + b"\0")
        return f"vibesim-profiler-{self.name}:{digest.hexdigest()[:16]}"


IMAGES = {
    image.name: image
    for image in (
        Image(
            "default",
            "profiling/container/default.Dockerfile",
            ("pyproject.toml", "uv.lock", ".python-version"),
            ("default_env", "flashinfer_pip_env", "causal_conv1d_env"),
        ),
    )
}


def image_for_env(env_name: str, root: Path = REPO_ROOT) -> str | None:
    """The image tag that runs ``env_name``, or None when no image does."""

    for image in IMAGES.values():
        if env_name in image.envs:
            return image.tag(root)
    return None


def build(name: str, root: Path = REPO_ROOT) -> str:
    image = IMAGES[name]
    tag = image.tag(root)
    revision = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(
        [
            "docker",
            "build",
            "--file",
            str(root / image.dockerfile),
            "--build-arg",
            f"VIBESIM_REVISION={revision or 'unknown'}",
            "--build-arg",
            f"VIBESIM_IMAGE_INPUTS={tag.rpartition(':')[2]}",
            "--tag",
            tag,
            str(root),
        ],
        check=True,
    )
    return tag


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m profiling.container.images")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("tag", "build"):
        commands.add_parser(command).add_argument("image", choices=sorted(IMAGES))
    commands.add_parser("env").add_argument("env")
    args = parser.parse_args(argv)
    if args.command == "tag":
        print(IMAGES[args.image].tag())
    elif args.command == "build":
        print(build(args.image))
    else:
        tag = image_for_env(args.env)
        if tag is None:
            print(f"no image runs profile env {args.env!r}", file=sys.stderr)
            return 1
        print(tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
