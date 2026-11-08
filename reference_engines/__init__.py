"""The serving engines alignment captures and L1 kernels run: an official base plus our patches.

Each engine is a directory beside this file (`vllm/`, `sglang/`):

- `BASE`: the upstream commit the patches apply to and the official image built
  from it, pinned by one digest per platform (`read_base`).
- `patches/`: the instrumentation, one `git format-patch` file per commit. They
  are the source of truth; a fork branch is only where they are edited
  (`export-patches`).
- `Dockerfile`: the engine's one image, used by both L1 kernel workers and
  alignment captures. It starts `FROM` the official image at the host
  platform's digest and applies the patches to the engine as that image
  installed it (`profiling/container/images.py` names and builds it).

`python -m reference_engines` (or `just ref`) materializes a source tree,
exports patches from a branch, bumps the base, and builds the image; see
README.md.
"""

from __future__ import annotations

import platform
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENGINES = ("vllm", "sglang")

_SHA1 = re.compile(r"[0-9a-f]{40}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_PLATFORMS = {
    "x86_64": "linux/amd64",
    "amd64": "linux/amd64",
    "aarch64": "linux/arm64",
    "arm64": "linux/arm64",
}


class BaseError(ValueError):
    """An engine's BASE file is missing or malformed."""


@dataclass(frozen=True)
class Base:
    """One engine's upstream base: what its patches apply to and the image built from it."""

    upstream_repo: str
    upstream_commit: str
    # The official image, as published (`repository:tag`).
    image: str
    # Platform (`linux/amd64`) -> the digest of that platform's image manifest.
    digests: dict[str, str]
    # Platform -> another `repository:tag`, for a platform the official image
    # above is not published for. Its digest is in `digests`.
    platform_images: dict[str, str]
    # Platform -> the upstream commit that platform's image was built from,
    # when it is not `upstream_commit` (the patches must still apply to it).
    platform_commits: dict[str, str]

    def image_for(self, platform_name: str) -> str:
        """The pinned reference (`repository@sha256:...`) to build `FROM` on ``platform_name``."""

        digest = self.digests.get(platform_name)
        if digest is None:
            raise BaseError(
                f"the base names no image for {platform_name} "
                f"(it has {', '.join(sorted(self.digests))})"
            )
        image = self.platform_images.get(platform_name, self.image)
        return f"{_repository(image)}@{digest}"

    def commit_for(self, platform_name: str) -> str:
        """The upstream commit ``platform_name``'s image was built from."""

        return self.platform_commits.get(platform_name, self.upstream_commit)


def _repository(image: str) -> str:
    """`registry/repo:tag` -> `registry/repo` (a registry port is not a tag)."""

    name, _, tag = image.rpartition(":")
    return name if name and "/" not in tag else image


def host_platform() -> str:
    machine = platform.machine()
    try:
        return _PLATFORMS[machine]
    except KeyError:
        raise BaseError(f"no reference-engine image for machine {machine!r}") from None


def engine_dir(engine: str) -> Path:
    if engine not in ENGINES:
        raise BaseError(f"unknown engine {engine!r} (one of {', '.join(ENGINES)})")
    return ROOT / engine


def base_path(engine: str) -> Path:
    return engine_dir(engine) / "BASE"


def patches_dir(engine: str) -> Path:
    return engine_dir(engine) / "patches"


def source_dir(engine: str) -> Path:
    """Where `materialize` puts the engine's source tree by default (git-ignored)."""

    return engine_dir(engine) / "src"


def parse_base(text: str, origin: str = "BASE") -> Base:
    """`key=value` lines; blank lines and `#` comments are ignored.

    Required: `upstream_repo`, `upstream_commit` (40 hex), `image`
    (`repository:tag`), and at least one `digest.<os>/<arch>=sha256:<64 hex>`.
    Optional, for a platform published under another tag:
    `image.<os>/<arch>=repository:tag` and, when that image is of another
    upstream commit, `upstream_commit.<os>/<arch>=<40 hex>`.
    """

    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key or not value:
            raise BaseError(f"{origin}:{number}: expected key=value, got {raw!r}")
        if key in values:
            raise BaseError(f"{origin}:{number}: {key} given twice")
        values[key] = value
    for key in ("upstream_repo", "upstream_commit", "image"):
        if key not in values:
            raise BaseError(f"{origin}: missing {key}")
    if not _SHA1.fullmatch(values["upstream_commit"]):
        raise BaseError(f"{origin}: upstream_commit must be a full 40-hex commit")
    if ":" not in values["image"].rpartition("/")[2] or "@" in values["image"]:
        raise BaseError(f"{origin}: image must be repository:tag, without a digest")
    digests: dict[str, str] = {}
    platform_images: dict[str, str] = {}
    platform_commits: dict[str, str] = {}
    for key, value in values.items():
        if key.startswith("digest."):
            if not _DIGEST.fullmatch(value):
                raise BaseError(f"{origin}: {key} must be sha256:<64 hex>")
            digests[key.removeprefix("digest.")] = value
        elif key.startswith("image."):
            platform_images[key.removeprefix("image.")] = value
        elif key.startswith("upstream_commit."):
            if not _SHA1.fullmatch(value):
                raise BaseError(f"{origin}: {key} must be a full 40-hex commit")
            platform_commits[key.removeprefix("upstream_commit.")] = value
        elif key not in ("upstream_repo", "upstream_commit", "image"):
            raise BaseError(f"{origin}: unknown key {key}")
    if not digests:
        raise BaseError(f"{origin}: no digest.<os>/<arch> entry")
    for prefix, entries in (("image", platform_images), ("upstream_commit", platform_commits)):
        stray = set(entries) - set(digests)
        if stray:
            raise BaseError(f"{origin}: {prefix}.{sorted(stray)[0]} has no digest")
    return Base(
        upstream_repo=values["upstream_repo"],
        upstream_commit=values["upstream_commit"],
        image=values["image"],
        digests=digests,
        platform_images=platform_images,
        platform_commits=platform_commits,
    )


def read_base(engine: str) -> Base:
    path = base_path(engine)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise BaseError(f"{path} is missing") from None
    return parse_base(text, str(path))


def format_base(base: Base, header: str) -> str:
    lines = [f"# {line}" if line else "#" for line in header.splitlines()]
    lines += [
        f"upstream_repo={base.upstream_repo}",
        f"upstream_commit={base.upstream_commit}",
        f"image={base.image}",
    ]
    for name in sorted(base.digests):
        if name in base.platform_images:
            lines.append(f"image.{name}={base.platform_images[name]}")
        if name in base.platform_commits:
            lines.append(f"upstream_commit.{name}={base.platform_commits[name]}")
        lines.append(f"digest.{name}={base.digests[name]}")
    return "\n".join(lines) + "\n"


def patch_files(engine: str) -> list[Path]:
    return sorted(patches_dir(engine).glob("*.patch"))
