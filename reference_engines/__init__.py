"""The serving engines alignment captures and L1 kernels run: an official base plus our patches.

Each engine is a directory beside this file (`vllm/`, `sglang/`):

- `BASE.yaml`: the upstream commit the patches apply to and the official image
  built from it, pinned by one digest per platform (`read_base`).
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

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
    # Platform -> why it is pinned the way it is (free text, kept in the file).
    notes: dict[str, str] = field(default_factory=dict)

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
    return engine_dir(engine) / "BASE.yaml"


def patches_dir(engine: str) -> Path:
    return engine_dir(engine) / "patches"


def source_dir(engine: str) -> Path:
    """Where `materialize` puts the engine's source tree by default (git-ignored)."""

    return engine_dir(engine) / "src"


_TOP_KEYS = {"upstream_repo", "upstream_commit", "image", "platforms"}
_PLATFORM_KEYS = {"digest", "image", "upstream_commit", "note"}


def _check_image(image: Any, where: str) -> None:
    # The digest lives in `platforms.<p>.digest`; one here too would build a
    # reference neither digest pins (`Base.reference`).
    if not isinstance(image, str) or ":" not in image.rpartition("/")[2] or "@" in image:
        raise BaseError(f"{where} must be repository:tag, without a digest")


def parse_base(text: str, origin: str = "BASE.yaml") -> Base:
    """A BASE.yaml document.

    Required: `upstream_repo`, `upstream_commit` (40 hex), `image`
    (`repository:tag`), and `platforms`, a mapping from `<os>/<arch>` to at
    least `digest: sha256:<64 hex>`. A platform published under another tag
    adds `image: repository:tag` and, when that image is of another upstream
    commit, `upstream_commit`. `note` is free text on why it is pinned so.
    """

    try:
        doc: Any = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise BaseError(f"{origin}: not YAML: {exc}") from None
    if not isinstance(doc, dict):
        raise BaseError(f"{origin}: expected a mapping")
    unknown = set(doc) - _TOP_KEYS
    if unknown:
        raise BaseError(f"{origin}: unknown key {sorted(unknown)[0]}")
    for key in _TOP_KEYS:
        if key not in doc:
            raise BaseError(f"{origin}: missing {key}")
    for key in ("upstream_repo", "upstream_commit", "image"):
        if not isinstance(doc[key], str):
            raise BaseError(f"{origin}: {key} must be a string")
    if not _SHA1.fullmatch(doc["upstream_commit"]):
        raise BaseError(f"{origin}: upstream_commit must be a full 40-hex commit")
    _check_image(doc["image"], f"{origin}: image")
    platforms = doc["platforms"]
    if not isinstance(platforms, dict) or not platforms:
        raise BaseError(f"{origin}: platforms must map <os>/<arch> to its pins")
    digests: dict[str, str] = {}
    platform_images: dict[str, str] = {}
    platform_commits: dict[str, str] = {}
    notes: dict[str, str] = {}
    for name, entry in platforms.items():
        where = f"{origin}: platforms.{name}"
        if not isinstance(name, str) or name.count("/") != 1:
            raise BaseError(f"{where}: a platform is <os>/<arch>")
        if not isinstance(entry, dict):
            raise BaseError(f"{where}: expected a mapping")
        unknown = set(entry) - _PLATFORM_KEYS
        if unknown:
            raise BaseError(f"{where}: unknown key {sorted(unknown)[0]}")
        if not isinstance(entry.get("digest"), str) or not _DIGEST.fullmatch(entry["digest"]):
            raise BaseError(f"{where}.digest must be sha256:<64 hex>")
        digests[name] = entry["digest"]
        if "image" in entry:
            _check_image(entry["image"], f"{where}.image")
            platform_images[name] = entry["image"]
        if "upstream_commit" in entry:
            if not _SHA1.fullmatch(str(entry["upstream_commit"])):
                raise BaseError(f"{where}.upstream_commit must be a full 40-hex commit")
            platform_commits[name] = str(entry["upstream_commit"])
        if "note" in entry:
            notes[name] = str(entry["note"]).strip()
    return Base(
        upstream_repo=doc["upstream_repo"],
        upstream_commit=doc["upstream_commit"],
        image=doc["image"],
        digests=digests,
        platform_images=platform_images,
        platform_commits=platform_commits,
        notes=notes,
    )


def read_base(engine: str) -> Base:
    path = base_path(engine)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise BaseError(f"{path} is missing") from None
    return parse_base(text, str(path))


def format_base(base: Base, header: str) -> str:
    platforms: dict[str, dict[str, str]] = {}
    for name in sorted(base.digests):
        entry: dict[str, str] = {}
        if name in base.notes:
            entry["note"] = base.notes[name]
        if name in base.platform_images:
            entry["image"] = base.platform_images[name]
        if name in base.platform_commits:
            entry["upstream_commit"] = base.platform_commits[name]
        entry["digest"] = base.digests[name]
        platforms[name] = entry
    doc = {
        "upstream_repo": base.upstream_repo,
        "upstream_commit": base.upstream_commit,
        "image": base.image,
        "platforms": platforms,
    }
    comment = "".join(f"# {line}\n" if line else "#\n" for line in header.splitlines())
    return comment + yaml.safe_dump(doc, sort_keys=False, width=float("inf"))


# What the image installs from a patch, per engine (the Dockerfile applies
# patches to the package as the official image installed it), and what a patch
# may also touch because no image runs it. Anything else, a native source
# above all, would build into an image that silently lacks it.
_INSTALLED = {"vllm": ("vllm/",), "sglang": ("python/sglang/",)}
_NOT_RUN = re.compile(r"(^|/)(tests?|docs)/|\.md$")
_RENAME_LINE = re.compile(r"(?:rename|copy) (?:from|to) (.+)")
_ESCAPES = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12, "r": 13, '"': 34, "\\": 92}


def _take_quoted(text: str) -> tuple[str, str]:
    """Split a leading C-quoted string (git's `core.quotePath` form) off
    `text`: (its decoded value, the rest after the closing quote)."""

    raw = bytearray()
    index = 1
    while index < len(text):
        char = text[index]
        if char == '"':
            return raw.decode("utf-8", "replace"), text[index + 1 :]
        if char == "\\" and index + 1 < len(text):
            following = text[index + 1]
            octal = text[index + 1 : index + 4]
            if len(octal) == 3 and all(digit in "01234567" for digit in octal):
                raw.append(int(octal, 8))
                index += 4
                continue
            raw.append(_ESCAPES.get(following, ord(following)))
            index += 2
            continue
        raw.extend(char.encode("utf-8"))
        index += 1
    raise ValueError(f"unterminated quoted path: {text!r}")


def _unquote(path: str) -> str:
    """A path as git wrote it: C-quoted ("a/\\303\\251.py") when unusual."""

    return _take_quoted(path)[0] if path.startswith('"') else path


def _header_paths(rest: str) -> tuple[str, ...]:
    """The two paths of `diff --git <rest>`, each `a/x` or `"a/..."`."""

    if rest.startswith('"'):
        old, tail = _take_quoted(rest)
        tail = tail.lstrip(" ")
        new = _take_quoted(tail)[0] if tail.startswith('"') else tail
    elif rest.startswith("a/"):
        if ' "b/' in rest:
            old, _, quoted = rest.partition(' "')
            new = _take_quoted('"' + quoted)[0]
        else:
            # `a/X b/X` when unrenamed: split in the middle, so a path that
            # holds " b/" itself is read whole; otherwise at " b/".
            half = (len(rest) - 1) // 2
            if rest[half : half + 3] == " b/" and rest[2:half] == rest[half + 3 :]:
                old, new = rest[:half], "b/" + rest[half + 3 :]
            else:
                old, _, tail = rest.partition(" b/")
                new = "b/" + tail
    else:
        return ()
    return tuple(path[2:] for path in (old, new) if path[:2] in ("a/", "b/"))


def patch_paths(text: str) -> list[str]:
    """Every path a patch touches, both sides of a rename or copy.

    Read from each file's header only (`diff --git` up to its first hunk or
    binary payload), so a body line that happens to read `--- a/x` is not
    taken for one.
    """

    paths: dict[str, None] = {}
    in_header = False
    for line in text.splitlines():
        if line.startswith("diff --git "):
            in_header = True
            paths.update(dict.fromkeys(_header_paths(line[len("diff --git ") :])))
            continue
        if line.startswith(("@@", "GIT binary patch", "Binary files ")):
            in_header = False
        if not in_header:
            continue
        if line.startswith(("--- ", "+++ ")):
            path = _unquote(line[4:].rstrip("\t"))
            if path != "/dev/null":
                paths[path[2:] if path.startswith(("a/", "b/")) else path] = None
        elif match := _RENAME_LINE.fullmatch(line):
            paths[_unquote(match.group(1))] = None
    return list(paths)


def patched_paths_not_installed(engine: str) -> list[str]:
    """Paths the patches change that the engine's image would not install."""

    stray = []
    for patch in patch_files(engine):
        # Both sides: a rename or copy into a native source counts too.
        for path in patch_paths(patch.read_text(encoding="utf-8", errors="replace")):
            if not path.startswith(_INSTALLED[engine]) and not _NOT_RUN.search(path):
                stray.append(f"{patch.name}: {path}")
    return stray


def patch_files(engine: str) -> list[Path]:
    return sorted(patches_dir(engine).glob("*.patch"))
