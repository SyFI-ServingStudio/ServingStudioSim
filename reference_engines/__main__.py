"""`python -m reference_engines <verb> <engine>` (`just ref <verb> <engine>`).

materialize ENGINE [--dest DIR] [--force]
    BASE's upstream commit plus patches/ as a git tree (default
    reference_engines/ENGINE/src): one commit per patch, with a fixed
    committer, so the same inputs give the same commit ids everywhere.
export-patches ENGINE [--from CHECKOUT] [--ref REF]
    Rewrite patches/ from the commits CHECKOUT (default: the materialized
    tree) has on top of BASE's upstream commit.
bump-base ENGINE --commit SHA --image REPO:TAG [--platform-image P=REPO:TAG]
    Point BASE at a new upstream commit and official image, pinning each
    platform's digest from the registry, then materialize on it. Patches
    that no longer apply stop `git am` in the tree; resolve there, then
    export-patches.
build-image ENGINE
    Build the engine's image for this host's platform
    (`profiling/container/images.py`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from reference_engines import (
    ENGINES,
    Base,
    BaseError,
    base_path,
    format_base,
    parse_base,
    patch_files,
    patches_dir,
    read_base,
    source_dir,
)

CACHE_DIR = Path(
    os.environ.get("VIBESIM_REFERENCE_CACHE", Path.home() / ".cache" / "vibesim-reference-engines")
)
# `git am` records these as every materialized commit's committer, and the
# author date as its date, so a commit id depends only on BASE and patches/.
_COMMITTER = {
    "GIT_COMMITTER_NAME": "ServingStudio reference_engines",
    "GIT_COMMITTER_EMAIL": "reference-engines@servingstudio.invalid",
}
# The fork records its base in this file; BASE replaces it, so it is not a patch.
_NOT_PATCHED = ("ALIGNMENT_BASE",)
_BASE_HEADER = {
    "vllm": (
        "vLLM reference engine: the upstream commit patches/ apply to and its official image.\n"
        "Read by reference_engines (README.md); change it with `just ref bump-base vllm`."
    ),
    "sglang": (
        "SGLang reference engine: the upstream commit patches/ apply to and its official image.\n"
        "Read by reference_engines (README.md); change it with `just ref bump-base sglang`."
    ),
}


class RefError(RuntimeError):
    pass


def _git(*args: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        where = f" in {cwd}" if cwd else ""
        raise RefError(f"git {' '.join(args)}{where}: {(result.stderr or result.stdout).strip()}")
    return result.stdout.strip()


def _has_commit(repo: Path, commit: str) -> bool:
    return (
        subprocess.run(
            ["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def _object_cache(engine: str, base: Base) -> tuple[Path, str]:
    """A bare repository holding upstream base commits, and the ref naming this one.

    Fetched from upstream once per base; every materialized tree then fetches
    it from here.
    """

    cache = CACHE_DIR / f"{engine}.git"
    ref = f"refs/bases/{base.upstream_commit}"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        _git("init", "--bare", "--quiet", str(cache))
    if not _has_commit(cache, ref):
        print(f"fetching {base.upstream_commit[:12]} from {base.upstream_repo}", flush=True)
        _git(
            "-C",
            str(cache),
            "fetch",
            "--quiet",
            "--depth=1",
            base.upstream_repo,
            f"{base.upstream_commit}:{ref}",
        )
    return cache, ref


def materialize(engine: str, dest: Path, *, force: bool = False) -> str:
    base = read_base(engine)
    patches = patch_files(engine)
    if dest.exists():
        if not force:
            raise RefError(f"{dest} exists; pass --force to replace it")
        if (dest / ".git").exists():
            dirty = _git("status", "--porcelain", "--untracked-files=no", cwd=dest)
            if dirty:
                raise RefError(f"{dest} has uncommitted changes; export or discard them first")
        shutil.rmtree(dest)
    cache, ref = _object_cache(engine, base)
    dest.mkdir(parents=True)
    _git("init", "--quiet", cwd=dest)
    _git("fetch", "--quiet", "--depth=1", str(cache), ref, cwd=dest)
    _git("checkout", "--quiet", "--detach", base.upstream_commit, cwd=dest)
    if patches:
        result = subprocess.run(
            ["git", "am", "--quiet", "--committer-date-is-author-date", *map(str, patches)],
            cwd=dest,
            env={**os.environ, **_COMMITTER},
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RefError(
                f"a patch does not apply to {base.upstream_commit[:12]} "
                f"({(result.stdout + result.stderr).strip()}); resolve it in {dest} "
                "(git am --continue), then `export-patches`"
            )
    head = _git("rev-parse", "HEAD", cwd=dest)
    _git("switch", "--quiet", "-C", f"reference/{engine}", cwd=dest)
    print(f"{dest}: {base.upstream_commit[:12]} + {len(patches)} patches = {head}")
    return head


def export_patches(engine: str, checkout: Path, ref: str) -> list[Path]:
    base = read_base(engine)
    if not _has_commit(checkout, base.upstream_commit):
        raise RefError(f"{checkout} does not have the base commit {base.upstream_commit[:12]}")
    head = _git("rev-parse", ref, cwd=checkout)
    if subprocess.run(
        ["git", "-C", str(checkout), "merge-base", "--is-ancestor", base.upstream_commit, head],
        check=False,
    ).returncode:
        raise RefError(f"{ref} ({head[:12]}) is not based on {base.upstream_commit[:12]}")
    merges = _git("rev-list", "--merges", f"{base.upstream_commit}..{head}", cwd=checkout)
    if merges:
        raise RefError(f"{ref} has merge commits over the base; rebase it to a linear series")
    with tempfile.TemporaryDirectory() as scratch:
        _git(
            "format-patch",
            "--quiet",
            # Each file must depend on its own commit only: no "n/m" in the
            # subject (adding a patch would rewrite every file), full blob ids
            # (the abbreviation length follows the repository's size), no
            # source commit id or git-version signature.
            "--no-numbered",
            "--full-index",
            "--zero-commit",
            "--no-signature",
            "--output-directory",
            scratch,
            f"{base.upstream_commit}..{head}",
            "--",
            ".",
            *(f":(exclude){path}" for path in _NOT_PATCHED),
            cwd=checkout,
        )
        target = patches_dir(engine)
        target.mkdir(parents=True, exist_ok=True)
        for old in patch_files(engine):
            old.unlink()
        written = []
        for patch in sorted(Path(scratch).glob("*.patch")):
            written.append(Path(shutil.copy2(patch, target / patch.name)))
    print(f"{target}: {len(written)} patches from {base.upstream_commit[:12]}..{head[:12]}")
    return written


def _registry_get(url: str, accept: str) -> tuple[bytes, dict[str, str]]:
    """GET with the anonymous bearer token the registry's challenge names."""

    request = urllib.request.Request(url, headers={"Accept": accept})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        challenge = exc.headers.get("WWW-Authenticate", "")
        if exc.code != 401 or not challenge.lower().startswith("bearer "):
            raise
    fields = dict(
        part.split("=", 1) for part in challenge[len("Bearer ") :].replace('"', "").split(",")
    )
    realm = fields.pop("realm")
    with urllib.request.urlopen(
        f"{realm}?{urllib.parse.urlencode(fields)}", timeout=60
    ) as response:
        token = json.load(response)
    token = token.get("token") or token["access_token"]
    request = urllib.request.Request(
        url, headers={"Accept": accept, "Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read(), dict(response.headers)


_INDEX_TYPES = (
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
)
_MANIFEST_TYPES = (
    "application/vnd.oci.image.manifest.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
)


def platform_digests(image: str) -> dict[str, str]:
    """`repository:tag` -> {platform: digest of that platform's image manifest}."""

    name, _, tag = image.rpartition(":")
    registry, _, path = name.partition("/")
    if "." not in registry and ":" not in registry and registry != "localhost":
        registry, path = "registry-1.docker.io", name
    if registry in ("docker.io", "index.docker.io"):
        registry = "registry-1.docker.io"
    if registry == "registry-1.docker.io" and "/" not in path:
        path = f"library/{path}"
    body, headers = _registry_get(
        f"https://{registry}/v2/{path}/manifests/{tag}", ",".join(_INDEX_TYPES + _MANIFEST_TYPES)
    )
    document = json.loads(body)
    media = document.get("mediaType") or headers.get("Content-Type", "")
    if media in _INDEX_TYPES or "manifests" in document:
        return {
            f"{entry['platform']['os']}/{entry['platform']['architecture']}": entry["digest"]
            for entry in document["manifests"]
            if entry.get("platform", {}).get("os") not in (None, "unknown")
        }
    config, _ = _registry_get(
        f"https://{registry}/v2/{path}/blobs/{document['config']['digest']}", "*/*"
    )
    config = json.loads(config)
    return {
        f"{config['os']}/{config['architecture']}": "sha256:" + hashlib.sha256(body).hexdigest()
    }


def bump_base(engine: str, commit: str, image: str, platform_images: dict[str, str]) -> None:
    old = read_base(engine)
    digests = platform_digests(image)
    for name, other in platform_images.items():
        found = platform_digests(other)
        if name not in found:
            raise RefError(f"{other} is not published for {name} (only {', '.join(found)})")
        digests[name] = found[name]
    base = Base(old.upstream_repo, commit, image, digests, platform_images)
    # Validate before writing: a malformed BASE would stop every later command.
    text = format_base(base, _BASE_HEADER[engine])
    parse_base(text)
    base_path(engine).write_text(text, encoding="utf-8")
    print(f"{base_path(engine)}: {old.upstream_commit[:12]} -> {commit[:12]}, {image}")
    for name, digest in sorted(digests.items()):
        print(f"  {name} {digest}")
    materialize(engine, source_dir(engine), force=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m reference_engines", description=__doc__.split("\n\n")[0]
    )
    verbs = parser.add_subparsers(dest="verb", required=True)
    verb = verbs.add_parser("materialize", help="BASE + patches/ -> a git source tree")
    verb.add_argument("engine", choices=ENGINES)
    verb.add_argument("--dest", type=Path)
    verb.add_argument("--force", action="store_true", help="replace an existing clean tree")
    verb = verbs.add_parser("export-patches", help="a branch over BASE -> patches/")
    verb.add_argument("engine", choices=ENGINES)
    verb.add_argument("--from", dest="checkout", type=Path)
    verb.add_argument("--ref", default="HEAD")
    verb = verbs.add_parser("bump-base", help="move BASE to a new upstream commit and image")
    verb.add_argument("engine", choices=ENGINES)
    verb.add_argument("--commit", required=True)
    verb.add_argument("--image", required=True, help="the official image, repository:tag")
    verb.add_argument(
        "--platform-image",
        action="append",
        default=[],
        metavar="OS/ARCH=REPOSITORY:TAG",
        help="a platform published under another tag",
    )
    verb = verbs.add_parser("build-image", help="build the engine image for this platform")
    verb.add_argument("engine", choices=ENGINES)
    args = parser.parse_args(argv)
    try:
        if args.verb == "materialize":
            materialize(args.engine, args.dest or source_dir(args.engine), force=args.force)
        elif args.verb == "export-patches":
            export_patches(args.engine, args.checkout or source_dir(args.engine), args.ref)
        elif args.verb == "bump-base":
            platform_images = dict(item.split("=", 1) for item in args.platform_image)
            bump_base(args.engine, args.commit, args.image, platform_images)
        else:
            from profiling.container.images import build

            print(build(args.engine))
    except (RefError, BaseError, subprocess.CalledProcessError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
