"""`python -m reference_engines <verb> <engine>` (`just ref <verb> <engine>`).

materialize ENGINE [--dest DIR] [--force]
    BASE.yaml's upstream commit plus patches/ as a git tree (default
    reference_engines/ENGINE/src): one commit per patch, with a fixed
    committer, so the same inputs give the same commit ids everywhere.
export-patches ENGINE [--from CHECKOUT] [--ref REF]
    Rewrite patches/ from the commits CHECKOUT (default: the materialized
    tree) has on top of BASE.yaml's upstream commit.
bump-base ENGINE --commit SHA --image REPO:TAG [--platform-image P=REPO:TAG]
    [--platform-commit P=SHA] [--platform-note P=TEXT]
    Point BASE.yaml at a new upstream commit and official image, pinning each
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
# author date as its date, so a commit id depends only on BASE.yaml and patches/.
_COMMITTER = {
    "GIT_COMMITTER_NAME": "ServingStudio reference_engines",
    "GIT_COMMITTER_EMAIL": "reference-engines@servingstudio.invalid",
}
# The fork records its base in this file; BASE.yaml replaces it, so it is not a patch.
_NOT_PATCHED = ("ALIGNMENT_BASE",)
# How export-patches writes a commit, and over which paths. Each file must
# depend on its own commit only: no "n/m" in the subject (adding a patch would
# rewrite every file), full blob ids (the abbreviation length follows the
# repository's size), no source commit id or git-version signature.
_FORMAT_PATCH = ("format-patch", "--no-numbered", "--full-index", "--zero-commit", "--no-signature")
_EXPORTED_PATHS = ("--", ".", *(f":(exclude){path}" for path in _NOT_PATCHED))
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


def _patch_id(text: str) -> str:
    result = subprocess.run(
        ["git", "patch-id", "--stable"], input=text, capture_output=True, text=True, check=True
    )
    return result.stdout.split(" ", 1)[0]


def _unexported_commits(dest: Path, patches: list[Path]) -> list[str]:
    """Commits over the base, on HEAD or any local branch, that no file in
    `patches` holds: each commit is written as export-patches writes it (so a
    rename is a rename on both sides) and compared by `git patch-id`.

    The base is the tree's root: materialize fetches it at depth 1, so it is
    the one commit without parents, whatever BASE.yaml says now.
    """

    exported = {_patch_id(path.read_text()) for path in patches}
    roots = _git("rev-list", "--max-parents=0", "HEAD", "--branches", cwd=dest).split()
    commits = _git(
        "rev-list", "--no-merges", "HEAD", "--branches", *(f"^{root}" for root in roots), cwd=dest
    ).split()
    unexported = []
    for commit in commits:
        written = _git(*_FORMAT_PATCH, "--stdout", "-1", commit, *_EXPORTED_PATHS, cwd=dest)
        patch_id = _patch_id(written + "\n")  # empty when the commit touches only _NOT_PATCHED
        if patch_id and patch_id not in exported:
            unexported.append(commit)
    return unexported


def _check_replaceable(dest: Path, *, force: bool, patches: list[Path]) -> None:
    """Refuse to delete `dest` unless it holds nothing `patches` does not.

    Untracked and ignored files count: a host venv (`src/.venv`) or a new file
    of an unexported patch lives there. So do commits not exported to
    `patches`, a stash, and an unfinished `git am` (a patch that did not apply
    leaves the tree clean, its work in `.git/rebase-apply`).
    """

    if not dest.exists():
        return
    if not force:
        raise RefError(f"{dest} exists; pass --force to replace it")
    if not (dest / ".git").exists():
        if any(dest.iterdir()):
            raise RefError(f"{dest} is not a materialized tree; move it away first")
        return
    status = _git("status", "--porcelain", "--ignored", "--untracked-files=normal", cwd=dest)
    if status:
        shown = "\n  ".join(status.splitlines()[:10])
        raise RefError(
            f"{dest} holds work a replace would delete; export, commit, move or remove "
            f"it first:\n  {shown}"
        )
    git_dir = Path(_git("rev-parse", "--absolute-git-dir", cwd=dest))
    pending = [
        name
        for name in ("rebase-apply", "rebase-merge", "MERGE_HEAD", "CHERRY_PICK_HEAD")
        if (git_dir / name).exists()
    ]
    if pending:
        raise RefError(
            f"{dest} has an unfinished git operation ({', '.join(pending)}); finish it and "
            "`export-patches`, or abort it, first"
        )
    if (git_dir / "refs" / "stash").exists() or (git_dir / "logs" / "refs" / "stash").exists():
        raise RefError(f"{dest} has a stash a replace would delete; apply or drop it first")
    unexported = _unexported_commits(dest, patches)
    if unexported:
        shown = "\n  ".join(
            _git("log", "--no-walk", "--format=%h %s", *unexported[:10], cwd=dest).splitlines()
        )
        raise RefError(
            f"{dest} has commits not in patches/ a replace would delete; `export-patches` "
            f"them (or drop them) first:\n  {shown}"
        )


def materialize(engine: str, dest: Path, *, force: bool = False) -> str:
    base = read_base(engine)
    patches = patch_files(engine)
    _check_replaceable(dest, force=force, patches=patches)
    if dest.exists():
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
            *_FORMAT_PATCH,
            "--quiet",
            "--output-directory",
            scratch,
            f"{base.upstream_commit}..{head}",
            *_EXPORTED_PATHS,
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


def bump_base(
    engine: str,
    commit: str,
    image: str,
    platform_images: dict[str, str],
    platform_commits: dict[str, str],
    notes: dict[str, str] | None = None,
) -> None:
    old = read_base(engine)
    # Before anything is written: a tree materialize cannot replace would
    # leave BASE.yaml moved and the tree on the old base.
    _check_replaceable(source_dir(engine), force=True, patches=patch_files(engine))
    digests = platform_digests(image)
    for name, other in platform_images.items():
        found = platform_digests(other)
        if name not in found:
            raise RefError(f"{other} is not published for {name} (only {', '.join(found)})")
        digests[name] = found[name]
    # Notes explain the old pins; a bump keeps only the ones it is given.
    base = Base(
        old.upstream_repo, commit, image, digests, platform_images, platform_commits, notes or {}
    )
    # Validate before writing: a malformed BASE.yaml would stop every later command.
    text = format_base(base, _BASE_HEADER[engine])
    parse_base(text)
    base_path(engine).write_text(text, encoding="utf-8")
    print(f"{base_path(engine)}: {old.upstream_commit[:12]} -> {commit[:12]}, {image}")
    for name, digest in sorted(digests.items()):
        print(f"  {name} {digest}")
    materialize(engine, source_dir(engine), force=True)


def warm_caches(engine: str, config: Path, startup_timeout: float | None = None) -> int:
    """Compile and tune `config`'s deployment into the persistent caches.

    The same container a capture of `config` runs in, with autotuning allowed,
    stopped once the engine is ready; it then writes the manifest a capture's
    preflight requires (alignment/warm_caches.py). Run it on the GPU type the
    captures will use: the caches are keyed by its architecture.
    `startup_timeout` overrides the config's for this pass only.
    """
    from alignment.image_runtime import run_profile_in_image
    from launcher.alignment_config import load_profile_config

    cfg = load_profile_config(config)
    if cfg.engine != engine:
        raise RefError(f"{config} runs {cfg.engine}, not {engine}")
    return run_profile_in_image(config, warm_caches=True, startup_timeout=startup_timeout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m reference_engines", description=__doc__.split("\n\n")[0]
    )
    verbs = parser.add_subparsers(dest="verb", required=True)
    verb = verbs.add_parser("materialize", help="BASE.yaml + patches/ -> a git source tree")
    verb.add_argument("engine", choices=ENGINES)
    verb.add_argument("--dest", type=Path)
    verb.add_argument("--force", action="store_true", help="replace an existing clean tree")
    verb = verbs.add_parser("export-patches", help="a branch over BASE.yaml -> patches/")
    verb.add_argument("engine", choices=ENGINES)
    verb.add_argument("--from", dest="checkout", type=Path)
    verb.add_argument("--ref", default="HEAD")
    verb = verbs.add_parser("bump-base", help="move BASE.yaml to a new upstream commit and image")
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
    verb.add_argument(
        "--platform-commit",
        action="append",
        default=[],
        metavar="OS/ARCH=COMMIT",
        help="the upstream commit a --platform-image was built from, when not --commit",
    )
    verb.add_argument(
        "--platform-note",
        action="append",
        default=[],
        metavar="OS/ARCH=TEXT",
        help="why a platform is pinned the way it is (kept in BASE.yaml)",
    )
    verb = verbs.add_parser("build-image", help="build the engine image for this platform")
    verb.add_argument("engine", choices=ENGINES)
    verb = verbs.add_parser(
        "warm-caches",
        help="start the engine of a capture config to ready with autotuning allowed, "
        "then record the manifest its captures need",
    )
    verb.add_argument("engine", choices=ENGINES)
    verb.add_argument("config", type=Path, help="the capture's alignment profile YAML")
    verb.add_argument(
        "--startup-timeout",
        type=float,
        metavar="SECONDS",
        help="seconds the engine may take to reach ready in this pass (default: the "
        "config's server.startup_timeout); a warm-up compiles and tunes, so it is slower",
    )
    args = parser.parse_args(argv)
    try:
        if args.verb == "materialize":
            materialize(args.engine, args.dest or source_dir(args.engine), force=args.force)
        elif args.verb == "export-patches":
            export_patches(args.engine, args.checkout or source_dir(args.engine), args.ref)
        elif args.verb == "bump-base":
            platform_images = dict(item.split("=", 1) for item in args.platform_image)
            platform_commits = dict(item.split("=", 1) for item in args.platform_commit)
            notes = dict(item.split("=", 1) for item in args.platform_note)
            bump_base(
                args.engine, args.commit, args.image, platform_images, platform_commits, notes
            )
        elif args.verb == "warm-caches":
            return warm_caches(args.engine, args.config, args.startup_timeout)
        else:
            from profiling.container.images import build

            print(build(args.engine))
    except (RefError, BaseError, subprocess.CalledProcessError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
