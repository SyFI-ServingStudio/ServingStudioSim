"""`bundle` / `verify-bundle`: hand a campaign's captures to someone else and back.

A contributor runs the profile passes of a pack on their own machine, then
`bundle` collects each case's capture directories into one publishable tree:

    <dest>/bundle.json                         what is inside: files, sha256, sizes, states
    <dest>/<case_slug>/<pass_name>/...         the pass directory, scrubbed
    <dest>/<case_slug>/<pass_name>.trace.csv   the trace that pass replayed

The maintainer runs `verify-bundle` against their own rendered campaign. Every
file is re-hashed; every pass name must be one of the case's profile passes;
the capture conditions (engine image, model snapshot, replayed trace, GPU) are
read again from the hashed files, never from `bundle.json`, and must equal what
this checkout renders. Only then is each complete pass placed, atomically, into
its case directory. The remaining phases (timing_predict, analysis_kernel,
simulation, analysis_e2e) run locally with `alignment-campaign run`, so every
number is recomputed from the raw capture.

**Scrubbing.** A bundle may be public, and host paths are machine facts, not
measurement. In text files:

- every host path is replaced by a placeholder: the capture dir
  (`${capture_dir}`), case dir, out root, checkout and its workspace, home, each
  host-profile path and checkpoint (`${host.<key>}`), and every mount source the
  capture recorded in `container_run.json` (`${mount:<container path>}`);
- the user name and the host name (short and FQDN) are replaced as whole
  words when they have at least `MIN_BARE_TOKEN` characters; a shorter name is
  not replaced (a user `vllm` would rewrite the engine's `vllm/` directory);
  each GPU UUID becomes `GPU-<n>`.

Binary files (`.nsys-rep`, parquet) cannot be rewritten, so afterwards every
file is searched for the home, every non-shared host path, and the user and
host name of at least `MIN_BARE_TOKEN` characters anywhere (also inside a
longer word such as `/raid/<user>_data`, which the replacement leaves); any
hit refuses the bundle. A hit in a binary is a leak to fix where
the capture writes it (as `container_cache_path` did for the autotune preflight
variable). The search sees only plain text, so in a compressed `.nsys-rep`
section it is a backstop, not a proof.

GPU UUIDs are replaced in text but not refused: nsys stores each device's bare
UUID in the `.nsys-rep`, and blocking it would refuse every capture. A
published `.nsys-rep` therefore names the GPUs it ran on (hardware serials, not
a person or a path).

`verify-bundle` restores `${capture_dir}`, `${case_dir}`, `${out_root}`, `${repo}`
and the replayed trace's mount to its own paths, which makes the capture's own
`log_dir` record match where it now lives
(`launcher/alignment.py:_load_profile_result`).

Not bundled: the nsys `.sqlite` export (several GB; `alignment profile
--resume` re-exports it from the `.nsys-rep`) and the `.warm-caches` sibling.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

import yaml

from .execute import _artifacts_for as execute_artifacts
from .execute import phase_complete
from .pack import REPO_ROOT, Case, HostProfile, Pack, PackError

BUNDLE_SCHEMA_VERSION = 1
BUNDLE_MANIFEST = "bundle.json"
TRACE_SUFFIX = ".trace.csv"

#: Suffixes rewritten as UTF-8 text. Everything else is copied byte for byte
#: and only searched.
TEXT_SUFFIXES = frozenset({".json", ".jsonl", ".yaml", ".yml", ".log", ".txt", ".csv", ".md"})
#: Never bundled: regenerable from what is bundled.
EXCLUDED_SUFFIXES = frozenset({".sqlite", ".sqlite-journal"})
#: Names shorter than this (user, host) are neither replaced nor searched, so a
#: user called `vllm` neither rewrites `"engine": "vllm"` nor refuses every
#: capture; the paths that would name them are covered as paths.
MIN_BARE_TOKEN = 5
#: Written last when placing a pass, so an interrupted copy never looks complete.
COMPLETE_MARKER = ".complete"
#: The container path the load generator reads the trace from (image runtime).
TRACE_MOUNT = "/work/trace.csv"
_CHUNK = 64 << 20
_REVISION = re.compile(r"[0-9a-f]{40}|refs/pr/\d+")


class BundleError(PackError):
    """A bundle that must not be published or placed. The message names why."""


# ── scrubbing ────────────────────────────────────────────────────────────────


def _absolute(path: str | Path) -> str:
    text = str(path)
    if not text.startswith("/"):
        text = str(REPO_ROOT / text)
    return os.path.normpath(text)


@dataclass
class Scrubber:
    """Literal path replacements, name replacements and the blocklist."""

    paths: list[tuple[str, str]] = field(default_factory=list)
    names: list[tuple[str, str]] = field(default_factory=list)
    blocklist: list[str] = field(default_factory=list)

    def copy(self) -> "Scrubber":
        return Scrubber(list(self.paths), list(self.names), list(self.blocklist))

    def add_path(self, path: str | Path | None, placeholder: str, *, block: bool = True) -> None:
        if not path:
            return
        text = _absolute(path)
        variants = {text}
        try:
            variants.add(str(Path(text).resolve()))
        except OSError:
            pass
        for variant in variants:
            if variant.count("/") < 2:  # never `/` or a top-level directory
                continue
            self.paths.append((variant, placeholder))
            if block:
                self.blocklist.append(variant)

    def add_name(self, name: str | None, placeholder: str) -> None:
        if name:
            self.names.append((name, placeholder))

    def scrub(self, text: str) -> str:
        # Longest first, so a capture dir is replaced before the out root that
        # contains it; the first placeholder registered for a path wins.
        chosen: dict[str, str] = {}
        for old, new in self.paths:
            chosen.setdefault(old, new)
        for old, new in sorted(chosen.items(), key=lambda item: -len(item[0])):
            text = re.sub(re.escape(old) + r"(?![\w.-])", lambda _m, new=new: new, text)
        for name, placeholder in self.names:
            pattern = _name_pattern(name)
            if pattern is not None:
                text = pattern.sub(lambda match, p=placeholder: match.group(1) + p, text)
        return text

    def pattern(self) -> re.Pattern[bytes]:
        """Paths as raw substrings, names as `_name_blocked` says."""
        alternatives = [re.escape(token.encode())
                        for token in sorted(set(self.blocklist), key=len, reverse=True)]
        alternatives += [blocked for name, _ in self.names if (blocked := _name_blocked(name))]
        return re.compile(b"|".join(alternatives)) if alternatives else re.compile(b"(?!)")


def _name_pattern(name: str) -> re.Pattern[str] | None:
    """A user or host name as a whole word; group 1 is the (empty) prefix kept
    in front of the placeholder. None for a short name, which is never
    replaced: `vllm` would rewrite the engine's own `vllm/` directory."""
    if len(name) < MIN_BARE_TOKEN:
        return None
    return re.compile(r"(^|(?<=[^\w.-]))" + re.escape(name) + r"(?![\w-])")


def _name_blocked(name: str) -> bytes | None:
    """What refuses a bundle: a long name anywhere, even inside a longer word
    (`/raid/<user>_data`), so what the replacement misses is refused rather
    than published. A short name is not searched: `vllm` would refuse every
    vLLM capture; its paths are covered as paths (home, workspace, mounts)."""
    if len(name) < MIN_BARE_TOKEN:
        return None
    return re.escape(name.encode())


def find_blocked(path: Path, pattern: re.Pattern[bytes]) -> str | None:
    """The first blocklisted string in `path`, or None. Reads in chunks with an
    overlap, so a token across a chunk boundary is still found."""
    overlap = 4096
    tail = b""
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                return None
            match = pattern.search(tail + chunk)
            if match:
                return match.group(0).decode(errors="replace")
            tail = chunk[-overlap:]


def _gpu_uuids(pass_dir: Path) -> list[str]:
    found: list[str] = []
    for name in ("capture_facts.json", "container_run.json"):
        path = pass_dir / name
        if path.is_file():
            found += re.findall(r"GPU-[0-9a-fA-F-]{36}", path.read_text(errors="replace"))
    return list(dict.fromkeys(found))


def _shared(path: str, host: HostProfile) -> bool:
    """A user-neutral install or cache path: replaced for portability, not
    refused (an image may hold nsys at the same path; a cluster may inject the
    hub root into every container)."""
    shared_roots = [host.hf_hub_root, host.nsys_executable]
    return any(root and (path == _absolute(root) or path.startswith(_absolute(root) + "/"))
               for root in shared_roots)


def host_scrubber(host: HostProfile, out_root: Path) -> Scrubber:
    """Replacements for this machine. Pass, case and mount paths are added per pass."""
    scrubber = Scrubber()
    scrubber.add_path(out_root, "${out_root}")
    scrubber.add_path(REPO_ROOT, "${repo}")
    # The workspace that holds the checkout (sibling worktrees, logs/, caches),
    # when it is the user's own: a checkout under `/opt` must not block `/opt`.
    workspace = REPO_ROOT.parent
    if getpass.getuser() in workspace.parts or Path.home() in (workspace, *workspace.parents):
        scrubber.add_path(workspace, "${workspace}")
    for key in ("text_corpus", "fork_python", "nsys_executable", "hf_hub_root"):
        value = getattr(host, key)
        if value:
            scrubber.add_path(value, "${host.%s}" % key, block=not _shared(_absolute(value), host))
    for key in host.checkpoints:
        # A checkpoint in the hub keeps `models--<org>--<name>/snapshots/<sha>`
        # behind the hub root's placeholder: the snapshot is a checked condition.
        path = host.checkpoint_path(key)
        if not _shared(_absolute(path), host):
            scrubber.add_path(path, "${host.checkpoints.%s}" % key)
    scrubber.add_path(Path.home(), "${home}")
    scrubber.add_name(getpass.getuser(), "${user}")
    for name in dict.fromkeys((socket.getfqdn(), socket.gethostname())):
        if name and name not in ("localhost", "localhost.localdomain"):
            scrubber.add_name(name, "${hostname}")
    return scrubber


def _mounts(pass_dir: Path) -> dict[str, str]:
    """Container path -> host path, as the capture recorded its mounts."""
    paths = _read_json(pass_dir / "container_run.json").get("paths")
    return {str(k): str(v) for k, v in paths.items()} if isinstance(paths, dict) else {}


def pass_scrubber(base: Scrubber, host: HostProfile, pass_dir: Path, case_dir: Path) -> Scrubber:
    scrubber = base.copy()
    # Registered first: the same string may also be a mount source (`/out`).
    scrubber.add_path(pass_dir, "${capture_dir}")
    # Where the capture was written, if it has moved since (a copied campaign,
    # or a pass dir linked in from another run root).
    recorded = _read_json(pass_dir / "profile_result.json").get("log_dir")
    if isinstance(recorded, str):
        scrubber.add_path(recorded, "${capture_dir}")
    scrubber.add_path(case_dir, "${case_dir}")
    for container, source in _mounts(pass_dir).items():
        scrubber.add_path(source, "${mount:%s}" % container,
                          block=not _shared(_absolute(source), host))
    for index, uuid in enumerate(_gpu_uuids(pass_dir)):
        # Replaced, not refused (see the module doc). Bare form too: some
        # records drop the `GPU-` prefix.
        scrubber.paths.append((uuid, f"GPU-{index}"))
        scrubber.paths.append((uuid.removeprefix("GPU-"), f"GPU-{index}"))
    return scrubber


# ── conditions ───────────────────────────────────────────────────────────────


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return document if isinstance(document, dict) else {}


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        document = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return {}
    return document if isinstance(document, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(model_path: Any) -> str | None:
    if not isinstance(model_path, str):
        return None
    match = re.search(r"/snapshots/([0-9a-f]{40})(?:/|$)", model_path)
    return match.group(1) if match else None


def capture_conditions(pass_dir: Path, trace: Path | None) -> dict[str, Any]:
    """The facts `verify-bundle` checks, read from a capture's own records.
    The same function reads the contributor's capture at bundle time and the
    hashed bundle at verify time, so a hand-edited summary cannot pass."""
    result = _read_json(pass_dir / "profile_result.json")
    config = _read_yaml(pass_dir / "profile.yaml")
    facts = _read_json(pass_dir / "capture_facts.json")
    gpus = facts.get("gpus") if isinstance(facts.get("gpus"), list) else []
    return {
        "image": _read_json(pass_dir / "container_run.json").get("image"),
        "model_snapshot": _snapshot((config.get("server") or {}).get("model_path")),
        "trace_sha256": _sha256(trace) if trace is not None and trace.is_file() else None,
        "gpu": result.get("gpu"),
        "gpu_count": len(gpus) or None,
        "driver_version": sorted({str(g.get("driver_version")) for g in gpus if isinstance(g, dict)})
        or None,
    }


#: Conditions that must match, each one present on both sides.
CHECKED_CONDITIONS = ("image", "model_snapshot", "trace_sha256", "gpu")


def _local_conditions(case_dir: Path, pass_name: str) -> dict[str, Any]:
    """What this machine's rendered campaign would capture with."""
    from alignment.image_runtime import image_tag

    config = _read_yaml(case_dir / f"{pass_name}.yaml")
    trace = _rendered_trace(case_dir, config)
    engine = config.get("engine")
    try:
        image = image_tag(engine) if engine else None
    except KeyError:
        image = None
    return {
        "image": image,
        "model_snapshot": _snapshot((config.get("server") or {}).get("model_path")),
        "trace_sha256": _sha256(trace) if trace is not None and trace.is_file() else None,
        "gpu": config.get("gpu"),
    }


def _rendered_trace(case_dir: Path, config: dict[str, Any]) -> Path | None:
    frontend = ((config.get("workload") or {}).get("frontend") or {})
    trace = frontend.get("path")
    return (case_dir / trace) if isinstance(trace, str) else None


def _replayed_trace(pass_dir: Path) -> Path | None:
    result = _read_json(pass_dir / "profile_result.json")
    source = (result.get("drive_summary") or {}).get("source_trace")
    return Path(source) if isinstance(source, str) else None


def _git(where: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(where), *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _pack_fingerprint(pack: Pack) -> str:
    return _sha256(pack.root / "campaign.yaml")


# ── bundle ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BundleReport:
    dest: Path
    manifest: dict[str, Any]

    @property
    def bytes(self) -> int:
        return sum(
            entry["size"]
            for case in self.manifest["cases"].values()
            for item in case["passes"].values()
            for entry in item.get("files", {}).values()
        )


def _iter_files(root: Path) -> Iterable[Path]:
    """Every regular file under `root`, following symlinked directories (a
    pass dir may be assembled from links), in a stable order."""
    for directory, subdirs, files in os.walk(root, followlinks=True):
        subdirs.sort()
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_file() and path.suffix not in EXCLUDED_SUFFIXES:
                yield path


def _write_scrubbed(source: Path, target: Path, scrubber: Scrubber) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix in TEXT_SUFFIXES:
        text = source.read_text(encoding="utf-8", errors="surrogateescape")
        target.write_text(scrubber.scrub(text), encoding="utf-8", errors="surrogateescape")
    else:
        shutil.copyfile(source, target)


def bundle(
    pack: Pack,
    host: HostProfile,
    out_root: Path,
    cases: list[Case],
    dest: Path,
) -> BundleReport:
    """Write the scrubbed bundle for `cases` under `dest` (which must not exist).

    `host` must be the profile the captures ran with: its paths are what gets
    replaced and refused.
    """
    # Absolute, so a relative --out-root is still replaced in what the capture
    # recorded (it records absolute paths).
    out_root = Path(out_root).resolve()
    dest = Path(dest)
    if dest.exists():
        raise BundleError(f"{dest} exists; bundle writes a fresh directory")
    base = host_scrubber(host, out_root)
    staged: list[tuple[str, str, Path, Path | None, Scrubber]] = []
    manifest_cases: dict[str, Any] = {}
    for case in cases:
        variant = pack.variant_of(case)
        case_dir = out_root / case.slug
        if not case_dir.is_dir():
            raise BundleError(f"{case.slug}: no run directory under {out_root}; render and run it first")
        passes: dict[str, Any] = {}
        for profile_pass in variant.profile_passes:
            link = case_dir / profile_pass.name
            if not link.is_dir():
                passes[profile_pass.name] = {"kind": profile_pass.kind, "state": "missing"}
                continue
            pass_dir = link.resolve()
            complete = phase_complete(case_dir, variant, profile_pass.name)
            trace = _replayed_trace(pass_dir)
            if complete and (trace is None or not trace.is_file()):
                raise BundleError(
                    f"{case.slug}/{profile_pass.name}: the replayed trace "
                    f"({trace or 'not recorded'}) is not readable here; bundle on the machine "
                    "that captured it"
                )
            passes[profile_pass.name] = {
                "kind": profile_pass.kind,
                "state": "complete" if complete else "incomplete",
                "files": {},
            }
            scrubber = pass_scrubber(base, host, pass_dir, case_dir.resolve())
            staged.append((case.slug, profile_pass.name, pass_dir, trace, scrubber))
        manifest_cases[case.slug] = {"phase": case.phase, "passes": passes}

    dest.mkdir(parents=True)
    try:
        for slug, pass_name, source_root, trace, scrubber in staged:
            pattern = scrubber.pattern()
            entry = manifest_cases[slug]["passes"][pass_name]
            sources = [(path, Path(pass_name) / path.relative_to(source_root))
                       for path in _iter_files(source_root)]
            if trace is not None and trace.is_file():
                sources.append((trace, Path(pass_name + TRACE_SUFFIX)))
            for source, relative in sources:
                target = dest / slug / relative
                _write_scrubbed(source, target, scrubber)
                hit = find_blocked(target, pattern)
                if hit is not None:
                    raise BundleError(
                        f"{slug}/{relative} still contains {hit!r} after scrubbing; fix where "
                        "the capture writes it (a binary cannot be scrubbed)"
                    )
                record = {"sha256": _sha256(target), "size": target.stat().st_size}
                if relative.name == pass_name + TRACE_SUFFIX:
                    entry["trace"] = record
                else:
                    entry["files"][relative.relative_to(pass_name).as_posix()] = record
            entry["conditions"] = capture_conditions(dest / slug / pass_name, dest / slug / f"{pass_name}{TRACE_SUFFIX}")
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "pack": pack.name,
            "pack_campaign_sha256": _pack_fingerprint(pack),
            "servingstudio_sim_commit": _git(pack.root, "rev-parse", "HEAD"),
            "servingstudio_sim_dirty": bool(
                _git(pack.root, "status", "--porcelain", "--untracked-files=no")
            ),
            "excluded_suffixes": sorted(EXCLUDED_SUFFIXES),
            "cases": manifest_cases,
        }
        (dest / BUNDLE_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    return BundleReport(dest, manifest)


def upload(dest: Path, repo_id: str, path_in_repo: str, *, create_pr: bool = True) -> str:
    """Upload a bundle directory to a Hugging Face dataset. Returns the commit
    or PR URL. A contributor without write access uploads as a PR."""
    from huggingface_hub import HfApi

    info = HfApi().upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(dest),
        path_in_repo=path_in_repo,
        commit_message=f"alignment bundle {path_in_repo}",
        create_pr=create_pr,
    )
    return str(getattr(info, "pr_url", None) or getattr(info, "commit_url", info))


# ── verify-bundle ────────────────────────────────────────────────────────────


def parse_hf_spec(spec: str) -> tuple[str, str, str]:
    """`<repo_id>@<revision>:<path>`; the revision must be a commit sha or a PR
    ref (`refs/pr/3`), so what is verified is pinned."""
    match = re.fullmatch(r"([^@:]+)@([^:]+):(.+)", spec)
    if match is None:
        raise BundleError(f"--from-hf wants <repo_id>@<revision>:<path>, got {spec!r}")
    repo_id, revision, path = match.groups()
    if not _REVISION.fullmatch(revision):
        raise BundleError(
            f"revision {revision!r} is not pinned; give a 40-hex commit sha or refs/pr/<n>"
        )
    return repo_id, revision, path.strip("/")


def download(spec: str, into: Path) -> Path:
    from huggingface_hub import snapshot_download

    repo_id, revision, path = parse_hf_spec(spec)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        allow_patterns=[f"{path}/**"],
        local_dir=str(into),
    )
    return Path(into) / path


@dataclass
class Verification:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    placed: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _safe_relative(relative: str) -> bool:
    path = PurePosixPath(relative)
    return bool(relative) and not path.is_absolute() and ".." not in path.parts and "." != relative


def _check_integrity(source: Path, manifest: dict[str, Any], pack: Pack, result: Verification) -> None:
    """Names, then every file's sha256; nothing unlisted may be present."""
    listed: set[Path] = {(source / BUNDLE_MANIFEST).resolve()}
    for slug, case_entry in (manifest.get("cases") or {}).items():
        case = pack.case_named(slug)
        if case is None or case.slug != slug:
            result.errors.append(f"{slug}: no such case in pack {pack.name}")
            continue
        passes = {item.name: item.kind for item in pack.variant_of(case).profile_passes}
        for pass_name, item in (case_entry.get("passes") or {}).items():
            if pass_name not in passes or item.get("kind") != passes[pass_name]:
                result.errors.append(
                    f"{slug}/{pass_name}: not a profile pass of this case "
                    f"(expected one of {sorted(passes)})"
                )
                continue
            entries = dict(item.get("files") or {})
            if item.get("trace") is not None:
                entries[f"../{pass_name}{TRACE_SUFFIX}"] = item["trace"]
            for relative, entry in entries.items():
                trace_entry = relative == f"../{pass_name}{TRACE_SUFFIX}"
                if not trace_entry and not _safe_relative(relative):
                    result.errors.append(f"{slug}/{pass_name}/{relative}: unsafe path")
                    continue
                path = (source / slug / pass_name / relative).resolve()
                listed.add(path)
                if not path.is_file():
                    result.errors.append(f"{slug}/{pass_name}/{relative}: missing")
                elif _sha256(path) != entry.get("sha256"):
                    result.errors.append(f"{slug}/{pass_name}/{relative}: sha256 mismatch")
    for path in _iter_files(source):
        if path.resolve() not in listed:
            result.errors.append(f"{path.relative_to(source)}: not in {BUNDLE_MANIFEST}")


def verify_bundle(
    pack: Pack,
    source: Path,
    out_root: Path,
    *,
    place: bool = True,
    replace: bool = False,
) -> Verification:
    """Check `source` against this pack and the rendered `out_root`, then place
    every complete pass. Nothing is placed unless every check passed."""
    source = Path(source)
    out_root = Path(out_root)
    result = Verification()
    manifest = _read_json(source / BUNDLE_MANIFEST)
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        raise BundleError(f"{source / BUNDLE_MANIFEST}: not a schema-{BUNDLE_SCHEMA_VERSION} bundle")
    if manifest.get("pack") != pack.name:
        raise BundleError(f"bundle is for pack {manifest.get('pack')!r}, not {pack.name!r}")
    if manifest.get("pack_campaign_sha256") != _pack_fingerprint(pack):
        result.errors.append(
            "campaign.yaml differs from the one the bundle was captured with "
            f"(bundle commit {manifest.get('servingstudio_sim_commit')})"
        )
    local_commit = _git(pack.root, "rev-parse", "HEAD")
    if manifest.get("servingstudio_sim_commit") != local_commit:
        result.warnings.append(
            f"captured at ServingStudioSim {manifest.get('servingstudio_sim_commit')}, "
            f"verifying at {local_commit}"
        )
    if manifest.get("servingstudio_sim_dirty"):
        result.warnings.append("the contributor's checkout had uncommitted changes")

    _check_integrity(source, manifest, pack, result)
    if result.errors:
        return result

    ready: list[tuple[str, str, Path, Path]] = []
    for slug, case_entry in manifest["cases"].items():
        case_dir = out_root / slug
        if not case_dir.is_dir():
            result.errors.append(f"{slug}: not rendered under {out_root}; run `render` first")
            continue
        for pass_name, item in case_entry["passes"].items():
            label = f"{slug}/{pass_name}"
            if item.get("state") != "complete":
                result.skipped.append(f"{label}: {item.get('state')} at the contributor")
                continue
            origin = source / slug / pass_name
            variant = pack.variant_of(pack.case_named(slug))
            absent = [name for name in execute_artifacts(variant, pass_name)
                      if not (origin / name).exists()]
            if absent:
                result.errors.append(f"{label}: marked complete but lacks {absent}")
            expected = _local_conditions(case_dir, pass_name)
            captured = capture_conditions(origin, source / slug / f"{pass_name}{TRACE_SUFFIX}")
            for key in CHECKED_CONDITIONS:
                want, got = expected.get(key), captured.get(key)
                if want is None:
                    result.errors.append(f"{label}: this checkout renders no {key} to check against")
                elif got is None:
                    result.errors.append(f"{label}: the capture records no {key}")
                elif want != got:
                    result.errors.append(f"{label}: {key} is {got}, this checkout expects {want}")
            target = case_dir / pass_name
            if (target.exists() or target.is_symlink()) and not replace:
                result.errors.append(f"{label}: {target} exists (--replace to overwrite)")
            ready.append((slug, pass_name, origin, case_dir))
    if result.errors or not place:
        return result

    for slug, pass_name, origin, case_dir in ready:
        _place_pass(origin, case_dir, pass_name, out_root)
        result.placed.append(f"{slug}/{pass_name}")
    return result


def _place_pass(origin: Path, case_dir: Path, pass_name: str, out_root: Path) -> None:
    """Copy one verified pass into `case_dir/pass_name` atomically: staged in a
    sibling, the completion marker written last, then renamed into place."""
    target = case_dir / pass_name
    staging = case_dir / f".{pass_name}.verify-bundle"
    if staging.exists():
        shutil.rmtree(staging)
    trace = _rendered_trace(case_dir, _read_yaml(case_dir / f"{pass_name}.yaml"))
    mapping = {
        "${capture_dir}": str(target.resolve()),
        "${case_dir}": str(case_dir.resolve()),
        "${out_root}": str(out_root.resolve()),
        "${repo}": str(REPO_ROOT),
    }
    if trace is not None:
        mapping["${mount:%s}" % TRACE_MOUNT] = str(trace.resolve())
    try:
        marker = None
        for path in _iter_files(origin):
            relative = path.relative_to(origin)
            if relative.as_posix() == COMPLETE_MARKER:
                marker = path
                continue
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix in TEXT_SUFFIXES:
                text = path.read_text(encoding="utf-8", errors="surrogateescape")
                for placeholder, value in mapping.items():
                    text = text.replace(placeholder, value)
                destination.write_text(text, encoding="utf-8", errors="surrogateescape")
            else:
                shutil.copyfile(path, destination)
        if marker is not None:
            shutil.copyfile(marker, staging / COMPLETE_MARKER)
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
