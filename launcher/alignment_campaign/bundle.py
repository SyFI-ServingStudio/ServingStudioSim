"""`bundle` / `verify-bundle`: hand a campaign's captures to someone else and back.

A contributor runs the profile passes of a pack on their own machine, then
`bundle` collects each case's capture directories into one publishable tree:

    <dest>/bundle.json                     what is inside, and under what conditions
    <dest>/<case_slug>/<pass_name>/...     the pass directory, scrubbed

The maintainer runs `verify-bundle` against their own rendered campaign. It
checks every file's sha256, checks the conditions against the local pack and
reference image, and places each complete pass into its case directory. The
remaining phases (timing_predict, analysis_kernel, simulation, analysis_e2e)
then run locally with `alignment-campaign run`, so every number is recomputed
from the raw capture rather than taken from the contributor.

**Scrubbing.** Host paths are machine facts, not measurement, and a bundle may
be public. Text files have every host-specific path replaced by a placeholder
(`{capture_dir}`, `{case_dir}`, `{out_root}`, `{repo}`, `{home}`, `{host.<key>}`
for each host-profile path), the user name by `{user}`, the host name by
`{hostname}` and each GPU UUID by `GPU-<n>`. Binary files (`.nsys-rep`,
parquet) cannot be rewritten, so the blocklist (user name, home, host name, GPU
UUIDs, every scrubbed path) is searched in every file after scrubbing, and any
hit refuses the bundle with the file and the string. A hit in a binary is a
leak to fix where the capture writes it (as `container_cache_path` did for the
autotune preflight variable), not to scrub here. The blocklist search over a
compressed section of an `.nsys-rep` sees only plain text, so it is a backstop,
not a proof.

`verify-bundle` restores the structural placeholders (`{capture_dir}`,
`{case_dir}`, `{out_root}`, `{repo}`) to its own paths, which makes the
capture's own `log_dir` record match where it now lives
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
from pathlib import Path
from typing import Any, Iterable

import yaml

from .execute import phase_complete
from .pack import REPO_ROOT, Case, HostProfile, Pack, PackError

BUNDLE_SCHEMA_VERSION = 1
BUNDLE_MANIFEST = "bundle.json"

#: Suffixes rewritten as UTF-8 text. Everything else is copied byte for byte
#: and only searched.
TEXT_SUFFIXES = frozenset({".json", ".jsonl", ".yaml", ".yml", ".log", ".txt", ".csv", ".md"})
#: Never bundled: regenerable from what is bundled.
EXCLUDED_SUFFIXES = frozenset({".sqlite", ".sqlite-journal"})
#: Placeholders `verify-bundle` restores to the maintainer's own paths.
STRUCTURAL = ("{capture_dir}", "{case_dir}", "{out_root}", "{repo}")
#: Tokens shorter than this are searched only as a path component (`/<token>/`),
#: so a short user name does not match random bytes in a binary.
MIN_BARE_TOKEN = 5
_CHUNK = 64 << 20


class BundleError(PackError):
    """A bundle that must not be published or placed. The message names why."""


# ── scrubbing ────────────────────────────────────────────────────────────────


@dataclass
class Scrubber:
    """Ordered literal replacements plus the blocklist searched afterwards."""

    replacements: list[tuple[str, str]] = field(default_factory=list)
    blocklist: list[str] = field(default_factory=list)

    def add_path(self, path: str | Path | None, placeholder: str, *, block: bool = True) -> None:
        if not path:
            return
        text = str(path).rstrip("/")
        if not text.startswith("/"):
            # A host-profile path relative to the checkout, as `HostProfile`
            # resolves it.
            text = str((REPO_ROOT / text).resolve())
        variants = {text}
        try:
            variants.add(str(Path(text).resolve()))
        except OSError:
            pass
        for variant in variants:
            if variant not in ("", "/"):
                self.replacements.append((variant, placeholder))
                if block:
                    self.blocklist.append(variant)

    def add_token(self, token: str | None, placeholder: str) -> None:
        if token:
            self.replacements.append((token, placeholder))
            self.blocklist.append(token)

    def ordered(self) -> list[tuple[str, str]]:
        # Longest first, so a capture dir is replaced before the out-root that
        # contains it, and both before the user name inside them.
        seen: dict[str, str] = {}
        for old, new in self.replacements:
            seen.setdefault(old, new)
        return sorted(seen.items(), key=lambda item: -len(item[0]))

    def scrub(self, text: str) -> str:
        for old, new in self.ordered():
            text = text.replace(old, new)
        return text

    def pattern(self) -> re.Pattern[bytes]:
        alternatives = []
        for token in sorted(set(self.blocklist), key=len, reverse=True):
            if "/" not in token and len(token) < MIN_BARE_TOKEN:
                alternatives.append(re.escape(f"/{token}/".encode()))
            else:
                alternatives.append(re.escape(token.encode()))
        return re.compile(b"|".join(alternatives)) if alternatives else re.compile(b"(?!)")


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


def host_scrubber(host: HostProfile, out_root: Path, *, extra_paths: Iterable[Path] = ()) -> Scrubber:
    """Replacements for this machine. Pass and case dirs are added per file."""
    scrubber = Scrubber()
    scrubber.add_path(out_root, "{out_root}")
    scrubber.add_path(REPO_ROOT, "{repo}")
    # The workspace that holds the checkout (sibling worktrees, logs/, caches),
    # when it is the user's own: a checkout under `/opt` must not block `/opt`.
    workspace = REPO_ROOT.parent
    if getpass.getuser() in workspace.parts or Path.home() in (workspace, *workspace.parents):
        scrubber.add_path(workspace, "{workspace}")
    for key in ("text_corpus", "fork_python"):
        scrubber.add_path(getattr(host, key), "{host.%s}" % key)
    # Shared, user-neutral install and cache roots: replaced for portability,
    # not blocked (an image may hold nsys at the same path; the cluster injects
    # the hub root into every container's HF_* variables).
    scrubber.add_path(host.nsys_executable, "{host.nsys_executable}", block=False)
    scrubber.add_path(host.hf_hub_root, "{host.hf_hub_root}", block=False)
    for extra in extra_paths:
        scrubber.add_path(extra, "{path}")
    scrubber.add_path(Path.home(), "{home}")
    scrubber.add_token(getpass.getuser(), "{user}")
    scrubber.add_token(socket.gethostname(), "{hostname}")
    return scrubber


# ── conditions ───────────────────────────────────────────────────────────────


def _read_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return document if isinstance(document, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(model_path: str | None) -> str | None:
    if not model_path:
        return None
    match = re.search(r"/snapshots/([0-9a-f]{40})", model_path)
    return match.group(1) if match else None


def pass_conditions(pass_dir: Path) -> dict[str, Any]:
    """The facts `verify-bundle` checks, read from the capture's own records.
    Read before scrubbing; none of them is a host path."""
    result = _read_json(pass_dir / "profile_result.json")
    container = _read_json(pass_dir / "container_run.json")
    facts = _read_json(pass_dir / "capture_facts.json")
    config: dict[str, Any] = {}
    if (pass_dir / "profile.yaml").is_file():
        loaded = yaml.safe_load((pass_dir / "profile.yaml").read_text())
        config = loaded if isinstance(loaded, dict) else {}
    source_trace = (result.get("drive_summary") or {}).get("source_trace")
    trace_sha = None
    if isinstance(source_trace, str) and Path(source_trace).is_file():
        trace_sha = _sha256(Path(source_trace))
    gpus = facts.get("gpus") if isinstance(facts.get("gpus"), list) else []
    return {
        "profile_kind": result.get("profile_kind") or config.get("profile_kind"),
        "engine": result.get("engine") or config.get("engine"),
        "image": container.get("image"),
        "image_id": container.get("image_id"),
        "model_snapshot": _snapshot((config.get("server") or {}).get("model_path")),
        "trace_sha256": trace_sha,
        "gpu": result.get("gpu") or config.get("gpu"),
        "gpu_count": len(gpus) or None,
        "driver_version": sorted({str(g.get("driver_version")) for g in gpus if isinstance(g, dict)})
        or None,
    }


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
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix not in EXCLUDED_SUFFIXES:
            yield path


def _place(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def bundle(
    pack: Pack,
    host: HostProfile,
    out_root: Path,
    cases: list[Case],
    dest: Path,
) -> BundleReport:
    """Write the scrubbed bundle for `cases` under `dest` (which must not exist)."""
    # Absolute, so a relative --out-root is still replaced in what the capture
    # recorded (it records absolute paths).
    out_root = Path(out_root).resolve()
    dest = Path(dest)
    if dest.exists():
        raise BundleError(f"{dest} exists; bundle writes a fresh directory")
    base = host_scrubber(host, out_root)
    staged: list[tuple[str, str, Path, Path, Scrubber]] = []
    manifest_cases: dict[str, Any] = {}
    for case in cases:
        variant = pack.variant_of(case)
        case_dir = out_root / case.slug
        if not case_dir.is_dir():
            raise BundleError(f"{case.slug}: no run directory under {out_root}; render and run it first")
        passes: dict[str, Any] = {}
        for profile_pass in variant.profile_passes:
            pass_dir = case_dir / profile_pass.name
            if not pass_dir.is_dir():
                passes[profile_pass.name] = {"kind": profile_pass.kind, "state": "missing"}
                continue
            complete = phase_complete(case_dir, variant, profile_pass.name)
            scrubber = Scrubber(list(base.replacements), list(base.blocklist))
            scrubber.add_path(pass_dir, "{capture_dir}")
            # Where the capture was written, if it has moved since (a copied
            # campaign, or a pass dir linked in from another run root).
            recorded = _read_json(pass_dir / "profile_result.json").get("log_dir")
            if isinstance(recorded, str):
                scrubber.add_path(recorded, "{capture_dir}")
            scrubber.add_path(case_dir, "{case_dir}")
            for index, uuid in enumerate(_gpu_uuids(pass_dir.resolve())):
                scrubber.add_token(uuid, f"GPU-{index}")
            passes[profile_pass.name] = {
                "kind": profile_pass.kind,
                "state": "complete" if complete else "incomplete",
                "conditions": pass_conditions(pass_dir.resolve()),
                "files": {},
            }
            staged.append((case.slug, profile_pass.name, pass_dir.resolve(), case_dir, scrubber))
        manifest_cases[case.slug] = {"phase": case.phase, "passes": passes}

    dest.mkdir(parents=True)
    try:
        for slug, pass_name, source_root, _case_dir, scrubber in staged:
            target_root = dest / slug / pass_name
            pattern = scrubber.pattern()
            entries = manifest_cases[slug]["passes"][pass_name]["files"]
            for source in _iter_files(source_root):
                relative = source.relative_to(source_root)
                target = target_root / relative
                if source.suffix in TEXT_SUFFIXES:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    text = source.read_text(encoding="utf-8", errors="surrogateescape")
                    target.write_text(scrubber.scrub(text), encoding="utf-8", errors="surrogateescape")
                else:
                    _place(source, target)
                hit = find_blocked(target, pattern)
                if hit is not None:
                    raise BundleError(
                        f"{slug}/{pass_name}/{relative} still contains {hit!r} after scrubbing; "
                        "fix where the capture writes it (a binary cannot be scrubbed)"
                    )
                entries[relative.as_posix()] = {"sha256": _sha256(target), "size": target.stat().st_size}
        manifest = {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "pack": pack.name,
            "pack_campaign_sha256": _pack_fingerprint(pack),
            "servingstudio_sim_commit": _git(pack.root, "rev-parse", "HEAD"),
            "servingstudio_sim_dirty": bool(_git(pack.root, "status", "--porcelain", "--untracked-files=no")),
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


def download(spec: str, into: Path) -> Path:
    """`<repo_id>@<revision>:<path>` -> local bundle directory. The revision is a
    commit sha or a PR ref (`refs/pr/3`), so the download is pinned."""
    from huggingface_hub import snapshot_download

    match = re.fullmatch(r"([^@:]+)@([^:]+):(.+)", spec)
    if match is None:
        raise BundleError(f"--from-hf wants <repo_id>@<revision>:<path>, got {spec!r}")
    repo_id, revision, path = match.groups()
    path = path.strip("/")
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


def _local_conditions(case_dir: Path, pass_name: str) -> dict[str, Any]:
    """What this machine's rendered campaign would capture with."""
    from alignment.image_runtime import image_tag

    config_path = case_dir / f"{pass_name}.yaml"
    if not config_path.is_file():
        return {}
    config = yaml.safe_load(config_path.read_text()) or {}
    frontend = ((config.get("workload") or {}).get("frontend") or {})
    trace = frontend.get("path")
    trace_path = (case_dir / trace) if isinstance(trace, str) else None
    engine = config.get("engine")
    try:
        image = image_tag(engine) if engine else None
    except Exception:  # noqa: BLE001 - an unknown engine is reported as missing
        image = None
    return {
        "image": image,
        "model_snapshot": _snapshot((config.get("server") or {}).get("model_path")),
        "trace_sha256": _sha256(trace_path) if trace_path and trace_path.is_file() else None,
        "gpu": config.get("gpu"),
    }


def _restore(text: str, mapping: dict[str, str]) -> str:
    for placeholder, value in mapping.items():
        text = text.replace(placeholder, value)
    return text


def verify_bundle(
    pack: Pack,
    source: Path,
    out_root: Path,
    *,
    place: bool = True,
    replace: bool = False,
) -> Verification:
    """Check `source` against this pack and the rendered `out_root`, then place
    every complete pass whose checks passed."""
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

    # Integrity first: nothing is placed from a bundle with a bad or extra file.
    listed: set[Path] = {source / BUNDLE_MANIFEST}
    for slug, case_entry in (manifest.get("cases") or {}).items():
        for pass_name, item in (case_entry.get("passes") or {}).items():
            for relative, entry in (item.get("files") or {}).items():
                path = source / slug / pass_name / relative
                listed.add(path)
                if not path.is_file():
                    result.errors.append(f"{slug}/{pass_name}/{relative}: missing")
                elif _sha256(path) != entry.get("sha256"):
                    result.errors.append(f"{slug}/{pass_name}/{relative}: sha256 mismatch")
    for path in source.rglob("*"):
        if path.is_file() and path not in listed and ".cache" not in path.relative_to(source).parts:
            result.errors.append(f"{path.relative_to(source)}: not in {BUNDLE_MANIFEST}")
    if result.errors:
        return result

    ready: list[tuple[str, str, Path, Path]] = []
    for slug, case_entry in manifest["cases"].items():
        case = pack.case_named(slug)
        if case is None or case.slug != slug:
            result.errors.append(f"{slug}: no such case in pack {pack.name}")
            continue
        case_dir = out_root / slug
        if not case_dir.is_dir():
            result.errors.append(f"{slug}: not rendered under {out_root}; run `render` first")
            continue
        for pass_name, item in case_entry["passes"].items():
            label = f"{slug}/{pass_name}"
            if item.get("state") != "complete":
                result.skipped.append(f"{label}: {item.get('state')} at the contributor")
                continue
            expected = _local_conditions(case_dir, pass_name)
            captured = item.get("conditions") or {}
            for key in ("image", "model_snapshot", "trace_sha256", "gpu"):
                want, got = expected.get(key), captured.get(key)
                if want is None or got is None:
                    result.warnings.append(f"{label}: {key} not checked (local {want}, bundle {got})")
                elif want != got:
                    result.errors.append(f"{label}: {key} is {got}, this checkout expects {want}")
            ready.append((slug, pass_name, source / slug / pass_name, case_dir))
    if result.errors or not place:
        return result

    for slug, pass_name, origin, case_dir in ready:
        target = case_dir / pass_name
        if target.exists() or target.is_symlink():
            if not replace:
                result.errors.append(f"{slug}/{pass_name}: {target} exists (--replace to overwrite)")
                continue
            if target.is_symlink() or target.is_file():
                target.unlink()
            else:
                shutil.rmtree(target)
        mapping = {
            "{capture_dir}": str(target.resolve()),
            "{case_dir}": str(case_dir.resolve()),
            "{out_root}": str(out_root.resolve()),
            "{repo}": str(REPO_ROOT),
        }
        for path in _iter_files(origin):
            relative = path.relative_to(origin)
            destination = target / relative
            if path.suffix in TEXT_SUFFIXES:
                destination.parent.mkdir(parents=True, exist_ok=True)
                text = path.read_text(encoding="utf-8", errors="surrogateescape")
                destination.write_text(_restore(text, mapping), encoding="utf-8", errors="surrogateescape")
            else:
                _place(path, destination)
        result.placed.append(f"{slug}/{pass_name}")
    return result
