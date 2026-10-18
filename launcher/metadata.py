"""Run metadata persistence — fired BEFORE the subprocess spawns (design §1.4
INV-2) so a crashed run still has triage info on disk.

Output layout (design §1.3.2):
    shared root: preset.json + git_snapshot/
    run log_dir: manifest.json + raw/params.json + raw/command.txt
    run subdirs: raw/ + plots/ + reports/ + payloads/ + traces/

This module finishes before spawn. Runtime stage state and completion evidence
are written separately by ``launcher.process.journal``.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .process import ProcessSpec, ProcessSupervisor

REPO_ROOT = Path(__file__).resolve().parents[1]
_PROCESS_SUPERVISOR = ProcessSupervisor()
GIT_SNAPSHOT_MAX_CHANGED_LINES_PER_FILE = 10_000
RAW_DIR = "raw"
PLOTS_DIR = "plots"
REPORTS_DIR = "reports"
PAYLOADS_DIR = "payloads"
TRACES_DIR = "traces"
RUN_OUTPUT_DIRS = (RAW_DIR, PLOTS_DIR, REPORTS_DIR, PAYLOADS_DIR, TRACES_DIR)

# Sidecar the rust binary writes (design §2.3.1 step 16) with the root seed.
# Absent until L7-β lands.
MANIFEST_SIDECAR = "manifest_sidecar.json"


def _strip_internal(params: dict) -> dict:
    """Drop launcher-internal `_`-prefixed keys (e.g. `_sweep_labels`)."""
    return {k: v for k, v in params.items() if not k.startswith("_")}


def ensure_run_layout(log_dir: Path) -> None:
    """Create the stable run artifact buckets from design §1.3.2."""
    log_dir.mkdir(parents=True, exist_ok=True)
    for output_dir in RUN_OUTPUT_DIRS:
        (log_dir / output_dir).mkdir(exist_ok=True)


def raw_dir(log_dir: Path) -> Path:
    return log_dir / RAW_DIR


def save_params_json(log_dir: Path, params: dict, start_ts: datetime) -> None:
    payload = _strip_internal(params)
    payload["start_ts"] = start_ts.isoformat()
    run_raw_dir = raw_dir(log_dir)
    run_raw_dir.mkdir(parents=True, exist_ok=True)
    (run_raw_dir / "params.json").write_text(
        json.dumps(payload, indent=2, default=str)
    )


def save_preset_json(log_dir: Path, preset: dict) -> None:
    (log_dir / "preset.json").write_text(json.dumps(preset, indent=2, default=str))


def save_command_log(log_dir: Path, cmd: list[str]) -> None:
    run_raw_dir = raw_dir(log_dir)
    run_raw_dir.mkdir(parents=True, exist_ok=True)
    (run_raw_dir / "command.txt").write_text(" ".join(cmd) + "\n")


# ── git snapshot (commit + capped diff) ─────────────────────────────────────


def _git(cmd: list[str]) -> str:
    result = _PROCESS_SUPERVISOR.run_sync(
        ProcessSpec(argv=cmd, cwd=REPO_ROOT, capture_output=True, name="git-metadata")
    )
    return result.output.strip() if result.succeeded else "N/A"


def _git_diff_capped(max_lines: int) -> str:
    """`git diff HEAD`, skipping files whose change count exceeds `max_lines`."""
    result = _PROCESS_SUPERVISOR.run_sync(
        ProcessSpec(
            argv=["git", "diff", "--numstat", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            name="git-numstat",
        )
    )
    if not result.succeeded:
        return "N/A"

    included: list[str] = []
    skipped: list[tuple[str, int]] = []
    for line in result.output.splitlines():
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        adds, dels, path = parts
        try:
            changed_lines = int(adds) + int(dels)
        except ValueError:
            included.append(path)  # binary diff (numstat shows '-')
            continue
        if changed_lines > max_lines:
            skipped.append((path, changed_lines))
        else:
            included.append(path)

    chunks: list[str] = []
    if skipped:
        chunks.append(f"Skipped files with more than {max_lines} changed lines:")
        chunks += [f"- {path} ({count} changed lines)" for path, count in skipped]
    diffs = [_git(["git", "diff", "HEAD", "--", p]) for p in included]
    diffs = [d for d in diffs if d and d != "N/A"]
    if diffs:
        if chunks:
            chunks.append("")
        chunks.append("\n".join(diffs))
    return "\n".join(chunks).strip() or "(no diff)"


def save_git_snapshot(log_dir: Path, timestamp: datetime) -> None:
    snapshot_dir = log_dir / "git_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    (snapshot_dir / "commit.txt").write_text(
        f"# {timestamp.isoformat()}\n" + _git(["git", "log", "-1", "--oneline"]) + "\n"
    )
    (snapshot_dir / "status.txt").write_text(_git(["git", "status", "--short"]) + "\n")
    (snapshot_dir / "diff.patch").write_text(
        _git_diff_capped(GIT_SNAPSHOT_MAX_CHANGED_LINES_PER_FILE) + "\n"
    )


# ── manifest (★ repro-gap fix) ──────────────────────────────────────────────


@functools.lru_cache(maxsize=None)
def _sha256(path: Path, size: int, mtime_ns: int) -> str:
    """SHA-256 of a file; size and mtime key the cache, so a sweep hashes each
    unchanged file once."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    """Resolved path and SHA-256 of an input the run reads (None if missing)."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "sha256": None}
    st = resolved.stat()
    return {"path": str(resolved), "sha256": _sha256(resolved, st.st_size, st.st_mtime_ns)}


def profile_db_path() -> Path:
    """The profile.db the binary reads: `VIBESIM_PROFILE_DB`, else the repo's."""
    return Path(os.environ.get("VIBESIM_PROFILE_DB", REPO_ROOT / "profiling" / "profile.db"))


def read_manifest_sidecar(log_dir: Path) -> dict[str, Any] | None:
    """Read the rust binary's manifest sidecar if present (seed chain).
    Returns None until L7-β writes it."""
    sidecar = log_dir / MANIFEST_SIDECAR
    if not sidecar.is_file():
        return None
    return json.loads(sidecar.read_text())


def save_manifest_json(
    log_dir: Path,
    binary: str | Path | None = None,
    seed_chain: dict[str, Any] | None = None,
) -> None:
    """Write `manifest.json`: the binary and the profile.db the run reads, each
    as resolved path + SHA-256 at launch (the git snapshot names the source,
    not what was built or which kernel table it read). The seed chain comes
    from the binary sidecar (L7-β); until that exists it is flagged
    `pending_l7_beta` so the gap is explicit rather than silently missing."""
    sidecar = read_manifest_sidecar(log_dir)
    if sidecar:
        seed_chain = seed_chain or sidecar.get("seed_chain")

    manifest: dict[str, Any] = {
        "binary": file_identity(binary) if binary is not None else None,
        "profile_db": file_identity(profile_db_path()),
        "seed_chain": seed_chain,
    }
    if seed_chain is None:
        manifest["pending_l7_beta"] = True
        manifest["note"] = (
            "the seed chain is produced by the rust binary (L7-β), which is "
            "not implemented yet."
        )
    (log_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def write_shared_metadata(root_dir: Path, preset: dict) -> None:
    """Invocation-level metadata, written once per single run or sweep."""
    root_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    save_preset_json(root_dir, preset)
    save_git_snapshot(root_dir, now)


def write_run_metadata(log_dir: Path, params: dict, cmd: list[str]) -> None:
    """Per-run metadata. In a sweep, shared preset/git metadata lives at the
    sweep root, so individual run dirs only carry run-specific files."""
    ensure_run_layout(log_dir)
    now = datetime.now(UTC)
    save_params_json(log_dir, params, now)
    save_command_log(log_dir, cmd)
    save_manifest_json(log_dir, binary=cmd[0])
