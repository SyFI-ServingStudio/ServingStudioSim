"""Run metadata persistence — fired BEFORE the subprocess spawns (design §1.4
INV-2) so a crashed run still has triage info on disk.

Output layout (design §1.3.2):
    shared root: preset.json + git_snapshot/
    run log_dir: manifest.json + raw/params.json + raw/command.txt
    run subdirs: raw/ + plots/ + reports/ + payloads/ + traces/

The rust binary owns everything after spawn; the launcher writes nothing more
once the subprocess starts.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
GIT_SNAPSHOT_MAX_CHANGED_LINES_PER_FILE = 10_000
RAW_DIR = "raw"
PLOTS_DIR = "plots"
REPORTS_DIR = "reports"
PAYLOADS_DIR = "payloads"
TRACES_DIR = "traces"
RUN_OUTPUT_DIRS = (RAW_DIR, PLOTS_DIR, REPORTS_DIR, PAYLOADS_DIR, TRACES_DIR)

# Sidecar the rust binary writes (design §2.3.1 step 16) with the real
# profile.db hash + root seed. Absent until L7-β lands.
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
    result = subprocess.run(
        cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, cwd=REPO_ROOT
    )
    return result.stdout.strip() if result.returncode == 0 else "N/A"


def _git_diff_capped(max_lines: int) -> str:
    """`git diff HEAD`, skipping files whose change count exceeds `max_lines`."""
    result = subprocess.run(
        ["git", "diff", "--numstat", "HEAD"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        return "N/A"

    included: list[str] = []
    skipped: list[tuple[str, int]] = []
    for line in result.stdout.splitlines():
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


def read_manifest_sidecar(log_dir: Path) -> dict[str, Any] | None:
    """Read the rust binary's manifest sidecar if present (profile.db hash +
    seed chain). Returns None until L7-β writes it."""
    sidecar = log_dir / MANIFEST_SIDECAR
    if not sidecar.is_file():
        return None
    return json.loads(sidecar.read_text())


def save_manifest_json(
    log_dir: Path,
    profile_db_hash: str | None = None,
    seed_chain: dict[str, Any] | None = None,
) -> None:
    """Write `manifest.json`. The real `profile.db` SHA + seed chain come from
    the binary sidecar (L7-β); until that exists, write a partial manifest
    flagged `pending_l7_beta` so the gap is explicit rather than silently
    missing."""
    sidecar = read_manifest_sidecar(log_dir)
    if sidecar:
        profile_db_hash = profile_db_hash or sidecar.get("profile_db_sha256")
        seed_chain = seed_chain or sidecar.get("seed_chain")

    manifest: dict[str, Any] = {
        "profile_db_sha256": profile_db_hash,
        "seed_chain": seed_chain,
    }
    if profile_db_hash is None and seed_chain is None:
        manifest["pending_l7_beta"] = True
        manifest["note"] = (
            "profile.db hash + seed chain are produced by the rust binary "
            "(L7-β), which is not implemented yet."
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
    save_manifest_json(log_dir)
