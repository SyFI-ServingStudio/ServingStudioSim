"""How a profile result names the files it wrote, so a copied profile still reads.

A profile phase writes ``profile_result.json`` naming its artifacts. Paths inside
the phase directory are stored relative to it, so the directory can be copied
(from a remote GPU host, or moved on disk) and read where it lands. Results
written before this stored absolute paths; `localize` rebases those onto the
directory they are read from when they point inside the recorded ``log_dir``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: Top-level result fields that name a file or directory the profile wrote.
ARTIFACT_KEYS = (
    "sqlite",
    "metrics_jsonl",
    "request_timings_jsonl",
    "server_log",
    "replay_result",
    "parsed_nsys",
    "parsed_kernel_rows",
    "kernel_sequences",
    "host_timeline",
    "token_corpus_manifest",
    "expert_load_jsonl",
    "expert_popularity_json",
    "draft_expert_load_jsonl",
    "draft_expert_popularity_json",
    "spec_decode_metrics_json",
)


def relative_to(path: Path | str, root: Path) -> str:
    """``path`` as recorded in a result written under ``root``.

    Relative to ``root`` when it lies inside it, spelled either way (``root``
    may sit behind a symlink that ``path`` was resolved through); else absolute.
    """
    path = Path(path)
    for candidate, base in ((path, root), (path.resolve(), root.resolve())):
        try:
            return candidate.relative_to(base).as_posix()
        except ValueError:
            continue
    return str(path)


def record(result: dict[str, Any], root: Path) -> dict[str, Any]:
    """A copy of ``result`` whose artifact paths are relative to ``root``."""
    return {
        key: relative_to(value, root) if key in ARTIFACT_KEYS and isinstance(value, str) else value
        for key, value in result.items()
    }


def resolve(value: str, root: Path, recorded_root: str | None = None) -> Path:
    """Where a recorded artifact is, for a result read from ``root``."""
    path = Path(value)
    if not path.is_absolute():
        return root / path
    if recorded_root:
        try:
            moved = root / path.relative_to(recorded_root)
        except ValueError:
            moved = None
        if moved is not None and (moved.exists() or not path.exists()):
            return moved
    return path


def resolve_beside(value: str, root: Path, recorded_root: str | None) -> Path:
    """Where a recorded input outside the phase directory is, for a result read from ``root``.

    An input such as the case's ``trace.csv`` sits beside or above the recorded
    ``log_dir``, so `resolve` cannot rebase it. When the recorded path is gone
    here, it is rebased from the nearest ancestor of ``recorded_root`` that
    contains it onto the same ancestor of ``root`` (a remote's out-root copied
    back to this host keeps the layout under it), if that file exists.
    """
    path = Path(value)
    if not path.is_absolute() or path.exists() or not recorded_root:
        return resolve(value, root, recorded_root)
    recorded = Path(recorded_root)
    for here, there in zip((root, *root.parents), (recorded, *recorded.parents)):
        if len(there.parts) <= 1:
            break
        try:
            moved = here / path.relative_to(there)
        except ValueError:
            continue
        return moved if moved.exists() else path
    return path


def localize(result: dict[str, Any], root: Path) -> dict[str, Any]:
    """A copy of ``result`` with every artifact path absolute, as it is here."""
    recorded_root = result.get("log_dir") if isinstance(result.get("log_dir"), str) else None
    return {
        key: str(resolve(value, root, recorded_root))
        if key in ARTIFACT_KEYS and isinstance(value, str)
        else value
        for key, value in result.items()
    }
