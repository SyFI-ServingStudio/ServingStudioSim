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


def localize(result: dict[str, Any], root: Path) -> dict[str, Any]:
    """A copy of ``result`` with every artifact path absolute, as it is here."""
    recorded_root = result.get("log_dir") if isinstance(result.get("log_dir"), str) else None
    return {
        key: str(resolve(value, root, recorded_root))
        if key in ARTIFACT_KEYS and isinstance(value, str)
        else value
        for key, value in result.items()
    }
