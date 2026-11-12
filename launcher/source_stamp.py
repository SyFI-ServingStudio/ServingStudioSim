"""The source identity of a checkout that has no git metadata.

`--remote` ships a checkout's files to another machine without `.git`
(`launcher/remote/source.py`). It writes this stamp at the shipped root so the
code that records provenance from git -- the run's `git_snapshot/` and the
profiler's `profiler_git_hash` -- can still say which commit and edits the code
came from. A checkout with git metadata ignores the stamp.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

STAMP_NAME = ".source-stamp.json"
STAMP_SCHEMA_VERSION = 1


def read_source_stamp(root: Path) -> dict[str, Any] | None:
    """The stamp at ``root``, or None when there is none.

    A stamp that exists but is unreadable is an error: provenance must not
    silently degrade to "unknown" on a shipped tree.
    """

    path = root / STAMP_NAME
    if not path.is_file():
        return None
    stamp = json.loads(path.read_text("utf-8"))
    if not isinstance(stamp, dict) or stamp.get("schema_version") != STAMP_SCHEMA_VERSION:
        raise ValueError(f"unsupported source stamp: {path}")
    return stamp


def has_git_metadata(root: Path) -> bool:
    # A worktree or submodule checkout has a `.git` file instead of a directory.
    return (root / ".git").exists()
