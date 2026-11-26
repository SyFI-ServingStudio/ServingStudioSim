"""Where profile rows live on disk: one SQLite file per kind.

The profile DB is a directory (``profiling/profile_db/`` by default, or
``VIBESIM_PROFILE_DB``). Each kind's table lives in ``<root>/<kind>.db``
together with the ``_profile_run`` rows its rows reference and the file's own
``_db_metadata``, so every file reads on its own: a kind table JOINs only
``_profile_run``, and a run is keyed by a hash of its contents
(``storage.run_key``), so the same run copied into several files is the same
row in each.

The directory used to be one file holding every table (``profile.db``). Such a
file is a *legacy* DB: ``split-db`` turns it into a directory, and ``merge-db``
accepts it as an input; every other reader and writer refuses it.
"""

from __future__ import annotations

import os
from pathlib import Path

SUFFIX = ".db"
# The tracked profile DB directory.
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "profile_db"


def configured_root() -> Path:
    """The profile DB directory this process reads: ``VIBESIM_PROFILE_DB`` or the default."""
    return Path(os.environ.get("VIBESIM_PROFILE_DB", DEFAULT_ROOT))


class LegacyProfileDbError(ValueError):
    """The profile DB path is a single-file DB from before the per-kind split."""


def kind_file(root: Path | str, kind: str) -> Path:
    """The file holding ``kind``'s table under ``root``."""
    return Path(root) / f"{kind}{SUFFIX}"


def kind_files(root: Path | str) -> list[Path]:
    """Every kind file under ``root``, sorted by kind; empty when ``root`` is absent."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.suffix == SUFFIX and path.is_file())


def kind_of(path: Path) -> str:
    """The kind a kind file holds."""
    return path.name[: -len(SUFFIX)]


def is_legacy_file(path: Path | str) -> bool:
    """Whether ``path`` is a single-file profile DB rather than a directory."""
    return Path(path).is_file()


def replace_directory(new: Path, target: Path) -> None:
    """Put the directory ``new`` at ``target``, replacing what is there.

    ``rename`` cannot replace a non-empty directory, so the old one is moved
    aside first and removed once ``new`` is in place. ``new`` takes the old
    directory's permission bits (a merge builds it in a private temporary
    directory).
    """
    import shutil

    old = None
    if target.exists():
        os.chmod(new, target.stat().st_mode & 0o7777)
        old = target.with_name(f".{target.name}.old-{os.getpid()}")
        target.rename(old)
    try:
        new.rename(target)
    except BaseException:
        if old is not None:
            old.rename(target)
        raise
    if old is not None:
        shutil.rmtree(old)


def require_directory(root: Path | str) -> Path:
    """``root``, refusing a legacy single-file DB with how to convert it."""
    root = Path(root)
    if is_legacy_file(root):
        raise LegacyProfileDbError(
            f"{root} is a single-file profile DB; the profile DB is now a directory of "
            "per-kind files. Convert it with "
            f"`python -m launcher kernel-profile split-db {root} --output <dir>`."
        )
    return root
