"""Read-only access to the two sources the service's documents read besides its
own code: simulator introspection and profile.db.

Everything else (the profiling registry, kind docs, GPU specs, the model
catalog, the public presets) is code in this checkout and is imported directly.
The two sources here sit behind one small class so tests can replace them with
fixtures:

- the release simulator's introspection commands, which print JSON and need no
  GPU, database or Python perf_api (``kernel-list``, ``list-params``,
  ``cost-trees``);
- profile.db, opened read-only for every query.

Introspection answers depend only on the binary, so they are cached until the
binary changes (:func:`introspect`, which the other introspection commands the
service runs share). Database aggregates are cached until the file changes.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from launcher.exec import _build_subprocess_env, binary_path
from profiling.db import layout
from profiling.db.migrate import require_current


def run(binary: Path, args: list[str], stdin: str | None = None) -> subprocess.CompletedProcess:
    """``binary args`` (with ``stdin``) in the launcher's subprocess env, its
    output captured; the caller reads its exit."""
    return subprocess.run(
        [str(binary), *args],
        input=stdin,
        capture_output=True,
        text=True,
        env=_build_subprocess_env(),
        check=False,
    )


def run_json(binary: Path, args: list[str], stdin: str | None = None) -> Any:
    """``binary args`` (with ``stdin``) as the JSON it prints; raises with its
    stderr when it fails."""
    result = run(binary, args, stdin)
    if result.returncode:
        raise RuntimeError(f"{Path(binary).name} {args[0]}: {result.stderr.strip()}")
    return json.loads(result.stdout)


_INTROSPECTED: dict[tuple, tuple[float, Any]] = {}
_INTROSPECTED_LOCK = threading.Lock()


def introspect(binary: Path, args: list[str]) -> Any:
    """:func:`run_json` of an introspection command (``simulator list-params``,
    ``trace-formats``, ``tracegen describe``), run again only when ``binary``
    changes."""
    key, stamp = (str(binary), *args), Path(binary).stat().st_mtime
    with _INTROSPECTED_LOCK:
        if key in _INTROSPECTED and _INTROSPECTED[key][0] == stamp:
            return _INTROSPECTED[key][1]
    value = run_json(binary, args)
    with _INTROSPECTED_LOCK:
        _INTROSPECTED[key] = (stamp, value)
    return value


class Sources:
    """The simulator binary and profile DB behind every document the service builds."""

    def __init__(self, *, db_path: Path, build_type: str = "release") -> None:
        # The profile DB directory: one SQLite file per kind (`profiling.db.layout`).
        self.db_path = layout.require_directory(Path(db_path).resolve())
        self.build_type = build_type
        self.binary = binary_path(build_type)
        self._lock = threading.Lock()
        self._db_cache: dict[Any, Any] = {}
        self._db_stamp: tuple | None = None

    # -- simulator introspection -------------------------------------------------

    def kernel_list(self) -> list[dict]:
        """Every Rust kernel kind with its config/input fields and dtype fields."""

        return introspect(self.binary, ["kernel-list"])

    def list_params(self) -> dict:
        """The param schema: every arch's params and the ``timing-predict`` case fields."""

        return introspect(self.binary, ["list-params"])

    def cost_trees(self, blocks: list[dict]) -> list[dict]:
        """``simulator cost-trees --kernel-configs`` of ``{gpu, arch}`` blocks:
        each one's cost tree, prediction case shape and kernel configs."""

        return run_json(self.binary, ["cost-trees", "-", "--kernel-configs"], json.dumps(blocks))

    # -- profile DB ----------------------------------------------------------------

    @contextmanager
    def connect(self, kind: str) -> Iterator[sqlite3.Connection]:
        """A read-only connection to ``kind``'s file; the file is never opened for
        writing. A kind without a file gets an empty database, which holds no table."""

        path = layout.kind_file(self.db_path, kind)
        if path.is_file():
            conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        else:
            conn = sqlite3.connect(":memory:")
        try:
            conn.execute("pragma query_only = on")
            require_current(conn, str(path))
            yield conn
        finally:
            conn.close()

    def cached_by_db(self, key: Any, compute: Callable[[], Any]) -> Any:
        """``compute()``, reused until any kind file of the profile DB changes."""

        stamp = tuple(
            (path.name, st.st_mtime_ns, st.st_size)
            for path in layout.kind_files(self.db_path)
            for st in (path.stat(),)
        )
        with self._lock:
            if stamp != self._db_stamp:
                self._db_cache.clear()
                self._db_stamp = stamp
            if key in self._db_cache:
                return self._db_cache[key]
        value = compute()
        with self._lock:
            self._db_cache[key] = value
        return value
