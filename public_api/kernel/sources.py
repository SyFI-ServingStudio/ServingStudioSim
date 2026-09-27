"""Read-only access to what the kernel library joins: simulator introspection and
profile.db.

Everything else the library shows (the profiling registry, kind docs, GPU specs,
the model catalog) is code in this checkout and is imported directly. The two
sources here sit behind one small class so tests can replace them with fixtures:

- the release simulator's introspection commands, which print JSON and need no
  GPU, database or Python perf_api (``list-params``, ``kernel-list``);
- profile.db, opened read-only for every query;
- the files a config's routing names: which of them this checkout tracks (git).

Introspection answers depend only on the binary, so they are cached until the
binary changes. Database aggregates are cached until the file, or a file the
library registers with :meth:`KernelSources.watch` (the model catalog), changes.
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

REPO_ROOT = Path(__file__).resolve().parents[2]


class KernelSources:
    """The simulator binary and profile.db behind the kernel library."""

    def __init__(self, *, db_path: Path, build_type: str = "release") -> None:
        self.db_path = Path(db_path)
        self.binary = binary_path(build_type)
        self._lock = threading.Lock()
        self._binary_cache: dict[Any, Any] = {}
        self._binary_stamp: float | None = None
        self._db_cache: dict[Any, Any] = {}
        self._db_stamp: tuple | None = None
        self._watched: list[Path] = []

    # -- simulator introspection -------------------------------------------------

    def _simulator(self, args: list[str], stdin: str | None = None) -> Any:
        result = subprocess.run(
            [str(self.binary), *args],
            input=stdin,
            capture_output=True,
            text=True,
            env=_build_subprocess_env(),
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"simulator {args[0]}: {result.stderr.strip()}")
        return json.loads(result.stdout)

    def cached_by_binary(self, key: Any, compute: Callable[[], Any]) -> Any:
        """``compute()``, reused until the simulator binary changes."""

        stamp = self.binary.stat().st_mtime
        with self._lock:
            if stamp != self._binary_stamp:
                self._binary_cache.clear()
                self._binary_stamp = stamp
            if key in self._binary_cache:
                return self._binary_cache[key]
        value = compute()
        with self._lock:
            self._binary_cache[key] = value
        return value

    def deployment_schema(self) -> dict:
        """``list-params``: every arch tag's params (type, default, whether it
        affects the kernel cache) and its ``#[supported]`` rows."""

        return self.cached_by_binary("list-params", lambda: self._simulator(["list-params"]))

    def kernel_list(self) -> list[dict]:
        """Every Rust kernel kind with its config/input fields and dtype fields."""

        return self.cached_by_binary("kernel-list", lambda: self._simulator(["kernel-list"]))

    # -- profile.db ----------------------------------------------------------------

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """A read-only connection; the file is never opened for writing."""

        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        try:
            conn.execute("pragma query_only = on")
            yield conn
        finally:
            conn.close()

    def watch(self, path: Path) -> None:
        """Also drop the :meth:`cached_by_db` results when ``path`` changes."""

        self._watched.append(Path(path))

    def cached_by_db(self, key: Any, compute: Callable[[], Any]) -> Any:
        """``compute()``, reused until profile.db or a watched file changes."""

        stamp = tuple(
            (st.st_mtime, st.st_size) for st in map(Path.stat, [self.db_path, *self._watched])
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

    def cached_by_db_and_binary(self, key: Any, compute: Callable[[], Any]) -> Any:
        """``compute()``, reused until profile.db, a watched file or the simulator
        binary changes: for a result that joins profile.db to the binary's
        introspection."""

        binary = self.binary.stat().st_mtime if self.binary.exists() else None
        return self.cached_by_db((key, binary), compute)

    # -- files a config names ----------------------------------------------------

    def tracked(self, paths: list[str]) -> set[str]:
        """The repo-relative ``paths`` this checkout's git tracks: files a reader
        of the repository can open at that path. Empty when git cannot say (no
        checkout)."""

        if not paths:
            return set()
        result = subprocess.run(
            ["git", "ls-files", "-z", "--", *paths],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return set(result.stdout.split("\0")) - {""} if result.returncode == 0 else set()
