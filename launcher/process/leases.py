"""Crash-safe cross-launcher resource leases backed by ``flock``.

Lock-file presence is never interpreted as ownership.  The kernel releases an
advisory lock when a launcher exits, including abrupt exits; per-holder JSON is
diagnostic only and may outlive a crashed process.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import socket
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Literal

LeaseMode = Literal["shared", "exclusive"]
_LEASE_POLL_SECONDS = 0.1


def launcher_lock_root(repository_root: Path) -> Path:
    """Return the stable cross-process lock root for one checkout and one user.

    The uid belongs in the *first* component below ``TMPDIR``, not in a
    subdirectory of a shared one.  A shared parent is precisely what breaks on a
    multi-user machine: whoever runs the launcher first creates it under their
    own umask, and every later user fails with ``EACCES`` trying to mkdir their
    checkout's directory inside it.  ``TMPDIR`` is world-writable and sticky, so
    a per-uid root is creatable by anyone and removable by no one else.

    Two users sharing one checkout therefore stop excluding each other.  That is
    a trade rather than a regression: cross-user exclusion only ever worked when
    the two also shared a group, and crashed the second launcher otherwise.
    """

    identity = hashlib.sha256(str(repository_root.resolve()).encode()).hexdigest()[:16]
    temporary_root = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
    return temporary_root / f"vibesim-launcher-locks-{os.getuid()}" / identity


@dataclass(slots=True)
class ResourceLease:
    """One shared/exclusive advisory lease with async and sync entry points."""

    lock_path: Path
    resource: str
    mode: LeaseMode
    _stream: IO[str] | None = field(default=None, init=False, repr=False)
    _holder_path: Path | None = field(default=None, init=False, repr=False)

    async def __aenter__(self) -> ResourceLease:
        await self.acquire()
        return self

    async def __aexit__(self, *_exception_info: object) -> None:
        self.release()

    def __enter__(self) -> ResourceLease:
        self.acquire_sync()
        return self

    def __exit__(self, *_exception_info: object) -> None:
        self.release()

    async def acquire(self) -> None:
        self._prepare_stream()
        assert self._stream is not None
        operation = self._operation() | fcntl.LOCK_NB
        try:
            while True:
                try:
                    fcntl.flock(self._stream.fileno(), operation)
                    break
                except BlockingIOError:
                    await asyncio.sleep(_LEASE_POLL_SECONDS)
            self._write_holder_record()
        except BaseException:
            self.release()
            raise

    def acquire_sync(self) -> None:
        self._prepare_stream()
        assert self._stream is not None
        try:
            fcntl.flock(self._stream.fileno(), self._operation())
            self._write_holder_record()
        except BaseException:
            self.release()
            raise

    def release(self) -> None:
        if self._stream is None:
            return
        try:
            if self._holder_path is not None:
                self._holder_path.unlink(missing_ok=True)
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
        finally:
            self._stream.close()
            self._stream = None
            self._holder_path = None

    def _prepare_stream(self) -> None:
        if self._stream is not None:
            raise RuntimeError(f"lease already acquired: {self.resource}")
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.lock_path.open("a+", encoding="utf-8")

    def _operation(self) -> int:
        return fcntl.LOCK_SH if self.mode == "shared" else fcntl.LOCK_EX

    def _write_holder_record(self) -> None:
        holders_dir = self.lock_path.parent / f"{self.lock_path.name}.holders"
        holders_dir.mkdir(parents=True, exist_ok=True)
        holder_path = holders_dir / f"{os.getpid()}-{uuid.uuid4().hex}.json"
        payload = {
            "schema_version": 1,
            "resource": self.resource,
            "mode": self.mode,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "acquired_at": datetime.now(UTC).isoformat(),
            "monotonic_seconds": time.monotonic(),
        }
        holder_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self._holder_path = holder_path


@dataclass(frozen=True, slots=True)
class LauncherLeases:
    """Canonical resource names shared by every launcher entry point."""

    repository_root: Path

    def build(self, _build_type: str) -> ResourceLease:
        target = (self.repository_root / "target").resolve()
        return self._lease(f"build:{target}", "exclusive")

    def profile_database(self, *, write: bool) -> ResourceLease:
        database = (self.repository_root / "profiling" / "profile.db").resolve()
        return self._lease(
            f"profile-db:{database}", "exclusive" if write else "shared"
        )

    def run_directory(self, log_dir: Path) -> ResourceLease:
        return self._lease(f"run-dir:{log_dir.resolve()}", "exclusive")

    def _lease(self, resource: str, mode: LeaseMode) -> ResourceLease:
        digest = hashlib.sha256(resource.encode()).hexdigest()
        lock_path = launcher_lock_root(self.repository_root) / f"{digest}.lock"
        return ResourceLease(lock_path=lock_path, resource=resource, mode=mode)
