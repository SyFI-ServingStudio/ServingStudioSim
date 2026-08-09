"""Typed input and output contracts for launcher-owned child processes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ProcessSpec:
    """One root process supervised independently of its output descriptors.

    ``log_path`` and ``capture_output`` are mutually exclusive.  Both use a
    regular file, never a pipe, so an inherited descriptor cannot delay root
    process completion.
    """

    argv: Sequence[str]
    cwd: Path
    env: Mapping[str, str] | None = None
    log_path: Path | None = None
    append_log: bool = False
    capture_output: bool = False
    input_bytes: bytes | None = None
    timeout_seconds: float | None = None
    name: str = "process"

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("process argv must not be empty")
        if self.log_path is not None and self.capture_output:
            raise ValueError("log_path and capture_output are mutually exclusive")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class ProcessResult:
    """Observed root-process outcome plus process-group cleanup evidence."""

    argv: tuple[str, ...]
    pid: int
    process_group_id: int
    exit_code: int
    elapsed_seconds: float
    output: str = ""
    leaked_descendants: bool = False
    termination_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return (
            self.exit_code == 0
            and not self.leaked_descendants
            and self.termination_reason is None
        )
