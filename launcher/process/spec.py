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
    #: Capture stderr into its own file instead of folding it into stdout.
    #: Needed when the child's stdout is a machine-readable document and its
    #: stderr is a human log — merging them corrupts the document. Off by
    #: default: for a child whose output is only ever read by a human, one
    #: interleaved stream preserves the ordering between the two.
    separate_stderr: bool = False
    input_bytes: bytes | None = None
    timeout_seconds: float | None = None
    name: str = "process"

    def __post_init__(self) -> None:
        if not self.argv:
            raise ValueError("process argv must not be empty")
        if self.log_path is not None and self.capture_output:
            raise ValueError("log_path and capture_output are mutually exclusive")
        if self.separate_stderr and not self.capture_output:
            raise ValueError("separate_stderr requires capture_output")
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
    #: The child's stderr, separately, when the spec asked for
    #: ``separate_stderr``. Empty otherwise — stderr is then already inside
    #: ``output``.
    stderr_output: str = ""
    leaked_descendants: bool = False
    termination_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return (
            self.exit_code == 0
            and not self.leaked_descendants
            and self.termination_reason is None
        )
