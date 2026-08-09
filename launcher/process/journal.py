"""Durable per-run stage journal.

The journal explains what the launcher was waiting for or validating after a
crash.  Writes use replace-on-the-same-filesystem so readers never observe a
partial JSON document.
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .spec import ProcessResult, ProcessSpec


class StageState(StrEnum):
    PENDING = "PENDING"
    WAITING_RESOURCE = "WAITING_RESOURCE"
    RUNNING = "RUNNING"
    EXITED = "EXITED"
    VALIDATING = "VALIDATING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class StageRecord:
    schema_version: int
    stage: str
    attempt: int
    state: str
    created_at: str
    updated_at: str
    waiting_since: str | None
    started_at: str | None
    exited_at: str | None
    finished_at: str | None
    launcher_pid: int
    launcher_host: str
    argv: list[str] | None = None
    resources: list[str] | None = None
    child_pid: int | None = None
    process_group_id: int | None = None
    exit_code: int | None = None
    elapsed_seconds: float | None = None
    leaked_descendants: bool | None = None
    termination_reason: str | None = None
    artifacts: list[str] | None = None
    error: str | None = None


class RunJournal:
    """Own ``<log_dir>/.launcher`` state for one concrete run."""

    def __init__(self, log_dir: Path) -> None:
        self.root = log_dir / ".launcher"
        self.stages_dir = self.root / "stages"

    def begin_attempts(self, stages: Sequence[str]) -> None:
        """Persist one new attempt for each selected stage in the compiled graph."""

        for stage in stages:
            stage_path = self.stages_dir / f"{stage}.json"
            self.update(
                stage,
                StageState.PENDING,
                new_attempt=stage_path.exists(),
            )

    def update(
        self,
        stage: str,
        state: StageState,
        *,
        spec: ProcessSpec | None = None,
        result: ProcessResult | None = None,
        resources: Sequence[str] | None = None,
        artifacts: Sequence[Path] | None = None,
        error: str | None = None,
        new_attempt: bool = False,
    ) -> None:
        stage_path = self.stages_dir / f"{stage}.json"
        previous = self._read_existing(stage_path)
        previous_attempt = previous.get("attempt", 1)
        terminal_states = {
            StageState.SUCCEEDED.value,
            StageState.FAILED.value,
            StageState.CANCELLED.value,
        }
        restarting = previous.get("state") in terminal_states and state in {
            StageState.WAITING_RESOURCE,
            StageState.RUNNING,
        }
        if new_attempt:
            attempt = int(previous_attempt) + (1 if previous else 0)
            previous = {}
        elif restarting:
            attempt = int(previous_attempt) + 1
            previous = {}
        else:
            attempt = int(previous_attempt)
        updated_at = datetime.now(UTC).isoformat()
        waiting_since = previous.get("waiting_since")
        started_at = previous.get("started_at")
        exited_at = previous.get("exited_at")
        finished_at = previous.get("finished_at")
        if state == StageState.WAITING_RESOURCE and waiting_since is None:
            waiting_since = updated_at
        if state == StageState.RUNNING and started_at is None:
            started_at = updated_at
        if state == StageState.EXITED:
            exited_at = updated_at
        if state in {
            StageState.SUCCEEDED,
            StageState.FAILED,
            StageState.CANCELLED,
        }:
            finished_at = updated_at

        record = StageRecord(
            schema_version=1,
            stage=stage,
            attempt=attempt,
            state=state.value,
            created_at=previous.get("created_at", updated_at),
            updated_at=updated_at,
            waiting_since=waiting_since,
            started_at=started_at,
            exited_at=exited_at,
            finished_at=finished_at,
            launcher_pid=os.getpid(),
            launcher_host=socket.gethostname(),
            argv=(
                [str(argument) for argument in spec.argv]
                if spec
                else previous.get("argv")
            ),
            resources=(list(resources) if resources else previous.get("resources")),
            child_pid=result.pid if result else previous.get("child_pid"),
            process_group_id=(
                result.process_group_id
                if result
                else previous.get("process_group_id")
            ),
            exit_code=result.exit_code if result else previous.get("exit_code"),
            elapsed_seconds=(
                result.elapsed_seconds
                if result
                else previous.get("elapsed_seconds")
            ),
            leaked_descendants=(
                result.leaked_descendants
                if result
                else previous.get("leaked_descendants")
            ),
            termination_reason=(
                result.termination_reason
                if result
                else previous.get("termination_reason")
            ),
            artifacts=(
                [str(path) for path in artifacts]
                if artifacts
                else previous.get("artifacts")
            ),
            error=error if error is not None else previous.get("error"),
        )
        self.stages_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_json(stage_path, asdict(record))
        self._write_run_state()

    @staticmethod
    def _read_existing(path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_run_state(self) -> None:
        stages: dict[str, Any] = {}
        for path in sorted(self.stages_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            stages[path.stem] = {
                "state": payload.get("state"),
                "updated_at": payload.get("updated_at"),
            }
        self._atomic_json(
            self.root / "run_state.json",
            {
                "schema_version": 1,
                "updated_at": datetime.now(UTC).isoformat(),
                "stages": stages,
            },
        )

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
