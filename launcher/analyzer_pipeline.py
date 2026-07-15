"""Durable publication state for one post-run analyzer generation.

The launcher owns this sidecar because it is the only component that observes
all three producer stages (compute, render, and trace).  Readers must use the
sidecar instead of inferring completion from whichever artifact happened to be
written last.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO, Final, Literal

PIPELINE_SCHEMA_VERSION: Final = 1
PIPELINE_STATE_RELATIVE_PATH: Final = Path("reports/analyzer_pipeline_state.json")
PIPELINE_PRODUCER_NAME: Final = "vibesim-analyzer"

StageName = Literal["compute", "render", "trace"]
StageStatus = Literal["not_started", "pending", "complete", "failed"]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def atomic_write_json(path: Path, value: object) -> None:
    """Durably replace ``path`` with one complete JSON value.

    The temporary file deliberately lives beside the destination: ``os.replace``
    is only guaranteed atomic within one filesystem.  Syncing both the file and
    parent directory makes the transition durable across a host crash.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as temporary_file:
            json.dump(value, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)

        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def discover_producer_identity(analyzer: Path, repo_root: Path) -> dict[str, object]:
    """Return the producer identity recorded in a new pipeline generation.

    The version comes from the binary that will actually run, not the serving
    binary.  The revision describes the source checkout from which the launcher
    builds that analyzer.  Failed generations may report an
    unavailable version, but successful generations should always have the
    concrete analyzer version enabled by clap's ``--version`` flag.
    """

    try:
        analyzer_stat = analyzer.stat()
    except OSError:
        analyzer_stat = None
    return dict(
        _discover_producer_identity_cached(
            str(analyzer),
            analyzer_stat.st_size if analyzer_stat else -1,
            analyzer_stat.st_mtime_ns if analyzer_stat else -1,
            str(repo_root),
        )
    )


@lru_cache(maxsize=8)
def _discover_producer_identity_cached(
    analyzer_name: str,
    analyzer_size: int,
    analyzer_mtime_ns: int,
    repo_root_name: str,
) -> tuple[tuple[str, object], ...]:
    """Cache identity work for the one binary shared by a large sweep."""

    del analyzer_size, analyzer_mtime_ns  # They intentionally form the cache key.
    analyzer = Path(analyzer_name)
    repo_root = Path(repo_root_name)
    version = "unavailable"
    binary_sha256 = "unavailable"
    if analyzer.is_file():
        result = subprocess.run(
            [str(analyzer), "--version"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            words = result.stdout.strip().split()
            if len(words) >= 2:
                version = words[-1]
        digest = sha256()
        with analyzer.open("rb") as analyzer_file:
            while chunk := analyzer_file.read(1024 * 1024):
                digest.update(chunk)
        binary_sha256 = f"sha256:{digest.hexdigest()}"

    revision_result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    revision = revision_result.stdout.strip() if revision_result.returncode == 0 else "unavailable"
    return tuple(
        {
            "name": PIPELINE_PRODUCER_NAME,
            "version": version,
            "revision": revision,
            "binary_sha256": binary_sha256,
        }.items()
    )


@dataclass
class AnalyzerPipelinePublisher:
    """Publish the state machine for one analyzer artifact generation."""

    state_path: Path
    state: dict[str, object]
    _lease_file: BinaryIO = field(repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def begin(
        cls,
        log_dir: Path,
        producer: dict[str, object],
        requested_subjects: list[str] | None = None,
    ) -> AnalyzerPipelinePublisher:
        state_path = log_dir / PIPELINE_STATE_RELATIVE_PATH
        state_path.parent.mkdir(parents=True, exist_ok=True)
        lease_path = state_path.parent / ".analyzer_pipeline_state.lock"
        lease_file = lease_path.open("a+b")
        # The lease spans compute, render, and trace. Callers acquire it in a
        # worker thread so a second generation may wait without blocking the
        # launcher's asyncio event loop.
        fcntl.flock(lease_file.fileno(), fcntl.LOCK_EX)
        generation_id = uuid.uuid4().hex
        timestamp = _utc_now()
        state: dict[str, object] = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "generation_id": generation_id,
            "artifact_revision": f"pipeline-{generation_id}",
            "status": "pending",
            "started_at": timestamp,
            "updated_at": timestamp,
            "producer": producer,
            "requested_subjects": (sorted(set(requested_subjects)) if requested_subjects else None),
            "stages": {
                "compute": {"status": "pending", "updated_at": timestamp},
                "render": {"status": "not_started", "updated_at": timestamp},
                "trace": {"status": "not_started", "updated_at": timestamp},
            },
        }
        publisher = cls(state_path=state_path, state=state, _lease_file=lease_file)
        try:
            publisher._publish(claim=True)
            return publisher
        except BaseException:
            publisher.close()
            raise

    def start_stage(self, stage: StageName) -> None:
        self._set_stage(stage, "pending")

    @property
    def generation_id(self) -> str:
        generation_id = self.state["generation_id"]
        assert isinstance(generation_id, str)
        return generation_id

    def complete_stage(self, stage: StageName, *, artifact: str | None = None) -> None:
        self._set_stage(stage, "complete", artifact=artifact)

    def fail_stage(self, stage: StageName, code: str) -> None:
        self._set_stage(stage, "failed", code=code)

    def finish(self) -> None:
        stages = self._stages()
        if stages["compute"]["status"] == "failed":
            self.state["status"] = "failed"
            self.state["failure_code"] = stages["compute"]["code"]
        elif (
            stages["compute"]["status"] == "complete"
            and stages["render"]["status"] in {"complete", "failed"}
            and stages["trace"]["status"] in {"complete", "failed"}
        ):
            # Complete means orchestration reached a terminal state. Optional
            # render/trace failures remain explicit on their stages without
            # hiding valid current-generation JSON subjects.
            self.state["status"] = "complete"
            self.state.pop("failure_code", None)
        else:
            raise ValueError("cannot finish an analyzer pipeline with unfinished stages")

        timestamp = _utc_now()
        self.state["updated_at"] = timestamp
        self.state["completed_at"] = timestamp
        self._publish()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            fcntl.flock(self._lease_file.fileno(), fcntl.LOCK_UN)
        finally:
            self._lease_file.close()

    def _set_stage(
        self,
        stage: StageName,
        status: StageStatus,
        *,
        code: str | None = None,
        artifact: str | None = None,
    ) -> None:
        timestamp = _utc_now()
        stage_state = self._stages()[stage]
        stage_state["status"] = status
        stage_state["updated_at"] = timestamp
        if code is None:
            stage_state.pop("code", None)
        else:
            stage_state["code"] = code
        if artifact is None:
            stage_state.pop("artifact", None)
        else:
            stage_state["artifact"] = artifact
        self.state["updated_at"] = timestamp
        self._publish()

    def _stages(self) -> dict[str, dict[str, object]]:
        stages = self.state["stages"]
        assert isinstance(stages, dict)
        return stages  # type: ignore[return-value]

    def _publish(self, *, claim: bool = False) -> None:
        if self._closed:
            raise RuntimeError("analyzer pipeline publisher no longer owns its run lease")
        if not claim and self.state_path.exists():
            try:
                published = json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise RuntimeError(
                    "cannot verify analyzer pipeline generation ownership"
                ) from error
            if published.get("generation_id") != self.generation_id:
                raise RuntimeError(
                    "analyzer pipeline generation was superseded by a newer publisher"
                )
        atomic_write_json(self.state_path, self.state)
