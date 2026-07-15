"""Durable publication state for one post-run analyzer generation.

The launcher owns this sidecar because it is the only component that observes
all three producer stages (compute, render, and trace).  Readers must use the
sidecar instead of inferring completion from whichever artifact happened to be
written last.
"""

from __future__ import annotations

import atexit
import fcntl
import json
import os
import shutil
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO, Final, Literal

PIPELINE_SCHEMA_VERSION: Final = 1
ANALYZER_IDENTITY_SCHEMA_VERSION: Final = 1
PIPELINE_STATE_RELATIVE_PATH: Final = Path("reports/analyzer_pipeline_state.json")
PIPELINE_PRODUCER_NAME: Final = "vibesim-analyzer"

StageName = Literal["compute", "render", "trace"]
StageStatus = Literal["not_started", "pending", "complete", "failed"]
PipelineStatus = Literal["pending", "complete", "failed"]


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


@dataclass(frozen=True)
class AnalyzerBinaryContract:
    """Machine contract owned by one concrete analyzer executable.

    ``subject_scopes`` comes from the Rust registry in that same executable, so
    the launcher can narrow intent without maintaining a second subject table.
    """

    version: str
    revision: str
    binary_sha256: str
    subject_scopes: tuple[tuple[str, str], ...]
    binary_fingerprint: tuple[int, int, int, int]

    @property
    def producer(self) -> dict[str, object]:
        return {
            "name": PIPELINE_PRODUCER_NAME,
            "version": self.version,
            "revision": self.revision,
            "binary_sha256": self.binary_sha256,
        }

    def canonical_subjects(
        self, requested: list[str] | None, *, scope: Literal["run", "alignment"]
    ) -> list[str] | None:
        """Validate explicit intent and return registry-order scope tokens."""

        if not requested:
            return None
        scope_by_name = dict(self.subject_scopes)
        for token in requested:
            token_scope = scope_by_name.get(token)
            if token_scope is None:
                known = [name for name, row_scope in self.subject_scopes if row_scope == scope]
                raise ValueError(
                    f"unknown {scope} analyzer subject {token!r}; known: {', '.join(known)}"
                )
            if token_scope != scope:
                raise ValueError(
                    f"analyzer subject {token!r} has {token_scope} scope and cannot be used "
                    f"with `analyze {scope}`"
                )
        wanted = set(requested)
        return [
            name for name, row_scope in self.subject_scopes if row_scope == scope and name in wanted
        ]

    def subjects_in_scope(self, scope: Literal["run", "alignment"]) -> list[str]:
        """Return every binary-registry token in one source scope."""

        return [name for name, row_scope in self.subject_scopes if row_scope == scope]

    def canonical_run_subjects(self, requested: list[str] | None) -> list[str] | None:
        return self.canonical_subjects(requested, scope="run")

    def matches_executable(self, analyzer: Path) -> bool:
        """Whether ``analyzer`` still names the executable that was hashed."""

        try:
            current = analyzer.stat()
        except OSError:
            return False
        return (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        ) == self.binary_fingerprint


_SNAPSHOT_LOCK = threading.Lock()
_SNAPSHOT_CACHE: dict[tuple[int, int, int, int], tuple[Path, AnalyzerBinaryContract]] = {}
_SNAPSHOT_DIRECTORIES: dict[Path, Path] = {}


def _stat_fingerprint(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _snapshot_directory(target_profile_directory: Path) -> Path:
    existing = _SNAPSHOT_DIRECTORIES.get(target_profile_directory)
    if existing is not None:
        return existing
    directory = (
        target_profile_directory / ".analyzer-snapshots" / f"{os.getpid()}-{uuid.uuid4().hex}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    _SNAPSHOT_DIRECTORIES[target_profile_directory] = directory
    atexit.register(shutil.rmtree, directory, ignore_errors=True)
    return directory


def discover_analyzer_contract(analyzer: Path, repo_root: Path) -> AnalyzerBinaryContract:
    """Read identity and subject metadata from the executable that will run."""

    analyzer_stat = analyzer.stat()
    return _discover_analyzer_contract_cached(
        str(analyzer),
        analyzer_stat.st_dev,
        analyzer_stat.st_ino,
        analyzer_stat.st_size,
        analyzer_stat.st_mtime_ns,
        str(repo_root),
    )


def snapshot_analyzer_binary(
    analyzer: Path, repo_root: Path
) -> tuple[Path, AnalyzerBinaryContract]:
    """Publish and validate the exact executable used by one generation.

    Cargo may atomically replace ``target/<profile>/analyze`` while a large
    sweep is still publishing earlier runs. An atomic hard link pins one inode
    without copying the (often ~1 GB debug) file. The process-wide lock/cache is
    also the singleflight boundary: one sweep hashes and queries each inode once,
    then every run executes the same pinned path. Per-process links are removed
    at normal interpreter exit.
    """

    with _SNAPSHOT_LOCK:
        for _attempt in range(3):
            source_before = _stat_fingerprint(analyzer)
            cached = _SNAPSHOT_CACHE.get(source_before)
            if cached is not None:
                snapshot, contract = cached
                if contract.matches_executable(snapshot):
                    return snapshot, contract
                _SNAPSHOT_CACHE.pop(source_before, None)

            directory = _snapshot_directory(analyzer.parent)
            temporary_path = directory / f".analyze-{uuid.uuid4().hex}.tmp"
            os.link(analyzer, temporary_path)
            pinned = _stat_fingerprint(temporary_path)
            try:
                source_after = _stat_fingerprint(analyzer)
            except OSError:
                source_after = None
            if source_before != pinned or source_after != pinned:
                temporary_path.unlink()
                continue

            try:
                contract = discover_analyzer_contract(temporary_path, repo_root)
                digest_hex = contract.binary_sha256.removeprefix("sha256:")
                if len(digest_hex) != 64:
                    raise RuntimeError("analyzer snapshot has no concrete binary digest")
                snapshot = directory / f"analyze-{digest_hex}"
                if snapshot.exists():
                    temporary_path.unlink()
                    existing_contract = next(
                        (
                            cached_contract
                            for cached_snapshot, cached_contract in _SNAPSHOT_CACHE.values()
                            if cached_snapshot == snapshot
                            and cached_contract.matches_executable(snapshot)
                        ),
                        None,
                    )
                    contract = existing_contract or discover_analyzer_contract(snapshot, repo_root)
                    if contract.binary_sha256 != f"sha256:{digest_hex}":
                        raise RuntimeError("existing analyzer snapshot violates its digest name")
                else:
                    temporary_path.rename(snapshot)
                if not contract.matches_executable(snapshot):
                    raise RuntimeError("analyzer snapshot inode changed during publication")
                _SNAPSHOT_CACHE[pinned] = (snapshot, contract)
                return snapshot, contract
            except BaseException:
                temporary_path.unlink(missing_ok=True)
                raise
        raise RuntimeError("Cargo analyzer output changed repeatedly while being pinned")


@lru_cache(maxsize=8)
def _discover_analyzer_contract_cached(
    analyzer_name: str,
    analyzer_device: int,
    analyzer_inode: int,
    analyzer_size: int,
    analyzer_mtime_ns: int,
    repo_root_name: str,
) -> AnalyzerBinaryContract:
    """Cache identity work for the one binary shared by a large sweep."""

    del analyzer_device, analyzer_inode, analyzer_size, analyzer_mtime_ns
    # The deleted values intentionally remain part of the cache key.
    analyzer = Path(analyzer_name)
    repo_root = Path(repo_root_name)
    before = analyzer.stat()
    result = subprocess.run(
        [str(analyzer), "identity"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"analyzer identity command failed with status {result.returncode}: "
            f"{result.stderr.strip()}"
        )
    try:
        identity = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("analyzer identity command returned invalid JSON") from error
    if not isinstance(identity, dict):
        raise RuntimeError("analyzer identity must be a JSON object")
    if identity.get("schema_version") != ANALYZER_IDENTITY_SCHEMA_VERSION:
        raise RuntimeError("unsupported analyzer identity schema_version")
    if identity.get("name") != PIPELINE_PRODUCER_NAME:
        raise RuntimeError("analyzer identity has an unexpected producer name")

    version = identity.get("version")
    revision = identity.get("revision")
    if not isinstance(version, str) or not version or len(version) > 160:
        raise RuntimeError("analyzer identity has no bounded concrete version")
    if version == "unavailable":
        raise RuntimeError("analyzer identity version is unavailable")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise RuntimeError("analyzer identity revision is not a full source revision")

    raw_subjects = identity.get("subjects")
    if not isinstance(raw_subjects, list) or not raw_subjects:
        raise RuntimeError("analyzer identity has no subject registry")
    subject_scopes: list[tuple[str, str]] = []
    seen_subjects: set[str] = set()
    for row in raw_subjects:
        if not isinstance(row, dict) or set(row) != {"name", "scope"}:
            raise RuntimeError("analyzer identity subject row has an invalid shape")
        name = row["name"]
        scope = row["scope"]
        if not isinstance(name, str) or not name or len(name) > 160 or name in seen_subjects:
            raise RuntimeError("analyzer identity contains a missing or duplicate subject")
        if scope not in {"run", "alignment"}:
            raise RuntimeError(f"analyzer identity subject {name!r} has invalid scope")
        seen_subjects.add(name)
        subject_scopes.append((name, scope))
    if not any(scope == "run" for _, scope in subject_scopes):
        raise RuntimeError("analyzer identity has no Run-scope subjects")

    digest = sha256()
    with analyzer.open("rb") as analyzer_file:
        while chunk := analyzer_file.read(1024 * 1024):
            digest.update(chunk)
    after = analyzer.stat()
    before_signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if after_signature != before_signature:
        raise RuntimeError("analyzer executable changed while its identity was read")
    return AnalyzerBinaryContract(
        version=version,
        revision=revision,
        binary_sha256=f"sha256:{digest.hexdigest()}",
        subject_scopes=tuple(subject_scopes),
        binary_fingerprint=before_signature,
    )


@dataclass
class AnalyzerPipelinePublisher:
    """Publish the state machine for one analyzer artifact generation.

    Public transition methods own whole stage boundaries. Keep the underlying
    stage mutation private: publishing half of a boundary can create a durable
    snapshot that the Rust lifecycle validator must reject.
    """

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
            # The launcher resolves this through the executing binary's registry
            # contract before entering the publication state machine.
            "requested_subjects": (list(requested_subjects) if requested_subjects else None),
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

    @property
    def generation_id(self) -> str:
        generation_id = self.state["generation_id"]
        assert isinstance(generation_id, str)
        return generation_id

    def complete_compute_and_start_render(self) -> None:
        """Atomically publish the successful compute-to-render handoff."""

        self._require_shape("pending", "pending", "not_started", "not_started")
        timestamp = _utc_now()
        self._update_stage("compute", "complete", timestamp)
        self._update_stage("render", "pending", timestamp)
        self._publish_at(timestamp)

    def fail_compute_and_finish(self, code: str) -> None:
        """Atomically publish a terminal compute failure."""

        self._require_shape("pending", "pending", "not_started", "not_started")
        timestamp = _utc_now()
        self._update_stage("compute", "failed", timestamp, code=code)
        self.state["status"] = "failed"
        self.state["failure_code"] = code
        self.state["completed_at"] = timestamp
        self._publish_at(timestamp)

    def complete_render_and_start_trace(self) -> None:
        """Atomically publish the successful render-to-trace handoff."""

        self._finish_render_and_start_trace("complete")

    def fail_render_and_start_trace(self, code: str) -> None:
        """Atomically publish an optional render failure and start trace."""

        self._finish_render_and_start_trace("failed", code=code)

    def complete_trace_and_finish(self, *, artifact: str) -> None:
        """Atomically publish a successful trace and terminal generation."""

        self._finish_trace("complete", artifact=artifact)

    def fail_trace_and_finish(self, code: str) -> None:
        """Atomically publish an optional trace failure and terminal generation."""

        self._finish_trace("failed", code=code)

    def _finish_render_and_start_trace(
        self,
        render_status: Literal["complete", "failed"],
        *,
        code: str | None = None,
    ) -> None:
        self._require_shape("pending", "complete", "pending", "not_started")
        timestamp = _utc_now()
        self._update_stage("render", render_status, timestamp, code=code)
        self._update_stage("trace", "pending", timestamp)
        self._publish_at(timestamp)

    def _finish_trace(
        self,
        trace_status: Literal["complete", "failed"],
        *,
        code: str | None = None,
        artifact: str | None = None,
    ) -> None:
        stages = self._stages()
        render_status = stages["render"]["status"]
        if render_status not in {"complete", "failed"}:
            raise ValueError("trace cannot finish before render reaches a terminal state")
        self._require_shape("pending", "complete", render_status, "pending")
        timestamp = _utc_now()
        self._update_stage(
            "trace",
            trace_status,
            timestamp,
            code=code,
            artifact=artifact,
        )
        # Complete means orchestration reached a terminal state. Optional
        # render/trace failures remain explicit on their stages without hiding
        # valid current-generation JSON subjects.
        self.state["status"] = "complete"
        self.state.pop("failure_code", None)
        self.state["completed_at"] = timestamp
        self._publish_at(timestamp)

    def _publish_at(self, timestamp: str) -> None:
        self.state["updated_at"] = timestamp
        self._publish()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            fcntl.flock(self._lease_file.fileno(), fcntl.LOCK_UN)
        finally:
            self._lease_file.close()

    def _update_stage(
        self,
        stage: StageName,
        status: StageStatus,
        timestamp: str,
        *,
        code: str | None = None,
        artifact: str | None = None,
    ) -> None:
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

    def _require_shape(
        self,
        pipeline: PipelineStatus,
        compute: StageStatus,
        render: StageStatus,
        trace: StageStatus,
    ) -> None:
        stages = self._stages()
        observed = (
            self.state["status"],
            stages["compute"]["status"],
            stages["render"]["status"],
            stages["trace"]["status"],
        )
        expected = (pipeline, compute, render, trace)
        if observed != expected:
            raise ValueError(
                f"analyzer pipeline transition expected {expected}, observed {observed}"
            )

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
