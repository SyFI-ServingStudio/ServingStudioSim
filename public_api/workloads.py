"""Where a simulation's requests come from, besides a capture: a trace that
req-frontend's ``tracegen`` draws, or a CSV a reader uploads.

Nothing here knows a trace's columns or rules. The generator's knobs are
``tracegen describe`` (its own parser, :data:`TRACEGEN`), and a request's
``generator`` object becomes that generator's command line; tracegen checks the
values. A trace, whatever made it, is read by ``simulator workload-plan``,
which loads a run config's ``workload`` block exactly as a run does (format,
tags, every row) and prints its requests; the formats an upload may declare are
``simulator trace-formats``. What this module adds is the service's own limits:
which generators it offers, how long one may run, how large an upload may be,
and how long an upload is kept.

Uploads live in their own directory, one per upload (``trace.csv`` and its
record, ``upload.json``), and are removed a day after they arrive.
"""

from __future__ import annotations

import json
import resource
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from alignment.load_generator.runner import TRACEGEN
from launcher.exec import _build_subprocess_env, binary_path

# The generators the service runs: `coding-session` materializes a corpus file
# from this host, which a reader cannot name.
GENERATORS = ("synthetic",)
# Arguments the service sets itself.
_SERVICE_ARGUMENTS = {"out"}
GENERATE_TIMEOUT_S = 30.0
# Address space a generator may take: a request for millions of rounds is
# stopped here rather than by the host.
_GENERATE_MEMORY = 2 << 30
MAX_UPLOAD_BYTES = 1 << 20
KEEP_S = 24 * 3600
# An upload's acceptance is the simulation's `accept_rate`, checked against the
# worker's draft width; a per-row column would bypass that check.
_UPLOAD_REFUSED_TAGS = {"speculative": "give acceptance as the simulation's workload.accept_rate"}
_RECORD = "upload.json"
_TRACE = "trace.csv"


class BadWorkload(ValueError):
    """A workload this member cannot run; the message says why."""


@dataclass(frozen=True)
class TraceSource:
    """The file a simulation's requests come from, as the simulator reads it."""

    path: Path
    input_file_format: str
    input_file_tags: tuple[str, ...] = ()
    # Which requests these are, in an error message: `capture X`, `upload <id>`.
    name: str = "the trace"


def _simulator(args: list[str], build_type: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(binary_path(build_type)), *args],
        capture_output=True,
        text=True,
        env=_build_subprocess_env(),
        check=False,
    )


def _error(stderr: str, *paths: Path | str) -> str:
    """A tool's error for a reader: without its `Error: ` prefix, and each of
    ``paths`` (this host's files) by its name only."""
    message = stderr.strip().removeprefix("Error: ")
    for path in sorted(map(str, paths), key=len, reverse=True):
        message = message.replace(path, Path(path).name)
    return message


def plan(block: dict, build_type: str = "release") -> list[dict]:
    """The requests a run of the ``workload`` block ``block`` releases, one per
    row, as ``simulator workload-plan`` loads them; raises
    :class:`BadWorkload` with the simulator's reason when a run could not."""
    with tempfile.TemporaryDirectory(prefix="public-workload-") as scratch:
        path = Path(scratch) / "workload.json"
        path.write_text(json.dumps(block))
        result = _simulator(["workload-plan", str(path)], build_type)
    if result.returncode:
        raise BadWorkload(_error(result.stderr, *block["trace_files"], path))
    return json.loads(result.stdout)["requests"]


def _limit_memory() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_GENERATE_MEMORY, _GENERATE_MEMORY))


class Workloads:
    """The generated and uploaded workloads: tracegen's description, its runs,
    and the uploads kept under ``uploads_dir``."""

    def __init__(self, uploads_dir: Path, build_type: str = "release") -> None:
        self.uploads_dir = Path(uploads_dir).resolve()
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.build_type = build_type
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[float, dict]] = {}

    def _cached(self, key: str, binary: Path, args: list[str]) -> dict:
        """A JSON introspection command's answer, reused until ``binary`` changes."""
        stamp = binary.stat().st_mtime
        with self._lock:
            if key in self._cache and self._cache[key][0] == stamp:
                return self._cache[key][1]
        result = subprocess.run(
            [str(binary), *args],
            capture_output=True,
            text=True,
            env=_build_subprocess_env(),
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"{binary.name} {args[0]}: {result.stderr.strip()}")
        value = json.loads(result.stdout)
        with self._lock:
            self._cache[key] = (stamp, value)
        return value

    # -- generated -------------------------------------------------------------

    def generators(self) -> dict:
        """``tracegen describe`` for the generators the service runs, without
        the arguments it sets itself."""
        described = self._cached("tracegen", TRACEGEN, ["describe"])
        return {
            "input_file_format": described["input_file_format"],
            "generators": [
                generator
                | {
                    "arguments": [
                        argument
                        for argument in generator["arguments"]
                        if argument["name"] not in _SERVICE_ARGUMENTS
                    ]
                }
                for generator in described["generators"]
                if generator["name"] in GENERATORS
            ],
        }

    def generate(self, spec: dict, out: Path) -> TraceSource:
        """Run the generator ``spec`` names (``{"type": ..., <argument>: value}``)
        into ``out``; tracegen writes its manifest and plan beside it."""
        offered = {generator["name"]: generator for generator in self.generators()["generators"]}
        kind = spec.get("type")
        if kind not in offered:
            raise BadWorkload(f"workload.generator.type must be one of {sorted(offered)}")
        arguments = {argument["name"]: argument for argument in offered[kind]["arguments"]}
        unknown = sorted(set(spec) - {"type"} - set(arguments))
        if unknown:
            raise BadWorkload(
                f"generator {kind} takes {sorted(arguments)}; unknown {unknown} "
                "(GET /workloads describes each)"
            )
        argv = [str(TRACEGEN), kind, "--out", str(out)]
        for name, value in spec.items():
            if name == "type" or value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise BadWorkload(f"generator {kind}: {name} takes one number or string")
            argv += [arguments[name]["flag"], str(value)]
        try:
            result = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=GENERATE_TIMEOUT_S,
                preexec_fn=_limit_memory,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise BadWorkload(
                f"generator {kind} did not finish within {GENERATE_TIMEOUT_S:.0f} s; "
                "ask for fewer sessions or rounds"
            ) from None
        if result.returncode:
            message = _error(result.stderr, out).split("\n\nFor more information", 1)[0]
            if not message:  # killed at the memory limit
                message = "it ran out of memory; ask for fewer sessions or rounds"
            raise BadWorkload(f"generator {kind}: {message}")
        return TraceSource(
            path=out,
            input_file_format=self.generators()["input_file_format"],
            name="the generated trace",
        )

    # -- uploaded --------------------------------------------------------------

    def trace_formats(self) -> dict:
        """The formats an upload may declare (``simulator trace-formats``), with
        the tags the service takes in an upload."""
        described = self._cached("trace-formats", binary_path(self.build_type), ["trace-formats"])
        return {
            "formats": [
                fmt | {"tags": [tag for tag in fmt["tags"] if tag not in _UPLOAD_REFUSED_TAGS]}
                for fmt in described["formats"]
            ],
            "tags": [tag for tag in described["tags"] if tag["name"] not in _UPLOAD_REFUSED_TAGS],
        }

    def upload(self, body: bytes, input_file_format: str, input_file_tags: list[str]) -> dict:
        """Keep an uploaded trace once the simulator reads it; its record."""
        from public_api.simulate import MAX_REQUESTS

        for tag in input_file_tags:
            if tag in _UPLOAD_REFUSED_TAGS:
                raise BadWorkload(f"an upload takes no `{tag}` tag: {_UPLOAD_REFUSED_TAGS[tag]}")
        self.prune()
        workload_id = uuid.uuid4().hex
        directory = self.uploads_dir / workload_id
        directory.mkdir()
        try:
            trace = directory / _TRACE
            trace.write_bytes(body)
            requests = plan(
                {
                    "trace_files": [str(trace)],
                    "input_file_format": input_file_format,
                    "input_file_tags": input_file_tags,
                    # Replay settings are the simulation's; these only read the file.
                    "arrival_mode": "trace_timed",
                    "session_dependency": "independent",
                    "request_rate": 1.0,
                    "run_to_end": True,
                    "duration_ms": 1.0,
                },
                self.build_type,
            )
            if len(requests) > MAX_REQUESTS:
                raise BadWorkload(
                    f"the upload has {len(requests)} requests; a simulation runs at most "
                    f"{MAX_REQUESTS}"
                )
            created = time.time()
            record = {
                "workload_id": workload_id,
                "input_file_format": input_file_format,
                "input_file_tags": input_file_tags,
                "requests": len(requests),
                "created_at": created,
            }
            (directory / _RECORD).write_text(json.dumps(record))
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return self._document(record)

    def _document(self, record: dict) -> dict:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record["created_at"] + KEEP_S))
        return {k: v for k, v in record.items() if k != "created_at"} | {"expires_at": stamp}

    def uploaded(self, workload_id: str) -> TraceSource:
        """The trace of a kept upload."""
        directory = self.uploads_dir / workload_id
        record_path = directory / _RECORD
        if not workload_id.isalnum() or not record_path.is_file():
            raise BadWorkload(
                f"no upload {workload_id!r}: POST /workloads first; an upload is kept "
                f"{KEEP_S // 3600} h"
            )
        record = json.loads(record_path.read_text())
        return TraceSource(
            path=directory / _TRACE,
            input_file_format=record["input_file_format"],
            input_file_tags=tuple(record["input_file_tags"]),
            name=f"upload {workload_id}",
        )

    def prune(self) -> None:
        """Remove the uploads older than :data:`KEEP_S`."""
        cutoff = time.time() - KEEP_S
        for directory in self.uploads_dir.iterdir():
            record = directory / _RECORD
            try:
                created = json.loads(record.read_text())["created_at"]
            except (OSError, ValueError, KeyError):
                created = directory.stat().st_mtime
            if created < cutoff:
                shutil.rmtree(directory, ignore_errors=True)


def describe(workloads: Workloads, limits: dict, schema: dict) -> dict:
    """``GET /workloads``: every source a simulation's requests can come from."""
    return {
        "sources": {
            "capture": {
                "summary": "A recorded trace a member offers: /simulations/presets lists "
                "each preset's captures. Its requests and its routing are the capture's.",
            },
            "generated": {
                "summary": "A trace req-frontend's tracegen draws from `generator`: "
                "`type` and the generator's arguments, each a string as its flag takes it.",
                **workloads.generators(),
            },
            "upload": {
                "summary": "A CSV you send to POST /workloads (body: the file; query: "
                "`format`, and `tags` comma-separated); the answer's workload_id is the "
                "simulation's workload.upload.",
                "max_bytes": MAX_UPLOAD_BYTES,
                "keep_s": KEEP_S,
                **workloads.trace_formats(),
            },
        },
        "routing": "An MoE member routes every workload's tokens as one of its captures does: "
        "workload.capture names it, its first by default.",
        "limits": limits,
        "workload": schema,
    }
