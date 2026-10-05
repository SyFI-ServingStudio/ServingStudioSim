"""Where a simulation's requests come from, besides a capture: a trace that
req-frontend's ``tracegen`` draws, or a CSV a reader uploads.

Nothing here knows a trace's columns or rules. The generator's knobs are
``tracegen describe`` (its own parser, :data:`TRACEGEN`), and a request's
``generator`` object becomes that generator's command line; tracegen checks the
values. A trace, whatever made it, is read by ``simulator workload-plan``,
which loads a run config's ``workload`` block exactly as a run does (format,
tags, every row) and prints its requests; given the whole run config, it also
refuses the requests a pool cannot serve, as the run would before its first
tick. The formats an upload may declare are ``simulator trace-formats``; an
upload that declares none is read as the one its header fits. What
this module adds is the service's own limits: which generators it offers, how
long one may run, how large an upload may be, and how long an upload is kept.

Uploads live in their own directory, one per upload (``trace.csv`` and its
record, ``upload.json``), and are removed a day after they arrive.
"""

from __future__ import annotations

import csv
import hashlib
import json
import resource
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from alignment.load_generator.runner import TRACEGEN
from launcher.exec import (
    ERROR_JSON,
    binary_error,
    binary_path,
    binary_too_long,
)
from public_api.predict import Refused, _cause
from public_api.sources import introspect, run

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
MAX_REQUESTS = 2000
KEEP_S = 24 * 3600
_RECORD = "upload.json"
_TRACE = "trace.csv"


class BadWorkload(Refused):
    """A workload this member cannot run; the message says why."""


def check_count(name: str, requests: int) -> None:
    """Refuse a trace of more requests than one simulation runs."""
    if requests > MAX_REQUESTS:
        raise BadWorkload(f"{name} has {requests} requests; at most {MAX_REQUESTS} per simulation")


@dataclass(frozen=True)
class TraceSource:
    """The file a simulation's requests come from, as the simulator reads it."""

    path: Path
    input_file_format: str
    input_file_tags: tuple[str, ...] = ()
    # Which requests these are, in an error message: `capture X`, `upload <id>`.
    name: str = "the trace"


def _workload_plan(flags: list[str], document: dict, block: dict, build_type: str) -> list[dict]:
    with tempfile.TemporaryDirectory(prefix="public-workload-") as directory:
        scratch = Path(directory)
        path, error = scratch / "workload.json", scratch / ERROR_JSON
        path.write_text(json.dumps(document))
        args = ["workload-plan", *flags, str(path), "--error-json", str(error)]
        result = run(binary_path(build_type), args)
        if result.returncode:
            traces = {Path(trace).parent for trace in block["trace_files"]}
            cause = _cause(binary_error(error), result.stderr, scratch, *traces)
            raise BadWorkload(cause, binary_too_long(error))
    return json.loads(result.stdout)["requests"]


def plan(block: dict, build_type: str = "release") -> list[dict]:
    """The requests a run of the ``workload`` block ``block`` releases, one per
    row, as ``simulator workload-plan`` loads them; raises
    :class:`BadWorkload` with the simulator's reason when a run could not."""
    return _workload_plan([], block, block, build_type)


def plan_run(config: dict, build_type: str = "release") -> list[dict]:
    """:func:`plan` of the run config ``config``'s workload, also refused, with
    the simulator's reason, when a request does not fit one of its pools
    (``simulator workload-plan --config``: the check a run makes first)."""
    return _workload_plan(["--config"], config, config["workload"], build_type)


def read_block(source: TraceSource) -> dict:
    """A ``workload`` block that only reads ``source``: the replay settings
    are a simulation's."""
    return {
        "trace_files": [str(source.path)],
        "input_file_format": source.input_file_format,
        "input_file_tags": list(source.input_file_tags),
        "arrival_mode": "trace_timed",
        "request_rate": 1.0,
        "run_to_end": True,
        "duration_ms": 1.0,
    }


def trace_facts(requests: list[dict]) -> dict:
    """What a reader is told about a trace, from :func:`plan`'s requests: how
    many there are, their mean prompt (carried prefix included) and output
    tokens, and the rate they arrive at. Only a session's first round arrives
    on the trace's clock (a later round follows its predecessor), so the rate
    counts those, per second over their span; None when they all arrive at
    once."""
    first = sorted(
        float(r["session_arrival_time_ms"]) for r in requests if r["predecessor_request_id"] is None
    )
    span_ms = first[-1] - first[0] if first else 0.0
    count = len(requests)
    mean = lambda values: sum(values) / count if count else None  # noqa: E731
    return {
        "requests": count,
        "sessions": len(first) < count,
        "prompt_tokens": mean(r["prefix_len"] + r["input_len"] for r in requests),
        "output_tokens": mean(r["output_len"] for r in requests),
        "rate": (len(first) - 1) / span_ms * 1000 if span_ms > 0 else None,
    }


def trace_formats(build_type: str = "release") -> dict:
    """``simulator trace-formats``: every trace format, its columns and tags,
    and each tag's columns; read again when the binary changes."""
    return introspect(binary_path(build_type), ["trace-formats"])


def _header(path: Path) -> list[str]:
    with open(path, newline="") as stream:
        return [column.strip() for column in next(csv.reader(stream), [])]


def trace_source(path: Path, name: str, build_type: str = "release") -> TraceSource:
    """A published trace as the simulator reads it: the format and tags its
    header fits (:func:`detect_format`). A speculative capture's trace carries
    the acceptance it recorded, a column of the ``speculative`` tag."""
    fmt, tags = detect_format(_header(path), trace_formats(build_type))
    return TraceSource(Path(path), fmt, tuple(tags), name)


def requests_digest(path: Path, build_type: str = "release") -> str:
    """A digest of a trace's requests: its format's own columns, row by row,
    without the columns a tag adds (such as the acceptance a speculative
    capture recorded), so two traces of the same requests digest alike."""
    source = trace_source(path, str(path), build_type)
    formats = {fmt["name"]: fmt for fmt in trace_formats(build_type)["formats"]}
    columns = formats[source.input_file_format]["columns"]
    digest = hashlib.sha256(source.input_file_format.encode())
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            digest.update(json.dumps([row[c] for c in columns]).encode())
    return digest.hexdigest()


def detect_format(header: list[str], formats: dict) -> tuple[str, list[str]]:
    """The trace format and tags whose columns are exactly ``header``, from
    ``simulator trace-formats``. A format fits when its columns and those of
    some of its tags make up the header; raises when none or several do."""
    have = set(header)
    tag_columns = {tag["name"]: set(tag["columns"]) for tag in formats["tags"]}
    fits = []
    for fmt in formats["formats"]:
        tags = [t for t in fmt["tags"] if tag_columns[t] <= have]
        if set(fmt["columns"]).union(*(tag_columns[t] for t in tags)) == have:
            fits.append((fmt["name"], tags))
    if len(fits) == 1:
        return fits[0]
    accepted = "; ".join(
        f"{fmt['name']}: {', '.join(fmt['columns'])}"
        + (f" (and optionally the columns of {', '.join(fmt['tags'])})" if fmt["tags"] else "")
        for fmt in formats["formats"]
    )
    found = ", ".join(header) or "no columns"
    if not fits:
        raise BadWorkload(f"the header ({found}) fits no trace format. They are {accepted}")
    raise BadWorkload(f"the header ({found}) fits several trace formats; give format")


def _limit_memory() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (_GENERATE_MEMORY, _GENERATE_MEMORY))


class Workloads:
    """The generated and uploaded workloads: tracegen's description, its runs,
    and the uploads kept under ``uploads_dir``."""

    def __init__(self, uploads_dir: Path, build_type: str = "release") -> None:
        self.uploads_dir = Path(uploads_dir).resolve()
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.build_type = build_type

    # -- generated -------------------------------------------------------------

    def generators(self) -> dict:
        """``tracegen describe`` for the generators the service runs, without
        the arguments it sets itself."""
        described = introspect(TRACEGEN, ["describe"])
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
            stderr = result.stderr.strip().removeprefix("Error: ")
            message = _cause(None, stderr, out.parent).split("\n\nFor more information", 1)[0]
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
        """The formats an upload may declare and their tags."""
        return trace_formats(self.build_type)

    def upload(
        self, body: bytes, input_file_format: str | None, input_file_tags: list[str] | None
    ) -> dict:
        """Keep an uploaded trace once the simulator reads it; its record. With
        no format, the format and tags are the ones its header fits
        (:func:`detect_format`)."""
        if input_file_format is None:
            if input_file_tags is not None:
                raise BadWorkload(
                    "tags need a format: give format with them, or neither and the header "
                    "picks both"
                )
            header = body.decode("utf-8", "replace").splitlines()[:1]
            columns = next(csv.reader(header), [])
            input_file_format, input_file_tags = detect_format(
                [c.strip() for c in columns], self.trace_formats()
            )
        self.prune()
        workload_id = uuid.uuid4().hex
        directory = self.uploads_dir / workload_id
        directory.mkdir()
        try:
            trace = directory / _TRACE
            trace.write_bytes(body)
            source = TraceSource(trace, input_file_format, tuple(input_file_tags or ()))
            requests = plan(read_block(source), self.build_type)
            check_count("the upload", len(requests))
            created = time.time()
            record = {
                "workload_id": workload_id,
                "input_file_format": input_file_format,
                "input_file_tags": list(input_file_tags or ()),
                **trace_facts(requests),
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
                "summary": "A CSV you send to POST /workloads (body: the file). Its format "
                "and tags are the ones its header fits, or the query's `format` and `tags` "
                "(comma-separated); the answer's workload_id is the simulation's "
                "workload.upload.",
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
