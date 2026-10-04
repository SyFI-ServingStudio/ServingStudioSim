"""``POST /predict``: one deployment's time for a reader's batches.

Each request is one ``launcher timing-predict`` run (``run_one``) of the
member's block, as the index built it (captures resolved, every param filled),
with the reader's cases. By default it is not analyzed and its directory is
removed once read. With ``analyze`` it is the launcher's standard analyzed
prediction (no plots) in its own directory under the service's runs directory,
which the Analyzer serves by ``prediction_id`` for Read more, until the service
removes it (:func:`prune`). The service sets ``SERVINGSTUDIO_NO_GPU``, so a row a
case needs and profile.db lacks fails the request with the step that would have
needed a GPU, instead of profiling it. Nothing writes profile.db.

The answer is in terms of the member's tree (:meth:`DeploymentIndex.tree`): per
case, per section, the total, each node's time by node id and each slot's time,
backend and coverage by slot index.

:func:`missing_specs` asks the same launcher entry, as a dry run, which profile.db
rows the member's kernels lack; a member that lacks any is not predictable.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import pyarrow.parquet as pq

from launcher import timing_predict as launcher
from public_api.deployments import Member, coverage_flags

MAX_CASES = 64
# `LeafMetrics::NO_BACKEND` (simulator `timing/cache/interp.rs`).
NO_BACKEND = 255


class BadCases(ValueError):
    """Cases the simulator rejects; the message is its own."""


class NotPredictable(RuntimeError):
    """The member or a case needs profile.db rows that are not measured."""


def _cause(stderr: str, *directories: Path) -> str:
    """anyhow's error chain (``Error: a`` / ``Caused by:`` / ``0: b``) as one
    line, with the paths under ``directories`` (the run's own) cut out."""
    lines = []
    for line in stderr.splitlines():
        text = line.strip()
        if text.startswith("Error: "):
            lines = [text.removeprefix("Error: ")]
        elif lines and text and text != "Caused by:":
            lines.append(text.split(": ", 1)[1] if text[:1].isdigit() else text)
    cause = ": ".join(lines) or stderr.strip()[-2000:]
    for directory in directories:
        cause = cause.replace(f"{directory}/", "")
    return cause


def _node_times(nodes: list[dict], slot_ms: list[float]) -> list[float]:
    """Each node's time, aggregated as ``CostTree::aggregate`` does."""
    out: list[float | None] = [None] * len(nodes)

    def time(index: int) -> float:
        if out[index] is None:
            node = nodes[index]
            kind = next(iter(node))
            if kind == "Leaf":
                value = slot_ms[node["Leaf"]]
            else:
                body = node[kind]
                children = [
                    time(c) for c in range(body["children"]["start"], body["children"]["end"])
                ]
                if kind == "Sum":
                    value = sum(children)
                elif kind == "Max":
                    value = max(children, default=0.0) / body["overlap"]
                else:
                    value = body["n"] * sum(children)
            out[index] = value
        return out[index]

    for index in range(len(nodes)):
        time(index)
    return out


def _write_config(member: Member, cases: list, directory: Path, log_dir: Path) -> Path:
    """A ``timing-predict`` config for ``member`` and ``cases`` in ``directory``."""
    (directory / "cases.json").write_text(json.dumps(cases))
    config = directory / "timing_predict.json"
    config.write_text(
        json.dumps(
            {
                "arch": {member.predict["selector"]: member.block["arch"]},
                "gpu": member.gpu,
                "log_dir": str(log_dir),
                "cases_file": str(directory / "cases.json"),
            }
        )
    )
    return config


def _probe_case(shape: dict) -> dict:
    """The smallest valid case of a member's prediction shape: one decode
    request per group (one token per group for a layer-wise FFN)."""
    groups = shape["groups"]
    if shape["selector"] == "ffn":
        return {"tokens_per_group": [1] * groups}
    if shape["selector"] == "speculative_iter":
        width = shape["query_width"]
        return {"groups": [{"decode_requests": [[width, width]]}] * groups}
    return {"groups": [{"decode_kv_lens": [1]}] * groups}


# One kernel's line of the dry-run report (`timing_predict::print_dry_run`).
_DRY_RUN_LINE = re.compile(r"^\s+(\S+)\s+\((\w+)\s*\)\s+(\d+) / (\d+)\s+missing$")


def missing_specs(member: Member, build_type: str = "release") -> dict[str, int]:
    """The profile.db rows ``member``'s kernels need and profile.db lacks, as
    ``{role: count}`` by the kernel's dotted role; empty when every one is
    measured."""
    if member.error or member.predict is None:
        raise NotPredictable(f"{member.preset} {member.params} does not build: {member.error}")
    with tempfile.TemporaryDirectory(prefix="public-dry-run-") as directory:
        scratch = Path(directory)
        config = _write_config(member, [_probe_case(member.predict)], scratch, scratch / "out")
        succeeded, stdout = launcher.dry_run_report(config, build_type)
    if not succeeded:
        raise RuntimeError(f"{member.preset} {member.params}: {_cause(stdout, scratch)}")
    missing: dict[str, int] = {}
    for line in stdout.splitlines():
        match = _DRY_RUN_LINE.match(line)
        if match and int(match[3]):
            missing[match[1]] = missing.get(match[1], 0) + int(match[3])
    if not stdout.rstrip().splitlines()[-1].startswith("total: "):
        raise RuntimeError(f"{member.preset}: unexpected dry-run report:\n{stdout[-2000:]}")
    return missing


def prune(runs_dir: Path, keep_seconds: float) -> None:
    """Remove the predictions older than ``keep_seconds``."""
    cutoff = time.time() - keep_seconds
    for run in runs_dir.iterdir() if runs_dir.is_dir() else []:
        if run.is_dir() and run.stat().st_mtime < cutoff:
            shutil.rmtree(run, ignore_errors=True)


def predict(
    runs_dir: Path,
    member: Member,
    cases: list,
    build_type: str = "release",
    *,
    analyze: bool = False,
) -> dict:
    """Cost ``cases`` on ``member``: per case, its sections' times. With
    ``analyze``, the prediction is also analyzed (without plots) and kept in
    ``runs_dir`` for the Analyzer, and its ``prediction_id`` returned; without,
    it is removed once read."""
    if member.error or member.predict is None:
        raise NotPredictable(f"{member.preset} {member.params} does not build: {member.error}")
    if member.missing:
        raise NotPredictable(
            f"{member.preset} {member.params} lacks profile.db rows: "
            + ", ".join(f"{kind} {count}" for kind, count in sorted(member.missing.items()))
        )
    if not isinstance(cases, list) or not cases:
        raise BadCases("cases must be a non-empty list")
    if len(cases) > MAX_CASES:
        raise BadCases(f"at most {MAX_CASES} cases per request")
    with tempfile.TemporaryDirectory(prefix="public-predict-") as directory:
        scratch = Path(directory)
        log_dir = runs_dir / uuid.uuid4().hex if analyze else scratch / "out"
        config = _write_config(member, cases, scratch, log_dir)
        try:
            run = launcher.run_one(config, build_type, analyze=analyze, render=False)
            if not asyncio.run(run):
                log = log_dir / "stdout.log"
                text = log.read_text(errors="replace") if log.exists() else ""
                cause = _cause(text, scratch, log_dir)
                if "needs a GPU" in cause:
                    raise NotPredictable(cause)
                raise BadCases(cause)
            prediction = _read(member, cases, log_dir)
        except BaseException:
            shutil.rmtree(log_dir, ignore_errors=True)
            raise
    if analyze:
        meta = json.loads((log_dir / "prediction.meta.json").read_text())
        prediction["prediction_id"] = meta["prediction_id"]
    return prediction


def _read(member: Member, cases: list, log_dir: Path) -> dict:
    """A finished prediction's times, per case and section, on ``member``'s tree."""
    raw = log_dir / "raw"
    manifest = json.loads((raw / "cost_manifest" / "worker_predict_0.json").read_text())
    rows = pq.read_table(
        raw / "cost_log" / "worker_predict_0.parquet",
        columns=[
            "iter_id",
            "section",
            "layer",
            "total_time_ms",
            "energy_j",
            "slot_time_ms",
            "slot_backend",
            "slot_coverage",
        ],
    ).to_pylist()
    sections = {s["section"]: s for s in manifest["sections"]}
    trees = {s["section"]: s for s in member.sections}
    out: list[dict] = [{"sections": []} for _ in cases]
    for row in rows:
        section = sections[row["section"]]
        tree = trees[row["section"]]
        # The run built the model the index built; a different tree means the
        # binary changed under the service.
        if [s["name"] for s in section["slots"]] != [s["name"] for s in tree["slots"]]:
            raise RuntimeError(f"{member.preset}: the predicted tree differs from the index's")
        slot_ms = row["slot_time_ms"]
        flags: dict[str, list[int]] = {}
        for index, bits in enumerate(row["slot_coverage"]):
            for name in coverage_flags(bits):
                flags.setdefault(name, []).append(index)
        out[row["iter_id"]]["sections"].append(
            {
                "section": row["section"],
                "layer": row["layer"],
                "total_ms": row["total_time_ms"],
                "energy_j": row["energy_j"],
                "node_ms": _node_times(section["nodes"], slot_ms),
                "slot_ms": slot_ms,
                # null: the leaf did not run (`LeafMetrics::NO_BACKEND`), as an
                # inter-node transfer on one node.
                "slot_backend": [
                    None if b == NO_BACKEND else slot["backends"][b]
                    for slot, b in zip(tree["slots"], row["slot_backend"], strict=True)
                ],
                "coverage": flags,
            }
        )
    return {"cases": out}
