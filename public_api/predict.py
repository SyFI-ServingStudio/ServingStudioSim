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

The answer is per case, per section: the total, the cost tree with each node's
time as the Analyzer reads it (``analyze gen-iter-breakdown``), and each slot's
backend and coverage by the slot index of :meth:`DeploymentIndex.tree`. Beside
the cases, ``kernel_time_share`` is the Analyzer's ``kernel-time-share`` over
all of them: every slot name's time, its layers summed, critical path only,
largest first, and their sums by kernel kind (``kinds``). A run that is not
analyzed runs only these two
(``--analyzer-essential-only``).

:func:`missing_specs` asks the same launcher entry, as a dry run, which profile.db
rows the member's kernels lack; a member that lacks any is not predictable.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import pyarrow.parquet as pq

from launcher import timing_predict as launcher
from launcher.exec import ERROR_JSON, binary_error
from public_api.deployments import Member

MAX_CASES = 64


class BadCases(ValueError):
    """Cases the simulator rejects; the message is its own."""


class NotPredictable(RuntimeError):
    """The member or a case needs profile.db rows that are not measured."""


def _cause(error: str | None, log: str, *directories: Path) -> str:
    """The binary's error (``--error-json``), or the end of its ``log`` when it
    wrote none (a panic), with the paths under ``directories`` (the run's own)
    cut out."""
    cause = error or log.strip()[-2000:]
    for directory in directories:
        cause = cause.replace(f"{directory}/", "")
    return cause


def missing_by_role(report: dict) -> dict[str, int]:
    """A dry run's ``--report-json`` document as the rows lacking, ``{kernel
    role: count}``; a role that lacks none is left out."""
    missing: dict[str, int] = {}
    for kernel in report["kernels"]:
        if kernel["missing"]:
            missing[kernel["name"]] = missing.get(kernel["name"], 0) + kernel["missing"]
    return missing


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


def missing_specs(member: Member, build_type: str = "release") -> dict[str, int]:
    """The profile.db rows ``member``'s kernels need and profile.db lacks, as
    ``{role: count}`` by the kernel's dotted role; empty when every one is
    measured."""
    if member.error or member.predict is None:
        raise NotPredictable(f"{member.preset} {member.params} does not build: {member.error}")
    with tempfile.TemporaryDirectory(prefix="public-dry-run-") as directory:
        scratch = Path(directory)
        config = _write_config(member, [_probe_case(member.predict)], scratch, scratch / "out")
        output, report, error = launcher.dry_run_report(config, build_type)
    if report is None:
        raise RuntimeError(f"{member.preset} {member.params}: {_cause(error, output, scratch)}")
    return missing_by_role(report)


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
            run = launcher.run_one(
                config, build_type, "full" if analyze else "essential", render=False
            )
            if not asyncio.run(run):
                log = log_dir / "stdout.log"
                text = log.read_text(errors="replace") if log.exists() else ""
                error = binary_error(log_dir / "raw" / ERROR_JSON)
                cause = _cause(error, text, scratch, log_dir)
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
    table = pq.read_table(
        raw / "cost_log" / "worker_predict_0.parquet",
        columns=[
            "iter_id",
            "section",
            "layer",
            "total_time_ms",
            "energy_j",
            "slot_backend",
            "slot_coverage",
        ],
    )
    rows = table.to_pylist()
    # The simulator names the coverage bits and the no-backend value in the columns.
    flag_names = table.schema.field("slot_coverage").metadata[b"flags"].decode().split(",")
    no_backend = int(table.schema.field("slot_backend").metadata[b"none"])
    sections = {s["section"]: s for s in manifest["sections"]}
    # The Analyzer's tree per row (`analyze gen-iter-breakdown`): each node's
    # time for one call, scaled, and its share of the row's time.
    breakdown = {
        (tree["iter_id"], tree["section"], tree["layer"]): tree["nodes"]
        for tree in json.loads((log_dir / "payloads" / "iter_breakdown.json").read_text())[
            "iterations"
        ]
    }
    trees = {s["section"]: s for s in member.sections}
    out: list[dict] = [{"sections": []} for _ in cases]
    for row in rows:
        section = sections[row["section"]]
        tree = trees[row["section"]]
        # The run built the model the index built; a different tree means the
        # binary changed under the service.
        if [s["name"] for s in section["slots"]] != [s["name"] for s in tree["slots"]]:
            raise RuntimeError(f"{member.preset}: the predicted tree differs from the index's")
        flags: dict[str, list[int]] = {}
        for index, bits in enumerate(row["slot_coverage"]):
            for bit, name in enumerate(flag_names):
                if bits >> bit & 1:
                    flags.setdefault(name, []).append(index)
        out[row["iter_id"]]["sections"].append(
            {
                "section": row["section"],
                "layer": row["layer"],
                "total_ms": row["total_time_ms"],
                "energy_j": row["energy_j"],
                "nodes": breakdown[row["iter_id"], row["section"], row["layer"]],
                # null: the leaf did not run (`LeafMetrics::NO_BACKEND`), as an
                # inter-node transfer on one node.
                "slot_backend": [
                    None if b == no_backend else slot["backends"][b]
                    for slot, b in zip(tree["slots"], row["slot_backend"], strict=True)
                ],
                "coverage": flags,
            }
        )
    # The Analyzer's kernel ranking over every case (`kernel-time-share`).
    share = json.loads((log_dir / "payloads" / "kernel_time_share_composition.json").read_text())
    return {"cases": out, "kernel_time_share": share["overall"]}
