"""Read a case's `from_capture` inputs out of its own workload pass.

Two calibrated case inputs are measurements the workload pass already makes:

- `speculative_acceptance`, the per-position conditional acceptance chain, from
  `spec_decode_metrics.json`: position 0 is `accepted_tokens_per_position["0"] /
  num_drafts`, position k is `accepted[k] / accepted[k-1]`.
- `speculative_acceptance_by_output_position`, the per-verify-round
  distribution of accepted drafts per bucket of output position, from the
  workload pass's `metrics_jsonl` (`buckets_from_metrics`, the estimator the
  Qwen3.8-Flash-Next and GLM-5.3 DFlash2 packs were built with). Render folds
  the buckets into each request's chain (`render.acceptance_chain`).
- `attn_gpu_memory_gb`, the per-rank KV pool, from the vLLM server log: the
  smallest rank's `Extended KV cache to N blocks (X GiB)` (the vLLM fork's
  extensible cache) or, without that line, `Available KV cache memory: X GiB`;
  `X * 2**30 / 1e9` GB. This is the conversion the hand-written packs used
  (`(x 1.073741824)` in their `derived_from`). Block counts are not converted:
  bytes per block depend on the model, and the log does not state them.

A case that says `{status: from_capture}` gets these at render or run time
instead of a hand copy. The resolved values, with the files they came from,
are written to `<case_dir>/calibration.json`, which `extract` carries into the
compare report and `compare --record` into the golden's provenance.

Rounding follows the scripts these replace: the chain to 6 decimals
(`collect_measured.py` of the DSv4.1 DSpark pack), the KV pool to 6 decimals of
a GB (that pack's `make_pack.py`).

SGLang is not covered. Its log states the pool as `max_total_num_tokens`, and
the existing packs turned that into GB with a per-token byte size that the log
does not carry, so an SGLang case must declare a measured value.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .pack import (
    ACCEPTANCE_RULE,
    BY_POSITION_RULE,
    KV_POOL_RULE,
    Calibrated,
    Case,
    Pack,
    PackError,
)

CALIBRATION_NAME = "calibration.json"
CALIBRATION_SCHEMA_VERSION = 1
GIB = 1 << 30

_EXTENDED = re.compile(r"Extended KV cache to (\d+) blocks \(([\d.]+) GiB\)")
_AVAILABLE = re.compile(r"Available KV cache memory: ([\d.]+) GiB")


class CalibrationError(PackError):
    """A `from_capture` field whose source is missing or unreadable."""


# ── the two extractors ───────────────────────────────────────────────────────


def acceptance_chain(metrics: dict[str, Any]) -> list[float]:
    """Conditional per-position acceptance from vLLM's spec-decode counters."""
    try:
        drafts = metrics["num_drafts"]
        accepted = metrics["accepted_tokens_per_position"]
    except (KeyError, TypeError):
        raise CalibrationError(
            "spec_decode_metrics.json needs num_drafts and accepted_tokens_per_position"
        ) from None
    counts = [accepted.get(str(position)) for position in range(len(accepted))]
    if not drafts or not counts or any(not isinstance(item, int) for item in counts):
        raise CalibrationError(
            f"spec_decode_metrics.json has no usable counters (num_drafts={drafts!r}, "
            f"accepted_tokens_per_position={accepted!r})"
        )
    chain = [counts[0] / drafts]
    for previous, current in zip(counts, counts[1:]):
        chain.append(current / previous if previous else 0.0)
    return [round(value, 6) for value in chain]


#: Lower bounds of the `output_tokens_before` buckets, and the fewest verify
#: rounds a bucket may hold before it merges into the previous one.
EDGES = (1, 32, 128, 512, 1024, 2048, 4096, 8192)
MIN_ROUNDS = 50


def buckets_from_metrics(path: Path, K: int) -> list[tuple[int, int | None, list[float]]]:
    """Accepted-draft distribution by output position, from one metrics JSONL.

    Every `decode_request_progress` row with both `output_tokens_before` and
    `accepted_draft_tokens` is one verify round. Rounds are binned by the
    `EDGES` bucket of `output_tokens_before` (position 0 joins the first), and
    accepted counts above `K` count as `K`. A bucket with fewer than
    `MIN_ROUNDS` rounds merges into the previous one; an undersized first
    bucket merges into the second. Returns `(from, to, p)` with `p` the
    distribution of 0..K accepted drafts; the last `to` is None.
    """
    counts = [Counter() for _ in EDGES]
    for line in open(path):
        for p in json.loads(line)["decode_request_progress"]:
            pos, a = p.get("output_tokens_before"), p.get("accepted_draft_tokens")
            if pos is None or a is None:
                continue
            b = max(i for i, lo in enumerate(EDGES) if pos >= lo) if pos >= 1 else 0
            counts[b][min(a, K)] += 1
    # Merge sparse buckets into the previous one; drop empty tails.
    merged: list[list] = []
    for i, c in enumerate(counts):
        n = sum(c.values())
        if n == 0:
            continue
        if merged and n < MIN_ROUNDS:
            merged[-1][1] += c
            continue
        merged.append([EDGES[i], Counter(c)])
    if len(merged) > 1 and sum(merged[0][1].values()) < MIN_ROUNDS:
        merged[1][1] += merged[0][1]
        merged[1][0] = merged[0][0]
        merged.pop(0)
    out = []
    for j, (lo, c) in enumerate(merged):
        hi = merged[j + 1][0] if j + 1 < len(merged) else None
        n = sum(c.values())
        out.append((lo, hi, [c[a] / n for a in range(K + 1)]))
    return out


def tokens_per_round(p: list[float]) -> float:
    """E = sum_a (a + 1) p(a): tokens one verify round emits under `p`."""
    return sum((accepted + 1) * share for accepted, share in enumerate(p))


@dataclass(frozen=True)
class KvPool:
    gb: float
    gib: float
    line: str
    blocks: int | None


def vllm_kv_pool(log_text: str) -> KvPool:
    """The smallest rank's KV pool in a vLLM server log."""
    extended = [
        (float(match.group(2)), int(match.group(1)), match.group(0))
        for match in _EXTENDED.finditer(log_text)
    ]
    if extended:
        gib, blocks, line = min(extended)
        return KvPool(gb=round(gib * GIB / 1e9, 6), gib=gib, line=line, blocks=blocks)
    available = [(float(match.group(1)), match.group(0)) for match in _AVAILABLE.finditer(log_text)]
    if available:
        gib, line = min(available)
        return KvPool(gb=round(gib * GIB / 1e9, 6), gib=gib, line=line, blocks=None)
    raise CalibrationError(
        "the server log has neither `Extended KV cache to N blocks (X GiB)` nor "
        "`Available KV cache memory: X GiB`"
    )


# ── one case ─────────────────────────────────────────────────────────────────


def workload_pass_name(pack: Pack, case: Case) -> str:
    variant = pack.variant_of(case)
    for item in variant.profile_passes:
        if item.kind == "workload_metrics":
            return item.name
    raise CalibrationError(
        f"{case.slug}: from_capture reads the workload_metrics pass, and variant "
        f"{variant.name} declares none"
    )


def _artifact(directory: Path, result: dict[str, Any], key: str) -> Path:
    """A file `profile_result.json` recorded, found inside this pass directory.

    The recorded path is absolute from wherever the profile ran; its position
    under the recorded `log_dir` is what survives a moved campaign.
    """
    recorded = result.get(key)
    if not isinstance(recorded, str) or not recorded:
        raise CalibrationError(f"{directory.name}/profile_result.json records no {key}")
    path = Path(recorded)
    try:
        relative = path.relative_to(str(result.get("log_dir", "")))
    except ValueError:
        relative = Path(path.name)
    candidate = directory / relative
    if not candidate.is_file():
        raise CalibrationError(f"{key} {relative} is not in {directory}")
    return candidate


def resolve(pack: Pack, case: Case, case_dir: Path) -> dict[str, dict[str, Any]]:
    """Every `from_capture` field of `case`, read from its workload pass.

    Read-only. Returns `{field: {value, source, ...}}` with `source` relative
    to `case_dir`; raises `CalibrationError` naming what is missing.
    """
    fields = case.capture_fields
    if not fields:
        return {}
    name = workload_pass_name(pack, case)
    directory = Path(case_dir) / name
    result_path = directory / "profile_result.json"
    if not result_path.is_file():
        raise CalibrationError(
            f"{', '.join(fields)} {'is' if len(fields) == 1 else 'are'} from_capture and "
            f"{name} has not run ({name}/profile_result.json absent)"
        )
    try:
        result = json.loads(result_path.read_text())
    except ValueError as error:
        raise CalibrationError(f"{name}/profile_result.json is not JSON: {error}") from None
    engine = result.get("engine") or pack.variant_of(case).engine

    resolved: dict[str, dict[str, Any]] = {}
    if "speculative_acceptance" in fields:
        source = _artifact(directory, result, "spec_decode_metrics_json")
        metrics = json.loads(source.read_text())
        resolved["speculative_acceptance"] = {
            "value": acceptance_chain(metrics),
            "source": str(source.relative_to(case_dir)),
            "rule": ACCEPTANCE_RULE,
            "num_drafts": metrics["num_drafts"],
            "accepted_tokens_per_position": metrics["accepted_tokens_per_position"],
        }
    if "speculative_acceptance_by_output_position" in fields:
        depth = pack.variant_of(case).arch.get("draft_tokens")
        if not isinstance(depth, int) or depth < 1:
            raise CalibrationError(
                f"speculative_acceptance_by_output_position: from_capture needs the variant's "
                f"arch.draft_tokens (got {depth!r})"
            )
        source = _artifact(directory, result, "metrics_jsonl")
        buckets = buckets_from_metrics(source, depth)
        if not buckets:
            raise CalibrationError(
                f"{source.name} has no decode_request_progress rows with "
                "output_tokens_before and accepted_draft_tokens"
            )
        resolved["speculative_acceptance_by_output_position"] = {
            "value": [{"from": lo, "p": p} for lo, _hi, p in buckets],
            "source": str(source.relative_to(case_dir)),
            "rule": BY_POSITION_RULE,
            "tokens_per_round": {str(lo): round(tokens_per_round(p), 6) for lo, _hi, p in buckets},
        }
    if "attn_gpu_memory_gb" in fields:
        if engine != "vllm":
            raise CalibrationError(
                f"attn_gpu_memory_gb: from_capture has no {engine} extractor. SGLang logs "
                "`max_total_num_tokens=C`, and turning that into GB needs the per-token KV "
                "bytes, which the log does not state; declare a measured value instead"
            )
        source = _artifact(directory, result, "server_log")
        pool = vllm_kv_pool(source.read_text(errors="replace"))
        resolved["attn_gpu_memory_gb"] = {
            "value": pool.gb,
            "source": str(source.relative_to(case_dir)),
            "rule": KV_POOL_RULE,
            "log_line": pool.line,
            "gib": pool.gib,
            "blocks": pool.blocks,
        }
    return resolved


def requirements(pack: Pack, case: Case, case_dir: Path) -> list[str]:
    """Why this case's from_capture fields cannot be read yet; empty when they can."""
    if not case.capture_fields:
        return []
    try:
        resolve(pack, case, case_dir)
    except (CalibrationError, OSError, ValueError, KeyError) as error:
        return [str(error)]
    return []


def resolved_case(case: Case, resolved: dict[str, dict[str, Any]], case_dir: Path) -> Case:
    """`case` with each from_capture field replaced by its measured value.

    A resolved field is `measured` from here on: everything downstream treats
    it exactly like a hand-entered measured value, with the source as evidence.
    """
    changes = {}
    for name, body in resolved.items():
        changes[name] = Calibrated(
            value=body["value"],
            status="measured",
            derived_from=body["rule"],
            evidence=str(Path(case_dir) / body["source"]),
        )
    missing = set(case.capture_fields) - set(changes)
    if missing:
        raise CalibrationError(f"{case.slug}: {sorted(missing)} not resolved")
    return dataclasses.replace(case, **changes)


def write(pack: Pack, case: Case, case_dir: Path, resolved: dict[str, dict[str, Any]]) -> Path:
    path = Path(case_dir) / CALIBRATION_NAME
    document = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "pack": pack.name,
        "case": case.slug,
        "workload_pass": workload_pass_name(pack, case),
        "fields": resolved,
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    return path


def load(case_dir: Path) -> dict[str, Any] | None:
    """A case directory's calibration.json, or None when it has none."""
    path = Path(case_dir) / CALIBRATION_NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text())
