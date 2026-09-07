"""Turn a server log into the alignment records the analyzer consumes.

Three extractors, one per record family: per-iteration batch shapes, per-request
latency splits, and MoE expert popularity. All three read plain log lines and
write JSONL, and none of them know which engine produced the log -- what varies
between engines is described by the `EngineRecords` passed in, not by a second
copy of the parsing.

The records are versioned contracts, so every extractor validates the fields it
is about to hand downstream and raises rather than emitting a row the analyzer
would have to guess about. A capture is expensive; a malformed record found here
is far cheaper than one found in a report.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path, PurePosixPath

from .engine_records import (
    API_DISPATCH_DURATION_FIELDS,
    VLLM_RECORDS,
    EngineRecords,
)

# Structured engine-owned records. Do not parse the human-readable iteration
# line an engine prints for its own UI: its wording is that engine's business,
# while this JSON is the versioned analyzer contract.
_ALIGNMENT_WORKER_RE = re.compile(r"VibeSimAlignmentWorker\s+(\{.*\})\s*$")
_ALIGNMENT_ITERATION_RE = re.compile(r"VibeSimAlignmentIteration\s+(\{.*\})\s*$")
_ALIGNMENT_REQUEST_TIMING_RE = re.compile(r"VibeSimAlignmentRequestTiming\s+(\{.*\})\s*$")
_ALIGNMENT_API_REQUEST_TIMING_RE = re.compile(r"VibeSimAlignmentApiRequestTiming\s+(\{.*\})\s*$")
_ALIGNMENT_EXPERT_LOAD_RE = re.compile(r"VibeSimAlignmentExpertLoad\s+(\{.*\})\s*$")
_HF_HUB_MODEL_DIR_RE = re.compile(r"^models--(?P<organization>.+?)--(?P<name>.+)$")
_VLLM_ENGINE_CORE_PREFIX_RE = re.compile(r"\(EngineCore(?:_DP(\d+))?\s+pid=(\d+)\)")
_VLLM_ENGINE_CORE_BODY_RE = re.compile(
    r"(?:INFO|WARNING|ERROR)\s+\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s+\[core\.py:\d+\]"
)
_VLLM_HUMAN_ITERATION_RE = re.compile(
    r"\[core\.py:434\]\s+Iteration\((\d+)\):.*"
    r"iteration elapsed time:\s+([0-9]+(?:\.[0-9]+)?)\s+ms\s*$"
)


def _vllm_engine_core_prefixes(line: str) -> list[int]:
    """Read all rank prefixes from one multiprocessing-coalesced log line."""
    ranks: list[int] = []
    cursor = 0
    while matched := _VLLM_ENGINE_CORE_PREFIX_RE.match(line, cursor):
        if matched.group(1) is not None:
            ranks.append(int(matched.group(1)))
        cursor = matched.end()
        while cursor < len(line) and line[cursor].isspace():
            cursor += 1
    return ranks


def _vllm_framed_rank_candidates(
    lines: list[str],
) -> tuple[dict[int, frozenset[int]], dict[int, int]]:
    """Pair coalesced rank prefixes with their consecutive EngineCore bodies."""
    candidates_by_line: dict[int, frozenset[int]] = {}
    framed_rank_by_line: dict[int, int] = {}
    pending_ranks: list[int] = []
    pending_body_lines: list[int] = []
    for line_index, line in enumerate(lines):
        ranks = _vllm_engine_core_prefixes(line)
        if _VLLM_ENGINE_CORE_BODY_RE.search(line) is None:
            pending_ranks.clear()
            pending_body_lines.clear()
            continue
        pending_ranks.extend(ranks)
        if ranks or pending_ranks:
            pending_body_lines.append(line_index)
        if len(pending_body_lines) > len(pending_ranks):
            raise ValueError("EngineCore log framing has more bodies than rank prefixes")
        if pending_ranks and len(pending_body_lines) == len(pending_ranks):
            candidates = frozenset(pending_ranks)
            for body_line, rank in zip(pending_body_lines, pending_ranks, strict=True):
                candidates_by_line[body_line] = candidates
                framed_rank_by_line[body_line] = rank
            pending_ranks.clear()
            pending_body_lines.clear()
    if pending_ranks or pending_body_lines:
        raise ValueError("EngineCore log framing ends with incomplete rank-prefix ownership")
    return candidates_by_line, framed_rank_by_line


def extract_metrics_jsonl(
    server_log: Path,
    out_jsonl: Path,
    *,
    dp_size: int = 1,
    records: EngineRecords = VLLM_RECORDS,
) -> int:
    """Extract canonical structured iteration records into a metrics JSONL.

    The fork emits one `VibeSimAlignmentIteration {json}` line per model step.
    Exact prefill/decode shapes are retained for the typed timing-predict input
    adapter; `nsys_parse` also uses `prefill_tokens` for stage tagging.

    Each row is stamped with the `dp_rank` of the EngineCore that emitted it.
    Under data parallelism every rank schedules an independent batch and numbers
    its own iterations, so `(dp_rank, iteration_index)` — not `iteration_index`
    alone — is the identity of one measured batch shape. A `dp_size > 1` capture
    therefore *requires* the rank tag and fails without it, while a single
    EngineCore needs no prefix and lands on rank 0.

    When an engine repeats scheduler records on every TP rank, its descriptor
    selects one owner inside each DP group before identities are validated.
    """
    required = {
        "schema_version",
        "input_adapter",
        "iteration_index",
        "prefill_tokens",
        "decode_requests",
        "decode_tokens_scheduled",
        "prefill_chunk_pairs",
        "decode_kv_lens",
    }
    lines = Path(server_log).read_text(errors="replace").splitlines()
    candidates_by_line: dict[int, frozenset[int]] = {}
    framed_rank_by_line: dict[int, int] = {}
    human_iterations: list[tuple[int, float, frozenset[int]]] = []
    if records is VLLM_RECORDS and dp_size > 1:
        candidates_by_line, framed_rank_by_line = _vllm_framed_rank_candidates(lines)
        for line_index, line in enumerate(lines):
            matched = _VLLM_HUMAN_ITERATION_RE.search(line)
            if matched is not None:
                human_iterations.append(
                    (
                        int(matched.group(1)),
                        float(matched.group(2)),
                        candidates_by_line.get(line_index, frozenset()),
                    )
                )

    used_human_iterations: set[int] = set()
    seen_rank_iterations: set[tuple[int, int]] = set()
    last_iteration_by_rank = [-1] * dp_size
    n = 0
    with Path(out_jsonl).open("w") as out:
        for line_index, line in enumerate(lines):
            m = _ALIGNMENT_ITERATION_RE.search(line)
            if not m:
                continue
            if not records.owns_scheduler_record(line):
                continue
            row = json.loads(m.group(1))
            missing = required - set(row)
            if missing:
                raise ValueError(f"alignment iteration record missing fields {sorted(missing)}")
            if records is VLLM_RECORDS and dp_size > 1:
                iteration_index = int(row["iteration_index"])
                observed_elapsed_ms = row.get("observed_elapsed_ms")
                matching_human = [
                    human_index
                    for human_index, (human_iteration, elapsed_ms, candidates) in enumerate(
                        human_iterations
                    )
                    if human_index not in used_human_iterations
                    and human_iteration == iteration_index
                    and observed_elapsed_ms is not None
                    and abs(elapsed_ms - observed_elapsed_ms) <= 0.005001
                    and candidates & candidates_by_line.get(line_index, frozenset())
                ]
                if len(matching_human) > 1:
                    raise ValueError(
                        "alignment iteration record matches multiple framed human records, "
                        f"so its DP rank is ambiguous: iteration={iteration_index} "
                        f"elapsed_ms={observed_elapsed_ms}"
                    )
                if matching_human:
                    human_index = matching_human[0]
                    used_human_iterations.add(human_index)
                    candidate_ranks = set(human_iterations[human_index][2])
                    candidate_ranks &= set(candidates_by_line.get(line_index, frozenset()))
                else:
                    candidate_ranks = set(candidates_by_line.get(line_index, frozenset()))
                    direct_ranks = _vllm_engine_core_prefixes(line)
                    if not candidate_ranks and len(direct_ranks) == 1:
                        candidate_ranks = {direct_ranks[0]}

                valid_ranks = {
                    rank
                    for rank in candidate_ranks
                    if 0 <= rank < dp_size
                    and (rank, iteration_index) not in seen_rank_iterations
                    and iteration_index > last_iteration_by_rank[rank]
                }
                preferred_rank = framed_rank_by_line.get(line_index)
                if preferred_rank in valid_ranks:
                    dp_rank = preferred_rank
                elif len(valid_ranks) == 1:
                    dp_rank = valid_ranks.pop()
                else:
                    raise ValueError(
                        "alignment iteration record carries no EngineCore_DP<k> log prefix, "
                        f"so its DP rank cannot be established in a dp_size={dp_size} "
                        f"capture: {line[:120]!r}"
                    )
                seen_rank_iterations.add((dp_rank, iteration_index))
                last_iteration_by_rank[dp_rank] = iteration_index
            else:
                dp_rank = records.rank_of(line, dp_size=dp_size)
                identity = (dp_rank, int(row["iteration_index"]))
                if identity in seen_rank_iterations:
                    raise ValueError(
                        "duplicate alignment iteration record for "
                        f"dp_rank={identity[0]} iteration={identity[1]}"
                    )
                seen_rank_iterations.add(identity)
            if row.get("dp_rank", dp_rank) != dp_rank:
                raise ValueError(
                    f"alignment iteration record claims dp_rank {row['dp_rank']} but was "
                    f"emitted by {records.rank_prefix_label} rank {dp_rank}"
                )
            row["dp_rank"] = dp_rank
            if row["schema_version"] not in {1, 2, 3, 4} or row["input_adapter"] != records.adapter:
                raise ValueError(
                    "unsupported alignment iteration record "
                    f"schema={row['schema_version']!r} adapter={row['input_adapter']!r}"
                )
            if row["schema_version"] >= 2:
                timing_fields = {
                    "observed_start_monotonic_ns",
                    "observed_end_monotonic_ns",
                    "observed_elapsed_ms",
                }
                missing_timing = timing_fields - set(row)
                if missing_timing:
                    raise ValueError(
                        "alignment iteration schema-v2 record missing fields "
                        f"{sorted(missing_timing)}"
                    )
                if row["observed_end_monotonic_ns"] < row["observed_start_monotonic_ns"]:
                    raise ValueError("alignment iteration observation ends before it starts")
            if row["schema_version"] >= 3:
                _validate_decode_queries(row)
            if row["schema_version"] >= 4:
                _validate_decode_request_progress(row, records)
            out.write(json.dumps(row) + "\n")
            n += 1
    return n


def _validate_decode_queries(row: dict) -> None:
    queries = row.get("decode_query_lens")
    if not isinstance(queries, list) or len(queries) != row["decode_requests"]:
        raise ValueError("decode_query_lens must contain one query length per decode request")
    if any(type(value) is not int or value <= 0 for value in queries):
        raise ValueError("decode_query_lens entries must be positive integers")
    if sum(queries) != row["decode_tokens_scheduled"]:
        raise ValueError("decode_query_lens disagrees with decode_tokens_scheduled")
    if len(row["decode_kv_lens"]) != len(queries):
        raise ValueError("decode_kv_lens disagrees with decode_query_lens")


def _validate_decode_request_progress(row: dict, records: EngineRecords) -> None:
    progress = row.get("decode_request_progress")
    if not isinstance(progress, list) or len(progress) != row["decode_requests"]:
        raise ValueError("decode_request_progress must contain one entry per decode request")
    counts = {
        "kv_len",
        "query_len",
        "output_tokens_before",
        "drafted_tokens",
        "accepted_draft_tokens",
        "emitted_tokens",
    }
    seen: set[str] = set()
    for index, request in enumerate(progress):
        if not isinstance(request, dict):
            raise ValueError("decode_request_progress entries must be objects")
        engine_id = request.get("engine_request_id")
        if not isinstance(engine_id, str) or not engine_id:
            raise ValueError("decode request engine_request_id must be non-empty")
        external_id = request.get("external_request_id")
        if external_id is None:
            # Early v4 producers exposed only the randomized internal id.
            external_id = engine_id
            if records.adapter == "vllm_text":
                external_id = re.sub(r"-[0-9a-f]{8}$", "", external_id)
        if not isinstance(external_id, str) or not external_id:
            raise ValueError("decode request external_request_id must be non-empty")
        if engine_id != external_id and re.fullmatch(re.escape(external_id) + r"-[0-9a-f]{8}", engine_id) is None:
            raise ValueError("decode request external_request_id does not match its engine id")
        request_id = records.unwrap_request_id(external_id)
        if request_id in seen:
            raise ValueError(f"duplicate canonical decode request id {request_id!r}")
        seen.add(request_id)
        finished = request.get("request_finished_before", False)
        if type(finished) is not bool:
            raise ValueError("decode request request_finished_before must be boolean")
        for field in counts:
            value = request.get(field)
            if field == "output_tokens_before" and value is None and finished:
                continue
            if type(value) is not int or value < 0:
                raise ValueError(f"decode request {field} must be a nonnegative integer")
        if request["kv_len"] != row["decode_kv_lens"][index]:
            raise ValueError("decode request kv_len disagrees with decode_kv_lens")
        if request["query_len"] != row["decode_query_lens"][index]:
            raise ValueError("decode request query_len disagrees with decode_query_lens")
        drafted = request["drafted_tokens"]
        accepted = request["accepted_draft_tokens"]
        emitted = request["emitted_tokens"]
        # A scheduler may suppress this request's sampled output (e.g. async
        # bookkeeping). Zero output is an observation, not a bonus token.
        if accepted > drafted or accepted != (max(emitted - 1, 0) if drafted else 0):
            raise ValueError("decode request accepted drafts disagree with sampled output")
        if not drafted and emitted > 1:
            raise ValueError("non-spec decode cannot emit multiple sampled tokens")
        request["request_id"] = request_id


def extract_request_timings_jsonl(
    server_log: Path,
    out_jsonl: Path,
    *,
    expected_request_ids: set[str] | None = None,
    records: EngineRecords = VLLM_RECORDS,
) -> int:
    """Extract one complete EngineCore timing record per request.

    Schema v1 contains TTFT only. Schema v2 adds first-token → last-token TPOT.
    When the server log also carries API/SSE timing records, this extractor
    joins them by request id and emits schema v3. Durations stay within their
    originating process clock; no cross-process absolute timestamps are mixed.
    """
    ttft_duration_fields = {
        "engine_core_ttft_ms",
        "engine_queue_wait_ms",
        "engine_first_schedule_to_first_token_ms",
    }
    tpot_fields = {
        "engine_core_decode_ms",
        "engine_core_tpot_ms",
        "num_output_tokens",
    }
    api_duration_fields = {
        "api_frontend_prepare_ms",
        "api_first_output_wait_ms",
        "api_first_output_serialize_ms",
        "api_token_output_receive_span_ms",
        "api_token_sse_yield_span_ms",
        "api_terminal_tail_ms",
    }
    api_v2_duration_fields = {
        "api_stream_activation_ms",
        "api_add_request_ms",
        "api_collector_wait_ms",
        "api_collector_wakeup_ms",
        "api_generator_resume_ms",
    }
    api_v3_duration_fields = {
        "api_engine_output_wait_ms",
        "api_output_fanout_ms",
    }
    # Which of these an engine can actually report differs; see engine_records.
    del api_duration_fields, api_v2_duration_fields, api_v3_duration_fields
    required = {"schema_version", *ttft_duration_fields}
    log_lines = Path(server_log).read_text(errors="replace").splitlines()
    api_rows: dict[str, dict] = {}
    for line in log_lines:
        api_match = _ALIGNMENT_API_REQUEST_TIMING_RE.search(line)
        if not api_match:
            continue
        api_row = json.loads(api_match.group(1))
        api_schema_version = api_row.get("schema_version")
        if api_schema_version not in {1, 2, 3}:
            raise ValueError(
                f"unsupported alignment API request timing schema {api_schema_version!r}"
            )
        missing = records.api_required_fields(api_schema_version) - set(api_row)
        if missing:
            raise ValueError(
                f"alignment API request timing record missing fields {sorted(missing)}"
            )
        api_request_id = api_row["api_request_id"]
        if not isinstance(api_request_id, str) or not api_request_id:
            raise ValueError("alignment API request timing api_request_id must be non-empty")
        request_id = records.unwrap_request_id(api_request_id)
        if expected_request_ids is not None and request_id not in expected_request_ids:
            continue
        if request_id in api_rows:
            raise ValueError(f"duplicate alignment API request timing for {request_id!r}")
        # Validate whatever this engine declared it reports, and only that.
        present = records.api_required_fields(api_schema_version)
        for field in present:
            if not field.endswith("_ms"):
                continue
            value = api_row[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"alignment API request timing {field} must be nonnegative")
        for field in {"output_tokens", "token_events", "first_token_event_tokens"} & present:
            value = api_row[field]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"alignment API request timing {field} must be a positive integer")
        for field in {"token_events", "first_token_event_tokens"} & present:
            if api_row[field] > api_row["output_tokens"]:
                raise ValueError(f"alignment API {field} cannot exceed output_tokens")
        # The wait for the first output is the parent span; the phases inside it
        # are what catches a phase boundary stamped in the wrong place. How
        # strongly that can be checked depends on how completely the engine
        # splits the span: an engine that names every part must add back up
        # exactly, while one that names only the dispatch phases can only be
        # held to them fitting inside. Requiring the sum of everyone would mean
        # rejecting a truthful partial split, and dropping the check entirely
        # would stop catching the exact bug it exists for.
        collector_split = {
            "api_collector_wait_ms",
            "api_collector_wakeup_ms",
            "api_generator_resume_ms",
        }
        if API_DISPATCH_DURATION_FIELDS <= present:
            components = API_DISPATCH_DURATION_FIELDS | (collector_split & present)
            first_output_wait_components_ms = sum(api_row[field] for field in components)
            partitions_the_span = collector_split <= present
            if partitions_the_span:
                if (
                    abs(api_row["api_first_output_wait_ms"] - first_output_wait_components_ms)
                    > 1e-6
                ):
                    raise ValueError(
                        "alignment API first-output components do not sum to "
                        "api_first_output_wait_ms"
                    )
            elif first_output_wait_components_ms > api_row["api_first_output_wait_ms"] + 1e-6:
                raise ValueError(
                    "alignment API dispatch phases do not fit inside api_first_output_wait_ms"
                )
        if {"api_engine_output_wait_ms", "api_output_fanout_ms"} <= present:
            collector_wait_components_ms = (
                api_row["api_engine_output_wait_ms"] + api_row["api_output_fanout_ms"]
            )
            if abs(api_row["api_collector_wait_ms"] - collector_wait_components_ms) > 1e-6:
                raise ValueError(
                    "alignment API collector components do not sum to api_collector_wait_ms"
                )
        api_rows[request_id] = api_row

    api_request_ids = set(api_rows)
    request_ids: set[str] = set()
    n = 0
    with Path(out_jsonl).open("w") as out:
        for line in log_lines:
            match = _ALIGNMENT_REQUEST_TIMING_RE.search(line)
            if not match:
                continue
            if not records.owns_scheduler_record(line):
                continue
            row = json.loads(match.group(1))
            missing = required - set(row)
            if missing:
                raise ValueError(
                    f"alignment request timing record missing fields {sorted(missing)}"
                )
            schema_version = row["schema_version"]
            if schema_version not in {1, 2}:
                raise ValueError(f"unsupported alignment request timing schema {schema_version!r}")
            if schema_version == 2:
                missing = tpot_fields - set(row)
                if missing:
                    raise ValueError(
                        "alignment request timing schema v2 record missing fields "
                        f"{sorted(missing)}"
                    )
            # The vLLM OpenAI completions frontend wraps X-Request-Id before
            # enqueueing it as `cmpl-<source-id>-0`. req-frontend sends one prompt
            # per request, so index 0 is the only supported alignment shape.
            # `request_id` was the raw field name in the first instrumented run;
            # accept it so an in-flight capture remains extractable.
            engine_request_id = row.get("engine_request_id", row.get("request_id"))
            if not isinstance(engine_request_id, str) or not engine_request_id:
                raise ValueError("alignment request timing engine_request_id must be non-empty")
            request_id = records.unwrap_request_id(engine_request_id)
            # vLLM may issue frontend-owned prefix-cache probes before req-frontend
            # starts the replay. The replay's successful request ids are the
            # authoritative experiment population; unrelated timing records are
            # intentionally excluded instead of relying on a fixed probe count.
            if expected_request_ids is not None and request_id not in expected_request_ids:
                continue
            if request_id in request_ids:
                raise ValueError(f"duplicate alignment request timing for {request_id!r}")
            request_ids.add(request_id)
            row["engine_request_id"] = engine_request_id
            row["request_id"] = request_id
            for field in ttft_duration_fields:
                value = row[field]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise ValueError(f"alignment request timing {field} must be nonnegative")
            components_ms = (
                row["engine_queue_wait_ms"] + row["engine_first_schedule_to_first_token_ms"]
            )
            if abs(row["engine_core_ttft_ms"] - components_ms) > 1e-6:
                raise ValueError(
                    "alignment request timing components do not sum to engine_core_ttft_ms"
                )
            if schema_version == 2:
                num_output_tokens = row["num_output_tokens"]
                if (
                    isinstance(num_output_tokens, bool)
                    or not isinstance(num_output_tokens, int)
                    or num_output_tokens <= 0
                ):
                    raise ValueError(
                        "alignment request timing num_output_tokens must be a positive integer"
                    )
                decode_ms = row["engine_core_decode_ms"]
                if (
                    isinstance(decode_ms, bool)
                    or not isinstance(decode_ms, (int, float))
                    or not math.isfinite(decode_ms)
                    or decode_ms < 0
                ):
                    raise ValueError(
                        "alignment request timing engine_core_decode_ms must be nonnegative"
                    )
                tpot_ms = row["engine_core_tpot_ms"]
                if num_output_tokens == 1:
                    if tpot_ms is not None:
                        raise ValueError(
                            "alignment request timing engine_core_tpot_ms must be null "
                            "for a one-token request"
                        )
                else:
                    if (
                        isinstance(tpot_ms, bool)
                        or not isinstance(tpot_ms, (int, float))
                        or not math.isfinite(tpot_ms)
                        or tpot_ms < 0
                    ):
                        raise ValueError(
                            "alignment request timing engine_core_tpot_ms must be "
                            "nonnegative for a multi-token request"
                        )
                    expected_tpot_ms = decode_ms / (num_output_tokens - 1)
                    if abs(tpot_ms - expected_tpot_ms) > 1e-6:
                        raise ValueError(
                            "alignment request timing engine_core_tpot_ms does not equal "
                            "engine_core_decode_ms/(num_output_tokens-1)"
                        )
            api_row = api_rows.pop(request_id, None)
            if api_row is not None:
                if schema_version != 2:
                    raise ValueError(
                        "alignment API timing requires EngineCore request timing schema v2"
                    )
                if api_row["output_tokens"] != row["num_output_tokens"]:
                    raise ValueError(
                        "alignment API output_tokens does not match EngineCore num_output_tokens"
                    )
                # When the API record echoes the engine's own durations, they
                # must agree -- that is what proves the two halves describe the
                # same request. Not every engine can echo them: vLLM's API
                # process receives them with the output, while SGLang's frontend
                # never sees the scheduler's numbers. There the exact request id
                # is the join, so the echo is checked when offered and not
                # demanded when it cannot be.
                for field in ("engine_core_ttft_ms", "engine_core_decode_ms"):
                    if field in api_row and abs(api_row[field] - row[field]) > 1e-6:
                        raise ValueError(
                            f"alignment API {field} does not match EngineCore timing record"
                        )
                row["schema_version"] = 3
                row["engine_timing_schema_version"] = 2
                row["api_timing_schema_version"] = api_row["schema_version"]
                row["api_request_id"] = api_row["api_request_id"]
                # Carry across exactly the fields this engine declared, so an
                # engine that reports fewer phases produces a narrower row
                # rather than a row with holes.
                carried = records.api_required_fields(api_row["schema_version"])
                if "token_events" in carried:
                    row["api_token_events"] = api_row["token_events"]
                    row["api_first_token_event_tokens"] = api_row["first_token_event_tokens"]
                for field in sorted(carried):
                    if field.endswith("_ms"):
                        row[field] = api_row[field]
            out.write(json.dumps(row) + "\n")
            n += 1
    if api_rows:
        raise ValueError(
            "alignment API request timings have no matching EngineCore records: "
            f"{sorted(api_rows)[:8]!r}"
        )
    if api_request_ids and api_request_ids != request_ids:
        missing = sorted(request_ids - api_request_ids)
        raise ValueError(
            "alignment API request timing ids do not cover EngineCore timing ids: "
            f"missing={missing[:8]!r}"
        )
    if expected_request_ids is not None and request_ids != expected_request_ids:
        missing = sorted(expected_request_ids - request_ids)
        extra = sorted(request_ids - expected_request_ids)
        raise ValueError(
            "alignment request timing ids do not match successful replay ids: "
            f"missing={missing[:8]!r} extra={extra[:8]!r}"
        )
    return n


def normalize_model_id(model: str) -> str:
    """Turn an absolute served-model path into a portable checkpoint identity."""
    path = PurePosixPath(model)
    if not path.is_absolute():
        return model

    parts = [part for part in path.parts if part != "/"]
    for part in reversed(parts):
        matched = _HF_HUB_MODEL_DIR_RE.fullmatch(part)
        if matched is not None:
            return f"{matched['organization']}/{matched['name']}"

    if len(parts) >= 2 and parts[-2] == "snapshots":
        parts = parts[:-2]
    return parts[-1] if parts else model


def extract_expert_popularity(
    server_log: Path,
    out_jsonl: Path,
    out_json: Path,
    *,
    max_tokens_per_step: int,
    expert_parallel_size: int,
    reduction_group_size: int,
    experts_per_token: int | None = None,
    model_role: str | None = None,
    replay_start_monotonic_ns: int | None = None,
    replay_end_monotonic_ns: int | None = None,
    dp_size: int = 1,
    records: EngineRecords = VLLM_RECORDS,
) -> int:
    """Extract and aggregate rank-synchronized logical-expert token counts.

    The vLLM fork emits one record per model step only when EPLB balancedness
    logging is explicitly enabled. Counts are already reduced and mapped from
    physical replicas back to logical expert ids.

    ``reduction_group_size`` is the number of ranks represented by those
    already-summed counts. It is independent of ``expert_parallel_size`` and
    both values are required rather than inferred from the selected engine.
    """
    if (
        isinstance(expert_parallel_size, bool)
        or not isinstance(expert_parallel_size, int)
        or expert_parallel_size <= 0
    ):
        raise ValueError("expert_parallel_size must be a positive integer")
    if experts_per_token is not None and experts_per_token <= 0:
        raise ValueError("experts_per_token must be positive")
    if (
        isinstance(reduction_group_size, bool)
        or not isinstance(reduction_group_size, int)
        or reduction_group_size <= 0
    ):
        raise ValueError("reduction_group_size must be a positive integer")
    if max_tokens_per_step <= 0:
        raise ValueError("max_tokens_per_step must be positive")
    if model_role is not None and model_role not in {"target", "draft"}:
        raise ValueError("model_role must be target or draft")
    window = (replay_start_monotonic_ns, replay_end_monotonic_ns)
    if model_role is not None:
        if any(type(value) is not int or value <= 0 for value in window):
            raise ValueError("role-specific expert popularity requires an exact replay window")
        if replay_end_monotonic_ns < replay_start_monotonic_ns:
            raise ValueError("replay monotonic window ends before it starts")
    elif any(value is not None for value in window):
        raise ValueError("replay window requires an explicit model_role")

    raw_records: list[dict] = []
    accepted_records: list[dict] = []
    discarded_oversized_steps: list[int] = []
    expected_shape: tuple[int, int] | None = None
    expected_model: str | None = None
    expected_forwards: int | None = None
    discarded_outside_window = 0
    aggregate_counts: list[list[int]] | None = None
    seen_owner_steps: set[tuple[int, int]] = set()
    with (
        Path(server_log).open(errors="replace") as source,
        Path(out_jsonl).open("w") as output_file,
    ):
        for line in source:
            match = _ALIGNMENT_EXPERT_LOAD_RE.search(line)
            if match is None:
                continue
            if not records.owns_scheduler_record(line):
                continue
            record = json.loads(match.group(1))
            required = {
                "schema_version",
                "model",
                "eplb_step",
                "logical_expert_counts",
            }
            missing = required - set(record)
            if missing:
                raise ValueError(f"alignment expert-load record missing {sorted(missing)}")
            if record["schema_version"] not in {1, 2, 3}:
                raise ValueError(
                    f"unsupported alignment expert-load schema {record['schema_version']!r}"
                )
            if model_role is not None and record["schema_version"] < 3:
                raise ValueError("role-specific expert popularity requires schema-v3 raw records")
            if record["schema_version"] >= 3:
                role = record.get("model_role")
                if role not in {"target", "draft"}:
                    raise ValueError("expert-load model_role must be target or draft")
                if model_role is None and role == "draft":
                    raise ValueError(
                        "draft expert-load records require explicit model_role selection"
                    )
                if model_role is not None and role != model_role:
                    continue
                forwards = record.get("max_forwards_per_step")
                observed = record.get("observed_monotonic_ns")
                if type(forwards) is not int or forwards <= 0:
                    raise ValueError("max_forwards_per_step must be a positive integer")
                if type(observed) is not int or observed <= 0:
                    raise ValueError("observed_monotonic_ns must be a positive integer")
                if role == "target" and forwards != 1:
                    raise ValueError("target expert-load must have one forward per step")
                if expected_forwards is not None and expected_forwards != forwards:
                    raise ValueError("expert-load max_forwards_per_step changed")
                expected_forwards = forwards
            if record["schema_version"] >= 2:
                for field_name in ("expert_parallel_size", "experts_per_token"):
                    value = record.get(field_name)
                    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                        raise ValueError(
                            f"alignment expert-load {field_name} must be a positive integer"
                        )
                record_ep_size = record["expert_parallel_size"]
                record_top_k = record["experts_per_token"]
                if record_ep_size != expert_parallel_size:
                    raise ValueError(
                        "expert-load expert_parallel_size changed from "
                        f"{expert_parallel_size} to {record_ep_size}"
                    )
                if experts_per_token is None:
                    experts_per_token = record_top_k
                elif record_top_k != experts_per_token:
                    raise ValueError(
                        "expert-load experts_per_token changed from "
                        f"{experts_per_token} to {record_top_k}"
                    )
            model = record["model"]
            if not isinstance(model, str) or not model:
                raise ValueError("alignment expert-load model must be a non-empty string")
            if expected_model is None:
                expected_model = model
            elif model != expected_model:
                raise ValueError(f"expert-load model changed from {expected_model!r} to {model!r}")
            eplb_step = record["eplb_step"]
            if isinstance(eplb_step, bool) or not isinstance(eplb_step, int) or eplb_step < 0:
                raise ValueError("alignment expert-load eplb_step must be a nonnegative integer")
            if records.scheduler_record_rank_re is not None:
                dp_rank = records.rank_of(line, dp_size=dp_size)
                identity = (dp_rank, eplb_step)
                if identity in seen_owner_steps:
                    raise ValueError(
                        "duplicate alignment expert-load record for "
                        f"dp_rank={dp_rank} eplb_step={eplb_step}"
                    )
                seen_owner_steps.add(identity)
            counts = record["logical_expert_counts"]
            if (
                not isinstance(counts, list)
                or not counts
                or not all(
                    isinstance(layer_counts, list) and layer_counts for layer_counts in counts
                )
            ):
                raise ValueError("logical_expert_counts must be a non-empty 2D array")
            shape = (len(counts), len(counts[0]))
            if any(len(layer_counts) != shape[1] for layer_counts in counts):
                raise ValueError("logical_expert_counts must be rectangular")
            if expected_shape is None:
                expected_shape = shape
                aggregate_counts = [[0] * shape[1] for _ in range(shape[0])]
                if experts_per_token is None:
                    raise ValueError(
                        "schema-v1 expert-load records require an explicit "
                        "experts_per_token fallback"
                    )
                if shape[1] % expert_parallel_size != 0:
                    raise ValueError(
                        f"num_logical_experts {shape[1]} must be divisible by "
                        f"expert_parallel_size {expert_parallel_size}"
                    )
                if experts_per_token > shape[1]:
                    raise ValueError(
                        f"experts_per_token {experts_per_token} exceeds "
                        f"num_logical_experts {shape[1]}"
                    )
            elif shape != expected_shape:
                raise ValueError(f"expert-load shape changed from {expected_shape} to {shape}")
            layer_totals: list[int] = []
            for layer_counts in counts:
                layer_total = 0
                for count in layer_counts:
                    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                        raise ValueError("expert-load counts must be nonnegative integers")
                    layer_total += count
                layer_totals.append(layer_total)
            output_file.write(json.dumps(record, separators=(",", ":")) + "\n")
            raw_records.append(record)

            if model_role is not None and not (
                replay_start_monotonic_ns
                <= record["observed_monotonic_ns"]
                <= replay_end_monotonic_ns
            ):
                discarded_outside_window += 1
                continue
            assert experts_per_token is not None
            assignment_ceiling = (
                max_tokens_per_step
                * reduction_group_size
                * experts_per_token
                * (expected_forwards or 1)
            )
            if any(total > assignment_ceiling for total in layer_totals):
                discarded_oversized_steps.append(eplb_step)
                continue

            assert aggregate_counts is not None
            for layer_index, layer_counts in enumerate(counts):
                for expert_index, count in enumerate(layer_counts):
                    aggregate_counts[layer_index][expert_index] += count
            accepted_records.append(record)

    if not raw_records or expected_shape is None or aggregate_counts is None:
        raise ValueError("no VibeSimAlignmentExpertLoad records found in server log")
    if not accepted_records:
        raise ValueError("no expert-load records remain within the configured token ceiling")
    assert expected_model is not None and experts_per_token is not None

    def normalize(counts: list[int]) -> list[float]:
        total = sum(counts)
        return [count / total for count in counts] if total else [0.0] * len(counts)

    all_layer_counts = [
        sum(layer[expert] for layer in aggregate_counts) for expert in range(expected_shape[1])
    ]
    summary = {
        "schema_version": 4 if model_role is not None else 3,
        # The raw JSONL retains the engine's exact path. The portable summary
        # records checkpoint identity rather than one capture host's location.
        "model": normalize_model_id(expected_model),
        "num_moe_layers": expected_shape[0],
        "num_logical_experts": expected_shape[1],
        "expert_parallel_size": expert_parallel_size,
        "experts_per_rank": expected_shape[1] // expert_parallel_size,
        "experts_per_token": experts_per_token,
        "count_semantics": "logical_routed_token_assignments",
        "aggregation": {
            "scope": "captured_eplb_steps_within_token_ceiling",
            "observed_eplb_step_min": min(record["eplb_step"] for record in accepted_records),
            "observed_eplb_step_max": max(record["eplb_step"] for record in accepted_records),
            "record_count": len(accepted_records),
            "raw_record_count": len(raw_records),
            "discarded_oversized_record_count": len(discarded_oversized_steps),
            "discarded_oversized_eplb_steps": discarded_oversized_steps,
            "max_tokens_per_step": max_tokens_per_step,
        },
        # The current simulator projects logical expert ids onto contiguous EP
        # rank shards before removing rank/expert identity. Name that modeling
        # assumption explicitly; this is not claimed to be an observed EPLB
        # physical placement map.
        "expert_partitioning": {
            "kind": "contiguous_logical_expert_ids",
            "layout": "rank_major",
        },
        "counts_by_layer": aggregate_counts,
        "probabilities_by_layer": [normalize(layer) for layer in aggregate_counts],
        "counts_all_layers": all_layer_counts,
        "probabilities_all_layers": normalize(all_layer_counts),
    }
    if model_role is not None:
        summary["model_role"] = model_role
        summary["aggregation"].update(
            scope="replay_window_within_role_specific_token_ceiling",
            replay_start_monotonic_ns=replay_start_monotonic_ns,
            replay_end_monotonic_ns=replay_end_monotonic_ns,
            observed_monotonic_ns_min=min(r["observed_monotonic_ns"] for r in accepted_records),
            observed_monotonic_ns_max=max(r["observed_monotonic_ns"] for r in accepted_records),
            discarded_outside_replay_window_record_count=discarded_outside_window,
            max_forwards_per_step=expected_forwards,
        )
    Path(out_json).write_text(json.dumps(summary, indent=2))
    return len(accepted_records)


def extract_dp_rank_by_device(
    server_log: Path,
    *,
    records: EngineRecords = VLLM_RECORDS,
) -> dict[int, int] | None:
    """CUDA device -> data-parallel rank, straight from the engine's own words.

    Returns `None` when this engine emits no worker record, which is not a
    failure: the vLLM path recovers the same mapping by joining its pid<->rank
    banner against the profiler's pid<->device knowledge. It is only the engines
    that state the device outright that come through here.

    Everything checked below is a way the mapping could be wrong rather than
    absent, and a wrong device<->rank map is worse than no capture: it silently
    relabels one rank's kernels as another's.
    """
    if not records.emits_worker_records:
        return None
    rows = []
    for line in Path(server_log).read_text(errors="replace").splitlines():
        match = _ALIGNMENT_WORKER_RE.search(line)
        if match is None:
            continue
        row = json.loads(match.group(1))
        required = {
            "schema_version",
            "input_adapter",
            "pid",
            "device_id",
            "visible_devices",
            "tp_rank",
            "dp_rank",
            "pp_rank",
            "tp_size",
            "dp_size",
        }
        missing = required - set(row)
        if missing:
            raise ValueError(f"alignment worker record missing fields {sorted(missing)}")
        if row["schema_version"] != 1:
            raise ValueError(f"unsupported alignment worker schema {row['schema_version']!r}")
        if row["input_adapter"] != records.adapter:
            raise ValueError(
                f"alignment worker record adapter {row['input_adapter']!r} is not "
                f"{records.adapter!r}"
            )
        rows.append(row)
    if not rows:
        return None

    # One process may restart; the same pid stating two different things is a
    # log with two runs in it, which is not a capture of one run.
    by_pid: dict[int, dict] = {}
    for row in rows:
        previous = by_pid.setdefault(row["pid"], row)
        if previous != row:
            raise ValueError(f"worker pid {row['pid']} reported two different identities")
    rows = list(by_pid.values())

    visible_sets = {row["visible_devices"] for row in rows}
    if len(rows) > 1 and len(visible_sets) > 1:
        raise ValueError(
            "alignment workers do not share one CUDA_VISIBLE_DEVICES, so their device "
            "ids are indices into different sets and cannot be compared: "
            f"{sorted(map(str, visible_sets))}. "
            "Capture with SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS unset."
        )

    dp_rank_by_device: dict[int, int] = {}
    for row in rows:
        device_id, dp_rank = row["device_id"], row["dp_rank"]
        existing = dp_rank_by_device.setdefault(device_id, dp_rank)
        if existing != dp_rank:
            raise ValueError(f"device {device_id} is claimed by DP ranks {existing} and {dp_rank}")
    observed_dp_size = {row["dp_size"] for row in rows}
    if len(observed_dp_size) != 1:
        raise ValueError(f"alignment workers disagree on dp_size: {sorted(observed_dp_size)}")
    declared_dp_size = next(iter(observed_dp_size))
    if sorted(set(dp_rank_by_device.values())) != list(range(declared_dp_size)):
        raise ValueError(
            f"alignment workers cover DP ranks {sorted(set(dp_rank_by_device.values()))}, "
            f"not the full 0..{declared_dp_size - 1} the run declares"
        )
    return dp_rank_by_device
