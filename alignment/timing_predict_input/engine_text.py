"""Convert canonical instrumented-engine text records into predictor cases.

The record is the same shape from either instrumented fork -- the `input_adapter`
tag is the only thing that differs -- so one adapter reads both. The tag is still
required and still checked: a record that names no engine, or names one this
build does not know, is a capture defect rather than something to assume about.
"""

from __future__ import annotations

from typing import Any

SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2, 3, 4})
SUPPORTED_INPUT_ADAPTERS = frozenset({"vllm_text", "sglang_text"})


def _validated_shape(
    metric: dict[str, Any], context: str, draft_tokens: int | None = None
) -> tuple[list[list[int]], list[int] | list[list[int]], int]:
    """Check one record's batch shape and return (chunk pairs, kv lens, tokens).

    The record is the measurement; a malformed one is a capture defect, never
    something to repair silently.
    """
    if metric.get("input_adapter") not in SUPPORTED_INPUT_ADAPTERS:
        raise ValueError(
            f"{context}: expected one of {sorted(SUPPORTED_INPUT_ADAPTERS)}, "
            f"got input_adapter {metric.get('input_adapter')!r}"
        )
    if metric.get("schema_version") not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"{context}: unsupported engine-text schema_version {metric.get('schema_version')!r}"
        )

    pairs = metric.get("prefill_chunk_pairs")
    decode_kv_lens = metric.get("decode_kv_lens")
    if not isinstance(pairs, list) or not isinstance(decode_kv_lens, list):
        raise ValueError(f"{context}: malformed text adapter lists")
    if any(
        not isinstance(pair, list)
        or len(pair) != 2
        or any(not isinstance(value, int) or value < 0 for value in pair)
        for pair in pairs
    ):
        raise ValueError(f"{context}: invalid prefill_chunk_pairs")
    minimum_kv = 0 if draft_tokens is not None else 1
    if any(type(value) is not int or value < minimum_kv for value in decode_kv_lens):
        raise ValueError(f"{context}: invalid decode_kv_lens")

    prefill_tokens = int(metric.get("prefill_tokens", -1))
    decode_requests = int(metric.get("decode_requests", -1))
    scheduled_decode = int(metric.get("decode_tokens_scheduled", -1))
    if sum(pair[1] for pair in pairs) != prefill_tokens:
        raise ValueError(f"{context}: prefill pair append sum does not match prefill_tokens")
    if len(decode_kv_lens) != decode_requests:
        raise ValueError(f"{context}: decode_kv_lens count does not match decode_requests")
    if draft_tokens is None:
        if scheduled_decode != decode_requests:
            raise ValueError(
                f"{context}: speculative/multi-token decode requires a speculative arch"
            )
        return pairs, decode_kv_lens, prefill_tokens + decode_requests
    if metric["input_adapter"] != "vllm_text":
        raise ValueError(f"{context}: speculative shape lowering is only defined for vllm_text")
    queries = metric.get("decode_query_lens")
    if metric["schema_version"] < 3 or not isinstance(queries, list):
        raise ValueError(f"{context}: speculative prediction requires recorded decode_query_lens")
    if len(queries) != decode_requests or any(
        type(query) is not int or query <= 0 for query in queries
    ):
        raise ValueError(f"{context}: invalid decode_query_lens")
    if sum(queries) != scheduled_decode:
        raise ValueError(f"{context}: decode_query_lens sum differs from scheduled decode tokens")
    progress = metric.get("decode_request_progress")
    lowered_pairs = list(pairs)
    decode = []
    for index, (kv, query) in enumerate(zip(decode_kv_lens, queries, strict=True)):
        if query != draft_tokens + 1:
            # Preempted decode requests can recompute context without drafts.
            # The MLA indexer uses query width, not request lifecycle, to split
            # prefill from decode (reorder_batch_threshold = 1 + draft_tokens).
            row = (
                progress[index]
                if isinstance(progress, list) and len(progress) == decode_requests
                else None
            )
            if (
                metric["schema_version"] >= 4
                and query > draft_tokens + 1
                and isinstance(row, dict)
                and row.get("kv_len") == kv
                and row.get("query_len") == query
                and type(row.get("drafted_tokens")) is int
                and row["drafted_tokens"] == 0
                and row.get("accepted_draft_tokens") == 0
            ):
                lowered_pairs.append([kv, query])
                continue
            raise ValueError(
                f"{context}: query lengths must equal draft_tokens + 1 ({draft_tokens + 1})"
            )
        if kv == 0:
            raise ValueError(f"{context}: invalid decode_kv_lens")
        # vLLM seq_lens includes the current input and every draft position.
        decode.append([kv + query, query])
    return lowered_pairs, decode, prefill_tokens + scheduled_decode


def _dp_group_count(parsed: dict[str, Any]) -> int:
    """How many attention-DP groups this capture observed.

    Taken from the parse's device → DP-rank map so the case width is a measured
    fact about the captured replica, not a guess. The predictor separately
    asserts it equals the arch's `num_attn_dp_groups`, which is where a genuine
    topology mismatch must surface.
    """
    dp_rank_by_device = parsed.get("dp_rank_by_device") or {}
    if not dp_rank_by_device:
        raise ValueError(
            "parsed NSYS has no dp_rank_by_device map; re-parse the capture with the "
            "server log so per-rank groups can be built"
        )
    ranks = sorted({int(rank) for rank in dp_rank_by_device.values()})
    if ranks != list(range(len(ranks))):
        raise ValueError(f"observed DP ranks are not a contiguous 0..N-1 range: {ranks}")
    return len(ranks)


def build_cases(
    parsed: dict[str, Any],
    measured_phase: str,
    group_assignment: str = "single",
    *,
    draft_tokens: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Preserve exact text batch shapes and their measured-iteration join.

    `group_assignment` selects how a measured step becomes predictor groups:

    - ``single`` — one group holding the whole replica's batch. Correct only when
      the arch has one attention-DP group.
    - ``per_dp_rank`` — one group per DP rank, each carrying that rank's own
      batch. Under data parallelism the ranks run *different* batches inside one
      synchronized step, so collapsing them into one group would model a batch
      that no rank ever executed.
    """
    if group_assignment not in {"single", "per_dp_rank"}:
        raise ValueError(f"unknown group_assignment {group_assignment!r}")
    if draft_tokens is not None and (type(draft_tokens) is not int or draft_tokens <= 0):
        raise ValueError("draft_tokens must be a positive integer")
    decode_key = "decode_kv_lens" if draft_tokens is None else "decode_requests"

    group_count = _dp_group_count(parsed) if group_assignment == "per_dp_rank" else 1
    cases: list[dict[str, Any]] = []
    case_map: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for detail in parsed.get("iteration_details", []):
        iteration = detail.get("iteration")
        metric = detail.get("metrics")
        phase_ranges = [
            item for item in detail.get("ranges", []) if item.get("phase") == measured_phase
        ]
        if not isinstance(metric, dict):
            excluded.append({"iteration": iteration, "reason": "missing iteration metrics"})
            continue
        if not phase_ranges or not any(item.get("kernel_count", 0) for item in phase_ranges):
            excluded.append(
                {"iteration": iteration, "reason": f"no kernels in {measured_phase!r} phase"}
            )
            continue

        # The aggregate always validates: it classifies the step and gates the
        # empty-batch exclusion regardless of how the groups are laid out.
        aggregate_pairs, _, aggregate_workload = _validated_shape(
            metric, f"iteration {iteration}", draft_tokens
        )
        if aggregate_workload == 0:
            excluded.append({"iteration": iteration, "reason": "empty model batch"})
            continue

        if group_assignment == "single":
            pairs, decode, _ = _validated_shape(metric, f"iteration {iteration}", draft_tokens)
            groups = [{"prefill_chunk_pairs": pairs, decode_key: decode}]
        else:
            by_rank = detail.get("metrics_by_dp_rank") or {}
            groups = []
            for dp_rank in range(group_count):
                rank_metric = by_rank.get(str(dp_rank))
                if rank_metric is None:
                    # The rank scheduled nothing this step. It still executes a
                    # forward pass to keep the EP collectives in lockstep, so the
                    # group exists and is empty rather than absent.
                    groups.append({"prefill_chunk_pairs": [], decode_key: []})
                    continue
                pairs, decode, _ = _validated_shape(
                    rank_metric, f"iteration {iteration} dp_rank {dp_rank}", draft_tokens
                )
                groups.append({"prefill_chunk_pairs": pairs, decode_key: decode})

        case_index = len(cases)
        cases.append({"groups": groups})
        case_map.append(
            {
                "case_index": case_index,
                "measured_iteration": int(iteration),
                "stage": "mixed" if any(pair[1] for pair in aggregate_pairs) else "decode",
            }
        )
    if not cases:
        raise ValueError("no measured iterations can be converted to timing-predict cases")
    return cases, case_map, excluded
