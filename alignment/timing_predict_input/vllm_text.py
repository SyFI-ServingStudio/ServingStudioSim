"""Convert canonical instrumented-vLLM text records into predictor cases."""

from __future__ import annotations

from typing import Any


def build_cases(
    parsed: dict[str, Any], measured_phase: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Preserve exact text batch shapes and their measured-iteration join."""
    cases = []
    case_map = []
    excluded = []
    for detail in parsed.get("iteration_details", []):
        iteration = detail.get("iteration")
        metric = detail.get("metrics")
        phase_ranges = [
            item for item in detail.get("ranges", []) if item.get("phase") == measured_phase
        ]
        if not isinstance(metric, dict):
            excluded.append({"iteration": iteration, "reason": "missing iteration metrics"})
            continue
        if metric.get("input_adapter") != "vllm_text" or metric.get("schema_version") != 1:
            raise ValueError(f"iteration {iteration}: expected vllm_text schema_version 1 record")
        if not phase_ranges or not any(item.get("kernel_count", 0) for item in phase_ranges):
            excluded.append(
                {"iteration": iteration, "reason": f"no kernels in {measured_phase!r} phase"}
            )
            continue

        pairs = metric.get("prefill_chunk_pairs")
        decode_kv_lens = metric.get("decode_kv_lens")
        if not isinstance(pairs, list) or not isinstance(decode_kv_lens, list):
            raise ValueError(f"iteration {iteration}: malformed text adapter lists")
        if any(
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(value, int) or value < 0 for value in pair)
            for pair in pairs
        ):
            raise ValueError(f"iteration {iteration}: invalid prefill_chunk_pairs")
        if any(not isinstance(value, int) or value <= 0 for value in decode_kv_lens):
            raise ValueError(f"iteration {iteration}: invalid decode_kv_lens")

        prefill_tokens = int(metric.get("prefill_tokens", -1))
        decode_requests = int(metric.get("decode_requests", -1))
        scheduled_decode = int(metric.get("decode_tokens_scheduled", -1))
        if sum(pair[1] for pair in pairs) != prefill_tokens:
            raise ValueError(
                f"iteration {iteration}: prefill pair append sum does not match prefill_tokens"
            )
        if len(decode_kv_lens) != decode_requests:
            raise ValueError(
                f"iteration {iteration}: decode_kv_lens count does not match decode_requests"
            )
        if scheduled_decode != decode_requests:
            raise ValueError(
                f"iteration {iteration}: speculative/multi-token decode is not supported by v1"
            )
        if prefill_tokens + decode_requests == 0:
            excluded.append({"iteration": iteration, "reason": "empty model batch"})
            continue

        case_index = len(cases)
        cases.append(
            {
                "groups": [
                    {
                        "prefill_chunk_pairs": pairs,
                        "decode_kv_lens": decode_kv_lens,
                    }
                ]
            }
        )
        case_map.append(
            {
                "case_index": case_index,
                "measured_iteration": int(iteration),
                "stage": "mixed" if prefill_tokens > 0 else "decode",
            }
        )
    if not cases:
        raise ValueError("no measured iterations can be converted to timing-predict cases")
    return cases, case_map, excluded
