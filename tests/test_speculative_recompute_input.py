"""Preempted decode context must remain work in offline prediction."""

from copy import deepcopy

import pytest

from alignment.timing_predict_input.engine_text import TruncatedDraft, _validated_shape, build_cases


def recompute_metric():
    return {
        "input_adapter": "vllm_text", "schema_version": 4,
        "prefill_chunk_pairs": [[10, 20]], "prefill_tokens": 20,
        "decode_requests": 3, "decode_tokens_scheduled": 2451,
        "decode_kv_lens": [100, 0, 65600],
        "decode_query_lens": [6, 2195, 250],
        "decode_request_progress": [
            {"kv_len": 100, "query_len": 6, "drafted_tokens": 5,
             "accepted_draft_tokens": 2},
            {"kv_len": 0, "query_len": 2195, "drafted_tokens": 0,
             "accepted_draft_tokens": 0},
            {"kv_len": 65600, "query_len": 250, "drafted_tokens": 0,
             "accepted_draft_tokens": 0},
        ],
    }


def test_recompute_preserves_rows_and_does_not_mutate_measurement():
    metric = recompute_metric()
    original = deepcopy(metric)
    pairs, decode, tokens = _validated_shape(metric, "test", 5)
    assert pairs == [[10, 20], [0, 2195], [65600, 250]]
    assert decode == [[106, 6]]
    assert tokens == 2471 == sum(p[1] for p in pairs + decode)
    assert metric == original


@pytest.mark.parametrize("defect", ["missing", "drafted", "mismatch", "legacy", "negative"])
def test_recompute_requires_matching_observed_evidence(defect):
    metric = recompute_metric()
    if defect == "missing":
        del metric["decode_request_progress"]
    elif defect == "drafted":
        metric["decode_request_progress"][1]["drafted_tokens"] = 5
    elif defect == "mismatch":
        metric["decode_request_progress"][1]["kv_len"] = 1
    elif defect == "legacy":
        metric["schema_version"] = 3
    else:
        metric["decode_kv_lens"][1] = -1
    with pytest.raises(ValueError):
        _validated_shape(metric, "test", 5)


def truncated_draft_detail():
    return {
        "iteration": 7,
        "ranges": [{"phase": "forward", "kernel_count": 10}],
        "metrics": {
            "input_adapter": "vllm_text", "schema_version": 4,
            "prefill_chunk_pairs": [], "prefill_tokens": 0,
            "decode_requests": 1, "decode_tokens_scheduled": 5,
            "decode_kv_lens": [131066], "decode_query_lens": [5],
            "decode_request_progress": [
                {"kv_len": 131066, "query_len": 5, "drafted_tokens": 4,
                 "accepted_draft_tokens": 4},
            ],
        },
    }


def test_truncated_draft_step_is_excluded_by_name():
    """At max_model_len the engine drafts fewer positions; that step has no
    uniform-width prediction and is excluded with its reason, not repaired."""
    detail = truncated_draft_detail()
    with pytest.raises(TruncatedDraft):
        _validated_shape(detail["metrics"], "test", 5)
    full = deepcopy(detail)
    full["iteration"] = 8
    full["metrics"].update(decode_tokens_scheduled=6, decode_kv_lens=[100],
                           decode_query_lens=[6], decode_request_progress=None)
    cases, case_map, excluded = build_cases(
        {"iteration_details": [detail, full]}, "forward", draft_tokens=5
    )
    assert [item["measured_iteration"] for item in case_map] == [8]
    assert excluded == [{"iteration": 7, "reason": "iteration 7: draft truncated to 4 of 5 tokens"}]


def test_short_query_without_truncation_evidence_still_fails():
    metric = truncated_draft_detail()["metrics"]
    metric["decode_request_progress"][0]["drafted_tokens"] = 5
    with pytest.raises(ValueError) as error:
        _validated_shape(metric, "test", 5)
    assert not isinstance(error.value, TruncatedDraft)
