"""Preempted decode context must remain work in offline prediction."""

from copy import deepcopy

import pytest

from alignment.timing_predict_input.engine_text import _validated_shape


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
