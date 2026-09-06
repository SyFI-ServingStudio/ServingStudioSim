import json

import pytest

from scripts.prepare_matrix_e2e_reuse import full_rates, request_rates


def test_full_rates_are_conditional_not_marginal():
    assert full_rates({
        "num_drafts": 100,
        "accepted_tokens_per_position": dict(zip(map(str, range(5)), [80, 40, 20, 10, 5])),
    }) == [0.8, 0.5, 0.5, 0.5, 0.5]


def test_request_acceptance_keeps_rejections_and_excludes_finished_queue(tmp_path):
    path = tmp_path / "metrics.jsonl"
    steps = [
        {"request_id": "independent_a", "drafted_tokens": 5, "accepted_draft_tokens": 2},
        {"request_id": "independent_a", "drafted_tokens": 5, "accepted_draft_tokens": 0},
        {"request_id": "independent_a", "drafted_tokens": 5, "accepted_draft_tokens": 5,
         "request_finished_before": True},
    ]
    path.write_text(json.dumps({"decode_request_progress": steps}) + "\n")
    rates, fallback = request_rates(path, {"a"}, [0.3] * 5)
    assert rates == {"a": [0.5, 1.0, 0.0, 0.3, 0.3]}
    assert fallback == 2
    with pytest.raises(ValueError, match="outside full trace"):
        request_rates(path, {"other"}, [0.3] * 5)
