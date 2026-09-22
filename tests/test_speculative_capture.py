"""Replay evidence must preserve measured counters and distinct model roles."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alignment import runner
from alignment.profiler import record_extraction, vllm_server
from alignment.profiler.config import ProfileConfig
from alignment.profiler.spec_decode import replay_delta


class Config(SimpleNamespace):
    """A profile config with only the fields one finalize path reads.

    It borrows the real derived property rather than restating it, so a change
    to what decides an expert-load capture reaches this test instead of quietly
    passing a stale answer.
    """

    captures_expert_load = ProfileConfig.captures_expert_load


def exposition(drafts, tokens, accepted, positions):
    values = {
        "num_drafts": drafts,
        "num_draft_tokens": tokens,
        "num_accepted_tokens": accepted,
    }
    lines = [
        f'vllm:spec_decode_{name}_total{{model_name="model with spaces",engine="0"}} {value}'
        for name, value in values.items()
    ]
    lines.extend(
        f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{position="{pos}"}} {count}'
        for pos, count in enumerate(positions)
    )
    return "\n".join(lines)


def test_replay_counters_exclude_warmup_and_survive_json():
    before = vllm_server._parse_spec_decode_metrics(exposition(10, 50, 20, [10, 5, 3, 2, 0]))
    after = vllm_server._parse_spec_decode_metrics(exposition(14, 70, 28, [14, 7, 4, 3, 0]))
    result = replay_delta(json.loads(json.dumps(before)), after)
    assert result["num_drafts"] == 4
    assert result["draft_tokens"] == 20
    assert result["accepted_tokens"] == 8
    assert result["mean_acceptance_length"] == 3
    assert result["per_position_acceptance_rates"] == {
        "0": 1,
        "1": 0.5,
        "2": 0.25,
        "3": 0.25,
        "4": 0,
    }


@pytest.mark.parametrize("value", ["NaN", "+Inf", "-1", "1.5"])
def test_invalid_prometheus_counter_is_rejected(value):
    with pytest.raises(ValueError, match="invalid spec-decode counter"):
        vllm_server._parse_spec_decode_metrics(exposition(value, 5, 0, [0] * 5))


def test_missing_or_reset_counters_fail_closed():
    with pytest.raises(ValueError, match="missing required"):
        vllm_server._parse_spec_decode_metrics("unrelated_total 42")
    before = vllm_server._parse_spec_decode_metrics(exposition(10, 50, 20, [10, 5, 3, 2, 0]))
    after = vllm_server._parse_spec_decode_metrics(exposition(0, 0, 0, [0] * 5))
    with pytest.raises(ValueError, match="decreased"):
        replay_delta(before, after)


def expert_rows():
    for step, timestamp in enumerate([99, 100, 150, 201]):
        for role in ("target", "draft"):
            yield {
                "schema_version": 3,
                "model": "moe/model",
                "eplb_step": step,
                "expert_parallel_size": 2,
                "experts_per_token": 2,
                "logical_expert_counts": [[2, 1, 1, 0], [0, 1, 1, 2]]
                if role == "target"
                else [[8, 4, 5, 3]],
                "model_role": role,
                "max_forwards_per_step": 1 if role == "target" else 5,
                "observed_monotonic_ns": timestamp,
            }


def test_popularity_separates_roles_and_preserves_reduction_group(tmp_path):
    source = tmp_path / "server.log"
    source.write_text(
        "\n".join("VibeSimAlignmentExpertLoad " + json.dumps(row) for row in expert_rows())
    )
    for role, expected in [("target", [[4, 2, 2, 0], [0, 2, 2, 4]]), ("draft", [[16, 8, 10, 6]])]:
        destination = tmp_path / f"{role}.json"
        assert (
            record_extraction.extract_expert_popularity(
                source,
                tmp_path / f"{role}.jsonl",
                destination,
                expert_parallel_size=2,
                reduction_group_size=1,
                max_tokens_per_step=2,
                model_role=role,
                replay_start_monotonic_ns=100,
                replay_end_monotonic_ns=200,
            )
            == 2
        )
        result = json.loads(destination.read_text())
        assert result["schema_version"] == 4
        assert result["model_role"] == role
        assert result["counts_by_layer"] == expected
        assert result["aggregation"]["discarded_outside_replay_window_record_count"] == 2


def test_finalize_speculative_popularity_produces_both_routing_artifacts(tmp_path):
    source = tmp_path / "server.log"
    source.write_text(
        "\n".join("VibeSimAlignmentExpertLoad " + json.dumps(row) for row in expert_rows())
    )
    cfg = Config(
        name="spec5",
        workload=SimpleNamespace(warmup=False),
        engine="vllm",
        profile_kind="expert_popularity",
        gpu="B200",
        cuda_visible_devices="0,1",
        server=SimpleNamespace(
            extra_args=['--speculative-config={"num_speculative_tokens":5}'],
            expert_parallel_size=2,
            expert_count_reduction_group_size=1,
            chunk_size=2,
            max_cudagraph_capture_size=None,
            tp_size=2,
            dp_size=1,
        ),
    )
    counters = vllm_server._parse_spec_decode_metrics(exposition(0, 0, 0, [0] * 5))
    summary = {
        "replay_start_monotonic_ns": 100,
        "replay_end_monotonic_ns": 200,
        "spec_decode_metrics_before": counters,
        "spec_decode_metrics_after": counters,
    }
    result = runner._finalize_profile(
        cfg,
        log_dir=tmp_path,
        engine_dir=tmp_path,
        server_log=source,
        out_rep=tmp_path / "unused",
        prepared_replay=SimpleNamespace(log_path=tmp_path / "replay"),
        drive_summary=summary,
        nsys_executable=None,
    )
    target = json.loads(Path(result["expert_popularity_json"]).read_text())
    draft = json.loads(Path(result["draft_expert_popularity_json"]).read_text())
    assert target["model_role"] == "target"
    assert draft["model_role"] == "draft"
    assert target["counts_by_layer"] != draft["counts_by_layer"]
    assert Path(result["spec_decode_metrics_json"]).is_file()
