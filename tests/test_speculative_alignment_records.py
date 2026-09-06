"""CPU checks across the pinned vLLM producer and alignment record consumer."""

from __future__ import annotations

import importlib.util
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

from alignment.nsys.parse import fold_rank_metrics
from alignment.profiler.record_extraction import extract_metrics_jsonl
from alignment.timing_predict_input import BuildRequest, build_inputs
from alignment.timing_predict_input.builder import EngineTextInputSpec


@pytest.fixture
def request_details():
    path = (
        Path(__file__).resolve().parents[1] / "alignment/profiler/vllm/vllm/v1/engine/alignment.py"
    )
    spec = importlib.util.spec_from_file_location("vllm_alignment_records", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return partial(module.iteration_request_details, requests={
        "decode-a": SimpleNamespace(num_output_tokens=7, is_finished=lambda: False),
        "decode-b": SimpleNamespace(num_output_tokens=9, is_finished=lambda: False),
    })


def scheduled_batch():
    return SimpleNamespace(
        scheduled_new_reqs=[SimpleNamespace(req_id="new", num_computed_tokens=0)],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["decode-a", "chunk", "paused", "decode-b"],
            num_computed_tokens=[100, 20, 50, 200],
            num_output_tokens=[7, 0, 3, 9],
            is_context_phase=lambda request_id: request_id == "chunk",
        ),
        num_scheduled_tokens={"new": 8, "decode-a": 6, "chunk": 4, "decode-b": 6},
        scheduled_spec_decode_tokens={"decode-a": [1] * 5, "decode-b": [2] * 5},
    )


def iteration_row(details):
    return {
        "schema_version": 4,
        "input_adapter": "vllm_text",
        "iteration_index": 0,
        "observed_start_monotonic_ns": 100,
        "observed_end_monotonic_ns": 200,
        "observed_elapsed_ms": 0.0001,
        "prefill_tokens": 12,
        "decode_requests": 2,
        "decode_tokens_scheduled": 12,
        **details,
    }


def extract(tmp_path, row):
    source = tmp_path / "server.log"
    source.write_text("VibeSimAlignmentIteration " + json.dumps(row) + "\n")
    output = tmp_path / "metrics.jsonl"
    assert extract_metrics_jsonl(source, output) == 1
    return json.loads(output.read_text())


def test_spec5_progress_round_trips_with_reordered_output(request_details, tmp_path):
    output = SimpleNamespace(
        req_id_to_index={"decode-b": 0, "decode-a": 1},
        sampled_token_ids=[[8], [1, 2, 3, 4]],
    )
    result = extract(tmp_path, iteration_row(request_details(scheduled_batch(), output)))
    assert result["prefill_chunk_pairs"] == [[0, 8], [20, 4]]
    assert result["decode_kv_lens"] == [100, 200]
    assert result["decode_query_lens"] == [6, 6]
    progress = result["decode_request_progress"]
    assert [request["accepted_draft_tokens"] for request in progress] == [3, 0]
    assert [request["output_tokens_before"] for request in progress] == [7, 9]
    assert [request["request_id"] for request in progress] == ["decode-a", "decode-b"]


def test_zero_sampled_output_does_not_invent_a_bonus_token(request_details, tmp_path):
    output = SimpleNamespace(
        req_id_to_index={"decode-a": 0, "decode-b": 1}, sampled_token_ids=[[], []]
    )
    result = extract(tmp_path, iteration_row(request_details(scheduled_batch(), output)))
    assert [request["emitted_tokens"] for request in result["decode_request_progress"]] == [0, 0]


def test_async_placeholders_do_not_advance_confirmed_progress(request_details, tmp_path):
    batch = scheduled_batch()
    batch.scheduled_cached_reqs.num_output_tokens = [68, 0, 3, 15]
    output = SimpleNamespace(
        req_id_to_index={"decode-a": 0, "decode-b": 1}, sampled_token_ids=[[1], [2]]
    )
    requests = {
        "decode-a": SimpleNamespace(num_output_tokens=60, is_finished=lambda: False),
    }
    result = extract(tmp_path, iteration_row(request_details(batch, output, requests=requests)))
    active, finished = result["decode_request_progress"]
    assert active["output_tokens_before"] == 60
    assert active["request_finished_before"] is False
    assert finished["output_tokens_before"] is None
    assert finished["request_finished_before"] is True
    assert finished["query_len"] == 6
    assert finished["emitted_tokens"] == 1


@pytest.mark.parametrize("randomized", [False, True])
def test_internal_id_normalization_preserves_a_client_hex_suffix(
    request_details, tmp_path, randomized,
):
    external = "generate-tokens-client-deadbeef"
    internal = external + "-12345678" if randomized else external
    batch = scheduled_batch()
    batch.scheduled_new_reqs = []
    batch.scheduled_cached_reqs = SimpleNamespace(
        req_ids=[internal], num_computed_tokens=[100], num_output_tokens=[1],
        is_context_phase=lambda _: False,
    )
    batch.num_scheduled_tokens = {internal: 6}
    batch.scheduled_spec_decode_tokens = {internal: [1] * 5}
    output = SimpleNamespace(req_id_to_index={internal: 0}, sampled_token_ids=[[1]])
    details = request_details(batch, output, request_ids_randomized=randomized, requests={
        internal: SimpleNamespace(num_output_tokens=1, is_finished=lambda: False),
    })
    row = iteration_row(details)
    row.update(prefill_tokens=0, decode_requests=1, decode_tokens_scheduled=6)
    result = extract(tmp_path, row)
    assert result["decode_request_progress"][0]["request_id"] == "client-deadbeef"


@pytest.mark.parametrize(
    "field,value",
    [
        ("query_len", 2),
        ("accepted_draft_tokens", 6),
        ("drafted_tokens", True),
        ("engine_request_id", "decode-b"),
        ("output_tokens_before", None),
        ("request_finished_before", 1),
    ],
)
def test_invalid_progress_is_rejected(request_details, tmp_path, field, value):
    output = SimpleNamespace(
        req_id_to_index={"decode-a": 0, "decode-b": 1}, sampled_token_ids=[[1], [2]]
    )
    row = iteration_row(request_details(scheduled_batch(), output))
    row["decode_request_progress"][0][field] = value
    with pytest.raises(ValueError):
        extract(tmp_path, row)


def test_producer_rejects_missing_or_impossible_output(request_details):
    with pytest.raises(ValueError, match="requires model output"):
        request_details(scheduled_batch(), None)
    output = SimpleNamespace(
        req_id_to_index={"decode-a": 0, "decode-b": 1}, sampled_token_ids=[[1] * 7, [2]]
    )
    with pytest.raises(ValueError, match="invalid alignment decode output"):
        request_details(scheduled_batch(), output)


def test_request_records_reach_speculative_predict_cases(request_details, tmp_path):
    output = SimpleNamespace(
        req_id_to_index={"decode-a": 0, "decode-b": 1}, sampled_token_ids=[[1], [2]]
    )
    row = extract(tmp_path, iteration_row(request_details(scheduled_batch(), output)))
    aggregate = fold_rank_metrics([row], 0)
    assert aggregate["decode_request_progress"] == row["decode_request_progress"]
    parsed = tmp_path / "parsed.json"
    parsed.write_text(
        json.dumps(
            {
                "iteration_details": [
                    {
                        "iteration": 0,
                        "metrics": aggregate,
                        "ranges": [{"phase": "forward", "kernel_count": 1}],
                    }
                ]
            }
        )
    )
    arch = {
        "type": "glm52_vllm_nvfp4_dsa_moe_speculative",
        "draft_tokens": 5,
        "model_config": "model/config/glm52_nvfp4.json",
    }
    result = build_inputs(
        BuildRequest(
            simulation_preset=tmp_path / "preset.json",
            profile_log_dir=tmp_path,
            parsed_nsys=parsed,
            output_dir=tmp_path / "predict",
            gpu="NVIDIA B200",
            arch=arch,
            backends={},
            input_spec=EngineTextInputSpec(),
        )
    )
    config = json.loads(result.predict_config.read_text())
    assert config["arch"] == {"speculative_iter": arch}
    assert json.loads(result.cases.read_text()) == [
        {
            "groups": [
                {
                    "prefill_chunk_pairs": [[0, 8], [20, 4]],
                    "decode_requests": [[106, 6], [206, 6]],
                }
            ]
        }
    ]
