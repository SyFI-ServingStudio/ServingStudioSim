"""CPU contract tests for the v1 alignment preparation and analyzer stack."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alignment.profiler import nsys_capture, vllm_server
from alignment.profiler.config import NsysConfig
from alignment.timing_predict_input.vllm_text import build_cases
from launcher.alignment_config import load_labeled_kernel_sequences

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cuda_profiler_api_nsys_prefix_targets_spawned_worker(tmp_path):
    executable = nsys_capture.ResolvedNsysExecutable(
        path=tmp_path / "nsys",
        version="NVIDIA Nsight Systems version 2025.1",
    )
    prefix = nsys_capture.build_nsys_prefix(
        executable,
        NsysConfig(capture_mode="cuda_profiler_api"),
        tmp_path / "profile",
    )

    assert prefix[0] == str(executable.path)
    assert "--trace-fork-before-exec=true" in prefix
    assert "--cuda-event-trace=false" in prefix
    assert "--capture-range=cudaProfilerApi" in prefix
    assert "--capture-range-end=stop" in prefix


def test_nsys_executable_requires_explicit_absolute_path(monkeypatch):
    monkeypatch.delenv("NSYS_BIN", raising=False)
    with pytest.raises(ValueError, match="NSYS executable is not configured"):
        nsys_capture.resolve_nsys_executable(None)
    with pytest.raises(ValueError, match="must be an absolute path"):
        nsys_capture.resolve_nsys_executable("nsys")


def test_nsys_executable_records_resolved_path_and_version(tmp_path, monkeypatch):
    executable_path = tmp_path / "nsys"
    executable_path.write_text("#!/bin/sh\nprintf 'Nsight Systems 2025.1\\n'\n")
    executable_path.chmod(0o755)
    monkeypatch.setenv("NSYS_BIN", str(executable_path))

    executable = nsys_capture.resolve_nsys_executable(None)

    assert executable.provenance() == {
        "executable": str(executable_path.resolve()),
        "version": "Nsight Systems 2025.1",
    }


def test_bounded_nsys_capture_requires_positive_cuda_profiler_window():
    NsysConfig(capture_duration_seconds=30.0).validate()

    with pytest.raises(ValueError, match="must be positive"):
        NsysConfig(capture_duration_seconds=0.0).validate()
    with pytest.raises(ValueError, match="requires capture_mode=cuda_profiler_api"):
        NsysConfig(capture_mode="full", capture_duration_seconds=30.0).validate()


def test_structured_vllm_iteration_record_is_the_only_metrics_contract(tmp_path):
    record = {
        "schema_version": 1,
        "input_adapter": "vllm_text",
        "iteration_index": 7,
        "prefill_tokens": 8,
        "decode_requests": 2,
        "decode_tokens_scheduled": 2,
        "prefill_chunk_pairs": [[4, 8]],
        "decode_kv_lens": [100, 120],
    }
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "Iteration(6): 1 context requests, 8 context tokens, "
        "2 generation requests, 2 generation tokens\n"
        f"INFO VibeSimAlignmentIteration {json.dumps(record)}\n"
    )
    output = tmp_path / "metrics.jsonl"

    assert vllm_server.extract_metrics_jsonl(server_log, output) == 1
    # A single-EngineCore capture needs no rank prefix and lands on rank 0.
    assert json.loads(output.read_text()) == {**record, "dp_rank": 0}


def test_structured_vllm_iteration_v2_requires_observed_timing(tmp_path):
    record = {
        "schema_version": 2,
        "input_adapter": "vllm_text",
        "iteration_index": 7,
        "observed_start_monotonic_ns": 1_000_000,
        "observed_end_monotonic_ns": 2_500_000,
        "observed_elapsed_ms": 1.5,
        "prefill_tokens": 0,
        "decode_requests": 2,
        "decode_tokens_scheduled": 2,
        "prefill_chunk_pairs": [],
        "decode_kv_lens": [100, 120],
    }
    server_log = tmp_path / "server.log"
    server_log.write_text(f"INFO VibeSimAlignmentIteration {json.dumps(record)}\n")
    output = tmp_path / "metrics.jsonl"

    assert vllm_server.extract_metrics_jsonl(server_log, output) == 1
    # A single-EngineCore capture needs no rank prefix and lands on rank 0.
    assert json.loads(output.read_text()) == {**record, "dp_rank": 0}


def _dp_iteration_record(iteration: int, prefill_tokens: int) -> dict:
    return {
        "schema_version": 1,
        "input_adapter": "vllm_text",
        "iteration_index": iteration,
        "prefill_tokens": prefill_tokens,
        "decode_requests": 1,
        "decode_tokens_scheduled": 1,
        "prefill_chunk_pairs": [[0, prefill_tokens]] if prefill_tokens else [],
        "decode_kv_lens": [64],
    }


def test_data_parallel_iteration_records_are_stamped_with_their_engine_rank(tmp_path):
    """Two ranks reuse one iteration index; only the rank tag keeps them apart."""
    rank0 = _dp_iteration_record(4, prefill_tokens=8)
    rank1 = _dp_iteration_record(4, prefill_tokens=0)
    server_log = tmp_path / "server.log"
    server_log.write_text(
        f"(EngineCore_DP0 pid=11) INFO VibeSimAlignmentIteration {json.dumps(rank0)}\n"
        f"(EngineCore_DP1 pid=12) INFO VibeSimAlignmentIteration {json.dumps(rank1)}\n"
    )
    output = tmp_path / "metrics.jsonl"

    assert vllm_server.extract_metrics_jsonl(server_log, output, dp_size=2) == 2
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows == [{**rank0, "dp_rank": 0}, {**rank1, "dp_rank": 1}]


def test_data_parallel_capture_rejects_untagged_iteration_records(tmp_path):
    """Silently folding every rank onto rank 0 would destroy dp_size-1 of the data."""
    server_log = tmp_path / "server.log"
    server_log.write_text(
        f"INFO VibeSimAlignmentIteration {json.dumps(_dp_iteration_record(4, 8))}\n"
    )

    with pytest.raises(ValueError, match="no EngineCore_DP<k> log prefix"):
        vllm_server.extract_metrics_jsonl(server_log, tmp_path / "metrics.jsonl", dp_size=2)


def test_worker_rank_banner_supplies_the_pid_to_rank_join(tmp_path):
    """nsys gives pid → device; this banner gives pid → rank. Together: rank → device."""
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "(Worker pid=101) INFO [parallel_state.py:1568] world_size=2 rank=1 local_rank=1 "
        "distributed_init_method=tcp://127.0.0.1:1 backend=nccl\n"
        "(Worker pid=100) INFO [parallel_state.py:1568] world_size=2 rank=0 local_rank=0 "
        "distributed_init_method=tcp://127.0.0.1:1 backend=nccl\n"
    )

    assert vllm_server.extract_worker_device_ranks(server_log) == {100: 0, 101: 1}


def test_worker_rank_banner_rejects_an_incomplete_population(tmp_path):
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "(Worker pid=100) INFO [parallel_state.py:1568] world_size=2 rank=0 local_rank=0 "
        "distributed_init_method=tcp://127.0.0.1:1 backend=nccl\n"
    )

    with pytest.raises(ValueError, match="world_size=2"):
        vllm_server.extract_worker_device_ranks(server_log)


def test_structured_vllm_request_timing_is_extracted_separately(tmp_path):
    record = {
        "schema_version": 1,
        "engine_request_id": "cmpl-vibesim_7-0",
        "engine_core_ttft_ms": 12.5,
        "engine_queue_wait_ms": 3.0,
        "engine_first_schedule_to_first_token_ms": 9.5,
    }
    server_log = tmp_path / "server.log"
    server_log.write_text(f"INFO VibeSimAlignmentRequestTiming {json.dumps(record)}\n")
    output = tmp_path / "request_timings.jsonl"

    assert vllm_server.extract_request_timings_jsonl(server_log, output) == 1
    assert json.loads(output.read_text()) == {**record, "request_id": "vibesim_7"}


def test_structured_vllm_request_timing_v2_includes_server_tpot(tmp_path):
    record = {
        "schema_version": 2,
        "engine_request_id": "cmpl-vibesim_7-0",
        "engine_core_ttft_ms": 12.5,
        "engine_queue_wait_ms": 3.0,
        "engine_first_schedule_to_first_token_ms": 9.5,
        "engine_core_decode_ms": 180.0,
        "num_output_tokens": 10,
        "engine_core_tpot_ms": 20.0,
    }
    server_log = tmp_path / "server.log"
    server_log.write_text(f"INFO VibeSimAlignmentRequestTiming {json.dumps(record)}\n")
    output = tmp_path / "request_timings.jsonl"

    assert vllm_server.extract_request_timings_jsonl(server_log, output) == 1
    assert json.loads(output.read_text()) == {**record, "request_id": "vibesim_7"}


def test_structured_vllm_tokens_request_timing_normalizes_native_envelope(tmp_path):
    record = {
        "schema_version": 2,
        "engine_request_id": "generate-tokens-vibesim_7",
        "engine_core_ttft_ms": 12.5,
        "engine_queue_wait_ms": 3.0,
        "engine_first_schedule_to_first_token_ms": 9.5,
        "engine_core_decode_ms": 180.0,
        "num_output_tokens": 10,
        "engine_core_tpot_ms": 20.0,
    }
    server_log = tmp_path / "server.log"
    server_log.write_text(f"INFO VibeSimAlignmentRequestTiming {json.dumps(record)}\n")
    output = tmp_path / "request_timings.jsonl"

    assert (
        vllm_server.extract_request_timings_jsonl(
            server_log, output, expected_request_ids={"vibesim_7"}
        )
        == 1
    )
    assert json.loads(output.read_text()) == {**record, "request_id": "vibesim_7"}


def test_structured_vllm_request_timing_v3_joins_api_sse_durations(tmp_path):
    engine_record = {
        "schema_version": 2,
        "engine_request_id": "cmpl-vibesim_7-0",
        "engine_core_ttft_ms": 12.5,
        "engine_queue_wait_ms": 3.0,
        "engine_first_schedule_to_first_token_ms": 9.5,
        "engine_core_decode_ms": 180.0,
        "num_output_tokens": 10,
        "engine_core_tpot_ms": 20.0,
    }
    api_record = {
        "schema_version": 3,
        "api_request_id": "cmpl-vibesim_7",
        "output_tokens": 10,
        "token_events": 9,
        "first_token_event_tokens": 1,
        "api_frontend_prepare_ms": 1.0,
        "api_first_output_wait_ms": 13.0,
        "api_stream_activation_ms": 1.0,
        "api_add_request_ms": 1.5,
        "api_collector_wait_ms": 9.0,
        "api_engine_output_wait_ms": 7.0,
        "api_output_fanout_ms": 2.0,
        "api_collector_wakeup_ms": 0.5,
        "api_generator_resume_ms": 1.0,
        "api_first_output_serialize_ms": 0.1,
        "api_token_output_receive_span_ms": 181.0,
        "api_token_sse_yield_span_ms": 182.0,
        "api_terminal_tail_ms": 0.2,
        "engine_core_ttft_ms": 12.5,
        "engine_core_decode_ms": 180.0,
    }
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "\n".join(
            [
                f"INFO VibeSimAlignmentRequestTiming {json.dumps(engine_record)}",
                f"INFO VibeSimAlignmentApiRequestTiming {json.dumps(api_record)}",
            ]
        )
    )
    output = tmp_path / "request_timings.jsonl"

    assert vllm_server.extract_request_timings_jsonl(server_log, output) == 1
    row = json.loads(output.read_text())
    assert row["schema_version"] == 3
    assert row["engine_timing_schema_version"] == 2
    assert row["api_timing_schema_version"] == 3
    assert row["request_id"] == "vibesim_7"
    assert row["api_request_id"] == "cmpl-vibesim_7"
    assert row["api_token_sse_yield_span_ms"] == 182.0
    assert row["api_collector_wait_ms"] == 9.0
    assert row["api_engine_output_wait_ms"] == 7.0
    assert row["api_output_fanout_ms"] == 2.0


def test_structured_vllm_request_timing_v2_rejects_inconsistent_tpot(tmp_path):
    record = {
        "schema_version": 2,
        "engine_request_id": "cmpl-vibesim_7-0",
        "engine_core_ttft_ms": 12.5,
        "engine_queue_wait_ms": 3.0,
        "engine_first_schedule_to_first_token_ms": 9.5,
        "engine_core_decode_ms": 180.0,
        "num_output_tokens": 10,
        "engine_core_tpot_ms": 19.0,
    }
    server_log = tmp_path / "server.log"
    server_log.write_text(f"INFO VibeSimAlignmentRequestTiming {json.dumps(record)}\n")

    with pytest.raises(ValueError, match="does not equal"):
        vllm_server.extract_request_timings_jsonl(server_log, tmp_path / "request_timings.jsonl")


def test_request_timing_extraction_filters_frontend_preflight_requests(tmp_path):
    def record(engine_request_id: str) -> dict:
        return {
            "schema_version": 1,
            "engine_request_id": engine_request_id,
            "engine_core_ttft_ms": 12.5,
            "engine_queue_wait_ms": 3.0,
            "engine_first_schedule_to_first_token_ms": 9.5,
        }

    server_log = tmp_path / "server.log"
    server_log.write_text(
        "\n".join(
            f"INFO VibeSimAlignmentRequestTiming {json.dumps(row)}"
            for row in [
                record("cmpl-prefix-cache-probe-0"),
                record("cmpl-vibesim_7-0"),
                record("cmpl-vibesim_8-0"),
            ]
        )
    )
    output = tmp_path / "request_timings.jsonl"

    assert (
        vllm_server.extract_request_timings_jsonl(
            server_log,
            output,
            expected_request_ids={"vibesim_7", "vibesim_8"},
        )
        == 2
    )
    assert [json.loads(line)["request_id"] for line in output.read_text().splitlines()] == [
        "vibesim_7",
        "vibesim_8",
    ]


def test_vllm_text_adapter_preserves_exact_shape_and_join():
    parsed = {
        "iteration_details": [
            {
                "iteration": 34,
                "metrics": {
                    "schema_version": 1,
                    "input_adapter": "vllm_text",
                    "prefill_tokens": 8,
                    "decode_requests": 2,
                    "decode_tokens_scheduled": 2,
                    "prefill_chunk_pairs": [[4, 8]],
                    "decode_kv_lens": [100, 120],
                },
                "ranges": [{"phase": "forward", "kernel_count": 2}],
            }
        ]
    }
    cases, case_map, excluded = build_cases(parsed, "forward")

    assert cases == [
        {
            "groups": [
                {
                    "prefill_chunk_pairs": [[4, 8]],
                    "decode_kv_lens": [100, 120],
                }
            ]
        }
    ]
    assert case_map == [{"case_index": 0, "measured_iteration": 34, "stage": "mixed"}]
    assert excluded == []


def _rank_record(prefill_tokens: int, decode_kv_lens: list[int]) -> dict:
    return {
        "schema_version": 1,
        "input_adapter": "vllm_text",
        "prefill_tokens": prefill_tokens,
        "decode_requests": len(decode_kv_lens),
        "decode_tokens_scheduled": len(decode_kv_lens),
        "prefill_chunk_pairs": [[0, prefill_tokens]] if prefill_tokens else [],
        "decode_kv_lens": decode_kv_lens,
    }


def _dp_parsed(metrics_by_dp_rank: dict[str, dict], aggregate: dict) -> dict:
    return {
        "dp_rank_by_device": {"0": 0, "1": 1, "2": 2},
        "iteration_details": [
            {
                "iteration": 34,
                "metrics": aggregate,
                "metrics_by_dp_rank": metrics_by_dp_rank,
                "ranges": [{"phase": "forward", "kernel_count": 2}],
            }
        ],
    }


def test_per_dp_rank_groups_keep_each_ranks_own_batch():
    """One synchronized step runs a different batch on every rank; one pooled
    group would model a batch that no rank ever executed."""
    parsed = _dp_parsed(
        {
            "0": _rank_record(8, [100]),
            "1": _rank_record(0, [120, 130]),
            "2": _rank_record(4, []),
        },
        aggregate=_rank_record(12, [100, 120, 130]),
    )

    cases, case_map, excluded = build_cases(parsed, "forward", "per_dp_rank")

    assert cases == [
        {
            "groups": [
                {"prefill_chunk_pairs": [[0, 8]], "decode_kv_lens": [100]},
                {"prefill_chunk_pairs": [], "decode_kv_lens": [120, 130]},
                {"prefill_chunk_pairs": [[0, 4]], "decode_kv_lens": []},
            ]
        }
    ]
    assert case_map == [{"case_index": 0, "measured_iteration": 34, "stage": "mixed"}]
    assert excluded == []


def test_per_dp_rank_gives_an_idle_rank_an_empty_group():
    """A rank that scheduled nothing still runs a forward pass to keep the EP
    collectives in lockstep, so its group is empty rather than absent."""
    parsed = _dp_parsed(
        {"0": _rank_record(0, [100]), "2": _rank_record(0, [140])},
        aggregate=_rank_record(0, [100, 140]),
    )

    cases, _, _ = build_cases(parsed, "forward", "per_dp_rank")

    assert cases[0]["groups"] == [
        {"prefill_chunk_pairs": [], "decode_kv_lens": [100]},
        {"prefill_chunk_pairs": [], "decode_kv_lens": []},
        {"prefill_chunk_pairs": [], "decode_kv_lens": [140]},
    ]


def test_per_dp_rank_requires_a_dp_aware_parse():
    parsed = {
        "iteration_details": [
            {
                "iteration": 34,
                "metrics": _rank_record(8, [100]),
                "ranges": [{"phase": "forward", "kernel_count": 2}],
            }
        ]
    }

    with pytest.raises(ValueError, match="no dp_rank_by_device map"):
        build_cases(parsed, "forward", "per_dp_rank")


def _labeled_doc(phases: dict | None = None) -> dict:
    return {
        "schema_version": 2,
        "encoding": "folded-v1",
        "source_parsed": "profile/parsed.json",
        "folding_policy": {
            "kind": "exact_contiguous_repeat",
            "match_fields": ["name", "suggested_category"],
            "row_identity": "sequence_id:expanded_ordinal",
        },
        "phases": phases
        or {
            "forward": {
                "unique_sequences": [
                    {
                        "sequence_id": "sequence_a",
                        "iterations": [34],
                        "expanded_kernel_count": 1,
                        "program": [
                            {
                                "kernels": [
                                    {
                                        "name": "attention_kernel",
                                        "suggested_category": "attention",
                                        "label": {
                                            "status": "mapped",
                                            "operation": "attention",
                                            "type": "attention",
                                            "role": "attention main",
                                            "simulated_slots": ["unified.attn"],
                                        },
                                    }
                                ]
                            }
                        ],
                    }
                ]
            }
        },
    }


def test_labeled_sequences_require_explicit_status(tmp_path):
    path = tmp_path / "kernel_sequences_labeled.json"
    doc = _labeled_doc()
    del doc["phases"]["forward"]["unique_sequences"][0]["program"][0]["kernels"][0]["label"]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="must contain name, suggested_category, and label"):
        load_labeled_kernel_sequences(path)


def test_labeled_sequences_validate_embedded_operation(tmp_path):
    path = tmp_path / "kernel_sequences_labeled.json"
    doc = _labeled_doc()
    path.write_text(json.dumps(doc))
    normalized = load_labeled_kernel_sequences(path)
    label = normalized["phases"]["forward"]["unique_sequences"][0]["program"][0]["kernels"][0][
        "label"
    ]
    assert label["operation"] == "attention"
    assert label["simulated_slots"] == ["unified.attn"]


def test_labeled_sequences_accept_multiple_simulated_slots(tmp_path):
    path = tmp_path / "kernel_sequences_labeled.json"
    doc = _labeled_doc()
    label = doc["phases"]["forward"]["unique_sequences"][0]["program"][0]["kernels"][0]["label"]
    label["simulated_slots"] = ["unified.attn.main", "unified.attn.combine"]
    path.write_text(json.dumps(doc))

    normalized = load_labeled_kernel_sequences(path)

    assert normalized["phases"]["forward"]["unique_sequences"][0]["program"][0]["kernels"][0][
        "label"
    ]["simulated_slots"] == ["unified.attn.main", "unified.attn.combine"]


@pytest.mark.parametrize("slots", [[], ["unified.attn", "unified.attn"]])
def test_labeled_sequences_reject_invalid_simulated_slots(tmp_path, slots):
    path = tmp_path / "kernel_sequences_labeled.json"
    doc = _labeled_doc()
    label = doc["phases"]["forward"]["unique_sequences"][0]["program"][0]["kernels"][0]["label"]
    label["simulated_slots"] = slots
    path.write_text(json.dumps(doc))

    with pytest.raises(ValueError, match="simulated_slots"):
        load_labeled_kernel_sequences(path)


def test_labeled_sequences_accept_slot_shared_by_different_operations(tmp_path):
    # A single simulated slot may be the aggregate boundary for several
    # operations (e.g. a fused vs. unfused all-reduce sharing one tp_allreduce
    # slot). The loader accepts this; the analyzer resolves the per-iteration
    # owner from the operations actually present.
    path = tmp_path / "kernel_sequences_labeled.json"
    doc = _labeled_doc()
    sequence = doc["phases"]["forward"]["unique_sequences"][0]
    second = json.loads(json.dumps(sequence["program"][0]["kernels"][0]))
    second["name"] = "second_kernel"
    second["label"]["operation"] = "second_operation"
    sequence["program"][0]["kernels"].append(second)
    sequence["expanded_kernel_count"] = 2
    path.write_text(json.dumps(doc))

    normalized = load_labeled_kernel_sequences(path)

    kernels = normalized["phases"]["forward"]["unique_sequences"][0]["program"][0]["kernels"]
    assert [kernel["label"]["operation"] for kernel in kernels] == [
        "attention",
        "second_operation",
    ]
    assert all(kernel["label"]["simulated_slots"] == ["unified.attn"] for kernel in kernels)


def test_labeled_sequences_reject_inconsistent_slots_for_one_operation(tmp_path):
    path = tmp_path / "kernel_sequences_labeled.json"
    doc = _labeled_doc()
    sequence = doc["phases"]["forward"]["unique_sequences"][0]
    second = json.loads(json.dumps(sequence["program"][0]["kernels"][0]))
    second["name"] = "second_kernel"
    second["label"]["simulated_slots"] = ["unified.attn.other"]
    sequence["program"][0]["kernels"].append(second)
    sequence["expanded_kernel_count"] = 2
    path.write_text(json.dumps(doc))

    with pytest.raises(ValueError, match="inconsistent label metadata"):
        load_labeled_kernel_sequences(path)


def test_alignment_analyzer_and_renderer_end_to_end(tmp_path):
    analyzer = REPO_ROOT / "target" / "debug" / "analyze"
    if not analyzer.is_file():
        pytest.skip("build analyzer first")

    profile = tmp_path / "profile"
    predict = tmp_path / "timing_predict"
    analysis = tmp_path / "analysis"
    sim = tmp_path / "sim"
    (predict / "raw" / "cost_manifest").mkdir(parents=True)
    (predict / "raw" / "cost_log").mkdir(parents=True)
    (sim / "raw").mkdir(parents=True)
    profile.mkdir(exist_ok=True)
    analysis.mkdir()
    (sim / "raw" / "params.json").write_text(
        json.dumps({"pools": {"main": {"groups": [{"worker": {"gpu_time_multiplier": 1.25}}]}}})
    )

    manifest = {
        "sections": [
            {
                "section": "iter",
                # Deliberately pin an archived manifest shape here: the current
                # simulator writes `kernel_config`, while Analyzer must still be
                # able to re-analyze runs written with `config` and `backends`.
                "slots": [
                    {
                        "name": "unified.qkv",
                        "kind": "single_gemm",
                        "config": "n=6144 k=4096",
                        "backends": ["torch", "torch_linear"],
                    },
                    {
                        "name": "unified.attn",
                        "kind": "flashinfer_attn_decode",
                        "config": "",
                    },
                    {
                        "name": "unified.attn.combine",
                        "kind": "elementwise",
                        "config": "",
                    },
                    {"name": "unified.lm_head", "kind": "single_gemm", "config": ""},
                ],
                "nodes": [
                    {"Sum": {"children": {"start": 1, "end": 5}}},
                    {"Leaf": 0},
                    {"Leaf": 1},
                    {"Leaf": 2},
                    {"Leaf": 3},
                ],
                "node_labels": ["unified", None, None, None, None],
            }
        ]
    }
    (predict / "raw" / "cost_manifest" / "worker_predict_0.json").write_text(json.dumps(manifest))
    pq.write_table(
        pa.table(
            {
                "pool_tag": ["predict", "predict"],
                "worker_id": pa.array([0, 0], type=pa.uint16()),
                "iter_id": pa.array([0, 1], type=pa.uint64()),
                "section": ["iter", "iter"],
                "total_time_ms": [3.8, 4.2],
                "slot_time_ms": pa.array(
                    [[2.2, 0.8, 0.3, 0.5], [2.4, 0.9, 0.3, 0.6]],
                    type=pa.list_(pa.float32()),
                ),
            }
        ),
        predict / "raw" / "cost_log" / "worker_predict_0.parquet",
    )

    parsed = {
        "schema_version": 2,
        "kernel_names": {"1": "nvjet_qkv", "2": "flashinfer_decode", "3": "nvjet_lm_head"},
        "iteration_details": [
            _measured_iteration(34, 2_000_000, 1_000_000, 500_000, start_ns=0),
            _measured_iteration(35, 2_100_000, 1_100_000, 600_000, start_ns=10_000_000),
        ],
    }
    parsed_path = profile / "parsed.json"
    parsed_path.write_text(json.dumps(parsed))
    metrics_path = profile / "metrics.jsonl"
    metrics_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "schema_version": 2,
                    "input_adapter": "vllm_text",
                    "iteration_index": iteration,
                    "observed_start_monotonic_ns": start_ns,
                    "observed_end_monotonic_ns": start_ns + 3_000_000,
                    "observed_elapsed_ms": 3.0,
                    "prefill_tokens": 0,
                    "decode_requests": 1,
                    "decode_tokens_scheduled": 1,
                    "prefill_chunk_pairs": [],
                    "decode_kv_lens": [100 + offset],
                }
            )
            for offset, (iteration, start_ns) in enumerate(
                [(34, 1_000_000_000), (35, 1_010_000_000)]
            )
        )
    )
    case_map = predict / "timing_predict_case_map.json"
    case_map.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {"case_index": 0, "measured_iteration": 34, "stage": "decode"},
                    {"case_index": 1, "measured_iteration": 35, "stage": "decode"},
                ],
            }
        )
    )
    labeled = predict / "kernel_sequences_labeled.json"

    def mapped(name, category, operation, kernel_type, role, slots):
        return {
            "name": name,
            "suggested_category": category,
            "label": {
                "status": "mapped",
                "operation": operation,
                "type": kernel_type,
                "role": role,
                "simulated_slots": slots,
            },
        }

    labeled.write_text(
        json.dumps(
            _labeled_doc(
                {
                    "forward": {
                        "unique_sequences": [
                            {
                                "sequence_id": "sequence_forward",
                                "iterations": [34, 35],
                                "expanded_kernel_count": 2,
                                "program": [
                                    {
                                        "kernels": [
                                            mapped(
                                                "nvjet_qkv",
                                                "gemm_or_cutlass",
                                                "dense_gemm",
                                                "gemm",
                                                "qkv projection",
                                                ["unified.qkv"],
                                            ),
                                            mapped(
                                                "flashinfer_decode",
                                                "attention",
                                                "attention",
                                                "attention",
                                                "decode attention",
                                                ["unified.attn", "unified.attn.combine"],
                                            ),
                                        ]
                                    }
                                ],
                            }
                        ]
                    },
                    "postprocess": {
                        "unique_sequences": [
                            {
                                "sequence_id": "sequence_postprocess",
                                "iterations": [34, 35],
                                "expanded_kernel_count": 1,
                                "program": [
                                    {
                                        "kernels": [
                                            mapped(
                                                "nvjet_lm_head",
                                                "gemm_or_cutlass",
                                                "model.lm_head",
                                                "gemm",
                                                "final vocabulary projection",
                                                ["unified.lm_head"],
                                            )
                                        ]
                                    }
                                ],
                            }
                        ]
                    },
                }
            )
        )
    )

    replay = profile / "replay.jsonl"
    replay.write_text("\n".join([_replay_row("1", 1.0, 1.2), _replay_row("2", 1.0, 1.3)]))
    request_timings = profile / "request_timings.jsonl"
    request_timings.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "schema_version": 2,
                        "request_id": "vibesim_1",
                        "engine_core_ttft_ms": 30.0,
                        "engine_core_decode_ms": 180.0,
                        "num_output_tokens": 10,
                        "engine_core_tpot_ms": 20.0,
                    }
                ),
                json.dumps(
                    {
                        "schema_version": 2,
                        "request_id": "vibesim_2",
                        "engine_core_ttft_ms": 35.0,
                        "engine_core_decode_ms": 225.0,
                        "num_output_tokens": 10,
                        "engine_core_tpot_ms": 25.0,
                    }
                ),
            ]
        )
    )
    pq.write_table(
        pa.table(
            {
                "request_id": pa.array([1, 2], type=pa.uint32()),
                "completed": [True, True],
                "arrival_time_ms": [0.0, 0.0],
                "num_output_tokens": pa.array([10, 10], type=pa.uint32()),
                "ttft_ms": pa.array([40.0, 50.0], type=pa.float32()),
                "tpot_mean_ms": pa.array([18.0, 25.0], type=pa.float32()),
                "finish_decode_time_ms": pa.array([200.0, 300.0], type=pa.float32()),
            }
        ),
        sim / "raw" / "request_slo.parquet",
    )

    (profile / "profile_result.json").write_text(json.dumps({"log_dir": str(profile)}))
    # End-to-end fixture exercises all three alignment subjects in one analyzer
    # invocation, so it writes the union of both typed manifests' fields. Each
    # subject deserializes only its own type (kernel-align vs e2e-align) and
    # ignores the other's fields.
    (analysis / "alignment_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 8,
                "profile_log_dir": str(profile),
                "workload_profile_log_dir": str(profile),
                "simulation_log_dir": str(sim),
                "analysis_log_dir": str(analysis),
                "parsed_nsys": str(parsed_path),
                "metrics_jsonl": str(metrics_path),
                "replay_result": str(replay),
                "request_timings_result": str(request_timings),
                "predict_log_dir": str(predict),
                "timing_predict_case_map": str(case_map),
                "labeled_kernel_sequences": str(labeled),
                "throughput_bins": 4,
            }
        )
    )

    subprocess.run([str(analyzer), "alignment", str(analysis)], cwd=REPO_ROOT, check=True)
    subprocess.run(
        [
            str(REPO_ROOT / ".venv" / "bin" / "python"),
            str(REPO_ROOT / "analyzer" / "python"),
            "render",
            str(analysis),
        ],
        cwd=REPO_ROOT,
        check=True,
    )

    iteration_report = json.loads(
        (analysis / "reports" / "alignment_iteration_report.json").read_text()
    )
    iteration_payload = json.loads(
        (analysis / "payloads" / "alignment_iteration_series.json").read_text()
    )
    e2e_report = json.loads((analysis / "reports" / "alignment_e2e_report.json").read_text())
    assert iteration_report["mapping"]["coverage"]["measured_duration_fraction"] == 1.0
    assert len(iteration_report["kernels"]) == 3
    assert iteration_report["iterations"][0]["measured_ms"] == 3.5
    # The iteration pass self-computes the duty-cycle multiplier from measured
    # quantities alone: Σ measured_gpu_cycle_ms / Σ measured_ms over iterations
    # that have a next-iteration cycle. Only iteration 0 has a cycle here
    # (10.0 ms), so recommended == 10.0 / 3.5. The sim params multiplier (1.25)
    # is deliberately ignored.
    recommended = 10.0 / 3.5
    assert iteration_report["meta"]["recommended_gpu_time_multiplier"] == pytest.approx(recommended)
    assert iteration_report["iterations"][0]["measured_gpu_cycle_ms"] == 10.0
    assert iteration_report["iterations"][0]["simulated_gpu_cycle_ms"] == pytest.approx(
        3.8 * recommended
    )
    assert iteration_report["iterations"][1]["measured_gpu_cycle_ms"] is None
    # Per-iteration detail lives in a byte-range-addressed shard so a client can
    # read one iteration without parsing the rest; the payload carries the index.
    detail = iteration_payload["breakdown_detail"]
    shard = (analysis / "payloads" / detail["file"]).read_bytes()
    offset, length = detail["byte_ranges"][str(iteration_report["iterations"][0]["iteration_id"])]
    breakdown = json.loads(shard[offset : offset + length])
    assert "operations" not in breakdown
    assert [kernel["name"] for kernel in breakdown["measured_kernels"]] == [
        "nvjet_qkv",
        "flashinfer_decode",
        "nvjet_lm_head",
    ]
    assert [kernel["phase"] for kernel in breakdown["measured_kernels"]] == [
        "forward",
        "forward",
        "postprocess",
    ]
    assert [kernel["calls"] for kernel in breakdown["measured_kernels"]] == [1, 1, 1]
    assert [kernel["name"] for kernel in breakdown["simulated_kernels"]] == [
        "unified.qkv",
        "unified.attn",
        "unified.attn.combine",
        "unified.lm_head",
    ]
    assert [kernel["multiplicity"] for kernel in breakdown["simulated_kernels"]] == [1, 1, 1, 1]
    assert [phase["phase"] for phase in breakdown["phase_summary"]] == [
        "forward",
        "postprocess",
    ]
    assert [item["operation"] for item in breakdown["operation_summary"]] == [
        "attention",
        "dense_gemm",
        "model.lm_head",
    ]
    attention = next(
        item for item in breakdown["operation_summary"] if item["operation"] == "attention"
    )
    assert attention["measured_ms"] == 1.0
    assert attention["simulated_ms"] == pytest.approx(1.1)
    attention_mapping = next(
        item
        for item in iteration_report["mapping"]["operations"]
        if item["operation"] == "attention"
    )
    assert attention_mapping["simulated_slots"] == [
        "unified.attn",
        "unified.attn.combine",
    ]
    assert e2e_report["meta"]["request_id_audit"]["shared_ids"] == 2
    assert e2e_report["latency"]["client_ttft"]["measured_ms"]["mean"] == 40.0
    assert e2e_report["latency"]["server_ttft"]["measured_ms"]["mean"] == 32.5
    assert e2e_report["latency"]["server_tpot"]["measured_ms"]["mean"] == 22.5
    assert e2e_report["throughput"]["measured_client_completion_tps"] == pytest.approx(20 / 0.3)
    assert e2e_report["throughput"]["measured_server_gpu_span_ms"] == 13.8
    assert e2e_report["throughput"]["measured_server_gpu_span_tps"] == pytest.approx(20 / 0.0138)
    e2e_payload = json.loads((analysis / "payloads" / "alignment_e2e_series.json").read_text())
    assert e2e_payload["throughput_summary"] == e2e_report["throughput"]
    assert (analysis / "plots" / "alignment_iteration_overview.png").is_file()
    assert (analysis / "plots" / "alignment_iteration_gpu_cycle_overview.png").is_file()
    assert (analysis / "plots" / "iter_34_to_35" / "iter_34_breakdown.png").is_file()
    assert (analysis / "plots" / "alignment_e2e_completion_throughput.png").is_file()
    assert (analysis / "plots" / "alignment_client_ttft_cdf_comparison.png").is_file()
    assert (analysis / "plots" / "alignment_server_ttft_cdf_comparison.png").is_file()
    assert (analysis / "plots" / "alignment_server_tpot_cdf_comparison.png").is_file()


def _measured_iteration(
    iteration: int,
    gemm_ns: int,
    attention_ns: int,
    lm_head_ns: int,
    *,
    start_ns: int,
) -> dict:
    return {
        "iteration": iteration,
        "iteration_type": "decode",
        "ranges": [
            {
                "device_id": 0,
                "phase": "forward",
                "start_ns": start_ns,
                "end_ns": start_ns + gemm_ns + attention_ns,
                "kernel_busy_union_ns": gemm_ns + attention_ns,
                "kernels": [
                    {
                        "name_id": 1,
                        "category": "gemm_or_cutlass",
                        "start_ns": start_ns,
                        "end_ns": start_ns + gemm_ns,
                    },
                    {
                        "name_id": 2,
                        "category": "attention",
                        "start_ns": start_ns + gemm_ns,
                        "end_ns": start_ns + gemm_ns + attention_ns,
                    },
                ],
            },
            {
                "device_id": 0,
                "phase": "postprocess",
                "start_ns": start_ns + gemm_ns + attention_ns,
                "end_ns": start_ns + gemm_ns + attention_ns + lm_head_ns,
                "kernels": [
                    {
                        "name_id": 3,
                        "category": "gemm_or_cutlass",
                        "start_ns": start_ns + gemm_ns + attention_ns,
                        "end_ns": start_ns + gemm_ns + attention_ns + lm_head_ns,
                    }
                ],
            },
        ],
    }


def _replay_row(request_id: str, submit: float, complete: float) -> str:
    return json.dumps(
        {
            "source": {"type": "independent_request", "data": {"id": request_id}},
            "outcome": {
                "status": "SUCCESS",
                "submit_timestamp": submit,
                "post_timestamp": submit,
                "complete_timestamp": complete,
                "output_len_actual": 10,
                "first_token_ms": 40.0,
                "total_duration_ms": (complete - submit) * 1000.0,
            },
        }
    )
