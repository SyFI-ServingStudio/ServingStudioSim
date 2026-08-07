"""Focused contract tests for alignment's indexed NSYS markers."""

import json
import sqlite3
from pathlib import Path

import pytest

from alignment.nsys.parse import (
    KernelEvent,
    RangeStats,
    Worker,
    _parse_label,
    build_host_timeline,
    aggregate_metrics_by_iteration,
    build_iteration_details,
    build_kernel_name_index,
    build_parser,
    ensure_query_indexes,
    iteration_kind,
    load_host_threads,
    load_ranges,
    owning_global_pid,
    parsed_window_ns,
    load_metrics,
    resolve_dp_rank_by_device,
)
from alignment.nsys.sequence import build_kernel_sequences, expand_program


def test_parse_label_accepts_only_indexed_iteration_markers():
    assert _parse_label("vllm_iteration(34): forward") == (34, "forward")
    assert _parse_label("sglang_iteration(9): sample") == (9, "sample")


def test_parse_defaults_to_all_indexed_phases():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--sqlite",
            "trace.sqlite",
            "--iteration-start",
            "1",
            "--iteration-end",
            "2",
        ]
    )

    assert args.range_mode == "phases"


def test_parse_persists_correlation_lookup_indexes_idempotently():
    con = sqlite3.connect(":memory:")
    con.executescript(
        """
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            globalTid INTEGER,
            start INTEGER,
            correlationId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            globalPid INTEGER,
            correlationId INTEGER,
            start INTEGER
        );
        """
    )

    ensure_query_indexes(con)
    ensure_query_indexes(con)

    runtime_indexes = con.execute(
        "PRAGMA index_list('CUPTI_ACTIVITY_KIND_RUNTIME')"
    ).fetchall()
    kernel_indexes = con.execute(
        "PRAGMA index_list('CUPTI_ACTIVITY_KIND_KERNEL')"
    ).fetchall()
    assert [row[1] for row in runtime_indexes] == ["vibesim_runtime_gtid_start_idx"]
    assert [row[1] for row in kernel_indexes] == ["vibesim_kernel_gpid_corr_start_idx"]
    assert not con.in_transaction


@pytest.mark.parametrize(
    "label",
    [
        "gpu_model_runner: forward",
        "execute_context_0(0)_generation_64(64)",
        "vllm_iteration(unknown): forward",
    ],
)
def test_parse_label_rejects_unindexed_markers(label):
    assert _parse_label(label) is None


def test_load_ranges_resolves_indexed_marker_from_string_ids():
    con = sqlite3.connect(":memory:")
    con.executescript(
        """
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE NVTX_EVENTS (
            start INTEGER,
            end INTEGER,
            globalTid INTEGER,
            text TEXT,
            textId INTEGER
        );
        INSERT INTO StringIds VALUES (1, 'vllm_iteration(34): forward');
        INSERT INTO StringIds VALUES (2, 'gpu_model_runner: forward');
        INSERT INTO NVTX_EVENTS VALUES (100, 200, 30, NULL, 1);
        INSERT INTO NVTX_EVENTS VALUES (100, 200, 30, NULL, 2);
        """
    )
    worker = Worker(global_pid=10, pid=20, name="VLLM::Worker", device_id=0)

    ranges = load_ranges(
        con,
        {30: worker},
        {34: {"prefill_tokens": 0, "decode_requests": 1}},
        34,
        34,
        "forward",
    )

    assert [(item.iteration, item.phase, item.stage) for item in ranges] == [
        (34, "forward", "decode")
    ]


def test_iteration_details_preserve_complete_kernel_launch_record():
    worker = Worker(global_pid=10, pid=20, name="VLLM::Worker", device_id=0)
    event = KernelEvent(
        start=120,
        end=180,
        name="full_demangled_kernel_name",
        category="attention",
        stream_id=3,
        correlation_id=42,
    )
    item = RangeStats(
        iteration=34,
        phase="forward",
        stage="mixed",
        worker=worker,
        start=100,
        end=200,
        intervals=[(event.start, event.end)],
        kernel_count=1,
        sum_ns=event.duration_ns,
        kernel_events=[event],
    )

    name_ids, kernel_names = build_kernel_name_index([item])
    details = build_iteration_details(
        [item],
        {34: {"iteration_index": 34, "prefill_tokens": 512, "decode_tokens": 64}},
        name_ids,
    )

    assert kernel_names == {1: "full_demangled_kernel_name"}
    assert details[0]["iteration_type"] == "prefill"
    assert details[0]["metrics"]["decode_tokens"] == 64
    kernel = details[0]["ranges"][0]["kernels"][0]
    assert kernel == {
        "ordinal": 1,
        "name_id": 1,
        "category": "attention",
        "start_ns": 120,
        "end_ns": 180,
        "stream_id": 3,
        "correlation_id": 42,
    }


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ({"prefill_tokens": 8, "decode_requests": 0}, "prefill"),
        ({"prefill_tokens": 0, "decode_requests": 8}, "decode"),
        ({"prefill_tokens": 8, "decode_requests": 8}, "mixed"),
    ],
)
def test_iteration_kind_distinguishes_prefill_decode_and_mixed(metric, expected):
    assert iteration_kind(metric) == expected


def test_full_sequence_folds_exact_repetition_losslessly():
    details = [
        {
            "iteration": 3,
            "iteration_type": "decode",
            "ranges": [
                {
                    "phase": "forward",
                    "kernels": [
                        {"name_id": name_id, "category": "other"} for name_id in (1, 2, 1, 2, 1, 2)
                    ],
                }
            ],
        }
    ]

    catalog = build_kernel_sequences(details, {1: "same_impl", 2: "other_impl"})["forward"]
    sequence = catalog["unique_sequences"][0]
    assert sequence["expanded_kernel_count"] == 6
    assert sequence["program"][0]["repeat"]["count"] == 3
    expanded = expand_program(sequence["program"])
    assert [row["name"] for row in expanded] == [
        "same_impl",
        "other_impl",
        "same_impl",
        "other_impl",
        "same_impl",
        "other_impl",
    ]
    assert sequence["occurrences"] == [{"device_id": 0, "iterations": [3]}]
    assert "iteration_assignments" not in catalog


def test_sequence_unique_ignores_iteration_semantic_labels():
    common_ranges = [
        {
            "phase": "forward",
            "kernels": [{"name_id": 1, "category": "other"}],
        }
    ]
    details = [
        {"iteration": 1, "iteration_type": "prefill", "ranges": common_ranges},
        {"iteration": 2, "iteration_type": "decode", "ranges": common_ranges},
    ]

    catalog = build_kernel_sequences(details, {1: "kernel"})["forward"]

    assert len(catalog["unique_sequences"]) == 1
    assert catalog["unique_sequences"][0]["occurrences"] == [
        {"device_id": 0, "iterations": [1, 2]}
    ]


def test_fold_prefers_layer_aligned_repeat_with_final_suffix():
    name_ids = (1, 2, 3, 4, 2, 5, 4, 2, 5, 4, 2, 5, 4)
    details = [
        {
            "iteration": 1,
            "iteration_type": "decode",
            "ranges": [
                {
                    "phase": "forward",
                    "kernels": [{"name_id": name_id, "category": "other"} for name_id in name_ids],
                }
            ],
        }
    ]

    sequence = build_kernel_sequences(
        details,
        {1: "first_a", 2: "shared", 3: "first_b", 4: "steady_a", 5: "steady_b"},
    )["forward"]["unique_sequences"][0]

    assert [
        len(sequence["program"][0]["kernels"]),
        sequence["program"][1]["repeat"]["count"],
        len(sequence["program"][2]["kernels"]),
    ] == [3, 3, 1]
    assert [kernel["name"] for kernel in expand_program(sequence["program"])] == [
        "first_a",
        "shared",
        "first_b",
        "steady_a",
        "shared",
        "steady_b",
        "steady_a",
        "shared",
        "steady_b",
        "steady_a",
        "shared",
        "steady_b",
        "steady_a",
    ]


def test_symmetric_ranks_collapse_to_one_labeling_decision():
    """Identical ranks share a sequence id, so they cost one label, not N."""
    rank_kernels = [
        {"name_id": 1, "category": "other"},
        {"name_id": 2, "category": "nccl_collective"},
    ]
    details = [
        {
            "iteration": 7,
            "iteration_type": "decode",
            "ranges": [
                {"device_id": 0, "phase": "forward", "kernels": rank_kernels},
                {"device_id": 1, "phase": "forward", "kernels": rank_kernels},
            ],
        }
    ]

    catalog = build_kernel_sequences(details, {1: "gemm", 2: "nccl"})["forward"]

    assert len(catalog["unique_sequences"]) == 1
    sequence = catalog["unique_sequences"][0]
    assert sequence["expanded_kernel_count"] == 2
    assert [row["name"] for row in expand_program(sequence["program"])] == [
        "gemm",
        "nccl",
    ]
    assert sequence["occurrences"] == [
        {"device_id": 0, "iterations": [7]},
        {"device_id": 1, "iterations": [7]},
    ]


def test_asymmetric_ranks_each_keep_their_own_sequence():
    """Data-parallel ranks schedule independent batches, so divergence is normal.

    Each device keeps the sequence it actually ran; nothing is merged across
    devices and nothing is dropped in favour of a representative.
    """
    details = [
        {
            "iteration": 7,
            "iteration_type": "decode",
            "ranges": [
                {
                    "device_id": 0,
                    "phase": "forward",
                    "kernels": [{"name_id": 1, "category": "other"}],
                },
                {
                    "device_id": 1,
                    "phase": "forward",
                    "kernels": [{"name_id": 2, "category": "other"}],
                },
            ],
        }
    ]

    catalog = build_kernel_sequences(details, {1: "rank0", 2: "rank1"})["forward"]

    sequences = catalog["unique_sequences"]
    assert len(sequences) == 2
    assert {
        expand_program(sequence["program"])[0]["name"]: sequence["occurrences"]
        for sequence in sequences
    } == {
        "rank0": [{"device_id": 0, "iterations": [7]}],
        "rank1": [{"device_id": 1, "iterations": [7]}],
    }




def test_idle_is_measured_against_the_kernel_span_not_the_nvtx_range():
    """Under CUDA graphs an NVTX range can close before its own kernels finish.

    The old definition was `nvtx_window_ms - busy_ms`, so in exactly that case
    the `max(0.0, ...)` clamped a real stall to zero. Idle is now the gap inside
    the correlated kernel span, which stays honest whatever the host did.
    """
    worker = Worker(global_pid=10, pid=20, name="VLLM::Worker", device_id=0)
    item = RangeStats(
        iteration=7,
        phase="forward",
        stage="decode",
        worker=worker,
        # The range opens at 100 and closes at 300, but the last kernel does not
        # end until 900 — the graph replay outlives its own marker.
        start=100,
        end=300,
        intervals=[(200, 400), (700, 900)],
    )

    assert item.nvtx_window_ms == pytest.approx(200 / 1e6)
    assert item.kernel_span_ms == pytest.approx(700 / 1e6)
    assert item.busy_ms == pytest.approx(400 / 1e6)
    # The 300 ns hole between the two kernels, which the old formula reported as
    # zero because busy (400) already exceeded the 200 ns window.
    assert item.idle_ms == pytest.approx(300 / 1e6)


def test_a_range_with_no_kernels_has_no_span_and_no_idle():
    worker = Worker(global_pid=10, pid=20, name="VLLM::Worker", device_id=0)
    item = RangeStats(
        iteration=7, phase="bookkeep", stage="decode", worker=worker, start=100, end=300
    )

    assert item.kernel_span_ms == 0.0
    assert item.idle_ms == 0.0


# ---------------------------------------------------------------------------
# Host (CPU) sidecar
# ---------------------------------------------------------------------------

# nsys packs a globalTid as `globalPid | thread_id` with the process id in the
# high bits, so these fixtures keep the low 24 bits of every globalPid clear.
WORKER_GLOBAL_PID = 1 << 24
SCHEDULER_GLOBAL_PID = 2 << 24
WORKER_MAIN_TID = WORKER_GLOBAL_PID + 7
WORKER_HELPER_TID = WORKER_GLOBAL_PID + 8
SCHEDULER_TID = SCHEDULER_GLOBAL_PID + 9


def host_capture() -> sqlite3.Connection:
    """A minimal capture with all three kinds of host thread on it."""
    con = sqlite3.connect(":memory:")
    con.executescript(
        f"""
        CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT);
        CREATE TABLE PROCESSES (globalPid INTEGER, pid INTEGER, name TEXT);
        CREATE TABLE NVTX_EVENTS (
            start INTEGER, end INTEGER, globalTid INTEGER, text TEXT, textId INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (
            globalPid INTEGER, deviceId INTEGER, correlationId INTEGER, start INTEGER
        );
        CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (
            globalTid INTEGER, start INTEGER, end INTEGER,
            nameId INTEGER, correlationId INTEGER
        );

        INSERT INTO StringIds VALUES (1, 'cudaLaunchKernel_v7000');
        INSERT INTO StringIds VALUES (2, 'cudaStreamSynchronize_v3020');

        INSERT INTO PROCESSES VALUES ({WORKER_GLOBAL_PID}, 7, 'VLLM::Worker0');
        INSERT INTO PROCESSES VALUES ({SCHEDULER_GLOBAL_PID}, 9, 'VLLM::EngineCor');

        -- Only the worker process ever put a kernel on a device.
        INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES ({WORKER_GLOBAL_PID}, 3, 1, 1000);

        INSERT INTO NVTX_EVENTS VALUES (1000, 2000, {WORKER_MAIN_TID}, 'vllm_iteration(7): forward', NULL);
        INSERT INTO NVTX_EVENTS VALUES (1100, 1500, {WORKER_MAIN_TID}, 'execute_context_0', NULL);
        INSERT INTO NVTX_EVENTS VALUES (900, 2100, {SCHEDULER_TID}, 'schedule', NULL);
        -- A mark the capture never closed.
        INSERT INTO NVTX_EVENTS VALUES (1200, NULL, {WORKER_MAIN_TID}, 'step', NULL);
        -- Entirely before the window.
        INSERT INTO NVTX_EVENTS VALUES (10, 20, {WORKER_MAIN_TID}, 'warmup', NULL);

        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES ({WORKER_MAIN_TID}, 1100, 1180, 1, 1);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES ({WORKER_MAIN_TID}, 1200, 1900, 2, 2);
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES ({WORKER_HELPER_TID}, 1300, 1310, 1, 3);
        -- Entirely before the window.
        INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES ({WORKER_MAIN_TID}, 10, 20, 1, 4);
        """
    )
    return con


def test_host_threads_separate_worker_main_helper_and_scheduler():
    """The scheduler thread is the reason the sidecar exists.

    It owns no device, so nothing on the kernel side can see its host time —
    and that is exactly where an iteration's unexplained wall clock hides.
    """
    threads = load_host_threads(host_capture())

    assert [(item.global_tid, item.role, item.device_id) for item in threads] == [
        (SCHEDULER_TID, "scheduler thread", None),
        (WORKER_MAIN_TID, "worker main thread", 3),
        (WORKER_HELPER_TID, "worker helper thread", 3),
    ]
    assert [item.process for item in threads] == [
        "VLLM::EngineCor",
        "VLLM::Worker0",
        "VLLM::Worker0",
    ]


def test_owning_global_pid_inverts_the_nsys_thread_packing():
    assert owning_global_pid(WORKER_HELPER_TID) == WORKER_GLOBAL_PID
    assert owning_global_pid(SCHEDULER_TID) == SCHEDULER_GLOBAL_PID


def test_host_timeline_keeps_the_runtime_call_bounds_attribution_throws_away():
    """Kernel attribution reads the same rows for `correlationId` alone.

    Keeping `(start, end, nameId)` is the whole of tier 2: it turns "a kernel
    belongs to this phase" into "the host spent 700 ns blocked in this call".
    """
    host = build_host_timeline(host_capture(), Path("trace.sqlite"), 1000, 2000)

    threads = [item["global_tid"] for item in host["threads"]]
    calls = [
        (threads[index], start, end, host["strings"][name], correlation_id)
        for index, start, end, name, correlation_id in host["api_calls"]
    ]
    assert calls == [
        (WORKER_MAIN_TID, 1100, 1180, "cudaLaunchKernel_v7000", 1),
        (WORKER_MAIN_TID, 1200, 1900, "cudaStreamSynchronize_v3020", 2),
        (WORKER_HELPER_TID, 1300, 1310, "cudaLaunchKernel_v7000", 3),
    ]


def test_host_timeline_keeps_every_nvtx_thread_not_just_iteration_markers():
    """`load_ranges` keeps only `vllm_iteration(N)` on device-owning workers.

    The host lane needs the nested ranges and the scheduler's marks too, or the
    time between phases has no name at all.
    """
    host = build_host_timeline(host_capture(), Path("trace.sqlite"), 1000, 2000)

    labels = sorted(host["strings"][name] for _index, _start, _end, name in host["nvtx_ranges"])
    assert labels == ["execute_context_0", "schedule", "vllm_iteration(7): forward"]


def test_host_timeline_counts_unclosed_marks_rather_than_dropping_them():
    host = build_host_timeline(host_capture(), Path("trace.sqlite"), 1000, 2000)

    assert host["unclosed_nvtx_marks"] == 1


def test_host_timeline_excludes_events_outside_the_parsed_window():
    """The sidecar accompanies one parse, so warmup and shutdown are not in it."""
    host = build_host_timeline(host_capture(), Path("trace.sqlite"), 1000, 2000)

    assert all(end >= 1000 for _index, _start, end, _name in host["nvtx_ranges"])
    assert all(
        start < 2000
        for _index, start, _end, _name, _correlation_id in host["api_calls"]
    )
    assert "warmup" not in host["strings"]


def test_parsed_window_spans_every_serialized_range():
    parsed = {
        "iteration_details": [
            {"ranges": [{"start_ns": 500, "end_ns": 900}, {"start_ns": 400, "end_ns": 700}]},
            {"ranges": [{"start_ns": 950, "end_ns": 1400}]},
        ]
    }

    assert parsed_window_ns(parsed) == (400, 1400)


def test_parsed_window_of_an_empty_parse_is_empty():
    assert parsed_window_ns({"iteration_details": []}) == (0, 0)

def _metrics_line(dp_rank: int, iteration: int, prefill_tokens: int, decode_kv_lens: list[int]):
    return json.dumps(
        {
            "schema_version": 1,
            "input_adapter": "vllm_text",
            "iteration_index": iteration,
            "dp_rank": dp_rank,
            "prefill_tokens": prefill_tokens,
            "decode_requests": len(decode_kv_lens),
            "decode_tokens_scheduled": len(decode_kv_lens),
            "prefill_chunk_pairs": [[0, prefill_tokens]] if prefill_tokens else [],
            "decode_kv_lens": decode_kv_lens,
        }
    )


def test_metrics_are_keyed_by_rank_and_iteration(tmp_path):
    """Every DP rank reuses the same iteration indices; keying on the index alone
    would keep only the last rank's batch shape."""
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                _metrics_line(0, 4, prefill_tokens=8, decode_kv_lens=[10]),
                _metrics_line(1, 4, prefill_tokens=0, decode_kv_lens=[20, 30]),
            ]
        )
        + "\n"
    )

    metrics = load_metrics(path)

    assert set(metrics) == {(0, 4), (1, 4)}
    assert metrics[(0, 4)]["prefill_tokens"] == 8
    assert metrics[(1, 4)]["decode_kv_lens"] == [20, 30]


def test_metrics_reject_a_repeated_rank_iteration_pair(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        _metrics_line(0, 4, 8, [10]) + "\n" + _metrics_line(0, 4, 8, [10]) + "\n"
    )

    with pytest.raises(ValueError, match="duplicate metrics record"):
        load_metrics(path)


def test_iteration_aggregate_is_the_union_of_the_ranks_batches():
    """DP ranks step in lockstep behind the EP collectives, so the replica's
    workload for one step is the union of its ranks' local batches."""
    rank_metrics = {
        (0, 4): json.loads(_metrics_line(0, 4, prefill_tokens=8, decode_kv_lens=[10])),
        (1, 4): json.loads(_metrics_line(1, 4, prefill_tokens=0, decode_kv_lens=[20, 30])),
    }

    aggregate = aggregate_metrics_by_iteration(rank_metrics)[4]

    assert aggregate["prefill_tokens"] == 8
    assert aggregate["decode_requests"] == 3
    assert aggregate["prefill_chunk_pairs"] == [[0, 8]]
    assert aggregate["decode_kv_lens"] == [10, 20, 30]
    assert aggregate["dp_ranks"] == [0, 1]
    # One rank prefilling while another only decodes is a mixed replica step.
    assert iteration_kind(aggregate) == "mixed"


def test_dp_rank_by_device_folds_global_ranks_by_the_tp_degree():
    workers = {
        1: Worker(global_pid=10, pid=100, name="VLLM::Worker", device_id=0),
        2: Worker(global_pid=20, pid=101, name="VLLM::Worker", device_id=1),
        3: Worker(global_pid=30, pid=102, name="VLLM::Worker", device_id=2),
        4: Worker(global_pid=40, pid=103, name="VLLM::Worker", device_id=3),
    }
    worker_ranks = {100: 0, 101: 1, 102: 2, 103: 3}

    assert resolve_dp_rank_by_device(workers, worker_ranks, tp_size=1) == {0: 0, 1: 1, 2: 2, 3: 3}
    assert resolve_dp_rank_by_device(workers, worker_ranks, tp_size=2) == {0: 0, 1: 0, 2: 1, 3: 1}
    # No banner at all is the historical single-process capture.
    assert resolve_dp_rank_by_device(workers, {}, tp_size=1) == {0: 0, 1: 0, 2: 0, 3: 0}


def test_dp_rank_by_device_rejects_a_device_with_no_rank_banner():
    workers = {1: Worker(global_pid=10, pid=100, name="VLLM::Worker", device_id=0)}

    with pytest.raises(ValueError, match="no rank banner"):
        resolve_dp_rank_by_device(workers, {999: 0}, tp_size=1)


# ---- wall-clock step alignment --------------------------------------------
#
# A data-parallel rank numbers its OWN scheduled steps. When one rank runs a
# prefill chunk alone its peers run dummy batches that join the expert-parallel
# collectives but are not scheduled iterations, so they never advance their
# counters and the ranks stay offset for the rest of the capture. Grouping by
# `iteration_index` then reduces kernels from steps that never coexisted.


def _window(device_id: int, iteration: int, start: int, end: int, phase: str = "forward"):
    return RangeStats(
        iteration=iteration,
        phase=phase,
        stage="decode",
        worker=Worker(global_pid=device_id, pid=100 + device_id, name="W", device_id=device_id),
        start=start,
        end=end,
        intervals=[(start, end)],
        kernel_count=1,
        sum_ns=end - start,
    )


def test_steps_group_by_measured_time_not_by_iteration_index():
    from alignment.nsys.parse import align_ranges_into_steps

    ranges = [
        _window(0, 5, 0, 10),
        _window(0, 6, 10, 20),
        _window(0, 7, 20, 30),
        # Device 1 is two steps behind on the counter but runs at the same time.
        _window(1, 3, 1, 11),
        _window(1, 4, 11, 21),
        _window(1, 5, 21, 31),
    ]
    steps, unpaired = align_ranges_into_steps(ranges)

    assert [step.iteration for step in steps] == [5, 6, 7]
    assert [step.index_by_device for step in steps] == [{0: 5, 1: 3}, {0: 6, 1: 4}, {0: 7, 1: 5}]
    assert unpaired == {}


def test_a_step_only_one_rank_ran_keeps_that_rank_and_names_the_absent_ones():
    from alignment.nsys.parse import align_ranges_into_steps

    ranges = [
        # Device 0 alone for the first two steps — the prefill chunks.
        _window(0, 8, 0, 100),
        _window(0, 9, 100, 200),
        _window(0, 10, 200, 210),
        _window(1, 8, 201, 211),
    ]
    steps, _ = align_ranges_into_steps(ranges)

    assert [step.index_by_device for step in steps] == [{0: 8}, {0: 9}, {0: 10, 1: 8}]

    name_ids, _ = build_kernel_name_index(ranges)
    details = build_iteration_details(ranges, {}, name_ids, {}, {0: 0, 1: 1})
    assert [detail["devices_absent"] for detail in details] == [[1], [1], []]
    assert details[0]["iteration_index_by_device"] == {"0": 8}
    assert details[2]["iteration_index_by_device"] == {"0": 10, "1": 8}


def test_each_range_carries_its_own_index_and_its_own_ranks_metrics():
    from alignment.nsys.parse import align_ranges_into_steps  # noqa: F401

    ranges = [_window(0, 5, 0, 10), _window(1, 3, 1, 11)]
    rank_metrics = {
        (0, 5): {"input_adapter": "vllm_text", "prefill_tokens": 8189, "decode_requests": 3},
        (1, 3): {"input_adapter": "vllm_text", "prefill_tokens": 0, "decode_requests": 7},
        # The peer's row under the STEP's id belongs to a different step and must
        # not be picked up.
        (1, 5): {"input_adapter": "vllm_text", "prefill_tokens": 0, "decode_requests": 99},
    }
    name_ids, _ = build_kernel_name_index(ranges)
    (detail,) = build_iteration_details(ranges, {}, name_ids, rank_metrics, {0: 0, 1: 1})

    assert [(row["device_id"], row["iteration_index"]) for row in detail["ranges"]] == [
        (0, 5),
        (1, 3),
    ]
    assert detail["ranges"][1]["metrics"]["decode_requests"] == 7
    assert detail["metrics"]["prefill_tokens"] == 8189
    assert detail["metrics"]["decode_requests"] == 10
    assert detail["iteration_type"] == "mixed"


def test_a_peer_step_with_no_reference_counterpart_is_counted_not_dropped():
    from alignment.nsys.parse import align_ranges_into_steps

    ranges = [
        _window(0, 5, 100, 110),
        # Device 1 ran a step before the reference's window opens at all.
        _window(1, 1, 0, 10),
        _window(1, 2, 101, 111),
    ]
    steps, unpaired = align_ranges_into_steps(ranges)

    assert [step.index_by_device for step in steps] == [{0: 5, 1: 2}]
    assert unpaired == {1: 1}


def test_the_window_follows_the_reference_rank_and_peers_join_by_time():
    """A peer numbering the same step lower must not be filtered out of it.

    In the GLM-5.2 DP8 capture the ranks that step together at t=15.67 s call it
    iteration 11, 7 and 6. A numeric window applied to every rank drops the peers
    from the first steps of the window and leaves one rank apparently running
    alone.
    """
    from alignment.nsys.parse import window_rows_by_reference_rank

    reference = Worker(global_pid=1, pid=1, name="W", device_id=0)
    peer = Worker(global_pid=2, pid=2, name="W", device_id=1)
    rows = [
        [7, "forward", reference, 0, 10],
        [8, "forward", reference, 10, 20],
        [9, "forward", reference, 20, 30],
        [3, "forward", peer, 1, 11],  # same step as the reference's 7 — before the window
        [4, "forward", peer, 11, 21],  # same step as the reference's 8
        [5, "forward", peer, 21, 31],  # same step as the reference's 9
    ]
    kept = window_rows_by_reference_rank(rows, 8, 9)

    assert sorted((row[0], row[2].device_id) for row in kept) == [(4, 1), (5, 1), (8, 0), (9, 0)]
