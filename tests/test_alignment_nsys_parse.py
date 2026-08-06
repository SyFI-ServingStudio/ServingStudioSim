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
    build_iteration_details,
    build_kernel_name_index,
    build_parser,
    ensure_query_indexes,
    iteration_kind,
    load_host_threads,
    load_ranges,
    owning_global_pid,
    parsed_window_ns,
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
    assert sequence["iterations"] == [3]
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
    assert catalog["unique_sequences"][0]["iterations"] == [1, 2]


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


def test_tp_sequence_inventory_keeps_one_validated_rank():
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

    sequence = catalog["unique_sequences"][0]
    assert sequence["expanded_kernel_count"] == 2
    assert [row["name"] for row in expand_program(sequence["program"])] == [
        "gemm",
        "nccl",
    ]


def test_tp_sequence_inventory_rejects_asymmetric_ranks():
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

    with pytest.raises(ValueError, match="not symmetric"):
        build_kernel_sequences(details, {1: "rank0", 2: "rank1"})


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
