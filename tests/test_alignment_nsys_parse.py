"""Focused contract tests for alignment's indexed NSYS markers."""

import sqlite3

import pytest

from alignment.nsys.parse import (
    KernelEvent,
    RangeStats,
    Worker,
    _parse_label,
    build_iteration_details,
    build_kernel_name_index,
    build_parser,
    iteration_kind,
    load_ranges,
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
