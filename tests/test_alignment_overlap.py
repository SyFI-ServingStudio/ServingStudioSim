from __future__ import annotations

import pytest

from alignment.nsys.overlap import build_overlap_diagnostics


def _kernel(
    ordinal: int,
    name_id: int,
    start_ns: int,
    end_ns: int,
    stream_id: int,
    category: str = "other",
) -> dict:
    return {
        "ordinal": ordinal,
        "name_id": name_id,
        "category": category,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "stream_id": stream_id,
        "correlation_id": ordinal,
        "track_index": stream_id,
    }


def _parsed(details: list[dict]) -> dict:
    return {
        "schema_version": 5,
        "sqlite": "/capture.sqlite",
        "iteration_start": 1,
        "iteration_end": 99,
        "range_mode": "phases",
        "tp_size": 4,
        "kernel_names": {1: "producer_a", 2: "consumer_b", 3: "side_c"},
        "iteration_details": details,
    }


def _detail(
    iteration: int,
    stage: str,
    reference_device_id: int,
    device_id: int,
    kernels: list[dict],
    phase: str = "forward",
) -> dict:
    return {
        "iteration": iteration,
        "stage": stage,
        "reference_device_id": reference_device_id,
        "ranges": [
            {
                "device_id": device_id,
                "worker": {"global_pid": 1000 + device_id},
                "phase": phase,
                "kernels": kernels,
            }
        ],
    }


def test_decomposes_pdl_and_multistream_overlap_without_double_counting() -> None:
    parsed = _parsed(
        [
            _detail(
                5,
                "decode",
                0,
                0,
                [
                    _kernel(1, 1, 0, 10, 7),
                    _kernel(2, 2, 4, 12, 7),
                    _kernel(3, 3, 8, 15, 9),
                ],
            )
        ]
    )

    report = build_overlap_diagnostics(parsed)

    assert report["source"] == {
        "parsed_schema_version": 5,
        "sqlite": "/capture.sqlite",
        "iteration_start": 1,
        "iteration_end": 99,
        "range_mode": "phases",
        "tp_size": 4,
    }
    assert report["overall"] == {
        "kernel_launches": 3,
        "raw_kernel_ns": 25,
        "per_stream_busy_union_ns": 19,
        "busy_union_ns": 15,
        "pdl_same_stream_trace_reduction_ns": 6,
        "multistream_trace_reduction_ns": 4,
        "invariant_residual_ns": 0,
        "iteration_device_rows": 1,
        "busy_union_pct_of_raw": 60.0,
        "pdl_same_stream_pct_of_raw": 24.0,
        "multistream_pct_of_raw": 16.0,
    }
    assert report["checks"] == {
        "raw_equals_busy_plus_pdl_plus_multistream": True,
        "pdl_pairs_sum_to_pdl_reduction": True,
        "pdl_pair_sum_ns": 6,
        "pdl_pair_rows_total": 1,
        "pdl_pair_rows_returned": 1,
        "returned_pdl_pair_sum_ns": 6,
        "omitted_pdl_pair_sum_ns": 0,
    }
    assert report["pdl_kernel_pairs"] == [
        {
            "producer_kernel": "producer_a",
            "consumer_kernel": "consumer_b",
            "producer_category": "other",
            "consumer_category": "other",
            "trace_reduction_ns": 6,
            "occurrences": 1,
            "mean_trace_reduction_ns": 6.0,
            "pct_of_pdl_trace_reduction": 100.0,
            "by_stage_ns": {"decode": 6},
        }
    ]


def test_pdl_pair_segments_telescope_through_triple_overlap() -> None:
    parsed = _parsed(
        [
            _detail(
                9,
                "decode",
                0,
                0,
                [
                    _kernel(1, 1, 0, 10, 7),
                    _kernel(2, 2, 2, 8, 7),
                    _kernel(3, 3, 4, 12, 7),
                ],
            )
        ]
    )

    report = build_overlap_diagnostics(parsed, top_pairs=None)
    reductions = {
        (row["producer_kernel"], row["consumer_kernel"]): row["trace_reduction_ns"]
        for row in report["pdl_kernel_pairs"]
    }

    assert report["overall"]["raw_kernel_ns"] == 24
    assert report["overall"]["busy_union_ns"] == 12
    assert report["overall"]["pdl_same_stream_trace_reduction_ns"] == 12
    assert report["overall"]["multistream_trace_reduction_ns"] == 0
    assert reductions == {
        ("producer_a", "consumer_b"): 6,
        ("consumer_b", "side_c"): 4,
        ("producer_a", "side_c"): 2,
    }

    truncated = build_overlap_diagnostics(parsed, top_pairs=1)
    assert len(truncated["pdl_kernel_pairs"]) == 1
    assert truncated["checks"]["pdl_pair_rows_total"] == 3
    assert truncated["checks"]["pdl_pair_rows_returned"] == 1
    assert truncated["checks"]["returned_pdl_pair_sum_ns"] == 6
    assert truncated["checks"]["omitted_pdl_pair_sum_ns"] == 6


def test_exact_start_tie_attributes_the_narrower_interval_to_the_wider_one() -> None:
    parsed = _parsed(
        [
            _detail(
                4,
                "decode",
                0,
                0,
                [
                    _kernel(1, 1, 0, 10, 7),
                    _kernel(2, 2, 0, 6, 7),
                ],
            )
        ]
    )

    report = build_overlap_diagnostics(parsed)

    assert report["overall"]["pdl_same_stream_trace_reduction_ns"] == 6
    assert report["pdl_kernel_pairs"][0]["producer_kernel"] == "producer_a"
    assert report["pdl_kernel_pairs"][0]["consumer_kernel"] == "consumer_b"
    assert report["pdl_kernel_pairs"][0]["trace_reduction_ns"] == 6


def test_equal_stream_ids_in_different_processes_are_not_pdl() -> None:
    parsed = _parsed(
        [
            {
                "iteration": 3,
                "stage": "decode",
                "reference_device_id": 0,
                "ranges": [
                    {
                        "device_id": 0,
                        "worker": {"global_pid": 1000},
                        "phase": "forward",
                        "kernels": [_kernel(1, 1, 0, 10, 7)],
                    },
                    {
                        "device_id": 0,
                        "worker": {"global_pid": 2000},
                        "phase": "forward",
                        "kernels": [_kernel(1, 2, 0, 10, 7)],
                    },
                ],
            }
        ]
    )

    report = build_overlap_diagnostics(parsed)

    assert report["overall"]["pdl_same_stream_trace_reduction_ns"] == 0
    assert report["overall"]["multistream_trace_reduction_ns"] == 10
    assert report["pdl_kernel_pairs"] == []


def test_defaults_to_reference_device_and_supports_explicit_filters() -> None:
    parsed = _parsed(
        [
            {
                "iteration": 1,
                "stage": "mixed",
                "reference_device_id": 1,
                "ranges": [
                    {
                        "device_id": 0,
                        "worker": {"global_pid": 1000},
                        "phase": "forward",
                        "kernels": [_kernel(1, 1, 0, 10, 7)],
                    },
                    {
                        "device_id": 1,
                        "worker": {"global_pid": 1001},
                        "phase": "forward",
                        "kernels": [_kernel(1, 2, 0, 20, 7)],
                    },
                    {
                        "device_id": 1,
                        "worker": {"global_pid": 1001},
                        "phase": "sample",
                        "kernels": [_kernel(1, 3, 20, 25, 7)],
                    },
                ],
            },
            _detail(2, "decode", 0, 0, [_kernel(1, 1, 30, 37, 7)]),
        ]
    )

    reference = build_overlap_diagnostics(parsed)
    explicit = build_overlap_diagnostics(
        parsed,
        device_ids={0},
        stages={"mixed"},
        phases={"forward"},
    )
    all_devices = build_overlap_diagnostics(parsed, all_devices=True, phases={"forward"})

    assert [(row["iteration"], row["device_id"]) for row in reference["by_iteration_device"]] == [
        (1, 1),
        (2, 0),
    ]
    assert reference["overall"]["raw_kernel_ns"] == 32
    assert [(row["iteration"], row["device_id"]) for row in explicit["by_iteration_device"]] == [
        (1, 0)
    ]
    assert explicit["overall"]["raw_kernel_ns"] == 10
    assert [(row["iteration"], row["device_id"]) for row in all_devices["by_iteration_device"]] == [
        (1, 0),
        (1, 1),
        (2, 0),
    ]


def test_rejects_conflicting_device_policies_and_bad_intervals() -> None:
    parsed = _parsed([_detail(1, "decode", 0, 0, [_kernel(1, 1, 10, 5, 7)])])

    with pytest.raises(ValueError, match="mutually exclusive"):
        build_overlap_diagnostics(parsed, device_ids={0}, all_devices=True)
    with pytest.raises(ValueError, match="non-negative"):
        build_overlap_diagnostics(parsed, top_pairs=-1)
    with pytest.raises(ValueError, match="unknown stage selection"):
        build_overlap_diagnostics(parsed, stages={"prefill"})
    with pytest.raises(ValueError, match="unknown phase selection"):
        build_overlap_diagnostics(parsed, phases={"sample"})
    with pytest.raises(ValueError, match="unknown device selection"):
        build_overlap_diagnostics(parsed, device_ids={7})
    with pytest.raises(ValueError, match="before start_ns"):
        build_overlap_diagnostics(parsed)


def test_rejects_valid_filters_with_no_joint_rows() -> None:
    parsed = _parsed(
        [
            _detail(1, "mixed", 0, 0, [_kernel(1, 1, 0, 10, 7)], phase="forward"),
            _detail(2, "decode", 0, 0, [_kernel(1, 2, 20, 30, 7)], phase="sample"),
        ]
    )

    with pytest.raises(ValueError, match="matched no iteration/device rows"):
        build_overlap_diagnostics(parsed, stages={"mixed"}, phases={"sample"})
