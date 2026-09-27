from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ANALYZER_PYTHON = Path(__file__).resolve().parents[1] / "analyzer" / "python"
sys.path.insert(0, str(ANALYZER_PYTHON))

from alignment_iteration import series_plot  # noqa: E402
from alignment_iteration.series_plot import (  # noqa: E402
    _display_stream_rows,
    _evenly_sample_iteration_ids,
    _mapping_center_pairs,
    _operation_comparison_rows,
    _output_is_current,
    _recommended_gpu_time_multiplier,
    _remove_stale_breakdown_outputs,
    _render_stream_breakdown,
    _simulated_width_cumulative_error_steps,
    _stream_operation_rows,
)
from common.layout import read_sharded_records  # noqa: E402


def test_evenly_sample_iteration_ids_caps_and_keeps_raw_endpoints() -> None:
    rows = [{"iteration_id": iteration_id} for iteration_id in range(6, 2471)]

    sampled_ids = _evenly_sample_iteration_ids(rows)

    assert len(sampled_ids) == 128
    assert sampled_ids[0] == 6
    assert sampled_ids[-1] == 2470
    assert sampled_ids == sorted(set(sampled_ids))
    gaps = [right - left for left, right in zip(sampled_ids, sampled_ids[1:])]
    assert max(gaps) - min(gaps) <= 1


def test_evenly_sample_iteration_ids_sorts_and_keeps_small_inputs() -> None:
    rows = [{"iteration_id": 9}, {"iteration_id": 3}, {"iteration_id": 7}]

    assert _evenly_sample_iteration_ids(rows) == [3, 7, 9]


@pytest.mark.parametrize("value", [None, 0.9812, float("nan"), True, "1.2"])
def test_invalid_duty_cycle_multiplier_does_not_block_kernel_plots(value) -> None:
    payload = {"meta": {"recommended_gpu_time_multiplier": value}}

    assert _recommended_gpu_time_multiplier(payload) is None


def test_valid_duty_cycle_multiplier_enables_gpu_cycle_plot() -> None:
    payload = {"meta": {"recommended_gpu_time_multiplier": 1.2}}

    assert _recommended_gpu_time_multiplier(payload) == 1.2


def test_sharded_records_are_read_by_byte_range_in_the_order_asked_for(tmp_path) -> None:
    """A renderer sampling 2 of 4 iterations must not parse the other 2."""
    import json

    payloads = tmp_path / "payloads"
    payloads.mkdir()
    records = [
        {"iteration_id": iteration_id, "payload": "x" * iteration_id}
        for iteration_id in (3, 7, 9, 11)
    ]
    byte_ranges, blob = {}, b""
    for record in records:
        line = json.dumps(record).encode()
        byte_ranges[str(record["iteration_id"])] = [len(blob), len(line)]
        blob += line + b"\n"
    (payloads / "shard.jsonl").write_bytes(blob)

    read = read_sharded_records(
        tmp_path, {"file": "shard.jsonl", "byte_ranges": byte_ranges}, [9, 3, 999]
    )

    # Order follows the request, and an id the index does not know is skipped
    # rather than raising: a payload written before its shard is a missing
    # figure, not a crash.
    assert [record["iteration_id"] for record in read] == [9, 3]


def test_zstd_frame_shards_decode_each_record_alone(tmp_path) -> None:
    import json

    import pyarrow as pa

    payloads = tmp_path / "payloads"
    payloads.mkdir()
    byte_ranges, decoded_lengths, blob = {}, {}, b""
    for iteration_id in (3, 7):
        line = json.dumps({"iteration_id": iteration_id}).encode() + b"\n"
        frame = pa.compress(line, codec="zstd", asbytes=True)
        byte_ranges[str(iteration_id)] = [len(blob), len(frame)]
        decoded_lengths[str(iteration_id)] = len(line)
        blob += frame
    (payloads / "shard.jsonl.zst").write_bytes(blob)
    shard = {
        "file": "shard.jsonl.zst",
        "encoding": "zstd-frames",
        "byte_ranges": byte_ranges,
        "decoded_lengths": decoded_lengths,
    }

    read = read_sharded_records(tmp_path, shard, [7, 3])

    assert [record["iteration_id"] for record in read] == [7, 3]

    # A range whose decoded length the index lost is a missing figure, not a crash.
    del shard["decoded_lengths"]["3"]
    read = read_sharded_records(tmp_path, shard, [7, 3])
    assert [record["iteration_id"] for record in read] == [7]


def test_mapping_center_pairs_keeps_one_to_many_simulated_slots() -> None:
    measured_centers = {"layer.attention": [1.5]}
    simulated_centers = {"layer.attention": [2.0, 3.0]}

    assert _mapping_center_pairs(measured_centers, simulated_centers) == [
        ("layer.attention", 1.5, 2.0),
        ("layer.attention", 1.5, 3.0),
    ]


def test_simulated_width_cumulative_error_steps_use_critical_path_widths() -> None:
    measured = [
        {
            "phase": "forward",
            "name": "attention",
            "operation": "layer.attention",
            "duration_ms": 1.5,
        },
        {
            "phase": "forward",
            "name": "helper",
            "operation": None,
            "duration_ms": 0.2,
        },
    ]
    simulated = [
        {"name": "prefill", "operation": "layer.attention", "critical_path_ms": 0.4},
        {"name": "decode", "operation": "layer.attention", "critical_path_ms": 0.6},
        {"name": "sim-only", "operation": None, "critical_path_ms": 0.3},
    ]

    baseline_ms, edges_ms, cumulative_errors_ms = _simulated_width_cumulative_error_steps(
        measured, simulated
    )

    assert baseline_ms == -0.2
    assert edges_ms == [0.0, 0.4, 1.0, 1.3]
    assert cumulative_errors_ms == pytest.approx([-0.4, -0.7, -0.4])
    assert cumulative_errors_ms[-1] == pytest.approx(1.3 - 1.7)


def test_operation_comparison_rows_format_cells_and_rank_absolute_error() -> None:
    rows = _operation_comparison_rows(
        ["attention", "sim-only"],
        [{"operation": "attention", "duration_ms": 0.5}],
        [
            {"operation": "attention", "duration_ms": 0.75},
            {"operation": "sim-only", "duration_ms": 0.2},
        ],
    )

    assert rows == [
        {
            "operation": "attention",
            "measured": "500 µs",
            "simulated": "750 µs",
            "relative": "+50.0%",
            "absolute_error_ms": 0.25,
            "is_top_error": True,
        },
        {
            "operation": "sim-only",
            "measured": "0 µs",
            "simulated": "200 µs",
            "relative": "n/a",
            "absolute_error_ms": 0.2,
            "is_top_error": True,
        },
    ]


def test_operation_comparison_rows_separate_directional_mapping_gaps() -> None:
    rows = _operation_comparison_rows(
        ["unmapped (measured)", "unmapped (simulated)"],
        [{"operation": None, "duration_ms": 0.4}],
        [{"operation": None, "duration_ms": 0.2}],
    )

    assert [(row["operation"], row["measured"], row["simulated"]) for row in rows] == [
        ("unmapped (measured)", "400 µs", "0 µs"),
        ("unmapped (simulated)", "0 µs", "200 µs"),
    ]


def test_output_is_current_tracks_every_render_input(tmp_path: Path) -> None:
    payload = tmp_path / "payload.json"
    renderer = tmp_path / "renderer.py"
    output = tmp_path / "plot.png"
    payload.write_text("payload")
    renderer.write_text("renderer")

    assert not _output_is_current(output, payload, renderer)

    output.write_bytes(b"png")
    os.utime(payload, ns=(100, 100))
    os.utime(renderer, ns=(200, 200))
    os.utime(output, ns=(300, 300))
    assert _output_is_current(output, payload, renderer)

    os.utime(payload, ns=(400, 400))
    assert not _output_is_current(output, payload, renderer)


def test_remove_stale_breakdowns_keeps_selected_jpg_fallback(tmp_path: Path) -> None:
    plots = tmp_path / "plots"
    selected_dir = plots / "iter_1_to_32"
    stale_dir = plots / "iter_33_to_64"
    selected_dir.mkdir(parents=True)
    stale_dir.mkdir()
    selected_png = selected_dir / "iter_1_breakdown.png"
    selected_jpg = selected_png.with_suffix(".jpg")
    stale_jpg = stale_dir / "iter_33_breakdown.jpg"
    selected_jpg.write_bytes(b"fallback")
    stale_jpg.write_bytes(b"stale")

    _remove_stale_breakdown_outputs(tmp_path, {selected_png})

    assert selected_jpg.is_file()
    assert not stale_jpg.exists()
    assert not stale_dir.exists()


def test_stream_breakdown_uses_reduced_work_and_aggregates_small_streams() -> None:
    kernels = [
        {
            "ph": "forward",
            "op": "main",
            "occ_ns": 100_000_000,
            "iv": [[0, 0, 100_000_000, 0, 0]],
        }
    ]
    for track_index in range(1, 10):
        kernels.append(
            {
                "ph": "forward",
                "op": "side",
                # Use reduced work, not the collective's long raw residency.
                "occ_ns": 100_000,
                "iv": [[0, 0, 28_000_000_000, track_index, track_index]],
            }
        )

    rows = _stream_operation_rows({"measured": {"kernels": kernels}}, device_id=0)
    displayed = _display_stream_rows(rows)

    assert sum(row["duration_ms"] for stream in rows.values() for row in stream) == pytest.approx(
        100.9
    )
    assert [label for label, _stream in displayed] == [
        "stream 0",
        "other 9 streams (aggregated)",
    ]
    assert sum(
        row["duration_ms"] for _label, stream in displayed for row in stream
    ) == pytest.approx(100.9)


def test_stream_breakdown_reserves_aligned_non_overlapping_table_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    operation_names = [f"attention.indexer.long_semantic_operation_{index}" for index in range(30)]
    kernels = [
        {
            "ph": "forward",
            "op": operation,
            "occ_ns": 1_000_000,
            "iv": [[0, index * 1_000_000, (index + 1) * 1_000_000, 0, 0]],
        }
        for index, operation in enumerate(operation_names)
    ]
    breakdown = {
        "iteration_id": 203,
        "stage": "decode",
        "critical_busy_ms_by_device": {"0": 30.0},
        "measured_ms": 30.0,
        "measured_kernels": [
            {"phase": "forward", "operation": operation, "duration_ms": 1.0}
            for operation in operation_names
        ],
        "simulated_kernels": [
            {"name": operation, "operation": operation, "critical_path_ms": 1.0}
            for operation in operation_names
        ],
    }
    captured: dict[str, object] = {}

    def capture_plot(fig, _path, **_kwargs) -> None:
        captured["figure"] = fig

    monkeypatch.setattr(series_plot, "save_plot", capture_plot)
    _render_stream_breakdown(
        breakdown,
        {"measured": {"kernels": kernels}},
        tmp_path / "breakdown.png",
        run_label="layout-regression",
    )

    figure = captured["figure"]
    figure.canvas.draw()
    axis = figure.axes[0]
    xlabel_bounds = axis.xaxis.label.get_window_extent(figure.canvas.get_renderer()).transformed(
        figure.transFigure.inverted()
    )

    assert axis.get_legend() is None
    assert not figure.legends
    table_axes = figure.axes[1:]
    assert len(table_axes) == 3
    assert all(table_axis.get_position().y1 < xlabel_bounds.y0 for table_axis in table_axes)
    first_table = table_axes[0].tables[0]
    assert [first_table[0, column].get_text().get_text() for column in range(5)] == [
        "",
        "operation",
        "measured",
        "simulated",
        "Δ",
    ]
    operation_cells = [
        table_axis.tables[0][row_index, 1]
        for table_axis in table_axes
        for row_index in range(1, len(table_axis.tables[0].get_celld()) // 5)
    ]
    assert all(cell.get_text().get_ha() == "left" for cell in operation_cells)
    assert sum(cell.get_text().get_weight() == "bold" for cell in operation_cells) == 5
    assert xlabel_bounds.y1 < axis.get_position().y0
    series_plot.plt.close(figure)


def test_stream_breakdown_draws_the_reported_path_on_the_busiest_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    breakdown = {
        "iteration_id": 8684,
        "stage": "decode",
        # Rank 1 owns more of the critical busy time, so its streams are drawn.
        "critical_busy_ms_by_device": {"0": 3.0, "1": 7.0},
        "critical_rank_switches": 4,
        "measured_ms": 10.0,
        "measured_kernel_sum_ms": 14.0,
        "measured_kernels": [
            {"phase": "forward", "operation": "attention.o_proj", "duration_ms": 6.0},
            {
                "phase": "forward",
                "operation": "moe.routed_experts.fused_moe",
                "duration_ms": 4.0,
            },
        ],
        "simulated_kernels": [
            {"name": "o", "operation": "attention.o_proj", "critical_path_ms": 7.0},
            {
                "name": "m",
                "operation": "moe.routed_experts.fused_moe",
                "critical_path_ms": 5.0,
            },
        ],
    }
    kernels = [
        {
            "ph": "forward",
            "op": "attention.o_proj",
            "occ_ns": 9_000_000,
            "iv": [[0, 0, 6_000_000, 0, 0], [1, 0, 5_000_000, 0, 0]],
        },
        {
            "ph": "forward",
            "op": "moe.routed_experts.fused_moe",
            "occ_ns": 7_000_000,
            "iv": [[1, 6_000_000, 10_000_000, 0, 0]],
        },
    ]
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        series_plot,
        "save_plot",
        lambda figure, _path, **_kwargs: captured.setdefault("figure", figure),
    )

    _render_stream_breakdown(
        breakdown,
        {"measured": {"kernels": kernels}},
        tmp_path / "breakdown.png",
        run_label="sign-regression",
    )

    figure = captured["figure"]
    axis = figure.axes[0]
    title = axis.get_title()
    assert "streams of GPU 1 · critical rank changed 4x" in title
    assert "Nsight 10.000 ms ·" in title
    assert "Timing-predict 12.000 ms" in title
    assert "delta +2.000 ms (+20.00%)" in title
    labels = [label.get_text() for label in axis.get_yticklabels()]
    assert labels[0].startswith("GPU 1 ")
    assert labels[-2:] == ["Nsight barrier critical path", "Timing-predict critical path"]

    tick_by_label = {
        label.get_text(): tick for label, tick in zip(axis.get_yticklabels(), axis.get_yticks())
    }

    def row_widths(label: str) -> list[float]:
        tick = tick_by_label[label]
        return [
            patch.get_width()
            for patch in axis.patches
            if patch.get_y() + patch.get_height() / 2 == pytest.approx(tick)
        ]

    assert sum(row_widths("Timing-predict critical path")) == pytest.approx(12.0)
    assert sum(row_widths("Nsight barrier critical path")) == pytest.approx(10.0)
    series_plot.plt.close(figure)
