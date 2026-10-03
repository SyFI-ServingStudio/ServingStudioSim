"""End-to-end CPU test for the rocpd alignment producer (``alignment/rocpd``).

Builds a synthetic rocpd SQLite database with roctx ``vllm_iteration(N): forward``
ranges and kernel dispatches inside them, then proves the whole offline path:

1. the timestamp-containment ownership join attributes each dispatch to the
   iteration range that holds its launch, and drops dispatches outside every
   range;
2. ``parse_trace`` + the shared writers emit a ``parsed.json`` +
   ``parsed.kernels.parquet`` whose columns are byte-schema-identical to the
   nsys producer (``_KERNEL_SCHEMA``, all non-null), and whose per-kernel
   ``category`` equals the folded ``suggested_category`` — the exact equality the
   Rust Check-1 reader asserts;
3. the folded ``kernel_sequences.json`` passes the ``label`` stage
   (initialize → walk → check) unchanged.

No GPU and no rocprofiler-sdk needed; the fixture is a plain sqlite file.
"""

from __future__ import annotations

import json
import sqlite3

import pyarrow.parquet as pq

from alignment.labeling import cli as labeling_cli
from alignment.nsys.parsed_io import _KERNEL_SCHEMA, kernel_rows_path, read_parsed, write_parsed
from alignment.nsys.sequence import expand_sequence
from alignment.rocpd.evidence import build_ranges_from_rocpd
from alignment.rocpd.parse import main as rocpd_main
from alignment.rocpd.parse import parse_trace
from profiling.profilers.rocprof_kernel_profiler import roctx_regions_from_rocpd

_GUID = "0000b1c7_c35b_735b_96a7_f0a02ff013cc"
_PID = 4242
_TID = 55
_AGENT = 1


def _write_synthetic_rocpd(path, dispatches, regions):
    """Write a rocpd db with roctx regions + full dispatch rows.

    ``dispatches`` is ``(start, end, name, stream_id)``; ``regions`` is
    ``(start, end, text)``. All dispatches share one process/device so the
    ownership join turns purely on timestamp containment.
    """
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE rocpd_string_{_GUID} (id INTEGER PRIMARY KEY, string TEXT)")
    conn.execute(
        f"CREATE TABLE rocpd_info_kernel_symbol_{_GUID} "
        "(id INTEGER PRIMARY KEY, kernel_name TEXT, display_name TEXT)"
    )
    conn.execute(
        f"CREATE TABLE rocpd_kernel_dispatch_{_GUID} "
        "(id INTEGER PRIMARY KEY, pid INTEGER, tid INTEGER, agent_id INTEGER, "
        "kernel_id INTEGER, dispatch_id INTEGER, queue_id INTEGER, stream_id INTEGER, "
        "start BIGINT, end BIGINT, region_name_id INTEGER)"
    )
    conn.execute(
        f"CREATE TABLE rocpd_region_{_GUID} "
        "(id INTEGER PRIMARY KEY, pid INTEGER, tid INTEGER, start BIGINT, end BIGINT, name_id INTEGER)"
    )
    strings: dict[str, int] = {}

    def intern(text):
        if text not in strings:
            strings[text] = len(strings) + 1
            conn.execute(
                f"INSERT INTO rocpd_string_{_GUID} (id, string) VALUES (?, ?)",
                (strings[text], text),
            )
        return strings[text]

    symbols: dict[str, int] = {}
    for _s, _e, name, _stream in dispatches:
        if name not in symbols:
            symbols[name] = len(symbols) + 1
            conn.execute(
                f"INSERT INTO rocpd_info_kernel_symbol_{_GUID} (id, kernel_name, display_name) "
                "VALUES (?, ?, ?)",
                (symbols[name], f"_mangled_{name}", name),
            )
    for i, (start, end, name, stream) in enumerate(dispatches, start=1):
        conn.execute(
            f"INSERT INTO rocpd_kernel_dispatch_{_GUID} "
            "(id, pid, tid, agent_id, kernel_id, dispatch_id, queue_id, stream_id, "
            "start, end, region_name_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (i, _PID, _TID, _AGENT, symbols[name], i, 0, stream, start, end, 0),
        )
    for i, (start, end, text) in enumerate(regions, start=1):
        conn.execute(
            f"INSERT INTO rocpd_region_{_GUID} (id, pid, tid, start, end, name_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (i, _PID, _TID, start, end, intern(text)),
        )
    conn.commit()
    conn.close()


def _one_iteration_fixture(path):
    """A single forward iteration: six dispatches inside, one warm-up outside."""
    _write_synthetic_rocpd(
        path,
        dispatches=[
            (500, 900, "warmup_fill_kernel", 0),  # BEFORE the range -> dropped
            (1_100, 1_300, "rms_norm_kernel", 0),
            (1_400, 2_000, "hipblaslt_gemm_f16", 0),
            (2_100, 2_900, "flash_fwd_attn_kernel", 0),
            (3_000, 3_200, "silu_and_mul_kernel", 0),
            (3_300, 3_900, "hipblaslt_gemm_f16", 0),
            (4_000, 4_200, "vectorized_elementwise_kernel", 0),
        ],
        regions=[(1_000, 5_000, "vllm_iteration(0): forward")],
    )


def test_containment_join_attributes_and_drops(tmp_path):
    db = tmp_path / "iter.db"
    _one_iteration_fixture(db)
    ranges = build_ranges_from_rocpd(str(db))
    assert len(ranges) == 1
    item = ranges[0]
    assert (item.iteration, item.phase) == (0, "forward")
    assert item.worker.device_id == _AGENT
    assert item.kernel_count == 6  # the warm-up dispatch before the range is dropped
    assert "warmup_fill_kernel" not in {event.name for event in item.kernel_events}
    # Kernels come back in launch order.
    assert [event.start for event in item.kernel_events] == sorted(
        event.start for event in item.kernel_events
    )


def test_amd_kernels_classify_off_the_other_bucket(tmp_path):
    db = tmp_path / "iter.db"
    _one_iteration_fixture(db)
    ranges = build_ranges_from_rocpd(str(db))
    by_name = {event.name: event.category for event in ranges[0].kernel_events}
    assert by_name["hipblaslt_gemm_f16"] == "gemm_or_cutlass"
    assert by_name["flash_fwd_attn_kernel"] == "attention"
    assert by_name["rms_norm_kernel"] == "norm_reduce"
    assert by_name["silu_and_mul_kernel"] == "activation"


def test_parsed_files_match_kernel_schema_and_category_parity(tmp_path):
    db = tmp_path / "iter.db"
    _one_iteration_fixture(db)
    parsed = parse_trace(db)

    out = tmp_path / "parsed.json"
    write_parsed(out, parsed)
    rows_path = kernel_rows_path(out)
    assert rows_path.exists()

    table = pq.read_table(rows_path)
    # Byte-schema parity with the nsys producer: identical columns and dtypes.
    assert table.schema.names == _KERNEL_SCHEMA.names
    for field in _KERNEL_SCHEMA:
        assert table.schema.field(field.name).type == field.type
    # Every column is non-null (the Rust reader requires it).
    for name in _KERNEL_SCHEMA.names:
        assert table.column(name).null_count == 0

    # The exact equality the Rust Check-1 reader asserts: each measured parquet
    # category equals the folded suggested_category at the same position.
    sequences = parsed["kernel_sequences"]["forward"]["unique_sequences"]
    assert len(sequences) == 1
    suggested = [kernel["suggested_category"] for kernel in expand_sequence(sequences[0])]
    reread = read_parsed(out)
    measured = [
        kernel["category"]
        for detail in reread["iteration_details"]
        for range_row in detail["ranges"]
        for kernel in range_row["kernels"]
    ]
    assert measured == suggested
    assert len(measured) == 6


def test_produced_files_pass_the_label_stage(tmp_path):
    db = tmp_path / "iter.db"
    _one_iteration_fixture(db)
    parsed_out = tmp_path / "parsed.json"
    sequences_out = tmp_path / "kernel_sequences.json"
    # Exercise the CLI path end to end (the `rocpd-parse` subcommand target).
    assert (
        rocpd_main(
            [
                "--db",
                str(db),
                "--output",
                str(parsed_out),
                "--sequences-output",
                str(sequences_out),
            ]
        )
        == 0
    )
    assert sequences_out.exists()

    labeled = tmp_path / "kernel_sequences_labeled.json"
    assert labeling_cli.main(["initialize", str(sequences_out), str(labeled)]) == 0
    assert labeled.exists()
    # `walk` reads the labeled inventory in program order; `check` must find no
    # error-severity defect in a freshly produced, consistently-categorized file.
    assert labeling_cli.main(["walk", str(labeled)]) == 0
    assert labeling_cli.main(["check", str(labeled)]) == 0


def _write_real_label_schema_rocpd(path, regions, dispatches):
    """Write a rocpd db mirroring a REAL rocprofv3-1.3.2 capture's label layout.

    The difference from ``_write_synthetic_rocpd`` is where the roctx label lives.
    On a real capture the region's ``name_id`` resolves (via ``rocpd_string``) only
    to the API op name ``roctxThreadRangeA`` — identical for every range — while
    the actual ``vllm_iteration(N): <phase>`` text is JSON in the event row the
    region points at via ``event_id``: ``rocpd_event.extdata`` = ``{"message": ...}``.
    This fixture reproduces that exactly: every region's ``name_id`` points at the
    decoy op-name string, and the real label is only in the joined event's extdata.

    ``regions`` is ``(start, end, label)``; ``dispatches`` is ``(start, end, name)``.
    """
    conn = sqlite3.connect(path)
    conn.execute(f"CREATE TABLE rocpd_string_{_GUID} (id INTEGER PRIMARY KEY, string TEXT)")
    conn.execute(
        f"CREATE TABLE rocpd_info_kernel_symbol_{_GUID} "
        "(id INTEGER PRIMARY KEY, kernel_name TEXT, display_name TEXT)"
    )
    conn.execute(
        f"CREATE TABLE rocpd_kernel_dispatch_{_GUID} "
        "(id INTEGER PRIMARY KEY, pid INTEGER, tid INTEGER, agent_id INTEGER, "
        "kernel_id INTEGER, dispatch_id INTEGER, queue_id INTEGER, stream_id INTEGER, "
        "start BIGINT, end BIGINT, region_name_id INTEGER)"
    )
    # The real region carries an ``event_id`` FK; the event row holds ``extdata``.
    conn.execute(
        f"CREATE TABLE rocpd_region_{_GUID} "
        "(id INTEGER PRIMARY KEY, pid INTEGER, tid INTEGER, start BIGINT, end BIGINT, "
        "name_id INTEGER, event_id INTEGER, extdata TEXT)"
    )
    conn.execute(
        f"CREATE TABLE rocpd_event_{_GUID} "
        "(id INTEGER PRIMARY KEY, category_id INTEGER, extdata TEXT)"
    )
    # The single decoy op-name string every region's name_id points at.
    decoy_id = 1
    conn.execute(
        f"INSERT INTO rocpd_string_{_GUID} (id, string) VALUES (?, ?)",
        (decoy_id, "roctxThreadRangeA"),
    )

    symbols: dict[str, int] = {}
    for _s, _e, name in dispatches:
        if name not in symbols:
            symbols[name] = len(symbols) + 1
            conn.execute(
                f"INSERT INTO rocpd_info_kernel_symbol_{_GUID} (id, kernel_name, display_name) "
                "VALUES (?, ?, ?)",
                (symbols[name], f"_mangled_{name}", name),
            )
    for i, (start, end, name) in enumerate(dispatches, start=1):
        conn.execute(
            f"INSERT INTO rocpd_kernel_dispatch_{_GUID} "
            "(id, pid, tid, agent_id, kernel_id, dispatch_id, queue_id, stream_id, "
            "start, end, region_name_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (i, _PID, _TID, _AGENT, symbols[name], i, 0, 0, start, end, 0),
        )
    for i, (start, end, label) in enumerate(regions, start=1):
        conn.execute(
            f"INSERT INTO rocpd_event_{_GUID} (id, category_id, extdata) VALUES (?, ?, ?)",
            (i, 39, json.dumps({"message": label})),
        )
        conn.execute(
            f"INSERT INTO rocpd_region_{_GUID} "
            "(id, pid, tid, start, end, name_id, event_id, extdata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (i, _PID, _TID, start, end, decoy_id, i, "{}"),
        )
    conn.commit()
    conn.close()


def test_label_recovered_from_event_extdata_not_name_id(tmp_path):
    """Gap 1 regression: the roctx label comes from event.extdata.message.

    A real rocprofv3 capture interns only ``roctxThreadRangeA`` in the region's
    ``name_id``; the true ``vllm_iteration(N)`` text is in the joined event's
    ``extdata`` JSON. The reader must recover the markers from extdata — a reader
    that reads ``name_id`` would see every range as ``roctxThreadRangeA`` and match
    no iteration, producing an empty alignment.
    """
    db = tmp_path / "real_label.db"
    _write_real_label_schema_rocpd(
        db,
        regions=[
            (1_000, 5_000, "vllm_iteration(0): forward"),
            (6_000, 10_000, "vllm_iteration(1): forward"),
        ],
        dispatches=[
            (500, 900, "warmup_fill_kernel"),  # before iter 0 -> dropped
            (1_500, 2_000, "hipblaslt_gemm_f16"),  # inside iter 0
            (6_500, 7_000, "flash_fwd_attn_kernel"),  # inside iter 1
        ],
    )

    # The reader surfaces the extdata message, NOT the decoy op name.
    names = {region.name for region in roctx_regions_from_rocpd(str(db))}
    assert names == {"vllm_iteration(0): forward", "vllm_iteration(1): forward"}
    assert "roctxThreadRangeA" not in names

    # ...and the full ownership join attributes dispatches to those iterations.
    ranges = build_ranges_from_rocpd(str(db))
    by_iter = {item.iteration: item for item in ranges}
    assert set(by_iter) == {0, 1}
    assert by_iter[0].kernel_count == 1 and by_iter[1].kernel_count == 1
    assert {e.name for item in ranges for e in item.kernel_events} == {
        "hipblaslt_gemm_f16",
        "flash_fwd_attn_kernel",
    }


def test_no_iteration_markers_raises(tmp_path):
    # The upstream (uninstrumented) capture has dispatches but no roctx iteration
    # ranges; the producer must refuse rather than emit an empty alignment.
    import pytest

    db = tmp_path / "bare.db"
    _write_synthetic_rocpd(
        db,
        dispatches=[(100, 200, "rms_norm_kernel", 0)],
        regions=[],
    )
    with pytest.raises(ValueError, match="no .*iteration"):
        build_ranges_from_rocpd(str(db))
