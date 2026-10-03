"""Host (no-GPU) tests for the rocpd per-rank merge (``alignment/rocpd/merge.py``).

A tensor-parallel rocpd capture produces one single-device ``parsed.json`` per
worker. ``rocpd-merge-ranks`` assembles the N of them into the ONE multi-device
document the Rust Check-1 reader needs (``device_ids = [0..N-1]``, ranks reduced
per-kernel-position by ``MeasuredRange.device_id``). These tests synthesize four
schema-valid per-rank documents from real ``--kernel-trace`` sentinel databases
with GLM-relevant kernel names and assert the merge is faithful, including the
load-bearing rule that device identity comes from the FILENAME RANK, never the
in-DB ``agent_id`` (which is 0 on every worker process).
"""

from __future__ import annotations

import json
import sqlite3

import pyarrow.parquet as pq

from alignment.nsys.parsed_io import _KERNEL_SCHEMA, kernel_rows_path, read_parsed
from alignment.nsys.sequence import expand_sequence
from alignment.profiler.roctx_shim import SENTINEL_GRID_Y_OFFSET, SENTINEL_KERNEL_NAME
from alignment.rocpd.merge import main as merge_main
from alignment.rocpd.merge import merge_documents
from alignment.rocpd.parse import main as rocpd_main

_GUID = "0000b1c7_c35b_735b_96a7_f0a02ff013cc"
_SENTINEL_DISPLAY = f"{SENTINEL_KERNEL_NAME}_0d1d"  # a Triton-mangled JIT name

# GLM-5.3-Flash-relevant AMD kernel names exercising the category buckets.
GLM_KERNELS = [
    ("rms_norm_kernel", "norm_reduce"),
    ("hipblaslt_gemm_f16", "gemm_or_cutlass"),
    ("dsa_sparse_mla_attn_fwd_kernel", "attention"),
    ("ck_moe_gemm_fused_moe_kernel", "fused_moe"),
    ("silu_and_mul_kernel", "activation"),
]


def _write_sentinel_rocpd(path, agent_id, pid, n_iters=3):
    """A real ``--kernel-trace`` capture: sentinel iteration markers (iteration
    encoded in ``grid_size_y``), GLM kernels, name on the joined symbol table.

    Every worker's ``agent_id`` defaults to 0, exactly as a real per-process
    rocpd DB stores it, so the merge cannot read device identity from here.
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
        "start BIGINT, end BIGINT, "
        "grid_size_x INTEGER, grid_size_y INTEGER, grid_size_z INTEGER, "
        "workgroup_size_x INTEGER, workgroup_size_y INTEGER, workgroup_size_z INTEGER, "
        "region_name_id INTEGER)"
    )
    dispatches = [(50, 90, "warmup_fill_kernel", 1)]  # before sentinel 0 -> dropped
    t = 100
    for iteration in range(n_iters):
        dispatches.append((t, t + 20, _SENTINEL_DISPLAY, iteration + SENTINEL_GRID_Y_OFFSET))
        t += 100
        for name, _category in GLM_KERNELS:
            dispatches.append((t, t + 80, name, 1))
            t += 100
        t += 100

    symbols: dict[str, int] = {}
    for _s, _e, name, _gy in dispatches:
        if name not in symbols:
            symbols[name] = len(symbols) + 1
            conn.execute(
                f"INSERT INTO rocpd_info_kernel_symbol_{_GUID} (id, kernel_name, display_name) "
                "VALUES (?, ?, ?)",
                (symbols[name], f"_mangled_{name}", name),
            )
    for i, (start, end, name, grid_y) in enumerate(dispatches, start=1):
        conn.execute(
            f"INSERT INTO rocpd_kernel_dispatch_{_GUID} "
            "(id, pid, tid, agent_id, kernel_id, dispatch_id, queue_id, stream_id, "
            "start, end, grid_size_x, grid_size_y, grid_size_z, "
            "workgroup_size_x, workgroup_size_y, workgroup_size_z, region_name_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (i, pid, 55, agent_id, symbols[name], i, 0, 0, start, end,
             64, grid_y, 1, 64, 1, 1, 0),
        )
    conn.commit()
    conn.close()


def _produce_per_rank(tmp_path, n_ranks=4, agent_id=0):
    """Write N per-rank DBs (all with the same in-DB agent_id) and parse each.

    Returns the ordered lists of parsed.json and kernel_sequences.json paths,
    named ``parsed.rank{N}.json`` so the merge's filename-rank rule applies.
    """
    parsed_paths = []
    sequences_paths = []
    for rank in range(n_ranks):
        db = tmp_path / f"glm_tp4_rank{rank}_results.db"
        _write_sentinel_rocpd(db, agent_id=agent_id, pid=4200 + rank)
        parsed_out = tmp_path / f"parsed.rank{rank}.json"
        sequences_out = tmp_path / f"kernel_sequences.rank{rank}.json"
        assert rocpd_main(
            ["--db", str(db), "--output", str(parsed_out),
             "--sequences-output", str(sequences_out)]
        ) == 0
        parsed_paths.append(parsed_out)
        sequences_paths.append(sequences_out)
    return parsed_paths, sequences_paths


def test_merge_produces_four_device_document(tmp_path):
    parsed_paths, sequences_paths = _produce_per_rank(tmp_path)
    out = tmp_path / "merged" / "parsed.json"
    seq_out = tmp_path / "merged" / "kernel_sequences.json"
    assert merge_main(
        ["--parsed", *map(str, parsed_paths),
         "--sequences", *map(str, sequences_paths),
         "--output", str(out), "--sequences-output", str(seq_out)]
    ) == 0

    merged = read_parsed(out)
    assert sorted(merged["device_ids"]) == [0, 1, 2, 3]
    assert merged["tp_size"] == 4
    assert merged["dp_rank_by_device"] == {"0": 0, "1": 0, "2": 0, "3": 0}
    assert merged["source"] == "rocpd"

    # Every logical step carries all four devices, each at its own device_id.
    for detail in merged["iteration_details"]:
        range_devices = sorted(r["device_id"] for r in detail["ranges"])
        assert range_devices == [0, 1, 2, 3]
        assert detail["devices_absent"] == []

    # The sentinel marker never enters the compared kernel-name index.
    assert not any(SENTINEL_KERNEL_NAME in name for name in merged["kernel_names"].values())

    # One consistent name-id space: every name_id in the parquet resolves, and
    # nothing in the dictionary dangles.
    table = pq.read_table(kernel_rows_path(out))
    assert table.schema.names == _KERNEL_SCHEMA.names
    for field in _KERNEL_SCHEMA:
        assert table.schema.field(field.name).type == field.type
    for name in _KERNEL_SCHEMA.names:
        assert table.column(name).null_count == 0
    used_ids = set(table.column("name_id").to_pylist())
    declared_ids = {int(key) for key in merged["kernel_names"]}
    assert used_ids == declared_ids  # no dangling id, none missing
    # 4 ranks x 3 iters x 5 GLM kernels, sentinels excluded.
    assert table.num_rows == 4 * 3 * len(GLM_KERNELS)

    # Merged labeled sequences: top-level device_ids lists all four, and the
    # symmetric TP sequence's occurrences carry every rank.
    sequences = json.loads(seq_out.read_text())
    assert sequences["schema_version"] >= 4  # union catalog (device axis)
    assert sorted(sequences["device_ids"]) == [0, 1, 2, 3]
    unique = sequences["phases"]["forward"]["unique_sequences"]
    assert len(unique) == 1
    assert sorted(o["device_id"] for o in unique[0]["occurrences"]) == [0, 1, 2, 3]

    # Category parity the Rust reader asserts, now per device: the measured
    # parquet category equals the folded suggested_category at each position.
    suggested = [kernel["suggested_category"] for kernel in expand_sequence(unique[0])]
    measured_by_device: dict[int, list[str]] = {}
    for detail in merged["iteration_details"]:
        for range_row in detail["ranges"]:
            measured_by_device.setdefault(range_row["device_id"], []).extend(
                kernel["category"] for kernel in range_row["kernels"]
            )
    for device in range(4):
        # One iteration's worth is five kernels; the per-device list repeats it.
        assert measured_by_device[device][: len(suggested)] == suggested


def test_device_id_comes_from_filename_not_in_db_agent_id(tmp_path):
    # Every worker DB carries agent_id 0; a merge that trusted the in-DB value
    # would collapse all four ranks onto device 0. Device identity must come
    # from the rankN filename instead.
    parsed_paths, sequences_paths = _produce_per_rank(tmp_path, agent_id=0)
    for path in parsed_paths:
        assert read_parsed(path)["device_ids"] == [0]  # each worker: in-DB agent 0

    out = tmp_path / "merged" / "parsed.json"
    seq_out = tmp_path / "merged" / "kernel_sequences.json"
    assert merge_main(
        ["--parsed", *map(str, parsed_paths),
         "--sequences", *map(str, sequences_paths),
         "--output", str(out), "--sequences-output", str(seq_out)]
    ) == 0
    merged = read_parsed(out)
    assert sorted(merged["device_ids"]) == [0, 1, 2, 3]
    for detail in merged["iteration_details"]:
        assert sorted(r["device_id"] for r in detail["ranges"]) == [0, 1, 2, 3]


def test_explicit_rank_overrides_filename_order(tmp_path):
    # Inputs given out of filename order, with --rank stating the device of each.
    parsed_paths, sequences_paths = _produce_per_rank(tmp_path)
    order = [2, 0, 3, 1]
    out = tmp_path / "merged" / "parsed.json"
    seq_out = tmp_path / "merged" / "kernel_sequences.json"
    assert merge_main(
        ["--parsed", *(str(parsed_paths[i]) for i in order),
         "--sequences", *(str(sequences_paths[i]) for i in order),
         "--rank", "2", "0", "3", "1",
         "--output", str(out), "--sequences-output", str(seq_out)]
    ) == 0
    merged = read_parsed(out)
    assert sorted(merged["device_ids"]) == [0, 1, 2, 3]


def test_merge_rejects_multi_device_input(tmp_path):
    import pytest

    parsed_paths, sequences_paths = _produce_per_rank(tmp_path, n_ranks=2)
    parsed = {0: read_parsed(parsed_paths[0]), 1: read_parsed(parsed_paths[1])}
    parsed[0]["device_ids"] = [0, 1]  # not a single-device per-rank doc
    sequences = {
        0: json.loads(sequences_paths[0].read_text()),
        1: json.loads(sequences_paths[1].read_text()),
    }
    with pytest.raises(ValueError, match="single-device"):
        merge_documents(parsed, sequences)
