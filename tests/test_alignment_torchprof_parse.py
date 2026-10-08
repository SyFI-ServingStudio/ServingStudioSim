"""End-to-end CPU test for the torch-profiler producer (``alignment/torchprof``).

Builds a synthetic PyTorch Kineto chrome trace -- the per-worker
``*.pt.trace.json[.gz]`` vLLM's native torch profiler writes -- and proves the
whole offline path, the torch-capture counterpart of
``test_alignment_rocpd_parse.py``:

1. the GPU ``kernel`` events are read (host ``cpu_op`` / ``cuda_runtime`` rows
   excluded), timestamps µs→ns, and the shared timestamp-containment join
   attributes each dispatch to the iteration range that holds its launch, via
   both the ``vllm_iteration(N)`` ``user_annotation`` ranges and the Option-B
   ``vibesim_sentinel`` marker kernels;
2. ``parse_trace`` + the shared writers emit a ``parsed.json`` +
   ``parsed.kernels.parquet`` whose columns are byte-schema-identical to the nsys
   producer and whose per-kernel ``category`` equals the folded
   ``suggested_category`` -- the exact equality the Rust Check-1 reader asserts;
3. the folded ``kernel_sequences.json`` passes the ``label`` stage unchanged;
4. four per-rank torch docs merge into one ``tp_size: 4`` multi-device document.

No GPU and no torch profiler needed; the fixture is a plain JSON trace.
"""

from __future__ import annotations

import gzip
import json

import pyarrow.parquet as pq
import pytest

from alignment.labeling import cli as labeling_cli
from alignment.nsys.parsed_io import _KERNEL_SCHEMA, kernel_rows_path, read_parsed, write_parsed
from alignment.nsys.sequence import expand_sequence
from alignment.profiler.roctx_shim import SENTINEL_GRID_Y_OFFSET, SENTINEL_KERNEL_NAME
from alignment.rocpd import merge as rocpd_merge
from alignment.torchprof.kineto import read_kineto_trace
from alignment.torchprof.parse import main as torch_main
from alignment.torchprof.parse import parse_trace

_SENTINEL_DISPLAY_NAME = f"{SENTINEL_KERNEL_NAME}_0d1d"  # a Triton-mangled JIT name


def _kernel_event(start_us, dur_us, name, *, stream=0, grid_y=1, device=0, corr=None):
    """A GPU ``kernel`` complete-event in the Kineto chrome-trace shape."""
    return {
        "ph": "X",
        "cat": "kernel",
        "name": name,
        "pid": device,  # Kineto sets a GPU event's pid to the device id, not an OS pid
        "tid": 7,
        "ts": start_us,
        "dur": dur_us,
        "args": {
            "device": device,
            "stream": stream,
            "correlation": corr if corr is not None else int(start_us),
            "grid": [64, grid_y, 1],
            "block": [256, 1, 1],
        },
    }


def _annotation_event(start_us, dur_us, label, *, tid=1):
    """A roctx/NVTX iteration marker as a ``user_annotation`` complete-event."""
    return {
        "ph": "X",
        "cat": "user_annotation",
        "name": label,
        "pid": 4242,
        "tid": tid,
        "ts": start_us,
        "dur": dur_us,
        "args": {},
    }


def _noise_events():
    """Host-side rows a real trace carries that the reader must NOT attribute."""
    return [
        {"ph": "X", "cat": "cpu_op", "name": "aten::mm", "pid": 4242, "tid": 1,
         "ts": 1.2, "dur": 0.4, "args": {}},
        {"ph": "X", "cat": "cuda_runtime", "name": "hipLaunchKernel", "pid": 4242,
         "tid": 1, "ts": 1.3, "dur": 0.05, "args": {"correlation": 999}},
        {"ph": "M", "name": "process_name", "pid": 0, "tid": 0,
         "args": {"name": "python"}},
    ]


def _write_trace(path, events, *, gzipped=False):
    document = {"schemaVersion": 1, "deviceProperties": [], "traceEvents": events}
    blob = json.dumps(document).encode()
    if gzipped:
        path.write_bytes(gzip.compress(blob))
    else:
        path.write_bytes(blob)


# --- roctx user_annotation iteration ranges ------------------------------------


def _one_iteration_events():
    """One forward marked by a user_annotation range; one warm-up kernel outside."""
    return [
        *_noise_events(),
        _kernel_event(0.5, 0.4, "warmup_fill_kernel"),  # before the range -> dropped
        _kernel_event(1.1, 0.2, "rms_norm_kernel"),
        _kernel_event(1.4, 0.6, "hipblaslt_gemm_f16"),
        _kernel_event(2.1, 0.8, "flash_fwd_attn_kernel"),
        _kernel_event(3.0, 0.2, "silu_and_mul_kernel"),
        _kernel_event(3.3, 0.6, "hipblaslt_gemm_f16"),
        _kernel_event(4.0, 0.2, "vectorized_elementwise_kernel"),
        _annotation_event(1.0, 4.0, "vllm_iteration(0): forward"),
    ]


def test_reader_keeps_only_gpu_kernels_and_iteration_markers(tmp_path):
    trace = tmp_path / "rank0.pt.trace.json"
    _write_trace(trace, _one_iteration_events())
    dispatches, regions = read_kineto_trace(trace)
    # cpu_op / cuda_runtime / metadata rows are excluded; only the 7 GPU kernels.
    assert len(dispatches) == 7
    assert all(d.name not in {"aten::mm", "hipLaunchKernel"} for d in dispatches)
    # µs -> ns conversion.
    rms = next(d for d in dispatches if d.name == "rms_norm_kernel")
    assert (rms.start_ns, rms.end_ns) == (1100, 1300)
    assert len(regions) == 1 and regions[0].name == "vllm_iteration(0): forward"


def test_containment_join_attributes_and_drops(tmp_path):
    trace = tmp_path / "rank0.pt.trace.json"
    _write_trace(trace, _one_iteration_events())
    parsed = parse_trace(trace)
    assert parsed["source"] == "torch"
    assert parsed["iterations"] == [0]
    details = parsed["iteration_details"]
    assert len(details) == 1
    kernels = [k for d in details for r in d["ranges"] for k in r["kernels"]]
    assert len(kernels) == 6  # the warm-up kernel before the range is dropped
    names = parsed["kernel_names"]
    assert "warmup_fill_kernel" not in names.values()


def test_parsed_files_match_kernel_schema_and_category_parity(tmp_path):
    trace = tmp_path / "rank0.pt.trace.json"
    _write_trace(trace, _one_iteration_events())
    parsed = parse_trace(trace)
    out = tmp_path / "parsed.json"
    write_parsed(out, parsed)
    rows_path = kernel_rows_path(out)
    assert rows_path.exists()

    table = pq.read_table(rows_path)
    assert table.schema.names == _KERNEL_SCHEMA.names
    for field in _KERNEL_SCHEMA:
        assert table.schema.field(field.name).type == field.type
    for name in _KERNEL_SCHEMA.names:
        assert table.column(name).null_count == 0

    # category == folded suggested_category at the same position (the Rust assert).
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
    # The AMD kernel names classify off the "other" bucket.
    by_name = {
        kernel["name_id"]: kernel["category"]
        for detail in reread["iteration_details"]
        for range_row in detail["ranges"]
        for kernel in range_row["kernels"]
    }
    names = reread["kernel_names"]
    name_cat = {names[str(nid)]: cat for nid, cat in by_name.items()}
    assert name_cat["hipblaslt_gemm_f16"] == "gemm_or_cutlass"
    assert name_cat["flash_fwd_attn_kernel"] == "attention"
    assert name_cat["rms_norm_kernel"] == "norm_reduce"
    assert name_cat["silu_and_mul_kernel"] == "activation"


def test_produced_files_pass_the_label_stage(tmp_path):
    trace = tmp_path / "rank0.pt.trace.json"
    _write_trace(trace, _one_iteration_events())
    parsed_out = tmp_path / "parsed.json"
    sequences_out = tmp_path / "kernel_sequences.json"
    assert torch_main(
        ["--trace", str(trace), "--output", str(parsed_out),
         "--sequences-output", str(sequences_out)]
    ) == 0
    assert sequences_out.exists()
    labeled = tmp_path / "kernel_sequences_labeled.json"
    assert labeling_cli.main(["initialize", str(sequences_out), str(labeled)]) == 0
    assert labeling_cli.main(["walk", str(labeled)]) == 0
    assert labeling_cli.main(["check", str(labeled)]) == 0


# --- Option B: sentinel-kernel iteration boundaries (no user_annotation) -------


def _three_iteration_sentinel_events():
    """Three forwards marked by sentinel kernels, no user_annotation ranges.

    sentinel 0 @100 | gemm, attn (iter 0) | sentinel 1 @1100 | gemm (iter 1) |
    sentinel 2 @2100 | attn, silu (iter 2). One warm-up kernel before sentinel 0.
    Sentinels encode their iteration ordinal in grid[1] (= iteration + offset).
    """
    def sentinel(start_us, iteration):
        return _kernel_event(
            start_us, 0.02, _SENTINEL_DISPLAY_NAME,
            grid_y=iteration + SENTINEL_GRID_Y_OFFSET,
        )

    return [
        *_noise_events(),
        _kernel_event(0.05, 0.04, "warmup_fill_kernel"),  # before sentinel 0 -> dropped
        sentinel(0.1, 0),
        _kernel_event(0.2, 0.2, "hipblaslt_gemm_f16"),
        _kernel_event(0.5, 0.4, "flash_fwd_attn_kernel"),
        sentinel(1.1, 1),
        _kernel_event(1.3, 0.4, "hipblaslt_gemm_f16"),
        sentinel(2.1, 2),
        _kernel_event(2.3, 0.2, "flash_fwd_attn_kernel"),
        _kernel_event(2.6, 0.2, "silu_and_mul_kernel"),
    ]


def test_sentinel_reconstruction_recovers_iterations_and_excludes_markers(tmp_path):
    trace = tmp_path / "rank0.pt.trace.json"
    _write_trace(trace, _three_iteration_sentinel_events())
    parsed = parse_trace(trace)
    assert parsed["iterations"] == [0, 1, 2]
    per_iter = {
        d["iteration"]: sum(len(r["kernels"]) for r in d["ranges"])
        for d in parsed["iteration_details"]
    }
    assert per_iter == {0: 2, 1: 1, 2: 2}
    attributed = set(parsed["kernel_names"].values())
    assert not any(SENTINEL_KERNEL_NAME in n for n in attributed)
    assert "warmup_fill_kernel" not in attributed
    assert attributed == {"hipblaslt_gemm_f16", "flash_fwd_attn_kernel", "silu_and_mul_kernel"}


def test_sentinel_path_produces_nonempty_artifacts_and_reads_gzip(tmp_path):
    # A real torch trace is gzip by default; the reader must handle *.json.gz.
    trace = tmp_path / "rank0.pt.trace.json.gz"
    _write_trace(trace, _three_iteration_sentinel_events(), gzipped=True)
    parsed_out = tmp_path / "parsed.json"
    sequences_out = tmp_path / "kernel_sequences.json"
    assert torch_main(
        ["--trace", str(trace), "--output", str(parsed_out),
         "--sequences-output", str(sequences_out)]
    ) == 0
    rows_path = kernel_rows_path(parsed_out)
    for artifact in (parsed_out, rows_path, sequences_out):
        assert artifact.exists() and artifact.stat().st_size > 0
    table = pq.read_table(rows_path)
    assert table.schema.names == _KERNEL_SCHEMA.names
    for name in _KERNEL_SCHEMA.names:
        assert table.column(name).null_count == 0
    # Five model kernels, the three sentinels excluded.
    assert table.num_rows == 5
    parsed = read_parsed(parsed_out)
    assert parsed["source"] == "torch"
    assert not any(SENTINEL_KERNEL_NAME in n for n in parsed["kernel_names"].values())


def test_sentinel_path_falls_back_to_ordinal_when_grid_absent(tmp_path):
    # A roctracer build that omits grid for HIP kernels -> grid_y 0 on every
    # sentinel; the reconstruction must fall back to dispatch order (0,1,2).
    events = [
        {"ph": "X", "cat": "kernel", "name": _SENTINEL_DISPLAY_NAME, "pid": 0, "tid": 7,
         "ts": start, "dur": 0.02, "args": {"device": 0, "stream": 0}}
        for start in (0.1, 1.1, 2.1)
    ]
    events += [
        _kernel_event(0.3, 0.2, "hipblaslt_gemm_f16"),
        _kernel_event(1.3, 0.2, "flash_fwd_attn_kernel"),
        _kernel_event(2.3, 0.2, "silu_and_mul_kernel"),
    ]
    trace = tmp_path / "rank0.pt.trace.json"
    _write_trace(trace, events)
    parsed = parse_trace(trace)
    assert parsed["iterations"] == [0, 1, 2]


def test_no_iteration_markers_raises(tmp_path):
    trace = tmp_path / "bare.pt.trace.json"
    _write_trace(trace, [*_noise_events(), _kernel_event(0.1, 0.1, "rms_norm_kernel")])
    with pytest.raises(ValueError, match="no .*iteration"):
        parse_trace(trace)


# --- per-rank merge (TP4) ------------------------------------------------------


def test_four_rank_merge_into_one_multidevice_document(tmp_path):
    parsed_paths = []
    seq_paths = []
    for rank in range(4):
        trace = tmp_path / f"glm_dp0_tp{rank}_ep{rank}_rank{rank}.pt.trace.json"
        _write_trace(trace, _three_iteration_sentinel_events())
        parsed_out = tmp_path / f"parsed.rank{rank}.json"
        seq_out = tmp_path / f"kernel_sequences.rank{rank}.json"
        assert torch_main(
            ["--trace", str(trace), "--output", str(parsed_out),
             "--sequences-output", str(seq_out)]
        ) == 0
        parsed_paths.append(str(parsed_out))
        seq_paths.append(str(seq_out))

    merged_out = tmp_path / "parsed.json"
    merged_seq = tmp_path / "kernel_sequences.json"
    assert rocpd_merge.main(
        ["--parsed", *parsed_paths, "--sequences", *seq_paths,
         "--output", str(merged_out), "--sequences-output", str(merged_seq)]
    ) == 0
    merged = read_parsed(merged_out)
    assert merged["source"] == "torch"
    assert merged["tp_size"] == 4
    assert merged["device_ids"] == [0, 1, 2, 3]
    # Every iteration now carries one range per device.
    for detail in merged["iteration_details"]:
        devices = {int(r["device_id"]) for r in detail["ranges"]}
        assert devices == {0, 1, 2, 3}
