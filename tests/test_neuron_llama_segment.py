"""Collective segments of the stock Neuron forward: scope, rule, gates and replay."""

from __future__ import annotations

import glob
import statistics
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from profiling.runners.neuron.parquet_columns import read_columns, read_int_arrays
from profiling.runners.neuron.vllm_forward_work import estimate_segment_work, estimate_work
from profiling.runners.neuron.vllm_segment import (
    capture_calls,
    check_capture_counts,
    validate_segment_shape,
)
from profiling.runners.neuron.vllm_segment_engine import capture_cases, capture_config
from profiling.runners.neuron.vllm_segment_trace import (
    SegmentTraceError,
    analyze_session,
    check_warnings,
    combine_ranks,
    composed_forward_ms,
    homogeneity_verdict,
    layer_homogeneity,
    segment_forward,
    segment_times,
    timing_verdict,
)

BASE = dict(
    phase="decode",
    token_bucket=16,
    max_model_len=512,
    kv_blocks=6782,
    block_size=32,
    tp_size=4,
    dtype="bf16",
    segment="mlp_block",
)
SHAPES = (("prefill", 512), ("decode", 1), ("decode", 16))
SEGMENTS = ("embedding", "attention_block", "mlp_block", "head")
DMA_LOSS = (
    "Notification from block DMA of type DMA were dropped. This can cause corresponding "
    "data to be incorrect."
)


def test_scope_accepts_only_the_twelve_verified_rows():
    """Catches device work on an unverified shape before any gate could reject it."""
    for phase, bucket in SHAPES:
        for segment in SEGMENTS:
            validate_segment_shape(
                **(BASE | dict(phase=phase, token_bucket=bucket, segment=segment))
            )
    for changed in (
        {"segment": "attention"},
        {"segment": "model"},
        {"token_bucket": 8},
        {"max_model_len": 128, "token_bucket": 16},
        {"phase": "prefill", "token_bucket": 2048, "max_model_len": 2048},
        {"kv_blocks": 6783},
        {"block_size": 64},
        {"tp_size": 2},
        {"dtype": "fp16"},
    ):
        with pytest.raises(ValueError):
            validate_segment_shape(**(BASE | changed))


# --- Synthetic collective timelines --------------------------------------------------


def synthetic_forward(phase, embedding=100, attention=None, mlp=None, head=900, start=1000):
    """CcOp rows and bounds of one forward with known segment durations (ns)."""
    attention = attention or [300] * 32
    mlp = mlp or [200] * 32
    reduction = {"decode": "AllReduce", "prefill": "ReduceScatter"}[phase]
    ops, t = [], start
    durations = [embedding] + [x for pair in zip(attention, mlp) for x in pair]
    for index, duration in enumerate(durations):
        if phase == "prefill" and index:  # Sequence-parallel gather opens each sublayer.
            ops.append({"operation": "AllGather", "start_ts": t + 5, "end_ts": t + 15})
        t += duration
        ops.append({"operation": reduction, "start_ts": t - 20, "end_ts": t})
    for k in range({"decode": 3, "prefill": 4}[phase]):
        ops.append(
            {"operation": "AllGather", "start_ts": t + 10 + 50 * k, "end_ts": t + 40 + 50 * k}
        )
    return ops, start, t + head


@pytest.mark.parametrize("phase", ["decode", "prefill"])
def test_rule_recovers_known_segments_and_partitions_the_span(phase):
    """Catches an off-by-one between reductions and the attention/MLP assignment."""
    attention = [300 + i for i in range(32)]
    mlp = [200 - i for i in range(32)]
    ops, start, end = synthetic_forward(phase, 70, attention, mlp, 450)
    forward = segment_forward(list(reversed(ops)), start, end, phase)
    assert forward["embedding_ns"] == 70
    assert forward["attention_ns"] == attention
    assert forward["mlp_ns"] == mlp
    assert forward["head_ns"] == 450
    total = 70 + sum(attention) + sum(mlp) + 450
    assert forward["span_ns"] == total == end - start


@pytest.mark.parametrize(
    "phase,mutate",
    [
        ("decode", lambda ops: ops[1:]),  # A missing reduction shifts every layer.
        ("decode", lambda ops: ops + [dict(ops[-1])]),  # An extra AllGather.
        ("decode", lambda ops: [dict(o, operation="ReduceScatter") for o in ops]),
        ("prefill", lambda ops: [o for o in ops if o["operation"] != "AllGather"]),
        # One sublayer gather moved into the previous interval: same counts, wrong layout.
        (
            "prefill",
            lambda ops: [dict(ops[1], start_ts=ops[0]["start_ts"] - 1)] + ops[:1] + ops[2:],
        ),
    ],
)
def test_structure_failures_are_rejected(phase, mutate):
    """Catches segmenting a forward whose collectives do not follow the layer layout."""
    ops, start, end = synthetic_forward(phase)
    with pytest.raises(SegmentTraceError):
        segment_forward(mutate(ops), start, end, phase)


def test_reductions_outside_the_execution_are_rejected():
    """Catches a head or embedding with zero or negative duration."""
    ops, start, end = synthetic_forward("decode", head=0)
    with pytest.raises(SegmentTraceError):
        segment_forward(ops, start, end, "decode")


def test_homogeneity_tolerates_ten_percent_and_rejects_more():
    """Catches folding a layer stack whose layers are not interchangeable."""
    assert layer_homogeneity([100] * 31 + [110.0])["passed"]
    assert not layer_homogeneity([100] * 31 + [112.0])["passed"]
    stats = layer_homogeneity([100, 200])
    assert stats["max_relative_deviation"] == pytest.approx(1 / 3)


def test_only_dma_loss_is_tolerated():
    """Catches accepting a trace with dropped instruction or event notifications."""
    assert check_warnings([{"message": DMA_LOSS}, {"message": "Missing HLO stats"}]) == [DMA_LOSS]
    for block in ("NC of type INSTRUCTION", "NC of type EVENT", "CC of type CC_CORE_EVENT"):
        with pytest.raises(SegmentTraceError):
            check_warnings([{"message": f"Notification from block {block} were dropped."}])


# --- Synthetic rank sessions written as Parquet --------------------------------------


def write_session(directory: Path, forwards, warnings=(), core_tail_ns=0, graph="a" * 32):
    """Write a rank session; ``forwards`` lists (phase, attention, mlp) per execution."""
    directory.mkdir(parents=True)
    executions, cc_ops, starts, ends, cores = [], [], [], [], []
    t = 0
    for index, (phase, attention, mlp) in enumerate(forwards):
        ops, start, end = synthetic_forward(phase, attention=attention, mlp=mlp, start=t)
        executions.append(
            dict(
                execution_index=index,
                neff_name=f"graph_{graph}.neff",
                execution_start_ts=start,
                execution_end_ts=end,
                num_subgraphs=2,
                row_count=4,
            )
        )
        cc_ops.extend(ops)
        for core in (0, 1):
            tail = core_tail_ns if core == 1 else 0
            starts += [start + 1, end - 30 - tail]
            ends += [start + 20, end - 1 - tail]
            cores += [core, core]
        t = end + 1000
    executions.append(
        dict(
            executions[0],
            execution_index=99,
            neff_name="barrier_x.neff",
            execution_start_ts=t,
            execution_end_ts=t + 10,
        )
    )
    pq.write_table(pa.Table.from_pylist(executions), directory / "ExecutionInfo.parquet")
    pq.write_table(pa.Table.from_pylist(cc_ops), directory / "CcOp.parquet")
    rows = [{"message": m, "category": "Missing Data"} for m in ("Missing HLO stats", *warnings)]
    pq.write_table(pa.Table.from_pylist(rows), directory / "Warning.parquet")
    pq.write_table(
        pa.table({"start_ts": starts, "end_ts": ends, "pcore_idx": cores}),
        directory / "Instruction.parquet",
    )
    return directory


def test_session_segments_every_stock_execution_and_skips_barriers(tmp_path):
    """Catches misreading Parquet bounds/collectives or binding the wrong shape."""
    flat = [300] * 32
    session = analyze_session(
        write_session(tmp_path / "rank0", [("decode", flat, [200] * 32)] * 2, [DMA_LOSS]),
        {"a" * 32: ("decode", 16)},
    )
    assert [f["token_bucket"] for f in session["forwards"]] == [16, 16]
    assert session["skipped_executions"] == ["barrier_x.neff"]
    assert session["tolerated_loss_warnings"] == [DMA_LOSS]
    forward = session["forwards"][0]
    assert forward["mlp_ns"] == [200] * 32 and forward["head_ns"] == 900
    assert forward["core_coverage"]["1"]["span_fraction"] > 0.95


def test_homogeneity_gate_rejects_a_slow_interior_layer_but_reports_edges(tmp_path):
    """Catches a 32x fold over a slow interior layer; systematic edge offsets only report."""
    slow_interior = [200] * 15 + [250] + [200] * 16
    directory = write_session(tmp_path / "rank0", [("prefill", [300] * 32, slow_interior)] * 3)
    verdict = homogeneity_verdict([analyze_session(directory, {"a" * 32: ("prefill", 512)})])
    row = verdict["prefill:512"]
    assert not row["passed"]
    assert row["failures"][0]["block"] == "mlp"
    assert row["failures"][0]["worst_interior_layer"] == 15
    assert row["failures"][0]["worst_interior_offset"] == pytest.approx(250 / (200 + 50 / 32) - 1)

    # Observed compiler-partition edges: slow first attention, fast last MLP.
    attention = [400] + [300] * 31
    mlp = [200] * 31 + [180]
    edges = write_session(tmp_path / "rank1", [("decode", attention, mlp)] * 3)
    row = homogeneity_verdict([analyze_session(edges, {"a" * 32: ("decode", 1)})])["decode:1"]
    assert row["passed"]
    stats = row["ranks"]["rank1"]
    assert stats["attention"]["first_layer_offset"] > 0.3
    assert stats["mlp"]["last_layer_offset"] < -0.09


def test_homogeneity_gate_ignores_one_off_spikes_in_single_forwards(tmp_path):
    """Catches failing the fold on noise: per-layer medians absorb a single slow forward."""
    spike = [300] * 4 + [380] + [300] * 27
    forwards = [("decode", [300] * 32, None)] * 4 + [("decode", spike, None)]
    session = analyze_session(
        write_session(tmp_path / "rank0", forwards), {"a" * 32: ("decode", 16)}
    )
    row = homogeneity_verdict([session])["decode:16"]
    assert row["passed"]
    assert row["ranks"]["rank0"]["attention"]["worst_single_forward_deviation"] > 0.2


def test_session_rejects_a_core_missing_its_tail(tmp_path):
    """Catches the retained failure mode: one core's instructions stop early."""
    directory = write_session(tmp_path / "rank0", [("decode", None, None)], core_tail_ns=5000)
    with pytest.raises(SegmentTraceError, match="lack part"):
        analyze_session(directory, {"a" * 32: ("decode", 16)})


def test_session_rejects_lost_instructions_and_unvalidated_graphs(tmp_path):
    """Catches a trace with dropped notifications or a graph outside the sealed set."""
    lost = "Notification from block NC of type INSTRUCTION were dropped."
    directory = write_session(tmp_path / "lost", [("decode", None, None)], [lost])
    with pytest.raises(SegmentTraceError, match="lost"):
        analyze_session(directory, {"a" * 32: ("decode", 16)})
    directory = write_session(tmp_path / "other", [("decode", None, None)])
    with pytest.raises(SegmentTraceError, match="unvalidated"):
        analyze_session(directory, {"b" * 32: ("decode", 16)})


def _forward(phase, bucket, embedding, attention, mlp, head, start):
    return {
        "phase": phase,
        "token_bucket": bucket,
        "start_ns": start,
        "embedding_ns": embedding,
        "attention_ns": attention,
        "mlp_ns": mlp,
        "head_ns": head,
        "span_ns": embedding + sum(attention) + sum(mlp) + head,
    }


def test_rank_average_keeps_the_exact_partition_and_pairs_forwards_in_order():
    """Catches pairing forwards across ranks by anything but execution order."""
    rank0 = {
        "forwards": [
            _forward("decode", 1, 100, [10] * 32, [20] * 32, 300, 0),
            _forward("decode", 1, 200, [10] * 32, [20] * 32, 300, 10_000),
        ]
    }
    rank1 = {
        "forwards": [
            _forward("decode", 1, 300, [30] * 32, [20] * 32, 500, 0),
            _forward("decode", 1, 400, [30] * 32, [20] * 32, 500, 10_000),
        ]
    }
    combined = combine_ranks([rank0, rank1])[("decode", 1)]
    assert [row["embedding"] for row in combined] == [200e-6, 300e-6]
    assert combined[0]["attention_block"] == pytest.approx(20e-6)
    for row in combined:
        assert composed_forward_ms(row) == pytest.approx(row["span"], rel=1e-12)
    rank1["forwards"].pop()
    with pytest.raises(SegmentTraceError, match="counts"):
        combine_ranks([rank0, rank1])


def test_timing_gate_composes_the_32x_fold_against_the_stock_median():
    """Catches a gate that compares one layer, not the composed forward."""
    times = {
        ("embedding", "decode", 16): [0.1, 0.1],
        ("attention_block", "decode", 16): [1.4, 1.5],
        ("mlp_block", "decode", 16): [0.2, 0.2],
        ("head", "decode", 16): [1.0, 1.2],
    }
    composed = 0.1 + 32 * (1.45 + 0.2) + 1.1
    verdict = timing_verdict(times, {("decode", 16): [composed * 1.04]}, {("decode", 16)})
    assert verdict["decode:16"]["passed"]
    assert verdict["decode:16"]["composed_ms"] == pytest.approx(composed)
    verdict = timing_verdict(times, {("decode", 16): [composed * 1.06]}, {("decode", 16)})
    assert not verdict["decode:16"]["passed"]
    assert set(segment_times({("decode", 16): []})) == {(s, "decode", 16) for s in SEGMENTS}


def test_capture_counts_require_every_traced_prefill_and_decode():
    """Catches a window that silently lost forwards at its edges."""
    combined = {("prefill", 512): [0] * 18, ("decode", 16): [0] * 7, ("decode", 1): [0] * 14}
    calls = [dict(batch=16, bucket=16), dict(batch=1, bucket=1), dict(batch=1, bucket=1)]
    assert check_capture_counts(combined, calls, 512)["decode:1"] == 14
    with pytest.raises(ValueError):
        check_capture_counts(combined | {("decode", 1): [0] * 13}, calls, 512)
    with pytest.raises(ValueError):
        check_capture_counts(combined | {("prefill", 512): [0] * 17}, calls, 512)


def test_capture_plan_and_config_change_only_the_profiler():
    """Catches a capture engine that would compile a different (non-stock) graph."""
    specs = [dict(phase="prefill", token_bucket=512), dict(phase="decode", token_bucket=1)]
    assert capture_calls(specs) == {"1": 2, "16": 1}
    cases = [dict(id="b1-0", bucket=1), dict(id="b1-1", bucket=1), dict(id="b16", bucket=16)]
    selected = capture_cases({"capture_calls": {"16": 1, "1": 2}}, cases)
    assert [case["id"] for case in selected] == ["b1-0", "b1-0", "b16"]
    profile = {
        "model": "/model",
        "max_num_seqs": 16,
        "profiler_config": {"profiler": "cuda"},
        "additional_config": {
            "neuron_config": {"num_seqs_buckets": [1, 16]},
            "neuron_profiler": {"activities": ["system_profile"], "neuron_cores": [0, 1, 2, 3]},
        },
    }
    config = capture_config(profile, Path("/out"), [0, 1, 2, 3])
    assert config["additional_config"]["neuron_profiler"]["activities"] == ["device_profile"]
    assert {k: v for k, v in config.items() if k != "additional_config"} == {
        k: v for k, v in profile.items() if k != "additional_config"
    }
    assert profile["additional_config"]["neuron_profiler"]["activities"] == ["system_profile"]


def test_work_split_is_exhaustive_under_the_32x_fold():
    """Catches double-counting or dropping work across segment boundaries."""
    for phase, bucket in SHAPES:
        whole = estimate_work(phase, bucket, 512)
        parts = {s: estimate_segment_work(s, phase, bucket, 512) for s in SEGMENTS}
        for name in whole:
            folded = (
                parts["embedding"][name]
                + 32 * (parts["attention_block"][name] + parts["mlp_block"][name])
                + parts["head"][name]
            )
            assert folded == whole[name]
        assert parts["embedding"]["contraction_flops_per_rank"] == 0
    with pytest.raises(ValueError):
        estimate_segment_work("model", "decode", 1, 512)


# --- Parquet reader and offline replay on the retained real captures ------------------

CAPTURES = Path("/home/ec2-user/ServingStudio/tmp/trainium-composition")
DECODE16 = "d0b185b8f48f643f8c317a262eb2a6f5"
PREFILL512 = "2ba3924da9e1b1ccb06b7daf7f5bf5a2"


def test_reader_matches_pyarrow_on_pyarrow_written_files(tmp_path):
    """Catches a reader that only works for one writer's encodings."""
    table = pa.table(
        {
            "i": [5, None, -7, 2**40],
            "s": ["AllReduce", None, "AllGather", "AllReduce"],
            "f": [1.5, 2.5, None, 0.0],
        }
    )
    for options in (
        dict(),
        dict(use_dictionary=False, compression="none", data_page_version="2.0"),
    ):
        path = tmp_path / f"t{len(options)}.parquet"
        pq.write_table(table, path, **options)
        assert read_columns(path) == table.to_pydict()


def _real_session(name):
    paths = glob.glob(str(CAPTURES / name / "instruction-parquet/*/"))
    if not paths:
        pytest.skip("retained device captures are not present on this host")
    return Path(paths[0])


@pytest.mark.parametrize("name", ["device-decode-v2", "device-prefill-v2"])
def test_reader_matches_pyarrow_on_real_neuron_explorer_exports(name):
    """Catches a decode error in the columns the gates read from parquet-go output."""
    directory = _real_session(name)
    for table in ("ExecutionInfo", "CcOp", "Warning"):
        expected = pq.read_table(directory / f"{table}.parquet").to_pydict()
        assert read_columns(directory / f"{table}.parquet") == expected
    columns = ["start_ts", "end_ts", "pcore_idx"]
    mine = read_int_arrays(directory / "Instruction.parquet", columns)
    expected = pq.read_table(directory / "Instruction.parquet", columns=columns)
    for column in columns:
        assert np.array_equal(mine[column], expected[column].to_numpy())


def test_offline_replay_reproduces_the_feasibility_decode16_numbers():
    """Catches drift from FEASIBILITY.md on the retained decode16 rank0 capture."""
    session = analyze_session(_real_session("device-decode-v2"), {DECODE16: ("decode", 16)})
    (forward,) = session["forwards"]
    assert forward["span_ns"] == 55_389_190
    assert forward["embedding_ns"] == 93_772
    assert statistics.median(forward["attention_ns"]) / 1e6 == pytest.approx(1.4723, abs=5e-5)
    assert min(forward["attention_ns"]) == 1_470_484 and max(forward["attention_ns"]) == 1_492_858
    assert statistics.median(forward["mlp_ns"]) / 1e6 == pytest.approx(0.2174, abs=5e-5)
    assert min(forward["mlp_ns"]) == 215_092 and max(forward["mlp_ns"]) == 236_192
    assert forward["head_ns"] == 1_148_996
    assert session["tolerated_loss_warnings"] == []


def test_offline_replay_reproduces_the_feasibility_prefill512_numbers():
    """Catches drift on the retained prefill rank0 capture (RS/AG layout, DMA loss only)."""
    session = analyze_session(_real_session("device-prefill-v2"), {PREFILL512: ("prefill", 512)})
    (forward,) = session["forwards"]
    assert forward["span_ns"] == 38_680_475
    assert statistics.fmean(forward["attention_ns"]) / 1e6 == pytest.approx(0.284, abs=5e-4)
    assert statistics.fmean(forward["mlp_ns"]) / 1e6 == pytest.approx(0.866, abs=5e-4)
    assert forward["mlp_homogeneity"]["max_relative_deviation"] < 0.10
    assert session["tolerated_loss_warnings"] == [DMA_LOSS]


def _instructions(starts, ends, cores):
    order = np.argsort(starts, kind="stable")
    rows = {
        "start_ts": np.asarray(starts, dtype=np.int64)[order],
        "end_ts": np.asarray(ends, dtype=np.int64)[order],
        "pcore_idx": np.asarray(cores, dtype=np.int64)[order],
    }
    rows["prefix_max_end"] = np.maximum.accumulate(rows["end_ts"])
    return rows


def test_timestamp_rounding_at_the_execution_end_is_tolerated_but_overlap_is_not():
    from profiling.runners.neuron.vllm_segment_trace import (
        BOUNDARY_TOLERANCE_NS,
        SegmentTraceError,
        boundary_overrun,
        core_coverage,
    )

    # Both cores span the execution; one final notification ends 1 ns late.
    rounded = _instructions([0, 0, 900_000], [999_000, 1_000_001, 1_000_001], [0, 1, 0])
    assert core_coverage(rounded, 0, 1_000_000)["0"]["span_fraction"] > 0.95
    assert boundary_overrun(rounded, 0, 1_000_000) == {"start": 0, "end": 1}
    overlap = _instructions(
        [0, 0, 900_000], [999_000, 999_000, 1_000_000 + BOUNDARY_TOLERANCE_NS + 1], [0, 1, 0]
    )
    with pytest.raises(SegmentTraceError, match="crosses the execution end"):
        core_coverage(overlap, 0, 1_000_000)
