"""Measured per-operation attribution: conservation and causal ownership tests."""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from profiling.profilers.neuron_ops import (
    NeuronOpsTraceError,
    _execution_roles,
    _static_labels,
    analyze_capture,
    interval_accounting,
)
from profiling.profilers.neuron_ops_dependencies import observed_dependencies
from profiling.profilers.neuron_ops_labels import SOURCE_HASHES, source_operation


def instruction(identifier, operation, start, end, **extra):
    return (
        dict(
            id=identifier,
            pcore_idx=0,
            engine="GpSimd",
            timeline_track="GpSimd",
            execution_unit="compute",
            compiler_pc=identifier,
            pc=identifier,
            start_ts=start,
            end_ts=end,
            opcode="NOP",
            operands="",
            nki_source_location=None,
            instruction_chain=None,
            operation=operation,
        )
        | extra
    )


def update(timestamp, value, source="AXI", semaphore=7, kind="sem_inc"):
    return dict(
        timestamp=timestamp,
        id=str(semaphore),
        value=value,
        update_source=source,
        update_operation_type=kind,
        pcore_idx=0,
    )


def test_overlaps_partition_elapsed_without_summing_engine_busy_time():
    rows = [
        instruction(1, "alpha", 0, 60),
        instruction(2, "beta", 40, 100),
        instruction(3, "alpha", 10, 50),
        instruction(4, "ignored", 0, 120, execution_unit="sequencer"),
    ]
    result = interval_accounting(rows, 0, 120)
    assert result == dict(
        active_ns=100,
        idle_ns=20,
        shared_ns=20,
        exclusive_ns={"alpha": 40, "beta": 40},
        per_op_active_ns={"alpha": 60, "beta": 60},
        elapsed_ns=120,
    )


def test_single_outstanding_dma_and_semaphore_reset_keep_correct_owner():
    rows = [
        instruction(
            1, "first", 1, 10, opcode="DMA_DIRECT2D", operands="semaphore=7 sem_increment=16"
        ),
        instruction(
            2, "second", 51, 55, opcode="DMA_DIRECT2D", operands="semaphore=7 sem_increment=16"
        ),
        instruction(3, "consumer", 60, 61, operands="S[7] (qGpSimdDynamic)>=16"),
        instruction(
            4, "third", 101, 110, opcode="DMA_DIRECT2D", operands="semaphore=7 sem_increment=16"
        ),
        instruction(5, "consumer", 170, 171, operands="S[7] (qGpSimdDynamic)>=16"),
    ]
    updates = [
        update(0, 0, "Tensor", kind="sem_wr_imm"),
        update(50, 16),
        update(80, 32),
        update(100, 0, "Tensor", kind="sem_wr_imm"),
        update(160, 16),
    ]
    result = observed_dependencies(rows, updates)
    assert [(row["operation"], row["completion_ts"]) for row in result["transfers"]] == [
        ("first", 50),
        ("second", 80),
        ("third", 160),
    ]
    assert {row["evidence"] for row in result["transfers"]} == {
        "counter_inferred_single_outstanding"
    }


def test_overlapping_same_semaphore_triggers_do_not_claim_individual_completion():
    rows = [
        instruction(
            1, "first", 1, 10, opcode="DMA_DIRECT2D", operands="semaphore=7 sem_increment=16"
        ),
        instruction(
            2, "second", 11, 20, opcode="DMA_DIRECT2D", operands="semaphore=7 sem_increment=16"
        ),
    ]
    result = observed_dependencies(
        rows, [update(0, 0, "Tensor", kind="sem_wr_imm"), update(50, 16), update(80, 32)]
    )
    assert [row["counter_crossing_ts"] for row in result["transfers"]] == [50, 80]
    assert all(
        row["completion_ts"] is None and row["latency_ns"] is None for row in result["transfers"]
    )
    assert {row["evidence"] for row in result["transfers"]} == {"ambiguous_counter_prefix"}


def test_exported_event_before_preceding_dma_recovers_tensor_wait():
    rows = [
        instruction(10, "glue", 0, 10, pc=0),
        instruction(11, "mlp", 100, 101, pc=2, opcode="EVENT_SEMAPHORE", operands="S[7]>=16"),
        instruction(
            12,
            "mlp",
            118,
            130,
            pc=1,
            opcode="DMA_DIRECT2D",
            operands="S[5] (Tensor)>=1 semaphore=7 sem_increment=16",
        ),
        instruction(13, "mlp", 0, 90, engine="Tensor", timeline_track="TensorMatrix"),
    ]
    updates = [update(5, 16), update(90, 1, "Tensor", semaphore=5)]
    gaps = observed_dependencies(rows, updates)["gaps"]
    gap = next(row for row in gaps if row["waiter_id"] == 11)
    assert gap["match"] == "adjacent_dma_timestamp_inversion"
    assert gap["source_id"] == 13
    assert gap["source_operation"] == "mlp"
    assert gap["explained_ns"] == 80
    assert gap["residual_ns"] == 10


def test_static_label_sandwich_stops_at_function_and_ambiguous_parent(tmp_path):
    data = {"instructions": []}
    for pc, function, kernels in [
        (0, "layer", ["qkv"]),
        (1, "layer", []),
        (2, "layer", ["qkv"]),
        (3, "head", []),
        (4, "head", ["qkv", "mlp_lnc"]),
        (5, "head", ["qkv"]),
        (6, "head", []),
        (7, "head", []),
        (8, "head", ["qkv"]),
    ]:
        data["instructions"].append(
            dict(
                pcore_idx=0,
                engine="Tensor",
                compiler_pc=pc,
                function=function,
                opcode_byte=211 if pc == 6 else 1,
                parents=[{"kernel_name": kernel} for kernel in kernels],
            )
        )
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(data))
    labels = _static_labels(path)
    assert labels[0, "Tensor", 1]["operation"] == "qkv_projection"
    assert labels[0, "Tensor", 1]["evidence"] == "same_kernel_sandwich"
    assert labels[0, "Tensor", 3]["operation"] is None
    assert labels[0, "Tensor", 4]["operation"] is None
    assert labels[0, "Tensor", 7]["operation"] is None


def test_changed_function_debug_transfer_requires_large_homogeneous_context(tmp_path):
    row = dict(
        pcore_idx=0,
        engine="Tensor",
        compiler_pc=0,
        function="head",
        opcode_byte=1,
        instruction_sha256="a" * 64,
        parents=[],
        matched_debug_parents=[{"kernel_name": "qkv"}],
        debug_match=dict(
            method="unique_binary_context",
            instruction_sha256="a" * 64,
            debug_neff_sha256="b" * 64,
            label_confidence="inferred_context",
            context_radius=0,
            homogeneous_context_owner="qkv_projection",
        ),
    )
    path = tmp_path / "mapped.json"
    path.write_text(json.dumps({"instructions": [row]}))
    with pytest.raises(NeuronOpsTraceError, match="strong semantic context"):
        _static_labels(path)
    row["debug_match"]["context_radius"] = 8
    path.write_text(json.dumps({"instructions": [row]}))
    assert _static_labels(path)[0, "Tensor", 0]["evidence"] == "inferred_context_debug_transfer"
    row["debug_match"].update(
        method="identical_function_stream",
        label_confidence="exact_function",
        stock_function_stream_sha256="c" * 64,
        debug_function_stream_sha256="c" * 64,
    )
    path.write_text(json.dumps({"instructions": [row]}))
    assert _static_labels(path)[0, "Tensor", 0]["evidence"] == "exact_function_debug_transfer"


@pytest.mark.parametrize(
    "stock,debug,operation,evidence",
    [
        (["qkv", None], [], None, "unresolved"),
        (["qkv", "missing"], [], None, "unresolved"),
        ([None], ["qkv", None], None, "unresolved"),
        ([None], ["qkv", "missing"], None, "unresolved"),
        ([None], ["qkv", "qkv"], "qkv_projection", "exact_function_debug_transfer"),
        (["qkv", None], ["qkv"], "qkv_projection", "exact_function_debug_transfer"),
        (["qkv"], ["qkv", None], "qkv_projection", "static_debug"),
        (["mlp_lnc"], ["qkv"], None, "unresolved"),
    ],
)
def test_incomplete_parent_sets_cannot_claim_unanimous_labels(
    tmp_path, stock, debug, operation, evidence
):
    def parents(values):
        return [
            {"id": index, "missing": True}
            if value == "missing"
            else {"id": index, "kernel_name": value}
            for index, value in enumerate(values)
        ]

    def row(pc, candidates):
        return dict(
            pcore_idx=0,
            engine="Tensor",
            compiler_pc=pc,
            function="layer",
            opcode_byte=1,
            instruction_sha256="a" * 64,
            parents=parents(candidates),
        )

    target = row(1, stock)
    if debug:
        target["matched_debug_parents"] = parents(debug)
        target["debug_match"] = dict(
            method="identical_function_stream",
            instruction_sha256="a" * 64,
            debug_neff_sha256="b" * 64,
            label_confidence="exact_function",
            stock_function_stream_sha256="c" * 64,
            debug_function_stream_sha256="c" * 64,
        )
    path = tmp_path / "parents.json"
    # Known neighbors must not promote ambiguous partial evidence via sandwich.
    path.write_text(json.dumps({"instructions": [row(0, ["qkv"]), target, row(2, ["qkv"])]}))
    actual = _static_labels(path)[0, "Tensor", 1]
    assert actual["operation"] == operation
    assert actual["evidence"] == evidence
    assert actual["ambiguous"] is (operation is None)


def capture_fixture(tmp_path, warning=None):
    session = tmp_path / "session"
    session.mkdir()
    executions, rows, neffs = [], [], []
    for index, phase, bucket in [(0, "prefill", 512), (1, "decode", 16)]:
        start = index * 1000
        executions.append(
            dict(
                execution_index=index,
                execution_start_ts=start,
                execution_end_ts=start + 100,
                neff_name=f"graph_{phase}.neff",
            )
        )
        for core in (0, 1):
            rows.extend(
                [
                    instruction(
                        index * 10,
                        "unused",
                        start,
                        start + 60,
                        pcore_idx=core,
                        engine="Tensor",
                        timeline_track="TensorMatrix",
                        compiler_pc=0,
                    ),
                    instruction(
                        index * 10 + 1,
                        "unused",
                        start + 40,
                        start + 100,
                        pcore_idx=core,
                        engine="Vector",
                        timeline_track="Vector",
                        compiler_pc=0,
                    ),
                ]
            )
        static = {
            f"sg{core:02}": {
                "PE": {"instr": [[1, "head" if index else "prologue", "", "alpha", ""]]},
                "DVE": {"instr": [[1, "head" if index else "prologue", "", "beta", ""]]},
            }
            for core in (0, 1)
        }
        binary = tmp_path / f"graph_{phase}.neff"
        binary.write_bytes(f"synthetic binary {phase}".encode())
        binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
        static["neff_sha256"] = binary_hash
        path = tmp_path / f"{phase}.json"
        path.write_text(json.dumps(static))
        neffs.append(
            dict(
                graph_id=phase,
                phase=phase,
                token_bucket=bucket,
                path=str(binary),
                sha256=binary_hash,
                static_instructions=str(path),
                static_instructions_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        )
    pq.write_table(pa.Table.from_pylist(rows), session / "Instruction.parquet")
    pq.write_table(pa.Table.from_pylist(executions), session / "ExecutionInfo.parquet")
    if warning:
        pq.write_table(pa.Table.from_pylist([{"message": warning}]), session / "Warning.parquet")
    manifest = dict(
        schema_version=1,
        shape={},
        hardware={"name": "AWS Trainium2", "physical_cores_per_rank": [0, 1]},
        sessions=[dict(rank=0, session_dir=str(session), neffs=neffs)],
    )
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_all_executions_prologue_head_and_per_core_conservation(tmp_path):
    manifest = capture_fixture(tmp_path)
    result = analyze_capture(manifest, tmp_path / "out")
    summary = result["summary"]
    assert summary["runtime_ms"] == {"median": 0.0001, "min": 0.0001, "max": 0.0001}
    assert len(summary["executions"]) == 4
    assert {row["phase"] for row in summary["per_op"]} == {"prefill", "decode"}
    assert all(row["dma_completion_latency_us"] is None for row in summary["per_op"])
    assert all(row["population"] == row["samples"] == 2 for row in summary["per_op"])
    for row in summary["executions"]:
        assert row["exclusive_us"] + row["shared_us"] + row["idle_us"] == pytest.approx(
            row["runtime_ms"] * 1000
        )
    attributed = pq.read_table(tmp_path / "out" / "instruction_attribution.parquet").to_pylist()
    assert len(attributed) == 8
    assert {row["function"] for row in attributed} == {"prologue", "head"}
    assert summary["quality"]["unseen_shape_prediction_supported"] is False
    assert summary["quality"]["labeled_active_fraction"] == 1
    assert summary["quality"]["per_engine_active_coverage"]["TensorMatrix"]["active_ns"] == 240
    assert (
        summary["quality"]["per_engine_active_coverage"]["TensorMatrix"]["labeled_active_fraction"]
        == 1
    )
    assert {row["execution_unit"] for row in attributed} == {"compute"}
    assert len({row["id"] for row in attributed}) < len(
        attributed
    )  # valid cross-core ID collisions


def selected_prefill_fixture(tmp_path):
    path = capture_fixture(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["shape"] = {"phase": "prefill", "token_bucket": 512}
    manifest["sessions"][0]["execution_selection"] = {
        "policy": "prefill_with_trailing_decode_guard_v1",
        "selected_indices": [0],
        "guard_indices": [1],
    }
    path.write_text(json.dumps(manifest))
    return path


def test_guard_retained_and_validated_but_excluded_from_reported_aggregates(tmp_path):
    path = selected_prefill_fixture(tmp_path)
    # Give the guard a distinct duration and a DMA trigger. Its work must stay
    # auditable without changing the selected prefill's runtime or DMA totals.
    raw = tmp_path / "session" / "Instruction.parquet"
    rows = pq.read_table(raw).to_pylist()
    for row in rows:
        if row["start_ts"] >= 1000:
            row["end_ts"] += 100
            if row["start_ts"] == 1000:
                row["operands"] = "semaphore=7 sem_increment=16"
    pq.write_table(pa.Table.from_pylist(rows), raw)
    execution_path = tmp_path / "session" / "ExecutionInfo.parquet"
    executions = pq.read_table(execution_path).to_pylist()
    executions[1]["execution_end_ts"] += 100
    pq.write_table(pa.Table.from_pylist(executions), execution_path)
    result = analyze_capture(path, tmp_path / "out")
    summary, quality = result["summary"], result["summary"]["quality"]
    assert summary["runtime_ms"] == {"median": 0.0001, "min": 0.0001, "max": 0.0001}
    assert {row["phase"] for row in summary["per_op"]} == {"prefill"}
    assert quality["input_instruction_count"] == 8
    assert quality["attributed_instruction_count"] == 4
    assert quality["excluded_execution_instruction_count"] == 4
    assert quality["excluded_control_spill_count"] == 0
    assert quality["active_instruction_ns"] == 200
    assert quality["per_engine_active_coverage"]["TensorMatrix"]["active_ns"] == 120
    assert sum(quality["instruction_label_counts"].values()) == 4
    assert quality["dma_triggers"] == 0
    attributed = pq.read_table(tmp_path / "out" / "instruction_attribution.parquet").to_pylist()
    assert len(attributed) == 8
    assert {row["execution_role"] for row in attributed} == {"selected", "guard"}
    dma = pq.read_table(tmp_path / "out" / "dma_transfers.parquet").to_pylist()
    assert len(dma) == 2
    assert {row["execution_role"] for row in dma} == {"guard"}
    guard = next(row for row in summary["execution_selection"] if row["execution_role"] == "guard")
    assert guard["execution_index"] == 1
    assert guard["phase"] == "decode" and guard["token_bucket"] == 16
    assert guard["neff_name"] == "graph_decode.neff"
    assert len(guard["neff_sha256"]) == 64
    assert guard["per_core_instruction_counts"] == {0: 2, 1: 2}
    assert guard["exclusion_reason"] == "trailing_decode_guard"
    assert str(tmp_path / "out" / "execution_selection.json") in result["artifacts"]


@pytest.mark.parametrize("guard_phase", ["prefill", "decode"])
def test_native_guard_v2_reuses_same_prefill_or_accepts_decode(tmp_path, guard_phase):
    path = selected_prefill_fixture(tmp_path)
    manifest = json.loads(path.read_text())
    session = manifest["sessions"][0]
    session["execution_selection"]["policy"] = "prefill_with_trailing_native_guard_v2"
    if guard_phase == "prefill":
        raw = tmp_path / "session" / "ExecutionInfo.parquet"
        rows = pq.read_table(raw).to_pylist()
        rows[1]["neff_name"] = rows[0]["neff_name"]
        pq.write_table(pa.Table.from_pylist(rows), raw)
    path.write_text(json.dumps(manifest))
    summary = analyze_capture(path, tmp_path / "out")["summary"]
    guard = next(row for row in summary["execution_selection"] if row["execution_role"] == "guard")
    assert guard["phase"] == guard_phase
    assert guard["exclusion_reason"] == "trailing_native_guard"
    assert summary["quality"]["attributed_instruction_count"] == 4
    assert summary["quality"]["excluded_execution_instruction_count"] == 4
    assert {row["phase"] for row in summary["per_op"]} == {"prefill"}
    assert all(row["samples"] == row["population"] == 2 for row in summary["per_op"])


def decode_tail_fixture(tmp_path, pattern=("prefill", "prefill", "decode")):
    path = selected_prefill_fixture(tmp_path)
    manifest = json.loads(path.read_text())
    raw = tmp_path / "session" / "Instruction.parquet"
    rows = pq.read_table(raw).to_pylist()
    rows.extend(
        [
            row
            | {
                "id": row["id"] + 100,
                "start_ts": row["start_ts"] + 1000,
                "end_ts": row["end_ts"] + 1000,
            }
            for row in rows
            if row["start_ts"] >= 1000
        ]
    )
    indices = [7, 9, 15]
    selected = 1 if pattern[1] == "prefill" else 0
    executions = [
        dict(
            execution_index=indices[i],
            execution_start_ts=i * 1000,
            execution_end_ts=i * 1000 + (100 if i == selected else 200),
            neff_name=f"graph_{phase}.neff",
        )
        for i, phase in enumerate(pattern)
    ]
    for row in rows:
        if row["start_ts"] // 1000 != selected:
            row["end_ts"] += 100
            if row["start_ts"] % 1000 == 0:
                row["operands"] = "semaphore=7 sem_increment=16"
    pq.write_table(pa.Table.from_pylist(rows), raw)
    pq.write_table(pa.Table.from_pylist(executions), tmp_path / "session" / "ExecutionInfo.parquet")
    manifest["sessions"][0]["execution_selection"] = {
        "policy": "prefill_with_decode_tail_v3",
        "selected_indices": [indices[selected]],
        "guard_indices": [index for i, index in enumerate(indices) if i != selected],
    }
    path.write_text(json.dumps(manifest))
    return path


@pytest.mark.parametrize(
    "pattern", [("prefill", "prefill", "decode"), ("prefill", "decode", "decode")]
)
def test_decode_tail_v3_selects_last_prefill_and_excludes_both_guards(tmp_path, pattern):
    path = decode_tail_fixture(tmp_path, pattern)
    raw = tmp_path / "session" / "ExecutionInfo.parquet"
    pq.write_table(pa.Table.from_pylist(pq.read_table(raw).to_pylist()[::-1]), raw)
    summary = analyze_capture(path, tmp_path / "out")["summary"]
    quality = summary["quality"]
    selected = next(
        row for row in summary["execution_selection"] if row["execution_role"] == "selected"
    )
    assert selected["execution_index"] == (9 if pattern[1] == "prefill" else 7)
    assert selected["guard_position"] is None
    guards = [row for row in summary["execution_selection"] if row["execution_role"] == "guard"]
    assert len(guards) == 2
    assert {row["exclusion_reason"] for row in guards} == {"decode_tail_guard"}
    assert sum(row["guard_position"] == "before_selected" for row in guards) == (
        pattern[1] == "prefill"
    )
    assert summary["runtime_ms"] == {"median": 0.0001, "min": 0.0001, "max": 0.0001}
    assert quality["input_instruction_count"] == 12
    assert quality["attributed_instruction_count"] == 4
    assert quality["excluded_execution_instruction_count"] == 8
    assert quality["dma_triggers"] == 0
    assert quality["active_instruction_ns"] == 200
    assert sum(quality["instruction_label_counts"].values()) == 4
    assert all(
        row["phase"] == "prefill" and row["samples"] == row["population"] == 2
        for row in summary["per_op"]
    )
    transfers = pq.read_table(tmp_path / "out" / "dma_transfers.parquet").to_pylist()
    assert len(transfers) == 4 and all(row["execution_role"] == "guard" for row in transfers)


@pytest.mark.parametrize("defect", ["count", "selected", "duplicate_guard", "DDD", "PPP", "PDP"])
def test_decode_tail_v3_rejects_wrong_pattern_partition_or_selected_index(tmp_path, defect):
    path = decode_tail_fixture(tmp_path)
    manifest = json.loads(path.read_text())
    selection = manifest["sessions"][0]["execution_selection"]
    raw = tmp_path / "session" / "ExecutionInfo.parquet"
    rows = pq.read_table(raw).to_pylist()
    if defect == "count":
        rows.pop()
    elif defect == "selected":
        selection.update(selected_indices=[7], guard_indices=[9, 15])
    elif defect == "duplicate_guard":
        selection["guard_indices"] = [7, 7]
    else:
        for row, phase in zip(rows, defect):
            row["neff_name"] = f"graph_{'prefill' if phase == 'P' else 'decode'}.neff"
    pq.write_table(pa.Table.from_pylist(rows), raw)
    path.write_text(json.dumps(manifest))
    with pytest.raises(NeuronOpsTraceError):
        analyze_capture(path, tmp_path / "bad")


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("defect", ["graph_id", "sha256", "token_bucket"])
def test_decode_tail_v3_requires_identical_repeated_compilation(phase, defect):
    phases = (
        ["prefill", "prefill", "decode"] if phase == "prefill" else ["prefill", "decode", "decode"]
    )
    neffs = [
        dict(
            graph_id=p,
            sha256=("a" if p == "prefill" else "b") * 64,
            phase=p,
            token_bucket=512 if p == "prefill" else 16,
            path=f"capture{i}.neff",
        )
        for i, p in enumerate(phases)
    ]
    target = 1 if phase == "prefill" else 2
    neffs[target][defect] = {
        "graph_id": "different_graph",
        "sha256": "c" * 64,
        "token_bucket": 1024,
    }[defect]
    chosen = 1 if phase == "prefill" else 0
    session = {
        "neffs": neffs,
        "execution_selection": {
            "policy": "prefill_with_decode_tail_v3",
            "selected_indices": [chosen],
            "guard_indices": [i for i in range(3) if i != chosen],
        },
    }
    executions = [
        dict(
            execution_index=i,
            neff_name=f"capture{i}.neff",
            execution_start_ts=i * 1000,
            execution_end_ts=i * 1000 + 100,
        )
        for i in range(3)
    ]
    with pytest.raises(NeuronOpsTraceError, match="identical graph"):
        _execution_roles(session, executions, {"phase": "prefill", "token_bucket": 512})


@pytest.mark.parametrize("defect", ["graph_id", "sha256", "token_bucket", "v1"])
def test_same_prefill_guard_requires_identical_compilation_and_v2(defect):
    # Distinct capture filenames may refer to the same declared compilation.
    # All three sealed identity fields must agree before accepting that claim.
    first = dict(
        graph_id="compiled_prefill",
        sha256="a" * 64,
        phase="prefill",
        token_bucket=512,
        path="first.neff",
    )
    guard = first | {"path": "second.neff"}
    policy = "prefill_with_trailing_native_guard_v2"
    if defect == "v1":
        policy = "prefill_with_trailing_decode_guard_v1"
    else:
        guard[defect] = {"graph_id": "other_prefill", "sha256": "b" * 64, "token_bucket": 1024}[
            defect
        ]
    session = {
        "neffs": [first, guard],
        "execution_selection": {"policy": policy, "selected_indices": [7], "guard_indices": [9]},
    }
    executions = [
        dict(execution_index=7, neff_name="first.neff", execution_start_ts=0, execution_end_ts=100),
        dict(
            execution_index=9, neff_name="second.neff", execution_start_ts=200, execution_end_ts=300
        ),
    ]
    with pytest.raises(NeuronOpsTraceError, match="geometry"):
        _execution_roles(session, executions, {"phase": "prefill", "token_bucket": 512})


@pytest.mark.parametrize("policy", ["legacy", "v1", "v2"])
@pytest.mark.parametrize("completion_ts", [1000, 1020])
def test_dependency_halo_cannot_borrow_adjacent_execution_writers(tmp_path, policy, completion_ts):
    path = capture_fixture(tmp_path) if policy == "legacy" else selected_prefill_fixture(tmp_path)
    if policy == "v2":
        manifest = json.loads(path.read_text())
        manifest["sessions"][0]["execution_selection"]["policy"] = (
            "prefill_with_trailing_native_guard_v2"
        )
        path.write_text(json.dumps(manifest))
        execution_path = tmp_path / "session" / "ExecutionInfo.parquet"
        executions = pq.read_table(execution_path).to_pylist()
        executions[1]["neff_name"] = executions[0]["neff_name"]
        pq.write_table(pa.Table.from_pylist(executions), execution_path)
    raw = tmp_path / "session" / "Instruction.parquet"
    rows = pq.read_table(raw).to_pylist()
    for index, start in enumerate([90, 1010]):
        rows.append(
            instruction(
                99 + index,
                "unused",
                start,
                start + 5,
                compiler_pc=99,
                operands="semaphore=7 sem_increment=16",
            )
        )
    pq.write_table(pa.Table.from_pylist(rows), raw)
    # The next execution also triggers this semaphore before the AXI increment.
    # Importing that increment without its trigger invents a single-outstanding
    # completion for the earlier execution. A shared boundary is excluded too.
    updates = [
        dict(
            pcore_idx=0,
            id="S7",
            timestamp=0,
            value=0,
            update_operation_type="sem_wr_imm",
            update_source="GpSimd",
        ),
        dict(
            pcore_idx=0,
            id="S7",
            timestamp=completion_ts,
            value=16,
            update_operation_type="sem_inc",
            update_source="AXI",
        ),
    ]
    pq.write_table(pa.Table.from_pylist(updates), tmp_path / "session" / "SemaphoreUpdate.parquet")
    analyze_capture(path, tmp_path / "out")
    transfers = pq.read_table(tmp_path / "out" / "dma_transfers.parquet").to_pylist()
    first = next(row for row in transfers if row["execution_index"] == 0)
    assert first["completion_ts"] is None
    assert first["counter_crossing_ts"] is None
    assert first["evidence"] == "unresolved_completion"
    second = next(row for row in transfers if row["execution_index"] == 1)
    assert second["reset_ts"] is None  # Reset belongs to the previous execution.
    assert second["completion_ts"] is None


@pytest.mark.parametrize("defect", ["loss", "negative", "missing_core", "binary"])
def test_guard_cannot_hide_global_loss_or_invalid_execution(tmp_path, defect):
    path = selected_prefill_fixture(tmp_path)
    if defect == "loss":
        pq.write_table(
            pa.Table.from_pylist([{"message": "DMA notifications dropped in guard"}]),
            tmp_path / "session" / "Warning.parquet",
        )
    elif defect == "binary":
        (tmp_path / "graph_decode.neff").write_bytes(b"wrong guard binary")
    else:
        raw = tmp_path / "session" / "Instruction.parquet"
        rows = pq.read_table(raw).to_pylist()
        if defect == "negative":
            rows[-1]["end_ts"] = rows[-1]["start_ts"] - 1
        else:
            rows = [row for row in rows if not (row["start_ts"] >= 1000 and row["pcore_idx"] == 1)]
        pq.write_table(pa.Table.from_pylist(rows), raw)
    with pytest.raises(NeuronOpsTraceError):
        analyze_capture(path, tmp_path / "bad")


@pytest.mark.parametrize(
    "defect",
    ["overlap", "missing", "unknown", "wrong_phase", "wrong_bucket", "guard_bucket", "wrong_order"],
)
def test_execution_selection_is_a_strict_partition_with_fixed_phase_order(tmp_path, defect):
    path = selected_prefill_fixture(tmp_path)
    manifest = json.loads(path.read_text())
    session = manifest["sessions"][0]
    selection = session["execution_selection"]
    if defect == "overlap":
        selection["guard_indices"] = [0]
    elif defect == "missing":
        selection["guard_indices"] = []
    elif defect == "unknown":
        selection["guard_indices"] = [2]
    elif defect == "wrong_phase":
        session["neffs"][1]["phase"] = "prefill"
    elif defect == "wrong_bucket":
        manifest["shape"]["token_bucket"] = 1024
    elif defect == "guard_bucket":
        session["neffs"][1]["token_bucket"] = 1
    else:
        raw = tmp_path / "session" / "ExecutionInfo.parquet"
        rows = pq.read_table(raw).to_pylist()
        rows[1]["execution_start_ts"] = 50
        pq.write_table(pa.Table.from_pylist(rows), raw)
    path.write_text(json.dumps(manifest))
    with pytest.raises(NeuronOpsTraceError):
        analyze_capture(path, tmp_path / "bad_selection")


@pytest.mark.parametrize("shape", [{"phase": "prefill"}, {"token_bucket": 512}])
def test_mixed_executions_without_selection_reject_declared_scope(tmp_path, shape):
    path = selected_prefill_fixture(tmp_path)
    manifest = json.loads(path.read_text())
    del manifest["sessions"][0]["execution_selection"]
    manifest["shape"] = shape
    path.write_text(json.dumps(manifest))
    with pytest.raises(NeuronOpsTraceError, match="requested geometry"):
        analyze_capture(path, tmp_path / "bad_scope")


def test_dma_packet_loss_rejected_and_map_hash_validated(tmp_path):
    path = capture_fixture(tmp_path, "Notification from block DMA of type DMA were dropped.")
    with pytest.raises(NeuronOpsTraceError, match="Incomplete"):
        analyze_capture(path, tmp_path / "bad")
    (tmp_path / "session" / "Warning.parquet").unlink()
    manifest = json.loads(path.read_text())
    manifest["sessions"][0]["neffs"][0]["static_instructions_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(NeuronOpsTraceError, match="hash"):
        analyze_capture(path, tmp_path / "bad_hash")


def test_all_numbered_table_shards_are_read_and_checked(tmp_path):
    path = capture_fixture(tmp_path)
    expected = analyze_capture(path, tmp_path / "original")["summary"]
    directory = tmp_path / "session"
    # Both base-plus-shards and shards-only are valid Explorer export layouts.
    for name, suffixes in [("Instruction", ("", "_1")), ("ExecutionInfo", ("_0", "_1"))]:
        base = directory / f"{name}.parquet"
        rows = pq.read_table(base).to_pylist()
        base.unlink()
        for index, suffix in enumerate(suffixes):
            pq.write_table(
                pa.Table.from_pylist(rows[index::2]), directory / f"{name}{suffix}.parquet"
            )
    for index, timestamp in enumerate((0, 1000)):
        pq.write_table(
            pa.Table.from_pylist([update(timestamp, 0, kind="sem_wr_imm")]),
            directory / f"SemaphoreUpdate_{index}.parquet",
        )
    pq.write_table(
        pa.Table.from_pylist([{"message": "missing optional metrics"}]),
        directory / "Warning_0.parquet",
    )
    actual = analyze_capture(path, tmp_path / "split")["summary"]
    assert actual["executions"] == expected["executions"]
    assert actual["per_op"] == expected["per_op"]
    assert actual["quality"]["input_instruction_count"] == 8
    assert actual["quality"]["missing_optional_tables"] == []
    assert actual["quality"]["warnings"] == ["missing optional metrics"]
    pq.write_table(
        pa.Table.from_pylist([{"message": "DMA notifications were dropped"}]),
        directory / "Warning_1.parquet",
    )
    with pytest.raises(NeuronOpsTraceError, match="Incomplete"):
        analyze_capture(path, tmp_path / "split_loss")


def test_summary_exposes_partial_dma_and_conditional_population(tmp_path):
    path = capture_fixture(tmp_path)
    raw = tmp_path / "session" / "Instruction.parquet"
    rows = pq.read_table(raw).to_pylist()
    for row in rows:
        if row["start_ts"] == 0:
            row["operands"] = "semaphore=7 sem_increment=16"
    pq.write_table(pa.Table.from_pylist(rows), raw)
    pq.write_table(
        pa.Table.from_pylist([update(0, 0, kind="sem_wr_imm"), update(65, 16)]),
        tmp_path / "session" / "SemaphoreUpdate.parquet",
    )
    manifest = json.loads(path.read_text())
    neff = manifest["sessions"][0]["neffs"][0]
    static_path = Path(neff["static_instructions"])
    static = json.loads(static_path.read_text())
    static["sg01"]["DVE"]["instr"][0][3] = ""
    static_path.write_text(json.dumps(static))
    neff["static_instructions_sha256"] = hashlib.sha256(static_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    summary = analyze_capture(path, tmp_path / "out")["summary"]
    prefill = {row["operation"]: row for row in summary["per_op"] if row["phase"] == "prefill"}
    assert prefill["alpha"]["dma_completion_fraction"] == 0.5
    assert prefill["alpha"]["dma_completion_count"] == 0.5
    assert prefill["alpha"]["dma_completion_latency_us"] == 0.005
    assert prefill["beta"]["samples"] == 1
    assert prefill["beta"]["population"] == 2


def test_compiler_primitive_measurements_do_not_inflate_model_label_coverage(tmp_path):
    path = capture_fixture(tmp_path)
    manifest = json.loads(path.read_text())
    for neff in manifest["sessions"][0]["neffs"]:
        static_path = Path(neff["static_instructions"])
        rows = []
        for core in (0, 1):
            for engine, kernel in (("Tensor", ""), ("Vector", "beta")):
                rows.append(
                    dict(
                        pcore_idx=core,
                        engine=engine,
                        compiler_pc=0,
                        function="layer",
                        opcode_byte=1,
                        parents=[{"kernel_name": kernel, "instruction_type": "TRANSPOSE"}],
                        compiler_operation="transpose",
                        compiler_operation_evidence="unanimous_backend_instruction_type",
                    )
                )
        static_path.write_text(json.dumps({"neff_sha256": neff["sha256"], "instructions": rows}))
        neff["static_instructions_sha256"] = hashlib.sha256(static_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    summary = analyze_capture(path, tmp_path / "out")["summary"]
    q = summary["quality"]
    assert q["labeled_active_fraction"] == 0.6
    assert q["compiler_primitive_active_fraction"] == 0.6
    assert q["unresolved_active_fraction"] == 0
    tensor = q["per_engine_active_coverage"]["TensorMatrix"]
    assert tensor["model_labeled_active_ns"] == 0
    assert tensor["compiler_primitive_active_ns"] == 240
    assert tensor["compiler_primitive_active_fraction"] == 1
    for row in summary["per_op"]:
        if row["operation"] == "compiler_transpose":
            assert row["attribution_category"] == "compiler_primitive"
        else:
            assert row["operation"] == "beta"
            assert row["attribution_category"] == "model_semantic"
    attributed = pq.read_table(tmp_path / "out" / "instruction_attribution.parquet").to_pylist()
    primitives = [row for row in attributed if row["attribution_category"] == "compiler_primitive"]
    assert len(primitives) == 4
    assert {row["label_evidence"] for row in primitives} == {"unanimous_backend_instruction_type"}


@pytest.mark.parametrize("other", [{"instruction_type": "REGULAR"}, {"missing": True}, {}])
def test_primitive_fallback_requires_unanimous_nonmissing_stock_types(tmp_path, other):
    row = dict(
        pcore_idx=0,
        engine="Tensor",
        compiler_pc=0,
        function="layer",
        opcode_byte=1,
        parents=[{"instruction_type": "TRANSPOSE"}, other],
        compiler_operation="transpose",
        compiler_operation_evidence="unanimous_backend_instruction_type",
    )
    path = tmp_path / "static.json"
    path.write_text(json.dumps({"instructions": [row]}))
    assert _static_labels(path)[0, "Tensor", 0]["compiler_operation"] is None


def test_source_line_semantics_require_verified_file_hash():
    source = "/opt/conda/site-packages/vllm_neuron/functional/attention/attention_decode.py:934"
    assert source_operation(source, SOURCE_HASHES) == "kv_gather"
    assert source_operation(source, {}) is None
    changed = dict(SOURCE_HASHES)
    changed["vllm_neuron/functional/attention/attention_decode.py"] = "0" * 64
    assert source_operation(source, changed) is None
    # Input norm is before the QKV matmul and must not inherit its label.
    assert source_operation(source.replace(":934", ":804"), SOURCE_HASHES) == "rmsnorm"
    assert source_operation(source.replace(":934", ":809"), SOURCE_HASHES) == "qkv_projection"


def test_unseen_sem_writer_invalidates_individual_dma_ownership():
    rows = [
        instruction(1, "dma", 1, 10, opcode="DMA_DIRECT2D", operands="semaphore=7 sem_increment=16")
    ]
    updates = [update(0, 0, "Tensor", kind="sem_wr_imm"), update(15, 1, "Scalar"), update(50, 16)]
    transfer = observed_dependencies(rows, updates)["transfers"][0]
    assert transfer["counter_crossing_ts"] == 50
    assert transfer["completion_ts"] is None
    assert transfer["evidence"] == "ambiguous_counter_prefix"


@pytest.mark.parametrize(
    "change,match",
    [
        ("hardware", "hardware"),
        ("missing_hash", "hash"),
        ("binary_changed", "NEFF hash"),
        ("map_neff", "different NEFF"),
    ],
)
def test_capture_identity_is_required_and_content_bound(tmp_path, change, match):
    path = capture_fixture(tmp_path)
    manifest = json.loads(path.read_text())
    neff = manifest["sessions"][0]["neffs"][0]
    if change == "hardware":
        del manifest["hardware"]["name"]
    elif change == "missing_hash":
        del neff["static_instructions_sha256"]
    elif change == "binary_changed":
        Path(neff["path"]).write_bytes(b"changed")
    elif change == "map_neff":
        static = json.loads(Path(neff["static_instructions"]).read_text())
        static["neff_sha256"] = "0" * 64
        Path(neff["static_instructions"]).write_text(json.dumps(static))
        neff["static_instructions_sha256"] = hashlib.sha256(
            Path(neff["static_instructions"]).read_bytes()
        ).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(NeuronOpsTraceError, match=match):
        analyze_capture(path, tmp_path / "bad_identity")


@pytest.mark.parametrize(
    "change,match",
    [
        ("negative", "invalid instruction"),
        ("duplicate", "Duplicate composite"),
        ("outside", "Unassigned non-control"),
    ],
)
def test_raw_instruction_defects_fail_without_silent_row_loss(tmp_path, change, match):
    path = capture_fixture(tmp_path)
    raw = tmp_path / "session" / "Instruction.parquet"
    rows = pq.read_table(raw).to_pylist()
    if change == "negative":
        rows[0]["end_ts"] = rows[0]["start_ts"] - 1
    elif change == "duplicate":
        rows.append(dict(rows[0]))
    else:
        rows.append(instruction(999, "bad", 200, 250, opcode="MATMUL"))
    pq.write_table(pa.Table.from_pylist(rows), raw)
    with pytest.raises(NeuronOpsTraceError, match=match):
        analyze_capture(path, tmp_path / "bad_rows")
