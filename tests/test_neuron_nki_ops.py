"""Standalone NKI per-op leaves: shape refusal, the raw-row and floor policy,
trace reduction, export naming, registry identity and the inventory shared
with the Rust kinds."""

import json
import subprocess

import pytest

from profiling.db.registry import find_kernel_profiler_spec
from profiling.runners.metrics import RunnerResult
from profiling.runners.neuron import nki_bench
from profiling.runners.neuron.stock_contract import CONTEXTS, largest_decode_bucket

KINDS = list(nki_bench.UNITS)
IDENTITY = {"tp_size": 4, "dtype": "bf16"}
KV = {"max_model_len": 512, "kv_blocks": 6782, "block_size": 32}


def test_every_kind_registers_the_one_core_nki_backend():
    for kind in KINDS:
        spec = find_kernel_profiler_spec(kind, "vllm_neuron_nki")
        assert spec.table_name == kind
        assert spec.neuron_logical_cores == 1
        assert spec.supports.device_family == "neuron"
        assert spec.runner_ref.function_name == f"profile_{kind}_batch"
        assert spec.list_native


def test_prefill_only_kinds_take_ladder_contexts():
    for kind in ("neuron_nki_qkv", "neuron_nki_attention_cte", "neuron_nki_output_projection"):
        for context in CONTEXTS:
            assert nki_bench.validate_shape(kind, {"token_bucket": context, **IDENTITY}) == (
                "prefill",
                context,
                context,
            )
        with pytest.raises(ValueError, match="one of"):
            nki_bench.validate_shape(kind, {"token_bucket": 504, **IDENTITY})


def test_shape_only_decode_kinds_take_powers_of_two_up_to_the_ladder_cap():
    for kind in ("neuron_nki_mlp", "neuron_glue"):
        assert nki_bench.validate_shape(
            kind, {"phase": "decode", "token_bucket": 512, **IDENTITY}
        ) == ("decode", 512, nki_bench.SHAPE_ONLY_CONTEXT)
        assert nki_bench.validate_shape(
            kind, {"phase": "prefill", "token_bucket": 8192, **IDENTITY}
        ) == ("prefill", 8192, 8192)
        for bucket, message in [(3, "powers of two"), (1024, "ladder")]:
            with pytest.raises(ValueError, match=message):
                nki_bench.validate_shape(
                    kind, {"phase": "decode", "token_bucket": bucket, **IDENTITY}
                )


def test_decode_attention_is_capped_by_its_context_and_pins_the_pool():
    kind = "neuron_nki_attention_tkg"
    for context in CONTEXTS:
        largest = largest_decode_bucket(context)
        spec = {"token_bucket": largest, **KV, "max_model_len": context, **IDENTITY}
        assert nki_bench.validate_shape(kind, spec) == ("decode", largest, context)
        spec["token_bucket"] = largest * 2
        with pytest.raises(ValueError, match="capacity"):
            nki_bench.validate_shape(kind, spec)
    with pytest.raises(ValueError, match="pool"):
        nki_bench.validate_shape(kind, {"token_bucket": 1, **KV, "kv_blocks": 100, **IDENTITY})
    with pytest.raises(ValueError, match="BF16"):
        nki_bench.validate_shape(kind, {"token_bucket": 1, **KV, "tp_size": 4, "dtype": "fp16"})
    with pytest.raises(ValueError, match="TP4"):
        nki_bench.validate_shape(kind, {"token_bucket": 1, **KV, "tp_size": 8, "dtype": "bf16"})


def _event(exec_id, start, duration, core):
    return {
        "name": "nc_exec_running",
        "exec_id": exec_id,
        "timestamp": start,
        "duration": duration,
        "model_name": "/cache/unit.neff",
        "device_core_idx": core,
    }


def test_execution_time_is_the_union_of_the_core_pair(tmp_path):
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "trace_event": [
                    # physical cores overlap 100..300 and 200..400 -> 300 ns
                    _event(1, 100, 200, 4),
                    _event(1, 200, 200, 5),
                    # disjoint core spans -> 50 + 50 ns
                    _event(2, 1000, 50, 4),
                    _event(2, 1100, 50, 5),
                    {"name": "other", "timestamp": 0, "duration": 5},
                ]
            }
        )
    )
    assert sorted(nki_bench.exec_times_ms(trace)) == [100 / 1e6, 300 / 1e6]
    # one span, or two spans on one core, would undercount the LNC2 pair
    for events in ([_event(1, 0, 10, 4)], [_event(1, 0, 10, 4), _event(1, 0, 10, 4)]):
        trace.write_text(json.dumps({"trace_event": events}))
        with pytest.raises(ValueError, match="two physical cores"):
            nki_bench.exec_times_ms(trace)
    trace.write_text(json.dumps({"trace_event": []}))
    with pytest.raises(ValueError, match="no device executions"):
        nki_bench.exec_times_ms(trace)


def test_export_names_one_session_per_unit(monkeypatch, tmp_path):
    prof = tmp_path / "prof_prefill_qkv_proj_512_512"
    (prof / "i-abc_pid_1" / "123").mkdir(parents=True)
    calls = []
    monkeypatch.setattr(nki_bench.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    assert nki_bench.export_trace(prof) == tmp_path / "prof_prefill_qkv_proj_512_512.json"
    assert calls[0][calls[0].index("-d") + 1] == str(prof / "i-abc_pid_1" / "123")
    (prof / "i-abc_pid_1" / "456").mkdir()
    with pytest.raises(ValueError, match="one profile session"):
        nki_bench.export_trace(prof)


def test_batch_keeps_rows_raw_and_errors_per_row(monkeypatch, tmp_path):
    monkeypatch.setenv("SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(nki_bench, "_versions", lambda: "vllm-neuron=test")
    measured = []

    def fake_measure(unit, tokens, ctx, root):
        measured.append((unit, tokens, ctx))
        if unit == "noop":
            return {"median_ms": 0.026, "min_ms": 0.026, "p90_ms": 0.026, "n": 50, "compile_s": 1.0}
        if tokens == 8192:
            raise RuntimeError("compiler out of memory")
        # decode glue at B1 runs below the floor; the row must still be its raw time
        median = 0.0215 if tokens == 1 else 0.12 + 0.001 * tokens
        return {"median_ms": median, "min_ms": 0.1, "p90_ms": 0.2, "n": 50, "compile_s": 30.0}

    monkeypatch.setattr(nki_bench, "measure_unit", fake_measure)
    specs = [
        {"phase": "decode", "token_bucket": 16, **IDENTITY},
        {"phase": "decode", "token_bucket": 3, **IDENTITY},
        {"phase": "prefill", "token_bucket": 8192, **IDENTITY},
        {"phase": "decode", "token_bucket": 1, **IDENTITY},
    ]
    results = nki_bench.profile_neuron_glue_batch(specs)
    assert [r.error is None for r in results] == [True, False, False, True]
    assert results[0].metrics.time_ms == pytest.approx(0.136)
    assert results[3].metrics.time_ms == pytest.approx(0.0215)
    assert "powers of two" in results[1].error
    assert "compiler out of memory" in results[2].error
    assert measured[0] == ("noop", 1, nki_bench.SHAPE_ONLY_CONTEXT)
    assert ("decode.glue", 3, nki_bench.SHAPE_ONLY_CONTEXT) not in measured
    provenance = nki_bench.row_provenance(**specs[0])
    assert "median_ms=0.1360" in provenance and "floor_ms=0.0260" in provenance
    assert "vllm-neuron=test" in provenance and str(tmp_path) in provenance
    assert nki_bench.row_provenance(**specs[1]) is None
    # a batch of only bad shapes never touches the device
    measured.clear()
    only_bad = nki_bench.profile_neuron_nki_mlp_batch(
        [{"phase": "decode", "token_bucket": 3, **IDENTITY}]
    )
    assert only_bad == [RunnerResult(error="decode buckets are powers of two")]
    assert measured == []


def test_a_failing_floor_is_every_rows_error(monkeypatch, tmp_path):
    monkeypatch.setenv("SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR", str(tmp_path))

    def no_explorer(unit, tokens, ctx, root):
        raise FileNotFoundError("/opt/aws/neuron/bin/neuron-explorer")

    monkeypatch.setattr(nki_bench, "measure_unit", no_explorer)
    results = nki_bench.profile_neuron_nki_qkv_batch(
        [{"token_bucket": 512, **IDENTITY}, {"token_bucket": 504, **IDENTITY}]
    )
    assert all("floor: FileNotFoundError" in r.error for r in results)


@pytest.mark.needs_binary
def test_python_and_rust_share_the_nki_inventories(sim_bin):
    """Every row the Rust kinds enumerate is a shape the Python profiler
    measures, and the Rust kinds refuse the inventories Python refuses."""
    from launcher.exec import _build_subprocess_env

    def rust(kind, prefill, decode, context=512):
        request = {
            "op": "rows",
            "kind": kind,
            "config": {
                "backends": ["vllm_neuron_nki"],
                "gpu_name": "AWS Trainium2 LNC2",
                "max_model_len": context,
                "kv_blocks": 6782,
                "block_size": 32,
                "tp_size": 4,
                "dtype": "bf16",
                "prefill_buckets": prefill,
                "decode_buckets": decode,
            },
        }
        result = subprocess.run(
            [str(sim_bin), "kernel-query"],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            env=_build_subprocess_env(),
        )
        if result.returncode:
            return None, result.stderr + result.stdout
        return json.loads(result.stdout)["rows"], ""

    inventories = {
        "neuron_nki_qkv": (list(CONTEXTS), []),
        "neuron_nki_attention_cte": (list(CONTEXTS), []),
        "neuron_nki_output_projection": (list(CONTEXTS), []),
        "neuron_nki_mlp": (list(CONTEXTS), [1, 16, 512]),
        "neuron_glue": ([512], [1, 16]),
        "neuron_nki_attention_tkg": ([], [1, 16, 512]),
    }
    for kind, (prefill, decode) in inventories.items():
        rows, _ = rust(kind, prefill, decode)
        assert rows is not None, kind
        assert len(rows) == len(prefill) + len(decode)
        for row in rows:
            nki_bench.validate_shape(kind, {k: v for k, v in row["args"].items() if k != "backend"})
    for kind, prefill, decode, context, reason in [
        ("neuron_nki_qkv", [512], [], 512, "two compiled variants"),
        ("neuron_nki_qkv", [512, 1024], [1], 512, "no decode inventory"),
        ("neuron_nki_attention_tkg", [], [1, 64], 8192, "within the context limit"),
    ]:
        rows, message = rust(kind, prefill, decode, context)
        assert rows is None and reason in message, (kind, message)
    assert rust("neuron_nki_attention_tkg", [], [1, 32], context=8192)[0] is not None
