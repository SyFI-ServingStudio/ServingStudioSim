"""Standalone launch-unit leaves (vLLM Neuron 0.29 image): the shape and work rules, the
raw-row and floor policy, the trace reduction, the collective launcher's core
binding, the pinned images, and the inventory shared with the Rust kinds."""

import json
import subprocess

import pytest

from profiling.container import images
from profiling.db.registry import find_kernel_profiler_spec
from profiling.exec import neuron
from profiling.runners.metrics import CommMetrics, RunnerResult
from profiling.runners.neuron import collective_bench, nki_bench
from profiling.runners.neuron.stock_contract import CONTEXTS, largest_decode_bucket

KINDS = list(nki_bench.UNITS)
IDENTITY = {"tp_size": 4, "dtype": "bf16"}
KV = {"max_model_len": 512, "kv_blocks": 6782, "block_size": 32}


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


def test_shape_only_kinds_take_powers_of_two_up_to_the_ladder_cap():
    for kind in nki_bench.SHAPE_ONLY_KINDS:
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
    # the head samples one row in prefill and every row in decode
    prefill_flops, _ = nki_bench.estimate_work("neuron_head", "prefill", 512, 512)
    decode_flops, _ = nki_bench.estimate_work("neuron_head", "decode", 16, 512)
    assert prefill_flops == 2 * 4096 * 32064
    assert decode_flops == 16 * prefill_flops


def test_decode_attention_is_capped_by_its_context_and_pins_the_pool():
    kind = "neuron_attention_decode"
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


def _event(exec_id, start, duration, core, name="nc_exec_running"):
    return {
        "name": name,
        "exec_id": exec_id,
        "timestamp": start,
        "duration": duration,
        "model_name": "graph.neff",
        "device_core_idx": core,
    }


def test_execution_time_is_the_union_of_the_core_pair():
    events = [
        # physical cores overlap 100..300 and 200..400 -> 300 ns
        _event(1, 100, 200, 4),
        _event(1, 200, 200, 5),
        # disjoint core spans -> 50 + 50 ns
        _event(2, 1000, 50, 4),
        _event(2, 1100, 50, 5),
        _event(3, 0, 5, 4, name="other"),
    ]
    assert sorted(nki_bench.exec_times_ms(events)) == [100 / 1e6, 300 / 1e6]
    # one span, or two spans on one core, would undercount the LNC2 pair
    for bad in ([_event(1, 0, 10, 4)], [_event(1, 0, 10, 4), _event(1, 0, 10, 4)]):
        with pytest.raises(ValueError, match="two physical cores"):
            nki_bench.exec_times_ms(bad)
    with pytest.raises(ValueError, match="no device executions"):
        nki_bench.exec_times_ms([])


def test_export_names_one_session_per_unit(monkeypatch, tmp_path):
    export = tmp_path / "prof.json"
    export.write_text(json.dumps({"trace_event": [_event(1, 0, 10, 4), _event(1, 0, 10, 5)]}))
    assert len(nki_bench.load_events(export)) == 2
    prof = tmp_path / "prof_prefill_qkv_proj_512_512"
    session = prof / "i-abc_pid_1" / "123"
    session.mkdir(parents=True)
    calls = []
    monkeypatch.setattr(nki_bench.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    assert nki_bench.export_trace(prof) == tmp_path / "prof_prefill_qkv_proj_512_512.json"
    assert calls[0][calls[0].index("-d") + 1] == str(session)
    assert "nc_exec_running" in calls[0]
    # a flat runtime capture is its own session
    flat = tmp_path / "flat"
    flat.mkdir()
    (flat / "ntrace.pb").write_bytes(b"")
    assert nki_bench.session_dir(flat) == flat
    (prof / "i-abc_pid_1" / "456").mkdir()
    with pytest.raises(ValueError, match="one profile session"):
        nki_bench.export_trace(prof)


def test_batch_keeps_rows_raw_and_errors_per_row(monkeypatch, tmp_path):
    monkeypatch.setenv("SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR", str(tmp_path))
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
    assert str(tmp_path) in provenance
    assert nki_bench.row_provenance(**specs[1]) is None
    # the head kind runs through the same batch path
    head = nki_bench.profile_neuron_head_batch(
        [{"phase": "decode", "token_bucket": 16, **IDENTITY}]
    )
    assert head[0].error is None and ("decode.head", 16, nki_bench.SHAPE_ONLY_CONTEXT) in measured
    # a batch of only bad shapes never touches the device
    measured.clear()
    only_bad = nki_bench.profile_neuron_nki_mlp_batch(
        [{"phase": "decode", "token_bucket": 3, **IDENTITY}]
    )
    assert only_bad == [RunnerResult(error="decode buckets are powers of two")]
    assert measured == []


def test_forward_rows_are_the_slope_and_intercept_of_two_depths(monkeypatch, tmp_path):
    from profiling.runners.neuron import forward_bench

    monkeypatch.setenv("SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(forward_bench, "visible_logical_cores", lambda: [0, 1, 2, 3])
    launches = []

    class FakeLauncher:
        def __init__(self, num_ranks, cores, *, own_core_only=False):
            # each rank compiles on neuron:0, so it must see its own core alone
            assert (num_ranks, cores, own_core_only) == (4, [0, 1, 2, 3], True)

        def run(self, fn, *, shapes, path, root):
            launches.append((tuple(shapes), path))
            # t(8) = 0.7 + 8 x 0.27, t(32) = 0.7 + 32 x 0.27 at B1
            return {
                f"{b}:{c}": {"8": 0.7 + 8 * 0.27 * b, "32": 0.7 + 32 * 0.27 * b} for b, c in shapes
            }

    monkeypatch.setattr(forward_bench, "NeuronMpLauncher", FakeLauncher)
    specs = [{"token_bucket": 1, **KV, **IDENTITY}, {"token_bucket": 3, **KV, **IDENTITY}]
    layer = forward_bench.profile_neuron_decoder_layer_batch(specs)
    assert layer[0].metrics.time_ms == pytest.approx(0.27)
    assert "powers of two" in layer[1].error
    assert launches == [(((1, 512),), "nki")]
    # the fixed term reuses the same two measurements: no second launch
    fixed = forward_bench.profile_neuron_decode_fixed_batch(specs[:1])
    assert fixed[0].metrics.time_ms == pytest.approx(0.7)
    assert len(launches) == 1
    provenance = forward_bench.row_provenance(**specs[0])
    assert "depth8_ms=" in provenance and "layer_ms=0.2700" in provenance
    # the torch path is a separate measurement
    forward_bench.profile_neuron_decoder_layer_torch_batch(specs[:1])
    assert launches[-1] == (((1, 512),), "torch")


def test_a_failed_forward_rank_group_is_every_measured_rows_error(monkeypatch, tmp_path):
    from profiling.runners.neuron import forward_bench

    monkeypatch.setenv("SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(forward_bench, "visible_logical_cores", lambda: [0, 1, 2, 3])

    class Broken:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            raise RuntimeError("rank 2: nrt_tensor_allocate status=4")

    monkeypatch.setattr(forward_bench, "NeuronMpLauncher", Broken)
    out = forward_bench.profile_neuron_decode_fixed_batch(
        [{"token_bucket": 16, **KV, **IDENTITY}, {"token_bucket": 64, **KV, **IDENTITY}]
    )
    assert all("nrt_tensor_allocate" in r.error for r in out)


def test_a_failing_floor_is_every_rows_error(monkeypatch, tmp_path):
    monkeypatch.setenv("SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR", str(tmp_path))

    def no_explorer(unit, tokens, ctx, root):
        raise FileNotFoundError("/opt/aws/neuron/bin/neuron-explorer")

    monkeypatch.setattr(nki_bench, "measure_unit", no_explorer)
    results = nki_bench.profile_neuron_nki_qkv_batch(
        [{"token_bucket": 512, **IDENTITY}, {"token_bucket": 504, **IDENTITY}]
    )
    assert all("floor: FileNotFoundError" in r.error for r in results)


def test_collective_ranks_bind_one_logical_core_each(monkeypatch):
    monkeypatch.setenv("NEURON_VISIBLE_DEVICES", "4-7")
    monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
    assert collective_bench.visible_logical_cores() == [4, 5, 6, 7]
    launcher = collective_bench.NeuronMpLauncher(4, [4, 5, 6, 7], master_port=29999)
    assert launcher.cores == [4, 5, 6, 7]
    # the torch rendezvous and the NRT root communicator never share a port
    assert launcher.root_comm_port != launcher.master_port
    with pytest.raises(ValueError, match="visible logical cores"):
        collective_bench.NeuronMpLauncher(4, [0, 1])
    monkeypatch.setenv("NEURON_VISIBLE_DEVICES", "2")
    assert collective_bench.visible_logical_cores() == [2]
    assert collective_bench.bus_bandwidth_factor("all_reduce", 4) == pytest.approx(1.5)
    assert collective_bench.bus_bandwidth_factor("all_gather", 4) == pytest.approx(0.75)
    assert collective_bench.bus_bandwidth_factor("reduce_scatter", 4) == pytest.approx(0.75)
    with pytest.raises(ValueError, match="unknown"):
        collective_bench.bus_bandwidth_factor("broadcast", 4)


def test_collective_batch_refuses_mixed_groups_and_unknown_ops(monkeypatch):
    monkeypatch.setenv("NEURON_VISIBLE_DEVICES", "0-3")
    base = {"num_ranks": 4, "message_size_bytes": 8192, "dtype": "bf16", "fabric": "neuron_link"}
    mixed = [{"op": "all_reduce", **base}, {"op": "all_reduce", **base, "num_ranks": 2}]
    results = collective_bench.profile_neuron_collective_batch(mixed)
    assert all("share num_ranks" in r.error for r in results)
    unknown = [{"op": "broadcast", **base}]
    assert "unknown" in collective_bench.profile_neuron_collective_batch(unknown)[0].error
    single = [{"op": "all_gather", **base, "num_ranks": 1}]
    assert "at least 2" in collective_bench.profile_neuron_collective_batch(single)[0].error
    # a successful chunk's rows carry the runner's note; a failed chunk's do not
    monkeypatch.setattr(
        collective_bench, "run_comm_batch", lambda launcher, fn, specs: [RunnerResult(error="x")]
    )
    ok = [{"op": "all_reduce", **base}]
    collective_bench.profile_neuron_collective_batch(ok)
    assert collective_bench.row_provenance(**ok[0]) is None
    monkeypatch.setattr(
        collective_bench,
        "run_comm_batch",
        lambda launcher, fn, specs: [RunnerResult(metrics=CommMetrics(1.0, 1.0, 1.0, 0.0))],
    )
    collective_bench.profile_neuron_collective_batch(ok)
    assert "cores=[0, 1, 2, 3]" in collective_bench.row_provenance(**ok[0])


def test_the_neuron_image_is_pinned_by_digest(monkeypatch, tmp_path):
    monkeypatch.setenv(neuron.DOCKER_HOST_ENV, f"unix://{tmp_path}/docker.sock")
    monkeypatch.setenv("VIBESIM_PROFILE_RUNTIME", "container")
    monkeypatch.delenv(images.IMAGES_ENV, raising=False)
    assert neuron.neuron_container("vllm_neuron_native_env").image.startswith("sha256:d6602265")
    # the digest override still maps an env to another image
    digest = "sha256:" + "ab" * 32
    monkeypatch.setenv(images.IMAGES_ENV, json.dumps({"vllm_neuron_native_env": digest}))
    assert neuron.neuron_container("vllm_neuron_native_env").image == digest


@pytest.mark.needs_binary
def test_python_and_rust_share_the_inventories(sim_bin):
    """Every row the Rust kinds enumerate is a shape the Python profiler
    measures, the Rust kinds refuse the inventories Python refuses, and the
    collective kind's rows carry the Python args."""
    from launcher.exec import _build_subprocess_env

    def rust(kind, config):
        request = {"op": "rows", "kind": kind, "config": config}
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

    def unit_config(prefill, decode, context=512):
        return {
            "backends": ["vllm_neuron_nki"],
            "gpu_name": "AWS Trainium2 LNC2",
            "max_model_len": context,
            "kv_blocks": 6782,
            "block_size": 32,
            "tp_size": 4,
            "dtype": "bf16",
            "prefill_buckets": prefill,
            "decode_buckets": decode,
        }

    inventories = {
        "neuron_nki_qkv": (list(CONTEXTS), []),
        "neuron_nki_attention_cte": (list(CONTEXTS), []),
        "neuron_nki_output_projection": (list(CONTEXTS), []),
        "neuron_nki_mlp": (list(CONTEXTS), [1, 16, 512]),
        "neuron_glue": ([512], [1, 16]),
        "neuron_head": ([128, 512], [1, 16]),
        "neuron_attention_decode": ([], [1, 16, 512]),
    }
    for kind, (prefill, decode) in inventories.items():
        rows, message = rust(kind, unit_config(prefill, decode))
        assert rows is not None, (kind, message)
        assert len(rows) == len(prefill) + len(decode)
        for row in rows:
            nki_bench.validate_shape(kind, {k: v for k, v in row["args"].items() if k != "backend"})
    for kind, prefill, decode, context, reason in [
        ("neuron_nki_qkv", [], [], 512, "at least one compiled variant"),
        ("neuron_nki_qkv", [512, 1024], [1], 512, "no decode inventory"),
        ("neuron_attention_decode", [], [1, 64], 8192, "within the context limit"),
    ]:
        rows, message = rust(kind, unit_config(prefill, decode, context))
        assert rows is None and reason in message, (kind, message)
    assert rust("neuron_attention_decode", unit_config([], [1, 32], 8192))[0] is not None
    rows, message = rust(
        "neuron_collective",
        {
            "backends": ["vllm_neuron_dist"],
            "gpu_name": "AWS Trainium2 LNC2",
            "op": "reduce_scatter",
            "num_ranks": 4,
            "fabric": "neuron_link",
        },
    )
    assert rows is not None, message
    assert len(rows) == 19
    spec = find_kernel_profiler_spec("neuron_collective", "vllm_neuron_dist")
    assert set(rows[0]["args"]) - {"backend"} == set(spec.args_schema.__dataclass_fields__)
    assert rows[0]["args"]["op"] == "reduce_scatter"
