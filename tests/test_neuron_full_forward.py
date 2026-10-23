"""Whole-forward boundaries: rank union, graph geometry, shape refusal and the
stock context ladder shared with the Rust kernel."""

from pathlib import Path

import pytest

from profiling.exec import neuron
from profiling.runners.neuron.vllm_forward import validate_shape, validate_whole_context_shape
from profiling.runners.neuron.vllm_forward_engine import cases_for_plan, validate_greedy_tokens
from profiling.runners.neuron.vllm_forward_trace import measure_trace, model_geometry
from profiling.runners.neuron.vllm_forward_work import estimate_work


def test_tp_reservation_exposes_four_logical_units_without_changing_tp1(monkeypatch):
    device = neuron.NeuronDevice(0, (0, 1, 2, 3), 2, 96 * 1024**3, False, "trn2.3xlarge")
    assert neuron._reservation_environment(device)["NEURON_RT_VISIBLE_CORES"] == "0"
    assert neuron._reservation_environment(device, 4)["NEURON_VISIBLE_DEVICES"] == "0-3"
    assert "NEURON_RT_VISIBLE_CORES" not in neuron._reservation_environment(device, 4)
    monkeypatch.delenv("NEURON_RT_VISIBLE_CORES", raising=False)
    monkeypatch.setattr(neuron, "neuron_devices", lambda: [device])
    for name, value in neuron._reservation_environment(device, 4).items():
        monkeypatch.setenv(name, value)
    assert neuron.observe_neuron_reservation(neuron._reservation_payload(device, 4)) == device
    monkeypatch.setenv("NEURON_RT_VISIBLE_CORES", "0")
    with pytest.raises(RuntimeError, match="reserved"):
        neuron.observe_neuron_reservation(neuron._reservation_payload(device, 4))
    with pytest.raises(ValueError, match="exceed"):
        neuron._reservation_environment(device, 5)


def test_reject_runtime_buffer_failure_and_unvalidated_cache_layout():
    args = dict(
        phase="decode",
        token_bucket=128,
        max_model_len=128,
        kv_blocks=6782,
        block_size=32,
        tp_size=4,
        dtype="bf16",
    )
    validate_shape(**args)
    for changed in (
        {"token_bucket": 256},
        {"block_size": 64},
        {"tp_size": 1},
        {"kv_blocks": 6783},
        {"max_model_len": 3072},
    ):
        with pytest.raises(ValueError):
            validate_shape(**(args | changed))


@pytest.mark.parametrize(
    ("context", "largest"),
    [(128, 128), (512, 512), (1024, 256), (2048, 128), (4096, 64), (8192, 32)],
)
def test_context_ladder_caps_decode_buckets_at_kv_resident_capacity(context, largest):
    args = dict(
        phase="decode",
        max_model_len=context,
        kv_blocks=6782,
        block_size=32,
        tp_size=4,
        dtype="bf16",
    )
    validate_shape(token_bucket=largest, **args)
    for bucket in (largest * 2, 3):
        with pytest.raises(ValueError):
            validate_shape(token_bucket=bucket, **args)
    # Prefill graphs: multiples of 128 up to the context.
    prefill = args | {"phase": "prefill"}
    for bucket in range(128, context + 1, 128):
        validate_shape(**prefill, token_bucket=bucket)
    for bucket in (context + 128, 100, 0):
        with pytest.raises(ValueError):
            validate_shape(**prefill, token_bucket=bucket)
    # Region, segment and per-operation captures keep the whole-context graph.
    validate_whole_context_shape(**prefill, token_bucket=context)
    if context > 128:
        with pytest.raises(ValueError, match="whole-context"):
            validate_whole_context_shape(**prefill, token_bucket=128)


def test_smaller_prefill_graphs_get_their_own_prompts_and_engine_buckets():
    from profiling.runners.neuron.vllm_forward import precision_shape
    from profiling.runners.neuron.vllm_forward_engine import (
        cases_for_plan,
        engine_config,
        prefill_buckets_of,
    )

    class Tokenizer:
        def encode(self, text):
            return list(range(len(text.split()) * 2))

    spec = dict(max_model_len=512, kv_blocks=6782, block_size=32, tp_size=4, dtype="bf16")
    specs = [dict(spec, phase="prefill", token_bucket=t) for t in (128, 384, 512)]
    specs.append(dict(spec, phase="decode", token_bucket=32))
    plan = {"context": 512, "specs": specs, "model": "/model"}
    assert prefill_buckets_of(plan) == [128, 384, 512]
    cases = cases_for_plan(plan, Tokenizer())
    shapes = {case["shape_id"]: case for case in cases}
    assert {"b16-s120", "b16-s376", "b16-s504", "b32-s504"} <= set(shapes)
    assert all(len(p) == 120 for p in shapes["b16-s120"]["prompts"])
    assert shapes["b16-s120"]["bucket"] == 16
    assert shapes["b16-s120"]["prefill_only"] and "prefill_only" not in shapes["b16-s504"]
    for s in specs[:3]:
        assert precision_shape(s, 512) in shapes
    config = engine_config(plan, cases, Path("/run"), "profile")
    nc = config["additional_config"]["neuron_config"]
    assert nc["num_batched_tokens_buckets"] == [128, 384, 512]
    assert config["max_num_batched_tokens"] == 512
    # A decode-only plan keeps the single whole-context graph.
    decode_only = {"context": 512, "specs": specs[3:], "model": "/model"}
    assert prefill_buckets_of(decode_only) == [512]


def test_native_timing_unions_rank_overlap_and_requires_every_rank():
    requests = [{"start_epoch_ns": 0, "stop_epoch_ns": 1000, "batch": 1, "bucket": 16}]
    models = {"prefill": ("prefill", 512), "decode": ("decode", 16)}
    events = []
    for execution in range(8):
        model = "prefill" if execution == 0 else "decode"
        for core in range(8):
            events.append(
                dict(
                    timestamp=execution * 100 + core,
                    duration=20,
                    exec_id=execution,
                    model_id=model,
                    process_id=core // 2,
                    device_core_idx=core,
                    model_name=f"/compile_cache/{model}/graph.neff",
                )
            )
    measured, records = measure_trace(events, requests, models, set(range(8)))
    assert measured[("decode", 16)] == [27 / 1e6] * 7
    assert len(records) == 8
    with pytest.raises(ValueError, match="missing a rank"):
        measure_trace(events[:-1], requests, models, set(range(8)))
    with pytest.raises(ValueError, match="missing output"):
        measure_trace(events[:-8], requests, models, set(range(8)))
    # A prefill_only request times its prefill graph only; a normal request
    # beside it at the same decode bucket still adds its decodes.
    shifted = [dict(e, timestamp=e["timestamp"] + 1000) for e in events]
    measured, records = measure_trace(
        events + shifted,
        [
            requests[0] | {"prefill_only": True},
            requests[0] | {"start_epoch_ns": 1000, "stop_epoch_ns": 2000},
        ],
        models,
        set(range(8)),
    )
    assert measured == {("prefill", 512): [27 / 1e6] * 2, ("decode", 16): [27 / 1e6] * 7}
    assert len(records) == 16


def test_compiled_input_geometry_refuses_changed_pool():
    metadata = (
        "Input 0:\n Shape: (512,)\nInput 5:\n Shape: (6782, 2, 32, 128)\n Dtype: bfloat16\nInput 6:"
    )
    assert model_geometry(metadata) == ("prefill", 512)
    with pytest.raises(ValueError, match="layout changed"):
        model_geometry(metadata.replace("6782", "6783"))


def test_small_batches_keep_the_same_prespecified_subject_coverage():
    class Tokenizer:
        def encode(self, text):
            return [text.split()[3]] * 2048

    plan = {
        "context": 512,
        "specs": [{"phase": "decode", "token_bucket": bucket} for bucket in (1, 2, 4, 8, 16)],
    }
    cases = cases_for_plan(plan, Tokenizer())
    subjects = {}
    for case in cases:
        subjects.setdefault(case["bucket"], set()).update(ids[0] for ids in case["prompts"])
    assert all(len(values) == 8 for values in subjects.values())
    assert len({case["id"] for case in cases}) == len(cases)


def test_graph_work_matches_independently_derived_tile_counts():
    decode = estimate_work("decode", 128, 128)
    assert decode["contraction_flops_per_rank"] == 482445623296
    assert decode["persistent_operand_bytes_per_rank"] == 4294975488
    for tokens, attention in ((128, 2147483648), (512, 27917287424), (2048, 317827579904)):
        assert estimate_work("prefill", tokens, tokens)["attention_flops_per_rank"] == attention


def test_greedy_consistency_accepts_unique_and_nonfirst_exact_tied_maxima():
    import numpy as np

    logits = np.array([[1, 3, 2], [9.5, 2, 9.5]], dtype=np.float32)
    validate_greedy_tokens(logits, [1, 2])


def test_greedy_consistency_rejects_one_stored_float_step_below_maximum():
    import numpy as np

    maximum = np.float32(9.5)
    below = np.nextafter(maximum, np.float32(-np.inf))
    with pytest.raises(RuntimeError, match="exact stored row maximum"):
        validate_greedy_tokens(np.array([[maximum, below]], dtype=np.float32), [1])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_greedy_consistency_rejects_nonfinite_unselected_logits(bad):
    import numpy as np

    with pytest.raises(RuntimeError, match="finite floating rows"):
        validate_greedy_tokens(np.array([[1, bad]], dtype=np.float32), [0])


@pytest.mark.parametrize("tokens", [[], [0, 0], [-1], [2], [1.0], [True], [[1]]])
def test_greedy_consistency_rejects_wrong_counts_or_invalid_indices(tokens):
    import numpy as np

    with pytest.raises(RuntimeError, match="integral in-range index"):
        validate_greedy_tokens(np.array([[1, 2]], dtype=np.float32), tokens)


@pytest.mark.parametrize("logits", [[], [[]], [1, 2], [[[1, 2]]]])
def test_greedy_consistency_rejects_missing_or_nonmatrix_logits(logits):
    with pytest.raises(RuntimeError, match="complete finite floating rows"):
        validate_greedy_tokens(logits, [0])


def test_accuracy_failure_retains_generated_history_and_native_vector(tmp_path, monkeypatch):
    import json
    import struct
    import sys
    from types import SimpleNamespace

    import numpy as np

    from profiling.runners.neuron import vllm_forward_engine as engine
    from profiling.runners.neuron import vllm_identity

    # This experimental caller has no coordinator-specific cache environment.
    monkeypatch.delenv("SERVINGSTUDIO_NEURON_PROFILE_CACHE_DIR", raising=False)
    monkeypatch.setattr(vllm_identity, "stock_compile_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(vllm_identity, "accuracy_source_identity", lambda: {})

    case = {"id": "test", "shape_id": "b1-s120", "bucket": 1, "prompts": [[1] * 120]}
    (tmp_path / "plan.json").write_text(json.dumps({"model": "unused", "context": 128}))
    monkeypatch.setattr(engine, "cases_for_plan", lambda *_: [case])
    tokenizer = SimpleNamespace(from_pretrained=lambda *_, **__: None)

    class LLM:
        def __init__(self, **config):
            self.raw = tmp_path / "raw-logits"
            self.raw.mkdir()

        def generate(self, prompts, params, **kwargs):
            logits = np.zeros(128256, dtype="<f4")
            logits[0], logits[2] = 2, 1
            data = struct.pack("qq", 8, logits.size)
            for position in range(119, 127):
                data += struct.pack("qq", 13, position) + logits.tobytes()
            (self.raw / "step.bin").write_bytes(data)
            return [
                SimpleNamespace(
                    prompt_token_ids=prompts[0]["prompt_token_ids"],
                    request_id="13",
                    outputs=[SimpleNamespace(token_ids=[2] * 8)],
                )
            ]

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=tokenizer))
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            LLM=LLM,
            SamplingParams=lambda **kwargs: kwargs,
        ),
    )
    with pytest.raises(RuntimeError, match="exact stored row maximum"):
        engine.main(tmp_path, "accuracy")
    rows = json.loads((tmp_path / "accuracy-outputs.json").read_text())
    assert rows[0]["token_ids"] == [2] * 8
    assert rows[0]["prompt_ids"] == case["prompts"][0]
    assert np.load(tmp_path / "native-test-slot0.npy").shape == (8, 128256)


def test_compiled_decode_graphs_omit_bucket_one_only_when_the_inventory_does():
    from profiling.runners.neuron.vllm_forward_engine import seqs_buckets

    def plan(*decode, prefill=True):
        specs = [{"phase": "decode", "token_bucket": b} for b in decode]
        return {"specs": specs + ([{"phase": "prefill", "token_bucket": 4096}] if prefill else [])}

    def cases(*buckets):
        return [{"bucket": b} for b in buckets]

    assert seqs_buckets(plan(1, 16), cases(1, 16)) == [1, 16]
    assert seqs_buckets(plan(), cases(16)) == [1, 16]
    assert seqs_buckets(plan(16, 32), cases(16, 32)) == [16, 32]


def test_coverage_allows_exactly_one_early_async_decode():
    from profiling.runners.neuron.vllm_forward_trace import check_forward_coverage

    request = {"batch": 2, "bucket": 2}

    def records(early, at_bucket):
        rows = [{"request_index": 0, "phase": "prefill", "token_bucket": 512}] * 2
        rows += [{"request_index": 0, "phase": "decode", "token_bucket": 1}] * early
        rows += [{"request_index": 0, "phase": "decode", "token_bucket": 2}] * at_bucket
        return rows

    check_forward_coverage(records(0, 7), [request])
    # Stock async scheduling: request 0 decodes alone before request 1's prefill.
    check_forward_coverage(records(2, 6), [request])
    with pytest.raises(ValueError, match="six times"):
        check_forward_coverage(records(2, 5), [request])
    with pytest.raises(ValueError, match="missing output decode"):
        check_forward_coverage(records(0, 6), [request])


@pytest.mark.needs_binary
def test_python_and_rust_share_the_stock_context_ladder(sim_bin):
    """The Rust kernel config accepts exactly the inventories the Python profiler
    measures: every context's largest decode bucket, never the next one, and
    prefill buckets on the same PREFILL_BUCKET_STEP grid."""
    import json
    import subprocess

    from launcher.exec import _build_subprocess_env
    from profiling.runners.neuron import stock_contract

    def rust_rows(context, buckets, prefill=()):
        request = {
            "op": "rows",
            "kind": "neuron_llama_forward",
            "config": {
                "backends": ["vllm_neuron"],
                "gpu_name": "AWS Trainium2 LNC2",
                "max_model_len": context,
                "kv_blocks": stock_contract.KV_BLOCKS,
                "block_size": stock_contract.BLOCK_SIZE,
                "tp_size": stock_contract.TP_SIZE,
                "dtype": "bf16",
                "decode_buckets": buckets,
                "prefill_buckets": list(prefill),
            },
        }
        result = subprocess.run(
            [str(sim_bin), "kernel-query"],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            env=_build_subprocess_env(),
        )
        return json.loads(result.stdout)["rows"] if result.returncode == 0 else None

    for context in stock_contract.CONTEXTS:
        largest = stock_contract.largest_decode_bucket(context)
        rows = rust_rows(context, [1, largest])
        assert rows is not None, context
        for row in rows:
            validate_shape(**{k: v for k, v in row["args"].items() if k != "backend"})
        assert rust_rows(context, [1, largest * 2]) is None, context
    assert rust_rows(3072, [1]) is None
    # Prefill inventories: the Rust and Python step rules agree.
    step = stock_contract.PREFILL_BUCKET_STEP
    rows = rust_rows(512, [1, 16], range(step, 513, step))
    assert rows is not None
    assert [r["args"]["token_bucket"] for r in rows if r["args"]["phase"] == "prefill"] == list(
        range(step, 513, step)
    )
    for row in rows:
        validate_shape(**{k: v for k, v in row["args"].items() if k != "backend"})
    assert rust_rows(512, [1, 16], [step // 2, 512]) is None
