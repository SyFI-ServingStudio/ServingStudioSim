"""Whole-forward boundaries: rank union, graph geometry and shape refusal."""

import pytest

from profiling.exec import neuron
from profiling.runners.neuron.vllm_forward import validate_shape
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
        {"max_model_len": 4096},
    ):
        with pytest.raises(ValueError):
            validate_shape(**(args | changed))


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
            return [SimpleNamespace(
                prompt_token_ids=prompts[0]["prompt_token_ids"], request_id="13",
                outputs=[SimpleNamespace(token_ids=[2] * 8)],
            )]

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=tokenizer))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        LLM=LLM, SamplingParams=lambda **kwargs: kwargs,
    ))
    with pytest.raises(RuntimeError, match="exact stored row maximum"):
        engine.main(tmp_path, "accuracy")
    rows = json.loads((tmp_path / "accuracy-outputs.json").read_text())
    assert rows[0]["token_ids"] == [2] * 8
    assert rows[0]["prompt_ids"] == case["prompts"][0]
    assert np.load(tmp_path / "native-test-slot0.npy").shape == (8, 128256)
