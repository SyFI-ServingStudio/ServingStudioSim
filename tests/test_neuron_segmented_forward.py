"""Segmented-prefill stock forward: scope, engine, references, prior mapping and work."""

import pytest

from profiling.runners.neuron.vllm_forward import precision_shape
from profiling.runners.neuron.vllm_forward_engine import (
    cases_for_plan,
    engine_config,
    profile_cases,
    segment_timing_cases,
)
from profiling.runners.neuron.vllm_forward_reference import reference_contract
from profiling.runners.neuron.vllm_forward_trace import (
    measure_segmented_trace,
    segment_count,
    segmented_model_geometry,
)
from profiling.runners.neuron.vllm_forward_work import estimate_segmented_work, estimate_work
from profiling.runners.neuron.vllm_segmented_forward import validate_segmented_shape

STOCK = dict(kv_blocks=6782, block_size=32, tp_size=4, dtype="bf16")


def spec(phase, bucket, prior, segment, context):
    return dict(
        phase=phase,
        token_bucket=bucket,
        prior_tokens=prior,
        segment_size=segment,
        max_model_len=context,
        **STOCK,
    )


@pytest.mark.parametrize(
    "args",
    [
        spec("prefill", 2048, 0, 2048, 16384),
        spec("prefill", 2048, 14336, 2048, 16384),  # last segment: 14336 + 1 <= 16376
        spec("prefill", 512, 32256, 512, 32768),
        spec("prefill", 4096, 4096, 4096, 8192),
        spec("decode", 16, 0, 2048, 16384),
        spec("decode", 8, 0, 1024, 32768),
        spec("decode", 1, 0, 1024, 32768),
    ],
)
def test_segmented_scope_accepts_contract_shapes(args):
    validate_segmented_shape(**args)


@pytest.mark.parametrize(
    "args",
    [
        spec("prefill", 2048, 0, 2048, 4096),  # context outside the segmented ladder
        spec("prefill", 768, 0, 768, 16384),  # segment size not in the contract
        spec("prefill", 8192, 0, 8192, 8192),  # segment must be below the context
        spec("prefill", 1024, 0, 2048, 16384),  # prefill bucket must equal S
        spec("prefill", 2048, 1024, 2048, 16384),  # prior is not a whole number of segments
        spec("prefill", 2048, 16384, 2048, 16384),  # no prompt token at this prior
        spec("prefill", 512, 32768 - 8, 512, 32768),  # prior + 1 > C - 8
        spec("prefill", 2048, -2048, 2048, 16384),
        spec("decode", 16, 2048, 2048, 16384),  # decode carries no prior
        spec("decode", 32, 0, 2048, 16384),  # above the C16384 cap of 16
        spec("decode", 16, 0, 2048, 32768),  # above the C32768 cap of 8
        spec("decode", 6, 0, 2048, 32768),  # not a power of two
        spec("prefill", 2048, 0, 2048, 16384) | {"kv_blocks": 6783},
        spec("prefill", 2048, 0, 2048, 16384) | {"dtype": "fp16"},
        spec("mixed", 2048, 0, 2048, 16384),
    ],
)
def test_segmented_scope_rejects_out_of_contract_shapes(args):
    with pytest.raises(ValueError):
        validate_segmented_shape(**args)


class Tokenizer:
    """One token per word, so the subject (4th word) identifies the prompt."""

    def encode(self, text):
        return text.split()


def test_long_context_validation_workload_fills_context_within_kv_capacity():
    """Catches a prompt generator that cannot fill C32768, a precision shape that differs
    from the case actually run, and a C32768 validation call needing a B16 decode graph."""
    for context, segment, batch, bucket in ((16384, 2048, 13, 16), (32768, 1024, 6, 8)):
        plan = {
            "context": context,
            "segment_size": segment,
            "specs": [spec("prefill", segment, 0, segment, context)],
        }
        cases = cases_for_plan(plan, Tokenizer())
        assert {case["bucket"] for case in cases} == {bucket}
        assert all(len(ids) == context - 8 for case in cases for ids in case["prompts"])
        shape = precision_shape(plan["specs"][0], context)
        assert shape == f"b{batch}-s{context - 8}"
        assert {case["shape_id"] for case in cases} == {shape}
        subjects = {ids[3] for case in cases for ids in case["prompts"]}
        assert len(subjects) == 8


def test_segment_timing_calls_replay_each_accepted_subject_alone():
    """Catches timing calls whose prompt or expected tokens belong to another request."""
    context = 32768
    plan = {
        "context": context,
        "segment_size": 2048,
        "specs": [spec("prefill", 2048, 0, 2048, context), spec("decode", 8, 0, 2048, context)],
    }
    cases = cases_for_plan(plan, Tokenizer())
    timing = segment_timing_cases(plan, cases)
    by_id = {
        f"{case['id']}-slot{slot}": ids
        for case in cases
        for slot, ids in enumerate(case["prompts"])
    }
    assert len(timing) == 8
    assert {case["prompts"][0][3] for case in timing} == {ids[3] for ids in by_id.values()}
    for case in timing:
        assert len(case["prompts"]) == 1 and case["segment_timing"]
        assert by_id[case["expected_case_id"]] == case["prompts"][0]
        assert case["bucket"] == 8  # the smallest compiled decode graph (no B1 requested)
    assert segment_timing_cases({k: v for k, v in plan.items() if k != "segment_size"}, cases) == []


def test_segmented_engine_config_uses_segment_budgets_and_whole_forward_is_unchanged(tmp_path):
    """Catches a segmented engine still budgeting the whole context, or drift in the
    whole-forward engine arguments."""
    whole = {"context": 8192, "model": "m", "specs": [spec("prefill", 8192, 0, 0, 8192)]}
    cases = [{"bucket": 16}]
    assert engine_config(whole, cases, tmp_path, "accuracy") == dict(
        model="m",
        dtype="bfloat16",
        tensor_parallel_size=4,
        max_num_seqs=16,
        max_model_len=8192,
        max_num_batched_tokens=8192,
        enable_prefix_caching=False,
        seed=0,
        additional_config={
            "neuron_config": {
                "num_batched_tokens_buckets": [8192],
                "num_seqs_buckets": [1, 16],
                "debug_logits_dir": str(tmp_path / "raw-logits"),
            }
        },
        num_gpu_blocks_override=6782,
    )
    segmented = whole | {"context": 16384, "segment_size": 2048}
    config = engine_config(segmented, cases, tmp_path, "profile")
    nc = config["additional_config"]["neuron_config"]
    assert config["max_model_len"] == 16384
    assert config["max_num_batched_tokens"] == 2048
    assert nc["num_batched_tokens_buckets"] == [2048]
    assert nc["kv_segment_size_buckets"] == [2048]
    assert nc["num_seqs_buckets"] == [1, 16]
    assert config["profiler_config"] == {"profiler": "cuda"}


def test_prefill_only_segmented_profile_runs_only_accepted_timing_calls():
    """Catches re-timing the large canonical calls when only prefill rows are requested."""
    plan = {"specs": [spec("prefill", 2048, 0, 2048, 16384)]}
    canonical = [{"shape_id": "b13-s16376"}]
    timing = [{"shape_id": "b13-s16376", "segment_timing": True}]
    passed = {"by_shape": {"b13-s16376": {"passed": True}}}
    assert profile_cases(plan, canonical, timing, passed) == timing
    with_decode = {"specs": plan["specs"] + [spec("decode", 16, 0, 2048, 16384)]}
    assert profile_cases(with_decode, canonical, timing, passed) == canonical + timing
    with pytest.raises(ValueError, match="no numerically accepted"):
        profile_cases(plan, canonical, timing, {"by_shape": {"b13-s16376": {"passed": False}}})


def test_reference_attention_switches_to_row_chunked_eager_only_above_8192_tokens():
    """Catches changing the cached eager contract (and its keys) for existing contexts."""
    short = reference_contract({"x": 1}, "fp32", [1] * 8192)
    assert short == {
        "identity": {"x": 1},
        "attention": "eager",
        "dtype": "fp32",
        "ids": [1] * 8192,
        "positions": 8,
    }
    long = reference_contract({"x": 1}, "fp32", [1] * 8193)
    assert long["attention"] == "eager_rowchunk1024"
    assert reference_contract({"x": 1}, "bf16", [1] * 32767)["attention"] == "eager_rowchunk1024"


def segmented_metadata(tokens, cached=(1, 1), table=(1, 512), pools=64):
    rows = [
        ((tokens,), "int64"),
        ((tokens,), "int32"),
        ((), "int32"),
        ((32064, 4096), "bfloat16"),
        ((4096,), "bfloat16"),
        (cached, "int32"),
        ((6782, 2, 32, 128), "bfloat16"),
        ((tokens,), "int64"),
        ((4096, 1536), "bfloat16"),
        (table, "int32"),
    ]
    rows += [((6782, 2, 32, 128), "bfloat16")] * (pools - 1)
    return "Example Inputs Metadata\n" + "".join(
        f"\nInput {i}:\n  Shape: ({', '.join(map(str, shape))}{',' if len(shape) == 1 else ''})"
        f"\n  Dtype: {dtype}\n  Device: meta\n"
        for i, (shape, dtype) in enumerate(rows)
    )


def test_segmented_geometry_binds_segment_size_and_block_table_width():
    """Catches timing a graph of another segment size, context or KV layout as a segment,
    or an unsegmented whole-context prefill graph inside a segmented run."""
    assert segmented_model_geometry(segmented_metadata(2048), 16384, 2048) == ("prefill", 2048)
    assert segmented_model_geometry(segmented_metadata(1024, table=(1, 1024)), 32768, 1024) == (
        "prefill",
        1024,
    )
    for metadata, context, segment in (
        (segmented_metadata(2048), 16384, 1024),
        (segmented_metadata(2048), 32768, 2048),
        (segmented_metadata(2048, pools=63), 16384, 2048),
    ):
        with pytest.raises(ValueError):
            segmented_model_geometry(metadata, context, segment)
    whole_prefill = (
        "Input 0:\n Shape: (16384,)\nInput 5:\n Shape: (6782, 2, 32, 128)\n Dtype: bfloat16\n"
        "Input 6:"
    )
    with pytest.raises(ValueError, match="unsegmented"):
        segmented_model_geometry(whole_prefill, 16384, 2048)
    decode = (
        "Input 0:\n Shape: (16,)\n Dtype: int64\nInput 5:\n Shape: (16, 512)\n Dtype: int32\n"
        "Input 6:\n Shape: (4096, 1536)\nInput 9:\n Shape: (6782, 2, 32, 128)\n Dtype: bfloat16\n"
    )
    assert segmented_model_geometry(decode, 16384, 2048) == ("decode", 16)


MODELS = {"seg": ("prefill", 2048), "d1": ("decode", 1), "d16": ("decode", 16)}


def call(start, executions, batch, bucket, timing=False):
    """Events of one public call: ``executions`` lists (model, duration_ns) in device order."""
    events, t = [], start
    for model, duration in executions:
        exec_id = len(EXEC_IDS)
        EXEC_IDS.append(exec_id)
        for core in range(8):
            events.append(
                dict(
                    timestamp=t + core,
                    duration=duration,
                    exec_id=exec_id,
                    model_id=model,
                    process_id=core // 2,
                    device_core_idx=core,
                    model_name=f"/compile_cache/{model}/graph.neff",
                )
            )
        t += duration + 100
    request = {"start_epoch_ns": start, "stop_epoch_ns": t, "batch": batch, "bucket": bucket}
    if timing:
        request["segment_timing"] = True
    return events, request


EXEC_IDS: list[int] = []


def test_segment_ordinal_maps_to_prior_per_batch1_timing_call():
    """Catches assigning a prior from the call's global execution order instead of the lone
    request's segment ordinal, or timing prefills of multi-request calls."""
    segments = segment_count(5000 + 8, 2048)  # three segments, the last partly padding
    assert segments == 3
    events, requests = [], []
    for index, start in enumerate((0, 100_000)):
        # Segment k of this call takes 1000 + 200 k + 10 index ns (core skew adds 7 ns).
        prefill = [("seg", 1000 + 200 * k + 10 * index) for k in range(segments)]
        e, r = call(start, prefill + [("d1", 50)] * 7, 1, 1, timing=True)
        events += e
        requests.append(r)
    # A batch-2 canonical call: its six prefills cover coverage but are never timed.
    e, r = call(200_000, [("seg", 9999)] * 6 + [("d16", 70)] * 7, 2, 16)
    events += e
    requests.append(r)
    measured, records = measure_segmented_trace(events, requests, MODELS, set(range(8)), segments)
    for k in range(segments):
        expected = sorted([(1000 + 200 * k + 7) / 1e6, (1010 + 200 * k + 7) / 1e6])
        assert sorted(measured[("prefill", 2048, 2048 * k)]) == pytest.approx(expected)
    assert set(measured) == {
        ("prefill", 2048, 0),
        ("prefill", 2048, 2048),
        ("prefill", 2048, 4096),
        ("decode", 1, 0),
        ("decode", 16, 0),
    }
    assert len(measured[("decode", 16, 0)]) == 7
    untimed = [r for r in records if r["phase"] == "prefill" and r["prior_tokens"] is None]
    assert len(untimed) == 6

    with pytest.raises(ValueError, match="missing a prefill segment"):
        measure_segmented_trace(events[8:], requests, MODELS, set(range(8)), segments)
    early, r = call(
        400_000,
        [("seg", 1000), ("d1", 50)] + [("seg", 1000)] * 2 + [("d1", 50)] * 6,
        1,
        1,
        timing=True,
    )
    with pytest.raises(ValueError, match="decode ran before"):
        measure_segmented_trace(early, [r], MODELS, set(range(8)), segments)


def test_segmented_work_without_prior_is_the_whole_forward_prefill_of_one_segment():
    """Catches double-counting the segment's own causal attention or the KV it writes."""
    for segment in (512, 1024, 2048, 4096):
        assert estimate_segmented_work("prefill", segment, 0, 32768) == estimate_work(
            "prefill", segment, segment
        )
    base = estimate_segmented_work("prefill", 2048, 0, 16384)
    later = estimate_segmented_work("prefill", 2048, 4096, 16384)
    prior_attention = 32 * 2 * 8 * 128 * 2 * 2048 * 4096
    assert later["attention_flops_per_rank"] - base["attention_flops_per_rank"] == prior_attention
    assert (
        later["persistent_operand_bytes_per_rank"] - base["persistent_operand_bytes_per_rank"]
    ) == 32768 * 4096
    assert estimate_segmented_work("decode", 8, 0, 32768) == estimate_work("decode", 8, 32768)
    with pytest.raises(ValueError):
        estimate_segmented_work("decode", 8, 2048, 32768)


def test_profile_stage_checks_timing_calls_against_the_accepted_subject_tokens(
    tmp_path, monkeypatch
):
    """Catches timing a batch-1 call whose production tokens were never validated, or
    looking its tokens up under the timing call's own (unvalidated) case id."""
    import json
    import sys
    from types import SimpleNamespace

    from profiling.runners.neuron import vllm_forward_engine as engine

    context = 16384
    plan = {
        "context": context,
        "segment_size": 2048,
        "model": "unused",
        "profile_repeats": 1,
        "specs": [spec("prefill", 2048, 0, 2048, context)],
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    cases = cases_for_plan(plan, Tokenizer())
    accepted = {}
    rows = []
    for case in cases:
        for slot, ids in enumerate(case["prompts"]):
            accepted.setdefault(tuple(ids), [len(ids) % 7, len(accepted)] * 4)
            rows.append({"case_id": f"{case['id']}-slot{slot}", "token_ids": accepted[tuple(ids)]})
    (tmp_path / "accuracy-outputs.json").write_text(json.dumps(rows))
    (tmp_path / "precision.json").write_text(
        json.dumps({"by_shape": {cases[0]["shape_id"]: {"passed": True}}})
    )
    calls = []

    class LLM:
        def __init__(self, **config):
            self.config = config
            self.flip = False

        def generate(self, prompts, params, **kwargs):
            calls.append(len(prompts))
            outputs = []
            for prompt in prompts:
                tokens = list(accepted[tuple(prompt["prompt_token_ids"])])
                if self.flip:
                    tokens[-1] += 1
                outputs.append(
                    SimpleNamespace(
                        prompt_token_ids=prompt["prompt_token_ids"],
                        request_id="1",
                        outputs=[SimpleNamespace(token_ids=tokens)],
                    )
                )
            return outputs

        def start_profile(self):
            pass

        def stop_profile(self):
            LLM.instance = self

    tokenizer = SimpleNamespace(from_pretrained=lambda *_, **__: Tokenizer())
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=tokenizer))
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(
            LLM=LLM,
            SamplingParams=lambda **kwargs: kwargs,
        ),
    )
    engine.main(tmp_path, "profile")
    profiled = json.loads((tmp_path / "profiled-outputs.json").read_text())
    assert calls == [1] * 16  # 8 warmup + 8 timed batch-1 calls; no canonical B13 call
    assert all(row["segment_timing"] and row["batch"] == 1 for row in profiled)
    config = json.loads((tmp_path / "profile-config.json").read_text())
    assert config["max_num_batched_tokens"] == 2048

    original = LLM.generate

    def flipped(self, prompts, params, **kwargs):
        self.flip = True
        return original(self, prompts, params, **kwargs)

    monkeypatch.setattr(LLM, "generate", flipped)
    with pytest.raises(RuntimeError, match="production sampling differs"):
        engine.main(tmp_path, "profile")
