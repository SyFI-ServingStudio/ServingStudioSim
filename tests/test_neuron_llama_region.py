"""Model/head region split: scope refusal, structural proof, attribution and gates."""

import operator

import numpy as np
import pytest
import torch

from profiling.runners.neuron.vllm_forward_work import estimate_region_work, estimate_work
from profiling.runners.neuron.vllm_region import (
    case_equivalence,
    compiled_shapes,
    equivalence_verdict,
    timing_verdict,
    validate_region_shape,
)
from profiling.runners.neuron.vllm_region_partition import (
    RegionPartitionError,
    check_kv_aliases,
    partition_model_and_head,
    signature,
)
from profiling.runners.neuron.vllm_region_trace import measure_region_trace

BASE = dict(
    phase="decode",
    token_bucket=16,
    max_model_len=512,
    kv_blocks=6782,
    block_size=32,
    tp_size=4,
    dtype="bf16",
    region="head",
)


def test_scope_accepts_only_the_six_verified_rows():
    """Catches compiling an unverified split before any gate could reject it."""
    for phase, bucket in (("prefill", 512), ("decode", 1), ("decode", 16)):
        for region in ("model", "head"):
            validate_region_shape(**(BASE | dict(phase=phase, token_bucket=bucket, region=region)))
    for changed in (
        {"region": "lm_head"},
        {"token_bucket": 8},
        {"max_model_len": 128, "token_bucket": 16},
        {"phase": "prefill", "token_bucket": 512, "max_model_len": 2048},
        {"kv_blocks": 6783},
        {"block_size": 64},
        {"tp_size": 2},
        {"dtype": "fp16"},
    ):
        with pytest.raises(ValueError):
            validate_region_shape(**(BASE | changed))


def test_compiled_shapes_include_the_engine_decode_bucket_one():
    """Catches a binding check that ignores graphs vLLM compiles but rows do not request."""
    prefill = [dict(phase="prefill", token_bucket=512)]
    assert compiled_shapes(prefill, 512) == {("prefill", 512), ("decode", 1), ("decode", 16)}


def test_work_split_is_exhaustive_and_head_is_lm_head_only():
    """Catches double-counting or dropping work at the region boundary."""
    for phase, bucket in (("prefill", 512), ("decode", 1), ("decode", 16)):
        whole = estimate_work(phase, bucket, 512)
        model = estimate_region_work("model", phase, bucket, 512)
        head = estimate_region_work("head", phase, bucket, 512)
        for name in whole:
            assert model[name] + head[name] == whole[name]
        rows = 1 if phase == "prefill" else bucket
        assert head["contraction_flops_per_rank"] == 2 * rows * 4096 * 32064


# --- structural partition on a synthetic Dynamo-like FX graph -----------------

MODEL = "L['self'].model"


def _label(node, *labels):
    node.meta["nn_module_stack"] = {label: (label, object) for label in labels}
    return node


def _stock_like_graph(layers=2, head_reads_cache=False, label_norm=True):
    graph = torch.fx.Graph()
    ids = graph.placeholder("l_kwargs_input_ids_")
    ids.meta["example_value"] = torch.zeros(4, dtype=torch.int32)
    embed = graph.placeholder("l_embed_weight_")
    caches = [
        (graph.placeholder(f"l_layers_{i}_k_cache"), graph.placeholder(f"l_layers_{i}_v_cache"))
        for i in range(layers)
    ]
    lm_head = graph.placeholder("l_lm_head_weight_")
    graph.call_function(torch._C._set_grad_enabled, (False,))
    hidden = _label(
        graph.call_function(torch.nn.functional.embedding, (ids, embed)),
        MODEL,
        f"{MODEL}.embed_tokens",
    )
    for i, (k_cache, v_cache) in enumerate(caches):
        layer = f"{MODEL}.layers.{i}"
        for cache in (k_cache, v_cache):
            _label(graph.call_method("index_copy_", (cache, 0, ids, hidden)), MODEL, layer)
        hidden = _label(graph.call_function(operator.add, (hidden, hidden)), MODEL, layer)
    norm_labels = (MODEL, f"{MODEL}.norm") if label_norm else (MODEL,)
    hidden = _label(graph.call_function(torch.rsqrt, (hidden,)), *norm_labels)
    rows = graph.call_function(torch.index_select, (hidden, 0, ids))
    if head_reads_cache:
        rows = graph.call_function(operator.add, (rows, caches[0][0]))
    logits = _label(
        graph.call_function(torch.nn.functional.linear, (rows, lm_head)), "L['self'].lm_head"
    )
    tokens = graph.call_function(torch.argmax, (logits,), {"dim": -1})
    graph.call_function(torch._C._set_grad_enabled, (True,))
    graph.output((tokens,))
    return torch.fx.GraphModule(torch.nn.Module(), graph)


def test_partition_keeps_every_stock_operation_and_kv_mutation_in_model():
    """Catches a split that moves, duplicates, or rewrites any stock operation."""
    gm = _stock_like_graph()
    original = {
        node.name: signature(node)
        for node in gm.graph.nodes
        if node.op not in ("placeholder", "get_attr", "output")
    }
    split, report = partition_model_and_head(gm, num_layers=2)
    assert report["phase"] == "prefill"
    assert report["token_bucket"] == 4
    model, head = report["regions"]["model"], report["regions"]["head"]
    assert sorted(model["cache_inputs"]) == sorted(
        f"l_layers_{i}_{kind}_cache" for i in range(2) for kind in ("k", "v")
    )
    assert head["cache_inputs"] == []
    assert model["operations"] + head["operations"] == len(original)
    head_ops = [
        node.target
        for node in split.get_submodule("submod_1").graph.nodes
        if node.op == "call_function"
    ]
    assert head_ops == [
        torch.index_select,
        torch.nn.functional.linear,
        torch.argmax,
        torch._C._set_grad_enabled,
    ]
    assert head["removed_grad_context"] == ["_set_grad_enabled"]  # split_module copy only
    for index in range(2):
        for node in split.get_submodule(f"submod_{index}").graph.nodes:
            if node.op not in ("placeholder", "get_attr", "output"):
                assert signature(node) == original[node.meta["stock_original_name"]]


@pytest.mark.parametrize(
    ("graph", "layers", "message"),
    [
        (dict(), 3, "expected 3 layers"),
        (dict(label_norm=False), 2, "final norm"),
        (dict(head_reads_cache=True), 2, "head region owns a KV cache input"),
    ],
)
def test_partition_refuses_ownership_violations(graph, layers, message):
    """Catches silently emitting a region pair that breaks the boundary contract."""
    with pytest.raises(RegionPartitionError, match=message):
        partition_model_and_head(_stock_like_graph(**graph), num_layers=layers)


def test_lowered_alias_check_requires_every_model_kv_and_no_head_alias():
    """Catches lowering that drops a KV in-place update or aliases head inputs."""
    _, report = partition_model_and_head(_stock_like_graph(), num_layers=2)
    caches = report["regions"]["model"]["cache_inputs"]
    with pytest.raises(RegionPartitionError, match="KV alias"):
        check_kv_aliases(report, {"model": {"0": caches[0]}, "head": {}})
    with pytest.raises(RegionPartitionError, match="head region aliases"):
        check_kv_aliases(
            report, {"model": {str(i): c for i, c in enumerate(caches)}, "head": {"0": "x"}}
        )
    check_kv_aliases(report, {"model": {str(i): c for i, c in enumerate(caches)}, "head": {}})


# --- native attribution ----------------------------------------------------------

INFO = {
    "mp": {"region": "model", "phase": "prefill", "token_bucket": 512},
    "hp": {"region": "head", "phase": "prefill", "token_bucket": 512},
    "md": {"region": "model", "phase": "decode", "token_bucket": 16},
    "hd": {"region": "head", "phase": "decode", "token_bucket": 16},
}


def _execution(exec_id, graph, start, duration):
    return [
        dict(
            timestamp=start + core,
            duration=duration,
            exec_id=exec_id,
            model_id=graph,
            process_id=core // 2,
            device_core_idx=core,
            model_name=f"/compile_cache/{graph}/graph.neff",
        )
        for core in range(8)
    ]


def _trace(gap=100):
    events, clock = [], 0
    for step in range(8):
        model, head = ("mp", "hp") if step == 0 else ("md", "hd")
        events += _execution(2 * step, model, clock, 50)
        events += _execution(2 * step + 1, head, clock + gap, 10)
        clock += 1000
    return events


REQUESTS = [{"start_epoch_ns": 0, "stop_epoch_ns": 10_000, "batch": 1, "bucket": 16}]


def test_region_time_is_each_neff_union_per_ordered_forward_pair():
    """Catches charging host gaps or the partner region to a region's time."""
    measured, records = measure_region_trace(_trace(), REQUESTS, INFO, set(range(8)))
    assert measured[("model", "decode", 16)] == [57 / 1e6] * 7
    assert measured[("head", "decode", 16)] == [17 / 1e6] * 7
    assert measured[("model", "prefill", 512)] == [57 / 1e6]
    assert len(records) == 8
    assert records[1]["whole_span_ms"] == 117 / 1e6


def test_region_attribution_refuses_overlap_order_orphans_and_unbound_graphs():
    """Catches attributing time when the model->head pairing is not proven."""
    cores = set(range(8))
    with pytest.raises(ValueError, match="overlap"):
        measure_region_trace(_trace(gap=40), REQUESTS, INFO, cores)
    swapped = {**INFO, "md": INFO["hd"], "hd": INFO["md"]}
    with pytest.raises(ValueError, match="ordered model then head"):
        measure_region_trace(_trace(), REQUESTS, swapped, cores)
    with pytest.raises(ValueError, match="partner"):
        measure_region_trace(_trace()[:-8], REQUESTS, INFO, cores)
    with pytest.raises(ValueError, match="not bound"):
        measure_region_trace(_trace(), REQUESTS, {k: INFO[k] for k in ("mp", "hp", "md")}, cores)
    with pytest.raises(ValueError, match="missing output"):
        measure_region_trace(_trace()[:-16], REQUESTS, INFO, cores)


# --- gates -----------------------------------------------------------------------


def test_equivalence_gate_requires_tokens_and_bounded_fp32_error_ratio():
    """Catches accepting a split that changes tokens or degrades precision >10%."""
    rng = np.random.default_rng(0)
    fp32 = rng.normal(size=(8, 32))
    stock = fp32 + 0.1 * rng.normal(size=fp32.shape)
    same = case_equivalence(stock.copy(), stock, fp32)
    assert same["bit_identical"] and same["error_ratio"] == 1.0
    worse = case_equivalence(fp32 + 1.2 * (stock - fp32), stock, fp32)
    assert worse["error_ratio"] == pytest.approx(1.2)
    slightly = case_equivalence(fp32 + 1.05 * (stock - fp32), stock, fp32)
    ok = {"tokens_identical": True, **same}
    assert equivalence_verdict([ok, {"tokens_identical": True, **slightly}])["passed"]
    verdict = equivalence_verdict([ok, {"tokens_identical": True, **worse}])
    assert not verdict["passed"] and verdict["bit_identical_cases"] == 1
    assert not equivalence_verdict([{**ok, "tokens_identical": False}])["passed"]


def test_timing_gate_compares_sum_of_region_medians_to_stock_median():
    """Catches summing per-forward totals or comparing against another shape."""
    regions = {
        ("model", "decode", 1): [9.0, 9.2, 100.0],
        ("head", "decode", 1): [0.9, 1.0, 0.8],
    }
    verdict = timing_verdict(regions, {("decode", 1): [9.9, 10.0, 10.1]}, {("decode", 1)})
    row = verdict["decode:1"]
    assert row["sum_region_medians_ms"] == pytest.approx(10.1)
    assert row["passed"] and row["error_pct"] == pytest.approx(1.0)
    failing = timing_verdict(regions, {("decode", 1): [9.5]}, {("decode", 1)})
    assert not failing["decode:1"]["passed"]
