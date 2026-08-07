"""Golden tests for the optimal necessary-work labeler (model/work/).

Every expected value below is HAND-DERIVED from the Llama3-8B config and asserted
against the labeler — the point is to catch a formula drift, so the expectations are
written out independently rather than recomputed through the code under test.

Llama3-8B: L=32, hidden=4096, num_qo=32, num_kv=8, head_dim=128, intermediate=14336,
vocab=128256, bf16 (2 B), lm_head untied.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model.work import Workload, load_model
from model.work import floors as work_floors
from model.work.core import AttnInteraction
from model.work.parameter_counts import compute_parameter_counts
from model.work.registry import UnknownArchitecture, build_model

CONFIG = Path(__file__).resolve().parents[1] / "model" / "config" / "llama3_8b.json"

L = 32
HIDDEN = 4096
HEAD_DIM = 128
NUM_QO = 32
NUM_KV = 8
INTERMEDIATE = 14336
VOCAB = 128256

# per-layer weight params
QKV = (NUM_QO + 2 * NUM_KV) * HEAD_DIM * HIDDEN  # 6144 * 4096 = 25_165_824
O_PROJ = (NUM_QO * HEAD_DIM) * HIDDEN  # 4096 * 4096 = 16_777_216
ATTN_LAYER = QKV + O_PROJ  # 41_943_040
FFN_LAYER = (2 * INTERMEDIATE) * HIDDEN + HIDDEN * INTERMEDIATE  # 176_160_768
EMBED = VOCAB * HIDDEN  # 525_336_576
NORMS = (2 * L + 1) * HIDDEN


@pytest.fixture(scope="module")
def model():
    return load_model(CONFIG)


def test_params(model):
    label = model.label(Workload.causal_lm(decode=[4096], sampled=1))
    activated = (ATTN_LAYER + FFN_LAYER) * L + NORMS
    total = activated + EMBED + EMBED  # + embedding + untied lm_head
    assert label.params["activated"]["layers"] == activated
    assert label.params["activated"]["with_embed_head"] == total  # dense: layers == total minus io
    assert label.params["total"] == total
    assert label.params["total"] == 8_030_261_248  # includes all RMSNorm scales
    breakdown = label.params["breakdown"]
    assert breakdown["attn"] == ATTN_LAYER * L
    assert breakdown["ffn"] == FFN_LAYER * L
    assert breakdown["embedding"] == EMBED
    assert breakdown["lm_head"] == EMBED
    assert breakdown["norm"] == NORMS
    assert breakdown["experts"] == 0 and breakdown["shared"] == 0


def test_parameter_count_subprocess_contract():
    counts = compute_parameter_counts(CONFIG)
    assert counts == {
        "total": 8_030_261_248,
        "active": 8_030_261_248,
        "active_layers": 6_979_588_096,
        "active_definition": "with_embed_head",
    }


def test_floor_batch_isolates_one_heterogeneous_level(monkeypatch, model):
    pool_specs = {
        "main": {"config": str(CONFIG), "gpu": "H200", "dtype": "bf16"},
        "other": {"config": "different-model.json", "gpu": "H200", "dtype": "bf16"},
    }
    monkeypatch.setattr(work_floors, "_pool_specs", lambda _log_dir: pool_specs)
    monkeypatch.setattr(work_floors, "_model", lambda _config_path: model)
    totals = {
        "matmul_tokens": 1,
        "prefill_tokens": 0,
        "decode_passes": 1,
        "prefill_pairs": 0,
        "prefill_cached": 0,
        "decode_kv": 1,
        "prefill_requests": 0,
    }

    result = work_floors.compute_floors(Path("unused"), {"cluster": totals, "main": totals})

    assert "error" in result["cluster"]
    assert result["main"]["necessary"] > 0
    assert result["main"]["segmented"] >= result["main"]["necessary"]


@pytest.mark.parametrize("force_direct_fallback", [False, True])
def test_locked_composition_evaluates_each_shape_before_addition(
    monkeypatch, model, force_direct_fallback
):
    spec = {"config": str(CONFIG), "gpu": "H200", "dtype": "bf16"}
    monkeypatch.setattr(work_floors, "_pool_specs", lambda _log_dir: {"main": spec})
    monkeypatch.setattr(work_floors, "_model", lambda _config_path: model)
    if force_direct_fallback:
        monkeypatch.setattr(work_floors, "_segment_work_matches", lambda *_values: False)
    decode_shape = {
        "matmul_tokens": 1,
        "prefill_tokens": 0,
        "decode_passes": 1,
        "prefill_pairs": 0,
        "prefill_cached": 0,
        "decode_kv": 4096,
        "prefill_requests": 0,
    }
    long_decode_shape = {**decode_shape, "decode_kv": 65536}
    prefill_shape = {
        "matmul_tokens": 128,
        "prefill_tokens": 128,
        "decode_passes": 0,
        "prefill_pairs": 128 * 129 // 2,
        "prefill_cached": 0,
        "decode_kv": 0,
        "prefill_requests": 1,
    }
    weighted_shapes = [
        {"occurrences": 3, "totals": decode_shape},
        {"occurrences": 4, "totals": long_decode_shape},
        {"occurrences": 2, "totals": prefill_shape},
    ]

    result = work_floors.compute_locked_compositions(Path("unused"), {"main/0": weighted_shapes})[
        "main/0"
    ]

    expected_fused = 0.0
    expected_segmented = 0.0
    for weighted_shape in weighted_shapes:
        label = model.label(work_floors._aggregate_workload(weighted_shape["totals"]))
        compute_ms, memory_ms, _bound = label.roofline_ms("H200", "bf16")
        expected_fused += max(compute_ms, memory_ms) / 1e3 * weighted_shape["occurrences"]
        expected_segmented += (
            label.segmented_lower_bound_ms("H200", "bf16") / 1e3 * weighted_shape["occurrences"]
        )
    assert result["necessary"] == pytest.approx(expected_fused)
    assert result["segmented"] == pytest.approx(expected_segmented)
    assert sum(segment["necessary"] for segment in result["segments"]) == pytest.approx(
        expected_segmented
    )
    assert result["composition"] == {
        "unique_shapes": 3,
        "iterations": 9,
        "affine_bases": int(not force_direct_fallback),
        "direct_fallback_bases": int(force_direct_fallback),
    }


def test_decode_flop_buckets(model):
    # 1 decode token at kv_len=4096: matmul_tokens=1, causal pairs = kv = 4096.
    label = model.label(Workload.causal_lm(decode=[4096], sampled=1))
    assert label.flops["attn_proj"] == 2 * 1 * ATTN_LAYER * L  # 2_684_354_560
    assert label.flops["ffn"] == 2 * 1 * FFN_LAYER * L  # 11_274_289_152
    assert label.flops["attn_internal"] == 4 * NUM_QO * HEAD_DIM * 4096 * L  # 2_147_483_648
    assert label.flops["lm_head"] == 2 * 1 * HIDDEN * VOCAB  # 1_050_673_152
    assert label.flops["router"] == 0
    assert label.flops_total == 17_156_800_512


def test_decode_kv_bytes(model):
    # Cached reads plus the compulsory new-token K/V write cover kv_len tokens.
    label = model.label(Workload.causal_lm(decode=[4096], sampled=1))
    per_token = 2 * NUM_KV * HEAD_DIM * 2  # 4096 bytes/token/layer
    assert label.bytes["kv"] == per_token * 4096 * L  # 536_870_912


def test_prefill_uses_causal_triangle(model):
    # append=8 from empty: pairs must be the triangle 8*9/2 = 36, NOT the square 64.
    label = model.label(Workload.causal_lm(prefill=[(8, 0)], sampled=1))
    triangle_pairs = 8 * 9 // 2
    assert label.flops["attn_internal"] == 4 * NUM_QO * HEAD_DIM * triangle_pairs * L
    # a full-rectangle bug would give 8*8 = 64 pairs -> ~1.78x larger
    assert label.flops["attn_internal"] != 4 * NUM_QO * HEAD_DIM * 64 * L


def test_full_mask_aggregation_equals_stepwise_sum(model):
    """The `model.work.floors` optimality-floor aggregation trick, validated.

    `internal_flops` and `kv_bytes` are LINEAR sums over `wl.attn`
    (`per_pair·Σ pairs()` and `per_cached·Σ num_cached_key`), so a whole batch of
    per-step causal interactions collapses to ONE `mask="full"` interaction carrying
    the summed pair / cached counts — which is exactly what `floors.py` builds from the
    Rust-aggregated scalars instead of materializing 1.7-billion decode interactions.
    Here we prove the collapse is exact against the step-by-step `causal_lm` reference.
    """
    # A realistic mixed batch: two chunked-prefill requests + a spread of decode steps.
    prefill = [(512, 0), (128, 512), (300, 0)]  # (append_len, prefix_len)
    decode = [4096, 4097, 1, 2048, 33, 100000]  # per-step kv_len
    reference = Workload.causal_lm(
        prefill=prefill, decode=decode, sampled=len(decode) + len(prefill)
    )

    # Collapse prefill and decode each into a single full interaction (floors.py shape):
    # num_key = Σ pairs(), num_cached_key = Σ num_cached_key — over that side's steps.
    def collapse(interactions: list[AttnInteraction]) -> AttnInteraction:
        total_pairs = sum(interaction.pairs() for interaction in interactions)
        total_cached = sum(interaction.num_cached_key for interaction in interactions)
        # pairs() for full = num_query·num_key, so num_query=1 makes pairs() == total_pairs.
        assert total_pairs == int(total_pairs)  # integer for integer step geometry
        return AttnInteraction(1, int(total_pairs), int(total_cached), "full")

    prefill_interactions = reference.attn[: len(prefill)]
    decode_interactions = reference.attn[len(prefill) :]
    aggregate = Workload(
        matmul_tokens=reference.matmul_tokens,
        head_positions=reference.head_positions,
        attn=[collapse(prefill_interactions), collapse(decode_interactions)],
        attention_step_count=len(reference.attn),
        attention_step_count_by_phase={
            "prefill": len(prefill_interactions),
            "decode": len(decode_interactions),
        },
        attention_tokens_by_phase={
            "prefill": sum(append_len for append_len, _prefix_len in prefill),
            "decode": len(decode),
        },
    )

    ref_label = model.label(reference)
    agg_label = model.label(aggregate)
    # The attention terms are the whole point of the trick — they must be bit-identical.
    assert agg_label.flops["attn_internal"] == ref_label.flops["attn_internal"]
    assert agg_label.bytes["kv"] == ref_label.bytes["kv"]
    # matmul_tokens / head_positions match, so every other bucket does too -> totals equal.
    assert agg_label.flops_total == ref_label.flops_total
    assert agg_label.bytes_total == ref_label.bytes_total


def test_roofline_memory_bound_decode(model):
    label = model.label(Workload.causal_lm(decode=[4096] * 256, sampled=256))
    compute_ms, memory_ms, bound = label.roofline_ms("H200", "bf16")
    assert bound == "memory"
    # memory floor = total_bytes / 4800 GB/s
    assert memory_ms == pytest.approx(label.bytes_total / 4800e9 * 1e3, rel=1e-9)
    assert compute_ms == pytest.approx(label.flops_total / 990e12 * 1e3, rel=1e-9)


def test_segments_roll_up_to_buckets(model):
    label = model.label(Workload.causal_lm(decode=[4096] * 256, sampled=256))
    assert sum(s.flops_total for s in label.segments) == pytest.approx(label.flops_total)
    assert sum(s.bytes_total for s in label.segments) == pytest.approx(label.bytes_total)


def test_segmented_bound_is_tighter_than_global(model):
    label = model.label(Workload.causal_lm(decode=[4096] * 256, sampled=256))
    compute_ms, memory_ms, _ = label.roofline_ms("H200", "bf16")
    global_floor = max(compute_ms, memory_ms)
    segmented = label.segmented_lower_bound_ms("H200", "bf16")
    # Σ per-segment max(compute, memory) >= max(Σcompute, Σmemory).
    assert segmented >= global_floor - 1e-9


def test_decode_segments_are_memory_bound(model):
    # 1 decode token: GEMMs read weights for a single token (memory-bound), and
    # attention reads a large KV cache (memory-bound).
    label = model.label(Workload.causal_lm(decode=[4096], sampled=1))
    rows = {row["name"]: row for row in label.segment_rows("H200", "bf16")}
    assert rows["qkv"]["bound"] == "memory"
    assert rows["attn.decode"]["bound"] == "memory"


def test_llama_semantic_segments_include_attention_phases_and_cache_write(model):
    label = model.label(Workload.causal_lm(prefill=[(8, 0)], decode=[4096], sampled=2))
    rows = {segment.name: segment for segment in label.segments}
    assert {"attn.prefill", "attn.decode", "kv_cache_append"} <= rows.keys()
    assert rows["attn.prefill"].flops_total > 0
    assert rows["attn.decode"].bytes_total > 0
    assert rows["kv_cache_append"].flops_total == 0
    assert rows["kv_cache_append"].bytes_total == 2 * NUM_KV * HEAD_DIM * 2 * 9 * L
    assert rows["input_norm"].bytes_total == HIDDEN * 2 * L
    assert rows["post_norm"].bytes_total == HIDDEN * 2 * L
    assert rows["final_norm"].bytes_total == HIDDEN * 2
    assert rows["input_norm"].flops_total == 0


def test_replicated_batch_amortizes_weights_but_preserves_per_iteration_kv(model):
    replication_factor = 1_000
    base = {s.name: s for s in model.label(Workload.causal_lm(decode=[4096])).segments}
    replicated = {
        s.name: s
        for s in model.label(Workload.causal_lm(decode=[4096] * replication_factor)).segments
    }

    # Replication means more independent batch entries, never a longer context.
    assert replicated["qkv"].flops_total / replication_factor == base["qkv"].flops_total
    assert (
        replicated["attn.decode"].bytes_total / replication_factor
        == base["attn.decode"].bytes_total
    )
    assert (
        replicated["kv_cache_append"].bytes_total / replication_factor
        == base["kv_cache_append"].bytes_total
    )
    # Dense weights load once for the mega-batch and are amortized on normalization.
    assert replicated["qkv"].bytes_total == base["qkv"].bytes_total
    assert replicated["qkv"].bytes_total / replication_factor < base["qkv"].bytes_total


# --------------------------------------------------------------------------- #
# Phase 2: MoE (Qwen3-MoE)
# --------------------------------------------------------------------------- #

QWEN_FIXTURE = Path(__file__).resolve().parents[1] / "model" / "config" / "qwen3_235b.json"


def test_qwen3_235b_params_exact():
    # Hand-derived from the real Qwen3-235B-A22B config (moe_intermediate=1536, 94 all-MoE
    # layers, vocab 151936): per layer = attn 71_303_168 + router 524_288 + 128 experts *
    # 18_874_368 = 2_487_746_560; total = 2_487_746_560*94 + 2*(151936*4096); the
    # activated.layers figure counts attn + router + only top_k=8 experts per layer.
    label = load_model(QWEN_FIXTURE).label(Workload.causal_lm(decode=[4096], sampled=1))
    assert label.params["total"] == 235_092_836_352
    assert label.params["activated"]["layers"] == 20_945_305_600


def test_qwen3_235b_matches_model_card():
    # The config file is the real checkpoint, so it must land on the published headline
    # "235B / A22B" figures — an independent model-card oracle over the exact test above.
    label = load_model(QWEN_FIXTURE).label(Workload.causal_lm(decode=[4096], sampled=1))
    assert label.params["total"] == pytest.approx(235e9, rel=0.01)
    # "A22B" activated counts embedding + lm_head alongside the per-token layer params
    assert label.params["activated"]["with_embed_head"] == pytest.approx(22e9, rel=0.02)
    # the layers-only figure (per-token compute) is lower
    assert label.params["activated"]["layers"] == pytest.approx(21e9, rel=0.03)


def test_moe_expert_loading_is_balls_in_bins():
    from model.work.core import MatmulGroup
    from model.work.ffn.moe import MoE

    expert = next(g for g in MoE(4096, 1536, 128, 8).matmul_groups() if g.name == "expert_gate_up")
    assert expert.routed and expert.total_count == 128 and expert.activated_mult == 8
    assert expert.loaded_instances(1) < 8.0  # 8 draws -> a few collisions, <8 distinct
    assert 40 < expert.loaded_instances(8) < 60  # 64 draws -> ~50 distinct
    assert expert.loaded_instances(8192) == pytest.approx(128, abs=1e-6)  # all experts loaded
    # a non-routed group loads all instances regardless of batch size
    assert MatmulGroup("x", 100, 100, total_count=4).loaded_instances(1) == 4.0


def test_moe_flops_use_top_k():
    # expert (ffn bucket) FLOPs scale with top_k, not num_experts; router is nonzero.
    label = load_model(QWEN_FIXTURE).label(Workload.causal_lm(decode=[1], sampled=1))
    per_expert = (2 * 1536) * 4096 + 4096 * 1536  # gate_up + down = 18_874_368
    assert label.flops["ffn"] == 2 * 1 * 8 * per_expert * 94
    assert label.flops["router"] == 2 * 1 * 128 * 4096 * 94


def test_unknown_architecture_errors():
    with pytest.raises(UnknownArchitecture):
        build_model({"architectures": ["NotARealForCausalLM"], "num_hidden_layers": 1})


# --------------------------------------------------------------------------- #
# Hybrid attention (Qwen3.6-27B: 48 GatedDeltaNet linear + 16 gated-full layers)
# --------------------------------------------------------------------------- #

QWEN36 = Path(__file__).resolve().parents[1] / "model" / "config" / "qwen3_6_27b.json"

GLM52 = Path(__file__).resolve().parents[1] / "model" / "config" / "glm52.json"

# GLM-5.2 hand-derived geometry. q_absorb and v_up partition kv_b_proj's learned
# matrix into the W_UK and W_UV per-head views, so their total parameters equal
# heads * (kv_lora * qk_nope + v_head * kv_lora).
GLM_LAYERS = 78
GLM_HIDDEN = 6144
GLM_HEADS = 64
GLM_Q_LORA = 2048
GLM_KV_LORA = 512
GLM_QK_NOPE = 192
GLM_QK_ROPE = 64
GLM_V_HEAD = 256
GLM_INDEX_HEADS = 32
GLM_INDEX_DIM = 128
GLM_INDEX_TOPK = 2048
GLM_VOCAB = 154880
GLM_ROUTED_EXPERTS = 256
GLM_TOPK = 8
GLM_MOE_INTERMEDIATE = 2048
GLM_DENSE_INTERMEDIATE = 12288
GLM_BF16_BYTES = 2

GLM_ATTN_PARAMS = (
    (GLM_Q_LORA + GLM_KV_LORA + GLM_QK_ROPE) * GLM_HIDDEN
    + GLM_HEADS * (GLM_QK_NOPE + GLM_QK_ROPE) * GLM_Q_LORA
    + GLM_HEADS * GLM_KV_LORA * GLM_QK_NOPE
    + GLM_HEADS * GLM_V_HEAD * GLM_KV_LORA
    + GLM_HIDDEN * GLM_HEADS * GLM_V_HEAD
)
GLM_INDEX_PARAMS = (
    GLM_INDEX_HEADS * GLM_INDEX_DIM * GLM_Q_LORA
    + (GLM_INDEX_DIM + GLM_INDEX_HEADS) * GLM_HIDDEN
)
GLM_DENSE_FFN_PARAMS = 3 * GLM_DENSE_INTERMEDIATE * GLM_HIDDEN
GLM_ROUTER_PARAMS = GLM_ROUTED_EXPERTS * GLM_HIDDEN
GLM_EXPERT_PARAMS = GLM_ROUTED_EXPERTS * (
    2 * GLM_MOE_INTERMEDIATE * GLM_HIDDEN + GLM_HIDDEN * GLM_MOE_INTERMEDIATE
)
GLM_SHARED_PARAMS = 3 * GLM_MOE_INTERMEDIATE * GLM_HIDDEN
GLM_NORM_PARAMS = (
    (GLM_HIDDEN + GLM_Q_LORA + GLM_KV_LORA + GLM_HIDDEN) * GLM_LAYERS
    + GLM_INDEX_DIM * 21
    + GLM_HIDDEN
)


def test_glm52_schedule_and_attention_holdouts():
    model = load_model(GLM52)
    assert model.num_layers == 78
    stacks = {stack.tag: stack for stack in model.layers}
    assert [(stack.tag, stack.count) for stack in model.layers] == [
        ("dense_full_index", 3),
        ("sparse_initial_index_share", 3),
        ("sparse_cycle_full_index", 18),
        ("sparse_cycle_index_share", 54),
    ]
    assert stacks["dense_full_index"].ffn.__class__.__name__ == "DenseSwiGLU"
    assert stacks["sparse_initial_index_share"].ffn.__class__.__name__ == "MoE"
    assert stacks["dense_full_index"].attn.full_index is True
    assert stacks["sparse_initial_index_share"].attn.full_index is False
    assert stacks["sparse_cycle_full_index"].attn.full_index is True
    assert stacks["sparse_cycle_index_share"].attn.full_index is False
    assert stacks["dense_full_index"].attn.mla_cache_dtype_bytes == GLM_BF16_BYTES
    assert stacks["dense_full_index"].attn.index_cache_dtype_bytes == 1

    full_names = {group.name for group in stacks["dense_full_index"].attn.matmul_groups()}
    shared_names = {
        group.name for group in stacks["sparse_initial_index_share"].attn.matmul_groups()
    }
    assert {"indexer.q_proj", "indexer.wk", "indexer.weights_proj"} <= full_names
    assert not any(name.startswith("indexer.") for name in shared_names)
    assert {"q_absorb", "v_up", "o_proj"} <= shared_names


def test_glm52_hand_derived_params_and_decode_goldens():
    label = load_model(GLM52).label(Workload.causal_lm(decode=[4096], sampled=1))
    dense_full = GLM_ATTN_PARAMS + GLM_INDEX_PARAMS + GLM_DENSE_FFN_PARAMS
    sparse_share = GLM_ATTN_PARAMS + GLM_ROUTER_PARAMS + GLM_EXPERT_PARAMS + GLM_SHARED_PARAMS
    sparse_full = sparse_share + GLM_INDEX_PARAMS
    expected_total = (
        3 * dense_full
        + 3 * sparse_share
        + 18 * sparse_full
        + 54 * sparse_share
        + GLM_NORM_PARAMS
        + 2 * GLM_VOCAB * GLM_HIDDEN
    )
    expected_active_layers = (
        3 * (GLM_ATTN_PARAMS + GLM_INDEX_PARAMS + GLM_DENSE_FFN_PARAMS)
        + 3 * GLM_ATTN_PARAMS
        + 18 * (GLM_ATTN_PARAMS + GLM_INDEX_PARAMS)
        + 54 * GLM_ATTN_PARAMS
        + 75
        * (
            GLM_ROUTER_PARAMS
            + GLM_TOPK
            * (2 * GLM_MOE_INTERMEDIATE * GLM_HIDDEN + GLM_HIDDEN * GLM_MOE_INTERMEDIATE)
            + GLM_SHARED_PARAMS
        )
        + GLM_NORM_PARAMS
    )
    assert label.params["total"] == expected_total == 743_376_998_016
    assert label.params["activated"]["layers"] == expected_active_layers == 39_347_342_976
    assert label.params["activated"]["with_embed_head"] == 41_250_508_416
    assert label.params["breakdown"] == {
        "embedding": GLM_VOCAB * GLM_HIDDEN,
        "norm": GLM_NORM_PARAMS,
        "attn": GLM_ATTN_PARAMS * 78 + GLM_INDEX_PARAMS * 21,
        "ffn": GLM_DENSE_FFN_PARAMS * 3,
        "experts": GLM_EXPERT_PARAMS * 75,
        "shared": GLM_SHARED_PARAMS * 75,
        "router": GLM_ROUTER_PARAMS * 75,
        "lm_head": GLM_VOCAB * GLM_HIDDEN,
    }

    # Decode: 4096 causal pairs, 2048 selected keys, and 4095 cached keys.
    expected_proj = 2 * GLM_ATTN_PARAMS * 78 + 2 * GLM_INDEX_PARAMS * 21
    expected_internal = (
        2 * GLM_HEADS * (GLM_KV_LORA + GLM_QK_ROPE + GLM_KV_LORA) * GLM_INDEX_TOPK * 78
        + 2 * GLM_INDEX_HEADS * GLM_INDEX_DIM * 4096 * 21
    )
    expected_ffn = 2 * (
        GLM_DENSE_FFN_PARAMS * 3
        + GLM_TOPK
        * (2 * GLM_MOE_INTERMEDIATE * GLM_HIDDEN + GLM_HIDDEN * GLM_MOE_INTERMEDIATE)
        * 75
        + GLM_SHARED_PARAMS * 75
    )
    expected_router = 2 * GLM_ROUTER_PARAMS * 75
    assert label.flops["attn_proj"] == expected_proj == 26_136_674_304
    assert label.flops["attn_internal"] == expected_internal == 22_951_231_488
    assert label.flops["ffn"] == expected_ffn == 52_319_748_096
    assert label.flops["router"] == expected_router == 235_929_600
    assert label.flops["lm_head"] == 2 * GLM_HIDDEN * GLM_VOCAB
    assert label.bytes["kv"] == 195_469_056


def test_glm52_segments_holdouts_and_unified_map_are_complete():
    model = load_model(GLM52)
    label = model.label(
        Workload.causal_lm(prefill=[(8, 0)], decode=[4096], sampled=2)
    )
    names = {segment.name for segment in label.segments}
    assert {
        "dense_full_index.indexer.prefill",
        "dense_full_index.indexer.decode",
        "dense_full_index.attn.prefill",
        "dense_full_index.attn.decode",
        "dense_full_index.index_cache_append",
        "dense_full_index.mla_cache_append",
    } <= names
    assert not any(
        "indexer." in segment.name for segment in label.segments if "index_share" in segment.name
    )
    assert not any(segment.name.endswith(".kv_cache_append") for segment in label.segments)
    assert sum(segment.flops_total for segment in label.segments) == pytest.approx(
        label.flops_total
    )
    assert sum(segment.bytes_total for segment in label.segments) == pytest.approx(
        label.bytes_total
    )

    prefill = model.label(Workload.causal_lm(prefill=[(8, 0)], sampled=1))
    # The empty-prefix causal triangle is 8*9/2 = 36 pairs. The selected-key
    # count is the same short triangle because index_topk is 2048.
    assert prefill.flops["attn_internal"] == (
        2 * GLM_INDEX_HEADS * GLM_INDEX_DIM * 36 * 21
        + 2 * GLM_HEADS * (GLM_KV_LORA + GLM_QK_ROPE + GLM_KV_LORA) * 36 * 78
    ) == 397_246_464
    assert prefill.bytes["kv"] == (
        36 * (GLM_KV_LORA + GLM_QK_ROPE) * GLM_BF16_BYTES * 78
        + 8 * (GLM_KV_LORA + GLM_QK_ROPE) * GLM_BF16_BYTES * 78
        + 8 * (GLM_INDEX_DIM + 4) * 21
    ) == 3_975_840

    map_path = (
        Path(__file__).resolve().parents[1]
        / "model"
        / "work"
        / "location_maps"
        / "glm52_dsa_moe_unified.json"
    )
    location_map = json.loads(map_path.read_text())
    mapped = [semantic for row in location_map["locations"] for semantic in row["semantics"]]
    assert location_map["schema_version"] == 1
    assert location_map["arch_types"] == ["glm52_dsa_moe"]
    assert len(location_map["locations"]) == len(
        {row["location"] for row in location_map["locations"]}
    ) == 126
    assert len(mapped) == len(set(mapped))
    assert set(mapped) == names
    assert all(
        ".dispatch." not in row["location"] and ".combine." not in row["location"]
        for row in location_map["locations"]
    )

# Real Qwen3.6-27B text_config: L=64, hidden=5120, intermediate=17408, vocab=248320,
# head_dim=256, num_qo=24, num_kv=4, attn_output_gate; GDN: v_heads=48, k_heads=16,
# d_k=d_v=128, conv_kernel=4. Hand-derived per-layer params:
#   full attn (gated GQA): qkv (24*256*2 + 2*4*256)=14336 *5120 + o 5120*(24*256)=6144
#                          = 73_400_320 + 31_457_280 = 104_857_600
#   linear GDN block: in_proj_qkv (2048*2+6144)=10240*5120 + z 6144*5120 + b/a 48*5120 (x2)
#                     + conv 10240*4 + out 5120*6144 = 115_875_840
#   dense MLP: 3*17408*5120 = 267_386_880 ; embed = lm_head = 248320*5120 = 1_271_398_400
Q36_FULL_ATTN = 104_857_600
Q36_LINEAR_ATTN = 115_875_840
Q36_MLP = 267_386_880
VOCAB_Q36 = 248320


def test_qwen3_6_hybrid_stack():
    model = load_model(QWEN36)
    assert model.num_layers == 64
    by_tag = {stack.tag: stack for stack in model.layers}
    assert by_tag["linear"].count == 48
    assert type(by_tag["linear"].attn).__name__ == "GatedDeltaNet"
    assert by_tag["full"].count == 16
    assert by_tag["full"].attn.output_gate is True  # Qwen3.6 gated full attention


def test_qwen3_6_params_exact():
    label = load_model(QWEN36).label(Workload.causal_lm(decode=[4096], sampled=1))
    assert label.params["total"] == 26_895_319_040  # "27B"
    # dense model: every layer param is activated, so activated == total minus embed/head.
    assert label.params["activated"]["layers"] == 24_352_522_240
    assert label.params["activated"]["with_embed_head"] == label.params["total"]
    breakdown = label.params["breakdown"]
    assert breakdown["attn"] == 48 * Q36_LINEAR_ATTN + 16 * Q36_FULL_ATTN  # 7_239_761_920
    assert breakdown["ffn"] == 64 * Q36_MLP  # 17_112_760_320
    assert breakdown["experts"] == 0 and breakdown["router"] == 0  # dense, not MoE
    assert breakdown["embedding"] == breakdown["lm_head"] == VOCAB_Q36 * 5120


def test_qwen3_6_matches_27b_card():
    label = load_model(QWEN36).label(Workload.causal_lm(decode=[1], sampled=1))
    assert label.params["total"] == pytest.approx(27e9, rel=0.01)


def test_qwen3_6_segments_roll_up():
    # the heterogeneous fold must still reconcile: Σ segment work == aggregate buckets.
    label = load_model(QWEN36).label(Workload.causal_lm(decode=[4096] * 32, sampled=32))
    assert sum(s.flops_total for s in label.segments) == pytest.approx(label.flops_total)
    assert sum(s.bytes_total for s in label.segments) == pytest.approx(label.bytes_total)


def test_collapsed_workload_preserves_linear_attention_state_transactions():
    """Geometry compression must retain recurrent-state read/write count."""
    model = load_model(QWEN36)
    reference = Workload.causal_lm(prefill=[(8, 0), (4, 8)], decode=[32, 64, 128])
    aggregate = Workload(
        matmul_tokens=reference.matmul_tokens,
        head_positions=reference.head_positions,
        attn=[AttnInteraction(1, 1, 0, "full")],
        attention_step_count=len(reference.attn),
    )

    reference_label = model.label(reference)
    aggregate_label = model.label(aggregate)
    reference_linear = next(
        segment for segment in reference_label.segments if segment.name == "linear.attn"
    )
    aggregate_linear = next(
        segment for segment in aggregate_label.segments if segment.name == "linear.attn"
    )
    assert aggregate_linear.bytes == reference_linear.bytes


def test_gdn_state_is_context_independent():
    model = load_model(QWEN36)
    short = {s.name: s for s in model.label(Workload.causal_lm(decode=[4096], sampled=1)).segments}
    long = {s.name: s for s in model.label(Workload.causal_lm(decode=[131072], sampled=1)).segments}
    # linear-attention recurrent state does NOT grow with context (its whole point)...
    assert short["linear.attn"].bytes == long["linear.attn"].bytes
    # ...while the full-attention KV cache does.
    assert long["full.attn.decode"].bytes > 10 * short["full.attn.decode"].bytes


def test_gdn_internal_flops_linear_and_state_scaling():
    from model.work.attention.linear import GatedDeltaNet

    gdn = GatedDeltaNet(
        hidden=5120,
        num_v_heads=48,
        num_k_heads=16,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel=4,
        state_dtype_bytes=4,
    )
    # O(T): 3 d_k·d_v products per (token, v-head) -> 6·num_v·d_k·d_v, linear in tokens.
    ten = Workload.causal_lm(decode=[4096] * 10, sampled=10)
    assert gdn.internal_flops(ten) == 6 * 48 * 128 * 128 * 10
    # fixed state per sequence (read+write), context-independent, scales with #sequences.
    one_short = Workload.causal_lm(decode=[128], sampled=1)
    one_long = Workload.causal_lm(decode=[1 << 17], sampled=1)
    assert gdn.kv_bytes(one_short) == gdn.kv_bytes(one_long) == 2 * (48 * 128 * 128 * 4)
    five = Workload.causal_lm(decode=[128] * 5, sampled=5)
    assert gdn.kv_bytes(five) == 5 * gdn.kv_bytes(one_short)


def test_gated_attention_doubles_q_projection():
    from model.work.attention.gqa import GQA

    plain = GQA(hidden=5120, num_qo_heads=24, num_kv_heads=4, head_dim=256, kv_dtype_bytes=2.0)
    gated = GQA(
        hidden=5120,
        num_qo_heads=24,
        num_kv_heads=4,
        head_dim=256,
        kv_dtype_bytes=2.0,
        output_gate=True,
    )
    qkv_plain = next(g for g in plain.matmul_groups() if g.name == "qkv")
    qkv_gated = next(g for g in gated.matmul_groups() if g.name == "qkv")
    assert qkv_plain.n == (24 + 2 * 4) * 256  # 8192
    assert qkv_gated.n == qkv_plain.n + 24 * 256  # + output gate on the q side = 14336
