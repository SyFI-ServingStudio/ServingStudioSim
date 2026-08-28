"""Golden tests for the optimal necessary-work labeler (model/work/).

Every expected value below is HAND-DERIVED from the Llama3-8B config and asserted
against the labeler — the point is to catch a formula drift, so the expectations are
written out independently rather than recomputed through the code under test.

Llama3-8B: L=32, hidden=4096, num_qo=32, num_kv=8, head_dim=128, intermediate=14336,
vocab=128256, bf16 (2 B), lm_head untied.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from model.work import Workload, load_model
from model.work import floors as work_floors
from model.work.attention.linear import GatedDeltaNet
from model.work.core import AttnInteraction, LayerStack, Model, gpu_peak_tflops
from model.work.ffn.moe import MoE
from model.work.parameter_counts import compute_parameter_counts
from model.work.quantization import parse_quantization_config
from model.work.registry import REGISTRY, UnknownArchitecture, build_model

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
        "main": {"config": str(CONFIG), "gpu": "H200", "dtype": "bf16", "arch_fp8": False},
        "other": {
            "config": "different-model.json",
            "gpu": "H200",
            "dtype": "bf16",
            "arch_fp8": False,
        },
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
    spec = {"config": str(CONFIG), "gpu": "H200", "dtype": "bf16", "arch_fp8": False}
    monkeypatch.setattr(work_floors, "_pool_specs", lambda _log_dir: {"main": spec})
    monkeypatch.setattr(work_floors, "_model", lambda _config_path: model)
    # Let group size out of the decision so this case is purely about the two
    # reduction paths; `_MIN_BASIS_GROUP` has its own test below.
    monkeypatch.setattr(work_floors, "_MIN_BASIS_GROUP", 1)
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
    # Two bases, not one: the pure-decode shapes and the prefill shape sit on
    # opposite sides of the mode-presence cliff, which the basis key pins.
    assert result["composition"] == {
        "unique_shapes": 3,
        "iterations": 9,
        "affine_bases": 0 if force_direct_fallback else 2,
        "direct_fallback_bases": 2 if force_direct_fallback else 0,
    }


def test_small_group_skips_the_basis_it_cannot_pay_for(monkeypatch, model):
    """A group below `_MIN_BASIS_GROUP` takes the direct path, same numbers.

    Fitting a basis costs ~22 `model.label` calls to then evaluate the group as
    array math, so on a handful of shapes it is strictly more work than labeling
    each one — and routed models make near-singleton prefill groups routine.
    """
    spec = {"config": str(CONFIG), "gpu": "H200", "dtype": "bf16", "arch_fp8": False}
    monkeypatch.setattr(work_floors, "_pool_specs", lambda _log_dir: {"main": spec})
    monkeypatch.setattr(work_floors, "_model", lambda _config_path: model)
    shapes = [
        {
            "occurrences": 1,
            "totals": {
                "matmul_tokens": 1,
                "prefill_tokens": 0,
                "decode_passes": 1,
                "prefill_pairs": 0,
                "prefill_cached": 0,
                "decode_kv": kv,
                "prefill_requests": 0,
            },
        }
        for kv in (4096, 8192, 16384)
    ]

    monkeypatch.setattr(work_floors, "_MIN_BASIS_GROUP", 24)
    small = work_floors.compute_locked_compositions(Path("unused"), {"main/0": shapes})["main/0"]
    monkeypatch.setattr(work_floors, "_MIN_BASIS_GROUP", 1)
    affine = work_floors.compute_locked_compositions(Path("unused"), {"main/0": shapes})["main/0"]

    assert small["composition"]["affine_bases"] == 0
    assert small["composition"]["direct_fallback_bases"] == 1
    assert affine["composition"]["affine_bases"] == 1
    assert small["necessary"] == pytest.approx(affine["necessary"])
    assert small["segmented"] == pytest.approx(affine["segmented"])


def test_qwen3_235b_fp8_prefills_precision_is_fp8():
    """The FP8 recipe serves prefill (FA3) with FP8 Q/KV, so its necessary math
    must be labeled at the FP8 tensor-core peak; decode QK stays BF16. A bf16
    pin here re-inflates R6 above R5 and hides every quant/norm excess."""
    from pathlib import Path

    qwen = load_model(str(Path("model/config/qwen3_235b_thinking_2507_fp8.json")))
    label = qwen.label(Workload.causal_lm(prefill=[(16384, 0)], sampled=1))
    dtype = {s.name: s.compute_dtype for s in label.segments}
    assert dtype["attn.prefill"] == "fp8"
    assert dtype["attn.decode"] != "fp8"  # decode QK in the FP8 recipe is BF16


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
QWEN36_MOE = Path(__file__).resolve().parents[1] / "model" / "config" / "qwen3_6_35b_a3b_fp8.json"
QWEN36_LOCAL_MAP = (
    Path(__file__).resolve().parents[1]
    / "model"
    / "work"
    / "location_maps"
    / "qwen36_local_unified.json"
)
QWEN36_LOCAL_LOCATIONS = [
    "unified.embedding",
    "unified.gdn.input_add_rms_norm",
    "unified.gdn.qkvz.input_quant",
    "unified.gdn.qkvz.gemm",
    "unified.gdn.ba",
    "unified.gdn.split_b",
    "unified.gdn.split_a",
    "unified.gdn.core_output_zero",
    "unified.gdn.state_gather",
    "unified.gdn.state_zero",
    "unified.gdn.prefill.causal_conv",
    "unified.gdn.prefill.post_conv",
    "unified.gdn.prefill.cumsum",
    "unified.gdn.prefill.kkt",
    "unified.gdn.prefill.solve",
    "unified.gdn.prefill.recompute_w_u",
    "unified.gdn.prefill.state_update",
    "unified.gdn.prefill.output",
    "unified.gdn.state_scatter",
    "unified.gdn.decode.causal_conv",
    "unified.gdn.decode.recurrent",
    "unified.gdn.core_output_copy",
    "unified.gdn.gated_norm",
    "unified.gdn.out_proj.input_quant",
    "unified.gdn.out_proj.gemm",
    "unified.gdn.post_attention_add_rms_norm",
    "unified.router.router.gemm",
    "unified.router.topk",
    "unified.router.align",
    "unified.routed_expert.gate_up.input_quant",
    "unified.routed_expert.gate_up.gemm",
    "unified.routed_expert.activation",
    "unified.routed_expert.down.input_quant",
    "unified.routed_expert.down.gemm",
    "unified.shared_expert.gate_up.input_quant",
    "unified.shared_expert.gate_up.gemm",
    "unified.shared_expert.silu_and_mul",
    "unified.shared_expert.down.input_quant",
    "unified.shared_expert.down.gemm",
    "unified.shared_expert.shared_gate",
    "unified.shared_expert.apply_shared_gate",
    "unified.finalize.finalize",
    "unified.finalize.shared_routed_add",
    "unified.gated_gqa.input_add_rms_norm",
    "unified.gated_gqa.qkv_gate.input_quant",
    "unified.gated_gqa.qkv_gate.gemm",
    "unified.gated_gqa.q_norm",
    "unified.gated_gqa.k_norm",
    "unified.gated_gqa.partial_rope",
    "unified.gated_gqa.attention.kv_cache_append",
    "unified.gated_gqa.attention.prefill",
    "unified.gated_gqa.attention.decode",
    "unified.gated_gqa.output_gate",
    "unified.gated_gqa.out_proj.input_quant",
    "unified.gated_gqa.out_proj.gemm",
    "unified.gated_gqa.post_attention_add_rms_norm",
    "unified.head.final_add_rms_norm",
    "unified.head.lm_head",
]

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
    GLM_INDEX_HEADS * GLM_INDEX_DIM * GLM_Q_LORA + (GLM_INDEX_DIM + GLM_INDEX_HEADS) * GLM_HIDDEN
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
            + GLM_TOPK * (2 * GLM_MOE_INTERMEDIATE * GLM_HIDDEN + GLM_HIDDEN * GLM_MOE_INTERMEDIATE)
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
    label = model.label(Workload.causal_lm(prefill=[(8, 0)], decode=[4096], sampled=2))
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
    assert (
        prefill.flops["attn_internal"]
        == (
            2 * GLM_INDEX_HEADS * GLM_INDEX_DIM * 36 * 21
            + 2 * GLM_HEADS * (GLM_KV_LORA + GLM_QK_ROPE + GLM_KV_LORA) * 36 * 78
        )
        == 397_246_464
    )
    assert (
        prefill.bytes["kv"]
        == (
            36 * (GLM_KV_LORA + GLM_QK_ROPE) * GLM_BF16_BYTES * 78
            + 8 * (GLM_KV_LORA + GLM_QK_ROPE) * GLM_BF16_BYTES * 78
            + 8 * (GLM_INDEX_DIM + 4) * 21
        )
        == 3_975_840
    )

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
    assert (
        len(location_map["locations"])
        == len({row["location"] for row in location_map["locations"]})
        == 126
    )
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
# Packed projections plus the attention-owned A-log and dt-bias. The gated
# RMSNorm scale is accounted independently in the norm breakdown.
Q36_LINEAR_ATTN = 115_875_840 + 48 + 48
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
    assert label.params["total"] == 26_895_329_792  # "27B"
    # dense model: every layer param is activated, so activated == total minus embed/head.
    assert label.params["activated"]["layers"] == 24_352_532_992
    assert label.params["activated"]["with_embed_head"] == label.params["total"]
    breakdown = label.params["breakdown"]
    assert breakdown["attn"] == 48 * Q36_LINEAR_ATTN + 16 * Q36_FULL_ATTN  # 7_239_766_528
    assert breakdown["norm"] == 48 * 128
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
        attn=[
            AttnInteraction(1, 1, 0, "full", phase="prefill"),
            AttnInteraction(1, 1, 0, "full", phase="decode"),
        ],
        attention_step_count=len(reference.attn),
        attention_step_count_by_phase={"prefill": 2, "decode": 3},
        attention_tokens_by_phase={"prefill": 12, "decode": 3},
        prefill_stateful_requests=1,
    )

    reference_label = model.label(reference)
    aggregate_label = model.label(aggregate)
    state_names = {
        "linear.recurrent_state.prefill_read",
        "linear.recurrent_state.prefill_write",
        "linear.recurrent_state.decode_read",
        "linear.recurrent_state.decode_write",
        "linear.conv_state.prefill_read",
        "linear.conv_state.prefill_write",
        "linear.conv_state.decode_read",
        "linear.conv_state.decode_write",
    }
    assert sum(s.bytes_total for s in aggregate_label.segments if s.name in state_names) == sum(
        s.bytes_total for s in reference_label.segments if s.name in state_names
    )


def test_gdn_state_is_context_independent():
    model = load_model(QWEN36)
    short = {s.name: s for s in model.label(Workload.causal_lm(decode=[4096], sampled=1)).segments}
    long = {s.name: s for s in model.label(Workload.causal_lm(decode=[131072], sampled=1)).segments}
    # linear-attention recurrent state does NOT grow with context (its whole point)...
    state_suffixes = ("state.decode_read", "state.decode_write")
    assert sum(v.bytes for k, v in short.items() if k.endswith(state_suffixes)) == sum(
        v.bytes for k, v in long.items() if k.endswith(state_suffixes)
    )
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
    state_and_conv = 48 * 128 * 128 * 4 + (2 * 16 * 128 + 48 * 128) * 4 * 2
    assert gdn.kv_bytes(one_short) == gdn.kv_bytes(one_long) == 2 * state_and_conv
    five = Workload.causal_lm(decode=[128] * 5, sampled=5)
    assert gdn.kv_bytes(five) == 5 * gdn.kv_bytes(one_short)


def _qwen_gdn() -> GatedDeltaNet:
    return GatedDeltaNet(
        hidden=2048,
        num_v_heads=32,
        num_k_heads=16,
        head_k_dim=128,
        head_v_dim=128,
        conv_kernel=4,
        state_dtype_bytes=4,
        activation_dtype_bytes=2,
    )


def test_workload_tracks_stateful_prefill_requests_and_legacy_default():
    fresh = Workload.causal_lm(prefill=[(128, 0)])
    stateful = Workload.causal_lm(prefill=[(64, 64)])
    decode = Workload.causal_lm(decode=[32, 64])
    mixed = Workload.causal_lm(prefill=[(3, 0), (4, 8)], decode=[32])
    assert fresh.prefill_stateful_requests == 0
    assert stateful.prefill_stateful_requests == 1
    assert decode.prefill_stateful_requests == 0
    assert mixed.prefill_stateful_requests == 1
    assert dict(mixed.attention_phases())["prefill"].prefill_stateful_requests == 1
    assert Workload(0, 0).prefill_stateful_requests == 0
    with pytest.raises(ValueError, match="non-negative integer"):
        Workload(0, 0, prefill_stateful_requests=-1)


def test_floors_stateful_prefill_axis_and_legacy_payload_default():
    totals = {
        "matmul_tokens": 65,
        "prefill_tokens": 64,
        "decode_passes": 1,
        "prefill_pairs": 64 * 129 - 64 * 63 // 2,
        "prefill_cached": 64,
        "decode_kv": 32,
        "prefill_requests": 1,
        "prefill_stateful_requests": 1,
    }
    assert work_floors._aggregate_workload(totals).prefill_stateful_requests == 1
    assert (
        work_floors._aggregate_workload(
            {k: v for k, v in totals.items() if k != "prefill_stateful_requests"}
        ).prefill_stateful_requests
        == 0
    )
    with pytest.raises(ValueError, match="cannot exceed"):
        work_floors._aggregate_workload({**totals, "prefill_stateful_requests": 2})
    with pytest.raises(ValueError, match="exact integer"):
        work_floors._aggregate_workload({**totals, "prefill_stateful_requests": 0.5})


def test_qwen_gdn_packed_groups_learned_weights_and_state_formula():
    gdn = _qwen_gdn()
    groups = {group.name: group for group in gdn.matmul_groups()}
    assert (groups["qkvz"].n, groups["qkvz"].k, groups["qkvz"].module) == (
        12288,
        2048,
        "linear_attn.in_proj_qkvz",
    )
    assert (groups["ba"].n, groups["ba"].k, groups["ba"].module) == (
        64,
        2048,
        "linear_attn.in_proj_ba",
    )
    assert (groups["conv1d"].n, groups["conv1d"].k, groups["conv1d"].module) == (
        8192,
        4,
        "linear_attn.conv1d",
    )
    assert (groups["out_proj"].n, groups["out_proj"].k) == (2048, 4096)
    assert [(w.name, w.elements, w.breakdown) for w in gdn.learned_weight_groups()] == [
        ("a_log", 32, "attn"),
        ("dt_bias", 32, "attn"),
        ("gated_norm", 128, "norm"),
    ]
    assert gdn.recurrent_state_bytes == 2_097_152
    assert gdn.convolution_state_bytes == 65_536
    unit = 2_097_152 + 65_536
    assert gdn.kv_bytes(Workload.causal_lm(prefill=[(3, 0)])) == unit
    assert gdn.kv_bytes(Workload.causal_lm(prefill=[(3, 8)])) == 2 * unit
    assert gdn.kv_bytes(Workload.causal_lm(decode=[32])) == 2 * unit
    assert gdn.kv_bytes(Workload.causal_lm(prefill=[(3, 0), (4, 8)], decode=[32, 64])) == 7 * unit
    assert gdn.internal_flops(Workload.causal_lm(decode=[32] * 5)) == 6 * 32 * 128 * 128 * 5


def test_gdn_weight_only_groups_add_params_and_bytes_but_no_flops():
    class EmptyFfn:
        def matmul_groups(self):
            return []

    gdn = _qwen_gdn()
    model = Model(
        name="test",
        hidden=2048,
        vocab=1,
        weight_dtype_bytes=2,
        tie_word_embeddings=True,
        layers=[LayerStack(gdn, EmptyFfn(), 30, "gdn")],
    )
    label = model.label(Workload.causal_lm(decode=[32], sampled=0))
    learned = [
        s for s in label.segments if s.name in {"gdn.a_log", "gdn.dt_bias", "gdn.gated_norm"}
    ]
    assert sum(s.bytes_total for s in learned) == 30 * 2 * (32 + 32 + 128)
    assert all(s.flops_total == 0 for s in learned)
    matrix_params = sum(g.total_params for g in gdn.matmul_groups())
    old_attn_classification = 30 * (matrix_params + 32 + 32 + 128)
    assert label.params["breakdown"]["attn"] == old_attn_classification - 3_840
    assert label.params["breakdown"]["norm"] == 3_840
    assert label.params["activated"]["layers"] == 30 * (matrix_params + 32 + 32 + 128)


def test_gdn_packed_quant_scale_is_counted_once():
    class EmptyFfn:
        def matmul_groups(self):
            return []

    raw = json.loads(
        (Path(__file__).resolve().parents[1] / "model/config/qwen3_6_35b_a3b_fp8.json").read_text()
    )
    model = Model(
        name="test",
        hidden=2048,
        vocab=1,
        weight_dtype_bytes=2,
        tie_word_embeddings=True,
        layers=[LayerStack(_qwen_gdn(), EmptyFfn(), 1, "gdn")],
        quant=parse_quantization_config(raw),
        master_dtype="bfloat16",
    )
    label = model.label(Workload.causal_lm(decode=[32], sampled=0))
    qkvz = next(segment for segment in label.segments if segment.name == "gdn.qkvz")
    expected_bytes = 12_288 * 2_048 + 4 * (12_288 // 128) * (2_048 // 128)
    assert qkvz.bytes_total == expected_bytes
    assert qkvz.compute_dtype == "fp8"
    assert not any(segment.name in {"gdn.qkv", "gdn.z"} for segment in label.segments)


# Qwen3.6-35B-A3B independent checkpoint arithmetic. Converted matrices carry
# one FP32 scale per 128x128 block; checkpoint-excluded matrices remain BF16.
def _qwen36_fp8_matrix_bytes(n: int, k: int) -> int:
    return n * k + 4 * math.ceil(n / 128) * math.ceil(k / 128)


def _qwen36_bf16_matrix_bytes(n: int, k: int) -> int:
    return 2 * n * k


def _qwen36_expected_loaded_experts(tokens: int) -> float:
    return 256 * (1 - (255 / 256) ** (8 * tokens))


def _qwen36_expected_weight_bytes(tokens: int) -> float:
    # G/F include their mechanism-specific learned vectors and external norms.
    gdn = 33_898_880
    gated_gqa = 27_278_848
    fixed_moe = 4_199_168
    routed_expert = 3_146_496
    return (
        30 * gdn
        + 10 * gated_gqa
        + 40 * (fixed_moe + _qwen36_expected_loaded_experts(tokens) * routed_expert)
        + 4096 * min(tokens, 248_320)
        + 2 * 2048 * 248_320
        + 4096
    )


@pytest.fixture(scope="module")
def qwen36_moe():
    return load_model(QWEN36_MOE)


def test_qwen36_moe_registry_geometry_schedule_and_quantization(qwen36_moe):
    from model.work.models import qwen3_6, qwen3_6_moe

    assert REGISTRY["Qwen3_5MoeForConditionalGeneration"] is qwen3_6_moe.build
    assert REGISTRY["Qwen3_5ForConditionalGeneration"] is qwen3_6.build
    assert qwen36_moe.name == "Qwen3_5MoeForConditionalGeneration"
    assert qwen36_moe.hidden == 2048
    assert qwen36_moe.vocab == 248_320
    assert qwen36_moe.tie_word_embeddings is False
    assert qwen36_moe.master_dtype == "bf16"
    assert qwen36_moe.quant.compute_dtype == "fp8"
    assert qwen36_moe.quant.block_shape == (128, 128)
    assert [(stack.tag, stack.count) for stack in qwen36_moe.layers] == [
        ("gdn", 30),
        ("gated_gqa", 10),
    ]
    gdn, gated_gqa = (stack.attn for stack in qwen36_moe.layers)
    assert (
        gdn.num_k_heads,
        gdn.num_v_heads,
        gdn.head_k_dim,
        gdn.head_v_dim,
        gdn.conv_kernel,
        gdn.state_dtype_bytes,
    ) == (16, 32, 128, 128, 4, 4)
    assert (
        gated_gqa.num_qo_heads,
        gated_gqa.num_kv_heads,
        gated_gqa.head_dim,
        gated_gqa.output_gate,
        gated_gqa.kv_dtype_bytes,
    ) == (16, 2, 256, True, 2)
    for stack in qwen36_moe.layers:
        assert stack.ffn.moe_intermediate == 512
        assert stack.ffn.num_experts == 256
        assert stack.ffn.top_k == 8
        assert stack.ffn.shared_intermediate == 512
        assert stack.ffn.shared_module == "shared_expert"
        assert stack.ffn.shared_gate is True
    # The nested checkpoint also contains vision and MTP metadata, but this
    # text-only accountant has exactly the two 40-layer text stacks above.
    assert qwen36_moe.num_layers == 40


def test_qwen36_moe_schedule_is_explicit_and_position_checked():
    raw = json.loads(QWEN36_MOE.read_text())
    missing = json.loads(json.dumps(raw))
    missing["text_config"].pop("layer_types")
    with pytest.raises(ValueError, match="explicit 40-entry"):
        build_model(missing)

    short = json.loads(json.dumps(raw))
    short["text_config"]["layer_types"].pop()
    with pytest.raises(ValueError, match="explicit 40-entry"):
        build_model(short)

    misplaced = json.loads(json.dumps(raw))
    misplaced["text_config"]["layer_types"][2:4] = ["full_attention", "linear_attention"]
    with pytest.raises(ValueError, match="three linear then one full"):
        build_model(misplaced)

    wrong_count = json.loads(json.dumps(raw))
    wrong_count["text_config"]["num_hidden_layers"] = 39
    with pytest.raises(ValueError, match="num_hidden_layers=40"):
        build_model(wrong_count)


def test_qwen36_moe_exact_parameter_inventory_and_subprocess_contract(qwen36_moe):
    label = qwen36_moe.label(Workload.causal_lm(decode=[4096], sampled=1))
    # Literal checkpoint inventory: I/O tables + attention + norm vectors +
    # all 256 routed experts + shared path + router, over 30/10 layers.
    embedding = lm_head = 248_320 * 2048
    gdn_attn = 30 * (12_288 * 2048 + 64 * 2048 + 8192 * 4 + 2048 * 4096 + 32 + 32)
    gated_gqa_attn = 10 * (9216 * 2048 + 2048 * 4096)
    norms = 30 * 128 + 30 * 2 * 2048 + 10 * (2 * 2048 + 2 * 256) + 2048
    expert_matrix = 1024 * 2048 + 2048 * 512
    experts = 40 * 256 * expert_matrix
    shared = 40 * (expert_matrix + 2048)
    router = 40 * 256 * 2048
    expected_breakdown = {
        "embedding": embedding,
        "norm": norms,
        "attn": gdn_attn + gated_gqa_attn,
        "ffn": 0,
        "experts": experts,
        "shared": shared,
        "router": router,
        "lm_head": lm_head,
    }
    assert expected_breakdown == {
        "embedding": 508_559_360,
        "norm": 174_848,
        "attn": 1_284_179_840,
        "ffn": 0,
        "experts": 32_212_254_720,
        "shared": 125_911_040,
        "router": 20_971_520,
        "lm_head": 508_559_360,
    }
    assert label.params["breakdown"] == expected_breakdown
    assert sum(expected_breakdown.values()) == 34_660_610_688
    activated_layers = (
        expected_breakdown["attn"]
        + expected_breakdown["norm"]
        + 40 * 8 * expert_matrix
        + expected_breakdown["shared"]
        + expected_breakdown["router"]
    )
    assert activated_layers == 2_437_870_208
    assert label.params == {
        "total": 34_660_610_688,
        "activated": {
            "layers": 2_437_870_208,
            "with_embed_head": 3_454_988_928,
        },
        "breakdown": expected_breakdown,
    }
    assert compute_parameter_counts(QWEN36_MOE) == {
        "total": 34_660_610_688,
        "active": 3_454_988_928,
        "active_layers": 2_437_870_208,
        "active_definition": "with_embed_head",
    }


def test_qwen36_moe_prefill_and_decode_work_goldens(qwen36_moe):
    prefill = qwen36_moe.label(Workload.causal_lm(prefill=[(128, 0)], sampled=1))
    assert prefill.flops == {
        "attn_proj": 328_749_547_520,
        "attn_internal": 13_432_258_560,
        "ffn": 289_931_264_000,
        "router": 5_368_709_120,
        "lm_head": 1_017_118_720,
    }
    assert prefill.flops_total == 638_498_897_920
    assert prefill.bytes["kv"] == 64_880_640 + 2_621_440 == 67_502_080
    assert prefill.bytes["weights"] == pytest.approx(_qwen36_expected_weight_bytes(128), rel=1e-14)
    assert prefill.bytes["weights"] == pytest.approx(34_109_960_070.600513, rel=1e-14)

    decode = qwen36_moe.label(Workload.causal_lm(decode=[4096], sampled=1))
    assert decode.flops == {
        "attn_proj": 2_568_355_840,
        "attn_internal": 765_460_480,
        "ffn": 2_265_088_000,
        "router": 41_943_040,
        "lm_head": 1_017_118_720,
    }
    assert decode.flops_total == 6_657_966_080
    assert decode.bytes["kv"] == 129_761_280 + 83_886_080 == 213_647_360
    assert decode.bytes["weights"] == pytest.approx(_qwen36_expected_weight_bytes(1), rel=1e-14)
    assert decode.bytes["weights"] == pytest.approx(3_468_068_334.75965, rel=1e-14)


def test_qwen36_moe_weight_formulas_and_precision_exclusions(qwen36_moe):
    assert _qwen36_fp8_matrix_bytes(12_288, 2048) == 25_171_968
    assert _qwen36_bf16_matrix_bytes(64, 2048) == 262_144
    # Fixed per-layer byte formulas, independently expanded from checkpoint math.
    gdn = (
        _qwen36_fp8_matrix_bytes(12_288, 2048)
        + _qwen36_bf16_matrix_bytes(64, 2048)
        + _qwen36_bf16_matrix_bytes(8192, 4)
        + _qwen36_fp8_matrix_bytes(2048, 4096)
        + 2 * (32 + 32 + 128 + 2 * 2048)
    )
    gated_gqa = (
        _qwen36_fp8_matrix_bytes(9216, 2048)
        + _qwen36_fp8_matrix_bytes(2048, 4096)
        + 2 * (2 * 2048 + 2 * 256)
    )
    fixed_moe = (
        _qwen36_bf16_matrix_bytes(256, 2048)
        + _qwen36_fp8_matrix_bytes(1024, 2048)
        + _qwen36_fp8_matrix_bytes(2048, 512)
        + _qwen36_bf16_matrix_bytes(1, 2048)
    )
    routed_expert = _qwen36_fp8_matrix_bytes(1024, 2048) + _qwen36_fp8_matrix_bytes(2048, 512)
    assert (gdn, gated_gqa, fixed_moe, routed_expert) == (
        33_898_880,
        27_278_848,
        4_199_168,
        3_146_496,
    )

    by_tag = {stack.tag: stack for stack in qwen36_moe.layers}
    gdn_groups = {group.name: group for group in by_tag["gdn"].attn.matmul_groups()}
    gqa_groups = {group.name: group for group in by_tag["gated_gqa"].attn.matmul_groups()}
    moe_groups = {group.name: group for group in by_tag["gdn"].ffn.matmul_groups()}
    assert {
        name for name, group in gdn_groups.items() if qwen36_moe.quant.is_converted(group.module)
    } == {
        "qkvz",
        "out_proj",
    }
    assert {
        name
        for name, group in gdn_groups.items()
        if not qwen36_moe.quant.is_converted(group.module)
    } == {
        "ba",
        "conv1d",
    }
    assert all(qwen36_moe.quant.is_converted(group.module) for group in gqa_groups.values())
    assert {
        name
        for name, group in moe_groups.items()
        if not qwen36_moe.quant.is_converted(group.module)
    } == {
        "router",
        "shared_gate",
    }
    assert all(
        qwen36_moe.quant.is_converted(moe_groups[name].module)
        for name in ("expert_gate_up", "expert_down", "shared_gate_up", "shared_down")
    )


def test_qwen36_moe_semantic_names_roll_up_and_sampled_head(qwen36_moe):
    prefill = qwen36_moe.label(Workload.causal_lm(prefill=[(128, 0)], sampled=1))
    names = {segment.name for segment in prefill.segments}
    assert names == {
        "gdn.qkvz",
        "gdn.ba",
        "gdn.conv1d",
        "gdn.out_proj",
        "gdn.router",
        "gdn.expert_gate_up",
        "gdn.expert_down",
        "gdn.shared_gate_up",
        "gdn.shared_down",
        "gdn.shared_gate",
        "gdn.a_log",
        "gdn.dt_bias",
        "gdn.gated_norm",
        "gdn.attn.prefill",
        "gdn.attn.decode",
        "gdn.recurrent_state.prefill_read",
        "gdn.recurrent_state.prefill_write",
        "gdn.recurrent_state.decode_read",
        "gdn.recurrent_state.decode_write",
        "gdn.conv_state.prefill_read",
        "gdn.conv_state.prefill_write",
        "gdn.conv_state.decode_read",
        "gdn.conv_state.decode_write",
        "gated_gqa.qkv",
        "gated_gqa.o",
        "gated_gqa.router",
        "gated_gqa.expert_gate_up",
        "gated_gqa.expert_down",
        "gated_gqa.shared_gate_up",
        "gated_gqa.shared_down",
        "gated_gqa.shared_gate",
        "gated_gqa.attn.prefill",
        "gated_gqa.attn.decode",
        "gated_gqa.kv_cache_append",
        "gdn.input_norm",
        "gdn.post_norm",
        "gated_gqa.input_norm",
        "gated_gqa.q_norm",
        "gated_gqa.k_norm",
        "gated_gqa.post_norm",
        "final_norm",
        "embedding",
        "lm_head",
    }
    segment_dtypes = {segment.name: segment.compute_dtype for segment in prefill.segments}
    assert segment_dtypes["gdn.qkvz"] == "fp8"
    assert segment_dtypes["gdn.out_proj"] == "fp8"
    for name in (
        "gdn.ba",
        "gdn.conv1d",
        "gdn.router",
        "gdn.shared_gate",
        "gdn.input_norm",
        "gdn.gated_norm",
        "final_norm",
        "embedding",
        "lm_head",
    ):
        assert segment_dtypes[name] == "bf16"
    assert sum(segment.flops_total for segment in prefill.segments) == prefill.flops_total
    assert sum(segment.bytes_total for segment in prefill.segments) == pytest.approx(
        prefill.bytes_total
    )
    assert next(segment for segment in prefill.segments if segment.name == "lm_head").flops == (
        2 * 2048 * 248_320
    )
    decode = qwen36_moe.label(Workload.causal_lm(decode=[4096], sampled=1))
    assert next(segment for segment in decode.segments if segment.name == "lm_head").flops == (
        2 * 2048 * 248_320
    )
    norm = next(segment for segment in decode.segments if segment.name == "final_norm")
    assert norm.flops_total == 0
    assert norm.bytes_total == 4096


def test_qwen36_local_location_map_identity_order_and_locality():
    location_map = json.loads(QWEN36_LOCAL_MAP.read_text())
    locations = [row["location"] for row in location_map["locations"]]
    assert location_map["schema_version"] == 1
    assert location_map["mapping_id"] == "qwen36-local-unified-v2"
    assert location_map["arch_types"] == ["qwen36_local"]
    assert locations == QWEN36_LOCAL_LOCATIONS
    assert len(locations) == len(set(locations)) == 58
    assert not any(
        component in location
        for location in locations
        for component in (
            "dispatch",
            "combine",
            "network",
            "collective",
            "all_reduce",
            "all_to_all",
        )
    )


def test_qwen36_local_location_map_consumes_mixed_semantics_exactly_once(qwen36_moe):
    location_map = json.loads(QWEN36_LOCAL_MAP.read_text())
    mapped = [semantic for row in location_map["locations"] for semantic in row["semantics"]]
    mixed = qwen36_moe.label(
        Workload.causal_lm(
            prefill=[(64, 64), (64, 0)],
            decode=[32] * 8,
            sampled=10,
        )
    )
    segment_names = [segment.name for segment in mixed.segments]
    assert len(mapped) == len(set(mapped)) == 43
    assert set(mapped) == set(segment_names)
    assert len(segment_names) == len(set(segment_names)) == 43
    assert sum(segment.flops_total for segment in mixed.segments) == mixed.flops_total
    assert sum(segment.bytes_total for segment in mixed.segments) == pytest.approx(
        mixed.bytes_total
    )


def test_qwen36_local_location_map_shared_rows_and_scale_multiplicity(qwen36_moe):
    location_map = json.loads(QWEN36_LOCAL_MAP.read_text())
    semantics = {row["location"]: row["semantics"] for row in location_map["locations"]}
    assert semantics["unified.router.router.gemm"] == ["gdn.router", "gated_gqa.router"]
    assert semantics["unified.routed_expert.gate_up.gemm"] == [
        "gdn.expert_gate_up",
        "gated_gqa.expert_gate_up",
    ]
    assert semantics["unified.routed_expert.down.gemm"] == [
        "gdn.expert_down",
        "gated_gqa.expert_down",
    ]
    assert semantics["unified.shared_expert.gate_up.gemm"] == [
        "gdn.shared_gate_up",
        "gated_gqa.shared_gate_up",
    ]
    assert semantics["unified.shared_expert.down.gemm"] == [
        "gdn.shared_down",
        "gated_gqa.shared_down",
    ]
    assert semantics["unified.shared_expert.shared_gate"] == [
        "gdn.shared_gate",
        "gated_gqa.shared_gate",
    ]
    assert [(stack.tag, stack.count) for stack in qwen36_moe.layers] == [
        ("gdn", 30),
        ("gated_gqa", 10),
    ]
    label = qwen36_moe.label(Workload.causal_lm(decode=[32], sampled=1))
    segments = {segment.name: segment for segment in label.segments}
    assert segments["gdn.router"].count == 30
    assert segments["gated_gqa.router"].count == 10
    # Scale multiplicity lives on semantic Segment.count, never duplicate map rows.
    assert (
        len([row for row in location_map["locations"] if row["location"].endswith("router.gemm")])
        == 1
    )


def test_qwen36_local_location_map_empty_fusion_placeholders_are_exact():
    location_map = json.loads(QWEN36_LOCAL_MAP.read_text())
    empty = {row["location"] for row in location_map["locations"] if not row["semantics"]}
    assert empty == {
        "unified.gdn.qkvz.input_quant",
        "unified.gdn.split_b",
        "unified.gdn.split_a",
        "unified.gdn.core_output_zero",
        "unified.gdn.state_zero",
        "unified.gdn.prefill.cumsum",
        "unified.gdn.prefill.kkt",
        "unified.gdn.prefill.solve",
        "unified.gdn.prefill.recompute_w_u",
        "unified.gdn.prefill.output",
        "unified.gdn.core_output_copy",
        "unified.gdn.out_proj.input_quant",
        "unified.router.topk",
        "unified.router.align",
        "unified.routed_expert.gate_up.input_quant",
        "unified.routed_expert.activation",
        "unified.routed_expert.down.input_quant",
        "unified.shared_expert.gate_up.input_quant",
        "unified.shared_expert.silu_and_mul",
        "unified.shared_expert.down.input_quant",
        "unified.shared_expert.apply_shared_gate",
        "unified.finalize.finalize",
        "unified.finalize.shared_routed_add",
        "unified.gated_gqa.qkv_gate.input_quant",
        "unified.gated_gqa.partial_rope",
        "unified.gated_gqa.output_gate",
        "unified.gated_gqa.out_proj.input_quant",
    }
    assert len(empty) == 27


def test_moe_shared_scalar_gate_is_opt_in():
    base = MoE(2048, 512, 256, 8, shared_intermediate=512, shared_module="shared_expert")
    assert "shared_gate" not in {g.name for g in base.matmul_groups()}
    enabled = MoE(
        2048,
        512,
        256,
        8,
        shared_intermediate=512,
        shared_module="shared_expert",
        shared_gate=True,
    )
    gate = next(g for g in enabled.matmul_groups() if g.name == "shared_gate")
    assert (gate.n, gate.k, gate.module, gate.bucket) == (
        1,
        2048,
        "mlp.shared_expert_gate",
        "shared_expert",
    )
    with pytest.raises(ValueError, match="requires shared_intermediate"):
        MoE(2048, 512, 256, 8, shared_gate=True).matmul_groups()


def test_nested_quantization_paths_preserve_component_boundaries():
    raw = json.loads(
        (Path(__file__).resolve().parents[1] / "model/config/qwen3_6_35b_a3b_fp8.json").read_text()
    )
    quant = parse_quantization_config(raw)
    assert quant is not None
    for module in (
        "linear_attn.in_proj_ba",
        "linear_attn.conv1d",
        "mlp.gate",
        "mlp.shared_expert_gate",
        "lm_head",
        "embed_tokens",
    ):
        assert not quant.is_converted(module)
    assert quant.is_converted("linear_attn.in_proj_qkvz")
    assert quant.is_converted("linear_attn.out_proj")
    assert quant.is_converted("mlp.experts.gate_up_proj")
    # Legacy model.layers paths still normalize, and component matching must not
    # make `mlp.gate` exclude `mlp.gate_proj`.
    legacy = parse_quantization_config(
        {
            "quantization_config": {
                "quant_method": "fp8",
                "modules_to_not_convert": ["model.layers.7.mlp.gate"],
            }
        }
    )
    assert not legacy.is_converted("mlp.gate")
    assert legacy.is_converted("mlp.gate_proj")


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


# --------------------------------------------------------------------------- #
# Mixed precision: a quantized checkpoint's weight bytes and per-segment peaks
# --------------------------------------------------------------------------- #

GLM52_FP8 = Path(__file__).resolve().parents[1] / "model" / "config" / "glm52_fp8.json"
GLM52_NVFP4 = Path(__file__).resolve().parents[1] / "model" / "config" / "glm52_nvfp4.json"
QWEN3_235B = (
    Path(__file__).resolve().parents[1] / "model" / "config" / "qwen3_235b_thinking_2507.json"
)
QWEN3_235B_FP8 = (
    Path(__file__).resolve().parents[1] / "model" / "config" / "qwen3_235b_thinking_2507_fp8.json"
)
QWEN3_235B_A22B = Path(__file__).resolve().parents[1] / "model" / "config" / "qwen3_235b.json"
QWEN3_235B_A22B_FP8 = (
    Path(__file__).resolve().parents[1] / "model" / "config" / "qwen3_235b_fp8.json"
)

# Enough tokens that balls-in-bins loads every routed expert, so weight bytes are
# the whole checkpoint rather than the hit subset.
FULL_LOAD = Workload.causal_lm(prefill=[(1_000_000, 0)], sampled=1)


@pytest.mark.parametrize(
    ("bf16_path", "fp8_path"),
    [
        (GLM52, GLM52_FP8),
        (QWEN3_235B, QWEN3_235B_FP8),
        (QWEN3_235B_A22B, QWEN3_235B_A22B_FP8),
    ],
)
def test_fp8_config_differs_from_its_bf16_twin_by_exactly_one_key(bf16_path, fp8_path):
    """Both files are verbatim HF downloads; the FP8 repo adds only the quant block.

    If either side is ever hand-edited, or upstream changes one and not the other,
    this drifts and the pair no longer describes the same model.
    """
    bf16 = json.loads(bf16_path.read_text())
    fp8 = json.loads(fp8_path.read_text())
    quantization = fp8.pop("quantization_config")
    assert fp8 == bf16
    assert quantization["quant_method"] == "fp8"
    assert quantization["fmt"] == "e4m3"
    assert quantization["weight_block_size"] == [128, 128]


def test_glm52_fp8_weight_bytes_are_half_the_bf16_checkpoint():
    """The strongest independent check: predict the FP8 byte total from scratch.

    741.35e9 converted weights at one byte + their FP32 block scales + the 2.03e9
    parameters the checkpoint declines to convert at two bytes. Cross-checked
    against the published `total_size` of 755,617,140,416 bytes minus the MTP
    layer model.work does not build (~9.96e9) = 745.7e9.
    """
    fp8 = load_model(GLM52_FP8)
    bf16 = load_model(GLM52)
    fp8_label = fp8.label(FULL_LOAD)
    bf16_label = bf16.label(FULL_LOAD)

    converted = not_converted = 0
    for stack in fp8.layers:
        for group in (*stack.attn.matmul_groups(), *stack.ffn.matmul_groups()):
            params = group.total_params * stack.count
            if fp8.quant.is_converted(group.module):
                converted += params
            else:
                not_converted += params
    norm_params = sum(norm.elements * norm.count for norm in fp8.norm_weights)
    embed_and_head = 2 * GLM_VOCAB * GLM_HIDDEN
    predicted = (
        converted
        + converted / (128 * 128) * 4  # one FP32 scale per 128x128 block
        + (not_converted + norm_params + embed_and_head) * GLM_BF16_BYTES
    )
    assert fp8_label.bytes["weights"] == pytest.approx(predicted)
    assert fp8_label.bytes["weights"] == pytest.approx(745.58e9, rel=1e-3)
    fp8_ratio = fp8_label.bytes["weights"] / bf16_label.bytes["weights"]
    assert fp8_ratio == pytest.approx(0.5015, abs=1e-4)

    # Quantization changes stored bytes, never parameter counts or FLOPs.
    assert fp8_label.params == bf16_label.params
    assert fp8_label.flops == bf16_label.flops
    assert fp8_label.bytes["kv"] == bf16_label.bytes["kv"]


def test_glm52_fp8_leaves_router_and_indexer_weights_at_the_master_dtype():
    """Exactly two matmul families sit in the checkpoint's modules_to_not_convert."""
    model = load_model(GLM52_FP8)
    unconverted = {
        group.name
        for stack in model.layers
        for group in (*stack.attn.matmul_groups(), *stack.ffn.matmul_groups())
        if not model.quant.is_converted(group.module)
    }
    assert unconverted == {"router", "indexer.weights_proj"}
    # `mlp.gate` must not swallow the dense layers' `mlp.gate_up_proj`.
    assert model.quant.is_converted("mlp.gate_up_proj")
    assert not model.quant.is_converted("mlp.gate")
    assert not model.quant.is_converted("mlp.gate.e_score_correction_bias")


def test_glm52_nvfp4_hand_derived_weight_and_work_goldens():
    """NVFP4 changes routed-expert storage, not model math or parameters."""
    nvfp4 = load_model(GLM52_NVFP4)
    bf16 = load_model(GLM52)
    nvfp4_label = nvfp4.label(FULL_LOAD)
    bf16_label = bf16.label(FULL_LOAD)

    routed_elements = 75 * GLM_EXPERT_PARAMS
    expected_weights = (
        bf16_label.bytes["weights"]
        - routed_elements * GLM_BF16_BYTES
        + routed_elements * (0.5 + 1 / 16)
    )
    assert nvfp4_label.bytes["weights"] == pytest.approx(expected_weights)
    assert nvfp4_label.bytes["weights"] == pytest.approx(444_888_882_432.0)
    assert nvfp4_label.params == bf16_label.params
    assert nvfp4_label.flops == bf16_label.flops
    assert gpu_peak_tflops("B200", "fp4") == 9000


def test_glm52_nvfp4_quantizes_only_routed_experts_and_uses_fp8_mla_cache():
    model = load_model(GLM52_NVFP4)
    assert model.quant.compute_dtype == "fp4"
    assert model.quant.bytes_per_weight == 0.5
    assert model.quant.block_shape == (1, 16)
    assert model.quant.scale_dtype_bytes == 1.0

    converted = {
        group.name
        for stack in model.layers
        for group in (*stack.attn.matmul_groups(), *stack.ffn.matmul_groups())
        if model.quant.is_converted(group.module)
    }
    assert converted == {"expert_gate_up", "expert_down"}
    assert all(stack.attn.mla_cache_dtype_bytes == 1.0 for stack in model.layers)

    sparse = next(stack for stack in model.layers if stack.tag == "sparse_cycle_full_index")
    gate_up = next(group for group in sparse.ffn.matmul_groups() if group.name == "expert_gate_up")
    elements = gate_up.n * gate_up.k
    assert model.weight_bytes_per_instance(gate_up) == elements * (0.5 + 1 / 16)


def test_glm52_nvfp4_location_map_consumes_every_semantic_once():
    names = {
        segment.name
        for segment in load_model(GLM52_NVFP4)
        .label(Workload.causal_lm(prefill=[(8, 0)], decode=[4096], sampled=2))
        .segments
    }
    map_path = (
        Path(__file__).resolve().parents[1]
        / "model"
        / "work"
        / "location_maps"
        / "glm52_vllm_nvfp4_dsa_moe_unified.json"
    )
    location_map = json.loads(map_path.read_text())
    locations = [row["location"] for row in location_map["locations"]]
    mapped = [semantic for row in location_map["locations"] for semantic in row["semantics"]]

    assert location_map["schema_version"] == 1
    assert location_map["arch_types"] == ["glm52_vllm_nvfp4_dsa_moe"]
    assert len(locations) == len(set(locations)) == 114
    assert len(mapped) == len(set(mapped))
    assert set(mapped) == names
    assert not any(location.endswith(".tp_allreduce") for location in locations)
    semantics = {row["location"]: row["semantics"] for row in location_map["locations"]}
    for tag in (
        "sparse_initial_index_share",
        "sparse_cycle_full_index",
        "sparse_cycle_index_share",
    ):
        base = f"unified.body.{tag}.moe.routed_experts"
        assert f"{base}.input_quant" in locations
        assert not semantics[f"{base}.input_quant"]
        assert semantics[f"{base}.fused_moe"] == [
            f"{tag}.expert_gate_up",
            f"{tag}.expert_down",
        ]


def test_glm52_fp8_compute_floor_is_mixed_not_globally_fp8():
    """~20% of the FLOPs stay on the BF16 tensor cores, so one global peak is wrong.

    Sparse MLA runs vLLM's BF16 FlashMLA kernel, and the router / lm_head /
    indexer weight projection are unconverted. H200's FP8 peak is 1979 against
    990 BF16, so the mixed compute floor exceeds the naive all-FP8 one by exactly
    `bf16_share * (1979/990 - 1)` — a fifth on the alignment run's prefill shape,
    not a rounding detail.
    """
    model = load_model(GLM52_FP8)
    for workload, expected_bf16_share in (
        (Workload.causal_lm(prefill=[(8192, 0)], sampled=1), 0.1994),
        (Workload.causal_lm(decode=[4096] * 64, sampled=64), 0.2356),
    ):
        label = model.label(workload)
        flops_by_dtype: dict[str, float] = {}
        for segment in label.segments:
            flops_by_dtype[segment.compute_dtype] = (
                flops_by_dtype.get(segment.compute_dtype, 0.0) + segment.flops_total
            )
        assert set(flops_by_dtype) == {"fp8", "bf16"}
        bf16_share = flops_by_dtype["bf16"] / sum(flops_by_dtype.values())
        assert bf16_share == pytest.approx(expected_bf16_share, abs=1e-4)

        mixed_compute_ms, _memory_ms, _bound = label.roofline_ms("H200", "fp8")
        naive_fp8_ms = label.flops_total / (1979.0 * 1e12) * 1e3
        penalty = bf16_share * (1979.0 / 990.0 - 1.0)
        assert mixed_compute_ms / naive_fp8_ms == pytest.approx(1.0 + penalty, rel=1e-9)

    # The BF16 twin makes no fp8 claim on its matmuls, but the DSA index logits
    # are FP8 by construction of the mechanism, not of the checkpoint.
    bf16_dtypes = {segment.compute_dtype for segment in load_model(GLM52).label(FULL_LOAD).segments}
    assert bf16_dtypes == {"bf16", "fp8"}


def test_floors_refuses_a_run_whose_arch_precision_contradicts_its_config():
    fp8_spec = {"config": str(GLM52_FP8), "gpu": "H200", "dtype": "fp8", "arch_fp8": True}
    bf16_spec = {"config": str(GLM52), "gpu": "H200", "dtype": "bf16", "arch_fp8": False}
    work_floors._check_precision(load_model(GLM52_FP8), fp8_spec)
    work_floors._check_precision(load_model(GLM52), bf16_spec)
    with pytest.raises(ValueError, match="no quantization_config"):
        work_floors._check_precision(load_model(GLM52), {**bf16_spec, "arch_fp8": True})
    with pytest.raises(ValueError, match="does not set fp8"):
        work_floors._check_precision(load_model(GLM52_FP8), {**fp8_spec, "arch_fp8": False})

    nvfp4_spec = {
        "config": str(GLM52_NVFP4),
        "gpu": "B200",
        "dtype": "bf16",
        "arch_fp8": False,
        "arch_quant_dtype": "fp4",
    }
    work_floors._check_precision(load_model(GLM52_NVFP4), nvfp4_spec)
    with pytest.raises(ValueError, match="declares fp4"):
        work_floors._check_precision(
            load_model(GLM52_NVFP4), {**nvfp4_spec, "arch_quant_dtype": "fp8"}
        )


def test_vllm_location_map_covers_every_non_communication_leaf():
    """Same semantic rows as the native map, over vLLM's finer leaf decomposition."""
    names = {
        segment.name
        for segment in load_model(GLM52_FP8)
        .label(Workload.causal_lm(prefill=[(8, 0)], decode=[4096], sampled=2))
        .segments
    }
    maps_dir = Path(__file__).resolve().parents[1] / "model" / "work" / "location_maps"
    location_map = json.loads((maps_dir / "glm52_vllm_dsa_moe_unified.json").read_text())
    assert location_map["schema_version"] == 1
    assert location_map["arch_types"] == ["glm52_vllm_dsa_moe"]
    locations = [row["location"] for row in location_map["locations"]]
    assert len(locations) == len(set(locations)) == 166
    mapped = [semantic for row in location_map["locations"] for semantic in row["semantics"]]
    assert len(mapped) == len(set(mapped))
    assert set(mapped) == names
    # The MoE exchange itself is communication and must not appear as a location.
    assert not any(location.endswith((".moe.dispatch", ".moe.combine")) for location in locations)
    # vLLM splits each routed-expert projection into a quantize and a grouped GEMM;
    # only the GEMM carries the semantic weight work.
    native = json.loads((maps_dir / "glm52_dsa_moe_unified.json").read_text())
    native_semantics = {row["location"]: row["semantics"] for row in native["locations"]}
    vllm_semantics = {row["location"]: row["semantics"] for row in location_map["locations"]}
    for tag in (
        "sparse_initial_index_share",
        "sparse_cycle_full_index",
        "sparse_cycle_index_share",
    ):
        for projection in ("gate_up", "down"):
            native_name = f"unified.body.{tag}.moe.routed_experts.{projection}"
            assert vllm_semantics[f"{native_name}.gemm"] == native_semantics[native_name]
            assert vllm_semantics[f"{native_name}.input_quant"] == []


def test_qwen3_moe_fp8_quantizes_experts_but_not_the_router():
    """The shared MoE/GQA specs carry checkpoint module names, so Qwen quantizes too."""
    fp8 = load_model(QWEN3_235B_A22B_FP8)
    bf16 = load_model(QWEN3_235B_A22B)
    unconverted = {
        group.name
        for stack in fp8.layers
        for group in (*stack.attn.matmul_groups(), *stack.ffn.matmul_groups())
        if not fp8.quant.is_converted(group.module)
    }
    assert unconverted == {"router"}

    workload = Workload.causal_lm(prefill=[(1_000_000, 0)], sampled=1)
    fp8_label, bf16_label = fp8.label(workload), bf16.label(workload)
    assert fp8_label.params == bf16_label.params
    assert fp8_label.flops == bf16_label.flops
    # lm_head and the embedding table stay BF16 in this checkpoint too, so the
    # ratio sits just above the 0.5 a fully converted model would reach.
    ratio = fp8_label.bytes["weights"] / bf16_label.bytes["weights"]
    assert 0.50 < ratio < 0.52

    # modules_to_not_convert is an exclusion list over the Linear modules the
    # quantizer walks, so it is silent about the embedding table here while GLM
    # lists it explicitly. Neither checkpoint stores a
    # `model.embed_tokens.weight_scale_inv`, so both must price it at BF16.
    assert fp8.quant.is_converted("embed_tokens")
    assert not load_model(GLM52_FP8).quant.is_converted("embed_tokens")
    for model, label in ((fp8, fp8_label), (load_model(GLM52_FP8), None)):
        label = label or model.label(FULL_LOAD)
        embedding = next(seg for seg in label.segments if seg.name == "embedding")
        assert embedding.compute_dtype == "bf16"
        assert embedding.bytes == min(1_000_000, model.vocab) * model.hidden * 2
