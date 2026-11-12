"""Label rules for Qwen3.8-Flash-Next-NVFP4 (MTP k=3) on one B200 (vLLM fork 892da082 / e1423a4d) against the
qwen4_exp_vllm_nvfp4_gdn_qsa_moe_speculative CostTree. Writes rules.json beside this file.

The kernel side is the diverse_100 capture at this pack's chunk-2048 serving config (Slurm job 6613,
logs/20261013_1_qf_diverse100/profile_nsys), checked against the folded inventories of the 20 campaign cases. The slot
side is the arch's cost manifest (`alignment label slots`).

There is no torch.compile here (breakable CUDA graphs, compile mode NONE), so every eager torch op is its own launch and
every dense linear is a BF16 cuBLAS GEMM (nvjet, a CUTLASS sm80 tile, or a gemv for the N=1 shared-expert gate, with an
optional split-K reduce). GEMMs are told apart by their neighbours only.

forward (one step, FULL graph for a pure verify step, breakable piecewise otherwise):
  prologue   input-id cast, embedding gather, hidden.repeat, PLE n-gram ids (+ the UVA lookup on a side stream)
  HC module  entry norm (hc_combine_norm, or grouped Gemma RMSNorm for layer 0 / after PLE) -> down GEMM (+ reduce)
             -> hc_silu -> up GEMM -> hc_gate_mix; two per layer plus the final mixer
  GDN        qkvz GEMM -> ba GEMM (+ reduce) -> core_attn_out zeros ->
               pure verify: causal_conv1d_update -> fused post-conv MTP recurrence + gated norm
               otherwise:   b / a contiguous copies, row index_selects (spec qkv/a/b, non-spec qkv/a/b),
                            spec causal_conv1d_update, causal_conv1d_fwd, spec rearrange copies + cat, post-conv,
                            sigmoid-gating recurrence (spec rows), state gather / zero / scatter around the FlashInfer
                            chunked delta rule (with their int64 casts), cu_seqlens cast, exp(g), index_copy_ x2,
                            output_gate copy, gated RMSNorm
             -> out_proj GEMM
  QSA        qkv GEMM -> q/k norm + RoPE + gate -> index_qk GEMM -> pre-indexer -> indexer logits (uniform / prefill)
             -> top-k (cooperative / filtered) -> expand -> FP8 KV append -> out zero -> sparse split-K attention
             (+ merge) -> o_proj GEMM
  PLE        (layer 1) hc_combine, finalize copy, fp8->bf16 cast, scale mul, kv_proj GEMM, ple_gate, ple_conv
             (+ writeback; twice in a mixed step)
  MoE        router GEMM (+ reduce) -> NVFP4 quant -> trtllm routing / FC1 / FC2 / finalize; BF16 shared expert
             (gate_up, SiluAndMul, down, gate GEMM or gemv, sigmoid, mul; on the aux stream when T <= 256);
             shared + routed add
draft: MTP step 0 then 2 endpoint passes: input prep (unmodeled), embed, Gemma RMSNorm (native launches) -> fc_embedding,
  Gemma RMSNorm over the 4 streams -> fc_hidden, then an HC + QSA + HC + FP8-block MoE layer, the final mixer and the
  greedy head (row gather, lm_head, argmax). Draft operations claim both the step_0 and the recurrent slots.
sample: logits row gather, lm_head, logits stats, rejection / resample / insert / count kernels.
preprocess: the input build (placeholder input_prep), the QSA metadata build and the Mamba align pre-copy.

Run from the repo root:
  uv run python presets/alignment/qwen38_flash_next_nvfp4_b200_tp1/label_rules/gen_rules.py
"""
import json
from pathlib import Path

SRC = "20261013_1_qf_diverse100/profile_nsys (Slurm job 6613) + 20261013_2_qf_cases 01-20"
NONE = "<none>"
rules: list[dict] = []
bodies: dict[str, tuple] = {}

GEMMS = ("nvjet_sm100_tst_", "cutlass_80_tensorop_s16816gemm_bf16")
SPLITK = "cublasLt::splitKreduce_kernel"
GEMV = "dot_kernel<float, (int)128, (int)0, cublasDotParams<cublasGemvTensorStridedBatched<const __nv_bfloat16>"
GEMV_REDUCE = "reduce_1Block_kernel<float, (int)128, (int)7, cublasGemvTensorStridedBatched<float>"
COMBINE_NORM = "_hc_combine_norm_kernel"
GROUPED_NORM = "_grouped_gemma_rmsnorm_kernel"
HC_SILU = "_hc_silu_kernel"
GATE_MIX = "_hc_gate_mix_kernel"
FP4_QUANT = "vllm::cvt_fp16_to_fp4"
FP8_QUANT = "per_token_group_quant_8bit_kernel"
ACT_MUL = "vllm::act_and_mul_kernel"
SIGMOID = "sigmoid_kernel_cuda"
MUL_BF16 = "BinaryFunctor<c10::BFloat16, c10::BFloat16, c10::BFloat16, at::native::binary_internal::MulFunctor"
ADD_BF16 = "CUDAFunctor_add<c10::BFloat16>"
FILL_BF16 = "FillFunctor<c10::BFloat16>"
BF16_COPY = "at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda"
LONG_COPY = "lambda() (instance 4)]::operator ()() const::[lambda(long) (instance 1)]"
UNROLLED_COPY = "at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda"
SCALE_COPY = "at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda"
GATHER = "vectorized_gather_kernel"
ISEL = "indexSelectSmallIndex"
CAT = "CatArrayBatchedCopy"
QK_ROPE = "_fused_qk_rmsnorm_rope_gate_kernel"
ATTN = "_qsa_sparse_paged_gqa_splitk_kernel"
MERGE = "_qsa_merge_splitk_kernel"
POST_CONV_MTP = "gdn_decode_post_conv_mtp_kernel"
GATED_NORM = "layer_norm_fwd_kernel"
ROUTING = "moe::dev::routing::"
FINALIZE = "moe::dev::finalize::"
ARGMAX = "ArgMaxOps"


def op(operation, type_, role, slots, phase="forward", exclude_mtp=True):
    key = (type_, role, tuple(slots), phase)
    assert bodies.setdefault(operation, key) == key, operation

    def rule(name, **evidence):
        r = {"operation": operation, "name": name, "type": type_, "role": role, "slot_suffixes": list(slots),
             "phase": phase}
        if exclude_mtp:
            r["excluded_slot_prefixes"] = ["unified.mtp."]
        r.update(evidence)
        r["source_experiment"] = SRC
        rules.append(r)
    return rule


def gemm(rule, operation, **evidence):
    """A GEMM identified by what precedes it: the GEMM, then its split-K reduce (if any) right after it."""
    for g in GEMMS:
        rule(g, **evidence)
    rule(SPLITK, after=operation)


def gemm_before(rule, operation, before_name=None, before=None):
    """A GEMM identified by what follows it, with or without a split-K reduce in between."""
    ev = {"before_name": before_name} if before_name else {"before": before}
    for g in GEMMS:
        rule(g, **ev)
        rule(g, before=operation, before_name=SPLITK)
    rule(SPLITK, **ev)


# ======================================================================================== forward + draft (shared)
def layer(phase):
    """The HC / QSA / MoE kernels both the target layers and the MTP layer launch."""
    draft = phase == "draft"
    P = "mtp." if draft else ""

    def o(name, type_, role, suffixes):
        if draft:
            slots = [f"{s}.{x}" for x in suffixes for s in ("step_0", "recurrent")]
            return op(P + name, type_, role, slots, phase=phase, exclude_mtp=False)
        return op(name, type_, role, suffixes, phase=phase)

    # HC module
    o("hc.combine_norm", "model", "hc_combine_norm: residual + injected block output, grouped norm",
      ["attn_hc.combine_norm", "mlp_hc.combine_norm", "final_mixer.combine_norm"])(COMBINE_NORM)
    down = o("hc.down", "model", "HC mix down-projection BF16 GEMM (10240 -> 336, final mixer 320)",
             ["attn_hc.down", "mlp_hc.down", "final_mixer.down"])
    gemm(down, P + "hc.down", after_name=COMBINE_NORM)
    if not draft:
        for g in GEMMS:
            down(g, after_name=GROUPED_NORM)
    o("hc.silu", "model", "hc_silu (x/4 then SiLU)", ["attn_hc.silu", "mlp_hc.silu", "final_mixer.silu"])(HC_SILU)
    up = o("hc.up", "model", "HC mix up-projection BF16 GEMM (320 -> 10240)",
           ["attn_hc.up", "mlp_hc.up", "final_mixer.up"])
    gemm(up, P + "hc.up", after_name=HC_SILU)
    o("hc.gate_mix", "model", "hc_gate_mix: sigmoid-gated mean over the 4 streams",
      ["attn_hc.gate_mix", "mlp_hc.gate_mix", "final_mixer.gate_mix"])(GATE_MIX)

    # QSA
    qkv = o("qsa.qkv_proj", "attention", "q/k/v + gate BF16 GEMM (2560 -> 13312)", ["qsa.qkv_proj"])
    gemm_before(qkv, P + "qsa.qkv_proj", before_name=QK_ROPE)
    o("qsa.qk_norm_rope_gate", "attention", "q/k RMSNorm + partial RoPE + gate split",
      ["qsa.qk_norm_rope_gate"])(QK_ROPE)
    idx = o("qsa.index_qk_proj", "attention", "indexer q/k BF16 GEMM (2560 -> 640)", ["qsa.index_qk_proj"])
    gemm(idx, P + "qsa.index_qk_proj", after_name=QK_ROPE)
    o("qsa.indexer.prepare", "attention", "qsa_pre_indexer: indexer q norm + RoPE, compressed key cache, ring",
      ["qsa.indexer.prepare"])("_qsa_pre_indexer_kernel")
    o("qsa.indexer.logits_decode", "attention", "indexer ReLU-sum logits over the paged compressed keys (uniform rows)",
      ["qsa.indexer.logits_decode"])("_qsa_mqa_paged_uniform_kernel")
    o("qsa.indexer.logits_prefill", "attention", "indexer logits of prefill rows (per chunk)",
      ["qsa.indexer.logits_prefill"])("_qsa_mqa_paged_prefill_kernel")
    o("qsa.indexer.topk_cooperative", "attention", "cooperative top-512 (<= 64 rows)",
      ["qsa.indexer.topk_cooperative"])("vllm::cooperative::cooperative_topk")
    topk = o("qsa.indexer.topk_persistent", "attention", "filtered / persistent top-512",
             ["qsa.indexer.topk_persistent"])
    topk("vllm::filtered_topk::FilteredTopKUnifiedKernel")
    topk("persistent_topk")
    o("qsa.indexer.expand", "attention", "expand compressed block indices to token indices",
      ["qsa.indexer.expand"])("_expand_qsa_indices_kernel")
    o("qsa.attention.kv_append", "attention", "FP8 paged KV append (reshape_and_cache_flash)",
      ["qsa.attention.kv_append"])("vllm::reshape_and_cache_flash_kernel")
    o("qsa.attention.out_zero", "attention", "output.zero_() before the sparse attention",
      ["qsa.attention.out_zero"])(FILL_BF16, after=P + "qsa.attention.kv_append")
    attn = o("qsa.attention", "attention", "sparse GQA paged split-K attention with the sigmoid gate (+ split merge)",
             ["qsa.attention.attn_uniform", "qsa.attention.attn_prefill_cfg"])
    attn(ATTN)
    attn(MERGE)
    o_proj = o("qsa.o_proj", "attention", "o_proj BF16 GEMM (6144 -> 2560)", ["qsa.o_proj"])
    gemm(o_proj, P + "qsa.o_proj", after_name=MERGE)
    gemm(o_proj, P + "qsa.o_proj", after_name=ATTN)

    # MoE: router, routed experts, BF16 gated shared expert, shared + routed add
    quant = FP8_QUANT if draft else FP4_QUANT
    router = o("moe.router", "ffn", "router BF16 GEMM (2560 -> 512)", ["moe.router"])
    gemm_before(router, P + "moe.router", before_name=quant)
    for g in GEMMS:
        router(g, before=P + "moe.shared.gate_up", before_name=g)
    o("moe.routed.input_quant", "ffn",
      "FP8 block-128 activation quant" if draft else "NVFP4 activation quant (swizzled scales)",
      ["moe.routed.input_quant"])(quant)
    fused = o("moe.routed.fused_moe", "ffn",
              "trtllm-gen " + ("FP8 block-scale" if draft else "NVFP4") + " MoE: routing, FC1, (activation,) FC2, finalize",
              ["moe.routed.fused_moe"])
    fused(ROUTING)
    fused("bmm_")
    fused(FINALIZE)
    if draft:
        fused("moe::dev::activation::")
        o("moe.routed_scale_transpose", "ffn", "FP8 activation-scale transpose copy",
          ["moe.routed_scale_transpose"])(SCALE_COPY, after_name=FP8_QUANT)
    sh = lambda leaf: [f"moe.shared.{leaf}", f"moe.shared_serial.{leaf}"]
    gate_up = o("moe.shared.gate_up", "ffn", "shared expert gate_up BF16 GEMM (2560 -> 1280)", sh("gate_up"))
    gemm_before(gate_up, P + "moe.shared.gate_up", before_name=ACT_MUL)
    o("moe.shared.act_and_mul", "ffn", "shared expert SiluAndMul (CUDA op)", sh("act_and_mul"))(ACT_MUL)
    sdown = o("moe.shared.down", "ffn", "shared expert down BF16 GEMM (640 -> 2560)", sh("down"))
    gemm(sdown, P + "moe.shared.down", after_name=ACT_MUL)
    sgate = o("moe.shared.gate", "ffn", "shared_expert_gate BF16 GEMM / gemv (2560 -> 1)", sh("gate"))
    gemm(sgate, P + "moe.shared.gate", after=P + "moe.shared.down")
    sgate(GEMV, after=P + "moe.shared.down")
    sgate(GEMV_REDUCE, after=P + "moe.shared.gate")
    o("moe.shared.gate_sigmoid", "ffn", "sigmoid of the shared-expert gate", sh("gate_sigmoid"))(SIGMOID)
    o("moe.shared.gate_mul", "ffn", "sigmoid(gate) * shared output (broadcast mul)",
      sh("gate_mul"))(MUL_BF16, after_name=SIGMOID)
    o("moe.shared_routed_add", "ffn", "shared + routed expert add", ["moe.shared_routed_add"])(ADD_BF16)
    return o


# ---------------------------------------------------------------- forward-only
fw = layer("forward")

# prologue
emb = op("model.embedding", "model", "embed_tokens row gather (BF16 vocabulary)", ["embedding"])
emb(GATHER, before_name=BF16_COPY)
emb(ISEL, before_name=BF16_COPY)
op("model.hidden_repeat", "model", "hidden.repeat(1, hc_count)", ["hidden_repeat"])(
    BF16_COPY, before_name="_ple_ngram_ids_kernel")
fw("hc.grouped_norm", "model", "grouped Gemma RMSNorm entry of layer 0 and of the layer after PLE",
   ["attn_hc.grouped_norm"])(GROUPED_NORM)

# PLE (layer 1)
op("ple.ngram_ids", "model", "PLE n-gram ids", ["ple.ngram_ids"])("_ple_ngram_ids_kernel")
op("ple.lookup", "model", "PLE FP8 row lookup from the pinned host table (UVA, side stream)",
   ["ple.lookup"])("_lookup_ple_embedding_from_pinned_kernel")
op("ple.hc_combine", "model", "standalone hc_combine before the PLE block", ["ple.hc_combine"])("_hc_combine_kernel")
fin = op("ple.finalize_copy", "model", "_finalize_prefetch copy of the looked-up rows", ["ple.finalize_copy"])
fin(UNROLLED_COPY, before_name="bfloat16_copy_kernel_cuda")
op("ple.dequant_cast", "model", "PLE dequant .to(bf16)", ["ple.dequant_cast"])("bfloat16_copy_kernel_cuda")
op("ple.dequant_scale", "model", "PLE dequant * scale", ["ple.dequant_scale"])(
    MUL_BF16, after_name="bfloat16_copy_kernel_cuda")
kvp = op("ple.kv_proj", "model", "PLE key/value BF16 GEMM (2560 -> 12800)", ["ple.kv_proj"])
gemm_before(kvp, "ple.kv_proj", before_name="_ple_gate_kernel")
op("ple.gate", "model", "PLE grouped norms + sign-sqrt-sigmoid gate", ["ple.gate"])("_ple_gate_kernel")
conv = op("ple.conv", "model", "PLE dilated depthwise short conv (+ state writeback); spec and prefill calls",
          ["ple.conv_decode", "ple.conv_spec", "ple.conv_prefill"])
conv("_ple_conv_kernel")
conv("_ple_conv_writeback_kernel")

# GDN
qkvz = op("gdn.qkvz", "attention", "in_proj_qkvz BF16 GEMM (2560 -> 16384)", ["gdn.qkvz"])
for g in GEMMS:
    for g2 in GEMMS:
        qkvz(g, before="gdn.ba", before_name=g2)
ba = op("gdn.ba", "attention", "in_proj_ba BF16 GEMM (2560 -> 96), with its split-K reduce", ["gdn.ba"])
gemm_before(ba, "gdn.ba", before_name=FILL_BF16)
op("gdn.core_output_zero", "attention", "torch.zeros(core_attn_out)", ["gdn.core_output_zero"])(
    FILL_BF16, after="gdn.ba")
bac = op("gdn.ba_contiguous", "attention", "b.contiguous() and a.contiguous()", ["gdn.ba_contiguous"])
bac(BF16_COPY, after="gdn.core_output_zero")
bac(BF16_COPY, after="gdn.ba_contiguous")
sel = op("gdn.row_select", "attention",
         "index_select of the spec rows (qkv, a, b) and the non-spec rows (qkv, a, b) of a mixed step",
         ["gdn.spec_qkv_select", "gdn.spec_ab_select", "gdn.non_spec_qkv_select", "gdn.non_spec_ab_select"])
for k in (GATHER, ISEL):
    sel(k, after="gdn.ba_contiguous")
    sel(k, after="gdn.row_select")
    sel(k, after="gdn.spec_rearrange")
    sel(k, before="gdn.row_select")
    sel(k, before_name="_causal_conv1d_update_kernel")
    sel(k, before_name="_fused_post_conv_kernel")
    sel(k, after_name=NONE, before_name=NONE,
        note="a lone folded gather body in the forward is always a run of GDN row selects")
cu = "_causal_conv1d_update_kernel"
op("gdn.verify_conv", "attention", "causal_conv1d_update of a pure verify step (fused path)",
   ["gdn.verify_conv"])(cu, before_name=POST_CONV_MTP)
op("gdn.spec_decode.causal_conv", "attention", "causal_conv1d_update of a mixed step's spec rows",
   ["gdn.spec_decode.causal_conv"])(cu, before_name="_causal_conv1d_fwd_kernel")
op("gdn.verify_recurrent_norm", "attention", "fused_gdn_decode_post_conv_mtp (recurrence + gated norm)",
   ["gdn.verify_recurrent_norm"])(POST_CONV_MTP)
op("gdn.prefill.causal_conv", "attention", "varlen causal_conv1d_fwd", ["gdn.prefill.causal_conv"])(
    "_causal_conv1d_fwd_kernel")
rea = op("gdn.spec_rearrange", "attention", "rearrange_mixed_qkv of the spec rows: q / k / v copies and their cat",
         ["gdn.spec_qk_copy", "gdn.spec_v_copy", "gdn.spec_qkv_cat"])
rea(BF16_COPY, after_name="_causal_conv1d_")
rea(BF16_COPY, after="gdn.spec_rearrange")
rea(BF16_COPY, after_name=NONE, before_name=NONE,
    note="a lone folded copy body in the forward is always the spec q/k/v rearrange")
rea(CAT)
op("gdn.prefill.post_conv", "attention", "post-conv split / gating of the prefill rows",
   ["gdn.prefill.post_conv"])("_fused_post_conv_kernel")
op("gdn.spec_decode.recurrent", "attention", "sigmoid-gating delta-rule update of the spec rows",
   ["gdn.spec_decode.recurrent"])("fused_sigmoid_gating_delta_rule_update_kernel")
sg = op("gdn.state_gather", "attention", "ssm_state[prefill indices] (int64 index cast + gather)", ["gdn.state_gather"])
sg(UNROLLED_COPY, after_name="fused_sigmoid_gating_delta_rule_update_kernel")
sg(UNROLLED_COPY, after_name="_fused_post_conv_kernel")
sg("index_kernel_impl<at::native::OpaqueType<(int)4>>", after="gdn.state_gather")
sg(GATHER, after="gdn.state_gather")
sz = op("gdn.state_zero", "attention", "initial_state[~has_initial_state] = 0", ["gdn.state_zero"])
sz("bitwise_not_kernel_cuda")
sz("masked_fill_kernel")
op("gdn.cu_seqlens_cast", "attention", "cu_seqlens.to(int64)", ["gdn.cu_seqlens_cast"])(
    UNROLLED_COPY, before_name="exp_kernel_cuda")
op("gdn.g_exp", "attention", "exp(g) of the prefill rows", ["gdn.g_exp"])("exp_kernel_cuda")
op("gdn.prefill.chunk_delta_rule", "attention", "FlashInfer Blackwell chunked gated delta rule",
   ["gdn.prefill.chunk_delta_rule"])("GatedDeltaNetChunkedKernel")
ss = op("gdn.state_scatter", "attention", "ssm_state[prefill indices] = last state (index cast + index_put)",
        ["gdn.state_scatter"])
ss(UNROLLED_COPY, before_name="index_put_kernel_impl")
ss("index_put_kernel_impl")
op("gdn.output_index_copy", "attention", "core_attn_out.index_copy_ of the spec and non-spec outputs",
   ["gdn.spec_output_index_copy", "gdn.non_spec_output_index_copy"])("index_copy_kernel_impl")
op("gdn.output_gate_copy", "attention", "output_gate reshape copy in the gated norm", ["gdn.output_gate_copy"])(
    BF16_COPY, before_name=GATED_NORM)
op("gdn.gated_norm", "attention", "gated RMSNorm (sigmoid gate)", ["gdn.gated_norm"])(GATED_NORM)
outp = op("gdn.out_proj", "attention", "out_proj BF16 GEMM (6144 -> 2560)", ["gdn.out_proj"])
gemm(outp, "gdn.out_proj", after_name=GATED_NORM)
gemm(outp, "gdn.out_proj", after_name=POST_CONV_MTP)

# ---------------------------------------------------------------- draft-only
dr = layer("draft")
D = lambda leaf: [f"step_0.{leaf}", f"recurrent.{leaf}"]
tok = op("mtp.prelude.token_embedding", "model", "MTP embed_tokens row gather", D("prelude.token_embedding"),
         phase="draft", exclude_mtp=False)
tok(GATHER, before_name=UNROLLED_COPY)
tok(ISEL, before_name=UNROLLED_COPY)
en = op("mtp.prelude.embedding_norm", "model", "pre_fc_norm_embedding: Gemma RMSNorm as native torch launches",
        D("prelude.embedding_norm"), phase="draft", exclude_mtp=False)
en("at::native::", after="mtp.prelude.token_embedding")
en("at::native::", after="mtp.prelude.embedding_norm")
fce = op("mtp.prelude.fc_embedding", "model", "fc_embedding BF16 GEMM (2560 -> 2560)", D("prelude.fc_embedding"),
         phase="draft", exclude_mtp=False)
gemm(fce, "mtp.prelude.fc_embedding", after="mtp.prelude.embedding_norm")
hn = op("mtp.prelude.hidden_norm", "model", "pre_fc_norm_hidden over the 4 streams: native Gemma RMSNorm",
        D("prelude.hidden_norm"), phase="draft", exclude_mtp=False)
hn("at::native::", after="mtp.prelude.fc_embedding")
hn("at::native::", after="mtp.prelude.hidden_norm")
fch = op("mtp.prelude.fc_hidden", "model", "fc_hidden BF16 GEMM (2560 -> 2560 per stream)", D("prelude.fc_hidden"),
         phase="draft", exclude_mtp=False)
gemm(fch, "mtp.prelude.fc_hidden", after="mtp.prelude.hidden_norm")
lg = op("mtp.head.logits_gather", "model", "endpoint row gather before the draft lm_head", D("head.logits_gather"),
        phase="draft", exclude_mtp=False)
for g in GEMMS:
    lg(GATHER, before_name=g)
    lg(ISEL, before_name=g)
dlm = op("mtp.head.lm_head", "model", "draft lm_head BF16 GEMM (2560 -> 248320)", D("head.lm_head"), phase="draft",
         exclude_mtp=False)
gemm_before(dlm, "mtp.head.lm_head", before_name=ARGMAX)
op("mtp.head.logits_reduce", "model", "greedy argmax over the draft logits", D("head.logits_reduce"), phase="draft",
   exclude_mtp=False)(ARGMAX)

# ---------------------------------------------------------------- sample (target head + rejection sampler)
sel_rows = op("head.logits_gather", "model", "hidden[logits_indices] row gather", ["head.logits_gather"],
              phase="sample")
for g in GEMMS:
    sel_rows(GATHER, before_name=g)
    sel_rows("index_kernel_impl", before_name=g)
lm = op("head.lm_head", "model", "target lm_head BF16 GEMM (2560 -> 248320)", ["head.lm_head"], phase="sample")
for g in GEMMS:
    lm(g)
lm(SPLITK, after="head.lm_head")
op("sample.logits_stats", "model", "per-vocab-block logits stats of the rejection sampler", ["sample.logits_stats"],
   phase="sample")("_compute_local_logits_stats_kernel")
rr = op("sample.rejection_resample", "model", "rejection, resample, insert and accepted-count kernels",
        ["sample.rejection_resample"], phase="sample")
for k in ("_rejection_kernel", "_resample_kernel", "_insert_resampled_kernel", "_get_num_sampled_and_rejected_kernel"):
    rr(k)

# ---------------------------------------------------------------- preprocess (decision D8 placeholder group)
op("prep.qsa_metadata", "model", "QSA sparse-attention metadata build", ["prep.qsa_metadata"],
   phase="preprocess")("_build_qsa_metadata_kernel")
op("prep.mamba_align_precopy", "model", "Mamba align-mode state pre-copy", ["prep.mamba_align_precopy"],
   phase="preprocess")("precopy_mamba_align_fused_kernel")
ip = op("prep.input_prep", "model", "Model Runner V2 input build (index / copy / fill / scan / sort and Triton prep)",
        ["prep.input_prep"], phase="preprocess")
for k in ("at::native::", "at_cuda_detail::", "_prepare_", "_combine_sampled_and_draft_tokens_kernel",
          "_gather_block_tables_kernel", "_compute_slot_mappings_kernel", "_zero_kv_blocks_kernel", "_apply_write_kernel",
          "_expand_idx_mapping_kernel", "compute_cuda_kernel", "preprocess_mamba_align_fused_kernel"):
    ip(k)

out = Path(__file__).resolve().parent / "rules.json"
out.write_text(json.dumps({
    "schema_version": 2,
    "note": "Generated by gen_rules.py against the qwen4_exp_vllm_nvfp4_gdn_qsa_moe_speculative CostTree.",
    "rules": rules,
}, indent=1) + "\n")
print(len(rules), "rules over", len(bodies), "operations")
