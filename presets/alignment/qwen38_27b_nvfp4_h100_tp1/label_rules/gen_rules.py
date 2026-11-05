"""Label rules for Qwen3.8-27B-NVFP4 on one Hopper GPU (H100 / H200; ServingStudio vLLM fork e1423a4d, TP1)
against the SM90 kernel path of the qwen35_vllm_nvfp4_gdn_dense CostTree. Writes rules.json beside this file.
The H100 and H200 packs carry the same file: both GPUs run the same kernels (the captures differ only in a few
cuBLASLt / Marlin tilings, which the name fragments below cover).

The kernel side is read from the folded inventories of the four campaign cases (03, 04, 05, 11) captured on H100
(ptc) and H200 (coriander) on 2026-11-03. The slot side is the SM90 cost manifest (`alignment label slots`).

One forward, as captured (FULL / PIECEWISE graph nodes, one track). Where it differs from the B200 graph:
  layer 0 input RMSNorm with the embedding gather and the FP8 row fused (Inductor)
  GDN layer: input norm (+FP8 row) -> in_proj_qkvz Cutlass SM90 FP8 GEMM -> in_proj_ba BF16 GEMM (cuBLASLt
             nvjet sm90, with a split-K reduce on some shapes) -> core_attn_out zeros -> b / a contiguous copies ->
             core, as on B200, except that the chunked delta rule is FlashInfer's SM90 kernels: the context-parallel
             set (MN precompute, T precompute, prefill, HMMA fixup) or the fully fused kernel, by shape
             -> core_attn_out copy -> gated RMSNorm -> FP8 quant -> out_proj FP8 GEMM
  GQA layer: input norm -> qkv FP8 GEMM -> q/k RMSNorm + partial RoPE + gate -> FP8 q quant -> FP8 KV append ->
             FlashAttention 3 (one `flash_attn_varlen_func` over the step's prefill and decode rows, then its
             split-KV combine) -> sigmoid gate + FP8 quant -> o_proj FP8 GEMM
  FFN: post-attention add + RMSNorm -> gate_up Marlin W4A16 GEMM -> SiLU*mul -> down Marlin W4A16 GEMM. Marlin
       takes the BF16 activation directly, so the three NVFP4 activation quants of the B200 graph do not exist.
       vLLM launches one Marlin GEMM as 1-3 kernels (it splits m); all of them belong to that GEMM.
  final add + RMSNorm; the sample phase runs the logits row select, the LM-head Marlin GEMM, then sampling.

Attention. vLLM's FLASH_ATTN backend makes one FA3 call per layer and step, and the simulator prices it as the
`prefill_attention` and `decode_attention` slots summed (`paged_attn_batch_vllm_fa3`: a mixed step is the two rows
summed; a step without prefill or decode rows prices that slot at zero). So one operation, `gqa.attention`,
owns both slots, and the FA3 main kernel and its combine are both that operation. FA3's
`prepare_varlen_num_blocks` runs in the preprocess phase and stays unmapped with the other input preparation.

Run from the repo root:
  uv run python presets/alignment/qwen38_27b_nvfp4_h100_tp1/label_rules/gen_rules.py
  cp presets/alignment/qwen38_27b_nvfp4_h100_tp1/label_rules/{gen_rules.py,rules.json,manifest.json} \
     presets/alignment/qwen38_27b_nvfp4_h200_tp1/label_rules/
"""
import json
from pathlib import Path

SRC = ("Hopper campaign captures 2026-11-03, cases 03/04/05/11: H100 80GB HBM3 (ptc ~/ss-hopper/campaign) and "
       "H200 (coriander /m-coriander/home/yilegu/ss-hopper/campaign), vLLM fork e1423a4d")
rules: list[dict] = []
bodies: dict[str, tuple] = {}


def op(operation, type_, role, slots):
    assert bodies.setdefault(operation, (type_, role, tuple(slots))) == (type_, role, tuple(slots)), operation

    def rule(name, **evidence):
        r = {"operation": operation, "name": name, "type": type_, "role": role,
             "slot_suffixes": list(slots), "phase": evidence.pop("phase", "forward")}
        r.update(evidence)
        r["source_experiment"] = SRC
        rules.append(r)
    return rule


FP8_GEMM = "cutlass_3x_gemm_sm90_fp8"
BF16_GEMM = "nvjet_sm90_tst_"
MARLIN = "marlin::Marlin<"
SPLITK_REDUCE = "cublasLt::splitKreduce_kernel"
NORM_FIRST = "triton_red_fused__to_copy_add_clamp_cutlass_scaled_mm_embedding_mul_reciprocal_rms_norm_"
NORM_FP8 = "triton_red_fused__to_copy_add_clamp_cutlass_scaled_mm_fused_add_rms_norm_mul_reciprocal_"
# The post-attention and final norms are the same Inductor kernel compiled into two graphs; their index suffix is
# the only difference and is the same in all eight captures.
NORM_POST = "triton_red_fused__to_copy_add_fused_add_rms_norm_1"
NORM_FINAL = "triton_red_fused__to_copy_add_fused_add_rms_norm_3"
SILU = "triton_poi_fused_mul_silu_slice_"
BF16_COPY = "at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda"
INDEX_COPY = "at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda"
CAT = "CatArrayBatchedCopy"
OUT_QUANT = "triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_view_"
GATE_QUANT = "triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_sigmoid_view_"
QK_ROPE = "_fused_qk_rmsnorm_rope_gate_kernel"
GDN_SM90 = "kernel_cutlass_kernel_flashinfergdn_kernelsdelta_rule_dsldelta_rule_"

# ---------------------------------------------------------------- norms
first_norm = op("model.first_input_norm", "model", "embedding gather + layer-0 input RMSNorm + FP8 row (Inductor)",
                ["gdn_first.input_norm"])
first_norm(NORM_FIRST)
gdn_norm = op("gdn.input_norm", "attention", "residual add + input RMSNorm + FP8 row of a GDN layer (Inductor)",
              ["gdn.input_norm"])
gdn_norm(NORM_FP8, before="gdn.qkvz")
gqa_norm = op("gqa.input_norm", "attention", "residual add + input RMSNorm + FP8 row of a GQA layer (Inductor)",
              ["gqa.input_norm"])
gqa_norm(NORM_FP8, before="gqa.qkv_proj")
op("ffn.post_attention_norm", "ffn", "post-attention add + RMSNorm (Inductor; BF16 out, Marlin reads it)",
   ["post_attention_norm"])(NORM_POST)
op("model.final_norm", "model", "final residual add + RMSNorm (Inductor)", ["final_norm"])(NORM_FINAL)

# ---------------------------------------------------------------- GDN
op("gdn.qkvz", "attention", "in_proj_qkvz static FP8 GEMM (Cutlass SM90)",
   ["gdn.qkvz"])(FP8_GEMM, before_name=BF16_GEMM)
ba = op("gdn.ba", "attention", "in_proj_ba BF16 GEMM (cuBLASLt nvjet sm90, with its split-K reduce)", ["gdn.ba"])
ba(BF16_GEMM, after="gdn.qkvz")
ba(SPLITK_REDUCE, after="gdn.ba")
op("gdn.core_output_zero", "attention", "core_attn_out torch.zeros (Inductor)",
   ["gdn.core_output_zero"])("triton_poi_fused_zeros_", after="gdn.ba")
ba_copy = op("gdn.ba_contiguous", "attention", "b.contiguous() and a.contiguous()", ["gdn.ba_contiguous"])
ba_copy(BF16_COPY, after="gdn.core_output_zero")
ba_copy(BF16_COPY, after="gdn.ba_contiguous", before_name="_causal_conv1d_")
op("gdn.decode.causal_conv", "attention", "causal_conv1d_update (decode-only step)",
   ["gdn.decode.causal_conv"])("_causal_conv1d_update_kernel")
op("gdn.decode.recurrent", "attention", "packed recurrent gated delta rule (decode-only step)",
   ["gdn.decode.recurrent"])("fused_recurrent_gated_delta_rule_packed_decode_kernel")
op("gdn.prefill.causal_conv", "attention", "varlen causal_conv1d_fwd over the step's tokens",
   ["gdn.prefill.causal_conv"])("_causal_conv1d_fwd_kernel")
op("gdn.prefill.post_conv", "attention", "post-conv split / gating of the prefill tokens",
   ["gdn.prefill.post_conv"])("_fused_post_conv_kernel")
dec_copy = op("gdn.decode_qkv_rearrange", "attention",
              "rearrange_mixed_qkv of a mixed step's decode rows: q / k / v copies and their cat",
              ["gdn.decode_qk_copy", "gdn.decode_v_copy", "gdn.decode_qkv_cat"])
dec_copy(BF16_COPY, after_name="_fused_post_conv_kernel")
dec_copy(BF16_COPY, after="gdn.decode_qkv_rearrange", before_name=BF16_COPY)
dec_copy(BF16_COPY, after="gdn.decode_qkv_rearrange", before_name=CAT)
dec_copy(CAT, before_name="fused_sigmoid_gating_delta_rule_update_kernel")
# The three identical copies sometimes fold into a repeat body of their own, with no neighbour in its segment. In
# all eight Hopper captures every such lone copy body repeats 3 times and is followed by the CAT that opens the next
# segment (130 bodies); the b / a copies and the core-output copy never fold alone.
dec_copy(BF16_COPY, after_name="<none>", before_name="<none>",
         note="lone folded copy body: always the decode q/k/v rearrange (x3, followed by the next segment's cat)")
op("gdn.mixed_decode_recurrent", "attention", "sigmoid-gating delta-rule update of a mixed step's decode rows",
   ["gdn.mixed_decode_recurrent"])("fused_sigmoid_gating_delta_rule_update_kernel")
op("gdn.prefill_index_copy", "attention", "int32 -> int64 copies of the prefill state indices / masks",
   ["gdn.prefill_index_copy"])(INDEX_COPY)
gather = op("gdn.state_gather", "attention", "ssm_state[prefill_state_indices] (index or gather)",
            ["gdn.state_gather"])
gather("index_kernel_impl<at::native::OpaqueType<(int)4>>", after="gdn.prefill_index_copy")
gather("vectorized_gather_kernel", after="gdn.prefill_index_copy")
zero = op("gdn.state_zero", "attention", "initial_state[~has_initial_state] = 0: the mask negation and masked_fill",
          ["gdn.state_zero"])
zero("bitwise_not_kernel_cuda", after="gdn.state_gather")
zero("masked_fill_kernel", after="gdn.state_zero")
op("gdn.g_exp", "attention", "g.exp() over the prefill tokens", ["gdn.g_exp"])("exp_kernel_cuda")
# FlashInfer picks the context-parallel kernel set or the fully fused kernel by shape; the profiled
# gdn_chunk_delta_rule row times the same dispatch, so every one of them is this operation.
op("gdn.prefill.chunk_delta_rule", "attention",
   "FlashInfer SM90 chunked gated delta rule (CP precompute / prefill / fixup kernels, or the fully fused kernel)",
   ["gdn.prefill.chunk_delta_rule"])(GDN_SM90)
op("gdn.state_scatter", "attention", "ssm_state[prefill_state_indices] = last_recurrent_state",
   ["gdn.state_scatter"])("index_put_kernel_impl")
op("gdn.output_cat", "attention", "cat of a mixed step's decode and prefill core outputs",
   ["gdn.output_cat"])(CAT, after="gdn.state_scatter")
op("gdn.core_output_copy", "attention", "core_attn_out[:num_actual_tokens] = core output",
   ["gdn.core_output_copy"])(BF16_COPY, before_name="layer_norm_fwd_kernel")
op("gdn.gated_norm", "attention", "gated RMSNorm of the core output", ["gdn.gated_norm"])("layer_norm_fwd_kernel")
op("gdn.out_quant", "attention", "static FP8 quant of the out_proj input (Inductor)", ["gdn.out_quant"])(OUT_QUANT)
op("gdn.out_proj", "attention", "out_proj static FP8 GEMM (Cutlass SM90)",
   ["gdn.out_proj"])(FP8_GEMM, after_name=OUT_QUANT)

# ---------------------------------------------------------------- GQA
op("gqa.qkv_proj", "attention", "q/k/v (+gate) static FP8 GEMM (Cutlass SM90)",
   ["gqa.qkv_proj"])(FP8_GEMM, before_name=QK_ROPE)
op("gqa.qk_norm_rope_gate", "attention", "q/k RMSNorm + partial RoPE + gate split",
   ["gqa.qk_norm_rope_gate"])(QK_ROPE)
op("gqa.query_quant", "attention", "static FP8 quant of q (Inductor)",
   ["gqa.query_quant"])("triton_poi_fused__to_copy_clamp_mul_reciprocal_", after_name=QK_ROPE)
op("gqa.kv_append", "attention", "FP8 paged KV append (reshape_and_cache_flash)",
   ["gqa.kv_append"])("vllm::reshape_and_cache_flash_kernel")
attention = op("gqa.attention", "attention",
               "FlashAttention 3 over the FP8 paged cache: one varlen call for the step's prefill and decode rows, "
               "then its split-KV combine",
               ["gqa.prefill_attention", "gqa.decode_attention"])
attention("flash::FlashAttnFwdSm90", after_name="vllm::reshape_and_cache_flash_kernel")
attention("flash::FlashAttnFwdCombine", after_name="flash::FlashAttnFwdSm90")
op("gqa.gate_quant", "attention", "sigmoid output gate + static FP8 quant (Inductor)", ["gqa.gate_quant"])(GATE_QUANT)
op("gqa.o_proj", "attention", "o_proj static FP8 GEMM (Cutlass SM90)",
   ["gqa.o_proj"])(FP8_GEMM, after_name=GATE_QUANT)

# ---------------------------------------------------------------- MLP
gate_up = op("ffn.gate_up", "ffn", "gate_up NVFP4 weight-only GEMM (Marlin W4A16; 1-3 launches)", ["mlp.gate_up"])
gate_up(MARLIN, after_name=NORM_POST)
gate_up(MARLIN, after="ffn.gate_up", after_name=MARLIN)
op("ffn.silu_and_mul", "ffn", "SiLU * up (Inductor)", ["mlp.silu_and_mul"])(SILU)
down = op("ffn.down", "ffn", "down NVFP4 weight-only GEMM (Marlin W4A16; 1-3 launches)", ["mlp.down"])
down(MARLIN, after_name=SILU)
down(MARLIN, after="ffn.down", after_name=MARLIN)

# ---------------------------------------------------------------- head (sample phase)
select = op("head.logits_select", "model", "hidden_states[logits_indices] row select", ["logits_select"])
select("index_elementwise_kernel", phase="sample", before_name=MARLIN)
select("vectorized_gather_kernel", phase="sample", before_name=MARLIN)
op("head.lm_head", "model", "LM head NVFP4 weight-only GEMM (Marlin W4A16)", ["lm_head"])(MARLIN, phase="sample")

out = Path(__file__).resolve().parent / "rules.json"
out.write_text(json.dumps({
    "schema_version": 2,
    "note": "Generated by gen_rules.py against the SM90 kernel path of the qwen35_vllm_nvfp4_gdn_dense CostTree.",
    "rules": rules,
}, indent=1) + "\n")
print(len(rules), "rules over", len(bodies), "operations")
