"""Label rules for Qwen3.8-27B-NVFP4 on one L40S (the vLLM Recipes deployment, pinned alignment fork) against the
qwen35_vllm_nvfp4_gdn_dense CostTree on its SM89 kernel path. Writes rules.json beside this file.

The kernel side is read from this pack's case 07 NSYS capture (logs/20261023_1_q38_l40s_campaign/
07_quadrant_interference_c8/profile_nsys) and checked against every case's inventory. The slot side is the arch's
cost manifest (`alignment label slots`). The inventories are initialized with `alignment label initialize
--unfold` (literal-v1): at large m Marlin issues several launches per GEMM, and a folded repeat body of one Marlin
launch has no neighbour, so gate_up and down would be indistinguishable; unfolded, each launch's nearest mapped
predecessor (the post-attention norm / gate_up, or SiLU-mul / down) names it.

One forward, as captured (the B200 sequence with SM89 kernels; see the B200 pack's gen_rules.py):
  layer 0 input RMSNorm with the embedding gather fused (Inductor, with the FP8 row)
  GDN layer: input norm (+FP8 row) -> in_proj_qkvz CUTLASS sm89 FP8 GEMM -> in_proj_ba BF16 GEMM (cutlass_80 wmma +
             split-K reduce at decode sizes, ampere s16816 at prefill sizes) -> core_attn_out zeros -> b / a copies ->
             core as on B200, except the chunked delta rule is FLA Triton (chunk_local_cumsum, scaled_dot_kkt, a BF16
             fill, merge_16x16_to_64x64_inverse, recompute_w_u, chunk_fwd_h, chunk_fwd_o) and there is no exp(g)
             -> core_attn_out copy -> gated RMSNorm -> FP8 quant -> out_proj FP8 GEMM
  GQA layer: input norm -> qkv FP8 GEMM -> q/k RMSNorm + partial RoPE + gate -> FP8 KV append (no query quant) ->
             FlashInfer FA2 BatchPrefillWithPagedKVCache (causal, prefill rows) then the same kernel non-causal +
             PersistentVariableLengthMergeStates (decode rows) -> sigmoid gate + FP8 quant -> o_proj FP8 GEMM
  FFN: post-attention add + RMSNorm (BF16 row) -> gate_up Marlin W4A16 GEMM (1..n launches) -> SiLU*mul (Inductor)
       -> down Marlin GEMM (1..n launches)
  final add + RMSNorm; the sample phase runs the logits row select and the LM-head Marlin GEMM, then sampling.

Run from the repo root:
  uv run python presets/alignment/qwen38_27b_nvfp4_l40s_tp1/label_rules/gen_rules.py
"""
import json
from pathlib import Path

SRC = "20261023_1_q38_l40s_campaign case 07 profile_nsys (and every case's inventory)"
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


FP8_GEMM = "enable_sm89_to_sm90<cutlass::gemm::kernel::DefaultGemmWithVisitor<cutlass::float_e4m3_t"
BF16_GEMMS = ("cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16", "ampere_bf16_s16816gemm_bf16",
              "cutlass_80_tensorop_bf16_s16816gemm_relu_bf16", "cutlass_80_tensorop_s16816gemm_bf16",
              # cuBLAS's BF16 GEMV at m = 1 (single-request decode).
              "internal::gemvx::kernel<int, int, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16")
MARLIN = "marlin::Marlin<"
SPLITK_REDUCE = "cublasLt::splitKreduce_kernel"
NORM_FIRST = "triton_red_fused__to_copy_add_clamp_cutlass_scaled_mm_embedding_mul_reciprocal_rms_norm_"
NORM_FP8 = "triton_red_fused__to_copy_add_clamp_cutlass_scaled_mm_fused_add_rms_norm_mul_reciprocal_"
NORM_BF16 = "triton_red_fused__to_copy_add_fused_add_rms_norm_"
BF16_COPY = "at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda"
INDEX_COPY = "at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda"
CAT = "CatArrayBatchedCopy"
OUT_QUANT = "triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_view_"
GATE_QUANT = "triton_poi_fused__to_copy_clamp_cutlass_scaled_mm_mul_reciprocal_sigmoid_view_"
QK_ROPE = "_fused_qk_rmsnorm_rope_gate_kernel"
FA2 = "flashinfer::BatchPrefillWithPagedKVCacheKernel<(bool)1, flashinfer::KernelTraits<(flashinfer::MaskMode)"
MERGE = "flashinfer::PersistentVariableLengthMergeStatesKernel"
FLA_CHUNK = ("chunk_local_cumsum_scalar_kernel", "chunk_scaled_dot_kkt_fwd_kernel", "merge_16x16_to_64x64_inverse_kernel",
             "recompute_w_u_fwd_kernel", "chunk_gated_delta_rule_fwd_kernel_h_blockdim64", "chunk_fwd_kernel_o")
BF16_FILL = "FillFunctor<c10::BFloat16>"

# ---------------------------------------------------------------- norms
op("model.first_input_norm", "model", "embedding gather + layer-0 input RMSNorm + FP8 row (Inductor)",
   ["gdn_first.input_norm"])(NORM_FIRST)
op("gdn.input_norm", "attention", "residual add + input RMSNorm + FP8 row of a GDN layer (Inductor)",
   ["gdn.input_norm"])(NORM_FP8, before="gdn.qkvz")
op("gqa.input_norm", "attention", "residual add + input RMSNorm + FP8 row of a GQA layer (Inductor)",
   ["gqa.input_norm"])(NORM_FP8, before="gqa.qkv_proj")
# The post-attention and final norms are the same Inductor template (BF16 row only); the final one is the forward's
# last kernel and the post-attention one feeds the gate_up Marlin GEMM.
op("ffn.post_attention_norm", "ffn", "post-attention add + RMSNorm, BF16 row (Inductor)",
   ["post_attention_norm"])(NORM_BF16, before_name=MARLIN)
op("model.final_norm", "model", "final residual add + RMSNorm (Inductor)", ["final_norm"])(NORM_BF16, before_name="<none>")

# ---------------------------------------------------------------- GDN
qkvz = op("gdn.qkvz", "attention", "in_proj_qkvz static FP8 GEMM (CUTLASS sm89)", ["gdn.qkvz"])
for ba_gemm in BF16_GEMMS:
    qkvz(FP8_GEMM, before_name=ba_gemm)
ba = op("gdn.ba", "attention", "in_proj_ba BF16 GEMM (cuBLAS cutlass_80 / ampere, with its split-K reduce)",
        ["gdn.ba"])
for gemm in BF16_GEMMS:
    ba(gemm, after="gdn.qkvz")
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
chunk = op("gdn.prefill.chunk_delta_rule", "attention",
           "FLA Triton chunked gated delta rule (cumsum, KKT, solve_tril with its BF16 zero-fill, w/u, h, o)",
           ["gdn.prefill.chunk_delta_rule"])
for name in FLA_CHUNK:
    chunk(name)
chunk(BF16_FILL, after_name="chunk_scaled_dot_kkt_fwd_kernel")
op("gdn.state_scatter", "attention", "ssm_state[prefill_state_indices] = last_recurrent_state",
   ["gdn.state_scatter"])("index_put_kernel_impl")
op("gdn.output_cat", "attention", "cat of a mixed step's decode and prefill core outputs",
   ["gdn.output_cat"])(CAT, after="gdn.state_scatter")
op("gdn.core_output_copy", "attention", "core_attn_out[:num_actual_tokens] = core output",
   ["gdn.core_output_copy"])(BF16_COPY, before_name="layer_norm_fwd_kernel")
op("gdn.gated_norm", "attention", "gated RMSNorm of the core output", ["gdn.gated_norm"])("layer_norm_fwd_kernel")
op("gdn.out_quant", "attention", "static FP8 quant of the out_proj input (Inductor)", ["gdn.out_quant"])(OUT_QUANT)
op("gdn.out_proj", "attention", "out_proj static FP8 GEMM (CUTLASS sm89)",
   ["gdn.out_proj"])(FP8_GEMM, after_name=OUT_QUANT)

# ---------------------------------------------------------------- GQA
op("gqa.qkv_proj", "attention", "q/k/v (+gate) static FP8 GEMM (CUTLASS sm89)",
   ["gqa.qkv_proj"])(FP8_GEMM, before_name=QK_ROPE)
op("gqa.qk_norm_rope_gate", "attention", "q/k RMSNorm + partial RoPE + gate split",
   ["gqa.qk_norm_rope_gate"])(QK_ROPE)
op("gqa.kv_append", "attention", "FP8 paged KV append (reshape_and_cache_flash)",
   ["gqa.kv_append"])("vllm::reshape_and_cache_flash_kernel")
pre = op("gqa.prefill_attention", "attention",
         "FlashInfer FA2 causal paged prefill (prefill rows), with its split-KV merge when the plan splits",
         ["gqa.prefill_attention"])
pre(FA2 + "1")
pre(MERGE, after="gqa.prefill_attention")
dec = op("gqa.decode_attention", "attention",
         "FlashInfer FA2 paged attention of the decode rows (tensor-core prefill kernel) + split-KV merge",
         ["gqa.decode_attention"])
dec(FA2 + "0")
dec(MERGE, after="gqa.decode_attention")
op("gqa.gate_quant", "attention", "sigmoid output gate + static FP8 quant (Inductor)", ["gqa.gate_quant"])(GATE_QUANT)
op("gqa.o_proj", "attention", "o_proj static FP8 GEMM (CUTLASS sm89)",
   ["gqa.o_proj"])(FP8_GEMM, after_name=GATE_QUANT)

# ---------------------------------------------------------------- MLP
gate_up = op("ffn.gate_up", "ffn", "gate_up NVFP4 W4A16 GEMM (Marlin, every launch)", ["mlp.gate_up"])
gate_up(MARLIN, after="ffn.post_attention_norm")
gate_up(MARLIN, after="ffn.gate_up")
op("ffn.silu_and_mul", "ffn", "SiLU * up (Inductor)", ["mlp.silu_and_mul"])("triton_poi_fused_mul_silu_slice_")
down = op("ffn.down", "ffn", "down NVFP4 W4A16 GEMM (Marlin, every launch)", ["mlp.down"])
down(MARLIN, after="ffn.silu_and_mul")
down(MARLIN, after="ffn.down")

# ---------------------------------------------------------------- head (sample phase)
select = op("head.logits_select", "model", "hidden_states[logits_indices] row select", ["logits_select"])
select("index_elementwise_kernel", phase="sample", before_name=MARLIN)
select("vectorized_gather_kernel", phase="sample", before_name=MARLIN)
op("head.lm_head", "model", "LM head NVFP4 W4A16 GEMM (Marlin)", ["lm_head"])(MARLIN, phase="sample")

out = Path(__file__).resolve().parent / "rules.json"
out.write_text(json.dumps({
    "schema_version": 2,
    "note": "Generated by gen_rules.py against the qwen35_vllm_nvfp4_gdn_dense CostTree (SM89 kernel path).",
    "rules": rules,
}, indent=1) + "\n")
print(len(rules), "rules over", len(bodies), "operations")
