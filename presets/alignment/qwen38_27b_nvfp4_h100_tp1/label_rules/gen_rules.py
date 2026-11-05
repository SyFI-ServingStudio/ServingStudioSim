"""Label rules for Qwen3.8-27B-NVFP4 on one B200 (vLLM fork 892da082) against the qwen35_vllm_nvfp4_gdn_dense
CostTree. Writes rules.json beside this file.

The kernel side is read from the diverse_100 capture at this pack's serving config (Slurm job 6443, wt-q38-capture
logs/20261012_2_q38_diverse100/profile_nsys) and checked against the folded inventories of all 20 campaign cases
(chunk 2048/4096/8192). The slot side is the arch's cost manifest (`alignment label slots`).

One forward, as captured (FULL / PIECEWISE graph nodes, one track):
  layer 0 input RMSNorm with the embedding gather fused (Inductor)
  GDN layer: input norm (+FP8 row) -> in_proj_qkvz FP8 GEMM -> in_proj_ba BF16 GEMM (+ split-K reduce) ->
             core_attn_out zeros -> b / a contiguous copies -> core:
               decode-only: causal_conv1d_update -> packed recurrent decode
               with a prefill: varlen causal_conv1d_fwd -> post-conv; a mixed step then copies the decode rows'
               q / k / v, cats them and runs the sigmoid-gating recurrent update on them; the prefills' int64
               index copies (x3), state gather (index or gather), ~mask + masked_fill (zero fresh states),
               exp(g), FlashInfer chunked delta rule, state scatter (index_put); a mixed step cats the decode
               and prefill outputs
             -> core_attn_out copy -> gated RMSNorm -> FP8 quant -> out_proj FP8 GEMM
  GQA layer: input norm -> qkv FP8 GEMM -> q/k RMSNorm + partial RoPE + gate -> FP8 q quant -> FP8 KV append ->
             trtllm-gen FMHA (PersistentContext for prefill rows, ForGen for decode rows; each preceded by a
             byte Fill) -> sigmoid gate + FP8 quant -> o_proj FP8 GEMM
  FFN: post-attention add + RMSNorm -> NVFP4 quant -> gate_up NVFP4 GEMM -> SiLU*mul (+FP4) -> NVFP4 quant ->
       down NVFP4 GEMM
  final add + RMSNorm; the sample phase runs the logits row select, the LM-head NVFP4 quant + GEMM, then sampling.

The FP8 GEMMs (FlashInfer FP8 per-tensor) launch cuBLASLt nvjet, a CUTLASS GemmUniversal or a cuDNN matmul by
shape, so every FP8 GEMM operation names all three and is told apart by its neighbours.

Run from the repo root:
  uv run python presets/alignment/qwen38_27b_nvfp4_b200_tp1/label_rules/gen_rules.py
"""
import json
from pathlib import Path

SRC = "20261012_2_q38_diverse100/profile_nsys (Slurm job 6443) + 20261012_1_q38_align_campaign cases 01-20"
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


FP8_GEMMS = ("nvjet_sm100_qqtst_", "cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal",
             "cudnn_generated_fort_native_sm100_matMul")
BF16_GEMMS = ("nvjet_sm100_tst_", "cutlass_80_tensorop_s16816gemm_bf16")
FP4_GEMM = "kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistentDenseGemmKernel"
FP4_QUANT = "vllm::cvt_fp16_to_fp4<"
SPLITK_REDUCE = "cublasLt::splitKreduce_kernel"
NORM_FIRST = "triton_red_fused__to_copy_add_clamp_embedding_mul_reciprocal_rms_norm_"
NORM_FP8 = "triton_red_fused__to_copy_add_clamp_fused_add_rms_norm_mul_reciprocal_view_"
NORM_POST = "triton_red_fused__to_copy_add_fused_add_rms_norm_scaled_fp4_quant_view_"
NORM_FINAL = "triton_red_fused__to_copy_add_fused_add_rms_norm_view_"
BF16_COPY = "at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda"
INDEX_COPY = "at::native::unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda"
CAT = "CatArrayBatchedCopy"
OUT_QUANT = "triton_poi_fused__to_copy_clamp_mul_reciprocal_view_"
GATE_QUANT = "triton_poi_fused__to_copy_clamp_mul_reciprocal_sigmoid_view_"
QK_ROPE = "_fused_qk_rmsnorm_rope_gate_kernel"

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
op("ffn.post_attention_norm", "ffn", "post-attention add + RMSNorm (Inductor)",
   ["post_attention_norm"])(NORM_POST)
op("model.final_norm", "model", "final residual add + RMSNorm (Inductor)", ["final_norm"])(NORM_FINAL)

# ---------------------------------------------------------------- GDN
qkvz = op("gdn.qkvz", "attention", "in_proj_qkvz static FP8 GEMM", ["gdn.qkvz"])
for gemm in FP8_GEMMS:
    for ba in BF16_GEMMS:
        qkvz(gemm, before_name=ba)
ba = op("gdn.ba", "attention", "in_proj_ba BF16 GEMM (cuBLASLt nvjet / CUTLASS sm80, with its split-K reduce)",
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
# The three identical copies sometimes fold into a repeat body of their own, with no neighbour in its segment. In
# all 21 captures every such lone copy body is this one (14946 positions = 3 x the 4982 cats that open the next
# segment); the b / a copies and the core-output copy never fold alone.
dec_copy(BF16_COPY, after_name="<none>", before_name="<none>",
         note="lone folded copy body: always the decode q/k/v rearrange (3 x the next segment's cat)")
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
op("gdn.prefill.chunk_delta_rule", "attention", "FlashInfer Blackwell chunked gated delta rule",
   ["gdn.prefill.chunk_delta_rule"])("flashinfergdn_kernelsblackwellgated_delta_net_chunkedGatedDeltaNetChunkedKernel")
op("gdn.state_scatter", "attention", "ssm_state[prefill_state_indices] = last_recurrent_state",
   ["gdn.state_scatter"])("index_put_kernel_impl")
op("gdn.output_cat", "attention", "cat of a mixed step's decode and prefill core outputs",
   ["gdn.output_cat"])(CAT, after="gdn.state_scatter")
op("gdn.core_output_copy", "attention", "core_attn_out[:num_actual_tokens] = core output",
   ["gdn.core_output_copy"])(BF16_COPY, before_name="layer_norm_fwd_kernel")
op("gdn.gated_norm", "attention", "gated RMSNorm of the core output", ["gdn.gated_norm"])("layer_norm_fwd_kernel")
op("gdn.out_quant", "attention", "static FP8 quant of the out_proj input (Inductor)", ["gdn.out_quant"])(OUT_QUANT)
out_proj = op("gdn.out_proj", "attention", "out_proj static FP8 GEMM", ["gdn.out_proj"])
for gemm in FP8_GEMMS:
    out_proj(gemm, after_name=OUT_QUANT)

# ---------------------------------------------------------------- GQA
qkv = op("gqa.qkv_proj", "attention", "q/k/v (+gate) static FP8 GEMM", ["gqa.qkv_proj"])
for gemm in FP8_GEMMS:
    qkv(gemm, before_name=QK_ROPE)
op("gqa.qk_norm_rope_gate", "attention", "q/k RMSNorm + partial RoPE + gate split",
   ["gqa.qk_norm_rope_gate"])(QK_ROPE)
op("gqa.query_quant", "attention", "static FP8 quant of q (Inductor)",
   ["gqa.query_quant"])("triton_poi_fused__to_copy_clamp_mul_reciprocal_", after_name=QK_ROPE)
op("gqa.kv_append", "attention", "FP8 paged KV append (reshape_and_cache_flash)",
   ["gqa.kv_append"])("vllm::reshape_and_cache_flash_kernel")
op("gqa.prefill_attention", "attention", "trtllm-gen FMHA persistent context kernel (prefill rows)",
   ["gqa.prefill_attention"])("fmhaSm100aKernel_QkvE4m3OBfloat16H256PagedKvCausalP32VarSeqQ128Kv128PersistentContext")
op("gqa.decode_attention", "attention", "trtllm-gen FMHA generation kernels (decode rows)",
   ["gqa.decode_attention"])("fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausal")
op("gqa.gate_quant", "attention", "sigmoid output gate + static FP8 quant (Inductor)", ["gqa.gate_quant"])(GATE_QUANT)
o_proj = op("gqa.o_proj", "attention", "o_proj static FP8 GEMM", ["gqa.o_proj"])
for gemm in FP8_GEMMS:
    o_proj(gemm, after_name=GATE_QUANT)

# ---------------------------------------------------------------- MLP
op("ffn.gate_up_input_quant", "ffn", "NVFP4 quant of the gate_up input",
   ["mlp.gate_up_input_quant"])(FP4_QUANT, after="ffn.post_attention_norm")
op("ffn.gate_up", "ffn", "gate_up NVFP4 group-16 GEMM (FlashInfer CuTeDSL)",
   ["mlp.gate_up"])(FP4_GEMM, after="ffn.gate_up_input_quant")
op("ffn.silu_and_mul", "ffn", "SiLU * up (+ FP4 scale, Inductor)",
   ["mlp.silu_and_mul"])("triton_poi_fused_mul_scaled_fp4_quant_silu_slice_")
op("ffn.down_input_quant", "ffn", "NVFP4 quant of the down input",
   ["mlp.down_input_quant"])(FP4_QUANT, after="ffn.silu_and_mul")
op("ffn.down", "ffn", "down NVFP4 group-16 GEMM (FlashInfer CuTeDSL)",
   ["mlp.down"])(FP4_GEMM, after="ffn.down_input_quant")

# ---------------------------------------------------------------- head (sample phase)
select = op("head.logits_select", "model", "hidden_states[logits_indices] row select", ["logits_select"])
select("index_elementwise_kernel", phase="sample", before_name=FP4_QUANT)
select("vectorized_gather_kernel", phase="sample", before_name=FP4_QUANT)
op("head.lm_head_input_quant", "model", "NVFP4 quant of the LM-head input",
   ["lm_head_input_quant"])(FP4_QUANT, phase="sample")
op("head.lm_head", "model", "LM head NVFP4 GEMM (FlashInfer CuTeDSL)", ["lm_head"])(FP4_GEMM, phase="sample")

out = Path(__file__).resolve().parent / "rules.json"
out.write_text(json.dumps({
    "schema_version": 2,
    "note": "Generated by gen_rules.py against the qwen35_vllm_nvfp4_gdn_dense CostTree.",
    "rules": rules,
}, indent=1) + "\n")
print(len(rules), "rules over", len(bodies), "operations")
