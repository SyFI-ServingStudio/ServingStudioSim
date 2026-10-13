"""Label rules for Qwen3.8-27B-NVFP4 on one B200 (vLLM fork 892da082). Writes rules.json beside this file.

DRAFT. The kernel side below is read from the diverse_100 capture at this pack's serving config (Slurm job
6443, wt-q38-capture logs/20261012_2_q38_diverse100/profile_nsys) and its folded inventory. The simulated side
does not exist yet: the clean arch's CostTree, and so its slot names, comes later. Each family therefore
carries `SLOTS[operation] = None` until its CostTree leaf is known, and only operations with slots are
emitted. Operation names are the draft's measured family names (shared brief, "Measured family shares").

One forward, as captured (FULL/PIECEWISE graph nodes, one track):
  embed + layer-0 RMSNorm (Inductor) ...
  GDN layer x3: input_fusion -> qkvz FP8 GEMM -> ba BF16 GEMM (+ splitK reduce) -> zeros/copies ->
                conv1d (update in decode, fwd + post_conv in prefill) -> recurrent packed decode
                (decode) | sigmoid-gating update + FlashInfer chunked GDN (prefill) -> copy ->
                gated RMSNorm (layer_norm_fwd_kernel) -> FP8 quant -> out_proj FP8 GEMM -> FFN
  GQA layer x1: input_fusion -> qkv FP8 GEMM -> qk RMSNorm + RoPE + gate -> FP8 quant -> KV append ->
                fill -> FlashInfer trtllm-gen FMHA (ForGen in decode, PersistentContext in prefill) ->
                sigmoid gate + FP8 quant -> o_proj FP8 GEMM -> FFN
  FFN: post-attn add+RMSNorm (+FP4 quant) -> FP4 quant -> gate_up NVFP4 GEMM -> SiLU*mul + FP4 quant ->
       FP4 quant -> down NVFP4 GEMM
  final add + RMSNorm; the LM head (FP4 quant + NVFP4 GEMM) runs in the `sample` phase.

Run from the repo root:
  uv run python presets/alignment/qwen38_27b_nvfp4_b200_tp1/label_rules/gen_rules.py
"""
import json
from pathlib import Path

SRC = "20261012_2_q38_diverse100/profile_nsys (Slurm job 6443)"

FP4_GEMM = "kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistentDenseGemmKernel"
FP8_GEMM = "nvjet_sm100_qqtst_"  # cuBLASLt FP8 tactics; the tile suffix moves with M
FP4_QUANT = "void vllm::cvt_fp16_to_fp4<__nv_bfloat16"
NORM_FP8 = "triton_red_fused__to_copy_add_clamp_fused_add_rms_norm_mul_reciprocal_view_3"
NORM_FP4 = "triton_red_fused__to_copy_add_fused_add_rms_norm_scaled_fp4_quant_view_1"

# (operation, type, role, kernel name, evidence). `name` matches by prefix as alignment.labeling does.
DRAFT = [
    ("gdn.input_fusion", "attention", "residual add + RMSNorm + FP8 quant of the GDN input (Inductor)",
     NORM_FP8, {"before_name": FP8_GEMM}),
    ("gdn.projection.qkvz", "attention", "in_proj_qkvz static FP8 GEMM", FP8_GEMM, {"after": "gdn.input_fusion"}),
    ("gdn.projection.ba", "attention", "in_proj_ba BF16 GEMM (nvjet/cutlass bf16, split-K)", "nvjet_sm100_tst_",
     {"after": "gdn.projection.qkvz"}),
    ("gdn.projection.ba", "attention", "in_proj_ba BF16 GEMM (Ampere-path cutlass tile)",
     "void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16", {"after": "gdn.projection.qkvz"}),
    ("gdn.projection.ba", "attention", "split-K reduce of in_proj_ba", "void cublasLt::splitKreduce_kernel",
     {"after": "gdn.projection.ba"}),
    ("gdn.conv.decode", "attention", "causal conv1d state update (decode)", "_causal_conv1d_update_kernel", {}),
    ("gdn.conv.prefill", "attention", "causal conv1d over the chunk (prefill)", "_causal_conv1d_fwd_kernel", {}),
    ("gdn.conv.prefill", "attention", "post-conv split/activation (prefill)", "_fused_post_conv_kernel", {}),
    ("gdn.recurrent.decode", "attention", "packed recurrent gated delta rule (decode)",
     "fused_recurrent_gated_delta_rule_packed_decode_kernel", {}),
    ("gdn.recurrent.prefill", "attention", "sigmoid gating + delta-rule state update (prefill)",
     "fused_sigmoid_gating_delta_rule_update_kernel", {}),
    ("gdn.chunk", "attention", "FlashInfer Blackwell chunked gated delta net (prefill)",
     "kernel_cutlass_kernel_flashinfergdn_kernelsblackwellgated_delta_net_chunkedGatedDeltaNetChunkedKernel", {}),
    ("gdn.gated_norm", "attention", "gated RMSNorm of the GDN output", "layer_norm_fwd_kernel", {}),
    ("gdn.out_quant", "attention", "FP8 quant of the out_proj input", "triton_poi_fused__to_copy_clamp_mul_reciprocal_view_0",
     {"after": "gdn.gated_norm"}),
    ("gdn.out_proj", "attention", "out_proj static FP8 GEMM", FP8_GEMM, {"after": "gdn.out_quant"}),
    ("gqa.input_fusion", "attention", "residual add + RMSNorm + FP8 quant of the GQA input (Inductor)",
     NORM_FP8, {"before_name": FP8_GEMM}),
    ("gqa.projection", "attention", "q/k/v static FP8 GEMM", FP8_GEMM, {"before_name": "_fused_qk_rmsnorm_rope_gate_kernel"}),
    ("gqa.qk_rope_gate", "attention", "q/k RMSNorm + partial RoPE + gate split", "_fused_qk_rmsnorm_rope_gate_kernel", {}),
    ("gqa.q_quant", "attention", "FP8 quant of q", "triton_poi_fused__to_copy_clamp_mul_reciprocal_4", {}),
    ("gqa.kv_append", "attention", "FP8 paged KV append", "void vllm::reshape_and_cache_flash_kernel", {}),
    ("gqa.reader.decode", "attention", "FlashInfer trtllm-gen FMHA, generation kernels (FP8 KV, head 256)",
     "fmhaSm100fKernel_QkvE4m3OBfloat16H256PagedKvCausal", {}),
    ("gqa.reader.context", "attention", "FlashInfer trtllm-gen FMHA, persistent context kernel (prefill)",
     "fmhaSm100aKernel_QkvE4m3OBfloat16H256PagedKvCausalP32VarSeqQ128Kv128PersistentContext", {}),
    ("gqa.output_gate", "attention", "sigmoid output gate + FP8 quant of the o_proj input",
     "triton_poi_fused__to_copy_clamp_mul_reciprocal_sigmoid_view_0", {}),
    ("gqa.o_proj", "attention", "o_proj static FP8 GEMM", FP8_GEMM, {"after": "gqa.output_gate"}),
    ("ffn.post_attn_norm", "ffn", "residual add + RMSNorm (+ FP4 quant) before the FFN", NORM_FP4, {}),
    ("ffn.gate_up_quant", "ffn", "NVFP4 quant of the gate_up input", FP4_QUANT, {"after": "ffn.post_attn_norm"}),
    ("ffn.gate_up", "ffn", "gate_up NVFP4 group-16 GEMM (FlashInfer CuTeDSL)", FP4_GEMM, {"after": "ffn.gate_up_quant"}),
    ("ffn.activation", "ffn", "SiLU * up + NVFP4 quant (Inductor)", "triton_poi_fused_mul_scaled_fp4_quant_silu_slice_2", {}),
    ("ffn.down_quant", "ffn", "NVFP4 quant of the down input", FP4_QUANT, {"after": "ffn.activation"}),
    ("ffn.down", "ffn", "down NVFP4 group-16 GEMM (FlashInfer CuTeDSL)", FP4_GEMM, {"after": "ffn.down_quant"}),
    ("prologue.embed_norm", "model", "embedding gather + layer-0 RMSNorm (Inductor)",
     "triton_red_fused__to_copy_add_clamp_embedding_mul_reciprocal_rms_norm_0", {}),
    ("epilogue.final_norm", "model", "final residual add + RMSNorm", "triton_red_fused__to_copy_add_fused_add_rms_norm_view_3", {}),
    ("head.quant", "model", "NVFP4 quant of the LM-head input", FP4_QUANT, {"phase": "sample"}),
    ("head.gemm", "model", "LM head NVFP4 GEMM", FP4_GEMM, {"phase": "sample"}),
]

# TODO(arch): CostTree slot suffixes per operation, from the clean arch's timing-predict cost manifest. Leave a
# helper (zeros/copy/index/cat/fill kernels, preprocess, sample bookkeeping) unmapped rather than attaching it here.
SLOTS: dict[str, list[str] | None] = {operation: None for operation, *_ in DRAFT}

rules = []
for operation, type_, role, name, evidence in DRAFT:
    if not SLOTS[operation]:
        continue
    rule = {"operation": operation, "name": name, "type": type_, "role": role,
            "slot_suffixes": SLOTS[operation], "phase": evidence.get("phase", "forward")}
    rule.update({k: v for k, v in evidence.items() if k != "phase"})
    rule["source_experiment"] = SRC
    rules.append(rule)

out = Path(__file__).resolve().parent / "rules.json"
out.write_text(json.dumps({
    "schema_version": 2,
    "note": "Generated by gen_rules.py. Empty until the clean arch's CostTree slots are filled into SLOTS.",
    "rules": rules,
}, indent=1) + "\n")
print(len(rules), "rules;", sum(v is None for v in SLOTS.values()), "operations awaiting slots")
