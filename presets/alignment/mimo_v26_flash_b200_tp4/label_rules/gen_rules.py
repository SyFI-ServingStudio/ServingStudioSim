"""Label rules for the MiMo-V2.6-Flash fused all-reduce capture (job 6264)
against the mimo_v2_vllm_fp8_swa_moe CostTree. Writes rules.json beside this file.

Taken from the arch's kernel-align run (wt-mimo-v2-clean
logs/20261012_2_mimo_arch_kernel_align/make_rules.py), which labeled the --unfold
inventory, plus the two body-start rules marked below so the folded inventory that
`alignment-campaign label` initializes is covered as fully as the unfolded one. The
sample-phase logits all-gather is mapped to the arch's
`logits_all_gather` slot, so no separate unmapped rule is needed.

Run from the repo root:
  uv run python presets/alignment/mimo_v26_flash_b200_tp4/label_rules/gen_rules.py
"""
import json
from pathlib import Path

SRC = "20261012_1_mimo_fused_ar_nsys/profile_nsys (Slurm job 6264)"
rules = []


def op(operation, type_, role, slots, *, sync=False):
    def rule(name, **evidence):
        r = {"operation": operation, "name": name, "type": type_, "role": role,
             "slot_suffixes": slots, "phase": evidence.pop("phase", "forward")}
        r.update(evidence)
        if sync:
            r["cross_rank"] = "synchronizing"
        r["source_experiment"] = SRC
        rules.append(r)
    return rule


AR_ONE = "oneshotAllreduceFusionKernel"
AR_TWO = "twoshotAllreduceKernel"
AR_NORM = "rmsNormLamport"


# Above vLLM's fused cap (4096 tokens at TP4; chunk-8192 captures, cases 17/19) the
# boundary runs unfused: a pynccl all-reduce, then Inductor's residual add + RMSNorm
# (some variants also emit a packed FP8 quant, but the separate quant kernel still
# follows). The embedding boundary's norm has no residual add.
AR_UNFUSED = "ncclDevKernel_AllReduce_Sum"
NORM_UNFUSED = "rms_norm"


def ar_norm(operation, slot, role, unfused_first=None, **first):
    rule = op(operation, "collective", role, [slot], sync=True)
    rule(AR_ONE, **first)
    rule(AR_TWO, **first)
    rule(AR_NORM, after=operation)
    op(f"{operation}_all_reduce", "collective", "unfused all-reduce above the fused cap (NCCL)",
       [f"{slot}_all_reduce"], sync=True)(AR_UNFUSED, **(unfused_first or first))
    op(f"{operation}_residual_norm", "model",
       "unfused residual add + RMSNorm above the fused cap (Inductor Triton)",
       [f"{slot}_residual_norm"])(NORM_UNFUSED, after=f"{operation}_all_reduce")


def after_ar_norm(rule, name, *boundaries):
    """`name` after a boundary, fused or unfused."""
    for boundary in boundaries:
        rule(name, after=boundary)
        rule(name, after=f"{boundary}_residual_norm")


embed = op("prologue.embed", "model", "vocab-parallel embedding gather", ["embed"])
embed("vocab_parallel_embedding_kernel")
ar_norm("prologue.embed_ar_norm", "embed.ar_norm",
        "embedding all-reduce + layer-0 input RMSNorm (FlashInfer MNNVL)",
        unfused_first={"after": "prologue.embed"}, after_name="triton_poi_fused_zeros_like_0")

quant_role = "FP8 per-token-group activation quant of the QKV input"
qkv_quant = op("attn.qkv_input_quant", "attention", quant_role, ["attn.qkv_proj.input_quant"])
after_ar_norm(qkv_quant, "per_token_group_quant_8bit_packed_register_kernel",
              "prologue.embed_ar_norm", "ffn.ar_norm", "moe.ar_norm")
op("attn.qkv_proj", "attention", "fused QKV projection (DeepGEMM FP8 block)",
   ["attn.qkv_proj.gemm"])("sm100_fp8_fp4_gemm_1d1d_impl", after="attn.qkv_input_quant")
rope = op("attn.rope", "attention", "partial rotary + v scale (Inductor Triton)", ["attn.rope"])
rope("triton_poi_fused_0", after="attn.qkv_proj")
rope("triton_poi_fused_1", after="attn.qkv_proj")
rope("triton_poi_fused_2", after="attn.qkv_proj")  # chunk 8192
rope("triton_poi_fused_3", after="attn.qkv_proj")
op("attn.kv_append", "attention", "DiffKV paged cache write",
   ["attn.kv_append"])("reshape_and_cache_kernel_flash_diffkv")
attn = op("attn.attention", "attention",
          "FA4 varlen attention: prepare_scheduler, mainloop, split-KV combine", ["attn.attention"])
attn("flash_attncuteprepare_scheduler")
attn("flash_attncuteflash_fwd_sm100")
attn("flash_attncuteflash_fwd_combine")
op("attn.o_proj", "attention", "BF16 output projection (cuBLAS)",
   ["attn.o_proj"])("nvjet_sm100_tst", after="attn.attention")
ar_norm("attn.ar_norm", "attn.ar_norm",
        "attention all-reduce + residual + RMSNorm (FlashInfer MNNVL)", after="attn.o_proj")

router = op("moe.router", "moe", "router GEMM with FP32 output (cuBLAS)", ["moe.router"])
after_ar_norm(router, "nvjet_sm100_tss", "attn.ar_norm")
router("splitKreduce_kernel", after="moe.router")
routed = op("moe.routed", "moe",
            "FlashInfer TRT-LLM MXFP4 x BF16 routed MoE: routing, FC1, FC2, finalize", ["fused_moe"])
routed("moe::dev::routing")
routed("bmm_Bfloat16_MxE2m1Bfloat16")
routed("moe::dev::finalize")
ar_norm("moe.ar_norm", "moe.ar_norm",
        "MoE all-reduce + residual + RMSNorm (FlashInfer MNNVL)", after="moe.routed")
# Added for the pack: `alignment-campaign label` labels the folded inventory, where a
# layer's repeat body can open on the previous layer's MoE all-reduce (capture 6264's
# prefill/mixed sequences: finalize ends one body, the AR opens the next). The opening
# kernel has no predecessor, so `after="moe.routed"` cannot fire there and the whole
# QKV chain behind it stays unmapped. Only that AR is ever a body's first forward
# kernel in the capture, so the empty-predecessor sentinel names it.
moe_ar_at_body_start = op("moe.ar_norm", "collective",
                          "MoE all-reduce + residual + RMSNorm (FlashInfer MNNVL)", ["moe.ar_norm"], sync=True)
moe_ar_at_body_start(AR_ONE, after_name="<none>")
moe_ar_at_body_start(AR_TWO, after_name="<none>")
op("moe.ar_norm_all_reduce", "collective", "unfused all-reduce above the fused cap (NCCL)",
   ["moe.ar_norm_all_reduce"], sync=True)(AR_UNFUSED, after_name="<none>")

gate_up_quant = op("ffn.gate_up_input_quant", "ffn", "FP8 quant of the dense FFN input",
                   ["ffn.gate_up_input_quant"])
after_ar_norm(gate_up_quant, "per_token_group_quant_8bit_packed_register_kernel", "attn.ar_norm")
op("ffn.gate_up", "ffn", "dense gate_up GEMM (DeepGEMM FP8 block)",
   ["ffn.gate_up"])("sm100_fp8_fp4_gemm_1d1d_impl", after="ffn.gate_up_input_quant")
op("ffn.act_and_mul", "ffn", "dense SiLU-and-mul (Inductor Triton)",
   ["ffn.act_and_mul"])("triton_poi_fused_mul_per_token_group_fp8_quant_packed_silu_slice")
op("ffn.down_input_quant", "ffn", "FP8 quant of the dense down input",
   ["ffn.down_input_quant"])("per_token_group_quant_8bit_packed_register_kernel", after="ffn.act_and_mul")
op("ffn.down", "ffn", "dense down GEMM (DeepGEMM FP8 block)",
   ["ffn.down"])("sm100_fp8_fp4_gemm_1d1d_impl", after="ffn.down_input_quant")
ar_norm("ffn.ar_norm", "ffn.ar_norm",
        "dense FFN all-reduce + residual + RMSNorm (FlashInfer MNNVL)", after="ffn.down")

select = op("epilogue.logits_select", "model", "gather the rows that need logits", ["logits_select"])
select("index_elementwise_kernel", phase="sample", before_name="nvjet_sm100_tst")
select("vectorized_gather_kernel", phase="sample", before_name="nvjet_sm100_tst")
lm_head = op("epilogue.lm_head", "model", "vocab-parallel LM head GEMM (cuBLAS)", ["lm_head"])
lm_head("nvjet_sm100_tst", phase="sample")
lm_head("splitKreduce_kernel", phase="sample", after="epilogue.lm_head")
op("epilogue.logits_all_gather", "collective", "vocab-parallel logits all-gather",
   ["logits_all_gather"], sync=True)("ncclDevKernel_AllGather", phase="sample")

Path(__file__).with_name("rules.json").write_text(
    json.dumps({"schema_version": 1, "note": "MiMo-V2.6-Flash-RL vLLM TP4/EP4 B200, capture job 6264, against "
                "mimo_v2_vllm_fp8_swa_moe. Residual zeros_like, preprocess and sampling stay unmapped.",
                "rules": rules}, indent=1) + "\n")
print(len(rules), "rules")
