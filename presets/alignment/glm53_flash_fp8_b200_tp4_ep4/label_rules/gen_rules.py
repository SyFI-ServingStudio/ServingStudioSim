"""Generate the GLM-5.3-Flash FP8 TP4/EP4 label rules (one order-free set).

Operations are named after the simulated slot suffix they own; one slot suffix
covers every layer tag (first_kda_dense, kda_dense, dsa_moe, kda_moe) and every
EP rank's routed leaf, so a measured kernel is compared with its slot summed
over all layers of the iteration.
"""
import json, sys

SRC = "logs/20260924_0_glm53_flash_b200_tp4_ep4_nsys"
rules = []

def R(operation, name, role, slots, typ="model", **ev):
    r = {"operation": operation, "name": name, "type": typ, "role": role,
         "slot_suffixes": list(slots), "source_experiment": SRC}
    if typ == "collective":
        r["cross_rank"] = "synchronizing"
    ev.setdefault("phase", "forward")
    r.update({k: v for k, v in ev.items() if v is not None})
    rules.append(r)

ROLE = {}
def op(operation, role, slots, typ="model"):
    ROLE[operation] = (role, slots, typ)
def M(operation, name, **ev):
    role, slots, typ = ROLE[operation]
    R(operation, name, role, slots, typ, **ev)

NVJ = "nvjet_sm100_tst_"
TSS = "nvjet_sm100_tss_"
SKR = "cublasLt::splitKreduce_kernel"
EW = "at::native::elementwise_kernel"
VEW = "at::native::vectorized_elementwise_kernel"
UEW = "at::native::unrolled_elementwise_kernel"
COPY = "unrolled_elementwise_kernel<at::native::direct_copy_kernel_cuda"
BIGFUSE = "mhc_pre_big_fuse_with_norm_tilelang_kernel"
PACKQ = "per_token_group_quant_8bit_packed_register_kernel"
ROUTEQ = "per_token_group_quant_8bit_kernel<"
DG = "deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl"
ACT = "vllm::act_and_mul_kernel"

TOP = "cutlass_80_tensorop"
WMMA = "cutlass_80_wmma_tensorop"
GEMMS = (NVJ, TOP, WMMA)

# ---------------- prologue / epilogue / boundaries ----------------
op("embedding", "vocab-parallel embedding lookup", ["embedding"])
op("hc_expand", "hyper-connection expand copy to [T,4,H]", ["hc_expand"])
op("tp.all_reduce", "TP all-reduce (flashinfer trtllm MNNVL one-shot / two-shot) after embedding, attention and FFN",
   ["embedding_all_reduce", "attn_all_reduce", "ffn_all_reduce"], "collective")
op("mhc.boundary", "mHC boundary: mhc_post, tf32 hc_prenorm GEMM and fused pre/RMSNorm between sublayers",
   ["attn_mhc_pre", "attn_mhc_post_pre", "ffn_mhc_post_pre", "final_mhc_post"])
op("hc_contract_mean", "hyper-connection contract x.mean(dim=1)", ["hc_contract_mean"])
op("final_norm", "final RMSNorm", ["final_norm"])
op("lm_head", "tensor-parallel language-model head projection", ["lm_head"])

M("embedding", "vocab_parallel_embedding_kernel", phase="forward")
for n in ("twoshotAllreduceKernel", "oneshotAllreduceFusionKernel"):
    M("tp.all_reduce", n, phase="forward")
M("hc_expand", EW, phase="forward", after="tp.all_reduce", before_name="hc_prenorm_gemm")
for n in ("mhc_post_tilelang_kernel", "hc_prenorm_gemm", BIGFUSE, "mhc_fused_tilelang_kernel"):
    M("mhc.boundary", n, phase="forward")
M("hc_contract_mean", "at::native::reduce_kernel", phase="forward", after="mhc.boundary")
M("final_norm", "vllm::rms_norm_kernel", phase="forward")
M("lm_head", NVJ, phase="sample")
M("lm_head", SKR, phase="sample", after="lm_head")
rules.append({"operation": "sample.logits.tp_allgather", "name": "ncclDevKernel_AllGather_RING_LL",
              "status": "unmapped", "cross_rank": "synchronizing", "phase": "sample",
              "note": "vocab-parallel logits all-gather before sampling; the unified CostTree has no sampling leaf",
              "source_experiment": SRC})

# ---------------- KDA ----------------
K = "kda."
for s, role in [("in_proj", "KDA in_proj_qkvbfg_a BF16 projection"),
                ("f_b_proj", "KDA f_b low-rank gate up projection"),
                ("g_b_proj", "KDA g_b low-rank output-gate projection"),
                ("short_conv_prefill", "KDA merged q/k/v causal conv (varlen prefill)"),
                ("short_conv_decode", "KDA merged q/k/v causal conv update (decode)"),
                ("decode_glue", "KDA decode split/copy launches between conv and recurrent core"),
                ("prefill_qkv_copy", "KDA prefill q/k/v contiguous copies around the qk l2norm"),
                ("prefill_glue", "KDA prefill small launches: beta sigmoid, index scans, gate views"),
                ("state_gather", "KDA prefill SSM state gather"),
                ("chunk_prefill", "KDA chunk prefill (chunk_kda_with_fused_gate: l2norm, gate cumsum, kkt, inverse, w/u, h, o)"),
                ("recurrent_decode", "KDA fused recurrent decode"),
                ("state_scatter", "KDA prefill SSM state scatter"),
                ("gated_norm", "KDA sigmoid-gated RMSNorm"),
                ("o_proj", "KDA output projection")]:
    op(K + s, role, [K + s])

for nxt in GEMMS:
    M(K + "in_proj", NVJ, after_name=BIGFUSE, before_name=nxt)
for g in GEMMS:
    M(K + "f_b_proj", g, after=K + "in_proj")
    M(K + "g_b_proj", g, after=K + "f_b_proj")
# folded bodies that open on the KDA projections: anchor on the conv instead
M(K + "g_b_proj", NVJ, before_name="_causal_conv1d")
M(K + "f_b_proj", NVJ, before=K + "g_b_proj")
M(K + "in_proj", NVJ, before=K + "f_b_proj")
M(K + "decode_glue", EW, after="<none>", before="<none>")
M(K + "prefill_qkv_copy", EW, before=K + "chunk_prefill")
M(K + "short_conv_prefill", "_causal_conv1d_fwd_kernel")
M(K + "short_conv_decode", "_causal_conv1d_update_kernel")
M(K + "decode_glue", EW, after=K + "short_conv_decode")
M(K + "decode_glue", EW, after=K + "decode_glue")
M(K + "state_gather", "_gather_initial_states_kernel")
M(K + "prefill_glue", "triton_poi_fused__to_copy_sigmoid_0")
M(K + "prefill_qkv_copy", EW, after=K + "prefill_glue")
M(K + "prefill_qkv_copy", EW, after=K + "chunk_prefill")
for n in ("l2norm_fwd_kernel2", "kda_gate_cumsum_fwd_kernel", "chunk_kda_scaled_dot_kkt_fwd_kernel",
          "merge_16x16_to_64x64_inverse_kernel", "recompute_w_u_fwd_kernel",
          "chunk_gated_delta_rule_fwd_kernel_h", "chunk_gla_fwd_kernel_o"):
    M(K + "chunk_prefill", n)
M(K + "recurrent_decode", "fused_recurrent_gated_delta_rule_fwd_kernel")
M(K + "state_scatter", "_scatter_states_kernel")
M(K + "gated_norm", "layer_norm_gated_fwd_kernel")
M(K + "o_proj", NVJ, after=K + "gated_norm")
# prefill small glue: between the qk copies and the chunk kernels
M(K + "prefill_glue", "CUDAFunctor_add<int>")
M(K + "prefill_glue", "CatArrayBatchedCopy_alignedK_contig")
M(K + "prefill_glue", "DeviceScanInitKernel")
M(K + "prefill_glue", "DeviceScanKernel")
for prev in (K + "prefill_glue", K + "chunk_prefill", K + "prefill_qkv_copy"):
    M(K + "prefill_glue", VEW, after=prev)
M(K + "prefill_glue", COPY, after=K + "prefill_glue")

# ---------------- DSA ----------------
D = "dsa."
for s, role in [("fused_qkv_a", "DSA fused Q and latent KV projection"),
                ("q_kv_a_norm", "DSA fused q/kv RMSNorm"),
                ("q_b_proj", "DSA query up projection"),
                ("indexer.wq_b", "DSA indexer query projection"),
                ("indexer.wk_weights", "DSA indexer key/weights projection"),
                ("indexer.head_weights", "DSA indexer fp32 head-weights torch.mm (cast copy, SIMT sgemm, split-K reduce)"),
                ("indexer.q_fwht_quant", "DSA indexer FWHT-128 + FP8 quant of q"),
                ("indexer.kpool_gate_score", "DSA indexer kpool gate-score projection"),
                ("indexer.prefill_gather", "DSA prefill k[idx] / gate_score[idx] gathers"),
                ("indexer.kpool_decode_update", "DSA kpool decode cache update"),
                ("indexer.kpool_prefill_write", "DSA kpool prefill softmax/rotate cache write"),
                ("indexer.kpool_tail_seed", "DSA kpool prefill tail seed"),
                ("indexer.mqa_logits_decode", "DSA indexer paged MQA logits (decode rows)"),
                ("indexer.topk_decode", "DSA indexer persistent top-k (decode rows)"),
                ("indexer.mqa_logits_prefill", "DSA indexer prefill MQA logits with its k-cache gather"),
                ("indexer.topk_prefill", "DSA indexer prefill top-k"),
                ("indexer.expand_pools", "DSA pool-to-token expansion"),
                ("q_absorb", "MLA W_UK query absorb bmm"),
                ("q_concat", "MLA query concat copy for FP8 MQA"),
                ("q_fp8_quant", "MLA query FP8 quant, with the two remap-buffer fills launched between it and the remap"),
                ("sparse_mla.mla_cache_append", "MLA latent cache append"),
                ("sparse_mla.index_remap", "sparse MLA request-to-global index remap (triton)"),
                ("sparse_mla.prefill", "token-sparse MLA attention, prefill-bearing variant"),
                ("sparse_mla.decode", "token-sparse MLA attention, decode-only variant"),
                ("output_copy", "sparse MLA output copy"),
                ("v_up", "MLA W_UV value up bmm"),
                ("o_proj", "DSA output projection"),
                ("glue", "DSA index/mask plumbing (fills, aranges, compares, copies)")]:
    op(D + s, role, [D + s])

# fused_qkv_a: plain GEMM straight into the norm, or split-K GEMM + reduce into the norm
M(D + "fused_qkv_a", NVJ, after_name=BIGFUSE, before_name="_fused_q_kv_rmsnorm")
M(D + "fused_qkv_a", SKR, before_name="_fused_q_kv_rmsnorm")
M(D + "fused_qkv_a", "_splitK_", before=D + "fused_qkv_a")
M(D + "q_kv_a_norm", "_fused_q_kv_rmsnorm_kernel")
M(D + "q_b_proj", NVJ, after=D + "q_kv_a_norm")
M(D + "indexer.wq_b", NVJ, after=D + "q_b_proj")
for g in GEMMS:
    M(D + "indexer.wk_weights", g, after=D + "indexer.wq_b")
M(D + "indexer.head_weights", COPY, after=D + "indexer.wk_weights")
M(D + "indexer.head_weights", "cutlass_80_simt_sgemm")
M(D + "indexer.q_fwht_quant", "_fwht_quant_kernel")
for g in GEMMS:
    M(D + "indexer.kpool_gate_score", g, after_name="triton_poi_fused_mul_unsqueeze_0")
M(D + "indexer.prefill_gather", "vectorized_gather_kernel", phase="forward")
M(D + "indexer.kpool_decode_update", "_kpool_decode_update_batched_kernel")
M(D + "indexer.kpool_prefill_write", "_kpool_softmax_rotate_write_cache_kernel")
M(D + "indexer.kpool_tail_seed", "_kpool_tail_seed_kernel")
M(D + "indexer.mqa_logits_decode", "deep_gemm::sm100_paged_mqa_logits<")
M(D + "indexer.topk_decode", "persistent_topk_kernel")
M(D + "indexer.mqa_logits_prefill", "cp_gather_indexer_k_quant_cache_kernel")
M(D + "indexer.mqa_logits_prefill", "deep_gemm::sm100_mqa_logits")
M(D + "indexer.topk_prefill", "topKPerRowPrefill")
M(D + "indexer.expand_pools", "_expand_pools_and_append_tail_kernel")
M(D + "sparse_mla.mla_cache_append", "concat_and_cache_mla_kernel")
M(D + "q_absorb", NVJ, after=D + "sparse_mla.mla_cache_append")
M(D + "q_concat", "CatArrayBatchedCopy<", after=D + "q_absorb")
M(D + "q_fp8_quant", "scaled_fp8_quant_kernel_strided_group_shape")
M(D + "sparse_mla.decode", "fmhaSm100fKernel_QkvE4m3OBfloat16H512PagedKvDenseDynamicTokenSparseP1MultiCtasKv")
M(D + "sparse_mla.decode", "fmhaSm100fKernel_QkvE4m3OBfloat16H512HVPerCta128PagedKvDenseDynamicTokenSparseP1MultiCtasKv")
M(D + "sparse_mla.prefill", "fmhaSm100fKernel_QkvE4m3OBfloat16H512PagedKvDenseDynamicTokenSparseP1VarSeq")
M(D + "output_copy", EW, after=D + "sparse_mla.decode")
M(D + "output_copy", EW, after=D + "sparse_mla.prefill")
M(D + "v_up", NVJ, after=D + "output_copy")
M(D + "o_proj", NVJ, after=D + "v_up")
# remap: third launch after the q FP8 quant. The two fills between them carry
# the quant's operation: a distinct fill operation would have to share a slot
# with dsa.glue, which the analyzer rejects when both are present.
M(D + "q_fp8_quant", VEW, after=D + "q_fp8_quant")
M(D + "q_fp8_quant", EW, after=D + "q_fp8_quant")
for prv in (VEW, EW):
    for nxt in (VEW, UEW):
        M(D + "sparse_mla.index_remap", "kernel", after=D + "q_fp8_quant", after_name=prv, before_name=nxt)

# ---------------- dense FFN / MoE ----------------
F = "dense_ffn."
for s, role in [("gate_up_input_quant", "dense FFN gate/up input FP8 group quant"),
                ("gate_up", "dense FFN gate/up deepgemm"),
                ("act_and_mul", "dense FFN SiLU and multiply"),
                ("down_input_quant", "dense FFN down input FP8 group quant"),
                ("down", "dense FFN down deepgemm")]:
    op(F + s, role, [F + s])
S = "moe.shared_expert."
for s, role in [("gate_up_input_quant", "shared-expert gate/up input FP8 group quant"),
                ("gate_up", "shared-expert gate/up deepgemm"),
                ("act_and_mul", "shared-expert SiLU and multiply"),
                ("down_input_quant", "shared-expert down input FP8 group quant"),
                ("down", "shared-expert down deepgemm")]:
    op(S + s, role, [S + s])
op("moe.router", "MoE router gate GEMM (fp32 out), computed twice by the fork", ["moe.router.gate", "moe.router.gate_recompute"])
op("moe.routed.input_quant", "routed-expert FP8 input group quant", ["input_quant"])
op("moe.input_glue", "routed-expert input copy", ["moe.input_glue"])
op("moe.routed.fused_moe", "TRT-LLM FP8 block-scale fused MoE: routing, FC1, activation, FC2, finalize", ["fused_moe"])
op("moe.combine_glue", "shared + routed add", ["moe.combine_glue"])

def ffn_chain(prefix, first_evidence):
    for ev in first_evidence:
        M(prefix + "gate_up_input_quant", PACKQ, **ev)
    M(prefix + "gate_up", DG, after=prefix + "gate_up_input_quant")
    M(prefix + "act_and_mul", ACT, after=prefix + "gate_up")
    M(prefix + "down_input_quant", PACKQ, after=prefix + "act_and_mul")
    M(prefix + "down", DG, after=prefix + "down_input_quant")

ffn_chain(F, [dict(after_name=BIGFUSE)])
ffn_chain(S, [dict(after="moe.router"), dict(after_name="<none>")])
M("moe.router", TSS)
M("moe.router", "linearcute_dsl_ll_bf16")
M("moe.routed.input_quant", ROUTEQ)
M("moe.input_glue", EW, after="moe.routed.input_quant")
for n in ("moe::dev::routing::", "bmm_E4m3_E4m3E4m3", "bmm_Bfloat16_E4m3E4m3", "activationDeepSeekKernel",
          "moe::dev::finalize::"):
    M("moe.routed.fused_moe", n)
M("moe.combine_glue", VEW, after="moe.routed.fused_moe")
M("moe.combine_glue", VEW, after=S + "down")
M("moe.combine_glue", VEW, after="<none>", before="tp.all_reduce")
# a folded body that opens mid shared expert: anchor on the add ahead of the reduce
M("moe.combine_glue", VEW, after_name=DG, before="tp.all_reduce")
M(S + "down", DG, before="moe.combine_glue")
M(S + "down_input_quant", PACKQ, before=S + "down")
M(S + "act_and_mul", ACT, before=S + "down_input_quant")
M(S + "gate_up", DG, before=S + "act_and_mul")

# ---------------- split-K reduces follow their GEMM ----------------
GEMM_OPS = [K + "in_proj", K + "f_b_proj", K + "g_b_proj", K + "o_proj", D + "q_b_proj", D + "indexer.wq_b",
            D + "indexer.wk_weights", D + "indexer.head_weights", D + "indexer.kpool_gate_score", D + "q_absorb",
            D + "v_up", D + "o_proj", "moe.router"]
for g in GEMM_OPS:
    M(g, SKR, after=g)

# ---------------- DSA glue chain ----------------
DSA_GLUE_FROM = [D + s for s in ("indexer.kpool_gate_score", "indexer.prefill_gather", "indexer.kpool_prefill_write",
                                 "indexer.kpool_tail_seed", "indexer.kpool_decode_update", "indexer.mqa_logits_decode",
                                 "indexer.topk_decode", "indexer.mqa_logits_prefill", "indexer.topk_prefill",
                                 "indexer.expand_pools", "sparse_mla.index_remap", "glue")]
GENERIC = [EW, VEW, UEW, "elementwise_kernel_with_index"]
for prev in DSA_GLUE_FROM:
    for n in GENERIC:
        M(D + "glue", n, after=prev)

json.dump({"schema_version": 1,
           "note": "GLM-5.3-Flash FP8 vLLM TP4/DP1/EP4 on B200 (glm53_flash_vllm_fp8_kda_dsa_moe). One order-free rule set; generated by gen_rules.py in this directory, then pruned to rules that match at the fixpoint.",
           "rules": rules}, open(sys.argv[1], "w"), indent=1)
print(len(rules), "rules")
