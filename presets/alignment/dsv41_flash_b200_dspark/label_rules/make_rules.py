"""Generate the dsv41_flash_b200_dspark label rules for one topology.

Usage: make_rules.py <cost_manifest.json> > <topo>/rules.json
The cost manifest is any timing-predict worker manifest of that topology
(timing_predict/raw/cost_manifest/worker_predict_0.json); its slots decide TP4
(per-layer TP all-reduces) vs DP4 (EP dispatch all-gather / combine
reduce-scatter), the swap-AB decode slot and the DSpark draft slots.

Evidence: logs/20261106_0_dsv41_dspark_capture/profile_{tp4ep4,dp4ep4}_{,diverse_}nsys
(vLLM fork, MRV2, DSpark). Rules are an order-independent set: every
ambiguous kernel name is split by its immediate neighbours so no two rules with
different operations claim one position.
"""
import json, sys

MANIFEST = json.load(open(sys.argv[1]))
SLOTNAMES = [sl["name"] for sec in MANIFEST["sections"] for sl in sec["slots"]]
# TP4 attention (per-layer TP all-reduces) vs DP4 attention + EP4 MoE (dispatch all-gather / combine reduce-scatter)
TP = any(n.endswith("tp_all_reduce") for n in SLOTNAMES)
SRC = "20261106_0_dsv41_dspark_capture/profile_%s_{,diverse_}nsys" % ("tp4ep4" if TP else "dp4ep4")
TARGET_EXCL = ["unified.dspark."]
rules = []


def R(name, op, slots, typ, role, *, excl=TARGET_EXCL, phase="forward", sync=False, note="", **m):
    r = {"name": name, "operation": op, "type": typ, "role": role, "slot_suffixes": slots,
         "phase": phase, "cross_rank": "synchronizing" if sync else "independent",
         "source_experiment": SRC}
    if excl:
        r["excluded_slot_prefixes"] = excl
    r.update(m)
    if note:
        r["note"] = note
    rules.append(r)


def both(*suffixes):
    """attn.* slot suffixes plus the bounded-replay cut layer's write_kv copies."""
    out = []
    for s in suffixes:
        out.append("attn." + s)
        out.append("write_kv." + s)
    return out


GEMM = "flashinfergemmkernelsdense_blockscaled_gemm_sm100"
QS5120 = "QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o512051201"
# per-rank shapes: TP4 shards wq_b / wo_b / the shared expert four ways; DP4 attention runs them whole
QS_DOWN = ("QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o5765761" if TP
           else "QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o230423041")
QS_DOWN_TAIL = QS_DOWN.split("gmemalign")[-1]
QS6144 = "QuantizeSwizzledKernel_object_at__tensorptrbf16gmemalign16o614461441"
QL5120 = "QuantizeLinearKernel_object_at__tensorptrbf16gmemalign16o512051201"
W = "dense_mxfp8_gemm_warp_splitkCpAsyncMxFp8WarpSplitKKernel_object_at__tensorptri16gmemalign32o"

# ---------------- preprocess / prologue ----------------
if TP:
    R("vocab_parallel_embedding_kernel", "prologue.embedding", ["prologue.embedding"], "model",
      "vocabulary-parallel token embedding lookup", phase="preprocess")
else:
    R("indexSelectSmallIndex<c10::BFloat16", "prologue.embedding", ["prologue.embedding"], "model",
      "token embedding lookup (TP1: a plain bf16 index_select)", phase="preprocess",
      note="the preprocess range's only bf16 index_select")
if TP:
    R("trtllm_mnnvl_allreduce::", "prologue.embedding_all_reduce", ["prologue.embedding_all_reduce"],
      "collective", "embedding output TP all-reduce", phase="preprocess", sync=True,
      note="The only all-reduce in the preprocess range is the vocab-parallel embedding reduction.")
R("_hash_ids_kernel", "prologue.engram_hash", ["prologue.engram_hash"], "model",
  "Engram n-gram hash of the input ids, with its pad-token masks")
R("CompareEqFunctor", "prologue.engram_hash", ["prologue.engram_hash"], "model",
  "Engram n-gram hash of the input ids, with its pad-token masks", note="pad-token mask before the hash")
R("bitwise_not_kernel_cuda", "prologue.engram_hash", ["prologue.engram_hash"], "model",
  "Engram n-gram hash of the input ids, with its pad-token masks")

# ---------------- Engram ----------------
R("_engram_lookup_kernel", "engram.lookup", ["engram_lookup"], "model", "Engram table lookup (prefetched)")
if TP:
    R("ncclDevKernel_AllGather", "engram.all_gather", ["engram.all_gather"], "collective",
      "Engram TP all-gather of the looked-up rows", sync=True,
      note="Every forward-range all-gather is Engram's; the logits all-gather is in the sample range.")
    R("elementwise_kernel", "engram.gather_copy", ["engram.gather_copy"], "model",
      "copy of the gathered Engram rows", after_name="ncclDevKernel_AllGather")
for frag in (QS6144,):
    R(frag, "engram.wkv", ["engram.wkv"], "model", "Engram wkv projection (MXFP8 quant + GEMM)")
R(GEMM, "engram.wkv", ["engram.wkv"], "model", "Engram wkv projection (MXFP8 quant + GEMM)",
  after_name="gmemalign16o614461441")
R("_fused_engram_post_wkv_kernel", "engram.post_wkv", ["engram.post_wkv"], "model", "Engram post-wkv gate")

# ---------------- mHC boundaries ----------------
MHC_SLOTS = ["attn.entry.hc_expand", "attn.entry.mhc_hc_prenorm", "attn.entry.mhc_pre_big_fuse",
             "entry.mega_mhc_post_pre", "entry.mhc_overlap.post", "entry.mhc_overlap.pre_input",
             "entry.mhc_overlap.coefficients", "entry.mhc_overlap.coefficients.serial",
             "engram.mhc_post", "engram.mhc_hc_prenorm", "engram.mhc_pre_big_fuse"]
MHC_ROLE = ("every mHC sublayer boundary: Mega-mHC post+pre, the FULL-graph overlap "
            "(side-stream coefficients, caller post + input), the unfused layer-0 / Engram entries")
for frag in ("sm100_mega_mhc_impl", "sm100_tf32_hc_prenorm_gemm_impl",
             "mhc_pre_big_fuse_with_norm_tilelang_kernel", "mhc_fused_tilelang_kernel", "_hc_expand_kernel"):
    R(frag, "mhc.boundary", MHC_SLOTS, "model", MHC_ROLE)
for nxt in ("mhc_pre_big_fuse", "ncclDevKernel_AllGather", "sm100_tf32_hc_prenorm_gemm", "reduce_kernel", "FillFunctor",
            "gmemalign16o614461441", "<none>"):
    R("mhc_post_tilelang_kernel", "mhc.boundary", MHC_SLOTS, "model", MHC_ROLE, before_name=nxt,
      note="post half of a sublayer boundary; the head's post (before _hc_head_reduce_store) is head.mhc_post")
R("reduce_kernel", "dspark.aux_hc_mean", ["dspark.context_kv.aux_hc_mean"], "model",
  "mean over hc copies of the post-mapped stream at a DSpark aux layer", excl=None,
  after_name="mhc_post_tilelang_kernel")

# ---------------- attention input / main ----------------
FWQ = both("input.fused_wqa_wkv")
R(QS5120, "attn.fused_wqa_wkv", FWQ, "attention", "fused wq_a + wkv projection (MXFP8 quant + GEMM)",
  before="attn.fused_wqa_wkv")
R(W + "1792256025601", "attn.fused_wqa_wkv", FWQ, "attention", "fused wq_a + wkv projection (MXFP8 quant + GEMM)")
for nxt in ("_q_kv_norm_quant_kernel", "nvjet_sm100_tst", "nvjet_sm100_tss", "<none>"):
    R(GEMM, "attn.fused_wqa_wkv", FWQ, "attention", "fused wq_a + wkv projection (MXFP8 quant + GEMM)",
      after_name=QS5120, before_name=nxt)
R(GEMM, "attn.fused_wqa_wkv", FWQ, "attention", "fused wq_a + wkv projection (MXFP8 quant + GEMM)",
  after_name=QS5120, before="attn.fused_wqa_wkv",
  note="track-final or followed by the attention input's own split-K projection; in every observed case "
       "the quant before it follows an attention-entry mHC (after the FFN all-reduce or Engram)")
# small-M bf16 path: cuBLAS split-K GEMM + reduce feeding the q/kv norm
R("splitKreduce_kernel", "attn.fused_wqa_wkv", FWQ, "attention", "fused wq_a + wkv projection (MXFP8 quant + GEMM)",
  before_name="_q_kv_norm_quant_kernel")
R("nvjet_sm100_", "attn.fused_wqa_wkv", FWQ, "attention", "fused wq_a + wkv projection (MXFP8 quant + GEMM)",
  before="attn.fused_wqa_wkv", before_name="splitKreduce_kernel")
R("_q_kv_norm_quant_kernel", "attn.qk_rmsnorm", both("main.qk_rmsnorm"), "attention",
  "q / kv RMSNorm with the wq_b input quant")
R(W + ("81926406401" if TP else "327686406401"), "attn.wq_b", both("main.wq_b"), "attention", "wq_b projection")
R(GEMM, "attn.wq_b", both("main.wq_b"), "attention", "wq_b projection", after_name="_q_kv_norm_quant_kernel")
R("fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel", "attn.kv_insert", both("main.kv_insert"), "attention",
  "q norm + rope, kv rope + quant + cache insert")

# ---------------- compressor / indexer ----------------
R("_fused_save_compress_norm_kernel", "attn.compressor.save_compress_norm",
  both("compressor.save_compress_norm", "compressor.save_compress_norm.serial"), "attention",
  "compressor state save + compress + norm")
R("_rope_quant_insert_nvfp4_kernel", "attn.compressor.nvfp4_insert",
  both("compressor.nvfp4_insert", "compressor.nvfp4_insert.serial"), "attention",
  "compressed KV rope + NVFP4 quant + insert")
R("_indexer_k_norm_rope_quant_store_kernel", "attn.indexer.k_store", both("indexer.k_store"), "attention",
  "indexer k norm + rope + quant + store")
R(W + "40966406401", "attn.indexer.wq_b", both("indexer.wq_b"), "attention", "indexer wq_b projection")
R(GEMM, "attn.indexer.wq_b", both("indexer.wq_b"), "attention", "indexer wq_b projection",
  before_name="kernel_cutlass_device_kernel_tensorptri64gmemo1")
R("kernel_cutlass_device_kernel_tensorptri64gmemo1", "attn.indexer.q_rope_quant", both("indexer.q_rope_quant"),
  "attention", "indexer q rope + quant")
SCORE = both("indexer.score.prefill_k_gather", "indexer.score.prefill_logits", "indexer.score.prefill_topk",
             "indexer.score.decode_logits", "indexer.score.decode_topk", "indexer.score.candidates")
for frag in ("cp_gather_indexer_k_quant_cache_kernel", "deep_gemm::sm100_mqa_logits", "sm100_paged_mqa_logits<",
             "sm100_paged_sparse_mqa_logits", "deep_gemm::sm100_sparse_mqa_logits<", "_upcast_logits_kernel",
             "topKPerRowPrefill", "topk_select_bf16_normal", "topk_select_fp32", "_sparse_topk_remap_kernel",
             "FilteredTopKUnifiedKernel", "cooperative_topk", "sampled_topk_kernel", "_block_scores_kernel",
             "_store_candidates_kernel", "_expand_candidates_kernel", "radixSortKVInPlace",
             "bitonicSortKVInPlace", "gatherTopK", "sm100_sparse_mqa_logits_metadata"):
    R(frag, "attn.indexer.score", SCORE, "attention",
      "indexer scoring: logits, top-k and sparse candidate selection, prefill and decode")
# small projections on the attention / FFN input, all cuBLAS(nvjet)/CUTLASS/cute-DSL GEMMs
SMALL = ["attn.compressor.kv_score_proj", "attn.compressor.kv_score_proj.serial", "attn.indexer.weights_proj",
         "attn.indexer.weights_proj.serial", "attn.indexer.wk",
         "write_kv.compressor.kv_score_proj", "write_kv.compressor.kv_score_proj.serial",
         "write_kv.indexer.weights_proj", "write_kv.indexer.weights_proj.serial", "write_kv.indexer.wk",
         "ffn.routed.router_gate"]
SMALL_ROLE = "small input projections: compressor kv_score, indexer weights / wk, MoE router gate"
for nxt in ("_indexer_k_norm_rope_quant_store_kernel", "_fused_save_compress_norm_kernel", "_dsv4_topk_kernel",
            "QuantizeSwizzled", "nvjet_sm100_tst"):
    R("nvjet_sm100_", "proj.small", SMALL, "model", SMALL_ROLE, before_name=nxt,
      note="split by successor; an nvjet split-K GEMM whose reduce feeds the q/kv norm is attn.fused_wqa_wkv")
for nxt in ("_fused_save_compress_norm_kernel", "_dsv4_topk_kernel", "QuantizeSwizzled"):
    R("splitKreduce_kernel", "proj.small", SMALL, "model", SMALL_ROLE, before_name=nxt)
R("nvjet_sm100_", "proj.small", SMALL, "model", SMALL_ROLE, before="proj.small")
R("splitKreduce_kernel", "proj.small", SMALL, "model", SMALL_ROLE, after="proj.small", before="proj.small")
R("cute_dsl_ll_bf16_splitkLLBf16SplitK", "proj.small", SMALL, "model", SMALL_ROLE)
# MXFP8 path of an aux-stream input GEMM (compressor fused_wkv_wgate / indexer weights_proj,
# DeepseekV4Attention._run_parallel_input_projections): quant + GEMM whose output is copied out
R(GEMM, "proj.small", SMALL, "model", SMALL_ROLE, after_name=QS5120, before_name="bfloat16_copy_kernel_cuda",
  note="an aux-stream input GEMM on the MXFP8 path; fused_wqa_wkv's GEMM feeds the q/kv norm instead")
R(QS5120, "proj.small", SMALL, "model", SMALL_ROLE, before="proj.small")
R("cutlass::Kernel2", "proj.small", SMALL, "model", SMALL_ROLE)

# ---------------- attention core / output ----------------
MEGA = both("attention.mega_attn.prefill", "attention.mega_attn.decode")
MEGA_ROLE = "FlashMLA-style sparse attention (prefill and decode kernels, incl. its top-k index prep)"
for frag in ("fused_norm_rope_attn_rope_cast_fwd::core_attn::fwd_kernel",
             "_dequantize_and_gather_k_nvfp4_kernel", "kernel_cutlass_device_kernel_tensorptrbf16gmemalign32o512i64"):
    R(frag, "attn.mega_attn", MEGA, "attention", MEGA_ROLE)
# decode global top-k remap (compute_global_topk_indices_and_lens): its own placeholder leaf
R("_compute_global_topk_indices_and_lens_kernel", "attn.global_topk_remap", both("attention.global_topk_remap"),
  "attention", "decode top-k local->global slot remap before the compressed-layer decode attention")
# swap-AB small-decode kernel (flash_mla_mega_attn.py _forward_decode_mega: compressed layers with 16/32
# live heads, steps of <= 16 tokens); its own slot once the arch prices it separately
SWAPAB = "kernel_cutlass_device_kernel_tensorptrbf16gmemalign16o5125121"
if any(n.endswith("attention.mega_attn.decode_swapab") for n in SLOTNAMES):
    R(SWAPAB, "attn.mega_attn_swapab", both("attention.mega_attn.decode_swapab"), "attention",
      "swap-AB decode attention (compressed layers, decode steps of <= 16 tokens)")
else:
    R(SWAPAB, "attn.mega_attn", MEGA, "attention", MEGA_ROLE)
R("kernel", "attn.mega_attn", MEGA, "attention", MEGA_ROLE, name_exact=True,
  before_name="fused_norm_rope_attn_rope_cast_fwd")
R("sm100_fp8_fp4_gemm_1d1d_impl", "attn.wo_a", both("output.wo_a_einsum"), "attention", "wo_a grouped einsum")
if TP:
    R(W + "5120102410241", "attn.wo_b", both("output.wo_b"), "attention", "wo_b projection")
R(GEMM, "attn.wo_b", both("output.wo_b"), "attention", "wo_b projection", after_name="sm100_fp8_fp4_gemm_1d1d_impl")
if TP:
    R("Allreduce", "layer.tp_all_reduce", ["attn.tp_all_reduce", "ffn.tp_all_reduce"], "collective",
      "attention-output and FFN-output TP all-reduces", sync=True)
else:
    R("ncclDevKernel_AllGather", "ffn.routed.dispatch_all_gather", ["ffn.routed.dispatch_all_gather"], "collective",
      "EP dispatch: all-gather of every DP rank's MoE input over the EP group", sync=True)
    R("ncclDevKernel_ReduceScatter", "ffn.routed.combine_reduce_scatter", ["ffn.routed.combine_reduce_scatter"],
      "collective", "EP combine: reduce-scatter of the routed-expert output back to its DP rank", sync=True)

# ---------------- FFN ----------------
GU = ["ffn.shared.gate_up", "ffn.shared.gate_up.serial"]
R(W + ("1152256025601" if TP else "4608256025601"), "ffn.shared.gate_up", GU, "ffn", "shared expert gate_up (MXFP8 quant + GEMM)")
R(GEMM, "ffn.shared.gate_up", GU, "ffn", "shared expert gate_up (MXFP8 quant + GEMM)",
  after_name=QS5120, before_name="act_and_mul_kernel")
R(QS5120, "ffn.shared.gate_up", GU, "ffn", "shared expert gate_up (MXFP8 quant + GEMM)", before="ffn.shared.gate_up")
R("act_and_mul_kernel", "ffn.shared.act_and_mul", ["ffn.shared.act_and_mul", "ffn.shared.act_and_mul.serial"],
  "ffn", "shared expert SiLU-and-mul")
R(QS_DOWN, "ffn.shared.down", ["ffn.shared.down", "ffn.shared.down.serial"], "ffn",
  "shared expert down (MXFP8 quant + GEMM)")
R(GEMM, "ffn.shared.down", ["ffn.shared.down", "ffn.shared.down.serial"], "ffn",
  "shared expert down (MXFP8 quant + GEMM)", after_name=QS_DOWN_TAIL)
if not TP:
    R(W + "5120115211521", "ffn.shared.down", ["ffn.shared.down", "ffn.shared.down.serial"], "ffn",
      "shared expert down (MXFP8 quant + GEMM)")
R("_dsv4_topk_kernel", "ffn.routed.topk", ["ffn.routed.topk"], "ffn", "router top-k")
R(QL5120, "ffn.routed.input_quant", ["ffn.routed.input_quant"], "ffn", "routed-expert input MXFP8 quant")
for frag in ("routingCustom::routingIndices", "bmm_MxE4m3_MxE2m1", "bmm_Bfloat16_MxE2m1", "moe::dev::finalize::finalizeKernel"):
    R(frag, "ffn.routed.fused_moe", ["ffn.routed.fused_moe"], "ffn",
      "TRT-LLM fused MoE: routing permute, FC1, FC2, finalize")
if TP:
    R("CUDAFunctor_add", "ffn.residual_add", ["ffn.residual_add"], "ffn", "shared + routed expert sum",
      before_name="Allreduce")
else:
    for prev in ("ffn.routed.fused_moe", "ffn.routed.combine_reduce_scatter"):
        R("CUDAFunctor_add", "ffn.residual_add", ["ffn.residual_add"], "ffn", "shared + routed expert sum",
          after=prev, note="DP4: the sum follows the routed output (or its combine reduce-scatter)")

# ---------------- bounded replay ----------------
for frag in ("vectorized_gather_kernel", "_scatter_gather_elementwise_kernel"):
    R(frag, "replay.gather", ["replay.gather"], "model",
      "bounded replay: gather the replay rows' residual, mixes and inputs above the cut layer",
      note="the forward range has no other gather; the chain ends at the replay batch's first mHC")
for frag in ("index_copy_kernel", "FillFunctor<c10::BFloat16>"):
    R(frag, "replay.scatter", ["replay.scatter"], "model",
      "bounded replay: zeros + index_copy of the replay rows back into the full batch",
      note="the forward range's only bf16 fill and index_copy; the chain ends at the head's hc collapse")

# ---------------- head ----------------
R("mhc_post_tilelang_kernel", "head.mhc_post", ["head.mhc_post"], "model", "last layer's mHC post",
  before_name="_hc_head_reduce_store_kernel")
R("_hc_head_reduce_store_kernel", "head.hc_head_collapse", ["head.hc_head_collapse"], "model", "hc head collapse")
R("rms_norm_kernel", "head.final_rms_norm", ["head.final_rms_norm"], "model", "final RMSNorm",
  after_name="_hc_head_reduce_store_kernel")

# ---------------- sample ----------------
R("nvjet_sm100_tst_", "head.lm_head", ["head.lm_head"], "model", "target lm_head (vocab-parallel)",
  phase="sample")
R("splitKreduce_kernel", "head.lm_head", ["head.lm_head"], "model", "target lm_head (vocab-parallel)",
  phase="sample")
if TP:
    R("ncclDevKernel_AllGather", "head.logits_all_gather", ["head.logits_all_gather"], "collective",
      "target logits all-gather", phase="sample", sync=True)
for frag in ("_compute_local_logits_stats_kernel", "_rejection_kernel", "_resample_kernel", "_insert_resampled_kernel"):
    R(frag, "sample.verify", ["sample.logits_fp32_cast", "sample.rejection_logits_stats", "sample.resample"],
      "model", "speculative verification: logits stats, rejection, resample", phase="sample", excl=None)

# ---------------- DSpark draft (one compute op + one collective op) ----------------
DRAFT_SLOTS = sorted({sl["name"][len("unified."):] for sec in MANIFEST["sections"] for sl in sec["slots"]
                      if sl["name"].startswith("unified.dspark.")
                      and sl["kind"] not in ("all_reduce", "all_reduce_fusion", "moe_ep_all_gather",
                                             "moe_ep_quantized_all_gather", "moe_ep_reduce_scatter")
                      and not sl["name"].endswith("aux_hc_mean")})
DRAFT_ROLE = "DSpark draft: context KV, three SWA-only layers, head and Markov steps"
for frag in ("vocab_parallel_embedding_kernel", "QuantizeSwizzled", "QuantizeLinear", GEMM, "dense_mxfp8_gemm_warp",
             "rms_norm_kernel", "fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel", "sm100_tf32_hc_prenorm_gemm",
             "mhc_pre_big_fuse", "sm100_mega_mhc_impl", "mhc_post_tilelang", "_q_kv_norm_quant_kernel",
             "fused_norm_rope_attn_rope_cast_fwd::core_attn::fwd_kernel", "sm100_fp8_fp4_gemm_1d1d_impl",
             "act_and_mul_kernel", "CUDAFunctor_add", "cute_dsl_ll_bf16", "topkGatingSoftplusSqrt",
             "routingCustom::routingIndices", "bmm_MxE4m3_MxE2m1", "bmm_Bfloat16_MxE2m1", "finalizeKernel",
             "_hc_head_reduce_store_kernel", "nvjet_sm100_", "splitKreduce_kernel", "_argmax_partial", "_argmax_finish",
             "CatArrayBatchedCopy", "indexSelectSmallIndex"):
    R(frag, "dspark.draft", DRAFT_SLOTS,
      "model", DRAFT_ROLE, phase="draft", excl=None)
if TP:
    for frag in ("Allreduce", "ncclDevKernel_AllGather"):
        R(frag, "dspark.collectives", ["dspark.embedding_all_reduce", "tp_all_reduce", "dspark.head.logits_all_gather"],
          "collective", "DSpark draft all-reduces and logits all-gather", phase="draft", sync=True,
          excl=["unified.layer"])
else:
    for frag in ("ncclDevKernel_AllGather", "ncclDevKernel_ReduceScatter"):
        R(frag, "dspark.collectives", ["routed.dispatch_all_gather", "routed.combine_reduce_scatter"],
          "collective", "DSpark draft MoE EP dispatch all-gather and combine reduce-scatter", phase="draft",
          sync=True, excl=["unified.layer"])

json.dump({"schema_version": 2, "note": __doc__.strip(), "rules": rules}, sys.stdout, indent=1)
