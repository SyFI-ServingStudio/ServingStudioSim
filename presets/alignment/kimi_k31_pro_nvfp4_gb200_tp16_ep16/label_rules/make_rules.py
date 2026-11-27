"""Generate the Kimi K3.1 Pro NVFP4 + DSpark label rules for one topology.

Usage: make_rules.py <cost_manifest.json> > rules.json

The cost manifest is any timing-predict worker manifest of the topology
(timing_predict/raw/cost_manifest/worker_predict_0.json). Its slots decide the
boundary family:

* TP16 attention (gb200_tp16_ep16): per-layer TP all-reduces
  (`all_reduce_fusion`), with the prefill o_proj fused into GemmRsAr;
* TP8 attention x DP2 (b200x2_tp8dp2_ep16): sequence-parallel row all-gather /
  reduce-scatter around attention and an EP16 quantized dispatch all-gather /
  combine reduce-scatter around the routed experts.

The compute rules are shared: both packs run the same vLLM model code
(vllm/models/kimi_k3/nvidia/{model,kda,mla,dsa,latent_moe_runner}.py) and
the same kernels, so a kernel's owner is read from its call site there and from
its neighbours in the capture. The rules are an order-independent set: every
name that serves several operations (the nvjet GEMM tiles, the MXFP8 quant, the
blockscaled GEMM, rms_norm, elementwise launches) is split by its immediate
neighbours, so no position is claimed by two operations.
"""
import json
import sys

MANIFEST = json.load(open(sys.argv[1]))
SLOTNAMES = [sl["name"] for sec in MANIFEST["sections"] for sl in sec["slots"]]
TP = any(n.endswith(".attn_all_reduce") and not n.startswith("unified.draft.") for n in SLOTNAMES)
SRC = "phase-1 diverse_100 nsys capture, %s" % ("TP16/EP16" if TP else "TP8xDP2/EP16")
DRAFT = "unified.draft."
# A barrier in the analyzer spans every rank of the iteration. Under TP16 the
# TP group is every rank, so its collectives are barriers. Under TP8 x DP2 the
# attention TP group is one DP group (8 of 16 ranks): its collectives (ShardQ,
# sequence-parallel all-gather / reduce-scatter, embedding and draft all-reduce,
# logits all-gathers) only wait on that group, and the analyzer has no
# partial-rank barrier, so they are classed independent there. Only the EP16
# dispatch / combine collectives span both DP groups.
GROUP_SYNC = TP
rules = []


def R(name, op, slots, typ, role, *, excl=(DRAFT,), phase="forward", sync=False, note="", **m):
    r = {"name": name, "operation": op, "type": typ, "role": role, "slot_suffixes": list(slots),
         "phase": phase, "cross_rank": "synchronizing" if sync else "independent",
         "source_experiment": SRC}
    if excl:
        r["excluded_slot_prefixes"] = list(excl)
    r.update(m)
    if note:
        r["note"] = note
    if not any(n.endswith("." + s) and not n.startswith(tuple(excl or ())) for n in SLOTNAMES for s in slots):
        raise SystemExit(f"rule {op} <- {name}: no slot ends with {slots}")
    rules.append(r)


def U(name, op, role, *, phase="forward", sync=True, **m):
    """A synchronizing collective no slot owns: stays unmapped, but the analyzer sees it wait.

    The labeler accepts an unmapped rule only as a synchronizing collective, so a
    DP-group collective (sync=False) gets no rule and keeps its initialized label.
    """
    if not sync:
        return
    r = {"name": name, "operation": op, "status": "unmapped", "cross_rank": "synchronizing",
         "note": role, "phase": phase, "source_experiment": SRC}
    r.update(m)
    rules.append(r)


def has(suffix):
    return any(n.endswith("." + suffix) for n in SLOTNAMES)


NV = "nvjet_sm100_tst_"          # bf16-output cuBLASLt GEMM
NV_FP32 = "nvjet_sm100_tss_"     # fp32-output cuBLASLt GEMM (the router)
SPLITK = "splitKreduce_kernel"
QUANT = "mxfp8_quantizeMXFP8QuantizeSwizzledKernel"
FI_GEMM = "kernel_cutlass_kernel_flashinfergemmkernelsdense_"   # blockscaled / mxfp8 warp split-K
RES = "attn_res_fwd_online_v2_kernel"
WMMA = "cutlass_80_wmma_tensorop_bf16"
ELEM = "at::native::elementwise_kernel<"
VEC = "at::native::vectorized_elementwise_kernel<"
DIRECT_COPY = "direct_copy_kernel_cuda"
INDEX_ELEM = "at::native::index_elementwise_kernel<"
# MXFP8 activation quant, by the row width it quantizes (per-rank K of the next GEMM).
Q_HIDDEN = "gmemalign16o716871681"   # 7168: KDA in_proj and shared-expert gate_up input
Q_KDA_O = "gmemalign16o7687681" if TP else "gmemalign16o153615361"   # KDA o_proj input (value heads / TP)
Q_SHARED_DOWN = "gmemalign16o3843841" if TP else "gmemalign16o614461441"  # shared down input

# ---------------- prologue / residual mixing ----------------
R("vocab_parallel_embedding_kernel", "embedding", ["embedding"], "model", "vocab-parallel embedding lookup")
R(RES, "attn_res", ["attn_res", "ffn_res", "final_attn_res"], "model",
  "attention-residual mixing (AttnRes) before every attention, FFN and the final norm")

# ---------------- KDA ----------------
KDA = "KDA linear attention (kda.py KimiDeltaAttention)"
for after_name in (RES, "<none>", "mnnvl_lamport_all_gather", "ncclDevKernel_AllGather"):
    R(QUANT + "_object_at__tensorptrbf16" + Q_HIDDEN, "kda.in_proj", ["kda.in_proj"], "model",
      KDA + ": fused q/k/v/g/beta in_proj, MXFP8 quant + blockscaled GEMM", after_name=after_name)
R(FI_GEMM, "kda.in_proj", ["kda.in_proj"], "model",
  KDA + ": fused q/k/v/g/beta in_proj, MXFP8 quant + blockscaled GEMM", after="kda.in_proj")
for frag in (NV, WMMA):
    R(frag, "kda.f_b", ["kda.f_b"], "model", KDA + ": bf16 f_b (forget-gate low-rank up) projection",
      after="kda.in_proj")
R(SPLITK, "kda.f_b", ["kda.f_b"], "model", KDA + ": bf16 f_b (forget-gate low-rank up) projection",
  after="kda.f_b")
R(ELEM, "kda.glue_copy", ["kda.glue_copy"], "model", KDA + ": q/k/v contiguous copy ahead of the conv",
  after="kda.f_b")
R(ELEM, "kda.glue_copy", ["kda.glue_copy"], "model", KDA + ": q/k/v contiguous copy ahead of the conv",
  before_name="_causal_conv1d_update_kernel")
R(ELEM, "kda.glue_copy", ["kda.glue_copy"], "model", KDA + ": q/k/v contiguous copy ahead of the conv",
  after_name="_gather_initial_states_kernel")
R("_causal_conv1d_update_kernel", "kda.conv_decode", ["kda.conv_decode"], "model", KDA + ": decode causal conv1d update")
R("_kda_recoverssm_verify_kernel", "kda.recoverssm_verify", ["kda.recoverssm_verify"], "model",
  KDA + ": RecoverSSM verify recurrence over the draft block")
R("_packed_causal_conv1d_fwd_kernel", "kda.conv_prefill", ["kda.conv_prefill"], "model", KDA + ": prefill packed causal conv1d")
for frag in ("_gather_initial_states_kernel", "store_cache_checkpoints_kernel"):
    R(frag, "kda.state_io", ["kda.state_io"], "model", KDA + ": prefill recurrent-state gather / checkpoint store")
for frag in ("_flash_kda_fwd_prepare", "_flash_kda_fwd_recurrence"):
    R(frag, "kda.chunk_prefill", ["kda.chunk_prefill"], "model", KDA + ": chunked prefill (prepare + recurrence)")
R("unrolled_elementwise_kernel<at::native::direct_copy", "kda.prefill_glue", ["kda.prefill_glue"], "model",
  KDA + ": prefill output scatter back to token order", after_name="store_cache_checkpoints_kernel")
R(INDEX_ELEM, "kda.prefill_glue", ["kda.prefill_glue"], "model",
  KDA + ": prefill output scatter back to token order", after="kda.prefill_glue")
R("layer_norm_gated_fwd_kernel", "kda.gated_norm", ["kda.gated_norm"], "model", KDA + ": sigmoid-gated RMSNorm")
R(QUANT + "_object_at__tensorptrbf16" + Q_KDA_O, "kda.o_proj", ["kda.o_proj"], "model",
  KDA + ": o_proj, MXFP8 quant + GEMM", before_name=FI_GEMM)
R(FI_GEMM, "kda.o_proj", ["kda.o_proj"], "model", KDA + ": o_proj, MXFP8 quant + GEMM", after="kda.o_proj")

# ---------------- dense FFN (layer 0) ----------------
DENSE = "layer-0 dense SiTU MLP (bf16)"
R(NV, "dense_ffn.gate_up", ["dense_ffn.gate_up"], "model", DENSE + ": gate_up", after_name=RES,
  before_name="situ_and_mul_kernel")
R(SPLITK, "dense_ffn.gate_up", ["dense_ffn.gate_up"], "model", DENSE + ": gate_up",
  before_name="situ_and_mul_kernel")
R(NV, "dense_ffn.gate_up", ["dense_ffn.gate_up"], "model", DENSE + ": gate_up", after_name=RES,
  before="dense_ffn.gate_up")
R("situ_and_mul_kernel", "dense_ffn.situ_and_mul", ["dense_ffn.situ_and_mul"], "model", DENSE + ": SiTU and mul",
  after="dense_ffn.gate_up")
R(NV, "dense_ffn.down", ["dense_ffn.down"], "model", DENSE + ": down", after="dense_ffn.situ_and_mul")
R(SPLITK, "dense_ffn.down", ["dense_ffn.down"], "model", DENSE + ": down", after="dense_ffn.down")

# ---------------- MLA + DSA indexer ----------------
MLA = "MLA attention with the DSA sparse indexer (mla.py / dsa.py)"
# fused_qkv_a: q_a + kv_a (+ g_proj when fused) off the AttnRes output, then the q_a / kv_a norms.
R(NV, "mla.fused_qkv_a", ["mla.fused_qkv_a"], "model", MLA + ": fused q_a/kv_a down-projection",
  before_name="rms_norm_kernel")
R(SPLITK, "mla.fused_qkv_a", ["mla.fused_qkv_a"], "model", MLA + ": fused q_a/kv_a down-projection",
  before_name="rms_norm_kernel")
R(NV, "mla.fused_qkv_a", ["mla.fused_qkv_a"], "model", MLA + ": fused q_a/kv_a down-projection",
  before="mla.fused_qkv_a")
# The small-M fused_a_gemm serves both the q_a/kv_a down-projection and, after the two norms, q_b.
R("fused_a_gemm_kernel", "mla.fused_qkv_a", ["mla.fused_qkv_a"], "model", MLA + ": fused q_a/kv_a down-projection",
  before_name="rms_norm_kernel")
R("rms_norm_kernel", "mla.q_a_norm", ["mla.q_a_norm"], "model", MLA + ": q_a RMSNorm", after="mla.fused_qkv_a")
R("rms_norm_kernel", "mla.kv_a_norm", ["mla.kv_a_norm"], "model", MLA + ": kv_a RMSNorm", after="mla.q_a_norm")
for frag in (NV, "fused_a_gemm_kernel"):
    R(frag, "mla.q_b", ["mla.q_b"], "model", MLA + ": q_b up-projection", after="mla.kv_a_norm")
R(NV, "mla.index_wq", ["mla.index_wq"], "model", MLA + ": indexer wq projection", after="mla.q_b")
R(NV, "mla.index_wk", ["mla.index_wk"], "model", MLA + ": indexer wk projection", after="mla.index_wq")
R(SPLITK, "mla.index_wk", ["mla.index_wk"], "model", MLA + ": indexer wk projection", after="mla.index_wk")
R("_mxfp4_k_norm_quant_insert_kernel", "mla.index_cache_append", ["mla.index_cache_append"], "model",
  MLA + ": indexer k norm + MXFP4 quant + cache insert")
R("_mxfp4_q_quant_kernel", "mla.index_q_quant", ["mla.index_q_quant"], "model", MLA + ": indexer q MXFP4 quant")
for frag in ("cp_gather_indexer_k_quant_cache_kernel", "deep_gemm::sm100_mqa_logits<"):
    R(frag, "mla.index_prefill_logits", ["mla.index_prefill_logits"], "model",
      MLA + ": prefill indexer (gather cached k, MQA logits)")
R("topKPerRowPrefill", "mla.index_prefill_topk", ["mla.index_prefill_topk"], "model", MLA + ": prefill indexer top-k")
R("deep_gemm::sm100_paged_mqa_logits<", "mla.index_decode_logits", ["mla.index_decode_logits"], "model",
  MLA + ": decode indexer paged MQA logits")
# The decode top-k is one of three slots by row count and graph mode
# (cooperative_topk up to 64 rows; persistent_topk eagerly or replayed from a
# full graph above). persistent_topk runs sampled_topk_kernel at a host
# max_seq_len >= 98304, else its persistent or filtered top-k kernel.
DECODE_TOPK = ["mla.index_decode_topk", "mla.index_decode_topk_persistent", "mla.index_decode_topk_graph"]
for frag in ("sampled_topk_kernel", "cooperative_topk", "persistent_topk_kernel", "filtered_topk::FilteredTopKUnifiedKernel"):
    R(frag, "mla.index_decode_topk", DECODE_TOPK, "model", MLA + ": decode indexer top-k")
R("concat_and_cache_mla_kernel", "mla.mla_cache_append", ["mla.mla_cache_append"], "model", MLA + ": latent KV cache append")
R("_bz_NNT", "mla.q_absorb", ["mla.q_absorb"], "model", MLA + ": q absorb (W_UK) batched GEMM")
R("scaled_fp8_quant_kernel_strided_group_shape", "mla.q_quant", ["mla.q_quant"], "model", MLA + ": static FP8 q quant")
R("HQk576HV512HVPerCta256", "mla.sparse_prefill", ["mla.sparse_prefill", "mla.sparse_prefill_shard_q"], "model",
  MLA + ": sparse MLA prefill attention (TRT-LLM gen fmha), plain or on the ShardQ token shards")
R("HQk576HV512PagedKv", "mla.sparse_decode", ["mla.sparse_decode"], "model", MLA + ": sparse MLA decode attention")
for frag, op in (("_shard_q_kernel", "mla.shard_q"), ("_unshard_out_kernel", "mla.unshard_out")):
    slot = op.split(".", 1)[1]
    role = MLA + ": ShardQ %s exchange (symmetric-memory push between two group barriers)" % slot
    R(frag, op, ["mla." + slot], "collective", role, sync=GROUP_SYNC)
    R("symmetric_memory::barrier_kernel", op, ["mla." + slot], "collective", role, sync=GROUP_SYNC, before_name=frag)
    R("symmetric_memory::barrier_kernel", op, ["mla." + slot], "collective", role, sync=GROUP_SYNC, after_name=frag)
R("sparse_mla_index_remap_kernel", "mla.index_remap", ["mla.index_remap"], "model", MLA + ": top-k index remap to cache slots")
R("_pad_query_heads_kernel", "mla.shard_q_pad_heads", ["mla.shard_q_pad_heads"], "model", MLA + ": ShardQ query head padding")
for after in ("mla.sparse_decode", "mla.unshard_out"):
    R(NV, "mla.v_up", ["mla.v_up"], "model", MLA + ": v up-projection (W_UV)", after=after)
# Prefill: the index remap of the next chunk is issued between the attention and v_up on the main stream.
R(NV, "mla.v_up", ["mla.v_up"], "model", MLA + ": v up-projection (W_UV)", after="mla.index_remap",
  stream_role="primary")
R(NV, "mla.g_proj", ["mla.g_proj"], "model", MLA + ": output gate projection", after="mla.v_up")
R(SPLITK, "mla.g_proj", ["mla.g_proj"], "model", MLA + ": output gate projection", after="mla.g_proj")
R(VEC, "mla.gate_mul", ["mla.gate_mul"], "model", MLA + ": sigmoid gate multiply", after="mla.g_proj")
R(VEC, "mla.gate_mul", ["mla.gate_mul"], "model", MLA + ": sigmoid gate multiply", after="mla.v_up")
R(VEC, "mla.gate_mul", ["mla.gate_mul"], "model", MLA + ": sigmoid gate multiply", after="mla.gate_mul")
R(NV, "mla.o_proj", ["mla.o_proj"], "model", MLA + ": o_proj", after="mla.gate_mul")

# ---------------- latent MoE ----------------
MOE = "latent MoE (latent_moe_runner.py): shared experts, router, latent down/up around NVFP4 routed experts"
R(NV_FP32, "moe.router.gate", ["moe.router.gate"], "model", MOE + ": router gate (fp32 logits)")
R(SPLITK, "moe.router.gate", ["moe.router.gate"], "model", MOE + ": router gate (fp32 logits)", after="moe.router.gate")
for frag in ("LLBf16Dotprod", "LLBf16SplitK"):
    R(frag, "moe.router.gate", ["moe.router.gate"], "model", MOE + ": router gate (fp32 logits)")
if has("moe.single_group_topk"):
    R("single_group_topk_block_kernel", "moe.single_group_topk", ["moe.single_group_topk"], "model", MOE + ": router top-k")
# latent down: right after the router (serial path) or ahead of the latent NVFP4 quant (aux stream).
R(NV, "moe.latent_down", ["moe.latent_down"], "model", MOE + ": latent down-projection", after="moe.router.gate")
R(NV, "moe.latent_down", ["moe.latent_down"], "model", MOE + ": latent down-projection", before_name="cvt_fp16_to_fp4_sf_major")
R(SPLITK, "moe.latent_down", ["moe.latent_down"], "model", MOE + ": latent down-projection", before_name="cvt_fp16_to_fp4_sf_major")
R(NV, "moe.latent_down", ["moe.latent_down"], "model", MOE + ": latent down-projection", before="moe.latent_down")
R("CuteSkinnyGemm", "moe.latent_down", ["moe.latent_down"], "model", MOE + ": latent down-projection",
  before_name="single_group_topk_block_kernel")
R(SPLITK, "moe.latent_down", ["moe.latent_down"], "model", MOE + ": latent down-projection",
  before_name="single_group_topk_block_kernel")
R("cvt_fp16_to_fp4_sf_major", "moe.latent_quant", ["moe.latent_quant"], "model", MOE + ": latent NVFP4 quant")
ROUTED = [f"moe.routed_rank{i}" for i in range(16)]
for frag in ("routingIndices", "bmm_E2m1", "bmm_Bfloat16_E2m1", "finalizeKernel"):
    R(frag, "moe.routed", ROUTED, "model", MOE + ": routed experts (TRT-LLM NVFP4 fused MoE: routing, FC1, FC2, finalize)")
R("rms_norm_kernel", "moe.latent_norm", ["moe.latent_norm"], "model", MOE + ": latent RMSNorm",
  after_name="twoshotAllreduceKernel" if TP else "ncclDevKernel_ReduceScatter")
R(NV, "moe.latent_up", ["moe.latent_up"], "model", MOE + ": latent up-projection", after="moe.latent_norm")
R("CuteSkinnyGemm", "moe.latent_up", ["moe.latent_up"], "model", MOE + ": latent up-projection", after="moe.latent_norm")
R(VEC, "moe.combine_add", ["moe.combine_add"], "model", MOE + ": shared + routed combine add", after="moe.latent_up")
SH = MOE + ": shared experts"
for after_name in (SPLITK, NV, NV_FP32, "LLBf16Dotprod", "LLBf16SplitK", "CuteSkinnyGemm"):
    R(QUANT + "_object_at__tensorptrbf16" + Q_HIDDEN, "moe.shared_experts.gate_up", ["moe.shared_experts.gate_up"],
      "model", SH + " gate_up, MXFP8 quant + GEMM", after_name=after_name)
R(FI_GEMM, "moe.shared_experts.gate_up", ["moe.shared_experts.gate_up"], "model", SH + " gate_up, MXFP8 quant + GEMM",
  after="moe.shared_experts.gate_up")
R("situ_and_mul_kernel", "moe.shared_experts.situ_and_mul", ["moe.shared_experts.situ_and_mul"], "model",
  SH + " SiTU and mul", after="moe.shared_experts.gate_up")
R(QUANT + "_object_at__tensorptrbf16" + Q_SHARED_DOWN, "moe.shared_experts.down", ["moe.shared_experts.down"],
  "model", SH + " down, MXFP8 quant + GEMM")
R(FI_GEMM, "moe.shared_experts.down", ["moe.shared_experts.down"], "model", SH + " down, MXFP8 quant + GEMM",
  after="moe.shared_experts.down")

# ---------------- boundaries ----------------
if TP:
    AR = "flashinfer::trtllm_mnnvl_allreduce::twoshotAllreduceKernel"
    R(AR, "embedding_all_reduce", ["embedding_all_reduce"], "collective", "embedding TP all-reduce",
      sync=True, after="embedding")
    R(AR, "kda.attn_all_reduce", ["kda_dense.attn_all_reduce", "kda_moe.attn_all_reduce"], "collective",
      "KDA o_proj TP all-reduce", sync=True, after="kda.o_proj")
    R(AR, "mla.attn_all_reduce", ["mla_moe.attn_all_reduce"], "collective", "MLA o_proj TP all-reduce",
      sync=True, after="mla.o_proj")
    R(AR, "dense_ffn.ffn_all_reduce", ["kda_dense.ffn_all_reduce"], "collective", "dense FFN down TP all-reduce",
      sync=True, after="dense_ffn.down")
    MOE_AR = ("MoE TP all-reduces: the latent all-reduce ahead of the latent norm and the shared-expert "
              "all-reduce. They run back to back and which is which is not visible in the capture, so one "
              "operation owns both slots")
    R(AR, "moe.all_reduces", ["moe.latent_all_reduce", "moe.shared_all_reduce"], "collective", MOE_AR,
      sync=True, before_name="twoshotAllreduceKernel")
    R(AR, "moe.all_reduces", ["moe.latent_all_reduce", "moe.shared_all_reduce"], "collective", MOE_AR,
      sync=True, after_name="twoshotAllreduceKernel")
    FUSED = ("prefill o_proj as one GemmRsAr kernel (GEMM + reduce-scatter + all-reduce over multimem) with "
             "its MXFP8 input quant; the simulator prices it as GEMM then all-reduce, so the operation owns "
             "both slots")
    GRA = "gemm_rs_arSm100GemmRsAr"
    R(QUANT + "_object_at__tensorptrbf16" + Q_KDA_O, "kda.o_proj_gemm_rs_ar",
      ["kda.o_proj", "kda_dense.attn_all_reduce", "kda_moe.attn_all_reduce"], "collective", "KDA " + FUSED,
      sync=True, before_name=GRA)
    R(GRA, "kda.o_proj_gemm_rs_ar", ["kda.o_proj", "kda_dense.attn_all_reduce", "kda_moe.attn_all_reduce"],
      "collective", "KDA " + FUSED, sync=True, after_name=Q_KDA_O)
    R(GRA, "dense_ffn.down_gemm_rs_ar", ["dense_ffn.down", "kda_dense.ffn_all_reduce"], "collective",
      "dense FFN down as one GemmRsAr kernel (bf16 GEMM + all-reduce); priced as GEMM then all-reduce",
      sync=True, after="dense_ffn.situ_and_mul")
else:
    SP_AG = "sequence-parallel row all-gather ahead of attention (TP8 attention over the DP group's rows)"
    SP_RS_SLOTS = ["kda_dense.sp_reduce_scatter", "kda_moe.sp_reduce_scatter", "mla_moe.sp_reduce_scatter"]
    SP_RS = "sequence-parallel reduce-scatter after the attention o_proj / embedding"
    for frag in ("mnnvl_lamport_all_gather",):
        R(frag, "sp_all_gather", ["sp_all_gather"], "collective", SP_AG, sync=GROUP_SYNC)
    R("ncclDevKernel_AllGather", "sp_all_gather", ["sp_all_gather"], "collective", SP_AG, sync=GROUP_SYNC, after="attn_res")
    for frag in ("mnnvl_lamport_reduce_scatter", "mnnvl_multimem_reduce_scatter"):
        R(frag, "sp_reduce_scatter", SP_RS_SLOTS, "collective", SP_RS, sync=GROUP_SYNC)
    for frag in ("mnnvl_multimem_reduce_scatter", ):
        R("mnnvl_multimem_barrier_kernel", "sp_reduce_scatter", SP_RS_SLOTS, "collective", SP_RS,
          sync=GROUP_SYNC, before_name=frag)
        R("mnnvl_multimem_barrier_kernel", "sp_reduce_scatter", SP_RS_SLOTS, "collective", SP_RS,
          sync=GROUP_SYNC, after_name=frag)
    EMB = ("embedding reduction: vLLM all-reduces the vocab-parallel embedding (one-shot or two-shot "
           "kernel by size); the simulator prices it as the sequence-parallel reduce-scatter")
    for frag in ("oneshotAllreduceFusionKernel", "twoshotAllreduceKernel"):
        R(frag, "embedding.sp_reduce_scatter", ["embedding.sp_reduce_scatter"], "collective", EMB, sync=GROUP_SYNC,
          after="embedding")
    for kw in ({"after": "moe.latent_quant"}, {"before_name": "routingIndices"}):
        R("ncclDevKernel_AllGather", "moe.ep_all_gather", ["moe.ep_all_gather"], "collective",
          "EP16 dispatch all-gather of the NVFP4 latent rows", sync=True, **kw)
    for kw in ({"after": "moe.routed"}, {"before_name": "rms_norm_kernel"}):
        R("ncclDevKernel_ReduceScatter", "moe.ep_reduce_scatter", ["moe.ep_reduce_scatter"], "collective",
          "EP16 combine reduce-scatter of the routed-expert output", sync=True, **kw)

# ---------------- epilogue (sample phase) ----------------
R("rms_norm_kernel", "final_norm", ["final_norm"], "model", "final RMSNorm", phase="sample")
R(NV, "lm_head", ["lm_head"], "model", "target lm_head (vocab-parallel)", phase="sample")
R(SPLITK, "lm_head", ["lm_head"], "model", "target lm_head (vocab-parallel)", phase="sample")
U("ncclDevKernel_AllGather", "lm_head.logits_all_gather", "target logits all-gather (no simulated slot)", phase="sample",
  sync=GROUP_SYNC)

# ---------------- DSpark draft (one compute op + one collective op) ----------------
DRAFT_SLOTS = sorted({n[len(DRAFT):] for n in SLOTNAMES if n.startswith(DRAFT)
                      and not n.endswith(("all_reduce", "all_reduce_large"))})
DRAFT_ROLE = "DSpark draft: fc + context K/V, five SWA layers, head and seven Markov steps"
for frag in ("vocab_parallel_embedding_kernel", NV, SPLITK, "rms_norm_kernel", "rotary_embedding_kernel",
             "reshape_and_cache_flash_kernel", "FlashPrepareScheduler", "FlashAttentionForwardSm100",
             "FlashAttentionForwardCombine", "act_and_mul_kernel",
             "indexSelectSmallIndex", "_gumbel_sample_kernel", "_gumbel_combine_kernel"):
    R(frag, "draft.compute", ["draft." + s for s in DRAFT_SLOTS], "model", DRAFT_ROLE, phase="draft", excl=None)
DRAFT_COLL = sorted({n[len(DRAFT):] for n in SLOTNAMES if n.startswith(DRAFT)
                     and n.endswith(("all_reduce", "all_reduce_large"))})
for frag in ("Allreduce",):
    R(frag, "draft.all_reduces", ["draft." + s for s in DRAFT_COLL], "collective",
      "DSpark draft TP all-reduces (embedding, attention, MLP)", phase="draft", sync=GROUP_SYNC, excl=None)
U("ncclDevKernel_AllGather", "draft.logits_all_gather", "DSpark draft logits all-gather (no simulated slot)", phase="draft",
  sync=GROUP_SYNC)

json.dump({"schema_version": 2, "note": __doc__.strip(), "rules": rules}, sys.stdout, indent=1)
