"""Label rules for DeepSeek-V4.1-Flash on SGLang v0.5.23, 4x RTX PRO 6000 (SM120), TP4/EP4, DSpark.

Read off the folded launch programs of this pack's first nsys capture
(`logs/b2_reextract/07_balanced_anchor_c16/profile_nsys`, B2 rental 2026-11-06), checked against the
time-ordered per-stream timeline of one decode-graph iteration and one 2048-token prefill chunk. Not
migrated from the B200 vLLM pack: the SM120 SGLang kernel vocabulary is different.

Writes rules.json beside this file. Run from the repo root:
  uv run python presets/alignment/dsv41_flash_rtxpro6000_sglang_tp4_ep4/label_rules/gen_rules.py

How the capture is shaped, and why the rules look the way they do:

* Every dense FP8 linear is the same two launches (`per_token_group_quant_8bit_v2` then the Triton
  `_w8a8_block_fp8_matmul`, no shape in either name). The GEMM is named by the kernel AFTER it
  (`before_name`): the q-a RMSNorm follows the fused q/kv-a projection, the q RoPE follows wq_b, the
  SiLU follows the shared gate-up, the indexer's RoPE/FP4 quant follows the indexer wq_b, the Engram
  gate follows wkv, the final-norm spelling follows the DSpark kv-inject projection. The quant launch
  takes its GEMM's operation through `before`. Where a GEMM ends a body (decode-graph tracks), the
  chain runs the other way: the quant is named by the kernel before it and the GEMM by `after`.
* The decode CUDA graph is captured on ~100 streams, so the inventory splits one iteration into ~100
  tracks, and neighbours are track-local. A few rules key on a track-local neighbour that is not the
  temporal predecessor (marked "graph track"); they are exact for this capture and are the first thing
  to re-check on a new capture.
* Where the measured program cannot separate two simulated slots of identical shape, one operation
  claims both, instead of guessing: the attention-side and FFN-side TP all-reduce, mHC stats /
  combine / post (one side stream serves both sublayers), the target and DSpark lm_head / final norm /
  hc collapse, sparse MLA decode and prefill (one row-count switch; every launch in this capture took
  the prefill kernels, since a 16x7-row verify is above 64 rows), and the attention KV norm/RoPE store
  with the DSpark kv-inject store. Target and DSpark-stage layer slots share suffixes, so each layer
  operation also covers the three draft stages.
* Left unmapped on purpose (no simulated slot; mapping them would hide work on both sides):
  scheduler/metadata prologue kernels, the routed+shared residual add, `_mhc_stream_mean_kernel`
  (DSpark hidden-state capture on layers 37-39), verify/accept bookkeeping (argmax, softcap,
  `_fused_correct_drafts_and_bonus`, accept-length broadcasts), the kv-inject RMSNorm, the folded
  `[2x: quant, GEMM]` bodies on the draft side streams (one shared-down and one kv-a projection folded
  into one position), the prefill Engram id sort (cub radix sort in repeat bodies with no neighbours)
  and the torch bookkeeping between the indexer projections and the FP4 logits.
"""

import json
from pathlib import Path

SRC = "logs/b2_reextract/07_balanced_anchor_c16/profile_nsys (B2 4x RTX PRO 6000, 2026-11-06)"

# kernel-name fragments
Q8 = "per_token_group_quant_8bit_v2_kernel"
MM = "_w8a8_block_fp8_matmul"
AR = "ncclDevKernel_AllReduce_Sum_bf16"
AG = "ncclDevKernel_AllGather"
BC = "ncclDevKernel_Broadcast"
CUTLASS = "cutlass::Kernel2<cutlass_"  # dense BF16 GEMMs (tile in the name, shape not)
SPLITK = "cublasLt::splitKreduce_kernel"
RMS_QA = "oi641280"  # flashinfer RMSNorm over the 1280 q-lora latent (two alignments)
RMS_5120_16 = "gmemalign16oi645120i64"  # layer-0 / draft-stage hc-less input norm
RMS_5120_128 = "oi64512051201"  # final norm (target and draft) and the kv-inject norm
KNR = "sglang::fused_k_norm_rope_flashmla"
ROPE = "sglang::deepseek_rope_kernel"
SILU = "sglang::silu_mul_clamp_kernel"
RTQ = "_rope_tail_fake_quant_fp4_kernel"
ENGRAM_GATHER = "_engram_gather_kernel"
ENGRAM_GATE = "_engram_gate_kernel"
POST = "_mhc_post_split_h_kernel"
MIX_PARTIAL = "_hc_mix_stats_partial_kernel"
MHC_COMBINE = "_mhc_combine_kernel"
EMB = "_vocab_parallel_embedding_kernel"
ROUTER = "_router_triton_kernel"
FLASH_C = "sglang::flash_c"  # flash_c1_decode / flash_c2_decode compressor stores
INDEX_K = "sglang::index_k_kernel"
GATHER = "at::native::vectorized_gather_kernel"
VEC_EW = "at::native::vectorized_elementwise_kernel"
TORCH = "at::native::"
MARKOV_PARTIAL = "_markov_sample_partial_kernel"
MARKOV_FINISH = "_markov_sample_finish_kernel"

rules = []
OPS = {}


def op(operation, type_, role, slots, *, sync=False):
    OPS[operation] = slots

    def rule(name, note="", **evidence):
        r = {"operation": operation, "name": name, "type": type_, "role": role,
             "slot_suffixes": slots, "phase": "forward"}
        r.update(evidence)
        r["cross_rank"] = "synchronizing" if sync else "independent"
        if note:
            r["note"] = note
        r["source_experiment"] = SRC
        rules.append(r)
    return rule


def gemm(operation, role, slots, *, before_names=(), after_names=(), chain=False, note=""):
    """One FP8 linear (quant + Triton GEMM), named by its neighbours."""
    rule = op(operation, "model", role, slots)
    for nxt in before_names:
        rule(MM, before_name=nxt, note=note)
    rule(Q8, before=operation)
    for prev in after_names:
        rule(Q8, after_name=prev, note=note)
    if chain or after_names:
        rule(MM, after=operation)
    return rule


# ---------------------------------------------------------------- prologue
emb = op("prologue.embedding", "model", "token embedding lookup (target forward and DSpark draft)",
         ["prologue.embedding", "dspark.embed"])
emb(EMB, note="One launch per forward; the DSpark draft re-embeds its block with the same kernel.")
op("prologue.embedding_all_reduce", "collective", "vocab-parallel embedding all-reduce",
   ["prologue.embedding_all_reduce"], sync=True)(
    AR, after_name=EMB,
    note="Both embeddings (target and draft) are followed by this all-reduce; the draft one has no "
         "separate slot, so the measured side carries two launches per speculative iteration.")
op("prologue.hc_broadcast", "model", "broadcast the embedding into the 4 hc streams",
   ["prologue.hc_broadcast"])("_hc_broadcast_kernel")
op("prologue.engram_hash", "model", "Engram n-gram hash", ["prologue.engram_hash"])("_engram_hash_kernel")

# ---------------------------------------------------------------- collectives
tp_ar = op("tp_all_reduce", "collective", "TP4 all-reduce closing an attention or FFN sublayer (NCCL ring LL)",
           ["attn.tp_all_reduce", "ffn.tp_all_reduce"], sync=True)
tp_ar(AR, before_name=POST,
      note="Decode graph: each sublayer's all-reduce opens a new stream/track, so attention vs FFN is "
           "not observable from neighbours. Both slots have the same tokens x 5120 bf16 shape.")
tp_ar(AR, before_name=MIX_PARTIAL, note="Eager prefill: mix stats run inline after the all-reduce.")

# ---------------------------------------------------------------- mHC
mix = op("mhc_mix_stats", "model", "hc mix stats + sinkhorn (side stream in decode, inline in prefill)",
         ["attn.mhc_mix_stats", "attn.mhc_mix_stats.serial", "ffn.mhc_mix_stats", "ffn.mhc_mix_stats.serial"])
mix(MIX_PARTIAL)
mix("_hc_mix_reduce_sinkhorn_kernel")
comb = op("mhc_combine_norm", "model", "hc combine + RMSNorm before a sublayer",
          ["attn.mhc_combine_norm", "ffn.mhc_combine_norm"])
comb("_hc_combine_norm")
comb(RMS_5120_16, note="Layer 0 (after hc_broadcast) and each DSpark stage's first sublayer: the hc "
                       "copies are identical, so SGLang runs a plain RMSNorm instead of _hc_combine_norm.")
post = op("mhc_post", "model", "hc post (residual split back into 4 streams)", ["attn.mhc_post", "ffn.mhc_post"])
post(POST)
post("mhc_post_tilelang_kernel", note="The eager (prefill) spelling of the same op.")

# ---------------------------------------------------------------- attention
gemm("attn.fused_wqa_wkv", "fused q-a + kv-a projection (FP8 block-32)", ["attn.fused_wqa_wkv"],
     before_names=[RMS_QA], after_names=[RMS_5120_16],
     note="The q-a RMSNorm follows it. Draft stages issue q-a and kv-a as two GEMMs on two streams; "
          "both are this operation.")
op("attn.qkv_rmsnorm", "model", "q-a latent RMSNorm", ["attn.qkv_rmsnorm"])(RMS_QA)
gemm("attn.wq_b", "q-b projection (FP8 block-32)", ["attn.wq_b"], before_names=[ROPE])
rope_store = op("attn.rope_kv_store", "model", "q RoPE, kv norm + RoPE + paged FP8 store",
                ["attn.rope_kv_store", "kv_inject.store"])
rope_store(KNR, note="Also the three DSpark kv-inject stores (one per stage), which the folded graph "
                     "places in neighbour-less repeat bodies; one operation claims both slots.")
rope_store(ROPE, after_name=MM, note="q RoPE right after wq_b.")
op("attn.inv_rope", "model", "inverse RoPE on the attention output", ["attn.inv_rope"])(
    ROPE, before_name=CUTLASS)
pm = op("attn.page_metadata", "model", "SWA / compressed page mark + split", ["attn.page_metadata"])
pm("_page_mark_kernel")
pm("_page_split_kernel")
op("attn.sparse_mla", "attention", "FlashInfer SM120 sparse MLA (decode <=64 rows, prefill kernels above)",
   ["attn.sparse_mla_decode", "attn.sparse_mla_prefill"])(
    "sparse_mla_", note="sparse_mla_prefill_mg(_dual)_kernel / sparse_mla_decode_dsv4. The model picks "
                        "one of its two slots per row count, so one operation claims both.")
op("attn.wo_a", "model", "grouped o-a projection (BF16 CUTLASS bmm)", ["attn.wo_a"])(
    CUTLASS, after_name=ROPE, note="The GEMM right after the inverse RoPE.")
wo_b = gemm("attn.wo_b", "o-b projection (FP8 block-32)", ["attn.wo_b"],
            before_names=[AR], chain=True,
            note="Prefill: the all-reduce follows it. Decode graph: it ends the attention track, so the "
                 "quant is named by wo_a before it (`after`).")
wo_b(Q8, after="attn.wo_a")

# compressor (index layers)
comp = op("attn.compressor_kv_score", "model", "compressor kv/score projection (BF16 CUTLASS, split-K)",
          ["attn.compressor_kv_score"])
comp(SPLITK, before_name=FLASH_C)
comp(CUTLASS, before="attn.compressor_kv_score")
comp(CUTLASS, before_name=FLASH_C, note="Prefill: no split-K, the store follows the GEMM directly.")
cs = op("attn.compressor_store", "model", "compressor state update + compressed KV store",
        ["attn.compressor_store"])
cs(FLASH_C)
cs("sglang::c2_prefill_store_kernel")

# ---------------------------------------------------------------- indexer (index layers)
wk = op("indexer.wk_store", "model", "indexer k projection + FP4 k store", ["indexer.wk_store"])
wk(INDEX_K)
wk(CUTLASS, before_name=INDEX_K, note="The BF16 k projection runs right before index_k.")
gemm("indexer.wq_b", "indexer q-b projection (FP8 block-32)", ["indexer.wq_b"], before_names=[RTQ])
op("indexer.q_rope_quant", "model", "indexer q RoPE + FP4 fake quant", ["indexer.q_rope_quant"])(RTQ)
wp = op("indexer.weights_proj", "model", "indexer head-weight projection (BF16 CUTLASS, split-K)",
        ["indexer.weights_proj"])
wp(CUTLASS, after_name=RTQ)
wp(SPLITK, after="indexer.weights_proj")
op("indexer.logits_decode", "model", "FP4 indexer logits (decode/verify)", ["indexer.logits_decode"])(
    "_fp4_index_logits_kernel")
op("indexer.logits_prefill", "model", "torch indexer logits einsum (prefill, per request)",
   ["indexer.logits_prefill"])(
    CUTLASS, after_name=GATHER, before_name=VEC_EW,
    note="The BF16 einsum over the gathered index K; the lm_head is the other GEMM after a gather and "
         "is followed by the logits all-gather instead.")
topk = op("indexer.torch_topk", "model", "torch top-k (radix select) over the index logits",
          ["indexer.torch_topk"])
for frag in ("computeBlockDigitCounts", "computeDigitCumSum", "computeBlockwiseWithinKCounts",
             "computeBlockwiseKthCounts", "gatherTopK", "DeviceScanByKeyInitKernel", "DeviceScanByKeyKernel"):
    topk(frag, note="torch.topk internals; in this capture torch.topk runs only in the indexer.")
topk("radixSortKVInPlace", before_name=TORCH, note="The sort of the selected indices (the Engram id "
                                                   "sort is the one followed by the gather).")

# ---------------------------------------------------------------- Engram (layers 1, 14)
lk = op("engram.lookup", "model", "Engram host-table gather (+ id sort)", ["engram.lookup"])
lk(ENGRAM_GATHER)
lk("radixSortKVInPlace", before_name=ENGRAM_GATHER)
gemm("engram.wkv", "Engram wkv projection (FP8 block-32)", ["engram.wkv"], before_names=[ENGRAM_GATE])
op("engram.gate", "model", "Engram gate", ["engram.gate"])(ENGRAM_GATE)

# ---------------------------------------------------------------- FFN
rt = op("ffn.router", "model", "router GEMM (BF16 CUTLASS, split-K) + top-k", ["ffn.router"])
rt(ROUTER)
rt(SPLITK, before_name=ROUTER)
rt(CUTLASS, before="ffn.router")
gemm("ffn.shared_gate_up", "shared expert gate-up (FP8 block-32, TP-sharded)", ["ffn.shared_gate_up"],
     before_names=[SILU])
op("ffn.shared_act", "model", "shared expert SiLU-mul clamp", ["ffn.shared_act"])(SILU)
gemm("ffn.shared_down", "shared expert down (FP8 block-32)", ["ffn.shared_down"],
     before_names=[CUTLASS], after_names=[SILU],
     note="Prefill: the router GEMM follows it. Decode graph: it ends the shared-expert track, so it is "
          "named by the SiLU before its quant.")
moe = op("ffn.routed_moe", "moe", "EP4 routed experts: MXFP8 act quant + TRT-LLM CUTLASS MXFP4 grouped GEMM",
         ["ffn.routed_moe"])
for frag in ("quantize_with_block_size", "ExpertPrefixSumKernel", "fusedBuildExpertMapsSortFirstTokenKernel",
             "expandInputRowsKernel", "computeStridesTmaWarpSpecializedKernel", "GroupProblemShape",
             "doActivationKernel", "finalizeMoeRoutingKernel"):
    moe(frag)

# ---------------------------------------------------------------- head (target and DSpark draft)
op("head.hc_collapse", "model", "hc collapse before the final norm", ["head.hc_collapse", "dspark.hc_collapse"])(
    MHC_COMBINE)
op("head.final_rmsnorm", "model", "final RMSNorm", ["head.final_rmsnorm", "dspark.final_rmsnorm"])(
    RMS_5120_128, after_name=MHC_COMBINE)
lm = op("head.lm_head", "model", "lm_head (BF16 CUTLASS, vocab / 4)", ["head.lm_head", "dspark.lm_head"])
lm(CUTLASS, after_name=RMS_5120_128, note="Decode: right after the final norm. Target and draft lm_head "
                                          "are the same GEMM; one operation claims both slots.")
lm(CUTLASS, before_name=AG, note="Prefill: last-token gather, then the GEMM, then the logits all-gather.")
lm("internal::gemvx::kernel", before_name=AG, note="cuBLAS gemv spelling for a one-row lm_head.")
lag = op("head.logits_all_gather", "collective", "logits all-gather", ["head.logits_all_gather"], sync=True)
lag(AG, after_name=CUTLASS)
lag(AG, after_name="internal::gemvx::kernel")
lag(AG, after_name=ROPE, note="graph track: in the decode graph the logits all-gather shares a stream "
                              "with layer 0's q path, so its track-local predecessor is the q RoPE.")

# ---------------------------------------------------------------- DSpark draft tail
gemm("dspark.kv_inject.kv_proj", "DSpark kv-inject projections of the target hidden states (FP8)",
     ["kv_inject.kv_proj"], before_names=[RMS_5120_128], after_names=[RMS_5120_128])
op("dspark.markov.w", "model", "Markov head low-rank projection (BF16 CUTLASS)", ["markov.w1", "markov.w2"])(
    CUTLASS, before_name=MARKOV_PARTIAL)
op("dspark.markov.sample_partial", "model", "Markov sampling, partial", ["markov.sample_partial"])(MARKOV_PARTIAL)
op("dspark.markov.sample_finish", "model", "Markov sampling, finish", ["markov.sample_finish"])(MARKOV_FINISH)
vag = op("dspark.markov.vocab_all_gather", "collective", "Markov vocab all-gather", ["markov.vocab_all_gather"],
         sync=True)
vag(AG, after_name=MARKOV_PARTIAL)
vag(AG, after_name=MM, note="graph track: round 1's all-gather shares a stream with the last draft "
                            "stage's wo_b, so that GEMM is its track-local predecessor.")
op("dspark.markov.token_broadcast", "collective", "Markov token broadcast", ["markov.token_broadcast"],
   sync=True)(BC, after_name=MARKOV_FINISH)

out =Path(__file__).with_name("rules.json")
out.write_text(json.dumps({"rules": rules}, indent=1) + "\n")
print(f"wrote {len(rules)} rules for {len(OPS)} operations to {out}")
