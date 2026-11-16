# deepseek_v41_sglang: graph design (TP4 attention, EP4 experts, DSpark, SM120)

The source of truth for op order is the SGLang v0.5.23 torch-profiler traces (decode batch 1, layer segments between
all-reduces). Shapes follow the vLLM arch's `DeepseekV41ModelCfg` and the shape math in its worklets; only the
SGLang differences are spelled out here. Kinds and backends are the ones committed on this branch (`SPLIT.md`).

## Model facts used (`model/config/deepseek_v41_flash.json`)

- **Size:** hidden 5120; 40 body layers; 64 heads, head_dim 512, rope 64 (MQA: one 512-wide latent KV).
- **Low-rank projections:** q_lora 1280; o_lora 1024 in 8 groups.
- **Indexer:** 32 heads x 128; top-k 512; index-source layers 2, 8, 14, 20, 24, 28, 32, 36.
- **Compression and SWA:** compress ratio 0 for layers 0-1, 2 for 2-19, 1 for 20-39; sliding window 128.
- **Candidate blocks:** source layer 20; 2048 blocks of 8.
- **MoE:** 384 routed experts, top-6, intermediate 2304; 1 shared expert.
- **hc:** hc_mult 4; 20 sinkhorn iterations.
- **Engram:** layers 1 and 14; 8 heads x 256 (24 local heads after the n-gram split, per the runner's report); max
  n-gram 4.
- **DSpark:** 3 stages fed from target layers 37-39. Each is a decoder layer: DSparkAttention (MQA) + MoE with 128
  experts, top-3. Markov head rank 256. The server ran block 6 (7 verify rows).

## Iteration (one TP rank; every rank holds every token after the attention all-reduce)

```text
prologue: embedding (+ all-reduce), engram hash, hc broadcast
for layer in 0..40:
  [Engram block before layers 1 and 14]
  attention sublayer (TP4)
  FFN sublayer (EP4 routed + shared)
head: hc collapse (mhc_combine), final RMSNorm, lm_head, logits all-gather
speculative only: DSpark draft (3 stages) + sampling chain
```

## Attention sublayer (per layer)

| # | Op | Kind / backend | Shape (TP4 rank) | Notes |
|---|---|---|---|---|
| a1 | hc mix stats + sinkhorn | `mhc_mix_stats_sinkhorn` / `sglang_triton` | (tokens, 5120, hc 4, 20 iters) | side stream in decode/verify; overlap with a2 |
| a2 | hc combine + RMSNorm | `mhc_combine_rms_norm` / `sglang_triton` | (tokens, 5120, 4) | |
| a3 | fused q-a + kv-a projection | `single_gemm` / `sglang_triton_block32_fp8` | m = tokens, k = 5120, n per vLLM `input.fused_wqa_wkv` | includes act quant |
| a4 | q-a RMSNorm, kv RMSNorm + RoPE | `elementwise` placeholders (FlashInfer RMSNorm, `fused_k_norm_rope_flashmla`) | | small |
| a5 | q-b projection | `single_gemm` / `sglang_triton_block32_fp8` | m = tokens, k = 1280, n = 16 heads x 512 | |
| a6 | q RoPE, KV store, page metadata | `elementwise` placeholders | | `deepseek_rope`, `_page_mark` / `_page_split` |
| a7 | indexer (index layers only) | `single_gemm` (wq/wk) + `sparse_indexer_logits_{decode,prefill}` / `sglang_{triton,torch}_fp4` + `logits_topk` | decode: (batch, next_n 1 or 7, num_keys = ctx/ratio, 32 x 128); prefill: per request | candidate-block selection not modeled yet |
| a8 | sparse MLA | `paged_sparse_mla` / `sglang_flashinfer_sm120` | (requests, rows/request, context_before, 16 heads, 512, swa 128, ratio, extra index 512, page 256, v4 layout) | one kind for decode, verify and prefill (switches at 64 rows) |
| a9 | inverse RoPE | `elementwise` | | |
| a10 | o-a grouped projection | `elementwise` placeholder (`_wo_a_partial` + `_wo_a_reduce`) | | 1.7% at bs1; promote after Phase 5 |
| a11 | o-b projection | `single_gemm` / `sglang_triton_block32_fp8` | m = tokens, k per vLLM `output.wo_b` | |
| a12 | TP all-reduce | `all_reduce` / `nccl` | tokens x 5120 bf16, 4 ranks | unfused: no residual/norm fusion on SM12x |
| a13 | hc post | `mhc_post` / `sglang_auto` | (tokens, 5120, 4) | |

## FFN sublayer (per layer)

| # | Op | Kind / backend | Shape | Notes |
|---|---|---|---|---|
| f1 | hc mix stats + sinkhorn | `mhc_mix_stats_sinkhorn` | as a1 | side stream |
| f2 | hc combine + RMSNorm | `mhc_combine_rms_norm` | as a2 | |
| f3 | router GEMM + top-k | `elementwise` placeholders (`tiny_n_gemm`, `_router_triton_kernel`) | | <= 1.2% |
| f4 | shared expert gate-up | `single_gemm` / `sglang_triton_block32_fp8` | m = tokens, k = 5120, n = 2 x 2304 / 4 = 1152 | TP-sharded (`DeepseekV2MLP(parallel_group="tp")`; replicated only with an all-to-all MoE backend or `SGLANG_SHARED_EXPERT_TP1`) |
| f5 | SiLU-mul clamp | `elementwise` | | |
| f6 | shared expert down | `single_gemm` / `sglang_triton_block32_fp8` | m = tokens, k = 576, n = 5120 | `reduce_results=False`: its partial sum joins the f8 all-reduce |
| f7 | routed experts | `nvfp4_fused_moe` / `flashinfer_cutlass_sm120_mxfp4` | tokens, 5120, 2304, 384 global / 96 local, top-6, mxfp4 group 32, `precomputed_dsv4`, per-expert demand from `token_corpus` | includes MXFP8 act quant; EP rank 0 |
| f8 | TP all-reduce | `all_reduce` / `nccl` | tokens x 5120 bf16 | |
| f9 | hc post | `mhc_post` | | |

## Engram block (layers 1 and 14)

`engram_lookup` / `sglang_triton` (host table, residency `host_shared_huge` once huge pages are set), then `wkv` as
`single_gemm` / `sglang_triton_block32_fp8` (m = tokens, n = 25600, k = 6144), then `engram_gate` /
`sglang_triton`.

## Head and DSpark

- **Head:**
  - `mhc_combine` (placeholder) and RMSNorm;
  - `lm_head` as `single_gemm` / `torch_cublas`, bf16, n = 129280 / 4;
  - logits `all_gather` (NCCL).
- **DSpark draft, speculative iterations:**
  - 3 stages, each a decoder layer built from the same worklets with the draft geometry (128 experts, top-3);
  - on the target layout (`DraftTPExecution.TARGET_LAYOUT`).
- **Sampling chain, 6 rounds of:**
  - `_markov_sample_partial` + `all_gather` + `_markov_sample_finish` + `broadcast`;
  - `elementwise` placeholders plus NCCL rows;
  - a new small `broadcast` comm kind, or `all_gather` rows at the broadcast size, until it ranks.

## Worker and KV

- **Worker:** `speculative`, with:
  - `draft_tokens: 6`;
  - `batch_policy: separate-prefill-priority`;
  - `max_batch_tokens: 2048`;
  - `max_running_requests: 16`;
  - `kv_admission_policy: bounded-future` (the GLM-5.2 SGLang pack's fields);
  - acceptance from the B2 `workload_metrics` pass (`prepare-workload`).
- **KV:** SGLang's DSV4 pools at `--mem-fraction-static 0.86`:
  - full 2,929,152 tokens;
  - swa 49,664 (cap mode);
  - c4 732,288;
  - c128 22,884;
  - 1,673.31 bytes per full token.

  These need a `kv_layout` builder for the SGLang V4 layout on SM12x (584-byte rows, 256-token pages).

## Open items before code

1. **Shared-expert sharding:** resolved, TP-sharded (`models/deepseek_v2.py` `DeepseekV2MoE.__init__`).
2. **Index layer, from the decode bs1 trace** (81 kernels between the two all-reduces; ratio-2 layer):
   1. hc pre;
   2. fused q/kv-a FP8 GEMM;
   3. RMSNorm;
   4. q-b FP8 GEMM;
   5. RoPE, fused k-norm + RoPE;
   6. **compressor:** a BF16 CUTLASS GEMM + split-K reduce, then `sglang::flash_c2_decode_kernel`;
   7. **indexer projections:** `sglang::n128k512_kernel`, then `sglang::index_k_kernel`;
   8. indexer FP8 GEMM, `_rope_tail_fake_quant_fp4`, `sglang::n32k5120_kernel` (head weights);
   9. **about 20 torch elementwise / index / copy kernels;**
   10. `_fp4_index_logits`;
   11. **about 30 torch top-k / scan / radix-sort kernels;**
   12. page mark/split, sparse MLA decode + merge, RoPE, `wo_a` partial + reduce, wo_b FP8 GEMM.

   Non-index layers skip steps 7-11. So the torch indexer bookkeeping and top-k account for much of the plumbing
   share at decode. Model them as an indexer-overhead op per index layer (`elementwise`, or a dedicated row taken
   from the nsys capture) instead of leaving them unattributed.
3. **Compressor ops (ratio 1 and 2 layers).** A BF16 GEMM (`single_gemm` / `torch_cublas`; shape from vLLM
   `compressor.kv_score_proj`) plus `flash_c2_decode_kernel` (decode) / `c2_prefill_store_kernel` (prefill).
   Placeholders first.
4. **What the bs8 decode window's 30% elementwise share is.** Needs the B2 nsys capture.
