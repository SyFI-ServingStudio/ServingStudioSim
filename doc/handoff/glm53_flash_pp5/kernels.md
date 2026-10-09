# Handoff: kernels of the simulated GLM-5.3-Flash NVFP4 PP5 B200 deployment

Scope: what one pipeline stage runs, how long each kernel takes at the operating point,
how to look the same kernel up in the profile DB, and where the real kernel call lives.
All facts were read on 2026-10-08 from the study code and re-checked on ServingStudioSim branch `pp-sched-tiers`
(draft PR #84), which carries this directory. Relative paths are from the ServingStudioSim root.
Numbers are copied as read. Anything inferred rather than read is marked **(inferred)**.

Reference run: `logs/glm53_flash_pp5/runs_8h/c3000` (called `$R` below), written by `pp5_8h.yaml` in this
directory. Its per-stage cost manifests are the same for every load and duration, so the 16-minute
`pp5_16min.yaml` run gives the same `raw/cost_manifest/` in minutes.

---

## 0. The deployment being modelled

From `$R/raw/run_config.yaml`:

| field | value |
|---|---|
| deployment | `pp`, one replica, 5 x `NVIDIA B200` |
| arch.type | `glm53_flash_vllm_nvfp4_pp_kda_dsa_moe`, `model_config: model/config/glm53_flash_nvfp4.json`, `fp8: false`, `pp_size: 5`, `max_model_len: 1048576` |
| kernel path ("fastk") | per-role `backends_file: backends.yaml`: KDA prefill `flashinfer_cute_persistent`, mHC fused boundary `deepgemm_mega_nonshifted`, short conv `dao_channellast`, top-k prefill `[deep_select, vllm_cuda]` (faster per launch). Arch knobs `mla_layout_copies: false`, `indexer_max_logits_mb: 512` |
| routing | `routing: corpus`, `token_corpus_file: hf://datasets/UW-SyFI/servingstudio-workload@c3f5ecaab0bbff757c64864bf2ec24f5e1f76b5c/glm53_flash_fp8/vllm/balanced_c32/capture/20260924/manifest.json` |
| cudagraph_capture_sizes | 1, 2, 4, 8, 16, 24, ... (step 8 to 256, then step 16 up to the list end) |
| worker | `pipeline_chunked_prefill`, `max_batch_tokens: 32768`, `prefill_chunk_alignment: plain`, `external_decode: true`, `microbatch_split: even`, `min_microbatch_tokens: 512`, `pending_order: shortest-prefill-first`, `srpt: true`, load-budget backlog 3,000,000 / 1,000,000 tokens, `attn_gpu_memory_gb: 108.96` |
| prefix tiers | DRAM 150 GB at 50 GB/s, SSD 8000 GB at 10 GB/s, warm start, `prefix_tier_max_read_wait_ms: 1000` |
| workload | `logs/glm53_flash_pp5/traces/closed_c3000.csv` (`make_traces.sh`), saturated closed loop, `max_concurrency: 3000`, 28,800,000 ms (8 h) |

**Important for the real system: this run is prefill-only.** `external_decode: true` means a request
completes at its first token "as if a decode instance took it"
(`simulator/src/worker/workers/pipeline/pipeline_head_worker.rs:30-36`). In stage 3's cost log every
iteration has `decode_request_count = 0`, so all decode leaves (`sparse_mla.decode`,
`mqa_logits_decode`, `topk_decode`, `kpool_decode_update`, `short_conv_decode`, `recurrent_decode`)
never run. The analyzer lists them under "never executed" in
`$R/reports/kernel_input_distribution_report.json` (`positions_omitted`).

Headline numbers (`$R/summary.json`): `prefill_tok_s` 182574.05, `total_tok_s_per_gpu` 36536.01,
`completed_req_s` 106.02. Busy fraction per stage (`$R/reports/utilization_report.json`,
`totals.per_worker[].avg_util`): stage 0 0.90473, stage 1 0.94147, stage 2 0.94146,
**stage 3 0.99490**, stage 4 0.94657, pool mean 0.94583.

Operating point of the bottleneck stage 3 (computed from `$R/raw/cost_log/worker_stage_3.parquet`,
192,866 microbatch iterations, `groups` column):

| quantity | mean | p10 | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| stage-3 microbatch time (ms, `total_time_ms`) | 148.6 | 107 | 154 | 181 | 193 | 262 |
| tokens per microbatch (all prefill) | 27263.4 | 19696 | 28369 | 32768 | 32768 | 32768 |
| prefill sequences per microbatch | 16.8 | 5 | 17 | 27 | 47 | 600 |
| per-sequence prefix (cached context) length | 148930.0 | 29550 | 117852 | 258618 | 834315 | 999698 |
| per-sequence append (new tokens this chunk) | 1620.9 | 40 | 237 | 4636 | 18071 | 32768 |

The sum of `total_time_ms` is 28653253.73 ms over a 28800041.96 ms span, which is the 99.49%.

---

## 1. CostTree structure

### 1.1 Where the tree comes from

- Arch wire file: `simulator/src/arch/glm53_flash_vllm_fp8_pp_kda_dsa_moe.rs`. One file serves both
  the FP8 and the NVFP4 tags. The NVFP4 tag is `arch/config.rs:865` (dispatched at `arch/build.rs:2216`), the module doc is at lines 1-45,
  `stage_ranges` is at line 94, the per-stage model `Glm53FlashVllmFp8PpStageModel::build` is at line 411,
  the `"{} x{count} (Scale {count}) layers {layers:?}"` label is at line 512, and
  `activation_bytes_per_token` is at line 718.
  The stage label says `Glm53FlashVllmFp8PpStageModel` even for NVFP4. That is the shared type name.
  The precision comes from the checkpoint's `quantization_config`.
- The leaves are the TP=EP graph's leaves, built at `tp_size = 1` without all-reduces:
  `simulator/src/arch/glm53_flash_vllm_fp8_kda_dsa_moe.rs`. The module doc at lines 1-65 covers the
  structure. Backend constants are at lines 116-160, `Glm53FlashKernelPath` (the per-role backend choice) at
  lines 480-500, `build_configs` at line 605, the shared-expert `Parallel` at line 895, `build` at
  line 1254, and `cost_tree` at line 1378.
- Worklets (L3): `simulator/src/worklet/glm53_dsa_attn_local.rs`, `glm53_kda_attn_local.rs`,
  `glm53_moe_local.rs` (router and routed experts), and `glm53_mlp_local.rs` (shared expert and
  dense FFN). Their module docs give the launch order. The kpool sparse-MLA compound op is
  `simulator/src/op/attention/glm53_kpool_sparse_mla.rs`.
- Per-stage trees as compiled for the run: `$R/raw/cost_manifest/worker_stage_{0..4}.json`. Each holds
  one section `iter` with `nodes` (flat BFS `Sum` / `Scale{n}` / `Max{overlap}` / `Parallel{overlap}` /
  `Leaf(slot)`), `node_labels`, and `slots[]` (`name`, `kind`, `kernel_config` = backends, gpu_name, and
  static shape params).
- Node algebra: `simulator/src/timing/COST_TREE.md`. `Sum` adds the times of serial children.
  `Scale{n}` multiplies one subtree by n. `Max` (EP ranks) and `Parallel` (aux stream) take the
  maximum child time divided by `overlap`. Flops and bytes always sum.

### 1.2 Layer types and stage split

Model (`model/config/glm53_flash_nvfp4.json`): 45 layers, hidden 4096, mHC `hc_mult` 4.
- The 11 DSA (MLA + kpool indexer) layers are `full_attn_layers` [3, 7, 11, ..., 43]. The other 34
  are KDA linear-attention layers.
- Attention: 64 heads, `q_lora_rank` 1536, `kv_lora_rank` 512, `qk_nope_head_dim` 256 (no rope),
  `v_head_dim` 256. Indexer: 32 heads x 128, `index_topk` 2048, `index_kpool` 4.
- KDA: 64 heads x 128, short conv 4.
- FFN: layers 0-2 are dense with intermediate 12288. Layers 3-44 are MoE with 288 routed experts,
  top-8, `moe_intermediate_size` 2048, 1 shared expert, sigmoid `noaux_tc`, routed scaling 2.5.
- NVFP4 (ModelOpt) applies to the dense FFN and the routed experts. Attention, routers, the shared
  experts and lm_head stay BF16. The KV cache is FP8.

The split is [9, 9, 9, 9, 9], 9 consecutive layers per stage. Verified from `node_labels`:

| stage | layers | composition (Scale groups) | extras |
|---|---|---|---|
| 0 | 0..9 | `first_kda_dense` x1 [0], `kda_dense` x2 [1,2], `dsa_moe` x2 [3,7], `kda_moe` x4 [4,5,6,8]; 7 KDA + 2 DSA | `pp.embedding`, `pp.hc_expand`; layer 0 opens with `mhc_pre_rms_norm` (vllm_tilelang) |
| 1 | 9..18 | `dsa_moe` x2 [11,15], `kda_moe` x7 [9,10,12,13,14,16,17] | - |
| 2 | 18..27 | `dsa_moe` x2 [19,23], `kda_moe` x7 [18,20,21,22,24,25,26] | - |
| **3** | 27..36 | **`dsa_moe` x3 [27,31,35], `kda_moe` x6 [28,29,30,32,33,34]**; 6 KDA + 3 DSA | - (this is why it is the bottleneck) |
| 4 | 36..45 | `dsa_moe` x2 [39,43], `kda_moe` x7 [36,37,38,40,41,42,44] | head: `pp.final_mhc_post` (mhc_fused, vllm_tilelang), `pp.hc_contract_mean`, `pp.final_norm` (rms_norm vllm_cuda), `pp.lm_head` (single_gemm n=154880 k=4096 bf16) |

A stage has no collectives (TP1/EP1). The stage-to-stage activation send is not a CostTree leaf. It is a
cluster transfer (`gpu_cluster.rs:360 submit_transfer`, kind `"pp_activation"`) that the follower pulls
while it computes the previous microbatch (`pipeline_stage_worker.rs:1-16`, `:131-150`). The payload is
`activation_bytes_per_token` = 4·4096·2 = 32768 B/token (arch line 718, test at line 1037). It therefore
never appears in busy time or kernel shares.

### 1.3 Stage-3 tree (75 leaves; stages 1/2/4 have the same layer subtrees with different Scale counts)

`Leaf[i]` is the slot index into the per-iteration buffer and into `slot_*` columns in `cost_log`.
"Dynamic key" is the per-iteration input field logged in `slot_input` (and named in
`kernel_input_distribution_report.json` `features_used`). Static params come from `kernel_config`.

```
Sum  stage 3 of 5 [layers 27..36: 6 KDA + 3 DSA]
  Scale 3  dsa_moe [27,31,35]
    Sum
      L0  attn_mhc_post_pre
      Sum dsa (Glm53DsaAttnLocalWorklet) [H=64, index heads=32 replicated, kpool top-2048 x 4]
        L1..L22  projections + indexer + q prep           (Scale 2 on L11 prefill_gather)
        Sum sparse_mla: L23 mla_cache_append, L24 index_remap, L25 prefill, L26 decode
        L27 output_copy, L28 v_up, L29 o_proj, Scale 17 {L30 glue}, Scale 8 {L31 prefill_glue}
      L32 ffn_mhc_post_pre
      Sum moe [shared expert overlaps routed at T<=256]
        Sum router: L33 gate, L34 gate_recompute
        L35 input_glue
        Max{1.0} ep_ranks (one EP rank)
          Sum
            Parallel{0.9} [ Sum shared_expert L36-38 , Sum routed_rank0 L39-40 ]   <- used when T<=256
            Sum routed_rank0 L41 input_quant, L42 fused_moe                         <- used when T>256
            Sum shared_expert L43 gate_up, L44 act_and_mul, L45 down               <- used when T>256
        L46 combine_glue
  Scale 6  kda_moe [28,29,30,32,33,34]
    Sum
      L47 attn_mhc_post_pre
      Sum kda (Glm53KdaAttnLocalWorklet) [H=64, D=128]
        L48 in_proj, L49 f_b_proj, L50 g_b_proj, L51 short_conv_prefill, L52 short_conv_decode,
        Scale 0 {L53 prefill_glue}, L54 state_gather, L55 chunk_prefill, L56 recurrent_decode,
        L57 state_scatter, L58 gated_norm, L59 o_proj
      L60 ffn_mhc_post_pre
      Sum moe  (same shape as the DSA layer's: L61-L74)
```

Only one of the two shared-expert copies is filled in each iteration (the unused copy logs backend 255
"not run"). The threshold is `SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD = 256` and the overlap is 0.9
(`glm53_flash_vllm_fp8_kda_dsa_moe.rs:107`, `:114`). At this operating point T > 256 always holds, so
the serial copy runs. `Scale 0` on KDA `prefill_glue` (the beta-sigmoid placeholder) is zero because the
flashinfer/flashkda prefill takes beta logits and applies the sigmoid itself
(`takes_beta_logits` at `simulator/src/timing/kernels/kda_chunk_prefill.rs:73`; the Scale count is
`beta_sigmoid_launches = u32::from(!backends_agree(..))` at `simulator/src/worklet/glm53_kda_attn_local.rs:291`).

### 1.4 Per-leaf table (stage 3 slots; DSA layer then KDA layer)

Backend column: the DB `backend` string, then the library or callable it times (from each backend's
registry `doc.summary`, see section 4).

| slot | leaf name | kind | backend → real callable | static key | dynamic key |
|---|---|---|---|---|---|
| 0,32,47,60 | `*.attn_mhc_post_pre`, `*.ffn_mhc_post_pre` | mhc_fused_post_pre_rms_norm | `deepgemm_mega_nonshifted` → one DeepGEMM `mega_mhc` launch without shifted collapse (post, TF32 pre GEMM, Sinkhorn, collapse + RMSNorm) | hidden_size 4096, hc_mult 4, bf16 | num_tokens |
| 1 | dsa.fused_qkv_a | single_gemm | `torch_linear_vllm` → `F.linear` (cuBLAS) | n 2048, k 4096, bf16 | m |
| 2 | dsa.q_kv_a_norm | q_kv_rms_norm | `vllm_triton` → `fused_q_kv_rmsnorm` (Triton) | q_dim 1536, kv_dim 512 | num_tokens |
| 3 | dsa.q_b_proj | single_gemm | torch_linear_vllm | n 16384, k 1536 | m |
| 4 | dsa.indexer.wq_b | single_gemm | torch_linear_vllm | n 4096, k 1536 | m |
| 5 | dsa.indexer.wk_weights | single_gemm | torch_linear_vllm | n 160, k 4096 | m |
| 6 | dsa.indexer.head_weights | gemm_fp32_output | `torch_cublas` → `torch.mm` fp32 | n 32, k 4096, input fp32 | m |
| 7,8,9 | indexer.k_norm / q_fwht_quant / weight_scale | elementwise | `triton` → byte-sized synthetic Triton copy (placeholder for Inductor/fork kernels) | in/out bytes per token 256/256, 40960/21120, 256/128 | num_tokens |
| 10 | indexer.kpool_gate_score | single_gemm | torch_linear_vllm | n 128, k 4096 | m |
| 11 | indexer.prefill_gather (x2) | elementwise | triton | 1024/1024 B/tok | num_tokens |
| 12 | indexer.kpool_decode_update | elementwise | triton | 908/545 | (decode only, not run) |
| 13,14 | indexer.kpool_prefill_write / kpool_tail_seed | elementwise | triton | 521/33, 8/4 | num_tokens |
| 15 | indexer.mqa_logits_decode | dsa_paged_mqa_logits_decode | `deepgemm_fp8` → DeepGEMM `fp8_paged_mqa_logits` | heads 32, dim 128, block 64, fp8 | (decode only) |
| 16 | indexer.topk_decode | dsa_persistent_topk_decode | `vllm_cuda` → `persistent_topk` (CUDA) | top_k 512 | (decode only) |
| 17 | indexer.mqa_logits_prefill | dsa_mqa_logits_prefill | `deepgemm_fp8` → DeepGEMM `fp8_mqa_logits` (fork's `fp8_fp4_mqa_logits` = `sm100_mqa_logits`) | heads 32, dim 128, fp8, span_mode single_causal_tail | `prefill_query_key_pairs` = list of (rows, pools) chunks; chunked so rows·pools·4 B ≤ 512 MiB |
| 18 | indexer.topk_prefill | dsa_topk_prefill | best-of-N over `deep_select` (DeepSelect `torch.ops.deep_select.topk`) and `vllm_cuda` (`topKPerRowPrefill`, CUDA) | top_k 512, single_causal_tail | same chunk pairs |
| 19 | indexer.expand_pools | elementwise | triton | 4100/8204 | num_tokens |
| 20 | dsa.q_absorb | batched_gemm | `torch_mla_q_absorb_no_rope` → `torch.bmm` on packed W_UK | batches 64, n 512, k 256 | m |
| 21 | dsa.q_concat | elementwise | triton | 655360/655360 | off (`mla_layout_copies: false`) |
| 22 | dsa.q_fp8_quant | elementwise | triton | 65536/33024 | num_tokens |
| 23 | sparse_mla.mla_cache_append | mla_cache_append | `vllm_cuda` → `concat_and_cache_mla` (CUDA) (BF16 to FP8) | kv_lora 512, block 64, plain | num_tokens |
| 24 | sparse_mla.index_remap | dsa_sparse_index_remap | `vllm_triton` → `triton_convert_req_index_to_global_index` | selected_k 2176, block 64 | num_queries, total_valid_count |
| 25 | sparse_mla.prefill | dsa_sparse_mla_attention | `flashinfer_trtllm_fp8` → FlashInfer `trtllm_batch_decode_with_kv_cache_mla` sparse mode, FP8 q + paged FP8 latent cache | H 64, kv_heads 1, selected_k 2176, latent 512, value 512, rope 0, `valid_counts_pattern: causal_tail`, layout `hnd_paged_mqa_fp8_latent` | `prefill_query_cache_pairs` (Q,S): ALL rows of the iteration as one query; S chosen so the causal ramp carries the same valid-slot total (`glm53_kpool_sparse_mla.rs:1-31`). At the operating point S = Q, e.g. `[[21301,21301]]` |
| 26 | sparse_mla.decode | dsa_sparse_mla_attention | same callable, `pooled_uniform_full` pattern | | (decode only) |
| 27 | dsa.output_copy | elementwise | triton | 229376/229376 | off (`mla_layout_copies: false`) |
| 28 | dsa.v_up | batched_gemm | `torch_mla_v_up_unpadded` → `torch.bmm` on packed W_UV | batches 64, n 256, k 512 | m |
| 29 | dsa.o_proj | single_gemm | torch_linear_vllm | n 4096, k 16384 | m |
| 30,31 | dsa.glue (x17), dsa.prefill_glue (x8) | elementwise | triton | 64/64, 8704/8704 | num_tokens |
| 33,34 | moe.router.gate / gate_recompute | gemm_fp32_output | `torch_cublas` → `torch.mm(out_dtype=fp32)` (priced as two launches; one is enough) | n 288, k 4096, bf16 | m |
| 35,46 | moe.input_glue / combine_glue | elementwise | triton | 8192/8192, 16384/8192 | num_tokens |
| 41 (39) | moe.routed_rank0.input_quant | nvfp4_quant | `vllm_cuda` → `scaled_fp4_quant` (CUDA), linear E4M3 scales | hidden 4096, group 16 | num_tokens |
| 42 (40) | moe.routed_rank0.fused_moe | nvfp4_fused_moe | `flashinfer_trtllm_sm100` → FlashInfer `trtllm_fp4_block_scale_moe` (routing + FC1 + act + FC2 + finalize as one boundary) | hidden 4096, inter 2048, experts 288 (all local), top_k 8, bf16 in, nvfp4_e2m1 g16, routing `deepseek_v3`, n_group 1, topk_group 1, scaling 5/2, `expert_demand: corpus` (routes.u16, 48896 tokens x 42 layers, seed 4028456685, `sampling_candidates` 16) | num_tokens (the per-expert histogram is derived from the corpus per grid point) |
| 43-45 (36-38) | moe.shared_expert.gate_up / act_and_mul / down | single_gemm / elementwise / single_gemm | torch_linear_vllm / triton | gate_up n 4096 (2048·2) k 4096; act 8192/4096; down n 4096 k 2048; BF16 | m / num_tokens |
| 48 | kda.in_proj | single_gemm | torch_linear_vllm | n 24896, k 4096 | m |
| 49,50 | kda.f_b_proj, g_b_proj | single_gemm | torch_linear_vllm | n 8192, k 128 | m |
| 51 | kda.short_conv_prefill | gdn_causal_conv_prefill | `dao_channellast` → Dao-AILab `causal-conv1d` channel-last kernel, **one launch per sequence**, final state written to the cache slot | channels 24576, kernel 4, bf16 | `sequence_lengths` (each prefill sequence priced as its own `(1, L_i)` row and summed) |
| 52 | kda.short_conv_decode | gdn_causal_conv_decode | `vllm_triton` → `causal_conv1d_update` | | (decode only) |
| 54,57 | kda.state_gather / state_scatter | elementwise | triton | 4096/4096 per 4-KiB unit | num_tokens = state units (12288 in the sample = 48 states x 256) |
| 55 | kda.chunk_prefill | kda_chunk_prefill | `flashinfer_cute_persistent` → FlashInfer `recurrent_kda(backend='cute-dsl-persistent')`: 3 q/k/v copies + 1 persistent CuTe-DSL kernel (L2 norms, gate, beta sigmoid, recurrence) | heads 64, dim 128, bf16 | num_tokens, max_sequence_length, num_decode_sequences |
| 56 | kda.recurrent_decode | kda_recurrent_decode | `vllm_triton` → `fused_recurrent_kda` | | (decode only) |
| 58 | kda.gated_norm | elementwise | triton | 32768/16384 | num_tokens |
| 59 | kda.o_proj | single_gemm | torch_linear_vllm | n 4096, k 8192 | m |

Stage-0-only leaves: `pp.embedding` (elementwise 8192/8192), `pp.hc_expand` (elementwise 8192/32768),
`first_kda_dense.attn_mhc_pre` (mhc_pre_rms_norm `vllm_tilelang`). The dense FFN is
`dense_ffn.gate_up_input_quant`/`down_input_quant` (nvfp4_quant `vllm_cuda`, `swizzled_e4m3`), then
`dense_ffn.gate_up` (single_gemm `flashinfer_cutedsl`, n 24576 = 12288·2, k 4096, `nvfp4_e2m1`),
`act_and_mul`, and `down` (n 4096, k 12288). `flashinfer_cutedsl` is FlashInfer
`mm_fp4(backend='cute-dsl')`.

Sample stage-3 iteration (row group 6, row 100 of `worker_stage_3.parquet`): 21301 prefill tokens over
12 sequences, `total_time_ms` 112.888. The per-launch times (ms) were:
- MoE: fused_moe 2.3879, input_quant 0.0380, shared gate_up 0.4893, shared down 0.2425
- KDA: in_proj 3.0124, chunk_prefill 1.8123, o_proj 1.0345, short_conv_prefill 0.5002
- DSA: sparse_mla.prefill 3.2208, mqa_logits_prefill 1.6475 (7 chunks), topk_prefill 0.8378,
  o_proj 2.0068, q_b_proj 0.7208
- mhc_fused 0.4516

---

## 2. Kernel time portions at the operating point

Source: `$R/reports/kernel_time_share_report.json`. Totals are exact
(`kernel_time_totals_exact: true`). Within-worker mixtures are estimated from a 1-in-4 iter_id sample
(`exact: false`). The report attributes time down the tree: a Sum adds its children, a Scale
multiplies its child, and a Max or Parallel goes to its critical child divided by overlap.
Communication share: **0%**. There are no collective leaves at TP1/EP1, and the PP p2p transfer is
outside the tree (section 1.2).

### 2.1 By kernel kind (share of busy time)

| kind | all 5 stages | stage 3 | stage 3 ms |
|---|---|---|---|
| single_gemm | 39.466% | 36.432% | 10438889.5 |
| nvfp4_fused_moe | 17.536% | 17.862% | 5118009.0 |
| dsa_sparse_mla_attention | 7.229% | 9.371% | 2685070.8 |
| kda_chunk_prefill | 10.099% | 8.472% | 2427393.3 |
| mhc_fused_post_pre_rms_norm | 7.249% | 6.928% | 1985152.2 |
| dsa_mqa_logits_prefill | 4.747% | 6.154% | 1763229.2 |
| elementwise | 4.439% | 4.690% | 1343876.0 |
| dsa_topk_prefill | 2.149% | 2.786% | 798210.7 |
| gdn_causal_conv_prefill | 3.099% | 2.600% | 744946.9 |
| batched_gemm | 1.653% | 2.142% | 613850.2 |
| gemm_fp32_output | 0.994% | 1.092% | 312938.1 |
| dsa_sparse_index_remap | 0.689% | 0.893% | 255890.5 |
| nvfp4_quant | 0.365% | 0.293% | 83967.3 |
| q_kv_rms_norm | 0.112% | 0.146% | 41784.1 |
| mla_cache_append | 0.108% | 0.140% | 40046.2 |
| mhc_pre_rms_norm | 0.053% | - | - |
| rms_norm | 0.014% | - | - |
| total kernel ms | 136199146.14 | 100% | 28653253.73 |

### 2.2 Grouped (by leaf name; computed from the report's `segments`)

| group | all stages | st 0 | st 1 | st 2 | **st 3** | st 4 |
|---|---|---|---|---|---|---|
| KDA projections (in_proj, f_b, g_b, o_proj; BF16) | 26.40% | 28.41% | 27.31% | 27.31% | **22.15%** | 27.16% |
| MoE routed (nvfp4_fused_moe + its nvfp4_quant) | 17.82% | 13.31% | 19.19% | 19.19% | **18.15%** | 19.08% |
| DSA sparse MLA (attn prefill, remap, cache append, q_absorb, v_up, q_fp8_quant) | 10.26% | 9.75% | 9.37% | 9.37% | **13.31%** | 9.32% |
| KDA core (chunk_prefill, short conv, state gather/scatter, gated norm) | 14.28% | 15.36% | 14.76% | 14.76% | **11.98%** | 14.68% |
| DSA indexer (mqa logits, top-k, its GEMMs, glue) | 8.20% | 7.79% | 7.49% | 7.49% | **10.63%** | 7.45% |
| DSA projections/norm/glue (fused_qkv_a, q_b, o_proj, q_kv norm, glue) | 7.13% | 6.77% | 6.51% | 6.51% | **9.24%** | 6.47% |
| mHC boundaries | 7.30% | 7.47% | 7.32% | 7.32% | **6.93%** | 7.49% |
| MoE shared expert (BF16) | 5.86% | 4.38% | 6.31% | 6.31% | **5.97%** | 6.28% |
| Dense FFN (NVFP4, layers 0-2) | 1.01% | 5.30% | - | - | - | - |
| MoE glue | 0.91% | 0.68% | 0.98% | 0.98% | **0.93%** | 0.98% |
| MoE router gate x2 | 0.71% | 0.53% | 0.76% | 0.76% | **0.72%** | 0.76% |
| embedding / head | 0.11% | 0.23% | - | - | - | 0.33% |
| comm / p2p | 0 | 0 | 0 | 0 | 0 | 0 |

Stage-3 top leaves (share of stage-3 busy time; DSA rows already include x3 and KDA rows x6):

| leaf | stage 3 share |
|---|---|
| kda.in_proj | 15.942% |
| kda_moe fused_moe | 11.908% |
| dsa sparse_mla.prefill | 9.371% |
| kda.chunk_prefill | 8.472% |
| dsa.indexer.mqa_logits_prefill | 6.154% |
| dsa_moe fused_moe | 5.954% |
| kda.o_proj | 5.441% |
| dsa.o_proj | 5.288% |
| dsa.indexer.topk_prefill | 2.786% |
| kda.short_conv_prefill | 2.600% |
| kda shared_expert.gate_up | 2.507% |
| kda attn_mhc / ffn_mhc | 2.309% each |
| dsa.q_b_proj | 2.002% |

### 2.3 Achieved throughput per leaf

Source: `$R/reports/kernel_throughput_report.json`. It samples 1 in 50 iter_ids and pools each
location across stages. Achieved = slot_flops / slot_time or slot_bytes / slot_time. Grid peak is from
`$R/raw/kernel_grid_peaks.json` (best row on the run's profiled grid for that config).

| leaf (kind, backend) | p50 TFLOP/s | mean TFLOP/s | p50 GB/s | grid peak TFLOP/s / GB/s |
|---|---|---|---|---|
| kda.in_proj (single_gemm torch_linear_vllm n24896 k4096) | 1418.1 | 1412.7 | 460.3 | 1590.6 / 5811.5 |
| kda.o_proj (n4096 k8192) | 1364.6 | 1360.2 | 560.3 | 1488.4 / 4502.5 |
| kda.f_b/g_b (n8192 k128) | 598.9 | 604.3 | 4773.1 | 692.9 / 5554.2 |
| dsa.o_proj (n4096 k16384) | 1390.6 | 1397.0 | 474.0 | 1512.4 / 5398.2 |
| dsa.q_b_proj (n16384 k1536) | 1395.4 | 1391.3 | 1049.6 | 1519.0 / 5040.1 |
| dsa.fused_qkv_a (n2048 k4096) | 1426.6 | 1409.0 | 1090.6 | 1638.9 / 2452.8 |
| shared_expert.gate_up (n4096 k4096) | 1460.4 | 1467.0 | 780.5 | 1617.2 / 4410.4 |
| shared_expert.down (n4096 k2048) | 1446.3 | 1439.3 | 1111.9 | 1616.0 / 3438.1 |
| routed fused_moe (nvfp4_fused_moe flashinfer_trtllm_sm100) | 3721.1 | 3685.8 | 1588.2 | 4082.1 / 6907.0 |
| routed input_quant (nvfp4_quant vllm_cuda) | - | - | 5914.3 | - / 5960.0 |
| sparse_mla.prefill (dsa_sparse_mla_attention flashinfer_trtllm_fp8) | 1547.7 | 1544.3 | 6649.2 | 1564.1 / 6699.2 (valid-slot TFLOPS, see 3.5) |
| indexer.mqa_logits_prefill (deepgemm_fp8) | 1876.5 | 1855.3 | 977.2 | 2055.1 / 1199.1 |
| indexer.topk_prefill (deep_select / vllm_cuda) | - | - | 1986.5 | - / 6306.3 |
| sparse_mla.index_remap (vllm_triton) | - | - | 1589.0 | - / 1576.3 |
| sparse_mla.mla_cache_append (vllm_cuda) | - | - | 593.9 | - / 740.1 |
| q_absorb (batched_gemm) | 807.2 | 816.2 | 4761.2 | 968.4 / 6094.7 |
| v_up (batched_gemm) | 929.1 | 918.7 | 5473.8 | 1036.9 / 6387.0 |
| kda.chunk_prefill (kda_chunk_prefill flashinfer_cute_persistent) | 132.5 | 129.1 | 1048.3 | 138.4 / 4065.2 |
| kda.short_conv_prefill (gdn_causal_conv_prefill dao_channellast) | 9.4 | 9.3 | 4170.2 | 10.3 / 4559.4 |
| mhc_fused_post_pre (deepgemm_mega_nonshifted) | - | - | 5497.9 | - / 5507.5 |
| router gate (gemm_fp32_output torch_cublas n288) | 1088.5 | 1078.3 | 4359.1 | 1153.4 / 4638.6 |
| indexer.head_weights (gemm_fp32_output n32, fp32 in) | 38.4 | 38.2 | 2422.2 | 43.5 / 2740.9 |
| dense_ffn.gate_up (single_gemm flashinfer_cutedsl nvfp4, stage 0) | 4557.4 | 4505.2 | 1215.9 | 5613.9 / 4895.1 |
| dense_ffn.down (flashinfer_cutedsl, stage 0) | 3935.8 | 3950.8 | 629.7 | 6663.9 / 3250.8 |
| lm_head (n154880 k4096, stage 4, one row per request) | 103.5 | 103.8 | 6116.0 | 1512.6 / 6363.8 |
| big elementwise (q_fp8_quant, gated_norm, act_and_mul, glue) | - | - | 7000-7400 | - / 7400-8400 |

`best-of-N` selection for `indexer.topk_prefill` (input distribution report `selection`):
`deep_select` count 1555, `vllm_cuda` count 8960. Every other leaf has one candidate.

---

## 3. Matching kernel performance against the profile DB

### 3.1 Which DB

- Path resolution: `profiling/perf_api.py:35`,
  `DB_PATH = $VIBESIM_PROFILE_DB or <checkout>/profiling/profile.db`. Leave `VIBESIM_PROFILE_DB` unset.
- `profiling/profile.db` at this commit (the same on main since PRs #82 and #85) holds every row the runs read.
  B200 row counts of the backends the fastk path adds:

| table / backend | rows |
|---|---|
| kda_chunk_prefill / flashinfer_cute_persistent | 1024 |
| kda_chunk_prefill / flashkda | 1536 |
| kda_chunk_prefill / vllm_triton | 1536 |
| gdn_causal_conv_prefill / dao_channellast | 432 |
| mhc_fused_post_pre_rms_norm / deepgemm_mega_nonshifted | 88 |
| dsa_topk_prefill / deep_select | 412 |

- Rows you profile locally land in this file; `profiling/db/merge.py` merges DBs by semantic key.
- Open it read-only, e.g. with sqlite URI `file:...?mode=ro`.

### 3.2 Schema (schema v3, `_db_metadata`: `schema_version` 3, `schema_hash` `l1-compact-keys-v3`)

- One SQLite table per kernel kind (table name = kind, e.g. `nvfp4_fused_moe`). Column order:
  `id, gpu_name, backend, <one column per KernelArgs field>, args_hash BLOB, run_key, profiler_run_at
  (epoch s), verified, time_ms, tflops, memory_bandwidth_gbps, energy_j, is_outlier, retry_count,
  outlier_reason, created_at`.
- Collective tables also carry `algbw_gbps` and `busbw_gbps` (`profiling/db/storage.py` `NON_ARG_COLUMNS`).
- Key: `UNIQUE(gpu_name, backend, args_hash)`. `args_hash` is the first 8 bytes of
  sha256(canonical JSON of every args column as SQLite stores it) (`profiling/db/storage.py:1-25`,
  `args_hash`). Readers still compare the args columns. **There is no spec-JSON column.** The spec is
  the args columns. List-valued args are TEXT encodings:
  - `nvfp4_fused_moe.per_expert_batches` is a JSON list of per-expert rows, sorted descending (folded
    histogram).
  - `dsa_sparse_mla_attention.valid_counts` is an RLE string: `c:1..N@2176` is a causal ramp capped at
    2176, `u:QxK` is uniform, `g:(...)xN` is a group pattern, `r:a..b` is a range.
- Provenance: `run_key` joins `_profile_run(run_key, profiler_git_hash, cuda_version,
  driver_version, backend_version)`.
- Metrics: `time_ms` is the CUPTI GPU-active time of one logical call, using a two-pass window
  (`profiling/README.md`, `Timer.cupti`). `tflops` and `memory_bandwidth_gbps` are logical
  FLOPs and bytes over time, using the kind DOC formulas (`profiling/kernels/<kind>.py`):
  - single_gemm `2·m·n·k`
  - nvfp4_fused_moe `2·local_rows·(H·2·I + I·H)`, with local_rows = Σ per_expert_batches
  - kda_chunk_prefill `2·T·H·(3·D² + 4·64·D)`
  - dsa_sparse_mla_attention `2·H·Σvalid_counts·(D + value_dim)`
  - dsa_mqa_logits_prefill `2·C·heads·head_dim`
  - `energy_j = 0.0` means not measured.
- `gpu_name` is the DB key (`NVIDIA B200`), not a device selector.
- The simulator does not read a row directly. At build time it profiles or loads each kernel config's
  sweep grid and fits an interpolating cache. At eval time it interpolates the grid (best-of-N over
  backends). An operating-point time like fused_moe at T=21301 (2.3879 ms) therefore lies between
  grid rows at 20480 and 24576 (`simulator/src/timing/README.md` "The two phases"; grids are in
  `simulator/src/timing/kernels/<kind>.rs` `sweep_grid`).
  - nvfp4_fused_moe: `token_axis`, `Cache1DLinear`
  - kda_chunk_prefill: `(L, R=(T-D)/L, D)`, `Cache3DLinear`
  - dsa_sparse_mla_attention: `(num_queries, num_cache_tokens)`, `Cache2DLinear`
  - gdn_causal_conv_prefill: `(batch, seq_len)`

### 3.3 Read-only CLI (preferred; same path the simulator uses)

```bash
# from the ServingStudioSim root
uv run python -m launcher kernel-profile query single_gemm --backend torch_linear_vllm \
  --gpu-name "NVIDIA B200" --spec '{"m":32768,"n":24896,"k":4096,"dtype":"bf16"}' --json
# -> status ok, time_ms 4.878206818877551, tflops 1369.964284153437, memory_bandwidth_gbps 431.2994143376872
```
`query` and `count-missing` never profile. `run` JIT-fills missing rows on a GPU, which needs the Slurm
GPU host (or a job on your cluster's scheduler) and writes to the DB. See `skills/operate-profile-existing-kernel/SKILL.md`.

### 3.4 Example SQL with real output

All queries ran with `sqlite3.connect("file:profiling/profile.db?mode=ro", uri=True)`.

**(a) Routed MoE, nvfp4_fused_moe at 16384 / 32768 tokens.** The grid has 73 rows, token counts
1..65536.
```sql
SELECT num_tokens, substr(per_expert_batches,1,60), time_ms, tflops, memory_bandwidth_gbps,
       datetime(profiler_run_at,'unixepoch')
FROM nvfp4_fused_moe
WHERE gpu_name='NVIDIA B200' AND backend='flashinfer_trtllm_sm100' AND hidden_size=4096
  AND intermediate_size=2048 AND num_local_experts=288 AND top_k=8 AND num_tokens IN (16384,32768);
```
```
32768 [8755,5281,4222,3857,3619,3405,...] 3.452993551724138  3821.0727404121394 1444.2633585312924 2026-10-06 19:45:34
16384 [4356,2639,2108,1928,1815,1712,...] 1.9312610384615385 3415.938930716114  2346.6274054853793 2026-10-06 19:45:34
```
Provenance: profiler git `0219dfc2...` (13 rows) and `e3c5fcf4...` (60 rows).

**(b) KDA chunk prefill, prefill-only, three backends.**
```sql
SELECT backend, num_tokens, max_sequence_length, num_decode_sequences, time_ms, tflops, memory_bandwidth_gbps
FROM kda_chunk_prefill
WHERE gpu_name='NVIDIA B200' AND num_heads=64 AND head_dim=128 AND dtype='bf16' AND num_decode_sequences=0
  AND ((max_sequence_length=8192 AND num_tokens=16384) OR (max_sequence_length=16384 AND num_tokens=32768)
       OR (max_sequence_length=1024 AND num_tokens=32768));
```
```
flashinfer_cute_persistent 16384  8192 0 1.299350233766234  132.2189255640741  1049.1003615313498
flashinfer_cute_persistent 32768  1024 0 2.521584475        136.26249173349626 1174.3325093243209
flashinfer_cute_persistent 32768 16384 0 2.6219738974358973 131.04531056392804 1033.3895339880069
flashkda                   16384  8192 0 1.8988716981132074  90.47409154115353  717.8730407928442
flashkda                   32768  1024 0 3.519176827586207   97.63572577160677  841.4407030609666
flashkda                   32768 16384 0 3.7678586296296297  91.19168669918348  719.1141309530339
vllm_triton                16384  8192 0 5.0545016           33.98924472395063  269.6900521309559
vllm_triton                32768  1024 0 9.962105272727273   34.49043894573653  297.24426142199695
vllm_triton                32768 16384 0 10.002961500000001  34.34956574410488  270.87181971059266
```
Provenance of `flashinfer_cute_persistent`: profiler `703dee71...`, CUDA 13.0, driver 580.178.04,
`flashinfer-python 0.7.2.dev20261008, nvidia-cutlass-dsl 4.8.0; flashinfer commit 73b63f88ce24a38f6b2addc9a6f051b3e50cd659`.

**(c) Sparse MLA prefill (dsa_sparse_mla_attention), the run's `causal_tail` encoding at Q = S.**
```sql
SELECT num_queries, num_cache_tokens, substr(valid_counts,1,40), time_ms, tflops, memory_bandwidth_gbps
FROM dsa_sparse_mla_attention
WHERE gpu_name='NVIDIA B200' AND backend='flashinfer_trtllm_fp8' AND num_heads=64 AND selected_k=2176
  AND latent_dim=512 AND rope_dim=0 AND num_queries=num_cache_tokens AND num_queries IN (8192,16384,32768);
```
```
 8192  8192 c:1..8192@2176                1.3498037162162162 1501.176359111032  6516.5123182925245
 8192  8192 g:(2048,2049,2050,2051)x2048  1.568716984375     1402.824021290727  6041.261733247435
16384 16384 g:(2048,2049,2050,2051)x4096  3.13450809375      1404.1334732399748 6046.900888146733
16384 16384 c:1..16384@2176               2.8391592222222224 1536.636481357059  6622.954338320738
32768 32768 g:(2048,2049,2050,2051)x8192  6.29649025         1398.0066868446274 6020.515882161495
32768 32768 c:1..32768@2176               5.838151666666667  1547.6953274245757 6649.166173198081
```
All rows have `index_distribution` `unique_scattered_pages`, `cache_layout` `hnd_paged_mqa_fp8_latent`,
and `softmax_scale` 0.0625. The run's p50 of 1547.7 TFLOP/s matches the `c:1..32768@2176` row.

**(d) Indexer prefill logits and top-k, and KDA in_proj.**
```sql
SELECT num_queries, num_keys, time_ms, tflops, memory_bandwidth_gbps FROM dsa_mqa_logits_prefill
WHERE gpu_name='NVIDIA B200' AND backend='deepgemm_fp8' AND num_heads=32 AND head_dim=128
  AND span_mode='single_causal_tail' AND num_queries IN (2048,8192) AND num_keys IN (16384,65536);
-- 2048 16384 0.14229579637080866 1826.0941506020913  967.7553625066191
-- 2048 65536 0.573365948355453   1891.4259651284472  953.7517349408167
-- 8192 16384 0.41331654703910614 2015.9455540626207 1073.4598534184047
-- 8192 65536 2.070390770212766   1995.6418847092305  995.3577603073558

SELECT backend, num_queries, num_keys, logits_row_stride, time_ms, memory_bandwidth_gbps FROM dsa_topk_prefill
WHERE gpu_name='NVIDIA B200' AND backend IN ('deep_select','vllm_cuda') AND top_k=512
  AND span_mode='single_causal_tail' AND num_queries=8192 AND num_keys IN (16384,65536);
-- deep_select 8192 16384 16640 0.4617049162790698 908.6156659991743
-- vllm_cuda   8192 16384 16640 0.24044833171912833 1744.7087987702873
-- deep_select 8192 65536 65792 0.837470575        2424.1150872673948
-- vllm_cuda   8192 65536 65792 0.8689580608695652 2336.2750717433432

SELECT m, time_ms, tflops, memory_bandwidth_gbps FROM single_gemm
WHERE gpu_name='NVIDIA B200' AND backend='torch_linear_vllm' AND n=24896 AND k=4096 AND dtype='bf16'
  AND m IN (16384,20480,24576,28672,32768);
-- 16384 2.429878516009852  1375.1652744249589 474.90353134811664
-- 20480 2.915081596302003  1432.8434924973112 477.33152779159605
-- 24576 3.4008453173431734 1473.8179384023495 478.987623369059
-- 28672 4.06926597005988   1437.0154264991313 458.6737003019084
-- 32768 4.878206818877551  1369.964284153437  431.2994143376872
```
More rows read for reference:
- `mhc_fused_post_pre_rms_norm` / `deepgemm_mega_nonshifted`, hidden 4096, hc 4: T=16384 → 0.35035646926983016 ms
  (5375.24 GB/s); T=32768 → 0.6847936896551724 ms (5497.89 GB/s).
- `gdn_causal_conv_prefill` / `dao_channellast`, channels 24576, B=1: L=256 → 0.009108405 ms;
  L=4096 → 0.091116114 ms; L=16384 → 0.3572282759856631 ms.
- `single_gemm` / `flashinfer_cutedsl`, n 24576, k 4096, nvfp4: m=16384 → 0.6762403662310866 ms
  (4877.76 TFLOP/s); m=32768 → 1.5400111178977272 ms (4283.78).

### 3.5 Caveat: sparse-MLA TFLOPS count valid slots only

The sparse-MLA FLOP formula is `2·H·Σvalid·(D+value_dim)`. The DOC is at
`profiling/kernels/dsa_sparse_mla_attention.py:61` and `:75`. Masked `selected_k` slots take time but do no attention
work.

PR #82 recomputed the derived `tflops` column from `time_ms` for 17543 `dsa_sparse_mla_attention` rows and 1630
`dsa_sparse_mla_prefill` rows. The grid peaks fell from 8260 to 1564 TFLOP/s and from 6991 to 1465. `time_ms` did
not change, so simulated timing did not change.

An older copy of the DB still holds the old values for the same args_hash, for example:
- (32768, 32768, `c:1..32768@2176`): now 1547.6953 TFLOP/s, old 1600.8232
- (16384, 16384, `c:...`): now 1536.6365, old 1645.8832
- (32768, 32768, `g:...x8192`): now 1398.0067, old 1484.2950

Use `time_ms` to compare kernels.

Other row caveats:
- B200 rows profiled before 2026-09-25 may carry a warm-L2 bias (`profiling/README.md`, Timer.cupti
  paragraph).
- Row provenance must match the production callable. See "Row provenance and production equivalence"
  in `skills/operate-profile-existing-kernel/SKILL.md`.

---

## 4. Where each real kernel call lives (call the same thing)

Each kind has a module `profiling/kernels/<kind>.py`. Its module docstring and the `doc=` of each
`register(...)` describe the measured boundary and the FLOP and byte formulas. Its `RunnerRef` names the
runner function. "env" is the subprocess interpreter (`profiling/README.md` explains each env):
- `vllm_env` is the profiling container whose kernel sources are pinned at commit 3f667d7.
- `vllm_upstream_fork_env` is `alignment/profiler/vllm` and its `.venv`.
- `flashinfer_kda_env` is `~/profile_envs/flashinfer_kda`, built by `profiling/exec/flashinfer_kda_env.sh`.
- `causal_conv1d_env` is the project interpreter plus `~/profile_envs/causal_conv1d` (causal-conv1d 1.7.0 wheel).

| kind / backend | runner (file:line of the profile fn) | env |
|---|---|---|
| mhc_fused_post_pre_rms_norm / deepgemm_mega_nonshifted | `profiling/runners/mhc/mhc_fused_post_pre_rms_norm_deepgemm_mega_nonshifted.py:148` | vllm_upstream_fork_env |
| mhc_fused_post_pre_rms_norm / vllm_tilelang (stage 4 final post) | `profiling/runners/mhc/mhc_fused_post_pre_rms_norm_vllm_tilelang.py:84` | vllm_env |
| mhc_pre_rms_norm / vllm_tilelang (layer 0) | `profiling/runners/mhc/mhc_pre_rms_norm_vllm_tilelang.py:62` | vllm_env |
| single_gemm / torch_linear_vllm | `profiling/runners/gemm/torch.py:75` (`profile_single_gemm_linear`, `F.linear`) | vllm_env |
| single_gemm / flashinfer_cutedsl (NVFP4 dense FFN) | `profiling/runners/gemm/flashinfer_fp4.py:140` | vllm_env |
| q_kv_rms_norm / vllm_triton | `profiling/runners/attention/q_kv_rms_norm_vllm_triton.py:39` | vllm_env |
| gemm_fp32_output / torch_cublas | `profiling/runners/gemm/gemm_fp32_output_torch_cublas.py:94` | vllm_env |
| elementwise / triton (placeholder copies) | `profiling/runners/elementwise/triton.py:123` | project |
| dsa_mqa_logits_prefill / deepgemm_fp8 | `profiling/runners/attention/dsa_mqa_logits_prefill.py:516` | vllm_env |
| dsa_topk_prefill / deep_select | `profiling/runners/attention/dsa_topk_prefill.py:316` | vllm_upstream_fork_env |
| dsa_topk_prefill / vllm_cuda | `profiling/runners/attention/dsa_topk_prefill.py:548` | vllm_env |
| dsa_paged_mqa_logits_decode / deepgemm_fp8 | `profiling/runners/attention/dsa_paged_mqa_logits_decode.py:819` | vllm_env |
| dsa_persistent_topk_decode / vllm_cuda | `profiling/runners/attention/dsa_persistent_topk_decode.py:650` | vllm_env |
| batched_gemm / torch_mla_q_absorb_no_rope | `profiling/runners/gemm/batched_gemm.py:463` | vllm_env |
| batched_gemm / torch_mla_v_up_unpadded | `profiling/runners/gemm/batched_gemm.py:479` | vllm_env |
| mla_cache_append / vllm_cuda | `profiling/runners/attention/mla_cache_append.py:375` | vllm_env |
| dsa_sparse_index_remap / vllm_triton | `profiling/runners/attention/dsa_sparse_index_remap.py:940` | vllm_env |
| dsa_sparse_mla_attention / flashinfer_trtllm_fp8 | `profiling/runners/attention/dsa_sparse_mla_attention.py:873` | vllm_env |
| nvfp4_quant / vllm_cuda | `profiling/runners/elementwise/nvfp4_quant.py:68` | vllm_env |
| nvfp4_fused_moe / flashinfer_trtllm_sm100 | `profiling/runners/moe/nvfp4_fused_moe.py:501` (`profile_nvfp4_fused_moe_sm100`) | vllm_env |
| gdn_causal_conv_prefill / dao_channellast | `profiling/runners/attention/gdn_causal_conv_prefill_dao_channellast.py:305` | causal_conv1d_env |
| gdn_causal_conv_decode / vllm_triton | `profiling/runners/attention/gdn_causal_conv_decode_vllm_triton.py:234` | vllm_env |
| kda_chunk_prefill / flashinfer_cute_persistent | `profiling/runners/attention/kda_chunk_prefill_flashinfer.py:322` | flashinfer_kda_env |
| kda_recurrent_decode / vllm_triton | `profiling/runners/attention/kda_recurrent_decode_vllm_triton.py:266` | vllm_env |
| rms_norm / vllm_cuda (stage 4 final norm) | `profiling/runners/norm/rms_norm_vllm_cuda.py:33` | vllm_env |

Registration sites (`register(` lines):
- `profiling/kernels/kda_chunk_prefill.py`: 146, 196, 227 (`flashinfer_cute_persistent` at line 230), 264
- `nvfp4_fused_moe.py`: 167 (`flashinfer_trtllm_sm100` at line 170), 196, 225, 253, 282
- `dsa_sparse_mla_attention.py`: 88, 111, 141
- `single_gemm.py`: 99, 119, 142, 162, 192, 226, 255, 287
- `mhc_fused_post_pre_rms_norm.py`: 71, 94, 124
- `gdn_causal_conv_prefill.py`: 90, 111, 140
- `dsa_topk_prefill.py`: 79, 98, 124, 151
- `dsa_mqa_logits_prefill.py`: 80, 105

Each runner builds its inputs (including the corpus-shaped expert histogram, causal-tail valid counts,
and the paged FP8 cache) outside the timed closure. It times the public callable with `Timer.cupti`.
The runner is the best template for calling the same kernel with the same layout. The Rust sides are
in `simulator/src/timing/kernels/<kind>.rs` (module docs explain cache axes and off-grid rules).

---

## 5. Docs and skills to read

CostTree and timing:
- `simulator/src/timing/COST_TREE.md`
- `simulator/src/timing/README.md`
- `doc/detailed_design/L1.md` through `L4.md`
- `simulator/src/arch/README.md`
- `simulator/src/worklet/README.md`
- `doc/architecture_compatibility.md` (line 46 lists this arch)

Profile DB:
- `profiling/README.md`
- `profiling/db/storage.py` (schema v3 docstring)
- `profiling/db/merge.py`
- `profiling/db/registry.py`

Reports:
- `doc/analyzer.md`
- `analyzer/README.md`

Skills (in `skills/`, linked as `.claude/skills`):
- `operate-profile-existing-kernel` (query and fill rows, provenance)
- `operate-use-analyzer` (citing analyzer results)
- `top-compose-real-framework-from-sim` (Tick/Tock/Probe loop, comparison contract)
- `operate-run-alignment` and `top-align-with-framework`
- `top-split-model-into-kernels`
- `impl-compose-arch`
- `dev-explore-kernel`
- `operate-align-moe-kernel`
