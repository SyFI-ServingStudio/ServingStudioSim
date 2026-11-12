# feat(qwen3.8): Qwen3.8-Flash-Next-NVFP4 on vLLM, one B200, MTP k=3, with kernel alignment and the 20-case campaign

## Purpose

Model `nvidia/Qwen3.8-Flash-Next-NVFP4` as the vLLM fork serves it on one B200 with native MTP drafting 3 tokens.
The model is `Qwen4ExpForConditionalGeneration`, served text-only:
- 48 layers, `[GDN ×3, QSA]` ×12, hidden 2560, `hc_count` 4 hyper-connection streams (low rank 320);
- Gated DeltaNet: 16 K heads / 48 V heads, head dim 128;
- QSA: sparse gated GQA, 24 Q / 2 KV heads, head dim 256, with a weight-free indexer (4 heads of 128, one key per
  4 tokens, top 2048 tokens per row);
- a PLE n-gram embedding block in layer 1, its 320,001,536-row FP8 table in pinned host memory, read over UVA;
- MoE in every layer: 512 routed experts, top-10, intermediate 640, and a sigmoid-gated shared expert;
- one MTP layer (QSA, FP8-block experts) that reuses the target's embedding and lm_head;
- ModelOpt `MIXED_PRECISION`: NVFP4 body experts, W8A8 FP8 block-128 MTP experts, an FP8 PLE table, BF16 elsewhere;
- FP8 KV cache, FP8 indexer cache, FP32 GDN state, prefix caching on.

The model is checked against vLLM captures: `diverse_100` and 20 operating points over chunk 2048, 4096 and 8192.

- **New L1 kinds:**
  - Hyper-connections: `hc_combine_norm`, `hc_combine`, `hc_gate_mix`, `grouped_rms_norm` and `scaled_silu`
    (`vllm_triton`).
  - PLE: `embedding_row_gather` (`vllm_triton` over a pinned host table, `torch` over a device table),
    `ple_ngram_ids`, `ple_gate` and `ple_short_conv` (decode, spec and prefill modes).
  - QSA: `sparse_gqa_paged_attention`, `sparse_indexer_logits_decode`, `sparse_indexer_logits_prefill`,
    `sparse_indexer_prepare` and `sparse_index_expand`.
  - GDN verify: `gdn_recurrent_decode_gated_norm` (`vllm_cuda`, `fused_gdn_decode_post_conv_mtp`).
- **New backends and axes on existing kinds:**
  - `dsa_persistent_topk_decode` / `vllm_cuda_cooperative`, and a `uniform_random` context mode. The QSA top-k
    reuses this kind; the Rust variant `sparse_indexer_topk` reads it on a 16384-row grid.
  - `gdn_causal_conv_decode` and `gdn_recurrent_decode` take an optional `num_tokens` for verify steps. One-token
    rows keep their keys.
  - `indexed_row_copy` gains `index_select` and `index_copy`, the eager token-row ops of a step that mixes
    speculative and plain rows.
  - `silu_and_mul` / `vllm_cuda` (`torch.ops._C.silu_and_mul`) and `rms_norm` / `vllm_gemma_native`.
  - Serving-env twins of the eager backends: `elementwise` / `torch_vllm`, `torch_strided_copy_vllm`,
    `torch_cat_vllm`, and `indexed_row_copy` / `torch_vllm`.
  - `nvfp4_fused_moe`: `renormalize_naive` routing on the TRT-LLM NVFP4 and FP8 block-scale backends.
- **Ops, worklets and the arch.**
  - L2 ops `SparseGqaIndexerOp` and `SparseGqaAttentionOp`.
  - Worklets: `hc_mix_local`, `ple_local`, `vllm_qsa_attn_local`, `vllm_bf16_gdn_spec_local`,
    `vllm_nvfp4_moe_shared_stream_local` and `vllm_fp8_moe_shared_stream_local` with
    `bf16_gated_shared_expert_local`, `mtp_fc_prelude_local`, `spec_head_local`, `spec_sample_local` and
    `engine_prep_local`.
  - The arch `qwen4_exp_vllm_nvfp4_gdn_qsa_moe_speculative` and its config reader `qwen4_exp_model_cfg`. Every
    dimension and every module's precision comes from `config.json` and `quantization_config`; other precision
    mixes are refused.
  - KV follows vLLM's hybrid allocator: a 3200-token block, 1950 blocks at 79.80 GiB, and 31.45x concurrency at
    131072 tokens, as vLLM logs.
- **Shared simulator changes** (each listed with its effect in
  [`README.md`](README.md#changes-that-reach-other-models)):
  - A running request holds the declared state copies; align mode's previous-step copy enters only vLLM's
    concurrency estimate.
  - The speculative recipe aligns prefill chunks to Mamba align-mode checkpoints.
  - Speculative verify rows are clipped at `max_model_len`, as vLLM's scheduler does.
  - Against `origin/main`, 356 of 356 comparable public members and 54 of 54 timing-predict shapes of other models
    are bit-identical. The 6 speculative ctx128k members (GLM-5.2 MTP, GLM-5.3 DFlash2) now serve the requests that
    end at `max_model_len` whole, moving every metric by at most 0.25%. In the Qwen3.8-27B and GLM-5.3-Flash
    alignment packs only the KV-overcommit case 15 moves (`logs/20261013_16_qf_cross_model_impact/`).
- **model.work:** the Qwen4Exp label (gated hyper-connections, the n-gram embedding, QSA, MTP passes),
  `attention/pooled_topk.py` shared with GLM-5.3-Flash kpool DSA, and two location maps.
- **Presets.**
  - Public arch and sim presets: chunk 2048, replicas 1 / 2, KV pool 85.6846 GB (79.80 GiB),
    `gpu_time_multiplier` 1.0881, acceptance from the capture's per-request `accept_rate`.
  - Catalog entries.
- **Alignment.** The `qwen38_flash_next_nvfp4_b200_tp1` pack covers 20 operating points:
  - cases 01-15 at chunk 2048, with KV pools pinned for 14 and 15;
  - cases 16-20 at chunk 4096 and 8192, with `max_cudagraph_capture_size` equal to the chunk;
  - MTP acceptance by output position, measured from each case's own workload pass.

  Every case runs at `gpu_time_multiplier` 1.0881: the median of the 21 kernel-align recommendations (the 20
  cases and `diverse_100`; range 1.030-1.482). The same value ships in the public sim preset.

[`README.md`](README.md) has the modeling notes, the evidence and the open items.
[`alignment_matrix.md`](alignment_matrix.md) is the generated campaign matrix.

## Test Plan

Test tiers at the branch head. `origin/main` (176ab0a3) is merged; since the tested 3df15294 it adds only skills:

```bash
just test-cpu      # CUDA_VISIBLE_DEVICES=""
just test-presets
just test-gpu      # on one B200, through Slurm
```

Also run:

- **Kernel alignment on `diverse_100`**: the nsys capture (job 6613) through `alignment timing-predict` and
  `alignment analyze`.
- **Simulation of `diverse_100`**, untrimmed, compared by the E2E Analyzer with the warm workload pass (job 6614,
  same server args).
- **The 20-case campaign** through `alignment-campaign`, from capture to `compare`, at the branch head, on CPU.
- **Other archs unchanged** by the shared worker and arch changes: timing predictions of GLM-5.2 MTP, GLM-5.3
  DFlash2 and Qwen3.5 are identical, manifest included, before and after the review fixes.
- **model.work labels** of every config and floor arch are bit-identical across the `pooled_topk.py` extraction and
  the generic FP8 floor knobs.
- **Cache fidelity** of every new kind and backend against fresh B200 measurements, on Slurm.

## Test Result

- `just test-cpu` at 2885464e (`CUDA_VISIBLE_DEVICES=""`): Rust 1577 passed; pytest 5002 passed, 5 failed,
  16 skipped. The 5 failures:
  - four `test_l1_single_gemm` energy tests, which fail whenever CUDA is hidden, on `main` as well;
  - the Flash-Next public preset test, which read the corpus path before the Hugging Face upload pinned it.
- `just test-presets` with the corpus pinned at `b4b749dc` and the 1.0881 multiplier: 9 passed.
  `test_every_member_is_measured` passes: every public member is fully measured in the committed profile.db.
- `just test-gpu` at 7a289808 (one B200, through Slurm): 14 passed, 3 skipped.
- **Kernel alignment on `diverse_100`** (1710 iterations). The signed error of the simulated critical path against
  the measured one, per iteration. Source: `logs/20261013_15_qf_campaign_final/d100/summary.md`:

  | Deployment | Decode median | Mixed median | All median | All total | Coverage | Recommended multiplier |
  |---|---:|---:|---:|---:|---:|---:|
  | Qwen3.8-Flash-Next-NVFP4 TP1 MTP-3 | -3.0% | +2.6% | -2.0% | +0.87% | 99.37% | 1.204 |

- **GPU time multiplier.** Each capture's kernel-align `recommended_gpu_time_multiplier` (measured GPU cycle over
  measured critical path, outliers excluded). The gap it carries is the mixed-step host launch time, which depends on
  the step's shape, so it varies by case. The deployment value is the median of the 21 captures, 1.0881. Sources:
  `logs/20261013_15_qf_campaign_final/*/analysis_kernel/reports/alignment_iteration_report.json`:

  | Capture | Recommended | Decode stage | Mixed stage | Mixed share of time |
  |---|---:|---:|---:|---:|
  | 01 micro_throughput_c256 | 1.482 | 1.083 | 1.518 | 92% |
  | 02 graph_boundaries_c128 | 1.192 | 1.060 | 1.216 | 84% |
  | 03 balanced_anchor_c32 | 1.117 | 1.067 | 1.139 | 69% |
  | 04 prefill_spectrum_c32 | 1.148 | 2.025 | 1.123 | 97% |
  | 05 decode_spectrum_c64 | 1.298 | 1.151 | 1.950 | 18% |
  | 06 long_long_c8 | 1.086 | - | 1.086 | 100% |
  | 07 quadrant_interference_c48 | 1.067 | 1.034 | 1.100 | 49% |
  | 08 completion_churn_c96 | 1.059 | 1.031 | 1.132 | 28% |
  | 09 rate_knee | 1.088 | 1.062 | 1.169 | 25% |
  | 10 full_saturation_rate | 1.065 | 1.025 | 1.071 | 87% |
  | 11 long_prefill_ladder_c1 | 1.075 | 1.188 | 1.062 | 90% |
  | 12 ultra_context_decode_c1 | 1.093 | 1.187 | 1.063 | 76% |
  | 13 generation_length_matrix_c8 | 1.073 | 1.060 | 1.079 | 70% |
  | 14 near_full_long_decode_c64 | 1.176 | 1.066 | 1.178 | 98% |
  | 15 kv_overcommit_mixed_churn_c32 | 1.118 | 1.190 | 1.117 | 99% |
  | 16 prefill_spectrum_chunk4k_c32 | 1.163 | - | 1.162 | 100% |
  | 17 prefill_spectrum_chunk8k_c32 | 1.030 | - | 1.030 | 100% |
  | 18 quadrant_interference_chunk4k_c48 | 1.056 | 1.034 | 1.063 | 76% |
  | 19 quadrant_interference_chunk8k_c48 | 1.034 | 1.034 | 1.034 | 96% |
  | 20 full_saturation_chunk8k | 1.035 | 1.021 | 1.044 | 61% |
  | `diverse_100` | 1.204 | 1.130 | 1.241 | 67% |

  Decode steps replay FULL graphs and sit at 1.02-1.09 except at concurrency 1 (about 1.19). Mixed steps run
  PIECEWISE and span 1.03 (chunk 8k, one request) to 1.52 (bursts of small requests at concurrency 256). One uniform
  multiplier therefore under-corrects bursty small-chunk cases and over-corrects single-request ones.

- **Simulation of `diverse_100`** at multiplier 1.0881 (100 requests, untrimmed trace: requests 34 and 41, which end
  at `max_model_len`, are served). Source: `logs/20261013_17_qf_campaign_mult/d100/metrics_d100.json`:

  | Deployment | Server TTFT mean | Server TPOT mean | E2E mean | Output throughput | Iterations |
  |---|---:|---:|---:|---:|---:|
  | Qwen3.8-Flash-Next-NVFP4 TP1 MTP-3 | +14.46% | -6.60% | +8.87% | +0.55% | -12.48% |

  `diverse_100` arrives on its recorded timestamps, so slower steps batch more requests per step: the iteration count
  drops and TTFT and E2E rise with the multiplier (at 1.0 they read -8.2% and -8.4%, at 1.0881 +14.5% and +8.9%).
  TPOT stays low through six short-output requests (3-33 tokens) whose few rounds landed on long mixed steps.

- **Campaign**, all 20 cases, at `gpu_time_multiplier` 1.0881, acceptance by output position (the predictive
  setup the pack ships). Kernel phases from `logs/20261013_15_qf_campaign_final`, simulation and E2E analysis rerun
  in `logs/20261013_17_qf_campaign_mult`. The table is the output of `alignment-campaign compare --markdown`, copied
  from [`alignment_matrix.md`](alignment_matrix.md) (`logs/20261013_17_qf_campaign_mult/compare.md`):

  | Case | kernel_abs_pct | kernel_signed_pct | map_coverage_pct | e2e_mean_pct | output_tps_pct | server_tpot_mean_pct | server_ttft_mean_pct |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | tp1/01_micro_throughput_c256 | +2.95% | +2.84% | 99.57% | +0.17% | -0.04% | +5.44% | +0.19% |
  | tp1/02_graph_boundaries_c128 | +2.06% | +1.46% | 99.52% | -9.25% | +9.85% | -5.90% | -9.01% |
  | tp1/03_balanced_anchor_c32 | +1.87% | +0.12% | 99.47% | +1.54% | -1.10% | +2.67% | +5.79% |
  | tp1/04_prefill_spectrum_c32 | +1.58% | +1.48% | 99.41% | +4.86% | -4.39% | +3.49% | +10.83% |
  | tp1/05_decode_spectrum_c64 | +3.10% | -2.78% | 99.56% | +4.18% | +3.70% | +4.81% | -4.97% |
  | tp1/06_long_long_c8 | +2.75% | +2.75% | 99.41% | +5.50% | -5.23% | +6.85% | +3.75% |
  | tp1/07_quadrant_interference_c48 | +2.60% | -1.46% | 99.51% | +4.26% | +9.96% | +2.04% | +8.50% |
  | tp1/08_completion_churn_c96 | +1.99% | -0.39% | 99.39% | +5.93% | +10.62% | +5.19% | +3.33% |
  | tp1/09_rate_knee | +4.28% | -2.90% | 99.47% | +2.69% | +2.08% | +1.88% | -8.56% |
  | tp1/10_full_saturation_rate | +1.16% | -0.38% | 99.39% | -0.29% | +1.78% | +1.28% | -1.69% |
  | tp1/11_long_prefill_ladder_c1 | +5.43% | +5.43% | 99.30% | +10.64% | -9.60% | +7.96% | +11.51% |
  | tp1/12_ultra_context_decode_c1 | +4.97% | +4.97% | 99.27% | +7.49% | -6.97% | +6.92% | +11.21% |
  | tp1/13_generation_length_matrix_c8 | +2.46% | +2.19% | 99.37% | +4.96% | -5.37% | +4.79% | +10.25% |
  | tp1_chunk4k/16_prefill_spectrum_chunk4k_c32 | +0.61% | +0.25% | 99.77% | +4.69% | -4.48% | +3.56% | +7.83% |
  | tp1_chunk4k/18_quadrant_interference_chunk4k_c48 | +1.53% | -1.34% | 99.44% | +6.99% | +8.59% | +8.07% | +3.55% |
  | tp1_chunk8k/17_prefill_spectrum_chunk8k_c32 | +1.17% | -0.84% | 99.33% | +5.68% | -5.57% | +4.34% | +9.49% |
  | tp1_chunk8k/19_quadrant_interference_chunk8k_c48 | +2.01% | -0.06% | 99.23% | +4.87% | +1.37% | +4.06% | +8.82% |
  | tp1_chunk8k/20_full_saturation_chunk8k | +1.67% | -1.46% | 99.32% | +3.51% | +0.44% | +3.45% | +3.12% |
  | tp1_kv14/14_near_full_long_decode_c64 | +2.44% | +2.44% | 99.43% | -4.97% | +5.73% | -3.47% | -15.61% |
  | tp1_kv15/15_kv_overcommit_mixed_churn_c32 | +2.83% | +2.82% | 99.43% | -5.99% | +3.64% | -8.41% | -0.27% |

  86 metrics are out of tolerance ([`alignment_matrix.md`](alignment_matrix.md), "Out of tolerance"): 66 of them are
  the ±1% workload-structure bands, 9 iteration cycle, 5 output throughput, 3 E2E (02, 11, 12), one TTFT (14) and
  case 11's kernel pair. Mapping coverage is 99.23-99.77% in every case (floor 97%). The causes:
  - **Per-request acceptance spread: the structure bands and the throughput misses** (07, 08, 18, 19; part of 05, 09,
    10 and 20). The by-position chain fixes the mean: decode request-rounds match within 0.3% (case 08: 57,265
    measured, 57,126 simulated). But within a case the measured per-request tokens per round run from about 2.0
    (p10) to 3.7 (p90). Every simulated request shares one chain and finishes with its neighbours; in vLLM the slow
    ones drain in a long small-batch tail. The same rounds spread over 15-29% more steps, so iterations read low and
    the per-step means (decode batch, prefill tokens, scheduled KV) high. The observed-conditioned run below confirms
    it.
  - **Mixed-step host launch time, around the median multiplier** (E2E 02, 11, 12). A PIECEWISE mixed step issues
    about 2,100 CUDA API calls and one pageable H2D copy plus `cudaStreamSynchronize` in preprocess. Case 02's own
    recommendation is 1.19, so 1.0881 leaves it fast (-9.3%); cases 11 and 12 run one request, recommend 1.075 and
    1.093 (their mixed steps 1.06), and read slow (+10.6%, +7.5%); case 11 also carries its +5.43% kernel error.
  - **Server TTFT** (14: -15.6%). vLLM's async scheduler takes about twice as long from first schedule to first
    token, and in this queued case the simulator drains the queue faster.
  - **Case 11 kernel +5.43%.** Its mixed stage is single-request 2048-token prefill, where mixed `fused_moe` (+14%)
    and `gdn.qkvz` (+15%) read high: the expert-histogram layer fold with a generated-only routing corpus, and the
    profiler's duty cycle at m=2048 (Known gaps in [`README.md`](README.md)).
  - The fork's per-iteration log is shifted by one to two steps, so logged decode steps before a mixed step carry
    part of its time (totals conserved). Per-step cycle strata read this artifact; request-level metrics do not.

- **Observed-conditioned campaign** (`predictive_alignment: false`; reported beside the predictive matrix, not in
  place of it). Each case's `trace_speculative.csv` is rebuilt by `alignment prepare-workload` from its own workload
  pass: every request gets its measured per-position acceptance (`--missing-acceptance run-aggregate`; the only
  fallbacks are requests with no verify round). Same multiplier 1.0881. Source:
  `logs/20261013_18_qf_campaign_observed/compare.md` and `metrics.json`:

  | Case | Iterations | Decode batch | Prefill tokens | Scheduled KV | Iteration cycle | E2E | Output tps | TPOT | TTFT |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | tp1/01_micro_throughput_c256 | -5.68% | +2.84% | +6.02% | +1.17% | +7.49% | +0.48% | -0.41% | +6.49% | +0.60% |
  | tp1/02_graph_boundaries_c128 | -3.55% | +0.68% | +3.42% | -0.21% | -2.51% | -8.26% | +7.80% | -3.66% | -8.33% |
  | tp1/03_balanced_anchor_c32 | -2.26% | +1.27% | +2.31% | +1.12% | +5.36% | +2.49% | -2.18% | +4.15% | -0.14% |
  | tp1/04_prefill_spectrum_c32 | -1.75% | -1.47% | +1.78% | -2.03% | +7.11% | +4.52% | -4.52% | +3.94% | +9.53% |
  | tp1/05_decode_spectrum_c64 | -0.34% | +0.18% | +0.34% | +4.75% | +2.93% | +2.07% | -2.36% | +3.42% | -5.35% |
  | tp1/06_long_long_c8 | -0.17% | -0.12% | +0.17% | -0.05% | +5.20% | +5.34% | -4.72% | +5.47% | +11.74% |
  | tp1/07_quadrant_interference_c48 | -1.45% | +1.16% | +1.47% | +3.98% | +4.09% | +2.32% | -2.35% | +0.41% | +8.16% |
  | tp1/08_completion_churn_c96 | -0.67% | +0.49% | +0.68% | +4.45% | +4.82% | +4.19% | -3.77% | +4.87% | +4.73% |
  | tp1/09_rate_knee | -1.91% | +1.80% | +1.95% | +4.53% | +2.34% | +2.18% | -0.37% | +2.10% | -8.68% |
  | tp1/10_full_saturation_rate | -0.05% | +0.00% | -7.29% | +2.84% | +1.49% | +0.19% | -1.33% | +1.39% | -0.22% |
  | tp1/11_long_prefill_ladder_c1 | -1.98% | -4.14% | +2.02% | -0.43% | +12.81% | +10.44% | -9.43% | +2.75% | +11.51% |
  | tp1/12_ultra_context_decode_c1 | -0.16% | -0.01% | +0.16% | +0.42% | +7.92% | +7.66% | -7.11% | +7.40% | +11.21% |
  | tp1/13_generation_length_matrix_c8 | -0.19% | +0.09% | +0.19% | +1.70% | +5.39% | +4.85% | -4.90% | +4.88% | -1.36% |
  | tp1_chunk4k/16_prefill_spectrum_chunk4k_c32 | +0.00% | -1.88% | +0.00% | -2.17% | +5.42% | +4.78% | -4.57% | +5.39% | +6.78% |
  | tp1_chunk4k/18_quadrant_interference_chunk4k_c48 | -0.42% | +0.38% | +0.42% | +2.15% | +4.56% | +3.76% | -3.81% | +6.16% | +5.00% |
  | tp1_chunk8k/17_prefill_spectrum_chunk8k_c32 | -1.12% | -2.26% | +1.13% | -2.58% | +7.33% | +5.75% | -5.58% | +5.01% | +9.09% |
  | tp1_chunk8k/19_quadrant_interference_chunk8k_c48 | -2.28% | +2.12% | +2.33% | +3.69% | +6.04% | +3.64% | -3.21% | +1.24% | +11.37% |
  | tp1_chunk8k/20_full_saturation_chunk8k | -0.63% | +0.53% | -5.61% | +3.59% | +4.38% | +3.54% | -3.51% | +3.52% | +3.59% |
  | tp1_kv14/14_near_full_long_decode_c64 | -0.04% | -0.13% | -5.38% | +0.69% | -4.21% | -4.88% | +4.49% | -3.46% | -14.88% |
  | tp1_kv15/15_kv_overcommit_mixed_churn_c32 | +0.75% | -0.80% | -17.68% | -0.30% | -5.18% | -7.19% | +4.72% | -10.04% | +0.21% |
  Out of tolerance: 86 predictive, 54 observed-conditioned. The structure bands drop from 66 to 45, iteration cycle
  from 9 to 1 and output throughput from 5 to 1; E2E stays at 4 (02, 11, 12, 15). The remaining structure misses are
  mostly 1-5%; the larger ones are the saturated bursts (01, 02: vLLM refills freed slots over more, smaller prefill
  steps) and the prefill tokens of 10, 15 and 20 (-6 to -18%, not yet attributed). `diverse_100` under the same
  conditioning: iterations -7.0%, E2E +9.2%, TTFT +15.1%, TPOT -8.1%, output tps -1.6%.

[`README.md`](README.md) has the per-operation breakdown and the remaining gaps. This is review evidence, not an
accepted baseline, and no golden is recorded.

## Dependencies

- **vLLM fork** `servingstudio-alignment`. The submodule moves from `892da082` to `e1423a4d`, where the captures
  ran: one commit that isolates the Qwen MTP aux config and records the drafter's routed experts (routed-expert
  capture only; no kernel changes).
- **Checkpoint** `nvidia/Qwen3.8-Flash-Next-NVFP4` at revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`.
  `model/config/qwen3_8_flash_next_nvfp4.json` is its `config.json`, kept verbatim.
- **Hugging Face capture upload.** The `diverse_100` capture goes to the dataset
  `UW-SyFI/servingstudio-workload` at `qwen38_flash_next_nvfp4/vllm_mtp_k3/diverse_100/capture/20261013/`:
  `manifest.json`, `routes.u16` (135,144 accepted tokens × 49 layers × top-10, 132,441,120 B), `trace.csv` with
  the measured per-request acceptance, and `provenance.json` (corpus pass job 6682). The public preset and the
  alignment pack read it at dataset commit `b4b749dc7f3fe4a8f2db45b76cfbabdba59515e0`; every uploaded file's sha256
  matches the local copy.
- **`profiling/profile.db`** gains 7,109 B200 rows, all measured by this branch, merged row-wise onto `main`'s DB:

  | Kind | Rows |
  |---|---:|
  | `elementwise` (`torch_vllm`, `triton`, `torch_strided_copy_vllm`, `torch_cat_vllm`) | 1,330 |
  | `dsa_persistent_topk_decode` (`vllm_cuda`, `vllm_cuda_cooperative`) | 1,044 |
  | `single_gemm` (`torch_linear_vllm`, BF16) | 1,020 |
  | `sparse_indexer_logits_decode` | 964 |
  | `sparse_gqa_paged_attention` | 868 |
  | `nvfp4_fused_moe` (NVFP4 and FP8 block, `renormalize_naive`, corpus routing) | 216 |
  | `ple_short_conv` | 195 |
  | `rms_norm` (`vllm_gemma_native`) | 126 |
  | `indexed_row_copy` (`torch_vllm`) | 123 |
  | `gdn_recurrent_decode_gated_norm` | 104 |
  | `sparse_indexer_logits_prefill` | 80 |
  | `embedding_row_gather` | 78 |
  | `gdn_causal_conv_decode` / `gdn_recurrent_decode` (verify rows) | 78 |
  | `kv_cache_append` | 70 |
  | `hc_combine`, `hc_combine_norm`, `hc_gate_mix`, `grouped_rms_norm`, `scaled_silu` | 68 each |
  | `qk_rms_norm_rope_gate`, `silu_and_mul` (`vllm_cuda`), `nvfp4_quant`, `fp8_per_token_group_quant` | 68 each |
  | `sparse_indexer_prepare`, `sparse_index_expand` | 68 each |
  | `ple_ngram_ids` | 40 |
  | `ple_gate` | 25 |

  It also gains 20 profiling-run records. No row of `main` is changed or removed; the GDN decode tables gain a
  `num_tokens` column, and their existing rows read the same.

## Contribution licensing

- [ ] I have read the project's CLA.
- [ ] I have the right to submit this contribution.
- [ ] I have disclosed any third-party code or licensing restrictions.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
