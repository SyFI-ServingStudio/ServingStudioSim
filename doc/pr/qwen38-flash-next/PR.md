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
  - Public arch and sim presets: chunk 2048, replicas 1 / 2, KV pool 85.6846 GB (79.80 GiB), multiplier 1.0,
    acceptance from the capture's per-request `accept_rate`.
  - Catalog entries.
- **Alignment.** The `qwen38_flash_next_nvfp4_b200_tp1` pack covers 20 operating points:
  - cases 01-15 at chunk 2048, with KV pools pinned for 14 and 15;
  - cases 16-20 at chunk 4096 and 8192, with `max_cudagraph_capture_size` equal to the chunk;
  - MTP acceptance by output position, measured from each case's own workload pass.

  Every case runs at `gpu_time_multiplier` 1.0.

[`README.md`](README.md) has the modeling notes, the evidence and the open items.
[`alignment_matrix.md`](alignment_matrix.md) is the generated campaign matrix.

## Test Plan

Test tiers at the branch head. `origin/main` (3df15294) is merged; it has no newer commits:

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
  - the Flash-Next public preset test, which reads the `PENDING_UPLOAD` corpus path until the Hugging Face upload.
- `just test-presets`: <<FINAL: passes once the Hugging Face sha replaces `PENDING_UPLOAD`>>.
  `test_every_member_is_measured` passes: every public member is fully measured in the committed profile.db.
- `just test-gpu`: <<FINAL: Slurm job 6756 result at HEAD>>.
- **Kernel alignment on `diverse_100`** (1710 iterations). The signed error of the simulated critical path against
  the measured one, per iteration. Source: `logs/20261013_15_qf_campaign_final/d100/summary.md`:

  | Deployment | Decode median | Mixed median | All median | All total | Coverage | Recommended multiplier |
  |---|---:|---:|---:|---:|---:|---:|
  | Qwen3.8-Flash-Next-NVFP4 TP1 MTP-3 | -3.0% | +2.6% | -2.0% | +0.87% | 99.37% | 1.204 (not applied) |

  The shipped multiplier is 1.0. The recommended 1.204 is the measured GPU cycle over the measured kernel time: it
  would absorb the mixed-step host launch time (below), not a kernel error, and as a uniform scale it would also
  stretch FULL-graph decode steps that have almost no host gap.

- **Simulation of `diverse_100`** at multiplier 1.0 (100 requests, untrimmed trace: requests 34 and 41, which end at
  `max_model_len`, are served). Source: `logs/20261013_15_qf_campaign_final/d100/summary.md` and
  `d100/metrics_d100.json`:

  | Deployment | Server TTFT mean | Server TPOT mean | E2E mean | Output throughput |
  |---|---:|---:|---:|---:|
  | Qwen3.8-Flash-Next-NVFP4 TP1 MTP-3 | -8.2% | -20.8% | -8.4% | +2.5% |

  Verify rounds match (41,757 measured, 41,760 simulated). TPOT misses through the mixed-step host launch time and
  six short-output requests (3-33 tokens) whose few rounds landed on long mixed steps: they measure 64-107 ms TPOT
  and carry about 3.0 of the 3.9 ms mean gap.

- **Campaign**, all 20 cases, at `gpu_time_multiplier` 1.0. The table is the output of
  `alignment-campaign compare --markdown`, copied from [`alignment_matrix.md`](alignment_matrix.md)
  (`logs/20261013_15_qf_campaign_final/compare.md`):

  | Case | kernel_abs_pct | kernel_signed_pct | map_coverage_pct | e2e_mean_pct | output_tps_pct | server_tpot_mean_pct | server_ttft_mean_pct |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | tp1/01_micro_throughput_c256 | +2.95% | +2.84% | 99.57% | -7.94% | +8.78% | -3.10% | -7.92% |
  | tp1/02_graph_boundaries_c128 | +2.06% | +1.46% | 99.52% | -16.60% | +19.53% | -13.52% | -16.37% |
  | tp1/03_balanced_anchor_c32 | +1.87% | +0.12% | 99.47% | -6.66% | +7.58% | -5.62% | -2.79% |
  | tp1/04_prefill_spectrum_c32 | +1.58% | +1.48% | 99.41% | -3.62% | +4.02% | -4.88% | +1.86% |
  | tp1/05_decode_spectrum_c64 | +3.10% | -2.78% | 99.56% | -4.23% | +12.83% | -3.65% | -12.66% |
  | tp1/06_long_long_c8 | +2.75% | +2.75% | 99.41% | -3.06% | +3.13% | -1.83% | -4.64% |
  | tp1/07_quadrant_interference_c48 | +2.60% | -1.46% | 99.51% | -4.17% | +19.65% | -6.23% | -0.29% |
  | tp1/08_completion_churn_c96 | +1.99% | -0.39% | 99.39% | -2.70% | +20.48% | -3.33% | -5.03% |
  | tp1/09_rate_knee | +4.28% | -2.90% | 99.47% | -13.10% | +3.51% | -14.47% | -18.76% |
  | tp1/10_full_saturation_rate | +1.16% | -0.38% | 99.39% | -8.36% | +10.74% | -6.91% | -9.64% |
  | tp1/11_long_prefill_ladder_c1 | +5.43% | +5.43% | 99.30% | +1.70% | -1.65% | -0.82% | +2.50% |
  | tp1/12_ultra_context_decode_c1 | +4.97% | +4.97% | 99.27% | -1.31% | +1.33% | -1.88% | +2.23% |
  | tp1/13_generation_length_matrix_c8 | +2.46% | +2.19% | 99.37% | -3.66% | +3.10% | -3.81% | +1.32% |
  | tp1_chunk4k/16_prefill_spectrum_chunk4k_c32 | +0.61% | +0.25% | 99.77% | -3.77% | +3.91% | -4.80% | -0.88% |
  | tp1_chunk4k/18_quadrant_interference_chunk4k_c48 | +1.53% | -1.34% | 99.44% | -1.66% | +18.15% | -0.67% | -4.83% |
  | tp1_chunk8k/17_prefill_spectrum_chunk8k_c32 | +1.17% | -0.84% | 99.33% | -2.88% | +2.74% | -4.11% | +0.62% |
  | tp1_chunk8k/19_quadrant_interference_chunk8k_c48 | +2.01% | -0.06% | 99.23% | -3.61% | +10.30% | -4.36% | +0.01% |
  | tp1_chunk8k/20_full_saturation_chunk8k | +1.67% | -1.46% | 99.32% | -4.87% | +9.29% | -4.92% | -5.22% |
  | tp1_kv14/14_near_full_long_decode_c64 | +2.44% | +2.44% | 99.43% | -12.67% | +15.05% | -11.29% | -22.44% |
  | tp1_kv15/15_kv_overcommit_mixed_churn_c32 | +2.83% | +2.82% | 99.43% | -13.64% | +12.82% | -15.85% | -8.36% |

  101 metrics are out of tolerance ([`alignment_matrix.md`](alignment_matrix.md), "Out of tolerance"). Mapping
  coverage is 99.23-99.77% in every case (floor 97%). The causes, from
  `logs/20261013_15_qf_campaign_final/summary.md`:
  - **E2E and TPOT sim-fast: mixed-step host launch time** (02, 09, 10, 14, 15; E2E and tps in 01; the
    iteration-cycle misses in 02, 09, 14 and 15). A PIECEWISE mixed step issues about 2,100 CUDA API calls and one
    pageable H2D copy plus `cudaStreamSynchronize` in preprocess. The measured mixed-step GPU cycle is 1.07-1.36x
    its kernel time, while the simulated mixed kernels are within +0.2% to +3.9%. The simulator has no host-time
    mechanism and the multiplier is 1.0.
  - **Output throughput high, iteration population off: per-request acceptance spread** (07, 08, 18, 19; part of
    05, 10 and 20). The by-position chain fixes the mean, but the last measured finishers run at 1.95-2.7 tokens per
    round against a case mean of 3.1-3.25. vLLM drains them in pure-decode steps the simulator does not run: p90
    completion matches within ±5%, while makespan is 9-17% short. Replaying per-request acceptance would condition
    on the observation.
  - **Server TTFT low** (02, 05, 09, 14). vLLM's async scheduler takes about twice as long from first schedule to
    first token (for example 117 vs 55 ms in 02); in case 14 the simulator also drains its queue faster.
  - **Case 11 kernel +5.43%.** Its mixed stage is single-request 2048-token prefill, where mixed `fused_moe` (+14%)
    and `gdn.qkvz` (+15%) read high: the expert-histogram layer fold with a generated-only routing corpus, and the
    profiler's duty cycle at m=2048 (Known gaps in [`README.md`](README.md)).
  - **Workload-structure 1% bands** (03, 04, 11, 12, 13, 16, 17): the pure-decode iteration count follows the
    acceptance spread (case 11: 331 vs 320 decode iterations), and in saturated bursts (01, 02) vLLM refills freed
    slots over more, smaller prefill steps.
  - The fork's per-iteration log is shifted by one to two steps, so logged decode steps before a mixed step carry
    part of its time (totals conserved). Per-step cycle strata read this artifact; request-level metrics do not.

[`README.md`](README.md) has the per-operation breakdown and the remaining gaps. This is review evidence, not an
accepted baseline, and no golden is recorded.

## Dependencies

- **vLLM fork** `servingstudio-alignment`. The submodule stays at `892da082`, the commit `main` uses. The captures
  ran at `e1423a4d`, which is `892da082` plus one commit that isolates the Qwen MTP aux config and records the
  drafter's routed experts (routed-expert capture only; no kernel changes). <<FINAL: push e1423a4d to the fork and
  decide whether the pointer moves>>
- **Checkpoint** `nvidia/Qwen3.8-Flash-Next-NVFP4` at revision `fc694b54fb0174e0913e6adf86691ef85a4ead47`.
  `model/config/qwen3_8_flash_next_nvfp4.json` is its `config.json`, kept verbatim.
- **Hugging Face capture upload (pending).** The `diverse_100` capture goes to the dataset
  `UW-SyFI/servingstudio-workload` at `qwen38_flash_next_nvfp4/vllm_mtp_k3/diverse_100/capture/20261013/`:
  `manifest.json`, `routes.u16` (135,144 accepted tokens × 49 layers × top-10, 132,441,120 B), `trace.csv` with
  the measured per-request acceptance, and `provenance.json` (corpus pass job 6682). The public preset reads it at
  `@PENDING_UPLOAD`. <<FINAL: dataset commit sha, then pin it in
  presets/public/Qwen3.8-Flash-Next-NVFP4/qwen4_exp_vllm_nvfp4_gdn_qsa_moe_speculative.yaml>>
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
