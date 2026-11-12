# feat(qwen3.8): Qwen3.8-27B-NVFP4 on vLLM, one B200, with kernel alignment and the 20-case campaign

## Purpose

Model `nvidia/Qwen3.8-27B-NVFP4` as the vLLM fork serves it on one B200. The model is
`Qwen3_5ForConditionalGeneration`, served text-only:
- 64 layers, `[GDN ×3, gated GQA]` ×16, hidden 5120, FFN 17408;
- Gated DeltaNet: 16 K heads / 48 V heads, head dim 128;
- gated GQA: 24 Q / 4 KV heads, head dim 256;
- ModelOpt `MIXED_PRECISION`:
  - FP8 per-tensor for the GDN and GQA projections;
  - NVFP4 for the dense SwiGLU MLP and the lm_head;
  - BF16 for `in_proj_ba`;
- FP8 KV cache.

The model is checked against vLLM captures: `diverse_100` and 20 operating points over chunk 2048, 4096 and
8192.

- **New L1 kinds:**
  - `multi_add_rms_norm`, `silu_and_mul` and `static_fp8_quant`. These are vLLM's Inductor-fused norm,
    activation and quant kernels, profiled through one shared runner, `profiling/runners/vllm_inductor.py`.
  - `qk_rms_norm_rope_gate` (`vllm_triton`).
  - `indexed_row_copy` (torch: gather, scatter and masked zero), for the GDN state copies.
  - `paged_attn_batch_decode` and `paged_attn_batch_prefill` (`flashinfer_trtllm_gen`). They price
    FlashInfer's trtllm-gen FMHA over an FP8 KV cache.
    - Decode is keyed on batch, max KV length and fill.
    - Prefill is keyed on sequences, query tiles and equivalent prefix.
- **New backends on existing kinds:**
  - `single_gemm` / `flashinfer_fp8_tensor` (`bmm_fp8`, auto). Its m axis brackets FlashInfer's hybrid
    autotune buckets and adds a step-64 scan.
  - `gdn_recurrent_decode` / `vllm_triton_sigmoid_gating`, the decode rows of a mixed step.
  - `gdn_chunk_delta_rule` / `flashinfer` on SM100.
  - `elementwise` / `torch_strided_copy` and `torch_cat`, for GDN's strided core-output copy and its cat.
- **Worklets and the arch.**
  - `qwen35_vllm_nvfp4_gdn_dense` and its config reader `qwen35_model_cfg`. Every dimension and every module's
    precision comes from `config.json` and `quantization_config`. Other precision mixes fail to fit.
  - Three worklets:
    - `vllm_qwen35_gdn_local`: every measured GDN launch, including the glue copies;
    - `vllm_qwen35_gated_gqa_local`;
    - `vllm_nvfp4_swiglu_mlp_local`.
  - FULL_AND_PIECEWISE graph padding:
    - The compiled pieces run at the padded rows.
    - A FULL-graph decode prices its conv and recurrent kernels at the real rows, because padded rows carry
      `PAD_SLOT_ID` and return early.
  - KV follows vLLM's hybrid allocator. The attention block is 1568 tokens, and each request also holds six
    GDN state blocks. This gives 29.13x concurrency at 131072 tokens; vLLM logs 29.14x.
- **Shared simulator changes:**
  - `kv_layout::hybrid_attention_block_tokens` and `hybrid_attention_state_layout`, shared with Qwen3.6 and
    GLM-5.3-Flash.
  - `GdnPrefillOp` takes `decode_rows`, so a mixed step's decode rows join the same varlen conv. A value of 0
    keeps the existing callers' cost.
  - The never-measured `trt` backend is removed from `flashinfer_attn_{prefill,decode,rect}`.
- **model.work:**
  - ModelOpt `MIXED_PRECISION` labeling; layer types come from the config.
  - The Qwen3.5 dense builder now counts its RMSNorm weights.
  - The served FP8 KV is applied to GQA through `_with_fp8_kv_cache`.
  - The `qwen35_vllm_nvfp4_gdn_dense` location map, with 35 rows.
- **Launcher and profiler:**
  - An alignment pack can declare one calibrated top-level `gpu_time_multiplier`.
  - `record_extraction` rejoins a structured record that another process's log line split.
- **Presets.**
  - Public arch and sim presets: chunk 2048, replicas 1 / 2, KV pool 125.52 GiB, multiplier 1.0483 (from
    `diverse_100`).
  - Catalog entries.
- **Alignment.** The `qwen38_27b_nvfp4_b200_tp1` pack covers 20 operating points:
  - cases 01-15 at chunk 2048, with KV pools pinned for 14 and 15;
  - cases 16-20 at chunk 4096 and 8192, with `max_cudagraph_capture_size` equal to the chunk.

  Every case uses the single `diverse_100` multiplier.
- **Changes that reach other models** are listed in [`README.md`](README.md#changes-that-reach-other-models).
  The only predictions that move are GLM-5.3-Flash's, by -0.9% to +0.5%, from the re-profiled B200 GDN
  causal-conv rows.

[`README.md`](README.md) has the modeling notes, the evidence and the open items.
[`alignment_matrix.md`](alignment_matrix.md) is the generated campaign matrix.

## Test Plan

Test tiers at the branch head. `origin/main` (54449112) has no new commits, so no merge was needed:

```bash
just test-cpu
just test-presets
just test-gpu    # on one B200, through Slurm
```

Also run:

- **Kernel alignment on `diverse_100`**: the nsys capture (job 6443) through `alignment timing-predict` and
  `alignment analyze`.
- **Simulation of `diverse_100`**, compared by the E2E Analyzer with the warm workload pass (job 6506, same
  server args).
- **The 20-case campaign** through `alignment-campaign`, from capture to `compare`, at the branch head.
- **Qwen3.6 bit-identity** across the shared KV-layout and `GdnPrefillOp` changes. A timing-predict on
  `qwen36_local` H200 gives an identical cost-log parquet and `iter_breakdown.json` before and after.
- **R6/R7** on one exact `batch_locked` iteration of a Qwen3.8 run. The per-kernel minimum FLOPs and bytes sum
  to the model.work label with zero relative error.

## Test Result

- `just test-cpu`: pytest 4796 passed, 19 skipped; Rust 1473 passed.
- `just test-presets`: passed. Every public member, Qwen3.8 included, is fully measured in the committed
  profile.db.
- `just test-gpu` (job 6577): 14 passed, 3 skipped.
- **Kernel alignment on `diverse_100`.** The signed error of the simulated critical path against the measured
  one, per iteration:

  | Deployment | Decode median | Mixed median | All median | All total | Coverage | Multiplier |
  |---|---:|---:|---:|---:|---:|---:|
  | Qwen3.8-27B-NVFP4 TP1 | +4.50% | -1.34% | +4.41% | +1.07% | 99.05% | 1.0483 |

- **Simulation of `diverse_100`** at the multiplier (100 requests):

  | Deployment | Server TTFT mean | Server TPOT mean | E2E mean | Output throughput |
  |---|---:|---:|---:|---:|
  | Qwen3.8-27B-NVFP4 TP1 | -7.08% | +0.36% | +2.47% | -3.98% |

- **Campaign**, all 20 cases, at `gpu_time_multiplier` 1.0483. The table is the output of
  `alignment-campaign compare --markdown`, copied from [`alignment_matrix.md`](alignment_matrix.md):

  | Case | kernel_abs_pct | kernel_signed_pct | map_coverage_pct | e2e_mean_pct | output_tps_pct | server_tpot_mean_pct | server_ttft_mean_pct |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | tp1/01_micro_throughput_c256 | +3.71% | +0.75% | 97.79% | -0.50% | +0.49% | +2.49% | +26.90% |
  | tp1/02_graph_boundaries_c128 | +3.74% | -1.13% | 97.98% | -6.64% | +7.22% | -2.33% | +4.11% |
  | tp1/03_balanced_anchor_c32 | +2.49% | -1.85% | 98.20% | -4.38% | +3.78% | -2.77% | -10.65% |
  | tp1/04_prefill_spectrum_c32 | +1.18% | +0.21% | 98.99% | -7.15% | +7.73% | -4.66% | -10.62% |
  | tp1/05_decode_spectrum_c64 | +5.61% | -5.55% | 97.48% | -5.69% | +4.77% | -6.79% | -12.60% |
  | tp1/06_long_long_c8 | +3.85% | -3.85% | 99.37% | +3.85% | -4.16% | +4.87% | -3.83% |
  | tp1/07_quadrant_interference_c48 | +3.71% | -3.26% | 98.03% | -5.88% | +5.56% | -6.63% | -19.35% |
  | tp1/08_completion_churn_c96 | +4.71% | -3.93% | 97.94% | -5.52% | +4.85% | -9.30% | -0.02% |
  | tp1/09_rate_knee | +2.71% | -2.19% | 98.30% | -2.47% | -0.34% | -2.62% | -13.68% |
  | tp1/10_full_saturation_rate | +1.54% | -0.55% | 98.95% | -0.79% | +0.41% | -1.02% | -2.29% |
  | tp1/11_long_prefill_ladder_c1 | +3.34% | -2.24% | 99.04% | -2.31% | +2.38% | +11.09% | -3.46% |
  | tp1/12_ultra_context_decode_c1 | +3.15% | -2.00% | 99.03% | +9.76% | -8.89% | +11.51% | -3.67% |
  | tp1/13_generation_length_matrix_c8 | +1.74% | -1.04% | 98.75% | +7.66% | -7.76% | +7.54% | -6.43% |
  | tp1_chunk4k/16_prefill_spectrum_chunk4k_c32 | +3.78% | -3.55% | 99.45% | -3.50% | +3.57% | -1.75% | -4.06% |
  | tp1_chunk4k/18_quadrant_interference_chunk4k_c48 | +4.40% | -3.80% | 98.76% | -4.15% | +3.79% | -3.41% | -3.37% |
  | tp1_chunk8k/17_prefill_spectrum_chunk8k_c32 | +2.77% | -2.70% | 99.65% | -0.59% | +0.58% | +1.16% | -0.61% |
  | tp1_chunk8k/19_quadrant_interference_chunk8k_c48 | +2.86% | -2.77% | 99.45% | -2.25% | +1.96% | -0.51% | +1.16% |
  | tp1_chunk8k/20_full_saturation_chunk8k | +3.64% | -3.52% | 99.09% | -1.50% | +1.59% | -1.28% | +1.35% |
  | tp1_kv14/14_near_full_long_decode_c64 | +4.21% | -4.07% | 99.23% | -0.85% | +0.27% | -0.83% | -2.75% |
  | tp1_kv15/15_kv_overcommit_mixed_churn_c32 | +3.91% | -3.68% | 99.36% | +3.06% | -3.91% | +3.06% | -0.26% |

  39 metrics are out of tolerance:
  - **Iteration population** of cases 01, 02, 04, 09, 15, 16 and 17. decode_batch, iterations,
    prefill_tokens and scheduled_kv are off by 1.2-2.6% against 1%.
  - **Server TTFT is low** in cases 03-05, 07 and 09 (-10.6% to -19.4%). Prefill-carrying PIECEWISE steps
    carry a 7-43 ms additive host gap that the single multiplier only averages.
  - **TPOT and E2E are high** in cases 11-13 (TPOT +7.5% to +11.5%, E2E up to +9.8%). nsys stretches c1/c8
    decode kernels about 3.5% over the workload pass, and the multiplier adds 4.8% to FULL-graph steps that
    have no host gap.
  - **Case 05 kernel -5.55%.** Small-M decode GEMMs run 9-21% slower in serving than the profiled rows. The
    serving process picks its runner from its own autotune cache (nvjet at B <= 16, CUTLASS at 17-64), and the
    profiler re-tunes in its own process.
  - **Case 01 server TTFT +26.9%** on a clean recapture. The cause has not been found.

[`README.md`](README.md) has the per-operation breakdown and the remaining gaps. This is review evidence, not
an accepted baseline, and no golden is recorded.

## Dependencies

- **vLLM fork** `servingstudio-alignment` at `892da082`, the same commit `main` uses.
- **Checkpoint** `nvidia/Qwen3.8-27B-NVFP4` at revision `482ca0f3832238542f8f5295dde86b5f22711d80`.
  `model/config/qwen3_8_27b_nvfp4.json` is byte-identical to its `config.json`.
- **No Hugging Face capture upload.** This is a dense model with no speculation, so the presets read no
  routing corpus.
- **`profiling/profile.db`** gains 4,053 B200 rows, all measured by this branch, merged row-wise onto
  `main`'s DB:

  | Kind | Rows |
  |---|---:|
  | `paged_attn_batch_prefill` | 1,080 |
  | `single_gemm` | 731 |
  | `paged_attn_batch_decode` | 701 |
  | `elementwise` | 433 |
  | `multi_add_rms_norm` | 340 |
  | `gdn_causal_conv_prefill` / `_decode` | 153 |
  | `nvfp4_quant` | 136 |
  | `static_fp8_quant` | 136 |
  | `kv_cache_append` | 70 |
  | `qk_rms_norm_rope_gate` | 68 |
  | `silu_and_mul` | 68 |
  | `indexed_row_copy` | 42 |
  | `gdn_chunk_delta_rule` | 42 |
  | `gdn_prefill_post_conv` | 20 |
  | `gdn_recurrent_decode` | 18 |
  | `gdn_gated_rms_norm` | 15 |

  It also gains 13 profiling-run records.
  - **459 existing B200 GDN causal-conv rows are re-measured** with vLLM 892da082's state-first conv-state
    layout. GLM-5.3-Flash reads them; its members move by -0.9% to +0.5%.
  - No row is removed.

## Contribution licensing

- [ ] I have read the project's CLA.
- [ ] I have the right to submit this contribution.
- [ ] I have disclosed any third-party code or licensing restrictions.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
