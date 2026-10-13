# feat(mimo): MiMo-V2.6-Flash-RL on vLLM, B200 TP4/EP4, with kernel alignment and the 19-case campaign

## Purpose

Model MiMo-V2.6-Flash-RL (`XiaomiMiMo/MiMo-V2.6-Flash-RL`; 48 layers, hidden 4096, 256 routed experts top-8)
as the vLLM fork serves it on four B200s:
- TP4 attention and EP4 routed experts over the same ranks;
- 9 global and 39 sliding-window (128-key) GQA layers with asymmetric QK 192 / V 128 heads and per-head sinks;
- FP8 block-quantized QKV and dense FFN;
- MXFP4 experts with BF16 activations;
- MNNVL fused all-reduce + RMSNorm.

Check the model against vLLM captures: `diverse_100` and 19 operating points over chunk 2048, 4096 and 8192.

- **L1 kinds and backends.**
  - One new kind, `fa4_varlen_attention` / `vllm_fa4`: vLLM's FA4 varlen call (FLASH_ATTN_DIFFKV).
    - It is priced in CTA waves and whole 128-key window tiles.
    - Rows are keyed on `cuda_graph`: a graph replay fixes the split choice at capture over `max_model_len`,
      an eager call picks it from the real batch. The capture bound is not a key.
    - Prefill knots sit on FlashPrepareScheduler's split cliffs.
    - It shares the FA4 batch projection with `rel_bias_attention`.
  - New backends on existing kinds:
    - `nvfp4_fused_moe` / `flashinfer_trtllm_sm100_mxfp4_bf16`: MXFP4 weights × BF16 activations, routing
      inside the call;
    - `all_reduce_residual_rms_norm` / `flashinfer_mnnvl`;
    - `kv_cache_append` / `vllm_triton_diffkv`, with an optional `head_dim_v` arg.
  - `#[compute_dtype_override]` lets a kernel config override its compute dtype from an optional field.
  - Fabric-memory profiling backends get the host's IMEX channel inside the profiler container.
- **Worklets and the arch.**
  - The `mimo_v2_vllm_fp8_swa_moe` arch, its model config reader and the attention worklet. Layer groups:
    `global_dense` ×1, `swa_moe` ×39, `global_moe` ×8.
  - The routed MoE reuses `Nvfp4MoeLocalWorklet` through a new `activation_format` (weight-only experts, no
    quant leaf).
  - KV follows vLLM's hybrid allocator (sliding-window groups padded to whole groups).
- **`TpAllReduceRmsNormOp`** (`simulator/src/op/comm/tp_all_reduce_rms_norm.rs`, shared op). Each TP boundary
  runs vLLM's fused MNNVL all-reduce + residual + RMSNorm up to its token cap (4096 at TP4). Above the cap it
  runs the unfused path, NCCL all-reduce then the Triton residual add + RMSNorm, instead of refusing the step.
  At TP1 only the norm runs. Only MiMo uses it.
- **model.work.**
  - A `SlidingWindowGQA` spec, plus GQA `v_head_dim` and `attention_sinks`.
  - FP8 `ignored_layers` and MXFP4 store dtype in `quantization.py`.
  - The MiMo accountant.
  - The TP4/EP4 location map, and a Rust test that checks it against the compiled locations.
- **Presets.**
  - Public arch and sim presets: replicas 1 / 2, the measured KV pool and multiplier.
  - Catalog entries in `model/catalog.yaml` and `model/arch_catalog.yaml`.
  - Routing reads the `diverse_100` token corpus in `UW-SyFI/servingstudio-workload` at `18506fd7`.
  - The timing-predict case presets moved to the experiment tree.
- **Alignment.** The `mimo_v26_flash_b200_tp4` pack covers 19 operating points:
  - cases 01-15: GLM-5.2's cases scaled to vLLM's `--max-num-seqs` 100 and a 131072 context;
  - cases 16-19: the prefill spectrum and IO quadrants at chunk 4096 and 8192, so steps cross the fused
    all-reduce cap;
  - warmed passes, with CUDA graphs captured to 2048 tokens;
  - KV pools pinned for cases 14/15 and measured per chunk for 16-19;
  - the default tolerances and the label rules, including the unfused boundary's kernels.
- **Changes that reach other models** are listed in [`README.md`](README.md#changes-that-reach-other-models).
  The one that moves a prediction is the refresh of five `single_gemm` / `torch_linear_vllm` rows (n 4096,
  k 2048, m 1-16), which GLM-5.3-Flash also reads.

[`README.md`](README.md) has the modeling notes, the evidence and the open items.
[`alignment_matrix.md`](alignment_matrix.md) is the generated campaign matrix.

## Test Plan

Test tiers at the branch head. `origin/main` (8ed83900) had no new commits, so no merge was needed:

```bash
just test-cpu
just test-presets
just test-gpu    # on one B200, through Slurm
```

Also run:

- **Kernel alignment on `diverse_100`** (`launcher alignment timing-predict` + `alignment analyze`) on the
  nsys capture, job 6264.
- **Simulation of `diverse_100`**, compared by the E2E Analyzer with the workload pass, job 4223 (same server
  args).
- **The 19-case campaign** through `alignment-campaign`, from capture to `compare`. The CPU phases of all 19
  cases were rerun at the branch head.
- **FA4 graph against eager rows** (job 6419, no DB writes), over global and SWA layers and context.
- **Cross-model checks** for the shared FA4 projection, `TpAllReduceRmsNormOp` and the o_proj rows. Each runs
  simulations before and after:
  - Inkling-NVFP4 campaign cases 01 and 10;
  - MiMo campaign cases 02, 10 and 11 under the fused cap;
  - GLM-5.3-Flash-NVFP4 DP8-attn/EP8 at c512;
  - GLM-5.3-Flash TP4 campaign case 12.
- **Cache fidelity** of `fa4_varlen_attention` against 94 off-grid B200 probes.

Not run: the product path, i.e. live predict and simulation on the public members through a local public API
and the Intro site.

## Test Result

- `just test-cpu`: pytest 4595 passed, 14 skipped; Rust 1439 passed.
- `just test-presets`: 9 passed. Every public member, MiMo included, is fully measured in the committed
  profile.db.
- `just test-gpu` (job 6417, before the unfused all-reduce op): 14 passed, 3 skipped. The throughput
  regression monitor raised no drift warning.
- **Kernel alignment on `diverse_100`.** The signed median of the simulated critical path against the
  measured one, per iteration:

  | Deployment | Decode | Mixed | All | Coverage | Multiplier | Source |
  |---|---:|---:|---:|---:|---:|---|
  | MiMo-V2.6-Flash-RL TP4/EP4 | +6.26% | -0.44% | +6.01% | 99.26% | 1.1103 | `logs/20261012_8_mimo_step11e/diverse100/check1_derived.md` |

- **Simulation of `diverse_100`** at the recommended multiplier (100 requests, all ids shared):

  | Deployment | TTFT client mean | TPOT client mean | E2E mean | Throughput | Source |
  |---|---:|---:|---:|---:|---|
  | MiMo-V2.6-Flash-RL TP4/EP4 | -21.9% | -1.9% | +2.7% | -3.8% | `logs/20261012_8_mimo_step11e/diverse100/e2e_table.md`, `mrec` |

  Server TTFT is -10.0%. The measured client TTFT runs 200 ms above the server's, and the simulator has no
  client leg.
- **Campaign**, all 19 cases. The table is the output of `alignment-campaign compare --markdown`, copied from
  [`alignment_matrix.md`](alignment_matrix.md). Each value is simulated against measured, at
  `gpu_time_multiplier` 1.0:

  | Case | kernel_abs_pct | kernel_signed_pct | map_coverage_pct | e2e_mean_pct | output_tps_pct | server_tpot_mean_pct | server_ttft_mean_pct |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | tp4_ep4/01_micro_throughput_c100 | +5.05% | -1.36% | 98.91% | -7.53% | +7.68% | -5.38% | -0.24% |
  | tp4_ep4/02_graph_boundaries_c100 | +5.17% | -2.58% | 98.72% | -7.65% | +6.99% | -5.58% | +2.01% |
  | tp4_ep4/03_balanced_anchor_c32 | +2.25% | -0.45% | 99.07% | -1.70% | +1.33% | -0.16% | -7.55% |
  | tp4_ep4/04_prefill_spectrum_c32 | +4.08% | -3.28% | 99.48% | -5.12% | +5.24% | -2.66% | -8.59% |
  | tp4_ep4/05_decode_spectrum_c64 | +2.55% | -2.31% | 98.93% | -1.62% | -0.35% | -2.22% | +1.87% |
  | tp4_ep4/06_long_long_c8 | +2.76% | -0.16% | 99.43% | +3.44% | -3.71% | +4.13% | +0.92% |
  | tp4_ep4/07_quadrant_interference_c48 | +1.87% | -1.62% | 99.03% | -0.17% | -0.47% | -1.39% | -3.29% |
  | tp4_ep4/08_completion_churn_c96 | +2.19% | -1.24% | 98.93% | -0.65% | -0.31% | -1.40% | +1.43% |
  | tp4_ep4/09_rate_knee | +3.89% | +3.49% | 98.94% | -3.46% | -0.13% | -3.28% | -9.82% |
  | tp4_ep4/10_full_saturation_rate | +4.32% | -3.86% | 99.11% | -3.69% | +2.50% | -4.24% | -1.71% |
  | tp4_ep4/11_long_prefill_ladder_c1 | +3.78% | -0.71% | 99.44% | -5.67% | +6.04% | +3.41% | -4.90% |
  | tp4_ep4/12_ultra_context_decode_c1 | +5.04% | +0.51% | 99.26% | +4.48% | -4.28% | +5.75% | -5.12% |
  | tp4_ep4/13_generation_length_matrix_c8 | +3.15% | +0.08% | 99.24% | +5.64% | -7.01% | +4.72% | -5.10% |
  | tp4_ep4/16_prefill_spectrum_chunk4k_c32 | +4.25% | -3.75% | 99.59% | -5.51% | +5.71% | -3.90% | -5.16% |
  | tp4_ep4/17_prefill_spectrum_chunk8k_c32 | +2.52% | +0.26% | 99.77% | -0.49% | +0.47% | +1.52% | -0.68% |
  | tp4_ep4/18_quadrant_interference_chunk4k_c48 | +3.03% | -2.94% | 99.05% | -1.61% | +1.00% | -4.15% | -1.02% |
  | tp4_ep4/19_quadrant_interference_chunk8k_c48 | +2.87% | -1.91% | 99.32% | -1.54% | +0.92% | -0.79% | -2.17% |
  | tp4_ep4_kv14/14_near_full_long_decode_c100 | +3.05% | -2.59% | 99.63% | -0.21% | -0.31% | -0.02% | -3.76% |
  | tp4_ep4_kv15/15_kv_overcommit_mixed_churn_c100 | +1.47% | -1.18% | 99.74% | +2.49% | -2.96% | +2.44% | +2.50% |

  247 judgements: 29 FAIL and 10 WARN. Out of tolerance:
  - **Cases 01 and 02.**
    - E2E is -7.53% / -7.65% (tolerance 7).
    - Kernel abs is 5.05% / 5.17% (tolerance 5). Signed is only -1.36% / -2.58%; the abs error is per-step
      MoE routing scatter at c100.
  - **Case 12 kernel abs 5.04%.** Its c1 decode steps read +6.8% for the same rows as `diverse_100` decode
    (router split-K, cold-L2 small launches).
  - **The iteration population of cases 01, 02, 04, 09, 16 and 17.** decode_batch, iterations,
    prefill_tokens and scheduled_kv are off by 3.2-3.3%, 1.9%, 2.5%, 2.2%, 1.7-1.9% and 1.5-1.6% against 1%.

  The E2E and population items come from per-step host time that the campaign leaves out by design:
  - The campaign renders the multiplier at 1.0.
  - At c100 the measured duty-cycle multiplier is 1.118 / 1.132 on 01 / 02, against about 1.02 elsewhere.
  - In 04, 09, 16 and 17, closed-loop requests re-enter 2-3 steps late (client/frontend round trip).

  A deployment preset absorbs this host time through the recommended `gpu_time_multiplier`, as the
  `diverse_100` simulation above shows.
- **The unfused boundary at chunk 8192** (cases 17 and 19). Above 4096 tokens vLLM runs NCCL all-reduce, then
  Inductor's residual add + RMSNorm; the FP8 group quant kernel still runs separately after it, so the
  simulator's quant leaves are not double-counted. Case 17 kernel signed error: decode -1.93%, mixed +0.61%.
  The NCCL slots read -3.2%; the quant-fused norm (-33%) and the mixed-step rope (-46%) are underpriced,
  listed in the README's known gaps.

[`README.md`](README.md) explains the `diverse_100` decode gap (+6.26%: padded-row MoE charge, router and
small-launch rows) and the remaining gaps. This is review evidence, not an accepted baseline. No golden is
recorded.

## Dependencies

- **vLLM fork** `servingstudio-alignment` at `892da082`, the same commit `main` uses.
- **Token corpus**: `mimo_v26_flash_rl/vllm/diverse_100/capture/20261007` in the Hugging Face dataset
  `UW-SyFI/servingstudio-workload`, pinned at `18506fd7afbaf925da2d31821287d079fffba5a6` (already
  uploaded; `manifest.json` and `routes.u16` match the local capture's sha256).
- **`profiling/profile.db`** gains 2,574 B200 rows, all measured by this branch. They were merged row-wise by
  semantic key onto `main`'s DB:

  | Kind | Rows |
  |---|---:|
  | `fa4_varlen_attention` | 1,136 |
  | `elementwise` | 378 |
  | `single_gemm` | 340 |
  | `nvfp4_fused_moe` | 292 |
  | `kv_cache_append` | 280 |
  | `all_reduce_residual_rms_norm` | 80 |
  | `gemm_fp32_output` | 68 |

  It also gains 16 profiling-run records.
  - **Five rows of `main` change:** `single_gemm` / `torch_linear_vllm`, n 4096, k 2048, bf16, m 1/2/4/8/16
    (job 6408, profiler c47ac0b1).
    - Their old values came from the upstream 3f667d7 profiler image. Re-measured on the serving fork's image,
      the values in µs (old → new) are: 8.456 → 6.234, 6.805 → 7.368, 7.108 → 7.339, 6.663 → 7.346 and
      5.007 → 6.030.
    - GLM-5.3-Flash reads these rows too (see README).
  - No row is removed. The file grows from 69 MB to 74 MB.

## Contribution licensing

- [ ] I have read the project's CLA.
- [ ] I have the right to submit this contribution.
- [ ] I have disclosed any third-party code or licensing restrictions.
- [ ] I understand that CLA acceptance must be recorded before this
      contribution can be merged.

Third-party material included or adapted here: none.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
