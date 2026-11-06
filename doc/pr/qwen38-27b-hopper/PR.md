# feat(qwen38): Qwen3.8-27B-NVFP4 on Hopper (H100 / H200, SM90), with kernel alignment

## Purpose

Model `nvidia/Qwen3.8-27B-NVFP4` as the vLLM fork serves it on one H100 or one H200. The B200 model
(`doc/pr/qwen38-27b/`) does not transfer: on H200, 4,016 of its 4,098 kernel specs had no rows, because Hopper
has no FP4 tensor cores and vLLM runs different kernels there. Captures on both GPUs (2026-11-03) showed which:

- **`qwen35_vllm_nvfp4_gdn_dense` picks a kernel path from the GPU** (`KernelPath::Sm90` for H100, H200, H800,
  H20, GH200; `Sm100` keeps the B200 graph unchanged):
  - NVFP4 linears (MLP, lm_head) run as Marlin W4A16 on the BF16 activation, so the MLP's two NVFP4 quants and
    the lm_head quant drop out (`vllm_nvfp4_swiglu_mlp_local` gains `weight_only`);
  - FP8 projections run on Cutlass SM90 (`cutlass_3x_gemm_sm90_fp8`);
  - attention is vLLM's FLASH_ATTN backend (FlashAttention 3) over the FP8 cache.
- **Three profiler runners**, each checked against an fp32 reference: `single_gemm` / `vllm_marlin_nvfp4`,
  `single_gemm` / `vllm_cutlass_fp8_tensor`, and `paged_attn_batch_{decode,prefill}` / `vllm_fa3`.
- **FA3 attention grids** add context knots 262,144 / 524,288 / 1,048,576 and place wave edges with Hopper's 132
  SMs. The B200 grids are unchanged.
- **`gdn_prefill_post_conv`** masks grid cells whose int32 offsets overflow (`num_tokens * stride > 2^31`): vLLM's
  kernel faults there with an illegal address.
- **Two alignment packs**, `qwen38_27b_nvfp4_{h100,h200}_tp1`: B200 cases 03, 04, 05 and 11 at chunk 2048. Each
  has SM90 label rules (50 rules over 38 operations; every forward kernel maps) and a declared
  `gpu_time_multiplier`, H100 1.01958 and H200 1.02157, from the pooled kernel alignments.
- **`profile.db`** gains 4,320 H100 and 4,238 H200 rows.

[`README.md`](README.md) has the modeling notes, the evidence and the known gaps.
[`alignment_matrix.md`](alignment_matrix.md) is the generated campaign matrix.

## Test Plan

At the branch head (`b7337f7b`), after merging `origin/main` (`a8501be5`):

```bash
just test-cpu      # pytest tier and Rust --lib; see Test Result for the Rust invocation
just test-presets
```

Also run:

- **Campaign, both packs, every case**, through `alignment-campaign` from timing-predict to `compare`, at the
  declared multipliers and at 1.0. The captures (nsys pass and workload pass per case) are from 2026-11-03.
- **`alignment label check`** on every labeled inventory.
- **B200 regression:** a Qwen3.8-27B-NVFP4 1×B200 simulation (956 requests, trace-timed, multiplier 1.0483) at
  the branch head against the same run at the branch's base `5c7cd27b`.

Not run:

- **`just test-gpu`.** The development host has no GPU; the Hopper rows were measured on the capture hosts.
- `test-bench` and `test-agent` (opt-in).
- No `diverse_100` capture on Hopper, and no case at chunk 4096 or 8192.

## Test Result

- **`test-cpu`, pytest tier:** 5,148 passed, 18 skipped.
- **`test-cpu`, Rust `--lib`:** 1,582 passed, 0 failed, 8 ignored (re-run after the rustfmt commit). On this host the test binary needs
  `LD_PRELOAD=/lib/x86_64-linux-gnu/libgcc_s.so.1`, because the venv's conda `libgcc_s` is older than the
  toolchain's. That is a host issue, not a code one.
- **`test-presets`:** 7 passed, 2 failed. The failing members are only GLM-5.2, GLM-5.2-FP8 and GLM-5.3-Flash DSA
  members (`dsa_sparse_mla_attention`, `dsa_mqa_logits_prefill`). They fail the same way with `origin/main`'s
  exact `profile.db`, so the gap predates this branch (it follows main's sparse-MLA cache re-key, #109). Every
  Qwen3.8 member is measured.
- **B200 regression:** all 956 requests are identical to the base run in every column (TTFT, TPOT, finish time,
  cache hits).
- **Labels:** every forward kernel of all 8 captures maps; `alignment label check` reports no findings.
- **Check 1** (kernel alignment, signed error of the simulated critical path, from each case's
  `alignment_iteration_report.json`):

  | GPU | Case | Iterations | Decode median | Mixed median | All median | All total | Coverage | Recommended multiplier |
  |---|---|---:|---:|---:|---:|---:|---:|---:|
  | H100 | 03_balanced_anchor_c32 | 185 | -2.57% | -1.03% | -2.41% | -1.42% | 99.74% | 1.0187 |
  | H100 | 04_prefill_spectrum_c32 | 74 | – | -1.03% | -1.03% | -0.97% | 99.87% | 1.0168 |
  | H100 | 05_decode_spectrum_c64 | 500 | -3.65% | -2.46% | -3.62% | -2.73% | 99.17% | 1.0334 |
  | H100 | 11_long_prefill_ladder_c1 | 380 | +1.73% | -0.12% | +1.61% | -0.30% | 99.85% | 1.0154 |
  | H200 | 03_balanced_anchor_c32 | 207 | -2.05% | -1.62% | -2.01% | -2.07% | 99.74% | 1.0199 |
  | H200 | 04_prefill_spectrum_c32 | 77 | – | -1.46% | -1.46% | -1.56% | 99.88% | 1.0148 |
  | H200 | 05_decode_spectrum_c64 | 521 | -3.76% | -2.94% | -3.75% | -2.87% | 99.13% | 1.0339 |
  | H200 | 11_long_prefill_ladder_c1 | 380 | +2.61% | -0.85% | +2.37% | -1.45% | 99.86% | 1.0203 |

- **Campaign** at the declared multipliers. Both tables are the output of `alignment-campaign compare --markdown`,
  copied from [`alignment_matrix.md`](alignment_matrix.md):

  H100 (`gpu_time_multiplier` 1.01958):

  | Case | kernel_abs_pct | kernel_signed_pct | map_coverage_pct | e2e_mean_pct | output_tps_pct | server_tpot_mean_pct | server_ttft_mean_pct |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | tp1/03_balanced_anchor_c32 | +1.59% | -1.42% | 99.74% | -1.16% | +1.34% | -0.18% | -0.74% |
  | tp1/04_prefill_spectrum_c32 | +1.46% | -0.97% | 99.87% | +0.76% | -0.68% | +3.24% | -3.55% |
  | tp1/05_decode_spectrum_c64 | +3.03% | -2.73% | 99.17% | -2.69% | +2.24% | -2.92% | +0.10% |
  | tp1/11_long_prefill_ladder_c1 | +0.82% | -0.30% | 99.85% | +0.43% | -0.43% | +0.97% | +0.58% |

  H200 (`gpu_time_multiplier` 1.02157):

  | Case | kernel_abs_pct | kernel_signed_pct | map_coverage_pct | e2e_mean_pct | output_tps_pct | server_tpot_mean_pct | server_ttft_mean_pct |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | tp1/03_balanced_anchor_c32 | +2.27% | -2.07% | 99.74% | -1.81% | +1.77% | -0.97% | -0.61% |
  | tp1/04_prefill_spectrum_c32 | +1.89% | -1.56% | 99.88% | -0.74% | +0.79% | +1.27% | -3.23% |
  | tp1/05_decode_spectrum_c64 | +3.41% | -2.87% | 99.13% | -2.62% | +1.52% | -3.52% | -4.28% |
  | tp1/11_long_prefill_ladder_c1 | +1.98% | -1.45% | 99.86% | +0.06% | -0.06% | +4.74% | +0.07% |

- **Before and after the multiplier** (the packs declared none before, i.e. 1.0): E2E moves from -1.2..-4.7% to
  -2.7..+0.8%, and TTFT from -1.4..-6.3% to -4.3..+0.6%. The per-case table is in the README's Simulation section.
  Kernel columns do not depend on the multiplier and are unchanged.
- **Out of tolerance:** only case 04's scheduler-structure counts, on both GPUs: iterations -1.75% (H100) / -1.47%
  (H200), with decode batch, prefill tokens and scheduled KV by the same amount, against 1%. B200 case 04 has the
  same signature (-2.04%), attributed to closed-loop requests re-entering a few steps late. E2E there is within
  ±0.8%.
- **Known gap:** FA3 decode attention reads -22% (H100) / -25% (H200) in c64 decode-only steps, within 1.4% at c1.
  The likely cause is the decode cache key inherited from trtllm-gen; the README has the analysis and the
  follow-up.

## Dependencies

- **vLLM fork** `e1423a4d` (the captures; no fork change in this PR).
- **`profiling/profile.db`** gains 8,558 Hopper rows, all measured by this branch's runners, merged row-wise by
  semantic key onto `origin/main`'s DB (`kernel-profile merge-db`): 8,567 inserted (8,558 kernel rows and 9
  `_profile_run` records), 0 conflicts. No row of `main` is removed or changed. The file is 75 MB.

  | Kind | H100 80GB HBM3 | H200 |
  |---|---:|---:|
  | `paged_attn_batch_prefill` | 1,534 | 1,534 |
  | `paged_attn_batch_decode` | 861 | 861 |
  | `elementwise` | 478 | 411 |
  | `single_gemm` | 476 | 476 |
  | `multi_add_rms_norm` | 340 | 340 |
  | `gdn_causal_conv_prefill` | 144 | 144 |
  | `static_fp8_quant` | 136 | 136 |
  | `kv_cache_append` | 70 | 70 |
  | `qk_rms_norm_rope_gate` | 68 | 68 |
  | `silu_and_mul` | 68 | 68 |
  | `gdn_chunk_delta_rule` | 42 | 42 |
  | `indexed_row_copy` | 42 | 42 |
  | `gdn_prefill_post_conv` | 19 | 19 |
  | `gdn_recurrent_decode` | 18 | 18 |
  | `gdn_gated_rms_norm` | 15 | 0 |
  | `gdn_causal_conv_decode` | 9 | 9 |

  H200 has fewer `elementwise` and `gdn_gated_rms_norm` rows because existing H200 rows (from the Qwen3.6 graph)
  already cover those shapes.

## Contribution licensing

- [ ] I have read the project's CLA.
- [ ] I have the right to submit this contribution.
- [ ] I have disclosed any third-party code or licensing restrictions.
- [ ] I understand that CLA acceptance must be recorded before this
      contribution can be merged.

Third-party material included or adapted here (project, URL, version, license — or "none"): none. The runners call
the vLLM fork's installed ops (`apply_fp4_marlin_linear`, `cutlass_scaled_mm`, `flash_attn_varlen_func`); no code is
copied.

---
<details>
<summary>Checklist</summary>

- [x] **Doc-backed — name the file.** `simulator/src/timing/README.md` (`infeasible_mask()`: physically
      unreachable cells are dropped rather than given a fabricated value),
      `launcher/alignment_campaign/README.md` (pack, `label_rules/`, declared `gpu_time_multiplier`),
      `alignment/labeling/README.md` (order-independent rule files). `doc/architecture_compatibility.md` is
      updated for the SM90 path.
- [x] **Formatting radius matches the change.** `git status --short` shows only intended files. Rust was
      formatted per file.
- [x] **`profiling/profile.db`:** merged row-wise with `kernel-profile merge-db` onto `origin/main`'s DB.
      8,558 rows over 16 tables (above), measured on H100 80GB HBM3 (ptc) and H200 (coriander). The 1,733 B200
      rows that main re-profiled keep main's values.
- [x] **Unmeasured shapes still fail.** A Hopper SKU missing from `KernelPath::of` falls to SM100, whose rows
      it does not have, so the build reports missing rows. The GDN post-conv mask marks cells the kernel cannot
      run (int32 overflow) infeasible, per the timing README, and borrows no neighbour's timing for them.
- [x] **Goldens.** No golden moves: B200 Qwen3.8 is bit-identical (956 / 956 requests) and there are no Hopper
      Qwen3.8 goldens. `test-gpu` was not run.

</details>

🤖 Generated with [Claude Code](https://claude.com/claude-code)
