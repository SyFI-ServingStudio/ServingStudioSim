# feat(glm53-flash): GLM-5.3-Flash NVFP4 + DFlash2 on B200, vLLM layout and KDA-TP fused design

## Purpose

Model GLM-5.3-Flash NVFP4 with the DFlash2 draft (7 draft tokens) on one
8x B200 node, attention DP8 and MoE EP8, in two forms, and publish both as
public presets:

- **`glm53_flash_vllm_nvfp4_dp_attn_ep_moe_dflash2`**: vLLM's deployment, on
  the kernels the vLLM fork calls (GateLinear router gate, Triton chunk KDA
  and causal-conv prefill, CuTe RecoverSSM, FlashInfer TRT-LLM routed MoE,
  vLLM CUDA top-k, TileLang mHC, eager-torch DSA glue). The fork's source line
  is cited beside each backend.
- **`glm53_flash_nvfp4_dp_attn_kda_tp_fused_ep_moe_dflash2`**: our design. KDA
  is tensor-parallel over `kda_tp_gpus` (1/2/4/8), the routed MoE is fused
  behind a triggered fixed-pattern MNNVL expert exchange, and every mHC
  boundary takes the faster measured kernel. Every mechanism is fixed; only
  `kda_tp_gpus`, `ep_size`, `max_model_len`, routing, capture sizes and draft
  tokens are parameters.

Under them:

- **L1.** 18 new kinds (KDA RecoverSSM verify / conv-verify / commit, DSA
  pooled indexer and page-table kinds, Hadamard FP8 quant, fused mHC sum
  boundary, sigmoid-gated RMSNorm, quantized EP all-gather, and others) and
  new backends or arguments on 16 existing kinds. Each has a Python runner, a Rust cache and
  tests. [`README.md`](README.md#kernels) lists them.
- **L2–L4.** The fixed-pattern expert exchange op, the DFlash2 checkpoint
  worklet, KDA per-role backends, the DSA `fork_glue_backends` field, two new
  arch files and the Flash DP / PP archs' ladder options.
- **Pricing fixes that reach other models.** A one-token append after a cached
  prefix runs the decode kernels (8a9f0bb5); a short eager DSA step prices a
  causal fill instead of the decode logits and top-k (131e1572).
- **Data.** A measured shared-expert exposure table
  (`model/glm53_flash_nvfp4_b200_ep8_moe_shared_exposure.json`), the DFlash2
  draft config, catalog entries, and the four public preset files.

[`README.md`](README.md) has the modeling notes, the evidence and the open
items.

## Test Plan

On the branch head, with the committed `profiling/profile.db`:

```bash
just test-cpu
just test-presets
just test-gpu        # one B200, through Slurm
```

Also run:

- **The public presets against the DFlash2 ladder** (the earlier study that
  stacked the design's mechanisms on the vLLM layout): every member of both
  sim presets at c1–c256 on the ladder's k7 workload, 3600 s, `--no-gpu`.
- **Rung 9 of the ladder replayed** on this branch and DB at c1–c32.
- **timing-predict on every public member** with `main`'s and this branch's
  profile.db row values, for the changes that reach other models.
- **The trimmed DB**: the committed DB gives the same simulation output as the
  full local DB on all 35 ladder runs.

Not run: alignment (no pack exists for these deployments, see below),
`just test-bench`.

## Test Result

- `just test-cpu`: 4 failed, 5599 passed, 36 skipped; Rust lib 1832 passed.
  The 4 failures are the baseline `test_l1_single_gemm::test_energy_*` NVML
  energy tests, which failed before this branch's changes too.
- `just test-presets`: 9 passed, 1 skipped.
- `just test-gpu` (one B200): 14 passed, 3 skipped (an sm_90a-only GEMM, the
  measure smoke test finding no idle GPU, and the opt-in bench tier). No golden
  warns. No golden was re-recorded.
- **Public presets against the DFlash2 ladder** (total tok/s; design column at
  its best `kda_tp_gpus` b):

  | c | ladder rung 0 | vLLM tag | ladder rung 9 | rung 9 replayed | design tag |
  |---|---|---|---|---|---|
  | 1 | 273.565 | 267.494 (−2.22%) | 417.934 | 406.306 | 405.684 b8 (−2.93% vs rung 9; −0.15% vs replay) |
  | 8 | 2161.074 | 2113.867 (−2.18%) | 3286.449 | 3192.762 | 3186.073 b8 (−3.05%; −0.21%) |
  | 16 | 4022.400 | 3942.215 (−1.99%) | 5971.058 | 5852.926 | 5868.301 b8 (−1.72%; +0.26%) |
  | 32 | 7028.777 | 6983.901 (−0.64%) | 10712.170 | 10640.849 | 10685.993 b8 (−0.24%; +0.42%) |
  | 64 | 12462.851 | 12278.718 (−1.48%) | 19503 | – | 19302.938 b4 (−1.03%) |
  | 128 | 20410.571 | 20258.740 (−0.74%) | 33866 | – | 33625.640 b2 (−0.71%) |
  | 256 | 28405.987 | 28272.891 (−0.47%) | 50778.049 | – | 50575.035 b1 (−0.40%) |

  On the ladder's own DB snapshot the code reproduces rung 0 (c32 differs by
  −0.0004%) and lane N exactly. The difference above is the DB: the ladder priced TileLang mHC on
  rows from 2026-09 that read 1–3 µs per boundary too fast; with the
  re-measured rows, rung 9 itself drops by 0.7–2.9% (the "replayed" column),
  and the design tag is within −0.21% to +0.42% of it. [`README.md`](README.md#simulation-against-the-dflash2-ladder)
  breaks the gap down.
- **Alignment.** No alignment Phase 1 result exists for either deployment.
  There is no pack for them under `presets/alignment/`
  (`glm53_flash_fp8_b200_tp4_ep4` is FP8 TP4/EP4 without speculation;
  `glm53_nvfp4_b200_dflash2` is GLM-5.3, not Flash). The only captures of the
  real deployment are three nsys captures of vLLM serving GLM-5.3-Flash NVFP4
  + DFlash2 at DP4/EP4 (c64, c128 x2), taken for a decode-path audit, not a
  diverse_100 replay and not run through Check 1/2/3. Alignment Phase 1 is
  missing and must be added before this merges under
  `dev-finalize-model-addition`; alignment Phase 2 is not run.

### Changes that reach other models

- **Code**: 8a9f0bb5 (one-token append runs the decode kernels; `append1`
  timing-predict case −6.0% to −32.8% on GLM-5.2 and GLM-5.3 presets, decode
  cases identical) and 131e1572 (short eager DSA step; −0.11% to −0.24% on the
  Flash DP/PP `shortmix` case). Numbers per preset in [`README.md`](README.md#code-from-flash-clean).
- **profile.db row values** (2,704 rows; timing-predict with only those rows
  at `main`'s values versus this branch's, median over members):

  | preset | members | decode cases (min / max) | prefill 2048 | mixed | moved by |
  |---|---:|---|---|---|---|
  | GLM-5.3-Flash FP8 `glm53_flash_vllm_fp8_kda_dsa_moe` | 30 | +0.51% / +3.38% | −0.18% | −0.06% | mhc +3.38% |
  | GLM-5.3-Flash FP8 `glm53_flash_vllm_fp8_dp_attn_ep_moe` | 12 | +0.17% / +1.88% | −0.10% | −0.02% | mhc +1.78%, smla −0.28%, bmm +0.19% |
  | GLM-5.3-Flash FP8 `glm53_flash_vllm_fp8_pp_kda_dsa_moe` | 12 | +0.11% / +2.00% | −0.11% | −0.02% | mhc +1.72%, smla −0.19%, bmm +0.26% |
  | GLM-5.3-Flash NVFP4 `glm53_flash_vllm_nvfp4_kda_dsa_moe` | 24 | +0.64% / +4.22% | −0.25% | −0.08% | mhc +4.22% |
  | GLM-5.3-Flash NVFP4 `glm53_flash_vllm_nvfp4_dp_attn_ep_moe` | 12 | −1.55% / +2.63% | −0.42% | −0.04% | mhc +2.25%, moe −2.20%, smla −0.36%, bmm +0.22% |
  | GLM-5.3-Flash NVFP4 `glm53_flash_vllm_nvfp4_pp_kda_dsa_moe` | 12 | +0.15% / +2.44% | −0.16% | −0.03% | mhc +2.32%, smla −0.27%, bmm +0.29% |

  `mhc` = TileLang mHC re-measure, `smla` = sparse MLA −1 padding re-measure,
  `moe` = NVFP4 routed MoE re-measure, `bmm` = token-major q_absorb; "moved
  by" is the largest per-case median with only that kind changed. Every other
  public preset (GLM-5.2, GLM-5.3 NVFP4 DFlash2, DeepSeek, Inkling, Llama,
  MiMo, Qwen) is unchanged on every case.

## Dependencies

- **Token corpus**: the `balanced_c32` capture of the Hugging Face dataset
  `UW-SyFI/servingstudio-workload`, pinned at `c3f5ecaa` (already used by the
  GLM-5.3-Flash presets on `main`). Nothing new is uploaded.
- **`profiling/profile.db`** (its own commit) is `main`'s DB plus 9,023 B200
  rows the public presets read and `main` lacks, found by the presets'
  missing-row export until nothing is missing. The file is 81.6 MB (`main`:
  78.4 MB). 2,704 rows of `main` take new values:

  | Kind / backend | Rows | Why |
  |---|---:|---|
  | `dsa_sparse_mla_attention` flashinfer_trtllm_fp8 | 2,251 | re-measured after the page table moved to vLLM's −1 padding |
  | `nvfp4_fused_moe` flashinfer_trtllm_routed_sm100 | 248 | re-measured with seeded realistic inputs and autotune off |
  | `batched_gemm` torch_mla_q_absorb_no_rope | 68 | the fork's token-major output |
  | `mhc_fused_post_pre_rms_norm` vllm_tilelang | 69 | September rows read 1–3 µs fast; median of three passes |
  | `mhc_pre_rms_norm` vllm_tilelang | 68 | same |

  Largest additions: `dsa_persistent_topk_decode` 1,584, `single_gemm`
  1,292, `flashinfer_attn_rect` 1,152, `nvfp4_fused_moe` 1,128,
  `kda_chunk_prefill` 1,024, and every row of the new kinds. No row of `main`
  is removed.

## Contribution licensing

- [ ] I have read the project's CLA.
- [ ] I have the right to submit this contribution.
- [ ] I have disclosed any third-party code or licensing restrictions.
- [ ] I understand that CLA acceptance must be recorded before this
      contribution can be merged.

Third-party material included or adapted here: none.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
