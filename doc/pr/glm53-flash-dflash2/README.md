# GLM-5.3-Flash NVFP4 + DFlash2 evidence (B200, DP8 attention / EP8 MoE)

Review evidence for the two DFlash2 tags on GLM-5.3-Flash NVFP4 and the kernels,
worklets and archs under them. There is no alignment Phase 1 result for these
deployments yet (see [Alignment](#alignment)). This is not an accepted baseline,
and no golden is recorded.

## Deployment modeled

One 8x B200 node serving `nvidia/GLM-5.3-Flash-NVFP4` with the DFlash2 draft
(`incoai/GLM-5.3-Flash-DFlash2`, `model/config/glm53_flash_dflash2.json`) at
7 draft tokens. Attention is data-parallel over the 8 GPUs, the routed experts
are expert-parallel (EP8), and the worker is the speculative worker.

| Tag | What it is | Public presets |
|---|---|---|
| `glm53_flash_vllm_nvfp4_dp_attn_ep_moe_dflash2` | vLLM's layout: every kernel the vLLM fork calls | `presets/public/GLM-5.3-Flash-NVFP4/glm53_flash_vllm_nvfp4_dp_attn_ep_moe_dflash2.yaml`, `presets/public_sim/.../*_speculative.yaml` |
| `glm53_flash_nvfp4_dp_attn_kda_tp_fused_ep_moe_dflash2` | Our design: KDA tensor-parallel over `kda_tp_gpus` (1/2/4/8), fused routed MoE behind a fixed-pattern expert exchange, every mechanism fixed | `presets/public/GLM-5.3-Flash-NVFP4/glm53_flash_nvfp4_dp_attn_kda_tp_fused_ep_moe_dflash2.yaml`, `presets/public_sim/.../*_speculative.yaml` |

Only `kda_tp_gpus`, `ep_size`, `max_model_len`, routing, capture sizes and
draft tokens are parameters. Routing reads the pinned `balanced_c32` token
corpus of the FP8 capture (the NVFP4 export keeps the router in BF16).

## How each model feature is modeled

- **KDA (linear attention).**
  - Per-role backends: `bf16_gemm_backends` covers `f_b` / `g_b`; `in_out_proj_backends` covers in_proj / o_proj; `conv_verify_backends` / `commit_backends` cover CuTe RecoverSSM. The FP8 Flash arch sets the new fields to its old constants.
  - Speculative verify per position with per-position state pages; RecoverSSM verify / conv-verify / commit are measured kinds.
  - Output norm on `sigmoid_gated_rms_norm`.
- **DSA (sparse MLA).**
  - Indexer: Hadamard FP8 quant (with and without head-weight scaling), pooled index-cache update, pooled top-k and logits page tables.
  - The fork's eager glue launch list is priced through `fork_glue_backends` (empty means `elementwise_backends`, so other archs are unchanged).
  - A one-token append after a cached prefix runs the decode kernels; an eager mixed step whose longest request is at most `index_topk` bills the short causal fill instead of the decode MQA logits and top-k.
- **MoE.** The vLLM tag uses FlashInfer TRT-LLM routed NVFP4 MoE with a BF16 GateLinear router gate. The design tag uses the fused routed MoE behind the triggered fixed-pattern MNNVL exchange (`op/comm/fixed_exchange.rs`). Shared-expert exposure comes from a shipped measurement table (`model/glm53_flash_nvfp4_b200_ep8_moe_shared_exposure.json`).
- **mHC.**
  - The vLLM tag calls TileLang at every boundary (fork `glm5next/common/model.py:405-408`).
  - The design tag lists `vllm_tilelang(_post_addend)` and `cuda_one_launch` at every boundary. The faster measured row at the boundary's rows per rank wins; there is no saving constant. The after-MoE boundary is one kernel either way, with the shared + routed add inside it.
- **DFlash2 draft.** Draft layers, selector, context K/V (all draft layers' K/V in one launch), draft query K/V append, and rect attention priced at the draft call's batch shape.

## Kernels

New L1 kinds, each with a Python runner, a Rust cache and tests:
`batched_gemm_static_fp8_quant`, `dsa_fp32_split_state_update`,
`dsa_pooled_index_cache_update`, `dsa_pooled_logits_page_table`,
`dsa_pooled_topk_page_table`, `fp32_row_split_layer_norm`,
`grouped_conv_residual_rms_norm`, `hadamard_fp8_quant`,
`hadamard_fp8_quant_weight_scale`, `kda_recoverssm_commit`,
`kda_recoverssm_conv_verify`, `kda_recoverssm_verify`,
`layered_kv_norm_rope_cache_write`, `mhc_fused_sum_post_pre_rms_norm`,
`moe_ep_quantize_and_all_gather`, `qk_norm_rope_fp8_kv_write`,
`row_all_gather`, `sigmoid_gated_rms_norm`.

Existing kinds with new backends or arguments: `all_reduce_fusion`,
`batched_gemm`, `dsa_paged_mqa_logits_decode`, `dsa_persistent_topk_decode`,
`flashinfer_attn_rect`, `gemm_fp32_output`, `kda_recurrent_decode`,
`logits_topk`, `mhc_fused_post_pre_rms_norm`, `moe_alltoall`,
`moe_alltoall_prepare`, `moe_ep_quantized_all_gather`,
`moe_ep_reduce_scatter`, `nvfp4_fused_moe`, `single_gemm`,
`static_fp8_quant`.

### mHC re-measure

The TileLang mHC rows on B200 at hidden 4096 were measured on 2026-09-24/25,
while the `cuda_one_launch` rows were measured in November. Re-measured in one
job (three passes per spec, median row):

| Kind / backend | Old rows | Change at ≤256 rows |
|---|---|---|
| `mhc_fused_post_pre_rms_norm` vllm_tilelang | 2026-09-25 | median +1.40 µs, up to +3.22 µs (1 row 8.06 → 9.60 µs; 8 rows 9.22 → 12.44 µs) |
| `mhc_pre_rms_norm` vllm_tilelang | 2026-09-24 | +1.10 to +1.68 µs |
| `mhc_fused_post_pre_rms_norm` cuda_one_launch | 2026-11-12 | within 0.02 µs |
| `mhc_fused_sum_post_pre_rms_norm` (all three backends) | 2026-11-12 / 14 | within 0.3 µs |

The three passes agree to a median spread of 0.1%. The November rows
reproduce, so only the September rows are replaced.

## Simulation against the DFlash2 ladder

The ladder is the earlier study that stacked the design's mechanisms one at a
time on the vLLM layout (rung 0) up to the design (rung 9). Setup: the public
preset members as shipped, on the committed `profiling/profile.db`; 3600 s,
`--no-gpu`, the ladder's k7 workload (`balanced_c32` routing). Values are total
tok/s; the design column shows the best `kda_tp_gpus`. "Rung 9 replayed" is the
ladder's rung-9 run_config (with its mHC saving constant), translated to this
branch and run on the same DB.

| c | ladder rung 0 | vLLM tag | ladder rung 9 | rung 9 replayed | design tag |
|---|---|---|---|---|---|
| 1 | 273.565 | 267.494 (−2.22%) | 417.934 | 406.306 | 405.684 b8 (−2.93% vs rung 9; −0.15% vs replay) |
| 8 | 2161.074 | 2113.867 (−2.18%) | 3286.449 | 3192.762 | 3186.073 b8 (−3.05%; −0.21%) |
| 16 | 4022.400 | 3942.215 (−1.99%) | 5971.058 | 5852.926 | 5868.301 b8 (−1.72%; +0.26%) |
| 32 | 7028.777 | 6983.901 (−0.64%) | 10712.170 | 10640.849 | 10685.993 b8 (−0.24%; +0.42%) |
| 64 | 12462.851 | 12278.718 (−1.48%) | 19503 | – | 19302.938 b4 (−1.03%) |
| 128 | 20410.571 | 20258.740 (−0.74%) | 33866 | – | 33625.640 b2 (−0.71%) |
| 256 | 28405.987 | 28272.891 (−0.47%) | 50778.049 | – | 50575.035 b1 (−0.40%) |

How the gaps break down:

1. **On the ladder's own DB snapshot the code reproduces the ladder.** The
   vLLM tag equals rung 0 (c32 −0.0004%: the fork calls Triton KDA prefill,
   `glm5next/nvidia/kda.py:23-24,456,539`, while the ladder used FlashInfer
   CuTe / Dao conv). The design tag equals lane N (rung 9 without the saving
   constant, with the after-MoE add in the TileLang post kernel) before the
   mHC best-of.
2. **The mHC re-measure is most of the gap.** The ladder priced TileLang mHC
   on the September rows. On the DB before the re-measure the vLLM tag was
   within −0.39% to +0.01% of rung 0; the re-measure alone moves it to the
   column above, since the fork has only TileLang at every boundary. Replaying
   rung 9 on the re-measured rows costs rung 9 itself 2.8% at c1, 2.9% at c8,
   2.0% at c16 and 0.7% at c32.
3. **On the same rows the design tag is −0.21% to +0.42% from rung 9 at
   c1–c32.** Rung 9 subtracts an extrapolated saving (1.9 → 0.5 µs per
   boundary, per load point) from TileLang rows and keeps the after-MoE add
   separate; the design tag prices each boundary at the faster measured kernel.
   Below about 64 rows per rank `cuda_one_launch` is 1–2 µs faster than
   TileLang; above about 128 rows TileLang is faster.
4. **Remaining DB differences against the ladder's snapshot.** 204
   `single_gemm` torch_linear_vllm rows from the same profiling run hold
   different values in the two DBs (snapshot median −1.8% against this DB,
   range −10.4% to +42.6%); they moved the vLLM tag by −0.37% at c16 and the
   design tag by up to +0.5%. 50 `sigmoid_gated_rms_norm` rows move it by
   −0.02% to +0.01%. Setting these rows to the snapshot's values reproduces the
   snapshot's numbers exactly.

The merge of `flash-clean`'s two pricing fixes (one-token append, short eager
DSA step) does not change these numbers: the trace has no cached prefix, and
the causal-fill branch is never taken (a stand-in row at 0.001 ms and at 1.0 ms
gives identical output).

## Changes that reach other models

### Code (from `flash-clean`)

Measured with timing-predict on the public members before and after each change:

- **8a9f0bb5**: a one-token append after a cached prefix now runs the decode
  kernels. `append1` case, median over members: GLM-5.2 FP8 −29.7%, GLM-5.2
  BF16 −19.5%, GLM-5.2 NVFP4 TP −14.7% / DP-attn −9.3% / PP −6.0%,
  GLM-5.3-Flash PP FP8 −25.0% / NVFP4 −32.8%, GLM-5.3 NVFP4 DFlash2 −6.5%.
  Decode-only cases are identical. Simulated tok/s on prefix workloads:
  GLM-5.3 DFlash2 −0.64%, GLM-5.2 NVFP4 TP4 +0.36%, Flash TP4/EP +0.13%; the
  others about 0. GLM-5.2 NVFP4 MTP is unchanged: its fork requires uniform
  decode without the FP4 indexer cache, so it keeps the append a prefill.
- **131e1572**: in an eager mixed step whose longest request is at most
  `index_topk`, the decode MQA logits, top-k and post-top-k glue bill zero and
  a 4-launch causal fill is priced instead; full-graph replays keep the long
  path. `shortmix` case: −0.11% to −0.24% on the Flash DP/PP arches.
- The KDA per-role backends, the DSA `fork_glue_backends` field and the
  `committed_rows` field on speculative decode inputs leave every other arch's
  output unchanged.

### profile.db row values

2,704 rows of `main`'s DB take new values (see PR.md, Dependencies). Effect on
every public member, timing-predict at the case set under Reproduce, with only
these rows at `main`'s values versus this branch's (both runs on the same code;
the medians are over members). "Moved by" is the largest per-case median when
only that kind's rows change: `mhc` = TileLang mHC re-measure, `smla` = sparse
MLA −1 padding re-measure, `moe` = NVFP4 routed MoE re-measure, `bmm` =
token-major q_absorb.

| preset | members | decode cases (min / max) | prefill 2048 | mixed | moved by |
|---|---:|---|---|---|---|
| GLM-5.3-Flash FP8 `glm53_flash_vllm_fp8_kda_dsa_moe` | 30 | +0.51% / +3.38% | −0.18% | −0.06% | mhc +3.38% |
| GLM-5.3-Flash FP8 `glm53_flash_vllm_fp8_dp_attn_ep_moe` | 12 | +0.17% / +1.88% | −0.10% | −0.02% | mhc +1.78%, smla −0.28%, bmm +0.19% |
| GLM-5.3-Flash FP8 `glm53_flash_vllm_fp8_pp_kda_dsa_moe` | 12 | +0.11% / +2.00% | −0.11% | −0.02% | mhc +1.72%, smla −0.19%, bmm +0.26% |
| GLM-5.3-Flash NVFP4 `glm53_flash_vllm_nvfp4_kda_dsa_moe` | 24 | +0.64% / +4.22% | −0.25% | −0.08% | mhc +4.22% |
| GLM-5.3-Flash NVFP4 `glm53_flash_vllm_nvfp4_dp_attn_ep_moe` | 12 | −1.55% / +2.63% | −0.42% | −0.04% | mhc +2.25%, moe −2.20%, smla −0.36%, bmm +0.22% |
| GLM-5.3-Flash NVFP4 `glm53_flash_vllm_nvfp4_pp_kda_dsa_moe` | 12 | +0.15% / +2.44% | −0.16% | −0.03% | mhc +2.32%, smla −0.27%, bmm +0.29% |
| New: `glm53_flash_vllm_nvfp4_dp_attn_ep_moe_dflash2` | 1 | −0.14% / +2.48% | −0.17% | +0.06% | mhc +2.29%, smla −0.36%, bmm +0.16% |
| New: `glm53_flash_nvfp4_dp_attn_kda_tp_fused_ep_moe_dflash2` | 4 | −0.55% / +1.11% | −0.11% | +0.03% | mhc +1.06%, smla −0.55% |

Every other public preset is unchanged (0.00% on every case): GLM-5.2 (BF16,
FP8, NVFP4: TP, DP-attention, PP, speculative, SGLang), GLM-5.3 NVFP4
DFlash2, DeepSeek-V4 / V4.1, Inkling, Llama-3.1, MiMo-V2.6, Qwen3 / Qwen3.6 /
Qwen3.8. The three FFN-only Qwen3 presets do not take this case set; they read
none of the replaced kinds. The TP-attention Flash arches do not read the
sparse MLA, q_absorb or routed MoE rows that changed (other head counts and
expert shapes).

The decode increase comes from the TileLang mHC rows (1–3 µs per boundary at
small batches). The NVFP4 DP-attention preset also gets faster at larger
decode batches from the routed MoE re-measure (d128x2k down to −3.74%).

## Modeling choices

- `attn_gpu_memory_gb: 110.12` in both sim presets is 149.84 − 39.72 GB.
  149.84 GB is vLLM's measured device budget at `gpu_memory_utilization` 0.85
  (a TP4 FP8 run: 81.00 GB weights + 68.84 GB KV); 39.72 GB is the NVFP4
  DP8/EP8 per-GPU weight footprint, computed, not measured.
- Shared-expert exposure is a measured table, not a constant: column
  `production` for the fork's MoE block, `fused_cap64_trig` for the design.
- No calibration multiplier: `gpu_time_multiplier` is 1.0 in both sim presets.

## Known gaps

- **No alignment Phase 1** for either deployment (below).
- **model.work location maps** for the two tags are not included; they need
  analyzer changes.
- **HiSparse load points** are not modeled.
- **Commits that do not build alone.** Two lane commits fail `cargo check` on
  their own and are fixed by their children: c82962fc (the DSA worklet's
  `Glm53KpoolSparseMlaInput` initializer is updated in 2a19abd0) and 39444a07
  (`hybrid_kv_cache_layout` gains an argument; the call site is updated in
  7be2652e). Both are second parents of merges; every first-parent commit
  builds, so `git bisect --first-parent` is unaffected. A squash merge removes
  them.

## Alignment

- No alignment pack exists for the DFlash2 Flash deployments.
  `presets/alignment/` has `glm53_flash_fp8_b200_tp4_ep4` (FP8, TP4/EP4, no
  speculation) and `glm53_nvfp4_b200_dflash2` (GLM-5.3, not Flash).
- Existing evidence on the real deployment is three nsys captures of vLLM
  serving GLM-5.3-Flash NVFP4 + DFlash2 k7 at DP4 attention / EP4 MoE (c64,
  c128 twice), taken to audit the decode path (graph-mode coverage, request
  padding, sampler segment). They are not a diverse_100 replay, were not run
  through Check 1/2/3, and are not in this PR.
- Missing: a pack with a `phase: 1` diverse_100 case for each deployment, its
  label rules, and the Check 1 / Check 2 / Check 3 tables.

## Reproduce

- `just test-presets` builds every public member and checks that it finds
  every profile.db row it reads.
- The ladder table runs each public sim member's resolved run config with
  `uv run python -m launcher <config>.yaml --no-gpu`, the workload replaced by
  the ladder's k7 trace (`reported_600k_decode_k7.csv`, sha256 prefix
  6f72ac0c) at `workload.max_concurrency` = c, 3600 s. The configs and runs
  live in an untracked experiment tree.
- The other-model table is timing-predict on every public member at a fixed
  case set (decode 1x1k, 8x4k, 32x4k, 32 ragged, 64x4k, 128x2k, 256 ragged,
  32x60k; prefill 2048; one mixed step), once on main's row values and once on
  this branch's.
