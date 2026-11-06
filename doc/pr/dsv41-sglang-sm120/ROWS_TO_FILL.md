# deepseek_v41_sglang_dspark: profile.db rows to fill on the rental

Generated on 2026-11-05 by the no-GPU dry run on GPU `NVIDIA RTX PRO 6000 Blackwell Server Edition`:

```bash
python -m launcher presets/dsv41_flash_rtxpro6000_sglang_tp4_ep4_{typical,peak}.yaml --cache-report
```

Both presets (now `deepseek_v41_sglang_dspark` + worker `speculative`, DSpark block 6) need the same set:
**4,168 rows across 52 distinct kernel configs**, all missing (profile.db has no SM12x rows). Of these, 582 rows
are the DSpark draft's own configs (KV inject, draft MoE at 128 experts top-3, embed, Markov chain); the draft's
attention, mHC, shared expert, all-reduces, lm_head and norms share configs with the target and add no rows. The
target's decode indexer is now captured at `next_n = 7` (the verify width). The non-speculative
`deepseek_v41_sglang` arch needs its `next_n = 1` indexer config instead (432 rows) and none of the draft rows.

## Per kind

| Kind | Backend | Rows | Distinct configs | Of which DSpark draft | Where |
|---|---|---:|---:|---:|---|
| `elementwise` | triton (byte placeholders) | 1429 | 24 | 378 | B1 (1 GPU) |
| `paged_sparse_mla` | sglang_flashinfer_sm120 | 780 | 3 | 0 | B1 (1 GPU) |
| `single_gemm` | sglang_triton_block32_fp8 (fp8) / torch_linear (bf16: wo_a, lm_head, DSpark lm_head + markov w2) | 748 | 11 | 136 | B1 (1 GPU) |
| `sparse_indexer_logits_decode` | sglang_triton_fp4 (next_n 7: DSpark verify) | 432 | 1 | 0 | B1 (1 GPU) |
| `sparse_indexer_logits_prefill` | sglang_torch_fp4 | 160 | 2 | 0 | B1 (1 GPU) |
| `nvfp4_fused_moe` | flashinfer_cutlass_sm120_mxfp4 (target 384/top-6 + draft 128/top-3) | 144 | 2 | 68 | B1 (1 GPU) |
| `gemm_fp32_output` | torch_cublas (compressor kv_score) | 136 | 2 | 0 | B1 (1 GPU) |
| `mhc_mix_stats_sinkhorn` | sglang_triton | 68 | 1 | 0 | B1 (1 GPU) |
| `mhc_combine_rms_norm` | sglang_triton | 68 | 1 | 0 | B1 (1 GPU) |
| `mhc_post` | sglang_auto | 68 | 1 | 0 | B1 (1 GPU) |
| `engram_gate` | sglang_triton | 68 | 1 | 0 | B1 (1 GPU) |
| `engram_lookup` | sglang_triton (host_shared_huge) | 48 | 2 | 0 | B1, host with about 110 GB free RAM and shmem_enabled=advise |
| `all_reduce` | nccl (4 ranks, PCIe) | 19 | 1 | 0 | B2 (4 GPUs; record P2P vs SHM) |
| **total** | | **4168** | **52** | **582** | |

Everything except `all_reduce` runs on one GPU (TP4/EP4 shard shapes), so B1 (1x RTX PRO 6000) can fill 4,149 of the
4,168 rows; the 19 `all_reduce` rows (one config serves the TP all-reduces, the logits gather and the DSpark
Markov gather/broadcast) need the 4-GPU B2 host.

## Per role (first role of each distinct config)

| Kind | Role | Rows |
|---|---|---:|
| `mhc_mix_stats_sinkhorn` | `unified.target.layer0.attn.mhc_mix_stats` | 68 |
| `mhc_combine_rms_norm` | `unified.target.layer0.attn.mhc_combine_norm` | 68 |
| `single_gemm` | `unified.target.layer0.attn.fused_wqa_wkv` | 68 |
| `elementwise` | `unified.target.layer0.attn.qkv_rmsnorm` | 63 |
| `single_gemm` | `unified.target.layer0.attn.wq_b` | 68 |
| `elementwise` | `unified.target.layer0.attn.rope_kv_store` | 63 |
| `elementwise` | `unified.target.layer0.attn.page_metadata` | 60 |
| `paged_sparse_mla` | `unified.target.layer0.attn.sparse_mla_decode` | 180 |
| `elementwise` | `unified.target.layer0.attn.inv_rope` | 54 |
| `single_gemm` | `unified.target.layer0.attn.wo_a` | 68 |
| `single_gemm` | `unified.target.layer0.attn.wo_b` | 68 |
| `all_reduce` | `unified.target.layer0.attn.tp_all_reduce` | 19 |
| `mhc_post` | `unified.target.layer0.attn.mhc_post` | 68 |
| `elementwise` | `unified.target.layer0.ffn.router` | 63 |
| `single_gemm` | `unified.target.layer0.ffn.shared_gate_up` | 68 |
| `elementwise` | `unified.target.layer0.ffn.shared_act` | 63 |
| `single_gemm` | `unified.target.layer0.ffn.shared_down` | 68 |
| `nvfp4_fused_moe` | `unified.target.layer0.ffn.routed_moe` | 76 |
| `engram_lookup` | `unified.target.layer1.engram.lookup` | 24 |
| `single_gemm` | `unified.target.layer1.engram.wkv` | 68 |
| `engram_gate` | `unified.target.layer1.engram.gate` | 68 |
| `gemm_fp32_output` | `unified.target.layers[2,8].attn.compressor_kv_score` | 68 |
| `elementwise` | `unified.target.layers[2,8].attn.compressor_store` | 63 |
| `elementwise` | `unified.target.layers[2,8].indexer.weights_proj` | 61 |
| `elementwise` | `unified.target.layers[2,8].indexer.wk_store` | 62 |
| `single_gemm` | `unified.target.layers[2,8].indexer.wq_b` | 68 |
| `elementwise` | `unified.target.layers[2,8].indexer.q_rope_quant` | 63 |
| `elementwise` | `unified.target.layers[2,8].indexer.torch_bookkeeping` | 28 |
| `sparse_indexer_logits_decode` | `unified.target.layers[2,8].indexer.logits_decode` | 432 |
| `sparse_indexer_logits_prefill` | `unified.target.layers[2,8].indexer.logits_prefill` | 80 |
| `paged_sparse_mla` | `unified.target.layers[2,8].attn.sparse_mla_decode` | 300 |
| `engram_lookup` | `unified.target.layer14.engram.lookup` | 24 |
| `gemm_fp32_output` | `unified.target.layer20.attn.compressor_kv_score` | 68 |
| `elementwise` | `unified.target.layer20.attn.compressor_store` | 63 |
| `elementwise` | `unified.target.layer20.indexer.wk_store` | 63 |
| `sparse_indexer_logits_prefill` | `unified.target.layer20.indexer.logits_prefill` | 80 |
| `paged_sparse_mla` | `unified.target.layer20.attn.sparse_mla_decode` | 300 |
| `elementwise` | `unified.target.prologue.embedding` | 63 |
| `elementwise` | `unified.target.prologue.engram_hash` | 63 |
| `elementwise` | `unified.target.prologue.hc_broadcast` | 63 |
| `elementwise` | `unified.target.head.hc_collapse` | 51 |
| `elementwise` | `unified.target.head.final_rmsnorm` | 42 |
| `single_gemm` | `unified.target.head.lm_head` | 68 |
| `elementwise` | `unified.dspark.stage.ffn.router` | 63 |
| `nvfp4_fused_moe` | `unified.dspark.stage.ffn.routed_moe` | 68 |
| `single_gemm` | `unified.dspark.kv_inject.kv_proj` | 68 |
| `elementwise` | `unified.dspark.kv_inject.store` | 63 |
| `elementwise` | `unified.dspark.embed` | 63 |
| `elementwise` | `unified.dspark.markov.w1` | 63 |
| `single_gemm` | `unified.dspark.markov.w2` | 68 |
| `elementwise` | `unified.dspark.markov.sample_partial` | 63 |
| `elementwise` | `unified.dspark.markov.sample_finish` | 63 |

## B1 result (2026-11-06, Massed Compute 1x RTX PRO 6000 VM, $2.79)

Filled **4,149 of 4,168** rows for the DSpark presets, and the non-speculative graph's extra configs (`next_n = 1`
decode indexer, no-draft MoE). That is 4,653 RTX PRO 6000 rows in all, merged by row into this worktree's
`profile.db` (4,656 inserted, 0 conflicts). Only the 19 `all_reduce` rows remain; they need the 4-GPU B2 host.

Every runner passed its reference check on SM120. Fixes made during bring-up (commits `045dd716` and two follow-ups):
- **`paged_sparse_mla`:** bound 5%. FlashInfer's SM120 decode kernel differs from SGLang's torch reference by a
  steady 3.05-3.07%; the prefill path stays under 2%.
- **`mhc_post`:** publish SGLang's default config (its TileLang path above 384 rows reads `get_parallel()`).
- **Indexer prefill:** a shape with no visible compressed key is a measured zero.
- **Compressor `kv_score`:** moved to `gemm_fp32_output/sglang_router_auto` (SGLang's `linear_bf16_fp32`). Its v0.5.23
  import paths are fixed.

Runtime used: `lmsysorg/sglang:v0.5.23-cu130` (checkout `297e54fd10`):
- flashinfer-python / cubin / jit-cache 0.7.2 (cu130);
- sglang-kernel 0.4.10;
- torch 2.14.1+cu130;
- triton 3.8.0.

In the container, `sglang_env`'s `.venv-sglang` must be a symlink to `/opt/sglang` (a symlink only to `bin/python`
loses the venv), and `mold` plus `binutils` must be installed for the simulator build. Logs and the rental's
`profile.db`: airbox-cost-calculator `docs/clients/decycle/sim/kt_bench/results_b1_1xpro6000/`.

## B2 result (2026-11-06, Massed Compute 4x RTX PRO 6000 VM without P2P, 1 h 21 m, $11.43)

- **`all_reduce` / `nccl`:** the 19 rows are filled (4 ranks, SHM transport) and merged by row (0 conflicts). **Both
  Decycle presets now resolve every row** (4,168 / 4,168 on RTX PRO 6000).
- **Full-day simulations ran.** They are uncalibrated (`gpu_time_multiplier` 1.0, placeholder DSpark acceptance 0.8):
  - typical: TTFT p50 / p90 / p99 0.68 / 3.6 / 13.6 s, TPOT p50 / p99 8.3 / 90 ms;
  - peak: 0.79 / 7.1 / 23.1 s and 12.4 / 111 ms.

  Outputs are in airbox-cost-calculator `docs/clients/decycle/sim/kt_bench/results_b2_4xpro6000/sim/`.
- **Campaign (DSpark variant, trimmed to fit 2 h):** 2 nsys captures (`07_balanced_anchor_c16`, warm rerun;
  `01_decode_graph_sizes_c16`, partial), and 1 workload-metrics pass (`07`). Every pass **stopped at extraction
  validation**: "alignment API output_tokens does not match EngineCore num_output_tokens". The raw artifacts (nsys
  reports, replay.jsonl, server logs with alignment records) are saved, so after the fix they re-extract with
  `--resume` on kisnis, without a GPU.

Blockers found, to fix before the next capture:
1. **`FLASHINFER_DISABLE_JIT=1` (required by the SGLang alignment validator)** breaks SM120: the image's FlashInfer
   JIT cache (0.7.2+cu130) has no SM120 build of `mxfp8_quantize` (the MoE path), so the server dies at its first
   forward. Workaround used: a `fork_python` wrapper that unsets the variable for the engine only. Proper fix: either
   add those modules to the pinned jit-cache contract, or let a pack opt out on SM12x.
2. **Speculative token accounting in the ported fork.** The API-side `output_tokens` and the engine-side
   `num_output_tokens` disagree under DSpark. Check the hand-ported `mark_vibesim_token_output` /
   `_maybe_emit_vibesim_api_timing` against how the scheduler counts accepted draft tokens.
3. **Non-fatal upstream SGLang error** during DSpark draft-block graph capture: `Cannot copy between CPU and CUDA
   tensors during CUDA graph capture` (`deepseek_v4_backend.py:2215`, `init_forward_metadata_dspark_draft_block`).
   The server still came up.
4. **The first nsys capture after a cold start recorded JIT compilation,** not steady-state work. Warm the caches
   (one throwaway server start) before capturing.

## Re-extraction on kisnis (2026-11-06, after the token-count fix)

**Root cause.** The ported fork's EngineCore record counted `len(req.output_ids)`. Under DSpark a verify step
appends the whole accepted block, so that runs up to 6 tokens past the stop condition; the API record counts the
delivered tokens (`finished_len`). 41 of 65 requests differed by 1-6.

**Fix.** `vibesim-alignment-v0.5.23` `60104cb296`: `visible_output_tokens()` counts through `finished_len`, with
3 unit tests (18 pass).

**Saved logs.** These predate the fix. `kt_bench/correct_spec_token_counts.py` set the engine count to the API
count in a copy and recomputed `engine_core_tpot_ms`; the originals are kept as `.orig`. Counts were changed only
where the overshoot was 1-6 tokens; any other difference would abort the script.

**Re-extraction with `--resume`, all OK:**
- uses Nsight Systems 2026.5.2 unpacked to `/mnt/storage/yilegu/tools/` (kisnis's 2025.3.2 cannot read 2026.4.1
  reports) and enwik9 at `/mnt/storage/yilegu/hf_stub/`;
- `07_balanced_anchor_c16/profile_nsys`: 94 iterations, 1.20M kernel rows, 64 request timings;
- `07_balanced_anchor_c16/profile_workload`: 488 iterations;
- `01_decode_graph_sizes_c16/profile_nsys`: 116 iterations, 1.43M kernel rows;
- artifacts in `logs/b2_reextract/` (local).

**DSpark acceptance on real text** (enwik9 prompts): mean accept length **2.50 tokens per verify step**, about 1.5 of
6 drafts accepted, which is a uniform per-position p of about 0.61. The placeholder 0.8 implies 3.95, and the random
token-id prompts gave 4.3+. The SGLang fork records no per-position counts (vLLM's `spec_decode_metrics` has no
SGLang counterpart yet), so only the mean is measured.

**Next:**
1. Write the label rules from `07_balanced_anchor_c16/profile_nsys/parsed.json` (the folded SM120 launch
   programs).
2. Run `alignment-campaign label` / `extract` / `analyze_kernel`, which gives `gpu_time_multiplier` and the
   per-op errors.
3. Re-simulate the 2026-11-05 windows with p of about 0.61 and the multiplier.
