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
