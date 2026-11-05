# deepseek_v41_sglang: profile.db rows to fill on the rental

Generated on 2026-11-05 by the no-GPU dry run on GPU `NVIDIA RTX PRO 6000 Blackwell Server Edition`:

```bash
python -m launcher presets/dsv41_flash_rtxpro6000_sglang_tp4_ep4_{typical,peak}.yaml --cache-report
```

Both presets need the same set: **3,583 rows across 43 distinct kernel configs**, all missing (profile.db has no SM12x
rows). The target model only: DSpark is not modeled yet, and its draft stages and sampling chain will add rows.

## Per kind

| Kind | Backend | Rows | Distinct configs | Where |
|---|---|---:|---:|---|
| `elementwise` | triton (byte placeholders) | 1051 | 18 | B1 (1 GPU) |
| `paged_sparse_mla` | sglang_flashinfer_sm120 | 780 | 3 | B1 (1 GPU) |
| `single_gemm` | sglang_triton_block32_fp8 (fp8) / torch_linear (bf16: wo_a, lm_head) | 612 | 9 | B1 (1 GPU) |
| `sparse_indexer_logits_decode` | sglang_triton_fp4 | 432 | 1 | B1 (1 GPU) |
| `sparse_indexer_logits_prefill` | sglang_torch_fp4 | 160 | 2 | B1 (1 GPU) |
| `gemm_fp32_output` | torch_cublas (compressor kv_score) | 136 | 2 | B1 (1 GPU) |
| `nvfp4_fused_moe` | flashinfer_cutlass_sm120_mxfp4 | 73 | 1 | B1 (1 GPU) |
| `mhc_mix_stats_sinkhorn` | sglang_triton | 68 | 1 | B1 (1 GPU) |
| `mhc_combine_rms_norm` | sglang_triton | 68 | 1 | B1 (1 GPU) |
| `mhc_post` | sglang_auto | 68 | 1 | B1 (1 GPU) |
| `engram_gate` | sglang_triton | 68 | 1 | B1 (1 GPU) |
| `engram_lookup` | sglang_triton (host_shared_huge) | 48 | 2 | B1, host with about 110 GB free RAM and shmem_enabled=advise |
| `all_reduce` | nccl (4 ranks, PCIe) | 19 | 1 | B2 (4 GPUs; record P2P vs SHM) |
| **total** | | **3583** | **43** | |

Everything except `all_reduce` runs on one GPU (TP4/EP4 shard shapes), so B1 (1x RTX PRO 6000) can fill 3,564 of the
3,583 rows. The 19 `all_reduce` rows need the 4-GPU B2 host. `elementwise` rows are byte-rate placeholders and are
cheap; `paged_sparse_mla` (780) and the indexer kinds (592) are the slow ones.

## Per role (first role of each distinct config)

| Kind | Role | Rows |
|---|---|---:|
| `mhc_mix_stats_sinkhorn` | `unified.layer0.attn.mhc_mix_stats` | 68 |
| `mhc_combine_rms_norm` | `unified.layer0.attn.mhc_combine_norm` | 68 |
| `single_gemm` | `unified.layer0.attn.fused_wqa_wkv` | 68 |
| `elementwise` | `unified.layer0.attn.qkv_rmsnorm` | 63 |
| `single_gemm` | `unified.layer0.attn.wq_b` | 68 |
| `elementwise` | `unified.layer0.attn.rope_kv_store` | 63 |
| `elementwise` | `unified.layer0.attn.page_metadata` | 60 |
| `paged_sparse_mla` | `unified.layer0.attn.sparse_mla_decode` | 180 |
| `elementwise` | `unified.layer0.attn.inv_rope` | 54 |
| `single_gemm` | `unified.layer0.attn.wo_a` | 68 |
| `single_gemm` | `unified.layer0.attn.wo_b` | 68 |
| `all_reduce` | `unified.layer0.attn.tp_all_reduce` | 19 |
| `mhc_post` | `unified.layer0.attn.mhc_post` | 68 |
| `elementwise` | `unified.layer0.ffn.router` | 63 |
| `single_gemm` | `unified.layer0.ffn.shared_gate_up` | 68 |
| `elementwise` | `unified.layer0.ffn.shared_act` | 63 |
| `single_gemm` | `unified.layer0.ffn.shared_down` | 68 |
| `nvfp4_fused_moe` | `unified.layer0.ffn.routed_moe` | 73 |
| `engram_lookup` | `unified.layer1.engram.lookup` | 24 |
| `single_gemm` | `unified.layer1.engram.wkv` | 68 |
| `engram_gate` | `unified.layer1.engram.gate` | 68 |
| `gemm_fp32_output` | `unified.layers[2,8].attn.compressor_kv_score` | 68 |
| `elementwise` | `unified.layers[2,8].attn.compressor_store` | 63 |
| `elementwise` | `unified.layers[2,8].indexer.weights_proj` | 61 |
| `elementwise` | `unified.layers[2,8].indexer.wk_store` | 62 |
| `single_gemm` | `unified.layers[2,8].indexer.wq_b` | 68 |
| `elementwise` | `unified.layers[2,8].indexer.q_rope_quant` | 63 |
| `elementwise` | `unified.layers[2,8].indexer.torch_bookkeeping` | 28 |
| `sparse_indexer_logits_decode` | `unified.layers[2,8].indexer.logits_decode` | 432 |
| `sparse_indexer_logits_prefill` | `unified.layers[2,8].indexer.logits_prefill` | 80 |
| `paged_sparse_mla` | `unified.layers[2,8].attn.sparse_mla_decode` | 300 |
| `engram_lookup` | `unified.layer14.engram.lookup` | 24 |
| `gemm_fp32_output` | `unified.layer20.attn.compressor_kv_score` | 68 |
| `elementwise` | `unified.layer20.attn.compressor_store` | 63 |
| `elementwise` | `unified.layer20.indexer.wk_store` | 63 |
| `sparse_indexer_logits_prefill` | `unified.layer20.indexer.logits_prefill` | 80 |
| `paged_sparse_mla` | `unified.layer20.attn.sparse_mla_decode` | 300 |
| `elementwise` | `unified.prologue.embedding` | 63 |
| `elementwise` | `unified.prologue.engram_hash` | 63 |
| `elementwise` | `unified.prologue.hc_broadcast` | 63 |
| `elementwise` | `unified.head.hc_collapse` | 51 |
| `elementwise` | `unified.head.final_rmsnorm` | 42 |
| `single_gemm` | `unified.head.lm_head` | 68 |
