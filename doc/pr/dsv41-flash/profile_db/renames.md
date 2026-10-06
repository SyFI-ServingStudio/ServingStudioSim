# Post-merge renames (Phase A step 2)

Convention (main 50be2ba / 2e85d96): a kind names a callable's mechanism, not a
model; a backend names the library/call, not a model; `*_vllm_fork` backends are
folded into their canonical sibling. Arch, op, worklet and model-config names
keep the model (main keeps `glm53_kpool_sparse_mla`, `deepseek_v4_vllm`), so the
V4.1 L2 op `deepseek_v41_mega_attn` (`DeepseekV41MegaAttnOp`) and the
`deepseek_v41_*` worklets/arch are unchanged.

DB mapping code: `tmp/dsv41/merge/db_renames.py` (run on a v2 copy of the
backup BEFORE `kernel-profile migrate-db`). Main has no in-repo rename
mechanism: 4cae722 and 592bfd6 edited the DB directly.

## Kinds (= DB tables)

| old kind / table | new kind / table | merged into an existing kind? |
|---|---|---|
| `deepseek_v41_mega_attn` | `compressed_sparse_mla_rope_cast` | No. It is one fused launch (Q RoPE, sparse MLA over the SWA window plus compressed top-k rows, inverse RoPE, FP8 cast). Main's `compressed_sparse_mla_{decode,prefill}` take other args and exclude the RoPE/cast, so the kind is new. Its name follows main's `compressed_sparse_mla_*` family and FlashMLA's `fused_norm_rope_attn_rope_cast_*`. |
| `deepseek_v41_qnorm_rope_kv_insert` | `q_pad_kv_rope_mxfp8_insert` | No. Same CUDA op as main's `qnorm_rope_kv_insert` (`fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`), but with other flags and other args. The V4.1 launch does no Q norm or Q RoPE; it only pads Q in the interleaved layout. The KV insert is MXFP8 into the SWA cache, keyed by `swa_cache_format`, and the kind has no `head_dim`/`rope_dim`/`rms_eps`/`cache_dtype`/`cache_layout`/`scale_format` args. Merging needs a Q-transform axis on main's schema, and without one its rows would silently change meaning. |
| `engram_lookup` | unchanged | "Engram" names the mechanism (hashed n-gram embedding lookup), not a model. |

Main's V4 renames, which the backup still has under the old names (same rows as
main's v3 DB; the mapping only lines them up):

| old table | new table |
|---|---|
| `deepseek_v4_fused_inv_rope_fp8_quant` | `inv_rope_fp8_quant` |
| `deepseek_v4_fused_q_kv_rmsnorm` | `q_kv_rms_norm` |
| `deepseek_v4_indexer_mqa_logits_prefill` | `dsa_compressed_mqa_logits_prefill` |
| `deepseek_v4_indexer_topk_prefill` | `dsa_compressed_topk_prefill` |
| `deepseek_v4_indexer_q_rope_quant` | `dsa_indexer_q_rope_quant_weight_fold` |
| `deepseek_v4_packed_cache_gather` | `packed_kv_cache_gather` |
| `deepseek_v4_qnorm_rope_kv_insert` | `qnorm_rope_kv_insert` |
| `deepseek_v4_sparse_attn_compress_store` | `kv_compress_store` |
| `deepseek_v4_sparse_mla_decode` | `compressed_sparse_mla_decode` |
| `deepseek_v4_sparse_mla_prefill` | `compressed_sparse_mla_prefill` |
| `deepseek_v4_terminal_mhc_head` | `mhc_terminal_head` |
| `deepseek_v4_indexer_mqa_logits_decode` | merged into `dsa_paged_mqa_logits_decode`, backend `vllm_deepgemm_fp8` -> `deepgemm_fp8` |
| `deepseek_v4_indexer_topk_decode` | merged into `dsa_persistent_topk_decode`, backend `vllm_cuda` -> `vllm_fork_cuda` |

The V4.1 Rust/op/worklet code used none of the renamed V4 kinds, so no V4.1
call site changed for them.

## Backends

| kind | old backend | new backend | notes |
|---|---|---|---|
| `batched_gemm` | `deepgemm_mxfp8_einsum_dsv41_wo_a` | `deepgemm_mxfp8_einsum_grouped_o_proj` | The name gives the layout role, as main's `torch_mla_q_absorb_no_rope` does. |
| `gemm_fp32_output` | `torch_cublas_vllm_fork` | `torch_cublas` (fold) | Main folded it in 2e85d96. The backup holds k=5120 bf16 rows under both names, 204 each, all measured by this branch; main has no k=5120 rows. The V4.1 arch read the fork rows, so they win (`on_conflict=replace`) and the timings stay the same. |
| `batched_gemm` | `torch_mla_{q_absorb,v_up}_glm52` | `torch_mla_{q_absorb,v_up}` | main 50be2ba |
| `kv_compress_store` | `vllm_deepseek_v4_{cutedsl,triton}` | `vllm_{cutedsl,triton}` | main 50be2ba |
| `packed_kv_cache_gather` | `vllm_deepseek_v4_cutedsl` | `vllm_cutedsl` | main 50be2ba |
| `single_gemm` `flashinfer_mxfp8`, `nvfp4_fused_moe` `flashinfer_trtllm_sm100_mxfp4`, `mhc_fused_post_pre_rms_norm` `deepgemm_mega`, `compressed_sparse_mla_rope_cast` `flashmla_mega`/`torch`, `q_pad_kv_rope_mxfp8_insert` `vllm_cuda`, `engram_lookup` `vllm_triton`, `all_reduce_fusion` `flashinfer_mnnvl` | unchanged | No model name in any of them. |

## Args values

| table / backend | column | old | new |
|---|---|---|---|
| `nvfp4_fused_moe` / `flashinfer_trtllm_sm100_mxfp4` | `weight_format` | `mxfp4_ue8m0` | `mxfp4_e2m1` |

In main, `weight_format` is a `DType` and the `#[compute_dtype]` (b55eec3). The
merge adds `DType::Mxfp4E2m1` / `DType.MXFP4_E2M1` (wire `mxfp4_e2m1`, named
like `nvfp4_e2m1`), and the MXFP4 backend declares `compute={mxfp4_e2m1}`.

## Code renames (files)

- `profiling/kernels/deepseek_v41_mega_attn.py` -> `compressed_sparse_mla_rope_cast.py` (`CompressedSparseMlaRopeCastArgs`)
- `profiling/runners/attention/deepseek_v41_mega_attn{,_reference}.py` -> `compressed_sparse_mla_rope_cast{,_reference}.py` (`profile_compressed_sparse_mla_rope_cast_{torch,flashmla_mega}`, `compressed_sparse_mla_rope_cast_reference`)
- `profiling/kernels/deepseek_v41_qnorm_rope_kv_insert.py` -> `q_pad_kv_rope_mxfp8_insert.py` (`QPadKvRopeMxfp8InsertArgs`)
- `profiling/runners/attention/deepseek_v41_qnorm_rope_kv_insert.py` -> `q_pad_kv_rope_mxfp8_insert_vllm_cuda.py` (`profile_q_pad_kv_rope_mxfp8_insert_vllm_cuda`)
- `profiling/runners/gemm/deepgemm_mxfp8_einsum.py`: `profile_batched_gemm_deepgemm_mxfp8_einsum_grouped_o_proj`
- `simulator/src/timing/kernels/deepseek_v41_mega_attn.rs` -> `compressed_sparse_mla_rope_cast.rs` (`CompressedSparseMlaRopeCast{Kernel,KernelConfig,KernelInput,Spec}`, slot variant `CompressedSparseMlaRopeCast`)
- `simulator/src/timing/kernels/deepseek_v41_qnorm_rope_kv_insert.rs` -> `q_pad_kv_rope_mxfp8_insert.rs` (`QPadKvRopeMxfp8Insert*`)
- tests: `test_compressed_sparse_mla_rope_cast.py`, `test_q_pad_kv_rope_mxfp8_insert.py`
- tools: `cache-fidelity-analyzer/q_pad_kv_rope_mxfp8_insert_fidelity.py`

Compat: presets and label rules name no kind or backend. Launcher-registered
kernel configs are not yet in main's DB for V4.1. So the only compat surface
is profile.db, and `db_renames.py` covers it. There are no old-name aliases
in code.
