# Kernels: micro-benchmark against the profile rows

The first milestone, before any server work. Every kernel the design runs must compute the right result and run as
fast as the measured row the targets were built from. This file lists those kernels, gives their rows, and says how
to measure them. Nothing here needs the pipeline, the scheduler or the cache.

## 1. Where the targets come from

Every compute time in this package's targets is built from kernel times read from `profiling/profile.db` at the
frozen commit. A row there is one kernel at one shape on a B200: a *spec*, which is the kernel's fixed configuration
(weight shape, dtypes, heads) plus the per-call sizes (tokens, context, routing), and the measured time. Between
measured sizes the time is interpolated. A kernel is aligned when the real implementation, measured the same way at the same spec,
matches the row.

Where one request's compute time goes (all 45 layers and both heads, §2.1 one-request runs):

| Kernel family | 8k prompt | 128k prompt |
|---|---|---|
| Linear layers (`single_gemm`: BF16 projections, shared expert, `lm_head`; NVFP4 dense FFN) | 38.1% | 39.7% |
| Routed experts (`nvfp4_fused_moe`) | 22.8% | 15.8% |
| KDA chunked prefill | 12.8% | 13.5% |
| mHC boundary (fused post + pre RMSNorm) | 7.2% | 7.5% |
| Sparse MLA attention | 6.7% | 7.8% |
| Batched MLA GEMMs, short conv, FP32-output GEMMs | 5.5% | 5.5% |
| Glue ops (§6) | 4.3% | 4.3% |
| DSA indexer logits / top-k / index remap | 2.0% | 5.2% |
| Quant, norms, cache append | 0.6% | 0.7% |

Align in this order: the first five families carry 84–88% of the time. Each layer type's cost tree, with the time
of every kernel in it, is in `data/layers/` (`3_layers.md`).

## 2. The kernels

Positions are named by layer type: `first_kda_dense` is layer 0, `kda_dense` layers 1–2, `kda_moe` the other KDA
layers, `dsa_moe` the DSA layers (3, 7, 11, …). Head and tail ops sit on stage 0 and stage 3.

| Position | Library call | DB kind / backend | Fixed configuration | Rows |
|---|---|---|---|---|
| layer 0, first pre-norm | TileLang mHC pre RMSNorm (vLLM) | `mhc_pre_rms_norm` / `vllm_tilelang` | hidden 4096, 4 streams, BF16 | 68 |
| every attention and FFN boundary | DeepGEMM `mega_mhc`, one launch, non-shifted | `mhc_fused_post_pre_rms_norm` / `deepgemm_mega_nonshifted` | hidden 4096, 4 streams | 88 |
| KDA `in_proj` | `F.linear`, BF16 | `single_gemm` / `torch_linear_vllm` | k 4096, n 24,896 | 68 |
| KDA `f_b_proj`, `g_b_proj` | `F.linear`, BF16 | `single_gemm` / `torch_linear_vllm` | k 128, n 8192 | 68 |
| KDA short conv | Dao-AILab `causal-conv1d`, channel-last, one launch per sequence | `gdn_causal_conv_prefill` / `dao_channellast` | 24,576 channels, width 4 | 144 |
| KDA chunked prefill | FlashInfer `recurrent_kda(backend='cute-dsl-persistent')` | `kda_chunk_prefill` / `flashinfer_cute_persistent` | 64 heads, head dim 128 | 512 |
| KDA `o_proj` | `F.linear`, BF16 | `single_gemm` / `torch_linear_vllm` | k 8192, n 4096 | 68 |
| dense FFN input quant (layers 0–2) | vLLM `scaled_fp4_quant`, swizzled scales | `nvfp4_quant` / `vllm_cuda` | hidden 4096 (gate_up), 12,288 (down) | 68 + 68 |
| dense FFN `gate_up`, `down` | FlashInfer `mm_fp4` (vLLM `FlashInferCuteDslNvFp4LinearKernel`) | `single_gemm` / `flashinfer_cutedsl` | k 4096 n 24,576; k 12,288 n 4096 | 68 + 68 |
| DSA `fused_qkv_a` | `F.linear`, BF16 | `single_gemm` / `torch_linear_vllm` | k 4096, n 2048 | 68 |
| DSA q/kv A-norm | vLLM `fused_q_kv_rmsnorm` (Triton) | `q_kv_rms_norm` / `vllm_triton` | q 1536, kv 512 | 68 |
| DSA `q_b_proj` | `F.linear`, BF16 | `single_gemm` / `torch_linear_vllm` | k 1536, n 16,384 | 68 |
| indexer `wq_b`, `wk_weights`, pool gate | `F.linear`, BF16 | `single_gemm` / `torch_linear_vllm` | k 1536 n 4096; k 4096 n 160; k 4096 n 128 | 68 each |
| indexer head weights | `torch.mm`, FP32 input and output | `gemm_fp32_output` / `torch_cublas` | k 4096, n 32 | 68 |
| indexer logits | DeepGEMM `fp8_mqa_logits` (vLLM wrapper) | `dsa_mqa_logits_prefill` / `deepgemm_fp8` | 32 heads, head dim 128, FP8 | 336 |
| indexer top-k | DeepSelect `torch.ops.deep_select.topk` **or** vLLM `topKPerRowPrefill`, faster per shape | `dsa_topk_prefill` / `deep_select`, `vllm_cuda` | top 512 pools, causal with tail | 412 + 412 |
| MLA q absorb | `torch.bmm`, BF16 | `batched_gemm` / `torch_mla_q_absorb_no_rope` | 64 × (k 256, n 512) | 68 |
| KV cache write | vLLM `concat_and_cache_mla` | `mla_cache_append` / `vllm_cuda` | block 64, FP8, latent 512 | 70 |
| index remap | vLLM `triton_convert_req_index_to_global_index` | `dsa_sparse_index_remap` / `vllm_triton` | 2,176 selected slots | 325 |
| sparse MLA | FlashInfer `trtllm_batch_decode_with_kv_cache_mla`, sparse mode | `dsa_sparse_mla_attention` / `flashinfer_trtllm_fp8` | 64 heads, latent 512, 2,176 selected, FP8 q and cache | 1,324 |
| MLA V up | `torch.bmm`, BF16, unpadded | `batched_gemm` / `torch_mla_v_up_unpadded` | 64 × (k 512, n 256) | 68 |
| DSA `o_proj` | `F.linear`, BF16 | `single_gemm` / `torch_linear_vllm` | k 16,384, n 4096 | 68 |
| router gate | `torch.mm`, BF16 input, FP32 output | `gemm_fp32_output` / `torch_cublas` | k 4096, n 288 | 68 |
| routed expert input quant | vLLM `scaled_fp4_quant`, linear scales | `nvfp4_quant` / `vllm_cuda` | hidden 4096 | 68 |
| routed experts | FlashInfer `trtllm_fp4_block_scale_moe` | `nvfp4_fused_moe` / `flashinfer_trtllm_sm100` | 288 experts, top 8, intermediate 2048, DeepSeek-V3 routing, scale 5/2 | 73 |
| shared expert `gate_up`, `down` | `F.linear`, BF16 | `single_gemm` / `torch_linear_vllm` | k 4096 n 4096; k 2048 n 4096 | 68 + 68 |
| final mHC post (stage 3) | TileLang mHC (vLLM) | `mhc_fused_post_pre_rms_norm` / `vllm_tilelang` | hidden 4096, 4 streams | 69 |
| final norm | vLLM `rms_norm` | `rms_norm` / `vllm_cuda` | hidden 4096 | 63 |
| `lm_head` | `F.linear`, BF16 | `single_gemm` / `torch_linear_vllm` | k 4096, n 154,880 | 68 |

The router gate runs twice per MoE layer (`gate` and `gate_recompute`); both use the same rows.

## 3. The rows

`data/kernel_specs.jsonl` holds every row of these 35 configurations: 5,392 lines, one per spec.

```json
{"kind": "single_gemm", "backend": "torch_linear_vllm", "spec": {"m": 8192, "n": 2048, "k": 4096, "dtype": "bf16"},
 "db_time_ms": 0.09928, "positions": ["stage 0: pp.dsa_moe.dsa.fused_qkv_a", "..."]}
```

- The rows are the micro-benchmark: measure every one. Most configurations have rows up to 64k tokens in one call;
  KDA chunked prefill goes to 16k-token sequences, index remap to 16k queries. Sparse MLA goes to 32k queries, the most
  one launch takes; a longer call is several launches. Larger calls of the one-request bring-up have no row (§7);
  `3_layers.md` measures them inside their layers.
- To look up one spec, or one the file does not hold, query the DB from a ServingStudioSim checkout at the frozen
  commit. It reads only; it runs nothing:

  ```bash
  uv run python -m profiling query single_gemm --backend torch_linear_vllm --gpu-name "NVIDIA B200" \
    --spec '{"m":8192,"n":2048,"k":4096,"dtype":"bf16"}' --json
  ```
- To regenerate the file: `data/kernel_specs.py` reads the stage graphs of the §2.1 one-request runs and selects the
  matching B200 rows (usage in its docstring).

Two kernels need more than the shape:
- **Routed experts.** The spec's `per_expert_batches` lists how many tokens each of the 288 experts receives, from the
  measured routing. Build router logits that produce exactly these counts. Do not use random or uniform routing: the
  time depends on the skew.
- **Indexer top-k.** Both libraries have rows. The design calls the faster one for each shape; the real system should
  do the same. DeepSelect takes k ≤ 4,096 and rows shorter than 2²³.

## 4. How to measure

Measure the way the rows were measured, or the comparison is not fair:
- **Kernel time only.** GPU kernel duration from CUPTI (or Nsight Systems), not host wall time. When one call launches
  several kernels on several streams, count overlapping time once.
- **Cold L2.** Before every launch, read a buffer of twice the L2 size (about 250 MB on B200) so the inputs and
  weights are not in L2. Some older rows were measured after a 64 MiB read, which leaves part of B200's L2 warm; if a
  small kernel misses its row, measure it both ways and record both.
- **Enough launches.** Average over at least 100 ms of launches after a few warm-up launches; if the per-launch time
  drifts, take 2 s.
- **Alone on the GPU** (`1_contract.md` §5). Record `uptime`.
- **Autotuned kernels** (FlashInfer): use the tactic the serving process will use, for example by loading its
  persisted autotune configs, and record the tactic per spec.

A spec passes when the real time is within ±10% of `db_time_ms`.
- **Slower by more than 10%:** an implementation gap (§8).
- **Faster by more than 10%:** not a failure, but record it: the targets assume the row's time.

To tell a node difference from an implementation difference, re-measure a few rows with the profiler that wrote them,
on the real node, without changing the DB:

```bash
uv run python -m profiling run single_gemm --backend torch_linear_vllm \
  --spec '{"m":8192,"n":2048,"k":4096,"dtype":"bf16"}' --fresh --json
```

## 5. Correctness

Check each kernel's output at its smallest and largest row against a plain PyTorch reference in FP32, with the
tolerance the library's own tests use for that op.

| Kernel | Reference |
|---|---|
| BF16 and FP32-output GEMMs, batched GEMMs | `torch.matmul` in FP32 |
| NVFP4 quant, `mm_fp4`, routed experts | the same op on dequantized inputs and weights, FP32 |
| RMSNorm, q/kv norm, mHC pre/post | the formula in FP32 |
| KDA short conv | direct causal convolution in FP32 |
| KDA chunked prefill | token-by-token recurrence in FP32 |
| indexer logits | FP32 logits from the dequantized FP8 inputs |
| indexer top-k | the selected set equals `torch.topk` of the same logits, ties aside |
| index remap, cache write | exact match |
| sparse MLA | dense attention over the selected tokens, FP32 |

The end-to-end check of `1_contract.md` §4 still decides whether the build counts; this one finds a broken kernel
before it is buried in a stage.

## 6. Glue ops

The ops between the library kernels (casts, quant, gathers, activations) have no row of their own. The targets price
each as a plain kernel that reads and writes a given number of bytes, measured on B200; for the few ops whose measured
kernels run far below streaming bandwidth, the target uses their measured time. Together they are 3.6–4.9% of each
stage's compute.

There is no pass rule for them: make them as fast as you can, by fusing them into a neighbouring kernel or at least
running each at memory bandwidth.

Largest first (share of one 8k request's compute, all stages):

| Glue op | Layer types | Bytes per token charged, in / out | Share |
|---|---|---|---|
| `gated_norm` (KDA output: rmsnorm(o) · w · sigmoid(g)) | KDA | 32,768 / 16,384 | 0.87% |
| `prefill_glue` (top-k index plumbing, 8 launches) | DSA | 8,704 / 8,704 | 0.69% |
| `q_fp8_quant` (MLA query to FP8) | DSA | 65,536 / 33,024 | 0.56% |
| `combine_glue` (shared + routed expert combine) | MoE | 16,384 / 8,192 | 0.54% |
| `q_fwht_quant` (indexer query Hadamard-128 + quant) | DSA | 40,960 / 21,120 | 0.35% |
| `input_glue` (routed-input copy) | MoE | 8,192 / 8,192 | 0.31% |
| `act_and_mul`, shared expert | MoE | 8,192 / 4,096 | 0.27% |
| `glue` (index and mask plumbing, 17 launches) | DSA | 64 / 64 | 0.19% |
| `act_and_mul`, dense FFN | layers 0–2 | 49,152 / 24,576 | 0.11% |
| `expand_pools` (selected pools to token slots) | DSA | 4,100 / 8,204 | 0.10% |
| the rest: `state_gather`, `state_scatter`, `prefill_gather`, `hc_expand`, `hc_contract_mean`, `k_norm`, `kpool_prefill_write`, `weight_scale`, `kpool_tail_seed`, `embedding` | | | 0.27% |

## 7. Where these targets may be off

- **Indexer logits.** The B200 rows were measured on 2026-08-29. Re-measured on 2026-10-05, the same kernel ran up to
  15% faster (median 8%), most at large key counts. A real time up to 15% below the row is expected here.
- **Autotuned tactics.** FlashInfer's autotuner picks a tactic per process from its own probe inputs. On another
  NVFP4 MoE, profiling and serving picked different tiles at the same shape, and the times differed by 20%. If a MoE
  row misses, compare the tactic first.
- **Past the largest row.** A call larger than every row is extrapolated from the largest rows. In the one-request
  bring-up this covers KDA chunked prefill from 32k tokens and most kernels at 128k
  (`data/layer_targets.jsonl` marks each such call `beyond_rows`).
- **Interpolation.** A size between two rows is priced by interpolating them. The micro-benchmark covers the rows
  only; a kernel with a cliff between rows shows up in the per-layer times of `3_layers.md`.

## 8. What to record

**Environment** (`measured/kernels_env.md`): the versions or commits of vLLM, FlashInfer, DeepGEMM, DeepSelect,
TileLang, causal-conv1d, PyTorch, CUDA and the driver; the GPU; `uptime` at the start of each batch.

**Every row** (`measured/kernels.jsonl`), one line per line of `data/kernel_specs.jsonl`:

```json
{"kind": "single_gemm", "backend": "torch_linear_vllm", "spec": {"m": 8192, "n": 2048, "k": 4096, "dtype": "bf16"},
 "db_time_ms": 0.09928, "real_time_ms": 0.1012, "ratio": 1.019, "kernels": ["..."], "tactic": null}
```

- `ratio` is real ÷ DB. `kernels` lists the GPU kernel names the call launched. `tactic` is the autotuned choice,
  where there is one.

**Per configuration** (`measured/kernels.md`), one row for each of the 35 configurations of §2:

| Kind / backend | Fixed configuration | Rows | Median ratio | Min ratio | Max ratio | Rows > 1.10 | Rows < 0.90 |
|---|---|---|---|---|---|---|---|

Below the table, for each configuration with rows above 1.10: the sizes where they fall (for example "tokens ≥ 32k")
and the suspected cause.

**Node check** (§4): the specs re-measured with the profiler, with its time beside the DB time and yours.

**Correctness** (§5): for each kernel, the rows checked, the largest error, the tolerance, and pass or fail.

**Glue ops** (§6): for each §2.1 prompt length, the total glue time per stage, and which glue ops were fused into
which kernel.

**Gaps** (`issues.md`): every configuration with a row above 1.10, and every correctness failure, with the spec.
