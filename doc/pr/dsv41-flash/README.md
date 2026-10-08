# DeepSeek-V4.1-Flash evidence (B200, vLLM TP4/EP4)

Review evidence for the `deepseek_v41_vllm` arch: what the simulator models,
how it was checked against a vLLM capture, and what is still open. This is not
an accepted baseline. The kernel alignment is within 5% overall, with
explained deviations left unfixed. TPOT and throughput have not been compared
on a pass without the profiler. No golden is recorded.

Raw captures, Analyzer bundles and profiling jobs stay in the experiment tree
`logs/20260924_0_dsv41_flash_capture/` and later `logs/2026092*_dsv41_*` runs
(untracked). Numbers are tagged *measured* (a vLLM capture or a B200 profiling
run), *simulated* (a timing prediction or simulation), *catalog* (the
checkpoint config, model card or fork source) or *derived* (arithmetic on
those).

## Deployment modeled

vLLM V1 on 4x B200: TP4 attention and EP4 experts over the same ranks, chunked
prefill at 2048 tokens, `--block-size 128`, `--kv-cache-dtype fp8`,
`max_cudagraph_capture_size 2048`, Engram tables CPU-offloaded. The server is
the vLLM fork `servingstudio-alignment` at `892da082`, rebased onto upstream
main `04730e8`, the newest main with a `cu130` wheel that serves
DeepSeek-V4.1.

| Capture | Workload | Use |
|---|---|---|
| 1 | `trace/quadrant_c48_96.csv` | CUDA graphs capped at 128 tokens, so every mixed iteration ran eager. KV reconciliation only. |
| corpus | `trace/diverse_100.csv` | Token corpus of routed-expert ids (135,144 generated tokens; prompt routes not recorded). Published as the `diverse_100` capture of the dataset `UW-SyFI/servingstudio-workload`. |
| 2 | `trace/quadrant_c48_96.csv` (96 requests, saturated at 48) | The reference capture: 2236 iterations (2113 decode, 122 mixed, 1 prefill). |

Capture 2's decode kernel time (*measured*, device 0, 12.877 ms per
iteration): routed MoE 38.2%, dense MXFP8 GEMMs 12.2%, mega attention 11.0%,
mHC 8.3%, all-reduce 8.1%.

## How each model feature is modeled

| Feature | vLLM | Simulator |
|---|---|---|
| Engram (layers 1, 14) | Hashed n-gram lookup from pinned host memory over UVA, on side streams, consumed before the Engram all-gather. | `engram_lookup` with the exact `table_rows`; the two lookups are charged serially after the layer-0 main path (below). |
| mHC (`hc_mult` 4) | One DeepGEMM `mega_mhc` launch per block boundary, 77 per iteration; non-fused kernels at layer 0, Engram and the head. | `mhc_fused_post_pre_rms_norm` / `deepgemm_mega`; the non-fused kernels are `elementwise` placeholders. |
| Compression ratios 0 / 2 / 1 | Layers 0-1 sliding window only, 2-19 pool two tokens, 20-39 one per token; every layer keeps a 128-token MXFP8 window. | `compress_ratio` is a config axis of `compressed_sparse_mla_rope_cast`; keys per token in closed form. |
| KV and index sharing | Only layers 2, 8, 14, 20 own a compressed cache; index sources 24-36 share layer 20's index cache. | Each layer maps to its source when its attention and indexer inputs are built. |
| Candidate selection | Layer 20 keeps the top 2048 8-token blocks; layers 24-36 score only those. | An `elementwise` placeholder sized by the logits it streams. |
| Causal encoder-decoder | The fork runs all 40 layers on every token. | All 40 layers, as the fork does. `decoder_swa_bounded_replay` (default off, no capture behind it) runs layers 21-39 on each prefill chunk's last 128 tokens, as vLLM PR #58132 and SGLang's `--enable-decoder-swa-bounded-replay` do. |
| Side streams | The shared expert runs beside the routed experts up to 256 tokens; the compressor and indexer input GEMMs beside `fused_wqa_wkv` up to 1024; the compressor state save only in FULL-graph decode. | `CostNode::Parallel` under those thresholds, a serial copy above. The `deepseek_v41_vllm_serial_streams` arch serializes all of them. |

MTP, the vision tower and `max_num_seqs` are not modeled.

### KV accounting

The arch charges 1056 B per token per rank, 4224 B over the four ranks
(*derived* from the fork's `get_kv_cache_spec`). vLLM forms seven KV groups
from one pool, every block 135,168 B: five sliding-window groups, one group of
the compressed and index pages of the four KV sources, and one compressor-state
ring. Only the compressed-and-index group grows with context: 135,168 B / 128
tokens = 1056 B. The sliding windows and rings are per-request constants (about
190 MB at 48 in flight), so the existing `FullAttnKv` store fits.

Checked against both captures by running the fork's `get_kv_cache_configs` on
CPU. Capture 1 predicted 53,181,494 tokens against 53,178,702 logged. Capture 2
logged 46,767,420, which inverts to 603,004 blocks = 81.5068e9 B on the
smallest rank. `attn_gpu_memory_gb: 81.506844672` therefore gives the simulator
77,184,512 tokens.

At `max_model_len` 1048576 the public sim presets use 76.066738176 GB (562,757
blocks). This is an estimate, not a measurement. It is the 131072 pool less
about 5.4 GB of workspace that `max_model_len` sizes: the FlashMLA prefill
workspace (0.61 to 4.84 GB) and the indexer K gather (about 1.2 GB).

## Kernels

| Kind / backend | What it times | Fidelity, cache vs fresh measurement |
|---|---|---|
| `compressed_sparse_mla_rope_cast` / `flashmla_mega` (new kind) | One fused FlashMLA launch: Q RoPE, sparse MLA over the window plus top-512 compressed rows, inverse RoPE, FP8 cast; prefill adds the chunk gather and index combine. | Decode 137/139 within ±15%, median 0.995; 200K-1M context 20/20, median 0.996. |
| `q_pad_kv_rope_mxfp8_insert` / `vllm_cuda` (new kind) | Q pad plus MXFP8 sliding-window KV insert. | 41/41 within ±15%, median 0.999. |
| `engram_lookup` / `vllm_triton` (new kind) | One Engram table gather, host UVA or device. | Host UVA 39/40 within ±15%, median 1.034. |
| `single_gemm` / `flashinfer_mxfp8` (new backend, dtype `mxfp8_e4m3`) | vLLM's MXFP8 linear. | 390/420 within ±15%; every decode-size probe within. |
| `nvfp4_fused_moe` / `flashinfer_trtllm_sm100_mxfp4` (new backend, `weight_format` `mxfp4_e2m1`) | MXFP8 activations x MXFP4 experts, routing to finalize in one call, priced on the critical EP rank. | 22/24 within ±15%, median 1.003. |
| `mhc_fused_post_pre_rms_norm` / `deepgemm_mega` (new backend) | DeepGEMM `mega_mhc` at hidden 5120. | 138/138 within ±15%, median 1.001. |
| `batched_gemm` / `deepgemm_mxfp8_einsum_grouped_o_proj` (new backend) | `wo_a` as the grouped MXFP8 einsum. | m <= 32768: 99.5% within ±15%. |
| `gemm_fp32_output` / `torch_cublas` (extended to B200, k 5120) | Router gate and compressor GEMMs. | 92.6-93.0% within ±15%; misses are cuBLAS tactic switches. |
| `all_reduce_fusion` / `flashinfer_mnnvl` (B200 rows) | MNNVL all-reduce on [T, 5120] over TP4. | 32/33 within ±15%, median 1.003. |

## Check 1: kernel alignment against capture 2

Duration-weighted signed error of the simulated critical path against measured,
over 2236 iterations (*simulated* vs *measured*):

| Round | All | Decode | Mixed |
|---|---:|---:|---:|
| Initial | −9.69% | −10.43% | −5.89% |
| After fix 1 | **−4.55%** | **−4.65%** | **−4.00%** |

Fix 1 was the CUPTI L2-flush fix (below) with the B200 rows re-measured, and
the serial Engram rule. Mapping covers 0.982 of measured critical-path time;
the remainder (position cast, sampling, metadata, indexer workspace fill) is
declared. The recommended `gpu_time_multiplier` is 1.0361, which the presets
use.

Per token bucket after fix 1 (ms per iteration):

| Bucket (n) | Measured | Simulated | Ratio |
|---|---:|---:|---:|
| decode (2113) | 11.117 | 10.600 | 0.953 |
| mixed 129-512 (30) | 16.084 | 15.384 | 0.956 |
| mixed 1025-1536 (12) | 34.118 | 32.535 | 0.954 |
| mixed 1537-2048 (80) | 45.413 | 43.712 | 0.963 |

Read the alignment by group. The Analyzer gives a hidden `Parallel` branch
zero simulated time and gives the measured time to the launch that started
first, so a row such as `ffn.shared.*` at T <= 256 is an attribution artifact,
not missing work.

What remains, with diagnoses:

1. **Routed MoE** reads +6-7% in mixed iterations. The corpus holds only
   generated-token routes while T ≈ 2048 iterations are prompt chunks; a
   contiguous corpus window already lowers the critical-rank time from 17.5 to
   16.65 ms per iteration, against 14.98 measured.
2. **Placeholders that should be L1 kinds**: `attn.indexer.candidates`
   0.07-0.10x, `attn.indexer.prefill_topk` 0.16-0.18x, `ffn.routed.topk` 23x at
   decode and 0.32x at large T, `attn.qk_rmsnorm` 0.58x at 2048. Together about
   −0.7 ms at T ≈ 2048.
3. **Isolated vs in-server timing.** At large T the prefill attention reads
   0.91-0.92x, the KV insert 0.81-0.86x and `mega_mhc` 0.92-0.94x, most of the
   attention gap. Small GEMMs read high after the L2 fix (router gate
   1.5-1.8x, `fused_wqa_wkv` 1.16-1.25x): a 253 MiB cold flush over-prices
   weights that stay L2-warm across graph replay. This needs a per-kind
   decision.
4. **Engram lookups** read 0.73-0.74x at large T; the window residual is only
   −0.13 to −0.15 ms.

## Simulation against capture 2

`presets/deepseek_v41_flash_b200_vllm_tp4_ep4.yaml`, `chunked_prefill`:

| Metric | Simulated | Measured |
|---|---|---|
| Requests | 96 | 96 |
| Makespan | 51.85 s | 54.0 s with the 124.9 s nsys stall removed (178.9 s raw) |
| Throughput | 6872.5 tok/s | 6598.9 tok/s stall-removed |
| TTFT mean / p50 / p90 | 539.2 / 147.0 / 1587.0 ms | 567.7 / 152.1 / 1700.7 ms (engine core) |

The stall is one gap after `cudaProfilerStop`, so capture 2's raw TPOT and
throughput measure the profiler. TPOT needs a pass without nsys.

## Bounded replay (counterfactual)

Timing predictions, ms (*simulated*):

| Case | Off | On |
|---|---:|---:|
| decode, 48 requests | 10.6022 | 10.6022 |
| mixed: chunks 1912 + 91, 45 decodes | 43.5739 | 32.3216 |
| 128-token chunk + 46 decodes | 15.4371 | 15.4371 |
| cold 2048-token chunk | 42.0314 | 28.8665 |

On capture 2's workload the simulation moves from 6872.5 to 7019.9 tok/s,
TTFT mean from 539.2 to 398.8 ms and TPOT mean from 14.09 to 12.99 ms.

## Long context

`max_model_len` defaults to the checkpoint's 1048576. The 1330
`compressed_sparse_mla_rope_cast` rows keyed on 1048576 were profiled on B200;
against the 131072 rows of the same shape they read median 1.007.

Timing predictions at `max_model_len` 1048576, ms (*simulated*, branch head):

| Case | Total |
|---|---:|
| decode 1 x 1,048,575 | 6.4124 |
| decode 8 x 1,048,575 | 8.5652 |
| 2048-token chunk at prefix 131,072 | 57.1686 |
| 2048-token chunk at prefix 524,288 | 122.3895 |
| 2048-token chunk at prefix 1,046,528 | 288.0901 |
| last chunk of a 500K-token request + 7 decodes at 500K | 114.2140 |

Attention and indexer-logits leaves are on the measured grid at every context.
What reads past the grid: the three indexer placeholders `prefill_topk`,
`prefill_k_gather` and `candidates`, whose context-sized inputs exceed the
`elementwise` grid (65,536 tokens) and which hold the grid edge's bandwidth;
and, in decode, `elementwise` leaves below the 32-token grid start of a small
batch. At a 500K prefix the held placeholders are about 7.5% of the iteration.

The 1M sim members ran `diverse_100` (100/100 requests), four 400K-token
prompts and two 900K-token prompts through the public API.

## Changes that reach other models

- **CUPTI L2 flush.** `default_l2_flush_bytes` read `props.l2_cache_size`,
  which torch spells `L2_cache_size`, so every cold-L2 `Timer.cupti` launch
  flushed only the 64 MiB floor. On B200 it now flushes 2 x L2. Rows profiled
  before the fix on GPUs with more than 32 MiB of L2 are warm-biased. Only
  the V4.1 B200 rows were re-measured, and 308 B200 `elementwise` / `triton`
  rows other models read take the re-measured values (median +3.3%, −31% to
  +104%).
- **`vllm_upstream_fork_env`**, a host profiling env on the fork's `.venv`
  (or `$VIBESIM_VLLM_FORK_ROOT`), because the `vllm_env` container is built
  from a vLLM with no DeepSeek-V4.1. Seven backends use it until the container
  is rebuilt from the rebased fork.
- **Dtypes** `mxfp8_e4m3` and `mxfp4_e2m1`, with record-size constants tied
  between Rust and Python by a test.
- **`name_exact`** label-rule key, because two vLLM launches are literally
  named `kernel`.
- **model.work**: `MatmulGroup` storage and compute dtypes, optional `mhc` and
  `engram` buckets, `scale_fmt: ue8m0` and `expert_dtype: fp4`. V4.1's
  logical parameters are 748,494,669,424, equal to the checkpoint tensors
  without MTP, the vision tower and the router `bias_vl`.

## Modeling choices

- **Engram lookups are serial.** Nothing waits on a lookup in capture 2, but
  the lookups take up to half the SMs for 0.6-0.8 ms and slow the main path
  beside them: the window grows about 0.70 ms per ms of lookup at T ≈ 2048.
  `Sum[main, lookups]` stands in for that contention.
- **All-gather as all-reduce.** There is no all-gather kind; the Engram and
  logits gathers are priced on the `all_reduce` curve at half the gathered
  bytes. Small gathers read high (about 28 vs 15 µs) and the logits gather
  low (41.6 vs 50.7 µs).
- **MoE histogram fold.** Per-layer corpus histograms are sorted by rank and
  averaged, and one critical-rank histogram is priced. That reads about
  −0.20 ms per decode iteration against pricing each layer's busiest rank.
- **CUDA-graph padding** to vLLM's capture sizes up to 2048.
- **Fixed step budget.** The attention config fixes `max_num_batched_tokens`
  at 2048; taking it from the worker needs rows at other budgets.

## Known gaps and follow-up

- Re-measure the warm-biased `Timer.cupti` rows, at least the 966
  indexer-logits rows V4.1 reads, then other models and GPUs.
- Fidelity round 2: prompt routes in the token corpus, an order-statistic MoE
  fold, L1 kinds for the placeholders above, a router-gate backend for T <= 16,
  a per-kind cold/warm L2 decision for small GEMMs.
- A workload pass without nsys to compare TPOT and throughput.
- model.work counts full prefill work; the encoder-decoder floor and
  per-request geometry are follow-up.
- Measure the 1M server's KV pool, and the held indexer placeholders at 524K
  and 1M.
- An SGLang capture with bounded replay on and off.
- `routing_method precomputed_dsv4` still names a model; renaming it is a DB
  value migration.

## Reproduce

From the repository root. Timing predict and simulation read the committed
profile.db and need no GPU.

```bash
uv run python -m launcher presets/deepseek_v41_flash_b200_vllm_tp4_ep4.yaml --dry-run --cache-report
uv run python -m launcher presets/deepseek_v41_flash_b200_vllm_tp4_ep4.yaml
```

Profiling a V4.1 row runs on a B200 node, one `--specs` call per job, into a
scratch DB merged afterwards with `python -m profiling merge-db`:

```bash
uv run python -m launcher kernel-profile run compressed_sparse_mla_rope_cast \
  --backend flashmla_mega --specs specs.json --gpu-name "NVIDIA B200" --db scratch.db --json
```

The backend resolves to `vllm_upstream_fork_env`, so the fork's `.venv` must
exist. Notes from the captures and profiling jobs:

- FlashInfer's all-reduce workspace and vLLM's ZMQ IPC create sockets under
  `TMPDIR`, limited to 107 bytes. Under a long path the all-reduce fusion falls
  back silently and changes the measured kernels; point `TMPDIR` and
  `VLLM_RPC_BASE_PATH` at a short directory.
- `--block-size 256` is rejected by the BLHNC KV layout at this upstream base.
- Without `max_cudagraph_capture_size: 2048`, vLLM caps graphs at 128 tokens
  and every mixed iteration runs eager.
- The MXFP8 GEMM and MoE backends need a warm autotune cache before a refill.
  The `flashmla_mega` prefill runner fails its reference check with NaN in
  about 2 of 571 specs; a retry passes.
