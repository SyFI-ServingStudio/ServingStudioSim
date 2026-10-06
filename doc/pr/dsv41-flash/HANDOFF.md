# DeepSeek-V4.1-Flash support: handoff (B200, vLLM TP4/EP4)

Handoff for branch `i-want-to-support-deepseek-v4-1-flash`, which adds
DeepSeek-V4.1-Flash (`deepseek-ai/DeepSeek-V4.1-Flash`, architecture
`DeepseekV41ForCausalLM`) to the simulator: L1 kernels, L2-L4 composition, a
worker pairing, a model.work label, and kernel alignment against a vLLM capture.
The simulation runs end to end. It is not an accepted baseline: the kernel
alignment is within 5% overall but still has explained, unfixed deviations, the
TPOT/throughput comparison has not been run on a clean pass, and nothing is
pushed. Section 8 lists what remains, in priority order.

**Provenance labels.** Every number carries one of these tags:

- *measured*: a real vLLM capture or a B200 profiling run;
- *simulated*: a timing prediction (`p_…`), simulation run (`r_…`) or sweep
  (`e_…`) registered in the Analyzer, with its citation token in backticks;
- *catalog*: the checkpoint config, HF README or fork source;
- *derived*: arithmetic on the above, with the inputs stated.

Paths are repo-relative. `tmp/` is gitignored and `logs/` is untracked, so the
files they name exist only in this worktree
(`/raid/kanzhu/ServingStudio/wt-i-want-to-support-deepseek-v4-1-flash`). The
tables this handoff depends on are copied below; the small files it depends on
are copied next to this document:

| Copy in `doc/pr/dsv41-flash/` | Original | Purpose |
|---|---|---|
| `profile_db/db_renames.py.txt` | `tmp/dsv41/merge/db_renames.py` | Kind/backend/value renames for a v2 profile.db copy (rename to `.py` to run; kept as `.txt` so repo lint skips it) |
| `profile_db/renames.md` | `tmp/dsv41/merge/renames.md` | Full rename table with the reason for each name |
| `profile_db/build_l2fix.py.txt` | `tmp/dsv41/merge2/db/build_l2fix.py` | Builds the DB copy with the 308 post-L2-fix elementwise rows |
| `profile_db/elementwise_conflicts_main_kept.csv` | elementwise rows of `tmp/dsv41/merge/conflicts_main_kept.csv` | The 308 conflicting elementwise rows: main's and the branch's time_ms |
| `public_preset_drafts/{public,public_sim}/DeepSeek-V4.1-Flash/*.yaml` | `tmp/dsv41/public/draft/` | Blocked public presets (Section 8a) |

## 1. Summary

- **Model.** A 40-layer MoE (hidden 5120, 384 routed experts, top-6, 1 shared
  expert) with mHC residual streams, two CPU-offloaded Engram n-gram layers,
  three compression ratios, cross-layer KV and index sharing, and
  candidate-block selection (*catalog*, `model/config/deepseek_v41_flash.json`).
  Text only; MTP and the vision tower are not modeled.
- **Deployment modeled.** vLLM V1 on 4x B200: TP4 attention, EP4 experts over
  the same ranks, chunked prefill at 2048 tokens, `--max-model-len 131072`, run
  on a local fork of vLLM rebased onto upstream 04730e8 (Section 7).
- **Kernels.** Three new L1 kinds (`compressed_sparse_mla_rope_cast`,
  `q_pad_kv_rope_mxfp8_insert`, `engram_lookup`) plus new or extended backends
  on six existing kinds. Every L1 cache passed its fidelity check at its stated bar
  (Section 3.2).
- **Kernel alignment (Check 1).** After one fix round, the simulated
  critical path is **−4.55%** against capture 2 over 2236 iterations: decode
  −4.65%, mixed −4.00% (*simulated vs measured*, alignment
  `al_6e04cf7b43e2774d90eea92fbfd65605fc2260f7e9f4a61bc3902194f81fd36c`).
  The recommended `gpu_time_multiplier` is 1.0361.
- **End to end.** On capture 2's 96-request workload the simulator gives a
  51.85 s makespan and 539.2 ms mean TTFT (*simulated*, run
  `r_be56f99c9ab1b6b7bbbe78b399d8f49f27c76d5c4670400dd2c911a8c14e853b`); the
  capture, with its 124.9 s nsys stall removed, took 54.0 s and 567.7 ms
  (*measured*). TPOT and throughput need a clean pass (Section 8e).
- **Beyond the capture.** A counterfactual flag `decoder_swa_bounded_replay`
  (no capture behind it) and an unpinned `max_model_len` defaulting to
  1048576, with rows profiled and validated out to 1M context.
- **Blocking.** Public presets wait on user approval to upload the corpus
  capture to Hugging Face (Section 8a); the fork is not pushed and its gitlink
  is not committed (Section 8b).

## 2. Architecture notes and how the simulator models each feature

Facts in this section come from the checkpoint config, the HF README in the
local snapshot (`/raid/hf/hub/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277/README.md`),
and the fork source (*catalog*). The simulator files are under
`simulator/src/{arch,worklet,op}/` (Section 3.3).

| Feature | What the model and vLLM do | How the simulator models it |
|---|---|---|
| **Engram** (layers 1 and 14) | Hashed n-gram embedding lookup: 24 hash columns per layer, head_dim 256, FP8 rows with UE8M0 block-32 scales. With `cpu_offload` (vLLM default, set explicitly in the capture) each rank holds its tables in pinned host memory and reads them over UVA: 96.0M rows x 264 B = 23.6 GiB per rank per layer. Both lookups launch right after the hash kernel on per-layer side streams and are consumed before the layer-1 / layer-14 Engram all-gather. | New kind `engram_lookup` with the exact `table_rows` in the config (the gather is TLB-bound, so table size matters). The two lookups are charged **serially after the layer-0 main path** (Section 6), not overlapped. |
| **mHC** (`hc_mult` 4) | The previous block's post-mix, the next block's pre-mix and its RMSNorm run as one DeepGEMM `mega_mhc` launch, 77 per iteration. Layer-0 pre, Engram pre/post and the final head use non-fused kernels. For T <= 16 in FULL decode graphs vLLM takes a non-fused path. | New backend `deepgemm_mega` of `mhc_fused_post_pre_rms_norm`, with grid points at the K-split switches and wave edges. The non-fused kernels are `elementwise` byte placeholders (each < 0.3% of time). The T <= 16 path is priced with the fused rows. |
| **Compression ratios** | `compress_ratios`: layers 0-1 ratio 0 (sliding window only), 2-19 ratio 2 (two tokens pooled into one latent with a softmax gate), 20-39 ratio 1 (per-token projection). Every layer also keeps a 128-token MXFP8 sliding window. | `compress_ratio` is a config axis of `compressed_sparse_mla_rope_cast`; keys per token are `min(pos+1, 128) + min((pos+1)//ratio, 512)` in closed form. The compressor GEMMs reuse `gemm_fp32_output`; the compressor's norm and NVFP4 store are placeholders. |
| **KV and index sharing** | Only the KV sources 2, 8, 14, 20 own a compressed cache (NVFP4, 288 B per state); the other layers own no compressed cache and read their source's. The index sources are 2, 8, 14, 20, 24, 28, 32, 36; the non-owning sources 24, 28, 32 and 36 share layer 20's FP8 index cache (132 B per key). | The arch maps each layer to its source when it builds the attention and indexer inputs. KV bytes count only the owning caches (below). |
| **Candidate selection** | Layer 20 scores 8-token blocks and keeps the top 2048 (`candidate_block_size` 8, `candidate_topk_blocks` 2048); layers 24/28/32/36 restrict their indexer to that pool. | An `elementwise` placeholder sized by the logits it streams (`attn.indexer.score.candidates`). It reads 0.07-0.10x of measured time (Section 5.1), so it is a promotion candidate. model.work treats it as zero necessary work and caps the consumers' index logits at 16,384 positions. |
| **Causal encoder-decoder** | The README describes layers 0-19 as a causal encoder and 20-39 as a decoder whose global KV is projected from the encoder output, so prefill needs about 8B active parameters per token and decode 16B. The fork does not exploit this: its layer loop runs all 40 layers on every token (`vllm/models/deepseek_v41/nvidia/model.py:832` in the fork), and the capture confirms it (the KV insert runs at full T in all 40 layers). | The simulator matches the fork and runs all layers. The opt-in `decoder_swa_bounded_replay` flag (default false) is a counterfactual: layers 21-39 see only each prefill chunk's last 128 tokens, as in open vLLM PR #58132 and SGLang's `--enable-decoder-swa-bounded-replay`. The model.work floor still counts full prefill work, which overstates the necessary work (Section 8f). |

### KV accounting

The arch charges **1056 B per token per rank, 4224 B per token over the four
ranks** (`total_kv_bytes_per_token`). The derivation follows vLLM's own grouping
(*catalog*, fork `get_kv_cache_spec` and `kv_cache_utils.py`; reconciled in
`tmp/dsv41/agent-trace/phase-sim-first.md`):

- vLLM forms 7 KV groups that draw from one pool, with every block 135,168 B:
  five sliding-window groups of 8 layers (8 x 32 x 528 B), one group holding the
  compressed and index pages of the four KV sources
  (3 x (18432 + 8704) + 36864 + 16896 B), and one compressor-state ring group.
  The server log line `kv cache group sizes [32, 32, 32, 32, 32, 128, 8]` matches.
- Per-token growth is the compressed+index group only: 135168 B / 128 tokens =
  1056 B (*derived*). The one KV head is replicated on all four ranks.
- The sliding window and compressor rings are per-request constants (about
  190 MB at 48 in flight against an 81 GB pool, *derived*), so the existing
  `FullAttnKv` store fits and no new KV store was added.
- **Reconciled against both captures** by running the fork's
  `get_kv_cache_configs` on CPU: capture 1 predicted 53,181,494 tokens vs
  53,178,702 logged; capture 2 logged 46,767,420, which inverts to 603,004
  blocks = 81.5068e9 B on the smallest rank (*measured* log, *derived* inversion).
  vLLM's logged "tokens" is concurrency at `max_model_len`, not per-token capacity.
- The preset's `attn_gpu_memory_gb: 81.506844672` therefore gives the
  simulator 77,184,512 tokens = 603,004 blocks x 128 (*derived*).

## 3. What was done, by layer

### 3.1 Captures

All three captures ran on host `cayenne`, 4x B200, from
`logs/20260924_0_dsv41_flash_capture/` (Slurm logs `slurm-{1178,1179,1185}.out`):

| Job | Pass | Workload | Use |
|---|---|---|---|
| 1178 | `profile_nsys` (capture 1) | `trace_quadrant_c48.csv` | Graphs capped at 128 tokens, so every mixed iteration ran eager (about 150 ms each, *measured*). KV reconciliation only. |
| 1179 | `profile_corpus` | `trace_diverse_100.csv` (100 requests) | Token corpus: 135,144 tokens of routed-expert ids, `profile_corpus/token_corpus/{manifest.json,routes.u16}`. Scope: accepted generated tokens only, no prompt routes. |
| 1185 | `profile_nsys_cg2048` (capture 2) | `trace_quadrant_c48.csv` (96 requests, saturated at 48) | The reference capture: `max_cudagraph_capture_size 2048`, 2236 iterations (2113 decode, 122 mixed, 1 prefill). |

Server flags common to the nsys passes: `--block-size 128`, `--max-model-len 131072`,
`--max-num-seqs 64`, `--kv-cache-dtype fp8`, chunk 2048, `--engram-config '{"cpu_offload": true}'`,
EP4. `phase1_evidence.md` in the capture directory holds the kernel inventory.
Capture 2's decode kernel time splits as follows (*measured*, device 0,
12.877 ms kernel sum per iteration): routed MoE 38.2%, dense MXFP8 GEMMs
12.2%, mega-attention 11.0%, mHC 8.3%, all-reduce 8.1%.

### 3.2 L1 kinds and backends (final post-merge names)

Each new kind or backend got a Python runner and a Rust cache, was filled on
B200, and was checked for cache fidelity. The fidelity column is the
final check against fresh `perf_api` measurements at off-grid points
(*measured* truth vs *simulated* cache). Traces: `tmp/dsv41/agent-trace/*.md`.

| Kind / backend | Status | What it times | Fidelity (cache/truth) |
|---|---|---|---|
| `compressed_sparse_mla_rope_cast` / `flashmla_mega` (+ `torch` reference) | new kind | One fused FlashMLA launch: Q RoPE, sparse MLA over the 128-token MXFP8 window plus top-512 compressed NVFP4 rows, inverse RoPE, FP8 cast; Q padded from 16 to 64 heads. Prefill adds the per-chunk gather and index-combine launches. Cache coordinates: query rows, keys per token, excess gather area (an exact replica of the fork's `get_prefill_chunk_plan`). | After the L2 fix: decode 137/139 within ±15%, median 0.995. Long context: 20/20 within ±15%, median 0.996, worst 1.051. |
| `q_pad_kv_rope_mxfp8_insert` / `vllm_cuda` | new kind | The fused Q pad (interleaved layout, no Q norm or RoPE) plus MXFP8 sliding-window KV insert; switches to `ReducedGrid` at T >= 1024. | 41/41 within ±15% per config, median 0.999. |
| `engram_lookup` / `vllm_triton` | new kind | One Engram table gather, `host_uva` or `device` residency, exact `table_rows`. | `host_uva` 39/40 within ±15%, median 1.034. |
| `single_gemm` / `flashinfer_mxfp8` | new backend; new `DType` `mxfp8_e4m3` | vLLM's MXFP8 linear: CuTe-DSL quantize plus `mm_mxfp8`, one slot. Grid aligned to FlashInfer tuning buckets. | After the L2 fix: 390/420 within ±15%; all decode-size probes within; extrapolation above m = 8192 is poor and unused by V4.1. |
| `nvfp4_fused_moe` / `flashinfer_trtllm_sm100_mxfp4` | new backend; `weight_format` `mxfp4_e2m1` (new `DType`) | MXFP8 activations x MXFP4 weights, routing + FC1 (clamped SwiGLU) + FC2 + finalize in one FlashInfer call, `routing_method precomputed_dsv4`, priced on the critical EP rank of the corpus-folded histogram. | 22/24 within ±15%, median 1.003; misses at T = 2100/2112 on a tactic step. Realistic histograms 8/8 against the slowest rank. |
| `mhc_fused_post_pre_rms_norm` / `deepgemm_mega` | new backend | DeepGEMM `mega_mhc`, hidden 5120, K-split 40/27/20/16 by T. | 138/138 within ±15%, median 1.001. |
| `all_reduce_fusion` / `flashinfer_mnnvl` | backend ported from the GLM-5.3 branch (7bcb2dd); now main's | Plain FlashInfer MNNVL all-reduce on [T, 5120] bf16 over TP4; one-shot/two-shot switch at T = 25/26, cap 3276 tokens. | 32/33 within ±15%, median 1.003. |
| `batched_gemm` / `deepgemm_mxfp8_einsum_grouped_o_proj` | new backend | `wo_a` as DeepGEMM's grouped MXFP8 einsum (2 groups per rank, n 1024, k 4096). | m <= 32768: 99.5% within ±15%, median 0.995. Above 32768 the tile heuristic flips and the cache over-predicts by 1.22-1.24. |
| `gemm_fp32_output` / `torch_cublas` | existing backend extended to B200 and k = 5120, n 384/512/1024 | Router gate (5120 -> 384, FP32 out) and the compressor GEMMs. | 92.6-93.0% within ±15%; misses are deterministic cuBLAS tactic picks at m 701-22435. Router m <= 16 uses a cute-DSL kernel no backend profiles. |

Existing kinds and backends V4.1 also reads: `dsa_paged_mqa_logits_decode` /
`deepgemm_fp8` and `dsa_mqa_logits_prefill` / `deepgemm_fp8` (indexer logits,
32 heads, existing rows; the ratio-1 index layers price on 64-key pages because
the runner supports no 128-key page), `all_reduce` / `nccl` (the all-gather
proxy, existing rows), `single_gemm` / `torch_linear` (`lm_head`, 68 rows
added), and `elementwise` / `triton` for every placeholder (1,449 rows added,
the merge-2 count).

**Rename table (merge 1, commit 72918d7).** Main names kinds and backends by
mechanism and folds `*_vllm_fork` backends, so the V4.1 names changed. Arch,
op and worklet names keep the model name, as main does.

| Old name | New name |
|---|---|
| kind `deepseek_v41_mega_attn` | `compressed_sparse_mla_rope_cast` |
| kind `deepseek_v41_qnorm_rope_kv_insert` | `q_pad_kv_rope_mxfp8_insert` |
| `batched_gemm` backend `deepgemm_mxfp8_einsum_dsv41_wo_a` | `deepgemm_mxfp8_einsum_grouped_o_proj` |
| `gemm_fp32_output` backend `torch_cublas_vllm_fork` | `torch_cublas` (fold; the fork rows replaced the colliding k = 5120 keys) |
| `nvfp4_fused_moe` `weight_format` value `mxfp4_ue8m0` | `mxfp4_e2m1` |
| profiling env `vllm_fork_env` | `vllm_upstream_fork_env` |

`profile_db/renames.md` has the full table, including main's V4 table renames
that `db_renames.py.txt` also applies to an old DB.

### 3.3 L2, L3 and L4

- **L2 ops** (`simulator/src/op/attention/`): `deepseek_v41_mega_attn.rs`
  (decode/prefill leaves, with the global top-k remap folded in) and
  `deepseek_v41_indexer.rs` (prefill K gather, prefill logits, prefill top-k,
  decode logits, candidates, decode top-k; prefill logits are split on the query
  axis past vLLM's 512 MiB logits budget).
- **L3 worklets** (`simulator/src/worklet/deepseek_v41_*.rs`): attention TP,
  Engram TP, Engram prefetch (local), MoE FFN EP, prologue TP, head TP, and
  `deepseek_v41_common.rs` (stream gates, the all-gather proxy, placeholder
  helpers). `deepseek_v41_iteration_tests.rs` checks whole-iteration launch
  counts against capture 2 (77 `mega_mhc`, 81 all-reduces, 40 KV inserts, 2
  lookups, ...).
- **L4 arch** (`simulator/src/arch/deepseek_v41_vllm.rs`): selectors
  `deepseek_v41_vllm` and `deepseek_v41_vllm_serial_streams` (gated side
  streams serialized). The layers fold into 10 bodies with `Scale` multiplicity.
  Cache-keyed params: `routing`, `routing_seed`, `expert_popularity_file`,
  `token_corpus_file`, `max_model_len` (default 1048576, refused above it).
  `decoder_swa_bounded_replay` (default false) is listed by `list-params` but
  does not enter the cache key.
- **Deployment** (`simulator/src/deployment/unified.rs`): paired with
  `chunked_prefill` (FIFO admission, 2048-token budget, mixed batches) and
  `hp_unified` (ff8cc7b).

### 3.4 model.work label

- Accountant `model/work/models/deepseek_v41.py` with the attention mechanism in
  `model/work/attention/deepseek_v41.py` (4efe135); location map
  `model/work/location_maps/deepseek_v41_vllm_tp4_ep4_unified.json`, mapping id
  `deepseek-v41-vllm-tp4-ep4-unified-v1` (9bba9c5).
- Logical parameters: **748,494,669,424**, pinned in `tests/test_model_work.py`.
  This equals the HF-reported 763,205,315,794 elements minus MTP (14,225,362,530),
  the vision tower and aligner (485,268,480) and the router `bias_vl` (15,360),
  and it matches the checkpoint tensor headers exactly (*catalog*, *derived*).
- Verified in prediction `p_26bbf6b3bfbf42adb308557d6255d3ed` (`pred.overview`;
  `logs/20260925_3_dsv41_model_work`): composed necessary work available, no
  caveats. The location map covers 292 non-communication locations; 99 are
  intentionally empty (quant, top-k, copies, `.serial` duplicates and similar).
- The `deepseek_v41_vllm_serial_streams` selector has no location map.

### 3.5 Label rules and presets

- **Label rules**: `presets/alignment/dsv41_flash_b200_tp4_ep4/label_rules/{rules.json,manifest.json}`,
  112 order-free rules (7a2eaaa). They need the new `name_exact` key because
  vLLM's FusedQKRMSNorm and combine-top-k launches are literally named `kernel`.
- **Simulation preset**: `presets/deepseek_v41_flash_b200_vllm_tp4_ep4.yaml`
  (capture-2 workload, corpus routing, `max_model_len: 131072` pinned,
  `gpu_time_multiplier: 1.0361`, `attn_gpu_memory_gb: 81.506844672`).
- **Timing-predict presets**: `presets/predict_deepseek_v41_vllm{,_serial}.json`
  with `predict_deepseek_v41_vllm_cases.json` (4 capture-shaped cases), and
  `presets/predict_deepseek_v41_vllm_longctx{,_cases}.json` (14 cases out to 1M,
  unpinned).
- The corpus preset paths point into `logs/20260924_0_dsv41_flash_capture/`,
  which is untracked; on another machine they need the corpus capture
  (Section 8a).

## 4. Changes that affect more than V4.1

- **CUPTI L2-flush fix (83c9816).** `default_l2_flush_bytes` read
  `props.l2_cache_size`, but torch spells it `L2_cache_size`, so the lookup
  always returned 0 and every `Timer.cupti` "cold-L2" launch flushed only the
  64 MiB floor. On B200 the flush is now 2 x L2 = 265,289,728 B. **origin/main
  still has the bug** (checked at 86fb0eb). Consequences:
  - Every `Timer.cupti` row on a GPU with more than 32 MiB of L2 (H100, H200,
    B200) profiled before this fix carries warm-L2 bias. Only the V4.1 B200
    rows were re-measured; the others are stale, including
    `dsa_paged_mqa_logits_decode` (630 rows) and `dsa_mqa_logits_prefill`
    (336 rows), which V4.1 reads.
  - The fix moved V4.1's simulated decode iteration by +0.61 ms (decode
    `mega_attn` 0.75x -> 0.95x of measured, Section 5.1).
  - **308 `elementwise` / `triton` rows** conflict between main and this
    branch: the branch re-measured them after the fix. Main's values were kept
    in the working DB, which makes V4.1 cases 0.04-0.08 ms faster than with the
    branch's values (Section 5.4). The rows are listed in
    `profile_db/elementwise_conflicts_main_kept.csv`; branch/main time_ms has
    median +3.3% and range −31% to +104% (*measured*).
- **`vllm_upstream_fork_env`** (42c377f, `profiling/exec/env.py`). A host
  profiling env on `alignment/profiler/vllm/.venv` (or `$VIBESIM_VLLM_FORK_ROOT`),
  because main's `vllm_env` container is built from fork 3f667d7, which has no
  DeepSeek-V4.1. Seven backends use it: `flashmla_mega`, `q_pad` `vllm_cuda`,
  `engram_lookup` `vllm_triton`, the grouped o-proj einsum, `flashinfer_mxfp8`,
  `flashinfer_trtllm_sm100_mxfp4` and `deepgemm_mega`. Remove it once the
  container is rebuilt from the rebased fork. `flashinfer_mnnvl` and
  `gemm_fp32_output` `torch_cublas` stay on main's `vllm_env`, but their V4.1
  rows were measured in the fork venv (FlashInfer 0.7.0 vs 0.6.18 in the
  container, and the fork's cuBLAS), so a `--force` refill would measure a
  different stack.
- **New dtypes** `mxfp8_e4m3` (1 B data) and `mxfp4_e2m1` (0.5 B) in
  `profiling/db/args.py` and `simulator/src/timing/bridge/payload.rs`, plus the
  MXFP8/NVFP4 record-size constants tied between Rust and Python by a test.
- **`byte_rate_placeholder_shape`** (`simulator/src/op/attention/deepseek_v41_indexer.rs`).
  The `elementwise` runner is a fan-in reduce, so a placeholder with a tiny
  output measures loop latency instead of bandwidth (about 180 µs per decode
  top-k in the first prediction). Above an 8:1 input:output ratio the same total
  bytes are split evenly between read and write. Only V4.1 calls it today.
- **`name_exact` label-rule key** (`alignment/labeling/rules.py`): matches the
  whole kernel name instead of a substring.
- **model.work extensions** (`model/work/core.py`, `model/work/quantization.py`):
  `MatmulGroup` storage/compute dtypes and rows, learned-weight dtype bytes and
  read/activated elements, optional `mhc` and `engram` buckets, `scale_fmt: ue8m0`
  (1-byte scales) and `expert_dtype: fp4`. main's GLM-5.3 labels mHC FLOPs as
  `residual_mix`; V4.1 keeps `mhc`, and unifying them would change V4.1's goldens.

## 5. Fidelity and results

### 5.1 Check 1: kernel alignment against capture 2

Two rounds ran against the same capture and the same labels:

| Round | Prediction | Alignment | All | Decode | Mixed |
|---|---|---|---|---|---|
| Initial | `p_06eace1e314f4a1c875c09e435d2a9d4` | `al_008ecc9c5b0646bdf5c393975cf1783ddabe1f4fb32025ff14f0ed4bb8bb5cdb` | −9.69% | −10.43% | −5.89% |
| After fix 1 | `p_bb36666131324917955cf7ffa1b6802c` | `al_6e04cf7b43e2774d90eea92fbfd65605fc2260f7e9f4a61bc3902194f81fd36c` | **−4.55%** | **−4.65%** | **−4.00%** |

These are duration-weighted signed errors of the simulated critical path
against measured (n = 2236, 2113 decode, 123 mixed; measured 28,034.1 ms,
simulated 26,759.9 ms after fix 1). The Analyzer returns no citation tokens for
alignments. In the initial round, mapping covered 0.982 of measured
critical-path time; the unmapped remainder (position cast, sampling, metadata,
`_post_update`, indexer workspace fill) is declared. Fix 1 reused the same
labels. `recommended_gpu_time_multiplier` = 1.036105864835686, with iteration
159 (an unfused eager one-off) excluded. Run bundle for fix 1:
`logs/20260925_2_dsv41_flash_fix1/`.

Fix 1 made two changes: the L2-flush refill (83c9816) and the serial Engram
rule (68dc853). The per-bucket ratios below (sim/measured, ms per iteration)
use the per-op devtable method; they are the "0.953/0.956/0.954/0.963" figures
quoted elsewhere, one per token bucket, and agree with the Analyzer's stage
totals above (decode 0.953 vs −4.65%). The table omits the (0,128] bucket, a
single eager one-off (iteration 158), which the Analyzer's 123 mixed iterations
include:

| Bucket (n) | Measured | Sim before | Sim after | Ratio after | From Engram rule | From DB refill |
|---|---|---|---|---|---|---|
| decode (2113) | 11.117 | 9.958 | 10.600 | 0.953 | +0.033 | +0.610 |
| mixed (128,512] (30) | 16.084 | 14.553 | 15.384 | 0.956 | +0.076 | +0.755 |
| mixed (1024,1536] (12) | 34.118 | 31.847 | 32.535 | 0.954 | +0.426 | +0.262 |
| mixed (1536,2048] (80) | 45.413 | 43.062 | 43.712 | 0.963 | +0.719 | −0.068 |

Read the alignment by **group, not by overlapped per-op rows**. The analyzer
gives a hidden `Parallel` branch zero simulated time and gives the measured time to
whichever launch started first, so rows such as `ffn.shared.*` (0 simulated at
T <= 256) are attribution artifacts, not missing work. Group deviations after
fix 1 (sim − measured, ms per iteration; winner-device spans; source
`tmp/dsv41/agent-trace/phase5-fix1-check.md`):

| Bucket | Attention (busy + AR net) | FFN group | Engram window |
|---|---|---|---|
| decode | −0.238 | −0.205 | −0.047 |
| (128,512] | −0.592 | +0.073 | −0.169 |
| (1024,1536] | −2.147 | +0.398 | −0.131 |
| (1536,2048] | −2.663 | +0.701 | −0.145 |

**What remains, with diagnoses:**

1. **Routed MoE.** Mixed `fused_moe` reads +6-7% (+0.49 to +0.94 ms/iter) and
   drives the positive FFN residual. It does not depend on the tactic cache.
   The corpus holds only generated-token routes, while T ≈ 2048 iterations are
   prompt chunks. Replaying corpus histograms through the runner at T = 2048,
   a contiguous corpus window already lowers the critical-rank MoE time from
   17.5 to 16.65 ms/iter, against 14.98 measured. In decode the −0.20 ms/iter comes from
   histogram compression in the fold (Section 6).
2. **Elementwise placeholders that should be L1 kinds**:
   `attn.indexer.candidates` 0.07-0.10x, `attn.indexer.prefill_topk` 0.16-0.18x,
   `ffn.routed.topk` 23x at decode but 0.32x at large T, `attn.qk_rmsnorm` 0.58x
   at 2048, `ffn.routed.input_quant` 1.6-2.2x at small T. Together about −0.7 ms
   at T ≈ 2048.
3. **Isolated vs in-server calibration.** At large T `mega_attn.prefill` reads
   0.91-0.92x, the KV insert 0.81-0.86x and `mega_mhc` 0.92-0.94x; this is most
   of the attention gap. About 7 µs per KV-insert launch at T = 2048 is
   unexplained. In the other direction, small GEMMs read high after the refill
   (`router_gate` 1.5-1.8x, `fused_wqa_wkv` 1.16-1.25x, `wq_b` 1.2x): the
   253 MiB cold flush over-prices weights that stay L2-warm in back-to-back graph
   replay. This needs a per-kind decision, not a global change.
4. **Engram.** `engram.lookup` reads 0.73-0.74x at large T because the
   contending in-server pair is longer than the isolated pair; the window
   residual is only −0.13 to −0.15 ms.
5. The router gate's T <= 16 tier (cute-DSL `ll_bf16`) has no backend; cuBLAS
   rows stand in.

### 5.2 First simulation vs capture 2

Run `r_be56f99c9ab1b6b7bbbe78b399d8f49f27c76d5c4670400dd2c911a8c14e853b`,
sweep `e_c22ea986ccda4c6d8290a6b0ac170fbd`, `logs/20260925_4_dsv41_first_sim/`,
preset as in Section 3.5, `chunked_prefill` worker:

| Metric | Simulated | Measured, capture 2 |
|---|---|---|
| Requests finished | 96 `exp.requests_finished` | 96 |
| Makespan | 51.85 s (span 51,851.8 ms, `run.cluster.throughput`) | 178.9 s raw; **54.0 s** with the 124.9 s nsys stall removed |
| Total throughput | 6872.5 tok/s `exp.throughput` | 1992.0 raw; 6598.9 stall-removed |
| TTFT mean / p50 / p90 | 539.2 / 147.0 / 1587.0 ms `run.cluster.slo-general` | 567.7 / 152.1 / 1700.7 ms (engine core) |
| TPOT mean / p50 / p90 | 14.09 / 11.89 / 20.28 ms `run.cluster.slo-general` | not comparable (48 requests straddle the stall) |
| GPU utilization | 96.3% `exp.utilization` | n/a |

The stall is the one 124.9 s gap after iteration 2392 (`cudaProfilerStop` and
the nsys flush), so capture 2's raw TPOT and throughput measure the profiler.
The stall-removed makespan is 4% longer than simulated, and TTFT agrees within
about 5% (*derived*). A fair TPOT/throughput check needs a workload pass without
nsys (Section 8e). Not modeled: `max_num_seqs 64` (48 < 64 here) and prefix
caching (a length-only trace cannot hit).

### 5.3 Decoder SWA bounded replay A/B (counterfactual)

The flag (2ebb6f5, 8dfb187) has no capture behind it. Layers 21-39 (after the
last KV source, layer 20) see `(min(append, 128), context)` per prefill chunk;
decode rows are unchanged. The approximation truncates to 128 rows at every late
layer, as SGLang does; the exact cone grows by 128 rows per layer.

Timing predictions (*simulated*, ms, `pred.overview`;
`logs/20260925_5_dsv41_swa_replay_predict/`):

| Case | Off `p_35762ab86ab94be0963bd081db074555` | On `p_d36b2cdf6c18452794de86ee07d7706e` |
|---|---|---|
| 0: decode, 48 requests | 10.6022 | 10.6022 |
| 1: mixed 1912 + 91 prefill + 45 decode | 43.5739 | 32.3216 |
| 2: 128-token chunk + 46 decode | 15.4371 | 15.4371 |
| 3: cold 2048-token chunk | 42.0314 | 28.8665 |

Case 2 does not change because a 128-token chunk is already within the window.
Simulation sweep `e_e1fa041324634f438b48c1f0d367bd8f`
(`logs/20260925_6_dsv41_swa_replay_ab/`), capture-2 workload:

| Metric | Off (`r_3ca946f5…`) | On (`r_35285a2f…`) |
|---|---|---|
| Total throughput | 6872.5 tok/s `exp.decoder_swa_bounded_replayfalse.throughput` | 7019.9 tok/s `exp.decoder_swa_bounded_replaytrue.throughput` |
| TTFT mean | 539.2 ms `exp.decoder_swa_bounded_replayfalse.ttft.mean` | 398.8 ms `exp.decoder_swa_bounded_replaytrue.ttft.mean` |
| TPOT mean | 14.09 ms `exp.decoder_swa_bounded_replayfalse.tpot.mean` | 12.99 ms `exp.decoder_swa_bounded_replaytrue.tpot.mean` |

Full run ids: off `r_3ca946f5459b206b7c3a1b3ae392cd3852e9ee364cfc128e9a8a350875665231`,
on `r_35285a2f57c24e40b3761418a1cb52692763545bfa54a9042f46848e3e11f9ae`. The
off run reproduces the first simulation.

### 5.4 Parity across the merges and the max_model_len change

Each step re-ran the same four cases with byte-identical configs (*simulated*,
ms, `pred.overview`). Off and on refer to the bounded-replay flag:

| Step | Off | On | Case 0 / 1 / 2 / 3 (off) | Case 1 / 3 (on) |
|---|---|---|---|---|
| Pre-merge (L2-fix elementwise rows) | `p_35762ab8…` | `p_d36b2cdf…` | 10.6022 / 43.5739 / 15.4371 / 42.0314 | 32.3216 / 28.8665 |
| Merge 1 (d8fc2f3) | `p_65e273e9f6da4002996c4e5a34ecd0d9` | `p_23fac0eb706c49998a4365b58b03a8e9` | 10.5645 / 43.5387 / 15.3594 / 41.9940 | 32.2430 / 28.8100 |
| Merge 2 (c669848b) | `p_0fb865ab79754b109bb24dfa8ea35589` | `p_c00a922d085a4d10958e2c70e858b941` | identical to merge 1 | identical |
| max_model_len param (ed4914b0) | `p_6461dd3d34e2433f8dde51c0757d2708` | `p_b7c9596da13b4a4bb12fef2b2a80015b` | identical to merge 1 | identical |

The only change in the whole sequence is the merge-1 step of −0.035 to −0.079
ms per case, and the per-slot diff traces all of it to the 308 elementwise rows
(Section 4): with the branch's rows swapped into a DB copy
(`tmp/dsv41/merge2/profile_l2fix_copy.db`, built by
`profile_db/build_l2fix.py.txt`) the current code reproduces the pre-merge
values bit for bit. Merge 2 and the `max_model_len` change are bit-identical
in every slot.

### 5.5 Long-context prediction (max_model_len 1048576)

The 1330 missing `compressed_sparse_mla_rope_cast` rows for
`max_model_len 1048576` were profiled on B200 (Slurm 3932 decode, 632 rows;
3933 prefill, 698 rows). Against the existing 131072 rows of the same shape they
read median 1.007; the outliers are 1-4-row decode batches about 3-4 µs slower,
which looks like session noise because the fork's decode never reads
`max_model_len` (*measured*). Cache fidelity at 200K-1M context: 20/20 within
±15% (Slurm 3934, Section 3.2).

Prediction `p_d1e9d4d712244e74a298640d291b9740` (`pred.overview`,
`logs/20261005_5_dsv41_longctx_predict/`), selected cases (*simulated*, ms):

| Case | Total |
|---|---|
| decode 1 x 1,048,575 context | 6.3488 |
| decode 8 x 1,048,575 | 8.5104 |
| 2048-token chunk at prefix 131,072 | 57.1302 |
| 2048-token chunk at prefix 524,288 | 122.3461 |
| 2048-token chunk at prefix 1,046,528 | 288.0400 |
| last chunk of a 500K-token request (prefix 497,952) | 111.8225 |

Attention and indexer-logits leaves are never flagged off-grid at any context.
The flagged leaves are the three indexer placeholders held at edge bandwidth in
prefill (Section 6) and decode `elementwise` leaves below the 32-token grid
start, which are unrelated to context.

## 6. Modeling choices and approximations

- **Engram lookups are a serial Sum.** In capture 2 nothing waits on a lookup
  (at least 0.24 ms of slack before the layer-1 all-gather), but the lookups take
  up to half the SMs for 0.6-0.8 ms and slow the main-path kernels beside them;
  the hash-to-all-gather window grows about 0.70 ms per ms of lookup at
  T ≈ 2048, so the pair is about 70% serial (*measured*). `Sum[main, lookups]`
  stands in for same-device contention; `Parallel{overlap < 1}` would need an
  overlap no capture calibrates. Whether the slowdown is SM
  occupancy or memory/TLB pressure was inferred, not isolated.
- **Parallel/Sum stream thresholds.** Shared expert ‖ routed expert overlaps under
  `Parallel` for T <= 256; stage A (`fused_wqa_wkv` ‖ compressor ‖ indexer
  `weights_proj`) for T <= 1024; `Sum` above. Both match vLLM's source
  thresholds; capture 2 has no mixed batch between T = 175 and 1116, so it is
  consistent with them but cannot tighten them. The compressor aux stream is
  live only in FULL-graph decode.
- **AllGather is proxied by the all-reduce curve.** There is no all-gather
  kind. The two Engram gathers and the logits gather are priced as `all_reduce` /
  `nccl` at half the gathered bytes; this over-prices small gathers (about 28 µs
  vs 15 µs measured for the 48-token Engram gather) and under-prices the logits
  gather (41.6 vs 50.7 µs).
- **Placeholders.** Ops without an L1 kind are `elementwise` byte-rate
  placeholders anchored to their fork source lines (table in
  `tmp/dsv41/agent-trace/phase2-l2l3.md`): mHC pre/post/head, QK RMSNorm,
  compressor norm and NVFP4 insert, indexer `wk`/`weights_proj`/K store/Q RoPE,
  all top-k and candidate kernels, Engram gather/post-wkv, routed top-k and input
  quant, shared `act_and_mul`, residual add, embedding, hash, final norm.
  Attention metadata launches are folded into overhead (not modeled).
- **MoE histogram fold.** The simulator draws real corpus batches, builds
  per-layer histograms, sorts ranks within each layer, averages the sorted
  histograms over layers and prices one critical-rank histogram. The price of
  the average (122.0 µs/layer) is below the average of per-layer busiest-rank
  prices (126.9 µs/layer), which costs about −0.20 ms per decode iteration
  (*measured* via the runner on real per-layer histograms). The corpus also lacks
  prompt routes, which leaves mixed iterations +6-7% (Section 5.1).
- **CUDA-graph padding.** Dense rows pad to vLLM's capture sizes: 1, 2, 4, 8,
  then every 8 to 256 and every 16 to 2048 (`cudagraph_rows`), matching
  `max_cudagraph_capture_size 2048`.
- **Prefill runs all 40 layers**, as the fork does; the encoder-decoder saving is
  only in the counterfactual flag (Section 2).
- **148 SMs.** The `compressed_sparse_mla_rope_cast` decode wave width is
  `B200_SMS * 64 / num_heads` with `B200_SMS = 148`. Since c03a8b49 the arch
  builds on other GPUs (B300, GB200), whose rows would still be read through the
  B200 wave model. The 128-head wave width (74) is not measured.
- **Indexer placeholders held at edge bandwidth past the grid.** The
  `elementwise` grid ends at 65,536 tokens. Long prefill contexts size
  `prefill_k_gather`, `prefill_topk` and `candidates` past it (2.1M tiles at
  1M context, ratio 1). The V4.1 indexer op holds the edge's bandwidth instead of
  extrapolating the last segment, which implied 18 TB/s before the fix. This is
  unvalidated at 512K and 1M (Section 8h). The shared `elementwise` kind was not
  changed, because that would move every arch.
- **Fixed step budget.** `max_num_batched_tokens` is 2048 in the
  `compressed_sparse_mla_rope_cast` config identity; main's 310390b7 (budget
  from the worker) was not mirrored (Section 8g).

## 7. Branch, merge and repository state

- **Branch.** `i-want-to-support-deepseek-v4-1-flash` at 51bd8476 before this
  document's commit; 85 commits ahead of origin/main by `git rev-list`, 54 on
  the first-parent line since the original base 8900dec (18 merges of L1 agent
  branches and 2 merges of origin/main among them). Nothing is pushed. origin/main
  has moved 5 commits past the last merge base, 865c11b9.
- **Merges.** d8fc2f3 merged origin/main 9161180 (25 conflicted files); c669848b
  merged 865c11b9 (10 files). Both took main's structure: renames by mechanism,
  no `#[supported]`, `OffGrid`, capability rules on `BackendSupport`. After merge
  2: `cargo test -p simulator --lib` 1214 passed (1219 after the max_model_len
  work); pytest 3938 passed, 0 failed, the same statuses as a clean origin/main
  worktree plus 72 new tests (*measured* test runs, `tmp/dsv41/merge2/phaseB.md`,
  `tmp/dsv41/maxlen/phaseB.md`).
- **profile.db.** The working copy carries the V4.1 rows and is marked
  skip-worktree (`git ls-files -v` shows `S profiling/profile.db`). It is
  **not committed**; HEAD's blob is main's.
  - Merge 2 inserted 4,139 branch rows into main's DB (0 conflicts); the
    max_model_len work added 1,330 rows (Slurm 3932/3933; fidelity 3934 wrote
    none).
  - Backups, in order: `tmp/dsv41/merge/profile_v41_backup.db` (pre-merge-1,
    schema v2), `tmp/dsv41/merge2/profile_pre_merge2.db`,
    `tmp/dsv41/maxlen/profile_before.db` and
    `tmp/dsv41/maxlen/phaseB/profile_pre_phaseB.db` (identical to the previous).
  - To rebuild from the v2 backup: copy it; run `db_renames.py` on the copy
    **before** `kernel-profile migrate-db` (a value rename on v3 would need
    `args_hash` recomputed, and the script refuses it); migrate; delete rows
    whose semantic key main already has (main wins) and main's deleted kinds;
    then `kernel-profile merge-db <main.db> <delta.db>`. The merge-2 row
    classification is `tmp/dsv41/merge2/db/analyze.py`; both phase reports
    (`tmp/dsv41/merge{,2}/phaseB.md`) give the exact commands and counts.
- **Fork submodule** `alignment/profiler/vllm`: local branch
  `servingstudio-alignment-v41` at 892da0822f, 16 commits on upstream 04730e8
  (instrumentation plus three docs commits), on no remote branch. The parent
  gitlink is still 3f667d7, so `git status` shows ` M alignment/profiler/vllm`.
  Its `.venv` is the profiling env above.
- **Untracked by design:** `6d67a4ff1fe9_plan.md`, `6d67a4ff1fe9_progress.md`
  (orchestrator notes), `logs/`, `tmp/`.

## 8. What is left and open decisions, in priority order

- **(a) Upload the corpus capture to Hugging Face (needs user approval).** The
  public presets must name a pinned `hf://datasets/UW-SyFI/servingstudio-workload@<sha>/…`
  capture (`tests/test_public_presets.py`). Proposed path
  `deepseek_v41_flash/vllm/diverse_100/capture/20260924/` with `manifest.json`,
  `routes.u16` and `trace.csv` (= `logs/20260924_0_dsv41_flash_capture/trace_diverse_100.csv`).
  A uniform-routing-only preset fails `test_every_member_is_measured`: each
  member misses 67 `nvfp4_fused_moe` rows (*measured*, re-checked on 2026-10-05
  with `tmp/dsv41/public/check_members.py`). The drafts are in
  `public_preset_drafts/` (arch: `max_model_len` [131072, 1048576] x replay
  [false, true]; sim: `chunked_prefill` 2048, replicas [1, 2], replay
  [false, true], pinned at 131072); add the pinned corpus row once uploaded.
- **(b) Push the fork, commit the gitlink, open the PR.** Push
  `servingstudio-alignment-v41` (892da0822f), commit the parent gitlink, and
  open the PR from this branch. origin/main is 5 commits past the last merge, so
  a third merge will likely come first.
- **(c) profile.db.** Re-measure the stale `Timer.cupti` rows after the L2 fix
  (at least the 966 indexer-logits rows V4.1 reads; then other models and GPUs),
  and decide the 308 elementwise rows: keep main's, or adopt the branch's
  post-fix values, which other models also read and so needs sign-off. Then
  decide how the DB lands in the PR.
- **(d) Fidelity round 2.** Capture prompt routes into the token corpus (vLLM
  `routed_experts` for prefill); replace the MoE fold with an order-statistic
  E[per-batch max rank]; promote placeholders to L1 kinds (indexer candidates,
  `prefill_topk`, routed top-k, `qk_rmsnorm`); add a router-gate `ll_bf16`
  backend for T <= 16; check cold vs warm L2 per kind for the small GEMMs; add an
  Engram same-device contention node.
- **(e) Check 3.** Run a `workload_metrics` pass (same config, no nsys) and
  compare TPOT and throughput.
- **(f) model.work prefill floor.** Count encoder-only prefill plus the decoder's
  last window, so the floor reflects the causal encoder-decoder; also carry
  per-request geometry so the mixed-batch floor is not built from a mean request
  (+21% attention in case 1).
- **(g) Step budget from the worker (main 310390b7).** Needs
  `compressed_sparse_mla_rope_cast` and indexer prefill rows at other
  `max_num_batched_tokens`; today the budget is fixed at 2048.
- **(h) 1M deployment.** Set `attn_gpu_memory_gb` for an unpinned 1M server
  (the FlashMLA prefill workspace grows from 0.61 GB to 4.84 GB at ratio 1,
  *catalog* fork source with *derived* sizes; the indexer
  workspace may grow too, `TODO(verify)`); re-measure the noisy 1-4-row decode
  rows in one job; validate the held placeholders with fresh measurements at
  524K and 1M tokens.
- **(i) SGLang capture with bounded replay on and off** to validate the
  counterfactual. The SGLang submodule is not initialized and there is no V4.1
  SGLang capture.
- **(j) Preset dry-run test for V4.1.** No test dry-runs
  `presets/deepseek_v41_flash_b200_vllm_tp4_ep4.yaml`; a retired key went
  unnoticed until merge 2 (fixed in 3fa8c983).
- **(k) Vision tower and MTP are not modeled** (3 MTP layers, 32-layer vision
  encoder).
- **(l) `routing_method precomputed_dsv4` still names a model.** It is a DB args
  value, so renaming it needs a value migration like `mxfp4_ue8m0` in merge 1.

Smaller open questions carried from the merge reports: relax the V4.1 runners'
production-only shape checks to the kernels' real limits (main's 4fd5bed9
convention); keep or drop the V4.1 `model/arch_catalog.yaml` entries while no
public preset uses them; let the DB-backed Rust tests set `sys.path`
themselves instead of needing `PYTHONPATH`.

## 9. How to reproduce

From the repository root, with `unset VIRTUAL_ENV`. Timing predict and
simulation read the working profile.db and need no GPU when every row exists;
with `--no-gpu` a missing row is an error instead of a GPU JIT.

```bash
# Timing predict: the four capture-shaped cases, and the long-context set
uv run python -m launcher timing-predict presets/predict_deepseek_v41_vllm.json --no-gpu
uv run python -m launcher timing-predict presets/predict_deepseek_v41_vllm_longctx.json --dry-run --no-gpu

# Simulation of capture 2's workload (dry run first, then the run)
uv run python -m launcher presets/deepseek_v41_flash_b200_vllm_tp4_ep4.yaml --dry-run --cache-report
uv run python -m launcher presets/deepseek_v41_flash_b200_vllm_tp4_ep4.yaml

# Check 1 against capture 2 (configs in the untracked capture directory)
uv run python -m launcher alignment timing-predict logs/20260924_0_dsv41_flash_capture/timing_predict_fix1.yaml
uv run python -m launcher alignment analyze logs/20260925_2_dsv41_flash_fix1/analyze_kernel.yaml

# DB-backed Rust tests (need the working profile.db)
PYTHONPATH=$PWD uv run cargo test --release -p simulator --lib bounded_replay -- --include-ignored
PYTHONPATH=$PWD uv run cargo test --release -p simulator --lib long_context -- --include-ignored
```

The Check-1 labels are reused unchanged
(`logs/20260924_0_dsv41_flash_capture/kernel_sequences_labeled.json`). To label
a new capture: `uv run python -m alignment label initialize <kernel_sequences.json> <labeled.json>`
with the rule preset of Section 3.5, iterated to a fixpoint as in
`tmp/dsv41/agent-trace/phase5-check1.md`. A new alignment needs a new bundle
directory, because the Analyzer keys one alignment per bundle.

**Profiling a V4.1 row** (example: `compressed_sparse_mla_rope_cast`). Run on a
Slurm node, never the login node, one complete `--specs` call per job, into a
scratch DB that is merged afterwards with `python -m profiling merge-db`:

```bash
#SBATCH -p main --gres=gpu:1 -c 16 --mem=96G
unset VIRTUAL_ENV VIBESIM_MANAGED_JOB_CONTEXT VIBESIM_MANAGED_RUN_CONTEXT
export TMPDIR=<short scratch dir>
uv run python -m launcher kernel-profile run compressed_sparse_mla_rope_cast \
  --backend flashmla_mega --specs specs.json --gpu-name "NVIDIA B200" --db scratch.db --json
uv run python -m launcher kernel-profile count-missing compressed_sparse_mla_rope_cast \
  --backend flashmla_mega --specs specs.json --gpu-name "NVIDIA B200" --db scratch.db --json
```

The backend resolves to `vllm_upstream_fork_env`, so the fork's `.venv` must
exist (`alignment/profiler/vllm/.venv`, or set `VIBESIM_VLLM_FORK_ROOT`). Generate
specs from `simulator cost-trees --kernel-configs` or a launcher `--cache-report`
dry run. The MXFP8 GEMM and MoE backends need a warm autotune cache before a
refill; a cold fill picked inconsistent tactics. Long multi-spec MoE runs come
out bimodal at T 4352-6144, so re-run those bands on their own. The
`flashmla_mega` prefill runner fails a reference check with NaN intermittently
(about 2 in 571 specs); a retry passes. A working job script is
`tmp/dsv41/maxlen/phaseB/profile.sbatch` (gitignored, this worktree only).

**Gotchas:**

- **AF_UNIX socket paths.** FlashInfer's all-reduce workspace and vLLM's ZMQ
  IPC create sockets under `TMPDIR`, limited to 107 bytes. Under the long
  workspace path the all-reduce fusion silently falls back and changes the
  measured kernels. The capture script sets `TMPDIR=/dev/shm/dsv41-tmp-$SLURM_JOB_ID`
  and `VLLM_RPC_BASE_PATH=/dev/shm/dsv41-rpc-$SLURM_JOB_ID`
  (`logs/20260924_0_dsv41_flash_capture/run_profile.sbatch`).
- **DB-backed tests need `PYTHONPATH=$PWD`.** They run from the crate directory
  and otherwise fail with `ModuleNotFoundError: No module named 'profiling'`.
  The same applies to scripts that import `profiling` directly.
- **Block size 128.** The capture uses `--block-size 128`; 256 is rejected by the
  BLHNC KV layout at this upstream base (smoke job 1169).
- **CUDA-graph capture size 2048.** Without `max_cudagraph_capture_size: 2048`,
  vLLM caps graphs at 128 tokens here (2 x `max-num-seqs`) and every mixed
  iteration runs eager, as in capture 1.
- **pytest on the login node.** Use `-m "not gpu and not agent and not bench"`;
  an unmarked run starts GPU tests on the login node's GPUs.
