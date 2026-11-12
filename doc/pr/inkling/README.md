# Inkling NVFP4 evidence (B200, vLLM TP4/EP4 and TP8/EP8)

Review evidence for the `inkling_vllm_nvfp4_rel_attn_moe` arch on
Inkling-Small-NVFP4 and Inkling-NVFP4: what the simulator models, how it was
checked against vLLM captures, and what is still open. This is not an accepted
baseline. Tolerance failures and the known gaps below stay explicit, and no
golden has been recorded.

Raw captures, Analyzer reports and profiling jobs stay in the experiment trees
(untracked):

- `logs/20261008_10_inkling_standard/`: Small, TP4/EP4, `diverse_100`;
- `logs/20261009_0_inkling_nvfp4_tp8ep8_diverse100/`: Inkling-NVFP4, TP8/EP8,
  `diverse_100`;
- `logs/20261009_1_inkling_nvfp4_tp8_campaign/`: Inkling-NVFP4, the 15-case
  campaign.

Numbers are tagged *measured* (a vLLM capture or a B200 profiling run),
*simulated* (a timing prediction or simulation), *catalog* (the checkpoint
config or fork source) or *derived* (arithmetic on those).

## Maintained artifacts

| Artifact | Responsibility |
|---|---|
| `presets/public/<checkpoint>/inkling_vllm_nvfp4_rel_attn_moe.yaml` | Arch presets. Inkling-NVFP4 sweeps `max_model_len` 131072 and 1048576; Inkling-Small-NVFP4 offers 131072 only |
| `presets/public_sim/<checkpoint>/inkling_vllm_nvfp4_rel_attn_moe_chunked_prefill.yaml` | Deployment presets, replicas 1 / 2. Inkling-NVFP4 adds `server` ctx128k / ctx1m |
| `model/work/location_maps/inkling_vllm_nvfp4_tp{4_ep4,8_ep8}_unified.json` | The Analyzer's necessary-work maps for both public layouts |
| `presets/alignment/inkling_nvfp4_b200_tp8/` | The 15-case pack: `campaign.yaml` (from `make_pack.py`), `acceptance.yaml`, `label_rules/` |
| `alignment_matrix.md` | Generated case matrix and tolerance failures |
| `campaign_metrics.json` | `extract` output behind the matrix. The committed copy drops the per-request identity lists and makes paths relative to the repository root. |

## Deployment modeled

vLLM V1 on B200, from the vLLM fork `servingstudio-alignment` at `892da082`.
Attention is TP, and the routed experts are EP over the same ranks
(`--enable-expert-parallel`). Chunked prefill runs at 2048 tokens with
`--max-model-len 131072` and `--gpu-memory-utilization 0.85`. MTP, vision and
audio are off.

| Checkpoint | GPUs | Workload | Captures |
|---|---|---|---|
| Inkling-Small-NVFP4 (42 layers, hidden 4096) | 4, TP4/EP4 | `trace/diverse_100.csv`, trace-timed | Workload pass and nsys pass (3910 iterations) in `logs/20261008_10_inkling_standard/`; token corpus (135,144 tokens, 40 MoE layers) in `logs/20261008_9_inkling_small_nvfp4_tp4ep4_diverse100/` |
| Inkling-NVFP4 (66 layers, hidden 6144) | 8, TP8/EP8 | `trace/diverse_100.csv`, trace-timed, c100 | Workload pass (job 5137, 100/100 requests, 12693 iterations), nsys pass (job 5138, 1655 iterations), token corpus (job 5139, 135144 tokens, 64 MoE layers) |
| Inkling-NVFP4 | 8, TP8/EP8 | The 15 cases of `inkling_nvfp4_b200_tp8` | Jobs 5273-5279, 5286-5292 and 5301 |

Both checkpoints have 256 routed experts, top-6, and `model_max_length`
1048576 (*catalog*, `model/config/inkling_{small_,}nvfp4.json`). Expert
routing reads the measured token corpus of each `diverse_100` capture, never
uniform. EPLB was off, so there is no popularity file.

## How each model feature is modeled

| Feature | vLLM | Simulator |
|---|---|---|
| Sliding-window and global relative-position attention | tml FA4 with a relative-position bias. On global layers the split count follows the request count. | `rel_bias_attention` / `tml_fa4`. Layers in `local_layer_ids` are sliding-window; the rest are global. |
| QKVR projection and prep | One fused projection, then `fused_qkvr_prep` on two streams. | `single_gemm` plus `sconv_qkvr_prep` / `vllm_triton`. Prep is timed as the interval union of both streams. |
| Short convolution and collectives | No stand-alone all-reduce or RMSNorm. Each sublayer ends in one Lamport call: reduce-scatter, short conv on the shard, all-gather, residual add and the next norm. | `rs_sconv_ag_residual_rms_norm` / `vllm_lamport_triton`, two calls per layer. The MoE call also sums the sink partial. |
| Router | Replicated on every rank. `_linear_with_fp32_out`: cute-DSL `ll_bf16` at m <= 64, `torch.mm` with fp32 output above. | `gemm_fp32_output` / `vllm_ll_bf16_auto`, with grid points on both sides of each kernel switch. |
| Gate select and top-k | One gate-select kernel, then top-k weight and id copies. The copies are no-ops on a one-row slice. | `moe_sink_gate_select` / `vllm_triton`. The top-k copies are charged only above one token. |
| Routed experts | NVFP4, except BF16 on layer 2, where the checkpoint excludes them. | `nvfp4_fused_moe` / `flashinfer_trtllm_routed_sm100_fork`, a `Max` over the EP ranks' ranked workloads. Layer 2 uses `bf16_fused_moe` (below). |
| Two sink experts | TP gated MLPs on vLLM's aux stream beside the routed call, at <= 256 tokens. A breakable CUDA graph capture drops the aux stream, so PIECEWISE steps run the sinks inline. | `CostNode::Parallel` with overlap 0.89 on uniform-decode FULL-graph or eager steps of <= 256 tokens; serial after the routed call otherwise. |
| Dense FFN | Layers `0..dense_mlp_idx`, TP gated MLP. | The GLM-5.3 gated MLP worklet. |
| CUDA graphs | FULL graphs for uniform decode, padded to the capture sizes, with zero-length padding requests in the attention metadata. | Tokens are padded to `cudagraph_capture_sizes`. Global attention on a uniform-decode step is priced at the padded request count. |
| Prologue and head | Embedding, `embed_norm` and log-scaling tau; `lm_head` over one row per request, then a logits all-gather. | 8 prologue launches: the fused embed and norm, then 7 for the log-scaling tau. The all-gather is priced on the `all_reduce` curve at half the gathered bytes. |

Sampling, MTP, vision and audio are not modeled.

### KV accounting

The arch charges K and V of every layer's KV heads in BF16, with
`PAGE_SIZE` 32, and counts sliding-window layers at full length. vLLM does
not:

- It draws every layer's pages from one block pool. Layers of one cache type
  form groups of equal layer count, and a request holds blocks in each group.
- A global layer holds one block per 32 tokens. A sliding-window layer holds
  only its 512-token window plus the tokens in flight, and frees the blocks
  behind it.
- The fork registers each layer's short-conv state as a sliding-window cache
  with a 4-token window. The arch does not count it.

These rules reproduce both servers' logged concurrency at 131072 tokens per
request: 37.61x on Inkling-NVFP4 TP8 (`GPU KV cache size: 4,930,120
tokens`) and 45.25x on Small TP4 (5,931,231). Under the arch's per-token
charge, the same pools hold 9.8 and 20.2 such requests (1.28M and 2.65M
tokens). For requests of that length, capacity is about 3.9x too small
on TP8 and 2.2x on Small. No simulation in this PR reaches it. The largest active KV is
786,028 tokens on Inkling-NVFP4 (campaign case 10, 61%) and 694,552 on
Small (`diverse_100`, 26%). A follow-up PR models the cache groups.

`attn_gpu_memory_gb` is each capture's logged per-GPU pool, x 1.073741824
(*measured*):

| Checkpoint | Logged pool | `attn_gpu_memory_gb` | Source |
|---|---:|---:|---|
| Inkling-NVFP4 | 73.85 GiB (4,930,120 tokens) | 79.2958 | `logs/20261009_0_inkling_nvfp4_tp8ep8_diverse100/summary.md` |
| Inkling-Small-NVFP4 | 106.11 GiB | 113.9349 | `logs/20261008_10_inkling_standard/profile_workload_metrics/` |

At `max_model_len` 1048576 the pool is not measured. The Inkling-NVFP4 ctx1m
preset keeps the 131072 pool, because the campaign servers logged 73.85 GiB at every
`--max-model-len` they ran, 8192 to 524288.

## Kernels

| Kind / backend | What it times | Fidelity, cache vs fresh measurement |
|---|---|---|
| `rel_bias_attention` / `tml_fa4` (new kind) | tml FA4 relative-position attention, local and global layers. Global contexts are gridded up to `max_model_len` keys within the 500-point ceiling. | Inkling-NVFP4 captured batches (job 5183): 30/30 within ±15%, median 0.983, worst 0.877. Global layers on mixed batches read 0.877-1.043, 10 of 12 below 1. |
| `sconv_qkvr_prep` / `vllm_triton` (new kind) | The fork's `fused_qkvr_prep`, both streams, interval union. | 124/124 probes within ±15%, worst 0.909 and 1.109 (commit 96c783ab). That check predates the move to interval union timing. |
| `rs_sconv_ag_residual_rms_norm` / `vllm_lamport_triton` (new kind) | The Lamport call: 30 back-to-back calls in one graph, host-timed, slowest rank, no L2 clear. | 74/74 probes within 5% on TP4 hidden 4096 (commit 4bfcda10). That check predates the back-to-back timing. |
| `moe_sink_gate_select` / `vllm_triton` (new kind) | Inkling's gate select. | 172/172 probes within ±15%, median 1.000, worst 0.959 (commit 8ec64b88). |
| `gemm_fp32_output` / `vllm_ll_bf16_auto` (new backend) | The router dispatch as one call, CUPTI busy union. Runs in `vllm_upstream_fork_env`. | A switch-aware fidelity caller exists. No result is recorded. |
| `nvfp4_fused_moe` / `flashinfer_trtllm_routed_sm100_fork` (new backend) | The routed NVFP4 MoE on the fork's FlashInfer 0.7.0. The pinned image has FlashInfer 0.6.18, whose FC1 tile runs about 2x slower at T >= ~1792 (355-390 us vs 205 us captured at T=2048). | The fork's FlashInfer reads 210 us at T=2048 (commit 781b7451). No cache fidelity run is recorded. |
| `bf16_fused_moe` (existing trtllm-gen backend) | Stands in for FlashInfer CUTLASS on layer 2's BF16 experts. | Existing backend; not re-checked. |

## Check 1: kernel alignment on `diverse_100`

Duration-weighted signed error of the simulated critical path against the
measured one, per iteration (*simulated* vs *measured*).

**Inkling-Small-NVFP4, TP4/EP4.** Source:
`logs/20261008_10_inkling_standard/check1e_derived.md`.

| stage | n | measured ms/iter (mean) | simulated ms/iter (mean) | signed rel. err median |
|---|---|---|---|---|
| decode | 3609 | 5.348 | 5.363 | -0.59% |
| mixed | 301 | 27.184 | 27.103 | -0.17% |
| all | 3910 | 7.029 | 7.037 | -0.56% |

Mapping covers 98.85% of the measured critical path. The recommended
`gpu_time_multiplier` is 1.0464802449385324; the presets use 1.0465.

**Inkling-NVFP4, TP8/EP8.** Source:
`logs/20261009_0_inkling_nvfp4_tp8ep8_diverse100/check1_derived.md`.

| stage | n | measured ms/iter (mean) | simulated ms/iter (mean) | signed rel. err median |
|---|---|---|---|---|
| decode | 1356 | 8.741 | 9.231 | +5.92% |
| mixed | 299 | 52.074 | 51.414 | -1.21% |
| all | 1655 | 16.569 | 16.852 | +5.69% |

Mapping covers 99.50% of the measured critical path. The recommended
`gpu_time_multiplier` is 1.0276.

Read the per-op rows by group. The Analyzer gives a hidden `Parallel` branch
zero simulated time and gives the measured time to the launch that started
first. So on decode steps `moe.routed_nvfp4` reads +30.6% and the sinks read
-100%; that is attribution, not cost.

The TP8 decode gap (+0.568 ms/iter against the mapped measured 8.663) splits
as follows. Source: `logs/20261009_0_inkling_nvfp4_tp8ep8_diverse100/summary.md`.

| part | gap ms/iter | cause |
|---|---|---|
| MoE input_quant + routed + sinks | +0.152 | `SINK_STREAM_OVERLAP` 0.89 is the Small TP4 calibration; on TP8 the measured union is 1.061x the isolated max |
| topk_copy | +0.093 | The `elementwise:torch` row is timed with L2 cleared; in serving the copy reads gate select's just-written output, warm in L2 (job 5179) |
| o_proj + qkvr_proj + router | +0.176 | The rows match each kernel's raw interval; the critical-path share is shorter. Mechanism not isolated. |
| Lamport attn + moe | +0.100 | Within the rows' repeatability (job 5184) |
| logits_all_gather | +0.036 | not examined |
| gate_select | +0.017 | not examined |

In mixed steps, `qkvr_prep` -23% is attribution: against the per-rank interval
union of its two streams the simulation reads 0.986-0.993. `rel_attention`
-5.6% is the cache's equivalent-context projection on global layers (job
5183).

## Simulation on `diverse_100`

Each capture's workload pass, simulated with `chunked_prefill` at the
recommended multiplier. Request ids: 100 shared, 0 one-sided in both.

**Inkling-Small-NVFP4, TP4/EP4, multiplier 1.0465.** Source:
`logs/20261008_10_inkling_standard/e2e_b2b_table.md`, column `b2b_m1046`.

| metric | stat | measured | sim b2b_m1046 | err b2b_m1046 |
|---|---|---|---|---|
| TTFT (client) (ms) | mean | 639.6 | 623.3 | -2.6% |
| TTFT (server) (ms) | mean | 606.5 | 623.3 | +2.8% |
| TPOT (client) (ms) | mean | 10.1 | 11.6 | +14.5% |
| TPOT (client) (ms) | p50 | 8.4 | 9.5 | +13.9% |
| TPOT (server) (ms) | mean | 10.5 | 11.6 | +10.3% |
| E2E latency (ms) | mean | 11924.7 | 13306.4 | +11.6% |
| output throughput (tok/s) | completion | 925.1 | 893.6 | -3.4% |

The completion span is 146192 ms measured and 151348 ms simulated. TPOT is a
known gap (below).

**Inkling-NVFP4, TP8/EP8, multiplier 1.0276.** Source:
`logs/20261009_0_inkling_nvfp4_tp8ep8_diverse100/e2e_table.md`.

| metric | stat | measured | sim m1028 | err m1028 |
|---|---|---|---|---|
| TTFT (client) (ms) | mean | 1839.1 | 1758.8 | -4.4% |
| TTFT (server) (ms) | mean | 1791.0 | 1758.8 | -1.8% |
| TPOT (client) (ms) | mean | 24.5 | 25.1 | +2.3% |
| TPOT (client) (ms) | p50 | 20.0 | 20.3 | +1.7% |
| TPOT (server) (ms) | mean | 25.0 | 25.1 | +0.4% |
| E2E latency (ms) | mean | 26419.4 | 27066.8 | +2.5% |
| output throughput (tok/s) | completion | 701.7 | 682.2 | -2.8% |

The completion span is 192731 ms measured and 198253 ms simulated.

## Campaign: 15 operating points (Inkling-NVFP4, TP8/EP8)

Source: [`alignment_matrix.md`](alignment_matrix.md), generated by
`alignment-campaign compare --markdown` from
`logs/20261009_1_inkling_nvfp4_tp8_campaign/metrics_run5.json`
(committed as `campaign_metrics.json`).

| Case | kernel_abs_pct | kernel_signed_pct | map_coverage_pct | e2e_mean_pct | output_tps_pct | server_tpot_mean_pct | server_ttft_mean_pct |
|---|---:|---:|---:|---:|---:|---:|---:|
| tp8_ep8/01_micro_throughput_c256 | +2.51% | -0.61% | 98.78% | -4.02% | +5.07% | -1.38% | +2.30% |
| tp8_ep8/02_graph_boundaries_c128 | +1.98% | +1.34% | 99.04% | -2.73% | +3.34% | +0.39% | -1.87% |
| tp8_ep8/03_balanced_anchor_c32 | +1.11% | -0.76% | 99.32% | -2.13% | +1.79% | -0.63% | -7.99% |
| tp8_ep8/04_prefill_spectrum_c32 | +1.49% | -0.96% | 99.50% | -3.91% | +3.94% | -1.73% | -3.79% |
| tp8_ep8/05_decode_spectrum_c64 | +0.65% | +0.42% | 99.13% | -0.01% | -0.97% | -0.33% | -8.53% |
| tp8_ep8/06_long_long_c8 | +5.47% | +5.40% | 99.15% | +4.11% | -3.88% | +4.23% | -9.06% |
| tp8_ep8/07_quadrant_interference_c48 | +1.76% | +1.42% | 99.24% | +0.61% | -0.54% | +0.22% | -3.04% |
| tp8_ep8/08_completion_churn_c96 | +1.77% | -0.03% | 99.16% | -0.87% | +0.68% | -1.27% | -1.55% |
| tp8_ep8/09_rate_knee | +5.24% | +5.19% | 99.18% | -1.27% | -0.17% | -1.15% | -10.26% |
| tp8_ep8/10_full_saturation_rate | +2.15% | -2.12% | 99.33% | -4.68% | +4.13% | -4.83% | -3.07% |
| tp8_ep8/11_long_prefill_ladder_c1 | +0.61% | +0.49% | 99.66% | -3.45% | +3.60% | -6.31% | -1.75% |
| tp8_ep8/12_ultra_context_decode_c1 | +0.88% | +0.83% | 99.66% | +0.09% | -0.09% | +1.59% | -1.63% |
| tp8_ep8/13_generation_length_matrix_c8 | +3.04% | +2.40% | 99.34% | +4.34% | -4.06% | +4.41% | -6.12% |
| tp8_ep8/14_near_full_long_decode_c21 | +1.58% | -0.19% | 99.61% | -0.66% | -0.07% | -0.74% | +11.45% |
| tp8_ep8/15_kv_overcommit_mixed_churn_c14 | +1.90% | -1.01% | 99.66% | +3.52% | -3.48% | +3.49% | +27.73% |

- **Coverage.** All 15 cases completed every phase. Each workload and nsys
  pass ran with a cold per-pass FlashInfer autotune cache and a 600 s stall
  watchdog. Case 09's timing prediction ran on CPU.
- **Calibrated inputs.** Every case logged a 73.85 GiB pool, so
  `attn_gpu_memory_gb` is 79.2958. Simulations run at `gpu_time_multiplier`
  1.0. Case 09 replays at the provisional rate of 1.0 req/s, as GLM's pack
  does.
- **Labels.** `label_rules/tp8_qkvr_rules.json` adds 7 rules for the TP8 QKVR
  launches. With them, case 01 coverage goes from 97.70% to 98.78% and case 02
  from 96.81% to 99.04%.
- **Tolerances.** `acceptance.yaml` copies the default tolerances of
  `glm52_nvfp4_b200_tp8` and has no per-case exceptions yet.

Out of tolerance, with cause (from
`logs/20261009_1_inkling_nvfp4_tp8_campaign/summary.md`):

1. **Iteration population, cases 01, 02 and 04.** Iterations, decode batch,
   prefill tokens and scheduled KV differ 2.69-2.77%, 2.51-2.58% and
   1.25-1.27%; the tolerance is 1%. At high concurrency the two engines batch
   the same trace differently, and kernel error in these cases is 1.1-2.5%. The
   GLM-5.2 pack measured 2.69-2.77%, 2.92-3.00% and 1.25-1.29% and carries
   per-case bounds of 3.0, 3.2 and 1.4.
2. **Decode kernel, cases 06 and 09: +5.4% and +5.2%** (tolerance 5%). Per
   decode iteration, case 06 splits as routed +0.73, sink -0.49, topk_copy
   +0.09, o_proj +0.07 and rs_sconv +0.10 ms; case 09 as routed +0.70, sink
   -0.49, topk_copy +0.09, rs_sconv +0.08 and o_proj +0.07 ms. These are the
   `diverse_100` decode cost items above.
3. **Case 15 server TTFT: +27.7%** (tolerance 12%). Case 14 reads +11.45%,
   inside the tolerance but slower. The gap sits in the cohort submitted
   together at t=0: case 15 +63.7%, case 14 +28.7%. The rest of case 15 reads
   -6.6%. The simulator prefills that cohort in trace-id order; vLLM's engine
   received it in another order, and the cause of that order is unverified.
   Per-request prefill time matches within about 5%. GLM's case 15 shows the
   same +35.1% and carries a 36% bound.

## Long context

The `rel_bias_attention` grid places global-layer context knots up to the
arch's `max_model_len`, so each length needs its own rows. Job 5317
(`logs/20261010_0_inkling_public_1m_fill/`) ran the public members' probe
timing predictions at 1048576 on B200 (*measured*):

| Checkpoint | Global specs profiled | Result |
|---|---:|---|
| Inkling-NVFP4, TP8 | 639 / 639 | Profiled. The public preset sweeps 131072 and 1048576. |
| Inkling-Small-NVFP4, TP4 | 631 / 639 | 8 specs with 256 decodes at 1048576 keys ran out of CUDA memory on one GPU. The public preset stays at 131072. |

The 284 sliding-window specs have the same shape on both checkpoints and were
all profiled. That makes 1,554 new rows.

The grid has no KV-capacity mask, so it asks for batches no TP4 server could
hold. Masking it would let Small offer 1M; that is a code change and awaits
approval.

Campaign cases 11 (long prefill ladder) and 12 (ultra-context decode), both
c1, read kernel abs error 0.61% and 0.88%.

## Changes that reach other models

No existing profile.db row is changed or removed, and the shared code below
adds new variants or opt-in fields. No other model's prediction is expected
to move; this was not checked by a run.

- **FlashInfer autotune cache** (`profiling/runners/autotune_cache.py`). A
  process now keeps the live tactics while it profiles specs of one
  configuration, and forgets them when the configuration changes. FlashInfer
  0.7 ignores the loaded tactic file while tuning when the op's profiling
  policy is not the legacy default. Clearing per spec therefore re-tuned every
  bucket on every spec: 46 s of a 46.3 s routed-MoE spec. This changes
  profiling time for every FlashInfer-autotuned backend. No existing row was
  re-measured.
- **CUPTI multi-launch check** (`profiling/profilers/cupti_kernel_profiler.py`).
  Under `interval_union`, a launch's kernel names are compared as a multiset,
  because a side stream's kernel may start first. Other timing modes are
  unchanged.
- **`gemm_fp32_output` grid.** The extra `m` points (3, 5, 65 and 8..128 step 8)
  apply only to `vllm_ll_bf16_auto`; other backends keep their grid.
- **`VllmTpGroupLauncher`** in `profiling/runners/comm/_launcher.py`, a launcher
  that sets up vLLM's own TP group for callables that look it up.
- **model.work.** `Model.param_vocab` (Inkling's padded embedding and head
  tables), `MoE.router_extra_logits` (the 2 sink logits) and a `short_conv`
  mixer. Their defaults leave other models' counts unchanged.
- **Shared worklets.** `Nvfp4MoeLocalWorklet` and `Bf16MoeLocalWorklet` gain
  an `eval_or_zero`, which pushes zeros in their own leaf order. GLM-5.2 and
  GLM-5.3 do not call it.
- **`profiling/runners/comm/_launcher.py`.** `VllmLauncher` and the new
  `VllmTpGroupLauncher` share one spawn helper. Spawn arguments and error
  forwarding are unchanged.
- **`flashinfer_trtllm_routed_sm100_fork`** runs in `vllm_upstream_fork_env`,
  so filling its rows needs the fork's `.venv`.

## Modeling choices

- **Sink overlap is one constant.** `SINK_STREAM_OVERLAP = 0.89` is 1/1.124.
  Over the 3609 decode steps of the Small capture, the measured union of input
  quant, routed call and sinks is 1.124x the isolated max of their rows. Both
  checkpoints use it; on TP8 the union is 1.061x (known gap).
- **Sinks inline on PIECEWISE steps.** Inkling auto-enables breakable CUDA
  graphs, and their capture drops the aux stream. So only uniform-decode FULL
  graph steps and eager steps at <= 256 tokens overlap the sinks.
- **Lamport calls are timed back to back.** 30 calls per graph replay, slowest
  rank, no L2 clear, as the other graph-captured collectives are timed. Every
  call reads that row, decode and mixed.
- **BF16 MoE stand-in.** Layer 2's BF16 experts run FlashInfer CUTLASS in
  production; the `bf16_fused_moe` trtllm-gen backend prices them.
- **All-gather as all-reduce.** There is no all-gather kind; the logits gather
  is priced on the `all_reduce` curve at half the gathered bytes. On TP8 decode
  it reads 0.058 ms simulated against 0.022 ms measured.
- **Padded global attention.** A uniform-decode FULL graph pads the attention
  metadata to the capture size, and the split count follows the padded count.
  The arch prices global attention there at the padded count.
- **Sliding-window KV at full length.** At 131072 tokens per request, KV
  capacity is about 3.9x too small on TP8 and 2.2x on Small. No simulation in this PR reaches it; a follow-up
  PR models vLLM's cache groups.

## Known gaps and follow-up

- **Small TPOT +14.5%** (client mean; server +10.3%), E2E mean +11.6%. Check 1
  is within 1%, so the gap is not kernel cost. The multiplier comes from the
  nsys capture, and the workload pass runs about 6% offset from it; the slower
  decode then grows the batches. Not fixed.
- **Large decode cost items** (+5.92% `diverse_100` decode; campaign 06 and
  09). Each needs approval, because it changes a model or a calibration:
  - recalibrate `SINK_STREAM_OVERLAP` for TP8 (1/1.061 = 0.943), which makes
    the constant deployment-dependent;
  - a warm-L2 `topk_copy` row, a timing-method change;
  - the critical-path share of o_proj, qkvr_proj and router, mechanism not
    isolated;
  - Lamport rows scatter ±20-30% at small T under host timing; tighter rows
    need more repeats.
- **`rel_bias_attention` global projection.** On mixed batches with a 2048
  chunk beside long decodes, the cache reads 0.88-0.97 of fresh truth on
  global layers.
- **Per-case acceptance bounds** for the iteration population (cases 01, 02,
  04) and case 15 TTFT, mirroring GLM. Awaiting approval.
- **t=0 cohort order** behind case 15 TTFT. Replaying the measured order would
  condition on the realized run.
- **Case 09 rate.** At 1.0 req/s the simulator shows 0 retractions and a 66 ms
  mean TTFT, so it is not a knee for Inkling. A calibrated knee needs a rate
  sweep.
- **Fidelity re-runs** for `sconv_qkvr_prep` and `rs_sconv_ag_residual_rms_norm`
  under their current timing methods, and a recorded result for
  `vllm_ll_bf16_auto` and the fork MoE backend.
- **1M context.** The 1M server's KV pool is not measured, and no capture ran
  at 1048576. Small offers 1M only after a KV-capacity mask on the
  `rel_bias_attention` grid (approval needed).
- **Several prefills in one step.** `rel_bias_attention` projects a step's
  prefill requests as one chunk of their summed query tokens. When the real
  batch has less work than that chunk at context 1, the projection clamps
  there and overprices: eight fresh 256-token prompts read as one fresh
  2048-token prompt. A unit test records the current behaviour. A fix adds
  a prefill-request coordinate or changes the projection, so it needs
  approval.
- **Small chunks at GQA 8.** Inkling-NVFP4's global layers run 8 query heads
  per KV head at TP8, where the q_stage switch falls at 16 tokens. The prefill
  axis starts at 32, so chunks of 16 tokens or fewer read the q_stage-2 row.
- **Grid size at 1048576.** The Inkling-NVFP4 global grid has 639 coordinates
  there, above the 500-point ceiling. A KV-capacity mask would bring it under.
- **Projection cost.** `rel_bias_attention` projects each batch by bisection
  on its CTA model, about 10-120 us per lookup. A closed form would remove most
  of it.
- Sliding-window KV at full length; sampling, MTP, vision and audio not
  modeled.

## Reproduce

From the repository root. The campaign, with
`PACK=presets/alignment/inkling_nvfp4_b200_tp8` and `OUT` a fresh experiment
directory:

```bash
uv run python -m launcher alignment-campaign check --pack "$PACK"
uv run python -m launcher alignment-campaign render --pack "$PACK" \
  --host <host> --out-root "$OUT"
for phase in profile_workload profile_nsys timing_predict; do
  uv run python -m launcher alignment-campaign run --pack "$PACK" --out-root "$OUT" --phase "$phase"
done
uv run python -m launcher alignment-campaign label --pack "$PACK" --out-root "$OUT"
for phase in analysis_kernel simulation analysis_e2e; do
  uv run python -m launcher alignment-campaign run --pack "$PACK" --out-root "$OUT" --phase "$phase"
done
uv run python -m launcher alignment-campaign extract --pack "$PACK" --runs "$OUT" --out metrics.json
uv run python -m launcher alignment-campaign compare --pack "$PACK" --measured metrics.json --markdown
```

- `<host>` names a host file under `presets/alignment/hosts/`, modeled on
  `example.yaml`.
- The profile phases need 8 B200s per case. Run each workload pass alone on its
  node.
- `campaign.yaml` is generated: edit `make_pack.py`, which reads
  `cudagraph_capture_sizes` from the public arch preset.

The public presets are checked by building every member against the
committed profile.db, with no GPU:

```bash
just test-presets
```

The `diverse_100` case configs moved out of `presets/` into the untracked
experiment trees (`presets_moved/` in each).

Notes from the captures and profiling jobs:

- vLLM's ZMQ IPC path is `TMPDIR` plus a uuid and must fit 107 bytes; point
  `TMPDIR` at a short directory.
- Give each pass a cold FlashInfer autotune cache and a stall watchdog, as
  the campaign did.
- The fork-env backends need the fork's `.venv` built.
