# GLM-5.3-Flash FP8 on Trainium2 (trn2.48xlarge)

ServingStudio Sim models `zai-org/GLM-5.3-Flash` (FP8 checkpoint) served by the glm5-trn2 seed
engine (`glmflash/neuron/DESIGN.md`, `engine/engine.py`, `engine/serve.py`) on AWS Trainium2 LNC2
logical cores, so that it can predict the engine's InferenceX 8k1k serving performance and rank its
configurations. The Trainium machinery comes from the (unmerged) `trn2-flashnext` branch, ported
onto main and cut to what this model needs.

**Status (2026-11-17).** Re-aligned out of sample on the round-1 R1 service's official measurement
(two replicas, decode-protected waves, C = 8): 40.6 vs 40.20 tok/s/chip, median TPOT 99.1 vs
98.66 ms (`ALIGNMENT.md`, last section). The service's scheduler is the worker's `wave_admission`
mode and its frontend the pool's `least-loaded` placement
(`presets/trn2/glm53_flash_fp8_trn2_service_r1.yaml`). The host floor is now measured (45 ms per
decode forward). The InferenceX traces draw output lengths in [819, 1024] and hold the measured
requests only. The seed-policy sections below predate those fixes except where marked.

**Before (2026-11-16).** The model is aligned on the seed engine's measured C = 16 run (one replica, dp4 x tp8).
It predicts 1107 vs 1099 measured tok/s, a median TPOT of 113.3 vs 114.1 ms, and a median TTFT of
6.47 vs 6.63 s; this is in sample (`ALIGNMENT.md`). The L1 rows are the complete glm5-trn2 benchmark
set (477 measured) plus 393 derived grid cells. The search answer is in `CONFIG_SEARCH.md`: no
seed-engine layout reaches 20 tok/s/user (best 9.7). Decode is device-bound on the MoE partial's
per-call FP8 upcast.

| Artifact | Role |
|---|---|
| `L1_CONTRACT.md` | The Neuron L1 kinds: names, args, grids, bench sources, the derived fill |
| `ALIGNMENT.md`, `alignment.json` | Fit of the host constants and the prefill device factor on the engine's serving timeline |
| `CONFIG_SEARCH.md` | Layout search on 8k1k, the interactivity-constrained answer, bottlenecks |
| `l1_coverage.json` | Measured vs derived rows per kind in the committed `profile.db` |
| `search/` | Raw search results (one InferenceX point per layout and C) |
| `simulator/src/arch/glm53_flash_neuron_fp8.rs` | The arch (`glm53_flash_neuron_fp8`, `glm53_flash_neuron_fp8_speculative`) |
| `simulator/src/arch/config.rs::Glm53NeuronSpec` | Every knob with its default |
| `simulator/src/timing/kernels/neuron_grid.rs` + the 11 kind files | Neuron grids and kinds |
| `profiling/kernels/neuron_common.py` | Imported-only Neuron backends (`neuron_<impl>`) |
| `tools/neuron-bench-import/` | profile.db rows from the engine's bench JSONLs (+ derived fill) |
| `tools/trn2-config-search/` | InferenceX metrics, the config search, the engine alignment (`engine_align.py`) |
| `trace/generate_inferencex.py` | The 8k1k closed-loop traces |
| `presets/trn2/glm53_flash_fp8_trn2_8k1k{,_mtp3}.yaml` | The seed engine at C = 4 ... 256 (and the MTP placeholder) |
| `presets/trn2/glm53_flash_fp8_trn2_align_seed_c16.yaml` | Replays the engine's measured C = 16 run (one replica) |

## Ported from trn2-flashnext

- `gpu/spec.json` `Trainium2-LNC2` (alias `AWS Trainium2 LNC2`), `profiling/gpu_catalog.py`
  vendor / architecture, and the registry's `device_family` / `architectures`
  (`BackendSupport.allows_gpu`: a Neuron backend never matches a CUDA device and vice versa).
- `common::Fabric::{NeuronC1, NeuronC2, NeuronC4}` (ranks per chip of a collective group).
- `timing/kernels/neuron_grid.rs` (trimmed), the `all_gather` / `reduce_scatter` kinds, and their
  imported in-graph collective rows (`tools/neuron-bench-import/data/`).
- The worker's `max_prefill_chunks_per_partition`, plus two new caps the engine needs:
  `max_prefill_chunk_tokens` (one compiled chunk size) and `max_requests_per_partition` (its fixed
  request slots per DP group).
- The pipelined region host model with the fitted constants as defaults.
- Not ported: live Neuron profiling (`profiling/exec/neuron.py` and the runners), the Qwen kinds,
  the P/D deployment, the trn2-engine-align and tune-loop tools. The Neuron backends here are
  imported-only: a missing row fails with the importer command.

## What one step costs

The arch's doc comment is the cost structure. In outline (per replica of `world` = 32 ranks):

- **Layout.** `dp = world / attn_tp` attention-DP groups of `attn_tp` contiguous ranks, one KV
  partition each. KDA and MLA heads are split over the TP group (`64 / tp` per rank); the indexer
  and the latent / indexer-key / gate caches are replicated in the group. The FFN is EP over the
  whole world: 9 experts per rank at EP32 plus a 1/32 shard of the shared expert (dense FFN layers
  0-2: a 1/32 column shard). FP8 weights stay FP8 in HBM and are upcast in-graph (inside the
  `moe_partial` / `dense_ffn_partial` rows).
- **Residual streams** are sequence-parallel in the TP group: a rank holds `R / tp` of its
  group's `R` rows. Per layer (one region): `mhc_pre` (own rows) -> `all_gather(tp)` -> KDA | DSA
  (all R rows) -> `reduce_scatter(tp)` -> `mhc_post`; then `mhc_pre` -> `all_gather(world)` ->
  `moe_partial` | `dense_ffn_partial` over `N = sum of R` rows -> `reduce_scatter(world)` ->
  `mhc_post`. Prologue: `embed` + `reduce_scatter(tp)`; epilogue: head region (`all_gather(tp)`,
  `lm_head` on the sampled rows) and the combine region (id gathers).
- **Step** (`serve.py`): a decode forward over every group's running requests, then a prefill
  forward with one chunk per DP group. Shapes are bucketed and SPMD: every group runs the largest
  group's decode bucket and context bucket (longest KV), and the longest chunk's bucket from the
  furthest `start + T`. `spmd_padding: false` is a what-if (each group its own shape).
- **Launch** (the trn2-flashnext pipelined model, constants fitted on this engine in
  `ALIGNMENT.md`). A forward costs `max(sum_r max(f_g device_r + region_gap_ms, region_host_ms),
  regions x region_host_ms + forward_overhead_ms)`, and the step adds `step_overhead_ms`. f_g is
  `region_device_factor` (1.0) on decode-side leaves and `prefill_device_factor` (1.78, the
  measured prefill-forward excess over its L1 rows) on prefill-forward compute leaves. Regions: 45 layers + embed + head +
  combine = 48 per forward (`region_granularity: block`, `block_layers: k` is a what-if).
- **HBM** (24 GiB per LNC2 core): FP8 FFN + BF16 attention / mHC / head weights per rank plus the
  engine's preallocated pools (`slots` x KDA state of `64 / tp` heads x 128 x 128 fp32 per KDA
  layer, `slots x cmax` x (512 + 128 + 128) bf16 per DSA layer) plus `hbm_reserve_gb`; the build
  refuses a layout that does not fit (TP1 never fits: 24.5 GB of weights per rank). The KV layout
  the worker charges is the same DSA cache (replicated over the group) and KDA state (sharded).
- **MTP placeholder** (`glm53_flash_neuron_fp8_speculative`, `draft_tokens` k): the decode
  forward verifies `B x (k + 1)` rows (KDA / DSA decode rows at that width), then one MTP pass
  (embed + one DSA + MoE layer + head) over the verify rows and `k - 1` draft passes of B rows; the
  prefill forward adds MTP passes over the chunk. Acceptance is synthetic per request
  (`accept_rate` trace column); `golden_accept_rate(k)` inverts InferenceX's glm-5.3-flash golden
  acceptance length (k = 1..5: 1.86, 2.52, 3.00, 3.35, 3.60 -> p = 0.860, 0.830, 0.811, 0.798, 0.790).

## Knobs (`Glm53NeuronSpec`)

| Knob | Default | Effect |
|---|---|---|
| `world` | 32 | ranks per replica (pool `replicas: 2` fills 16 chips) |
| `attn_tp` | 8 | 1, 2, 4, 8, 16; buckets must be multiples of it |
| `moe_ep` | world | must equal `world` (the engine's FFN layout) |
| `prefill_buckets`, `decode_buckets`, `context_buckets` | [2048], [8, 16], [2048, 4096, 8192, 9216] | compiled shapes (serve.py) |
| `region_granularity`, `block_layers` | layer, 4 | `block` = what-if |
| `spmd_padding` | true | what-if off |
| `region_host_ms` / `region_gap_ms` / `forward_overhead_ms` / `mtp_forward_overhead_ms` / `step_overhead_ms` | 1.1 (layer, fitted), 3.7 (block) / 0.1 / 19 / 2 / 0.12 | host model (ALIGNMENT.md) |
| `host_time_scale` | 1.0 | what-if: scales every host constant |
| `region_device_factor`, `prefill_device_factor` | 1.0, 1.78 | f_g on decode-side / prefill-forward compute leaves |
| `kind_time_scale` | [] | what-if `kind=factor` / `kind@min_rows=factor` (`kda`, `dsa`, `moe_partial`, ...) |
| `slots`, `cmax` | 17, 9216 | the engine's pools (HBM check) |
| `hbm_per_core_gb`, `hbm_reserve_gb` | 25.77, 2.0 | HBM check |
| `*_impls` | torch / radix / [dense_local, cte_hybrid] / torch_bf16 | bench impls; several = best measured per shape |
| worker `max_prefill_chunk_tokens`, `max_prefill_chunks_per_partition`, `max_requests_per_partition` | (presets: 2048, 1, 16) | the engine's chunk, one chunk per group, slots - 1 |

## Running it

```bash
# On a host without CUDA, `uv run --no-sync` and `--no-gpu` (rows are imported, never profiled).
# 1. rows: re-run whenever the bench JSONLs grow (measured rows replace derived ones)
uv run --no-sync python tools/neuron-bench-import/import_bench.py \
    --bench-dir /home/ec2-user/glm5-trn2/logs/bench --fill-grid \
    --report doc/pr/trn2-glm53-flash/l1_coverage.json
# 2. the 8k1k sweep (7 concurrencies), then one InferenceX point per run
uv run --no-sync python -m launcher presets/trn2/glm53_flash_fp8_trn2_8k1k.yaml --no-gpu --no-analyze
uv run --no-sync python tools/trn2-config-search/inferencex.py \
    logs/trn2/glm53_flash_fp8_trn2_8k1k/c16 --conc 16 --breakdown
# 3. the config search
uv run --no-sync python tools/trn2-config-search/search.py --stage main --out <dir>
```

Closed-loop fixed concurrency is `arrival_mode: saturated` + `max_concurrency: C` over a trace of
`12C` requests (`trace/generate_inferencex.py`); the metrics drop the first `2C` completions.

## Preset results (8k1k, seed layout, aligned predictions)

Re-run 2026-11-17 with the corrected workload (outputs U[819, 1024], 10C measured requests from
an idle server) and the measured host floor; the seed's continuous batching (serve.py):

| C | tok/s/chip | intvty (tok/s/user) | median TPOT (ms) | median TTFT (s) | median E2E (s) | total tok/s |
|---|---|---|---|---|---|---|
| 4 | 20.3 | 9.61 | 104.1 | 5.21 | 97.1 | 325 |
| 8 | 37.3 | 8.75 | 114.3 | 5.21 | 108.3 | 598 |
| 16 | 64.6 | 7.46 | 134.1 | 5.22 | 126.1 | 1033 |
| 32 | 102.7 | 5.87 | 170.4 | 5.22 | 159.1 | 1643 |
| 64 | 147.1 | 4.09 | 244.2 | 5.22 | 227.1 | 2354 |
| 128 | 197.3 | 2.75 | 363.7 | 6.57 | 337.8 | 3157 |
| 256 | 183.5 | 2.52 | 396.3 | 373.53 | 734.6 | 2937 |

With random output lengths the closed loop desynchronizes the groups and prefill chunks stall
running decodes; the R1 service's decode-protected waves avoid that (C 64: 240 tok/s/chip at
9.6 tok/s/user, `glm53_flash_fp8_trn2_service_r1.yaml`). The table below is the original
(exact 1024-token outputs, 2C warm-up dropped from a continuous loop):

`presets/trn2/glm53_flash_fp8_trn2_8k1k.yaml`: 2 replicas x (dp4 x tp8, EP32), layer regions,
chunk 2048, decode buckets 8/16, 16 requests per DP group (128 per instance).

| C | tok/s/chip | intvty (tok/s/user) | median TPOT (ms) | median TTFT (s) | median E2E (s) | total tok/s |
|---|---|---|---|---|---|---|
| 4 | 18.8 | 9.70 | 103.1 | 5.73 | 111.3 | 300 |
| 8 | 35.5 | 9.47 | 105.5 | 6.37 | 113.3 | 568 |
| 16 | 64.4 | 8.29 | 120.7 | 6.47 | 129.9 | 1031 |
| 32 | 108.2 | 7.04 | 142.0 | 6.63 | 151.9 | 1731 |
| 64 | 181.4 | 5.77 | 173.4 | 6.64 | 183.4 | 2902 |
| 128 | 297.2 | 4.83 | 207.0 | 6.84 | 218.6 | 4755 |
| 256 | 318.3 | 5.22 | 191.6 | 206.44 | 402.4 | 5093 |

At C 256 only 128 requests fit the engine's slots; the rest queue (hence the TTFT). The measured
engine point for comparison is one replica at C 16, i.e. this preset's C 32: 1099 tok/s per
replica = 137 tok/s/chip at 8.77 tok/s/user. The Sim reproduces that on one replica: 137 tok/s/chip
and 8.8 tok/s/user, both for the replay (`ALIGNMENT.md`) and for this preset's own 8k1k trace.

The C 32 row here is lower (108 tok/s/chip, 7.0 tok/s/user). With two replicas behind one
closed loop, a finished request's replacement may land on the other replica. The waves of
same-length requests then desynchronize: about 90% of prefill steps carry one group's chunk while
the other three groups run the SPMD-padded shape idle, and there are 2.3x more prefill steps per
replica. On one replica, the 16 requests start and finish together and every prefill step fills
all 4 groups.

Which regime the real 8k1k benchmark lands in depends on how its client and router spread
requests over the replicas. Synchronized waves are the engine's best case; the SPMD padding of
idle groups is what desynchronization costs.

The two regimes, for one replica at C 16 (2 replicas at C 32):

| regime | tok/s/chip | intvty (tok/s/user) |
|---|---|---|
| engine, measured (one replica, synchronized) | 137 | 8.77 |
| Sim, one replica (synchronized) | 137 | 8.8 |
| Sim, 2 replicas (desynchronized) | 108 | 7.0 |

MTP placeholder (k = 3, acceptance length 3.00), same layout:

| C | tok/s/chip | interactivity (tok/s/user) |
|---|---|---|
| 4 | 31.4 | 17.2 |
| 16 | 93.7 | 12.5 |
| 64 | 190.5 | 6.1 |
| 128 | 279.4 | 4.8 |

## How a VibeSys tick uses this

A VibeSys round proposes an engine change. To predict the 8k1k headline (best tok/s/chip with
median interactivity >= X) before building it:

1. **Express the change as knobs.**
   - Layout and shape changes are arch / worker knobs: `attn_tp`, the buckets, `slots` /
     `max_requests_per_partition`, `max_prefill_chunk_tokens`, `region_granularity: block` +
     `block_layers`.
   - A faster or slower op is `kind_time_scale`, for example `["moe_partial=0.3"]` for a native FP8
     expert GEMM, or `["moe_partial@1024=0.5"]` for prefill-sized calls only. The kinds are `kda`,
     `dsa`, `moe_partial`, `dense_ffn_partial`, `mhc_pre`, `mhc_post`, `lm_head` and `embed`.
   - Less host work is `host_time_scale` or the individual host constants.
   - If the change has its own bench rows (a new op impl), import them instead: add the impl to
     `profiling/kernels/neuron_common.py::GLM_IMPLS`, re-run `tools/neuron-bench-import`, and list
     the impl in the arch's `*_impls`.
2. **Sweep C for the changed layout.** Run the launcher on `glm53_flash_fp8_trn2_8k1k.yaml` with
   `--override pools.main.groups.0.arch.kind_time_scale='["moe_partial=0.3"]'` (and any other
   overrides), `--no-gpu --no-analyze`. Then run `tools/trn2-config-search/inferencex.py` on each
   `c*` run. To search layouts as well, add a stage to `search.py` (as `kernels` does) and read the
   per-X answer it prints.
3. **Read the headline:** the max `tok_s_per_chip` over the points with `intvty >= X`.
   `inferencex.py --breakdown` names the kinds that now dominate the decode and prefill steps:
   the next tick's target.
4. **After the engine run, re-align.** Feed its `serve.py` JSONL to
   `tools/trn2-config-search/engine_align.py`. If the fitted constants move, the change altered
   host or composition behavior. Update the defaults when the fit holds out of sample.

The prefill factor (1.78) was fitted at dp4 x tp8 with chunk 2048. A change that moves work
between prefill kinds or changes the prefill region's composition (fusion, collectives) needs a
re-fit before its prediction is trusted to better than about 10%.

## Open

- **Alignment is in sample at one load.** The constants come from one run (C 16, dp4 x tp8,
  chunk 2048). Validate at another C and layout. `forward_overhead_ms` is not identified,
  because the seed engine's forwards are device-bound. The prefill factor of 1.78 is an
  unexplained composition excess and is applied to every layout.
- **Rows.** 393 grid cells are derived (`l1_coverage.json`). These are DSA at tp 2 / 16, above 64
  rows and at context 32768; KDA at tp 1 / 2; and MoE above 16384 rows. Bench those shapes and
  re-run the importer. The collective rows are the Qwen campaign's and were not re-measured.
- **Engine details not modeled.** First-use compile/load stalls of a new shape; the per-step
  `rt.wait` every 24 launches; the engine's admission (least-loaded group at a free slot; the sim
  places at arrival with vLLM least-loaded); sampling (greedy only); the MTP `eh_proj`.
- **MTP** is a placeholder until the engine has one: the KDA / DSA verify rows reuse the decode
  rows at `B x (k + 1)` rows, the MTP layer is costed as a DSA + MoE layer.
- **Profiling.** The Neuron backends are imported-only; porting trn2-flashnext's
  `profiling/exec/neuron.py` would let the launcher fill missing cells live.
