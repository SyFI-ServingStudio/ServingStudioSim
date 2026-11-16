# GLM-5.3-Flash FP8 on Trainium2 (trn2.48xlarge)

ServingStudio Sim models `zai-org/GLM-5.3-Flash` (FP8 checkpoint) served by the glm5-trn2 seed
engine (`glmflash/neuron/DESIGN.md`, `engine/engine.py`, `engine/serve.py`) on AWS Trainium2 LNC2
logical cores, so that it can predict the engine's InferenceX 8k1k serving performance and rank its
configurations. The Trainium machinery comes from the (unmerged) `trn2-flashnext` branch, ported
onto main and cut to what this model needs.

| Artifact | Role |
|---|---|
| `L1_CONTRACT.md` | The Neuron L1 kinds: names, args, grids, bench sources, the derived fill |
| `CONFIG_SEARCH.md` | Layout search on 8k1k, the interactivity-constrained answer, bottlenecks |
| `l1_coverage.json` | Measured vs derived rows per kind in the committed `profile.db` |
| `search/` | Raw search results (one InferenceX point per layout and C) |
| `simulator/src/arch/glm53_flash_neuron_fp8.rs` | The arch (`glm53_flash_neuron_fp8`, `glm53_flash_neuron_fp8_speculative`) |
| `simulator/src/arch/config.rs::Glm53NeuronSpec` | Every knob with its default |
| `simulator/src/timing/kernels/neuron_grid.rs` + the 11 kind files | Neuron grids and kinds |
| `profiling/kernels/neuron_common.py` | Imported-only Neuron backends (`neuron_<impl>`) |
| `tools/neuron-bench-import/` | profile.db rows from the engine's bench JSONLs (+ derived fill) |
| `tools/trn2-config-search/` | InferenceX metrics, the config search |
| `trace/generate_inferencex.py` | The 8k1k closed-loop traces |
| `presets/trn2/glm53_flash_fp8_trn2_8k1k{,_mtp3}.yaml` | The seed engine at C = 4 ... 256 (and the MTP placeholder) |

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
- **Launch** (`trn2-flashnext` fit, barrier-off engine): a forward costs
  `max(sum_r max(device_r + region_gap_ms, region_host_ms), regions x region_host_ms +
  forward_overhead_ms)`, the step adds `step_overhead_ms`. Regions: 45 layers + embed + head +
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
| `region_host_ms` / `region_gap_ms` / `forward_overhead_ms` / `mtp_forward_overhead_ms` / `step_overhead_ms` | 1.3 (layer), 3.7 (block) / 0.17 / 19 / 2 / 2.5 | host model |
| `host_time_scale` | 1.0 | what-if: scales every host constant |
| `region_device_factor` | 1.0 | f_g on every compute leaf |
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

## Preset results (8k1k, seed layout, predictions)

`presets/trn2/glm53_flash_fp8_trn2_8k1k.yaml`: 2 replicas x (dp4 x tp8, EP32), layer regions,
chunk 2048, decode buckets 8/16, 16 requests per DP group (128 per instance). Predictions from the
rows of `l1_coverage.json` (477 measured, 393 derived) and the unaligned trn2-flashnext host fit.

| C | tok/s/chip | intvty (tok/s/user) | median TPOT (ms) | median TTFT (s) | median E2E (s) | total tok/s |
|---|---|---|---|---|---|---|
| 4 | 18.2 | 9.22 | 108.5 | 3.54 | 114.6 | 292 |
| 8 | 35.2 | 9.08 | 110.1 | 3.96 | 116.0 | 563 |
| 16 | 66.8 | 8.42 | 118.7 | 4.07 | 125.3 | 1069 |
| 32 | 122.5 | 7.78 | 128.5 | 4.16 | 135.1 | 1961 |
| 64 | 210.7 | 6.70 | 149.3 | 4.16 | 156.9 | 3372 |
| 128 | 340.5 | 5.59 | 179.0 | 4.32 | 187.5 | 5448 |
| 256 | 361.5 | 5.89 | 169.9 | 180.42 | 354.2 | 5784 |

At C 256 only 128 requests fit the engine's slots; the rest queue (TTFT). MTP placeholder (k = 3,
AL 3.00), same layout: C 4 / 16 / 64 / 128 -> 31.8 / 105.7 / 254.3 / 348.7 tok/s/chip at
16.9 / 13.7 / 8.3 / 6.0 tok/s/user.

## Open

- **Alignment.** Nothing is aligned against the GLM engine yet: run `serve.py` (it writes per-step
  decode / prefill ms and the InferenceX summary) at a few C and fit `region_host_ms`,
  `forward_overhead_ms`, `step_overhead_ms` (and `region_device_factor`) as trn2-flashnext's
  `ALIGNMENT.md` did; the host constants here are the Qwen engine's.
- **Rows.** 393 grid cells are derived (`l1_coverage.json`); re-run the importer as the benches
  fill in (DSA at tp 2 / 16 and above 64 rows / 16384 context, KDA at tp 1 / 2, MoE above 16384
  rows). Collective rows are the Qwen campaign's, not re-measured.
- **Engine details not modeled.** First-use compile/load stalls of a new shape; the per-step
  `rt.wait` every 24 launches; the engine's admission (least-loaded group at a free slot; the sim
  places at arrival with vLLM least-loaded); sampling (greedy only); the MTP `eh_proj`.
- **MTP** is a placeholder until the engine has one: the KDA / DSA verify rows reuse the decode
  rows at `B x (k + 1)` rows, the MTP layer is costed as a DSA + MoE layer.
- **Profiling.** The Neuron backends are imported-only; porting trn2-flashnext's
  `profiling/exec/neuron.py` would let the launcher fill missing cells live.
