# Qwen3.8-Flash-Next BF16 on Trainium2 (trn2.48xlarge)

ServingStudio Sim models `Qwen/Qwen3.8-Flash-Next` (BF16, text only) served by the
flashnext-neuron engine (`framework/flashnext-neuron`) on AWS Trainium2 logical
cores at LNC2. The engine runs one process per logical core, and each compiled region
holds static shapes for one bucket. This file records what the simulator models, how
to run it, and what is still open.

| Artifact | Role |
|---|---|
| `L1_CONTRACT.md` | The Neuron L1 kinds: names, args in order, backends, grids |
| `simulator/src/arch/qwen4_exp_neuron_bf16.rs` | The L4 arch, with the `qwen4_exp_neuron_bf16` and `qwen4_exp_neuron_bf16_speculative` selectors |
| `simulator/src/arch/config.rs::NeuronFlashNextSpec` | Every knob, with its default |
| `profiling/kernels/neuron_common.py`, `profiling/runners/neuron/` | The Neuron backends (`neuron_<impl>`) that time the engine's own op modules |
| `flashnext-neuron/flashnext/neuron/sim_l1/` | The project-side adapter: one region per spec, built the way the bench scripts build inputs |
| `tools/neuron-bench-import/` | Seeds `profile.db` from the project's `logs/bench/*.jsonl` |
| `presets/trn2/` | Predict configs and simulation presets |
| `model/config/qwen3_8_flash_next_bf16.json` | The checkpoint's `config.json`, verbatim |

## Hardware model

The hardware is one trn2.48xlarge: 16 Trainium2 chips in a 4x4 torus. At LNC2 each
chip exposes 4 logical cores, so the host has 64 ranks. Each rank has 24 GiB of HBM
(`gpu/spec.json` `Trainium2-LNC2`).

**Ranks are logical cores.** A "TP8" group is 8 cores, which is 2 chips. In
collective rows, the fabric names how many of a group's ranks share each chip:

| Fabric | Ranks per chip |
|---|---|
| `neuron_c4` | 4 (every core of each chip) |
| `neuron_c2` | 2 |
| `neuron_c1` | 1 (ranks strided across chips) |

**Every leaf time is NRT device time** from the engine's op modules. Host time (region
dispatch, per-forward input work) lives in each forward's launch leaf
(`launch_model = pipelined`, below). The composition is aligned with the full engine's
serving runs: `ALIGNMENT.md`.

## What one step costs

The cost structure is the doc comment of `qwen4_exp_neuron_bf16.rs`. In outline:

- **DP groups.** A replica is `attn_dp` groups of `G = max(attn_tp, gdn_tp)` ranks.
  - QSA runs `G / attn_tp` sub-replicas of `attn_tp` ranks; GDN runs `G / gdn_tp`
    sub-replicas of `gdn_tp` ranks.
  - Each group's requests are spread over its sub-replicas. Prefill chunks are
    balanced largest first.
  - MoE spans every rank: `moe_ep x moe_tp = attn_dp x G`.
- **Per layer:**
  1. The attn HC module: combine and mix fused into one region.
  2. The mixer: one call per prefill chunk and one for the decode batch, plus its TP
     collective.
  3. The mlp HC module.
  4. MoE over every group's tokens, with its dispatch collectives.
- **PLE block.** It runs before layer 1's attention:
  1. n-gram ids.
  2. The local table gather.
  3. A reduce-scatter over the shards.
  4. The fused post region.
- **Prologue and epilogue.** The prologue is a vocab-parallel embed plus an
  all-reduce. The epilogue is the final HC combine and mixer, the fused lm_head +
  argmax, and an all-gather of the (value, index) pairs.
- **MTP** (speculative selector only).
  1. Pass 0 runs over every target row, verify rows and prefill tokens alike.
  2. `draft_tokens - 1` more passes run, one row per decode request.
  3. Each pass is embed, fc GEMM, HC, the QSA layer, MoE and the head.
- **Step.** A decode forward (+ its MTP passes) then a prefill forward (+ its MTP
  passes), separate programs (`step_forwards: separate`); at most one prefill chunk per
  DP group per step (worker `max_prefill_chunks_per_partition: 1`).
- **SPMD.** Every DP group runs the forward's padded shape, idle groups included
  (`spmd_padding`); world gathers count the "dup" rows of a batch that does not split
  over its TP ranks.
- **Composition** (`launch_model = pipelined`). Every compute leaf is its L1 row (f_g
  1.0). A forward costs `max(Σ_r max(device_r + region_gap_ms, region_host_ms), regions x
  region_host_ms + forward_overhead_ms)`, and the step adds `step_overhead_ms`: the
  host dispatches each region and does its per-forward work while the device runs; a
  region shorter than its dispatch waits for the next launch. Constants fitted on the
  engine (`ALIGNMENT.md`).
- **Padding.** Every shape is padded to its bucket, and the padding costs kernel time.
  If a shape exceeds the largest bucket, it runs as several calls.
- **KV and state per DP partition:**
  - QSA K/V of the rank's KV head on each of the sub-replica's ranks;
  - indexer raw and compressed keys, replicated;
  - GDN SSM and conv state per request, about 115 MB summed over its shards;
  - PLE short-conv state per request.
- **Memory checks.**
  - BF16 weights per rank plus `hbm_reserve_gb` are charged as fixed device bytes, so
    the KV pool is what is left of `attn_gpu_memory_gb` (set this to the whole core:
    25.77 GB).
  - The build refuses a layout whose weights plus reserve exceed `hbm_per_core_gb`.
    16 ranks need about 24 GB of weights each and are refused; 32 ranks need
    12.5 GB each.

## Knobs

| Knob | Values (default) | Effect |
|---|---|---|
| `attn_dp` | power of two (1) | attention-DP groups; one KV partition each |
| `attn_tp` | 1, 2, 4, 8 (8) | QSA TP. tp16 is refused: 24 heads do not split 16 ways |
| `gdn_tp` | 1..16 (= attn_tp) | GDN TP; the group spans max(attn_tp, gdn_tp) ranks |
| `moe_ep`, `moe_tp` | (all ranks / moe_tp), (1) | MoE layout; ep x tp must equal the replica's ranks |
| `lnc` | 2 | the LNC axis. Only LNC2 rows exist; other values are refused |
| `prefill_buckets` | 128 .. 8192 | compiled prefill chunk sizes |
| `decode_buckets` | 1 .. 256 | compiled decode batch sizes |
| `context_buckets` | 2048 .. 131072 | compiled QSA context (block-table) sizes; a prefill within the 2048-token selection budget runs QSA's dense path |
| worker `max_batch_tokens` | (2048 in the presets) | chunked-prefill chunk size (worker knob) |
| `draft_tokens` (speculative tag) and worker `draft_tokens` | 3 | MTP k |
| trace `accept_rate` | per request | MTP acceptance input (`trace/generate_fixed_shape.py --accept-rate`) |
| `region_granularity` | per_op, mixer, layer, block (mixer) | op-level leaves (one region each); the engine's fused regions; one region per layer (QSA front \| back); or cut only at QSA front \| back (14 regions) |
| `step_forwards` | separate, fused (separate) | the engine's decode forward then prefill forward, or one fused forward (what-if) |
| `spmd_padding` | bool (true) | every DP group runs the padded shape; false = each group pays its own (what-if) |
| `region_device_factor` | (1.0) | f_g on every compute leaf |
| `launch_model` | pipelined, calibrated, l1_additive (pipelined) | pipelined: above; calibrated: per region max(device, L_r) + `step_overhead_ms`; l1_additive: the `region_launch_overhead` row per forward |
| `region_host_ms` | (per_op 0.3, mixer 1.0, layer 1.3, block 3.7) | host dispatch per region and its device floor |
| `forward_overhead_ms`, `mtp_forward_overhead_ms`, `region_gap_ms`, `step_overhead_ms` | 19, 2, 0.17, 2.5 | host work per target / MTP forward (overlaps the device), device per region launch, host per step |
| `host_time_scale` | (1.0) | what-if: scales every host constant |
| `launch_coll_ms`, `launch_plain_ms` | 0.15, 0.15 | calibrated only: L_r of a region with / without a collective (barrier-off dispatch) |
| `*_impls` lists | op IMPLS keys | per-op implementation. Several keys = the best measured row per shape. Per-role `backends_file` overrides also apply |
| `tp_collective` | all_reduce, reduce_scatter_all_gather | the latter runs HC and MoE dispatch sequence-parallel on 1/G of the tokens, in a forward whose rows split over the TP group |
| `moe_dispatch` | all_gather, all_to_all | fused block over every group's rows, or the engine's A2A: each destination gets the S-row slice (exact capacity: the experts see the all-gather's rows), the shared expert local |
| `moe_a2a_capacity_factor`, `moe_a2a_pack_us_per_row` | 0 (exact), 0.8 | A2A rows per destination `min(S, ceil(cf S top_k / ranks))`; the engine's packing per received row (fitted) |
| worker `max_prefill_chunks_per_partition` | (unbounded; presets 1) | prefill chunks per DP group per step |
| `ple_table_shards` | replica, dp_group, host | row-sharded over all ranks or one group's ranks. host is refused: a compiled region cannot read host memory |
| `hbm_per_core_gb`, `hbm_reserve_gb` | 25.77, 2.0 | the HBM check and the runtime reserve |
| `kind_time_scale` | list of `kind=factor` or `kind@min_rows=factor` (none) | what-if only: multiplies the device time of every compute leaf of that L1 kind, at calls of at least `min_rows` rows, before f_g and the launch floor. `moe_block@256=0.333` is "MoE prefill 3x faster". Collectives and the launch leaf are never scaled |
| deployment `pd` | | separate prefill and decode instances, each with its own layout and core set; the KV handoff is a 2-rank `neuron_c1` all-gather |

## Running it

```bash
source /home/ec2-user/ServingStudio/wt-trn2-flashnext/neuron_env.sh   # Neuron profiling env (chips 8-15)
export PATH=$HOME/.cargo/bin:$PATH
# offline cost of explicit batch shapes
uv run --no-sync python -m launcher timing-predict presets/trn2/predict_qwen38_flash_next_bf16_trn2_8chip.json
# a simulation (8 chips; --override pools.main.groups.0.replicas=2 for all 16)
uv run --no-sync python -m launcher presets/trn2/qwen38_flash_next_bf16_trn2_8chip.yaml
uv run --no-sync python -m launcher presets/trn2/qwen38_flash_next_bf16_trn2_8chip_mtp.yaml
uv run --no-sync python -m launcher presets/trn2/qwen38_flash_next_bf16_trn2_pd_8x8chip.yaml
# knobs, e.g.
uv run --no-sync python -m launcher presets/trn2/qwen38_flash_next_bf16_trn2_8chip.yaml \
    --override pools.main.groups.0.arch.tp_collective=reduce_scatter_all_gather
```

On this host `uv sync` cannot build deep-gemm (no CUDA), so use `uv run --no-sync`.
`just test-cpu` runs `just sync` first, so on this host run its two halves by hand:

- `cargo test -p simulator --lib`, with libpython on `LD_LIBRARY_PATH`;
- `pytest -m "not gpu and not agent and not bench and not presets" -n 32`.

A missing `profile.db` row is profiled on chips 8-15 (cores 32-63) when the run is not
`--no-gpu`. Multi-chip collectives cannot run through the Sim pool, which reserves one
chip per chunk. They come from the project's standalone sweep
(`flashnext/neuron/sim_l1/collectives.py`) and are imported.

## Headroom ladder (optimal_analyze)

| Rungs | Status |
|---|---|
| R0–R2 | Generic. DP groups are `Max` nodes, so R2 balances them. |
| R3–R5 | Need each row's flops and bytes. Importer and live rows carry analytic estimates from the op shapes. A row without them keeps its observed time at every rung, so it shows no fake headroom. `region_launch_overhead` has no work and stays at its observed time, so it lands in "excess over necessary work". Collectives (`all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`) are on the analyzer's communication list and are removed at R4. |
| R4a | The catalog has no `sustained` block for Trainium2, so R4a = R4b. |
| R4b | Needs `single_gemm` / `elementwise` copy rows on the device. There are none for Neuron yet (only m <= 8192 GEMMs), so R4b falls back as documented. |
| R6/R7 scope bounds | `model.work` already builds `Qwen4ExpForConditionalGeneration` from the config. |
| Per-location R6 | Needs a location map for this arch's manifest, `model/work/location_maps/qwen4_exp_neuron_bf16*_unified.json`. It does not exist yet. |

Run the launcher with `UV_NO_SYNC=1` (set in `wt-trn2-flashnext/neuron_env.sh`). The analyzer shells out to
`uv run`, and a sync attempt fails on deep-gemm. Without it, R3 (grid peaks), R4b and R6/R7 drop out with a
caveat.

Status of the ladder pieces:

- **Location maps.** The location map covers the ordinary unified layout at mixer granularity. The other layouts
  run, but without per-location R6. Those are the speculative tag, PD (leaf prefix `pd.`), per_op, layer, and
  the RS/AG or A2A variants. Scope R6/R7 still evaluates for all of them.
- **Launch floors.** Under `launch_model = calibrated`, a region's launch floor is part of its leaf time. R3
  removes it as "batching", because it bounds the leaf by its rate. Read the R2 − R3 gap as launch-bound
  regions plus genuinely small batches.

## Config search

`CONFIG_SEARCH.md` holds the simulator-only config search over these knobs: the best
configs per workload, knob sensitivity, the headroom ladder, and what-if scenarios for
the launch constants and the MoE prefill kernel. Tool: `tools/trn2-config-search/`;
skill: `operate-search-trn2-configs`.

## Tune loop (engine in the loop)

`TUNE_LOOP.md` holds blind predictions for engine-unmeasured configs, then the engine
runs and predicted vs measured results. On W64, the Sim's ranking of four configs held
once first-use stalls are removed. The best measured config is dp4-tp8-g8 block +
RS/AG + EP32, k = 1, budget 4096: 62.8 s, 1.37x the engine's previous best. The loop
also found a2a mis-modeled at dp8 and a dp8 block prefill that does not fit HBM. It
gives the headroom of the best config. Tool: `tools/trn2-tune-loop/`; skill:
`operate-tune-trn2-with-engine`.

## Preset results (aligned, 2026-10-25)

The presets replay the W64 trace (64 requests, 2048 in / 512 out, 50 ms apart) at its
own arrival times with the engine's knobs (layer regions, its compiled buckets, one
chunk per DP group).

| Preset | Ranks | Duration (s) | Out tok/s | Engine |
|---|---|---|---|---|
| 8-chip DP4 x TP8 EP32, layer, chunk 2048 | 32 | 163.7 | 200 | 169.3 s, 194 tok/s (`serve_w64_2048_512_layer_k0`) |
| same + MTP k=1, uniform 0.67 acceptance | 32 | 152.4 | 215 | 183.4 s at per-request acceptance 0.01-0.97 (mean 0.65); with those per-request rates the Sim gives 170.8 s (`ALIGNMENT.md`) |
| P/D, 8 + 8 chips (mixer regions, default buckets) | 64 | 190.8 | 172 | not in the engine |

The first results of step 4 (1326 tok/s on 8 chips) replayed the trace 10x compressed
(`request_rate` default 10) under the barrier-on launch constants; they are superseded.
