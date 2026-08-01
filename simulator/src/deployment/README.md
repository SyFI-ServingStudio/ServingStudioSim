# Deployment — config → a runnable Flow

The deployment layer is the assembly glue: it deserializes a **structured run
config** and builds the concrete L6b `Flow` the sim driver ticks. The
`deployment` tag is the top-level switch that fixes, in one choice, the
orchestrator (1:1), the pool topology, and each pool's contract class (which arch
+ worker types are legal). It composes the lower layers; it contains no cost or
scheduling logic of its own.

This is the practical, code-matching reference; the code is the ground truth.
For the layer overview see `doc/detailed_design/L6.md`; the archived config-tree
design intent is `old-doc/new-interface-design.md`.

## The `Deployment` trait + dispatch

```rust
pub trait Deployment {
    const NAME: &'static str;     // serde `deployment` tag + list-params key, e.g. "unified"
    type Config;                  // the deserialized per-deployment config
    fn build(cfg, bridge, store) -> anyhow::Result<Box<dyn Flow>>;
}
```

`build_flow(&RunConfig, bridge, store)` is the single dispatch point that routes
the deserialized config to its deployment's `build`:

| `deployment` tag | status |
|---|---|
| `unified` | **wired** — `UnifiedDeployment::build` |
| `pd` | **wired** — `PdDeployment::build` for the supported prefill/decode arch pairings |
| `afd` | **wired** — `AfdDeployment::build` for Qwen3 attention/FFN disaggregation |

## The config shape (`config.rs`)

`RunConfig` is a serde enum tagged on `deployment`. Each per-deployment config
embeds the run-global specs and fixes which **pool roles** exist:

```jsonc
{ "deployment": "unified",
  "workload": { trace_files, duration_ms, run_to_end, request_rate },
  "io":       { log_dir, log_level, quiet, force_cache_build },
  "pools":    { "main": { placement, groups: [ { gpu, replicas, arch, worker } ] } } }
```

- A per-deployment `pools` map names the **roles** and binds each to a
  contract-class-typed `PoolSpec<Arch, Worker>` — unified's `main` is
  `PoolSpec<IterArchSel, IterWorkerSel>`, so an iter-wise arch can only pair an
  iter-wise worker (the type parameters enforce the contract).
- `model_config` and the model dims live **inside the arch tag**, not at the
  root; only `workload` / `io` are run-global.
- These types carry **no `#[serde(default)]` for valued params**: the launcher
  expands sweeps, fills defaults, and writes one fully-concrete config per run, so
  the Rust binary only ever reads a complete config. `#[derive(ParamStruct)]`
  emits the flat fields for the launcher's `list-params` schema.

## The build cascade (`UnifiedDeployment`)

`build` reads a `UnifiedConfig` and runs the lower layers in order:

1. Take the single homogeneous group of the `main` pool (heterogeneous `groups`
   is parse-only this round).
2. Load `ModelCfg` from the arch tag's `model_config` JSON, applying any
   `sim_num_layers` / `num_layers` truncation **before** config build.
3. Build a `WorkerConfig` (`attn_kv_bytes` from the worker tag's
   `attn_gpu_memory_gb`, plus hot-path logging controls from `io`). The worker is
   itself a tagged enum (`IterWorkerSel`): `Barebone` and `HpUnified` are wired
   for `unified`; `ChunkedPrefill` parses but bails (`not wired yet`), and the PD
   worker tags are rejected here because they belong to the `pd` deployment.
4. **Select the arch by its explicit tag** — the wired unified arms are
   `Llama3Dense`, `Llama3DenseTp`, `Llama3DpAttnTpFfn`, and
   `Qwen3MoeDpAttnEpFfn`, and `Glm52DsaMoe`. Dispatch is provider-first, *not* a
   `tp_size` dispatch. Each arm runs the L4 cascade
   `build_configs → resolve_configs → build` with its resolved parallel layout,
   producing a concrete model type `M`.
5. `assemble_flow::<M>` wraps the `Arc<M>` in a `UnifiedWorkerFactory` (threading
   `gpu_name` + `model.gpus_per_replica()` for the GPU inventory) and a
   `SimpleDpFlow`, erasing to `Box<dyn Flow>`.

That `Box<dyn Flow>` is the **single `dyn` erasure point** — each concrete model
monomorphizes its own flow, so the per-iter cost path stays `dyn`-free.

GLM-5.2 pairs with the existing `hp_unified` recipe. Its attention is TP1 local,
so the L4 model reports one attention-DP group per EP rank; `FullAttnKv` creates
one independent sticky KV partition for each group and `UnifiedIterExecution`
emits one matching `ArchGroupInput`. The worker shell, admission and selection
policy, request lifecycle, event ownership, and `SimpleDpFlow` remain the same as
for the existing multi-partition DP-attention/MoE families.

## The build cascade (`PdDeployment`)

`pd` has two fixed pool roles, `prefill` and `decode`, each with one homogeneous
group in the current implementation. The prefill pool must use the `pd_prefill`
worker and the decode pool must use `pd_decode`; both pools must point at the
same model config so the request/KV semantics line up across the handoff. Wired
arch pairings are:

- `llama3_dense_tp -> llama3_dense_tp`
- `llama3_dense_tp -> llama3_dp_attn_tp_ffn`

Each side builds its own concrete model (so TP/layout may differ), then
`assemble_pd_flow` builds a `PdFlow` with one `UnifiedWorkerFactory` per pool.

## The build cascade (`AfdDeployment`)

`afd` has an attention pool and an FFN pool. The wired arch pair is
`qwen3_attn_tp -> qwen3_ffn_moe`; the attention pool uses `disagg_attn` and the
FFN pool uses `disagg_ffn`.

The deployment validates the shared model config, builds each pool's
layer-wise L4 model under its own backend overrides, builds the profiled
attention↔FFN transfer cost once, and assembles an `AfdFlow`. Attention replicas
are independent sticky request/KV shards. FFN replicas receive complete
section tasks from the AFD pool controller; L6 owns the cross-pool layer barrier
and routing.

## Up / down

- **Above (consumer):** the L7 sim driver calls `build_flow` to get the `Flow`,
  and reads `workload` / `io` off the `RunConfig` to drive the run.
- **Below (assembled):** the L4 arch model build cascade, L5 worker recipes and
  `WorkerConfig`, and the matching L6 flow/pool controllers.
