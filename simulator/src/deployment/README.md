# Deployment — config → a runnable Flow

The deployment layer is the assembly glue: it deserializes a **structured run
config** and builds the concrete L6b `Flow` the sim driver ticks. The
`deployment` tag is the top-level switch that fixes, in one choice, the
orchestrator (1:1), the pool topology, and each pool's contract class (which arch
+ worker types are legal). It composes the lower layers; it contains no cost or
scheduling logic of its own.

This is the practical, code-matching reference; the code is the ground truth.
For deeper design intent see `docs/new-interface-design.md` (§2/§4/§6/§10) and
`docs/detailed_design/L6/`.

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
| `pd`, `afd` | parse (so `list-params` advertises them), then error cleanly until wired |

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
3. Build a `WorkerConfig` (today only `attn_kv_bytes`, from the worker tag's
   `attn_gpu_memory_gb`). The worker is itself a tagged enum (`IterWorkerSel`):
   `Barebone` is **wired**, `ChunkedPrefill` is parsed but bails (`not wired
   yet`).
4. **Select the arch by its explicit tag** — `IterArchSel` has **three** arms,
   `Llama3Dense` / `Llama3DenseTp` (both **wired**) and `DeepseekMoe` (parsed but
   bails). Dispatch is provider-first, *not* a `tp_size` dispatch; `tp_size`
   exists only on the TP tag. Each wired arm runs the L4 cascade `build_configs →
   resolve_configs → build` with the right `ParallelCfg`, producing a concrete
   model type `M`.
5. `assemble_flow::<M>` wraps the `Arc<M>` in a `UnifiedWorkerFactory` (threading
   `gpu_name` + `model.gpus_per_replica()` for the GPU inventory) and a
   `SimpleDpFlow`, erasing to `Box<dyn Flow>`.

That `Box<dyn Flow>` is the **single `dyn` erasure point** — each concrete model
monomorphizes its own flow, so the per-iter cost path stays `dyn`-free.

## Up / down

- **Above (consumer):** the L7 sim driver calls `build_flow` to get the `Flow`,
  and reads `workload` / `io` off the `RunConfig` to drive the run.
- **Below (assembled):** the L4 arch model build cascade, the L5 `WorkerConfig`,
  and the L6 orchestrator (`SimpleDpFlow` + `UnifiedWorkerFactory`).
