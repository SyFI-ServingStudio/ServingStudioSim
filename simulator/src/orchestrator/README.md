# L6 Orchestrator — pools & deployment flow

The orchestrator models the **serving topology**: how a deployment's requests are
routed across pools of workers. It owns request routing and the run's GPU
inventory, and defers the actual per-iteration cost + request lifecycle to the L5
workers. To read this layer you first need its abstraction.

This is the practical, code-matching reference; the code is the ground truth.
For deeper design intent see `docs/detailed_design/L6/design.md`.

## The abstraction: deployment → pool → group → worker

A run's topology is a four-level hierarchy. The orchestrator owns the top two
levels; L5/L4 own the bottom two.

- **Deployment** — the serving shape (unified / PD / AFD / …). It fixes which
  *roles* exist. One deployment = one L6b `Flow`.
- **Pool** — **the set of workers exposing one L6-facing role.** A pool is the
  deployment's *role boundary*, **not** a GPU-type or Rust-worker-type boundary:
  one logical role may mix GPU models inside a single pool. Unified has one pool
  (`main`); AFD has `attn` + `ffn` (`AfdPools`); PD has `prefill` + `decode`
  (`PdPools`) — see `deployment/config.rs`. A role is split into multiple pools only when the flow needs to
  *route* between them explicitly (fast vs. cheap tier, expert shard, …).
- **Group** — one *homogeneous* slice of a pool: a single GPU type + a `replicas`
  count (the data-parallel fan-out) + an arch (L4) + worker (L5) provider. A
  homogeneous pool is one group; a heterogeneous pool lists several (parse-only
  this round, `build()` takes one).
- **Worker** — one model replica's execution (an L5 `BareboneWorker`), running on
  `gpus_per_worker` GPUs.

The **pool boundary is L6's only abstraction**: no code outside L6 ever holds a
`&Worker` (Invariant 1). Everything above the worker is routing; everything at or
below it is cost + lifecycle.

## Two halves, split at the pool boundary

- **L6a — pool-local orchestration** (*inside* one pool): build and hold the
  worker instances, pick a worker (placement / load balance), tick them, and
  collect self-tagged `WorkerEvent`s into a caller-owned sink, translating them
  to `PoolEvent`s for L6b. Workers do not keep a drainable outbox on the current
  iter-wise path. L6a never decides deployment-level phase jumps.
- **L6b — inter-pool deployment flow** (*between* pools): consume external
  arrivals and L6a's `PoolEvent`s, decide which pool gets work next, and hold any
  deployment-level phase/priority state. **L6b is the only object L7 holds**,
  through the `Flow` trait:

```rust
pub trait Flow {
    fn on_arrival(&mut self, req: Request);           // insert facts into the shared store, admit the id
    fn tick(&mut self, now: Time) -> Vec<OrchAction>; // drive pools; surface deployment actions
    fn inventory(&self) -> &GpuInventory;             // GPUs occupied → L7 writes raw/run_meta.json
}
```

`OrchAction` is the deployment-level result L7 consumes (today only
`Complete { req }`). The concrete `Flow` for a run is **chosen and built by the
`deployment` layer** from the run config — this module supplies the trait and the
implementations.

## Directory map

```
mod.rs        The Flow trait + re-exports. Nothing else (the L7-facing surface).
common.rs     Shared L6 vocabulary, kept out of mod.rs:
                OrchAction / PoolEvent  — the deployment-action + pool-event enums
                GpuInfo / GpuInventory  — the run's flat GPU registry
                UnifiedWorkerFactory    — stamps identical workers for a pool
config.rs     Pool + group topology: PoolSpec / GroupSpec / PlacementPolicy.
impls/        Concrete deployments.
  simple_dp.rs  SimpleDpPoolController (L6a) + SimpleDpFlow (L6b) in one file.
  pd.rs         PdFlow: prefill pool -> decode pool handoff using two simple-DP pools.
```

## `simple_dp` — the one wired deployment

One pool (`main`) of identical unified workers, one homogeneous group. L6a and
L6b stay two structs even though the file is small:

- **`SimpleDpPoolController` (L6a)** owns the pool's iter-wise workers.
  `admit(rid)` picks a worker by `DpPlacementPolicy` (`LeastQueued` /
  `RoundRobin`) and enqueues it. `tick_collect(now, events)` first checks each
  worker's next wakeup and only enters due workers; those workers push self-tagged
  `WorkerEvent`s into `events`. The flow maps those events through
  `to_pool_event(pool, event)`. Note `DpPlacementPolicy` (`simple_dp.rs`, the
  impl's own enum) and config's `PlacementPolicy` (`config.rs`, the wire/launcher
  policy) are two distinct types, bridged by `placement_into` in the `deployment`
  layer.
- **`SimpleDpFlow` (L6b)** implements `Flow`. `on_arrival` inserts the request into
  the shared `RequestStore` then admits its id to the pool; `tick` ticks the pool
  and maps each `PoolEvent` to an `OrchAction::Complete`.

## Worker stamping & the GPU inventory

A pool's workers are built by a **`UnifiedWorkerFactory`**: identical unified
workers, each sharing one `Arc<Model>` and the one `SharedRequests` handle, with
the run's `log_dir` (for cost_log) and a `WorkerConfig`. The factory **carries**
`gpu_name` / `gpus_per_worker` (read off the L4 `ParallelCfg` by the deployment) —
it doesn't derive them; it threads them so L6 can build the run's
**`GpuInventory`**: a flat `Vec<GpuInfo>` (`id`, `name`, `pool`, `worker_id`).
`allocate(pool, worker, n, name)` appends a contiguous id block per worker, so ids
stay dense and **globally unique across pools**. The **flow owns** the inventory
(`SimpleDpFlow::new` builds it); the controller (`SimpleDpPoolController::new`)
takes it by `&mut` and populates it via `allocate` — it does *not* own it. That is
exactly what keeps ids globally unique: a future multi-pool deployment threads the
same `&mut` inventory through each pool's `new`, so ids keep counting up instead of
each pool restarting at zero. As one combined run-level artifact it is the
**reporting source** L7 serializes to `raw/run_meta.json` for the analyzer's
per-GPU normalization.

## `pd` — prefill/decode disaggregation

`PdFlow` composes two `SimpleDpPoolController`s: a `prefill` pool of
`PdPrefillWorker`s and a `decode` pool of `PdDecodeWorker`s. The flow ticks the
producer first, maps `PrefillDone` events to same-tick decode admissions, then
ticks the decode pool and maps decode completions to `OrchAction::Complete`. The
pool tags (`prefill` / `decode`) are also threaded into cost-log file names and
manifests so `(pool_tag, worker_id)` is the stable per-worker key.

## Topology config (`config.rs`)

The on-disk shape of the hierarchy above. A pool is always `{ placement,
groups: [...] }` — never inlined. `PoolSpec<Arch, Worker>` / `GroupSpec<Arch,
Worker>` are **generic over the provider types**, and those type parameters *are*
the contract constraint — a layer-wise arch cannot be paired with an iter-wise
worker. `placement` is the only pool-level policy; everything provider-specific
lives on the arch/worker tags. `#[derive(ParamStruct)]` emits the flat pool/group
fields (`placement`, `gpu`, `replicas`) for the launcher schema; `groups` /
`arch` / `worker` are `#[param(skip)]` nested sub-trees.

## Up / down

- **Above (consumer):** the L7 sim driver calls `Flow::{on_arrival, tick,
  inventory}`. The `deployment` layer constructs the concrete `Flow`.
- **Below (driven):** L5 workers (`enqueue` / `tick` / `status`)
  and, through the factory, the L4 model (`Arc<M>`).
