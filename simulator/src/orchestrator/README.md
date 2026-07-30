# L6 Orchestrator — pools and deployment flow

`orchestrator/` owns routing and cross-worker protocol state. It drives typed L5
workers but does not select their KV/admission/execution composition; deployment
code passes family-local worker build recipes.

The canonical layer contract is `doc/detailed_design/L6.md`; this README is the
production source-tree guide.

## Read the tree in this order

1. `mod.rs` — the L7-facing `Flow` contract.
2. `common.rs` — deployment actions and worker factory contracts.
3. `config.rs` — pool/group topology types.
4. `impls/simple_dp.rs` — reusable single-pool controller.
5. `impls/pd.rs` — prefill/decode flow.
6. `impls/afd_attn_pool.rs` — AFD placement, aggregation, scatter, barriers.
7. `impls/afd_ffn_pool.rs` — AFD FFN task routing.
8. `impls/afd.rs` — AFD flow ordering.

## Topology vocabulary

```text
deployment → pool(role) → homogeneous group → worker replica
```

- unified: `main`
- PD: `prefill`, `decode`
- AFD: `attn`, `ffn`

`PoolSpec<Arch, Worker>` and `GroupSpec<Arch, Worker>` keep the L4/L5 contract
class typed. Current deployment builders accept one homogeneous group per pool;
the config shape retains a group list for future heterogeneous routing.

## L7-facing surface

```rust
pub trait Flow {
    fn on_arrival(&mut self, request: Request);
    fn tick(&mut self, now: Time) -> Vec<OrchAction>;
    fn cluster(&self) -> &SharedGpuCluster;
}
```

`Flow` is the only object L7 holds. `on_arrival` records request facts before
routing the id. `tick` advances pool protocols and currently surfaces
`OrchAction::Complete`. `cluster` exposes the shared GPU registry and transfer
oracle for run metadata/logging.

The concrete model/worker/controller stack is erased to `Box<dyn Flow>` only in
the deployment layer. L5 cost paths remain statically dispatched.

## Shared cluster and worker construction

`GpuCluster` is canonically defined under `worker/gpu_cluster.rs` because workers
register and use its runtime resources. L6 owns the shared handle across pools.
It contains:

- dense global GPU ids and ownership facts;
- registered communication groups;
- profiled/analytic transfer timing and network logging;
- registered KV token capacities.

Every L5 build recipe allocates its real GPU block using the L4 model's
`gpus_per_replica()`. KV-owning recipes register partition capacity; transfer
workers register communication groups. This makes worker construction the single
source of topology truth.

For iter-wise pools, `UnifiedWorkerFactory` holds the shared model, request store,
worker config, logging path, GPU name, pool tag, and the selected
`build_*_worker` function. `WorkerFactory` lets a specialized factory use the
same controller without teaching L6 a new L5 composition.

## Unified (`simple_dp.rs`)

`SimpleDpPoolController` owns one pool's workers. `admit` selects by
`DpPlacementPolicy` (`LeastQueued` or `RoundRobin`), while `tick_collect` enters
only workers whose stored wakeup is due.

`SimpleDpFlow` inserts arrivals, admits them to the controller, and maps worker
completion to `OrchAction::Complete`.

## PD (`pd.rs`)

`PdFlow` composes two simple-DP controllers:

```text
prefill PrefillDone
  → decode Handoff
  → KV pull
  → source-specific PullComplete/ReleaseKv ack
  → decode RequestComplete
```

The producer is ticked first so same-time handoffs can be consumed without an
extra global tick. Pool tags are preserved in cost logs and cluster metadata.

## AFD (`afd_*.rs`)

AFD has a different L6 protocol and therefore dedicated controllers.

`AfdAttnPoolController` owns:

- least-estimated-peak-KV placement, sticky for the request lifetime;
- one start barrier and one layer barrier per slot;
- aggregation of all attention shards into `FfnTask`s;
- FFN-output scatter back to every attention worker;
- terminal KV release and completion.

Once a slot becomes active, every attention worker reports every layer,
including a zero-token shard. The barrier divisor stays `workers.len()`, and
`SlotFlushed` advances all workers together.

`AfdFfnPoolController` accepts complete section tasks. It round-robins independent
tasks across FFN replicas, ticks due workers, and returns `SectionReady` or
`IterComplete` events.

`AfdFlow` drives attention, forwards aggregated tasks to FFN, applies FFN events
back to attention, and surfaces completed requests. Cross-pool transfers use the
same shared `GpuCluster`.

## Ownership rules

- L6 owns routing, barriers, and protocol message movement.
- L5 owns queues that are local lifecycle/cadence state and all KV mutation.
- Deployment owns legal arch/worker pairing and concrete assembly.
- L7 owns the global tick loop and sees no pool/worker internals.
