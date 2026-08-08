# L5 Worker — composition and cadence

L5 models one serving worker replica. A worker may own one GPU or a multi-GPU
parallel group. It turns L6 messages into model-cost queries, advances request
and KV state, emits typed events, and reports its next wakeup.

The canonical ownership, compatibility, and extension rules live in
`doc/detailed_design/L5.md`; this README is the production source-tree guide.

The production implementation is single-track. There is no old/v2 runtime
selector: deployments construct the composed types under `workers/` directly.
L6 still sees only `IterWorker` (plus the narrow AFD capabilities).

## Read the tree in this order

1. `iter_worker.rs` — the stable L6-facing protocol.
2. `shared/` — facts shared by components (`WorkerContext`, `AdvanceScope`).
3. `kv/` — resource ownership and request membership.
4. `admission/` — lifecycle and selection.
5. `execution/` — model input construction and cost evaluation.
6. `workers/` — cadence shells and one `build_*_worker.rs` recipe per concrete
   production worker.

The organizing equation is:

```text
Worker = <KV, Admission<Selection>, Execution> × Shell(cadence)
```

The three components are reusable axes. The shell is deliberately concrete:
whole-iteration, pull+decode, layer-slot pipeline, and FFN double buffer have
different timelines and should not be hidden behind one giant FSM abstraction.

## Production compositions

| Worker selector | KV | Admission | Execution | Shell |
|---|---|---|---|---|
| `barebone` | `FullAttnKv` | `LocalPrefillDecodeAdmission<FifoOrder>` | `UnifiedIterExecution` | `IterBatchWorker` |
| `hp_unified` | `FullAttnKv` with N partitions | `LocalPrefillDecodeAdmission<FifoOrder>` with prefix affinity then RR misses | `UnifiedIterExecution` | `IterBatchWorker` |
| `pd_prefill` | `FullAttnKv` with held-KV ledger | `PrefillHandoffAdmission<FifoOrder>` | `UnifiedIterExecution` | `IterBatchWorker` |
| `pd_decode` | `FullAttnKv` | thin inline landed-request ingress | `UnifiedIterExecution` | `PullDecodeWorker` |
| `disagg_attn` | `FullAttnKv` | `FreshRequestSlotAdmission<FifoOrder>` | `AttentionLayerExecutionAdapter` | `SlotAttentionWorker` + private `AttentionSlotPipeline` |
| `disagg_ffn` | none | none; L6 sends complete tasks | `FfnSectionExecutionAdapter` | `BufferedFfnWorker` |

Aliases such as `BareboneWorker<M>` and `DisaggAttnWorker<M>` name these concrete
generic compositions. They are not wrapper runtimes.

## L6-facing surface

Every worker implements:

```rust
trait IterWorker {
    type Msg;
    type Event;

    fn id(&self) -> WorkerId;
    fn enqueue(&mut self, msg: Self::Msg);
    fn tick(&mut self, now: Time, events: &mut Vec<Self::Event>) -> Option<Time>;
    fn status(&self) -> WorkerStatus;
}
```

`tick` advances to a local fixpoint at `now` and returns the next useful
timestamp. `None` means quiescent. Events are pushed into the L6-owned sink and
carry their source worker id.

AFD adds only protocol-specific views:

- `AfdAttnWorker::estimated_peak_kv` for sticky least-KV placement.
- `AfdFfnWorker` as a type-level guarantee of the FFN task/event pair.

Construction remains outside these traits. L6 passes family-local
`build_*_worker` recipes; it does not select KV/admission/execution components
itself.

## KV axis

`KvStore` owns the common resource lifecycle:

```text
footprint → fits → reserve → commit_resident → advance → release
```

`FullAttnKv` owns:

- one private `FullAttnPartitionState` per independent KV partition;
- the promised reservation ledger;
- the PD-prefill held-KV ledger;
- runtime session-prefix resolution and one retained-prefix cache per partition;
- worker-local retained-prefix partition lookup;
- sticky request→partition ownership;
- strict admission and KV sampling.

Family capabilities expose only the views their cadence needs:

- `IterWorkerKv` — prefill admits and live decode membership.
- `PrefixKv` — resolved resident/recomputed prefix work and retained-session release.
- `SlotPipelineKv` — per-request current KV, reservation membership, and
  projected peak.
- `HandoffKv` — hold/complete semantics for PD prefill.

Iteration and slot input builders consume borrowed membership visitors/slices,
so composition does not require cloning the live batch on the hot path.

The partition-local implementation is split by ownership: resident decode state
and capacity live in `kv/full_attn_partition.rs`, while `FullAttnKv::fits`
directly owns the only strict capacity gate. Partition placement and the prefill
token gate belong to admission (`admission/placement.rs` and
`admission/token_budget.rs`); there is no mixed `admission_helpers` module or
one-variant capacity-policy seam.

`PrefixInput` remains an immutable request declaration. Admission asks
`PrefixKv` to locate the best retained match before fallback placement, resolve
that partition's hit, gate the actual compute
`fresh + declared - resident`, and reserve the full context `fresh + declared`.
An HP request waits when its retained owner is full; only a cold or evicted
session advances RR. Execution reads the resolution from KV; request progress
records processed prefill work, not cache residency.

Retained prefix KV is evictable occupancy inside the same total attention
capacity as active, promised, and PD-held KV. `prefix_cache_capacity_bytes` is
only a cache ceiling: active reservations shrink/evict retained entries to
preserve the physical capacity invariant. A hit transfers ownership out of the
cache until completion, so there is no inter-request sharing. PD prefill returns
held KV to the cache only after decode acknowledges the pull. The implementation
lives in `kv/prefix_cache.rs`; `kv/full_attn.rs` owns the combined ledger and
capacity gate.

## Admission axis

Admission owns lifecycle gates, not pending membership, KV arithmetic, or model
input. Its `PendingOrderPolicy` type parameter owns the pending set and exposes
one stable head through `peek`/`pop`:

- `LocalPrefillDecodeAdmission<P>` admits local prefills and completes the
  prefill→decode→done lifecycle.
- `PrefillHandoffAdmission<P>` completes prefill, holds KV until decode acks the
  pull, and emits the handoff.
- `FreshRequestSlotAdmission<P>` implements AFD's two-level admission: enqueue
  into `P` first, then reserve fitting heads and hand them to the slot shell.

The lifecycle freezes `AdmissionCandidate` facts once at enqueue. `FifoOrder`
uses `VecDeque`; `ShortestJobFirst` maintains a `BinaryHeap` incrementally and
uses a per-admission monotonic enqueue sequence for deterministic equal-work
ties. The current production builders explicitly choose `FifoOrder`; the SJF
type is available as a composition seam but is not a deployment selector yet.
Queues that represent active cadence state—PD pull/decode timelines, AFD slot
work, and FFN tasks—remain in their shells and are not selection policies.

The AFD FFN family has no admission component because L6 already gives it a
complete `FfnTask`.

## Execution axis

Execution owns the model, reusable input buffer shape, and `CostBuffers`:

- `UnifiedIterExecution` builds one `UnifiedArchInput` group per KV partition.
- `AttentionLayerExecutionAdapter` builds one slot's `AttnArchInput` and costs
  one attention layer.
- `FfnSectionExecutionAdapter` splits token counts across FFN DP groups and
  costs Bootstrap/Bridge/Terminal sections.

The shell treats `E::Input` as opaque. Transfer submission stays in the shell
because it is part of cadence overlap, not model math.

## Four production cadence families

### 1. Whole iteration: `workers/iter/`

`IterBatchWorker` runs:

```text
form_batch → build/evaluate iteration → complete_iteration → repeat
```

Barebone, HP, and PD-prefill differ only in their build recipe. HP changes the
number of KV partitions and placement; PD-prefill changes admission and adds a
send comm group.

### 2. Pull plus decode: `workers/pd_decode/`

`PullDecodeWorker` coordinates two timelines:

- one serialized KV pull with a bounded landed/in-transit backlog;
- one whole-iteration decode FSM.

A request enters the decode partition only after its prompt KV lands. Pull
completion acks the exact prefill worker that still holds the source capacity.

### 3. Layer-slot attention: `workers/afd_attention/`

`SlotAttentionWorker` combines fresh-request admission with a private
three-slot pipeline. Each slot walks:

```text
Wait → WaitComplete → Pull → PullComplete → Compute → AwaitFlush
```

At most one slot pulls and one computes. `AwaitFlush` is released by L6's
all-worker layer barrier, keeping empty shards in lockstep. The KV partition is
shared across slots; slot identity is cadence, not a KV partition.

### 4. Buffered FFN: `workers/afd_ffn/`

`BufferedFfnWorker` has no KV/admission axes. It overlaps one incoming gather
with one current section compute:

```text
incoming → pulling_task → computing_task → SectionReady / IterComplete
```

Terminal owns token emission and completion but preserves the sticky attention
worker as the request's recorded location.

## Construction and deployment wiring

Every concrete recipe is isolated in a `build_*_worker.rs` file. Repeated
allocation/capacity/sampler/cost setup lives in a family-neutral or
family-private essentials helper, while the recipe visibly selects its concrete
KV, admission, execution, and shell.

Deployments and pool controllers only call those recipes:

- unified → `build_barebone_worker` / `build_hp_worker`
- PD → `build_pd_prefill_worker` / `build_pd_decode_worker`
- AFD attention → `build_afd_attention_worker`
- AFD FFN → `build_afd_ffn_worker`

Message/event enums, pool behavior, and flow barriers remain L6 contracts; the
composition refactor does not create a second deployment layer.
