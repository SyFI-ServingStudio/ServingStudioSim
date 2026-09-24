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
| `barebone` | `FullAttnKv` | `LocalPrefillDecodeAdmission<SessionStartOrder>` | `UnifiedIterExecution` | `IterBatchWorker` |
| `hp_unified` | `FullAttnKv` with N partitions | `LocalPrefillDecodeAdmission<SessionStartOrder>` with prefix affinity then RR misses | `UnifiedIterExecution` | `IterBatchWorker` |
| `chunked_prefill` | `FullAttnKv` with N partitions | `ChunkedPrefillAdmission<PendingOrder>` | `UnifiedIterExecution` | `IterBatchWorker` |
| `speculative` | `FullAttnKv` with N partitions | `ChunkedPrefillAdmission<PendingOrder, SpeculativeDecodeCompletion>` | `SpeculativeIterExecution` | `IterBatchWorker` |
| `pd_prefill` | `FullAttnKv` with held-KV ledger | `PrefillHandoffAdmission<SessionStartOrder>` | `UnifiedIterExecution` | `IterBatchWorker` |
| `pd_decode` | `FullAttnKv` | thin inline landed-request ingress | `UnifiedIterExecution` | `PullDecodeWorker` |
| `disagg_attn` | `FullAttnKv` | `FreshRequestSlotAdmission<SessionStartOrder>` | `AttentionLayerExecutionAdapter` | `SlotAttentionWorker` + private `AttentionSlotPipeline` |
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
- resolved request-prefill contexts and one retained-prefix cache per partition;
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

`SessionInput` remains one closed immutable request declaration: either
`Standalone`, or a session id plus its first trace arrival and declared reusable
prefix. Admission asks
`PrefixKv` to locate the best retained match before fallback placement, resolve
that partition's hit, gate the actual compute
`fresh + declared - resident`, and reserve the post-prefill context
`fresh + declared`.
An HP request waits when its retained owner is full; only a cold or evicted
session advances RR. Execution reads `ResolvedPrefillContext` from KV; request
progress records processed prefill work, not cache residency. At successful
admission, the resolved resident count is copied once to
`RequestTelemetry.prefix_cache_hit_tokens`; L7 writes that nullable observation
beside the immutable declaration in `request_slo` (`None` = never resolved,
`Some(0)` = resolved miss).

Retained prefix KV is evictable occupancy inside the same total attention
capacity as active, promised, and PD-held KV. `WorkerConfig::prefix_cache` is a
validated typed contract: `Opportunistic` uses all dynamically available slack
by default and may carry an optional retained-byte ceiling; `Disabled` is the
explicit no-reuse baseline. Active reservations shrink/evict retained entries
to preserve the physical capacity invariant. A hit transfers ownership out of
the cache until completion, so there is no inter-request sharing. PD prefill
returns held KV to the cache only after decode acknowledges the pull. The
implementation lives in `kv/prefix_cache.rs`; `kv/full_attn.rs` owns the
combined ledger and capacity gate.

The same combined accounting feeds `kv_snapshot`: `active_kv` remains total
committed attention KV, while `retained_prefix_kv` exposes its retained-cache
component. The sampler keeps both values from the same throttle-window peak
submit, so their difference is a valid non-prefix occupancy rather than a
difference between unrelated maxima.

Every prefix-capable KV worker also owns one sparse `PrefixCacheLogger`. It
writes `prefix_cache_event` from mutation receipts returned by
`kv/prefix_cache.rs`: admission-time ownership transfer (`activate`), exact
victim removal and reason (`evict`), and completion/PD-ack return (`retain`). A
worker-local sequence orders equal-time mutations, and entry/cache before/after
counts make the stream replayable. The logger is an observer of the same
`FullAttnKv`; it does not infer operations from `kv_snapshot` or maintain shadow
residency.

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

`DecodeCompletion` is a sub-axis of the chunked-prefill lifecycle, not a second
lifecycle: it owns only how far one resident decode moves per iteration.
`SingleTokenDecodeCompletion` is the ordinary one-row-in/one-token-out engine.
`SpeculativeDecodeCompletion` submits a fixed `draft_tokens + 1` verify window
per resident request and retires the target's own token plus the leading run of
accepted drafts, so its requests advance by different distances in the same
iteration. Membership, capacity, and batch composition stay with the lifecycle;
the query width it publishes is what sizes the mixed-iteration token budget and
what the execution adapter lowers into the L4 input.

Its acceptance draw is keyed by `(seed, request, tokens already emitted, draft
position)` rather than drawn from one stream, so a request accepts the same
chain no matter which requests it was batched with. Comparing two schedulers on
one trace therefore measures the schedulers, not a reshuffled random stream.

The lifecycle freezes `AdmissionCandidate` facts once at enqueue. Its
`conversation_start_time` is a session's first trace-declared arrival, or the
standalone request's own release. Production builders choose `SessionStartOrder`,
which ranks that key oldest-first and uses the per-admission monotonic enqueue
sequence for ties. `FifoOrder` and `ShortestJobFirst` remain available as
explicit composition seams.
Queues that represent active cadence state—PD pull/decode timelines, AFD slot
work, and FFN tasks—remain in their shells and are not selection policies.

The AFD FFN family has no admission component because L6 already gives it a
complete `FfnTask`.

## Execution axis

Execution owns the model, reusable input buffer shape, and `CostBuffers`:

- `UnifiedIterExecution` builds one `UnifiedArchInput` group per KV partition.
  Multi-partition workers also retain the exact per-partition token vector for
  ragged EP communication; the execution adapter derives it from those same
  groups without changing placement.
- `SpeculativeIterExecution` is its sibling for a model that implements
  `SpeculativeUnifiedModel` instead of `IterwiseUnifiedModel`. It builds a
  `SpeculativeArchInput`, whose decode side carries a `(kv_len, query_len)` pair
  per request rather than one KV length: a verify pass submits `draft_tokens + 1`
  rows per resident decode, so `decode_tokens` counts query rows and request
  cardinality is no longer recoverable from it. The width is a worker-lifetime
  constant because it selects a profiled kernel shape; how far a request actually
  advances after the verify belongs to the lifecycle's `DecodeCompletion`, not
  here.
  The complete verify window must fit within `max_model_len`. A boundary batch
  that requires fewer query rows fails explicitly; clipping only the context
  would discard resident keys and invalidate necessary-work conservation.
  vLLM can shorten these boundary batches, which this fixed-width model does
  not yet represent.
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
KV, admission, execution, and shell. The whole-iteration helper
(`workers/iter_build_essentials.rs`) takes plain model facts — GPUs per replica,
attention shard count, KV bytes per token, cost manifest — instead of a model
handle, so it names no L4 model trait and cannot select an execution adapter. It
returns the `CostBuffers`; the recipe wraps them.

Deployments and pool controllers only call those recipes:

- unified → `build_barebone_worker` / `build_hp_worker` /
  `build_chunked_prefill_worker`
- PD → `build_pd_prefill_worker` / `build_pd_decode_worker`
- AFD attention → `build_afd_attention_worker`
- AFD FFN → `build_afd_ffn_worker`

Message/event enums, pool behavior, and flow barriers remain L6 contracts; the
composition refactor does not create a second deployment layer.

`chunked_prefill` is a generic whole-iteration recipe, not a model-specific
scheduler. It uses the ordinary pending-order and partition-placement
contracts and exposes prompt chunks bounded by `max_batch_tokens`. Its
shell-owned iteration plan expresses whether resident decode shares that
iteration (`mix`) or waits behind a runnable prefill
(`separate-prefill-priority`); KV membership is unchanged in both cases. It
does not replay an observed DP rank or rewrite request shapes.

KV admission is a separate selector component (`kv_admission_policy`), shared
by `chunked_prefill` and `speculative`:

- `full-footprint` (default) reserves the complete request KV footprint once
  and never retracts.
- `bounded-future` admits against a near-future estimate. It then checks
  every decode step's physical allocation, sized at `min(step, remaining
  output)` per request, where the step is the verify width under speculation.
  On a shortfall it retracts a decode for recompute. A `mix` engine claims that
  step before admitting and admits nothing in a step that retracted, as vLLM
  does; a `separate-prefill-priority` engine checks only on its decode-only
  steps, as SGLang does.
- `decode_retraction_policy` selects the victim. `length` (SGLang) takes the
  fewest emitted tokens and requeues behind waiting requests. `fcfs` (vLLM)
  takes the most recently admitted request and requeues it at the head.
