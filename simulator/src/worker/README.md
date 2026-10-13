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
| `barebone` | `KvCacheStore` | `LocalPrefillDecodeAdmission<SessionStartOrder>` | `UnifiedIterExecution` | `IterBatchWorker` |
| `hp_unified` | `KvCacheStore` with N partitions | `LocalPrefillDecodeAdmission<SessionStartOrder>` with prefix affinity then RR misses | `UnifiedIterExecution` | `IterBatchWorker` |
| `chunked_prefill` | `KvCacheStore` with N partitions | `ChunkedPrefillAdmission<PendingOrder>` | `UnifiedIterExecution` | `IterBatchWorker` |
| `speculative` | `KvCacheStore` with N partitions | `ChunkedPrefillAdmission<PendingOrder, SpeculativeDecodeCompletion>` | `SpeculativeIterExecution` | `IterBatchWorker` |
| `pd_prefill` | `KvCacheStore` with held-KV ledger | `PrefillHandoffAdmission<SessionStartOrder>` | `UnifiedIterExecution` | `IterBatchWorker` |
| `pd_decode` | `KvCacheStore` less the reserved pull budget, no prefix tier | thin inline landed-request ingress | `UnifiedIterExecution` | `PullDecodeWorker` |
| `disagg_attn` | `KvCacheStore` | `FreshRequestSlotAdmission<SessionStartOrder>` | `AttentionLayerExecutionAdapter` | `SlotAttentionWorker` + private `AttentionSlotPipeline` |
| `disagg_ffn` | none | none; L6 sends complete tasks | `FfnSectionExecutionAdapter` | `BufferedFfnWorker` |
| `pipeline_chunked_prefill`, stage 0 | `KvCacheStore`, one partition, sized by the most constrained stage's `KvCacheLayout` | `PipelinedChunkedPrefillAdmission<PendingOrder>` | `UnifiedIterExecution` | `PipelineHeadWorker` |
| `pipeline_chunked_prefill`, stages 1..N-1 | none | none; L6 hands over microbatches | `UnifiedIterExecution` (as `PipelineStageExecution`) | `PipelineStageWorker` |

Hybrid archs (sliding-window, recurrent, or compressed state: Inkling,
Qwen3.6, GLM-5.3-Flash, DeepSeek-V4) use the same selectors and recipes; only
their `KvCacheLayout` differs.

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

`KvCacheStore` (`kv/kv_cache_store.rs`) is the only store. It owns:

- one private `ResidentPartitionState` per independent KV partition;
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
and capacity live in `kv/shared/resident_partition.rs`, while
`KvCacheStore::fits` directly owns the only strict capacity gate. Partition placement and the prefill
token gate belong to admission (`admission/placement.rs` and
`admission/token_budget.rs`); there is no mixed `admission_helpers` module or
one-variant capacity-policy seam.

Every charge is in KV units of the partition's `KvAccounting`
(`kv/shared/kv_accounting.rs`), resolved once at build from the arch's
`KvCacheLayout`; the store never sees layout parts. One unit is the gcd of the
parts' entry bytes, so a layout of one per-token part counts plain tokens. The
accounting answers what a running request holds (`held_while_running`), what a
waiting one needs free (`needed_to_admit`), and what a retained prefix entry
keeps (`entry_charge`). `current_kv` stays context tokens for the arch input;
charge never replaces it. The worker's flattened `KvAccountingSpec` picks
`kv_overhead` (`vllm` default: padding, state copies, engine blocks, in-flight
window tokens; `none`: model state only), `prefix_resume_points`,
`prefix_resume_interval`, and `kv_max_in_flight_tokens`. Rules and rationale:
`doc/detailed_design/L5.md` §3.1–3.2.

`SessionInput` remains one closed immutable request declaration: `Standalone`;
`PinnedPrefix`, a standalone request whose trace declares its prefix resident on
arrival (resolved by every prefix-capable KV as a full hit on any partition,
reserved and released with the request, never retained or evicted); or a
session id plus its first trace arrival and declared reusable prefix. Admission asks
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
returns held KV to the cache only after decode acknowledges the pull.

An entry resumes only at the `ResumePositions` its request left behind: any
length for per-token parts, but only where a sliding-window tail or recurrent
state was kept for the others (`prefix_resume_points`, default `prompt_end`
under `vllm`). A hit is the largest position at or below the declared prefix.
A running request's tails and states at positions it has already passed are
live checkpoints: like retained entries they are a soft, evictable tier that
never enters `fits` or `kv_usage`, and they win the slack over retained
entries. The implementation lives in `kv/shared/prefix_cache.rs`;
`kv/kv_cache_store.rs` owns the combined ledger and capacity gate.

The same combined accounting feeds `kv_snapshot`, in KV units: `active_kv`
remains total committed attention KV (live checkpoints included), while `retained_prefix_kv` exposes its retained-cache
component. The sampler keeps both values from the same throttle-window peak
submit, so their difference is a valid non-prefix occupancy rather than a
difference between unrelated maxima.

Every prefix-capable KV worker also owns one sparse `PrefixCacheLogger`. It
writes `prefix_cache_event` from mutation receipts returned by
`kv/shared/prefix_cache.rs`: admission-time ownership transfer (`activate`), exact
victim removal and reason (`evict`), and completion/PD-ack return (`retain`). A
worker-local sequence orders equal-time mutations, and entry/cache before/after
counts make the stream replayable. An `activate` row's `full_hit_tokens` is the
hit the entry would give if every position were resumable. The logger is an
observer of the same `KvCacheStore`; it does not infer operations from `kv_snapshot` or maintain shadow
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
`SpeculativeDecodeCompletion` submits a `draft_tokens + 1` verify window per
resident request (clipped at `max_model_len`, below) and retires the target's
own token plus the leading run of accepted drafts, so its requests advance by
different distances in the same iteration. Membership, capacity, and batch
composition stay with the lifecycle; the query width it publishes is what sizes
the mixed-iteration token budget and what the execution adapter lowers into the
L4 input.

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
  At the context limit the width follows vLLM's scheduler: a request with
  `resident` keys verifies `min(draft_tokens + 1, max_model_len - resident - 1)`
  rows (`SpeculativeDecodeInput::verify_query_len`), so the position it samples
  stays inside `max_model_len`. Its `kv_len` and `query_len` shrink together,
  keeping necessary-work conservation, and `DecodeCompletion` retires no more
  tokens than those rows. Arch models whose decode kernels take one uniform
  width price that rare ragged row on their prefill (varlen) path, which is
  also how vLLM leaves the uniform-decode graph. The mixed-iteration token
  budget still reserves the full width for it.
- `AttentionLayerExecutionAdapter` builds one slot's `AttnArchInput` and costs
  one attention layer.
- `FfnSectionExecutionAdapter` splits token counts across FFN DP groups and
  costs Bootstrap/Bridge/Terminal sections.

The shell treats `E::Input` as opaque. Transfer submission stays in the shell
because it is part of cadence overlap, not model math.

## Production cadence families

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

### 5. Pipeline stages: `workers/pipeline/`

A pipeline-parallel replica is one head and `depth - 1` followers, one GPU each.
`PipelineHeadWorker` owns admission and KV for the whole pipeline and stage 0's
compute. It forms a microbatch whenever stage 0 is idle and fewer than `depth`
microbatches are in flight, and starts it at the exact time that became true
(stage 0's finish, an exit, or the tick a request arrived on), not on the tick
grid:

```text
form_microbatch → build input → commit_microbatch → stage-0 compute
  → MicrobatchLaunched … MicrobatchExit → complete_microbatch
```

`PipelinedChunkedPrefillAdmission` commits a chunk's prefill progress when it is
scheduled (vLLM V1), so the next microbatch can carry the same prompt's next
chunk while the previous one is on a later stage. Token emission, decode KV
advance, `Done`, and KV release wait for the exit, stamped with the exact exit
time. A decode step needs the previous token, so a running request has at most
one microbatch in flight; each microbatch takes the ready decodes first, then
started and fresh prompts, under one token budget. Ready decodes all join by
default (vLLM); `balance_decode_microbatches` caps them at
`ceil(resident decodes / depth)` per microbatch. Admission writes the chosen
decode subset into `IterBatchPlan`, and input lowering skips the resident decodes
outside it.

`PipelineStageWorker` has no KV/admission axes. It runs microbatches FIFO,
double-buffering one activation pull against one compute, at exact times derived
from the previous stage's `ready_at`. It stamps no request stage, so requests stay
located on their head.

## Construction and deployment wiring

Every concrete recipe is isolated in a `build_*_worker.rs` file. Repeated
allocation/capacity/sampler/cost setup lives in a family-neutral or
family-private essentials helper, while the recipe visibly selects its concrete
KV, admission, execution, and shell. The whole-iteration helper
(`workers/iter_build_essentials.rs`) takes plain model facts — GPUs per replica,
the resolved `KvAccounting`, partition count, cost manifest — instead of a model
handle, so it names no L4 model trait and cannot select an execution adapter. It
returns the `CostBuffers`; the recipe wraps them. The same file holds the KV
helpers every KV-owning recipe shares: `resolve_kv_accounting` (layout, attention
shard count and worker options to `KvAccounting`), `kv_cache_store`, and
`prefill_chunk_end_quantum`. PD decode sets its pull budget aside with
`KvAccounting::reserving`; the pipeline head resolves the most constrained
stage's layout (`pipeline_kv_cache_layout`).

Deployments and pool controllers only call those recipes:

- unified → `build_barebone_worker` / `build_hp_worker` /
  `build_chunked_prefill_worker` / `build_speculative_worker`
- PD → `build_pd_prefill_worker` / `build_pd_decode_worker`
- AFD attention → `build_afd_attention_worker`
- AFD FFN → `build_afd_ffn_worker`
- PP → `build_pipeline_head_worker` / `build_pipeline_stage_worker`

Message/event enums, pool behavior, and flow barriers remain L6 contracts; the
composition refactor does not create a second deployment layer.

`chunked_prefill` is a generic whole-iteration recipe, not a model-specific
scheduler. It uses the ordinary pending-order and partition-placement
contracts and exposes prompt chunks bounded by `max_batch_tokens`. Its
shell-owned iteration plan expresses whether resident decode shares that
iteration (`mix`) or waits behind a runnable prefill
(`separate-prefill-priority`); KV membership is unchanged in both cases. It
does not replay an observed DP rank or rewrite request shapes. An optional
chunk-end alignment (`with_chunk_end_alignment`) ports vLLM's Mamba `align`
split: non-final chunks end on recurrent-state checkpoint boundaries, and the
budget a clipped chunk leaves may start further prompts.
`prefill_chunk_end_quantum` sets its block to the layout's engine block when
the layout keeps a resumable per-request state, prefix caching is on, and
`prefill_chunk_alignment` resolves to `checkpoint`; unset follows
`kv_overhead` (`vllm` aligns, `none` does not), and `plain` gives
`min(remaining, budget)` chunks. The `speculative` recipe applies the same
block; when its arch's drafter prunes the last matching prefix block
(`SpeculativeUnifiedModel::prunes_last_prefix_block`, vLLM's
`use_eagle_block_drop` for MTP/EAGLE/DFlash), the prompt's last checkpoint, and
therefore the stop no chunk may run past, sits one block earlier. Only that
chunk-end stop reads the hook; prefix-hit lengths do not yet drop the block.

With several attention partitions (an attention-DP arch), `dp_placement`
chooses where a cold request lands. `round-robin` (default) rotates blindly;
`vllm-least-loaded` ports vLLM's DP load balancer
(`DPLBAsyncMPClient.get_core_engine_for_request`): lowest `waiting + running`,
waiting requests penalised by `6 * max(0, kv_usage - 0.5)` each, scan start
rotated per placement. Retained-prefix affinity still wins over both.

With several attention partitions (an attention-DP arch), `dp_placement`
chooses where a cold request lands. `round-robin` (default) rotates blindly;
`vllm-least-loaded` ports vLLM's DP load balancer
(`DPLBAsyncMPClient.get_core_engine_for_request`): lowest `waiting + running`,
waiting requests penalised by `6 * max(0, kv_usage - 0.5)` each, scan start
rotated per placement. Retained-prefix affinity still wins over both.

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
