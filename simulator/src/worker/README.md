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
| `pipeline_chunked_prefill`, stage 0 | `FullAttnKv`, one partition, sized by the stage with the most KV per token | `PipelinedChunkedPrefillAdmission<PendingOrder>` | `UnifiedIterExecution` | `PipelineHeadWorker` |
| `pipeline_chunked_prefill`, stages 1..N-1 | none | none; L6 hands over microbatches | `UnifiedIterExecution` (as `PipelineStageExecution`) | `PipelineStageWorker` |

Hybrid recurrent + attention archs (Qwen3.6 local, GLM-5.3-Flash) swap in
`HybridGdnKv` under `barebone` (`Qwen36HybridWorker`) and `chunked_prefill`
(`HybridChunkedPrefillWorker`).

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

Prefill fills each microbatch greedily up to `max_batch_tokens` by default
(`microbatch_split: greedy`, vLLM). `microbatch_split: even` sizes it to
`clamp(ceil((pending + in-flight prefill tokens) / depth), min_microbatch_tokens,
max_batch_tokens)`: pending is started prompts' unscheduled tokens plus the
policy's `queued_prompt_tokens()`, in-flight is the prefill of formed microbatches
that have not exited. One prompt into an empty pipeline therefore runs as `depth`
equal slices. The split never holds a microbatch back, and
`min_microbatch_tokens` is rejected unless the split is `even`.

Three knobs compose with either split. `long_prefill_token_threshold` caps any
one request's chunk per microbatch, as vLLM's knob of that name does.
`load_budget_low_tokens` (not vLLM) runs microbatches at that budget while the
prefill backlog (queued, started and in-flight tokens) is at most
`load_budget_backlog_lo_tokens`, rising linearly to `max_batch_tokens` at
`load_budget_backlog_hi_tokens`; both bounds are required with the floor.
`srpt` (not vLLM) orders prefill by remaining tokens across started and queued
prompts instead of started first; pair it with `pending_order:
shortest-prefill-first` so the queue offers its shortest prompt first. The
budget (`LoadBudget`) and the srpt order (`admission/srpt.rs`) are shared with
`chunked_prefill`, which applies both per attention DP rank.
`force_schedule_after_ms` (not vLLM; also on `chunked_prefill`) serves a request
that has waited that long since arrival before every other one, oldest first:
queued ones leave the pending order for an overdue queue admission takes first,
and with `srpt` overdue started prompts also go first. Shortest-first alone can
leave a long prompt waiting for hours under a saturated closed loop
(`admission/overdue.rs`).

A hybrid head (`HybridGdnKv`) follows `prefill_chunk_alignment`. `checkpoint` is
vLLM's Mamba `align` mode: non-final chunks end on the KV block and prefix hits
floor to it, one state per block. `plain` chunks plainly and keeps one state per
retained session, at its context's end: a hit is the whole context when the
prompt extends it, and nothing when the prompt ends inside it. The tiers follow
the same rule (`ResumePoint` in `kv/shared/prefix_cache.rs`).
`recurrent_state_checkpoints: k` (not vLLM; default 1, needs `plain`) keeps the
latest k round-end states per retained context: its end plus earlier prompt ends
and handed-back decode ends (a hand-back keeps the prompt-end state its prefill
wrote); fewer only when the session has had fewer passes or a cut removed some.
Every kept state sits at a prefix of the entry's context, so only states within
a new round's declared prefix carry over (a round that does not extend the
context cuts it there, keeping the states at or before the cut). A prompt ending
inside the context resumes at the latest state at or below it. One rule holds in
HBM, DRAM and SSD: a retained context is one unit, its KV plus its states,
charged `tokens + states x state` and evicted whole by LRU, never a state alone.
Whatever moves it between tiers moves all its states: an HBM eviction into a
victim tier, a DRAM -> SSD demotion. A read into HBM moves the KV and only the
states that survive its round: the state it resumes from and the latest earlier
ones still among the session's latest k after the round adds its own (its
prefill's end, and with `external_decode` the hand-back's end), i.e.
`max(k - added, 1)` states; k = 2 with a hand-back moves the resume state
alone, k = 4 one earlier state beside it. The rest stay in an inclusive tier's
entry and are dropped with the entry an exclusive DRAM releases. A request that takes an HBM entry
carries its states (still charged to HBM's retained tier, not evictable) and
hands them back with its context, which appends its new end state and drops the
oldest beyond k.

Behind HBM's retained tier a head can keep host DRAM and SSD tiers
(`dram_tier_gb`, `ssd_tier_gb`, per GPU, with read bandwidths). They are
write-through LRUs of each session's latest context, plus one state for a hybrid
model (`kv/shared/prefix_tiers.rs`). With `dram_tier_writes: hbm-evictions`
(not vLLM) DRAM is instead HBM's victim tier: the KV store logs each context its
prefix cache evicts or cannot keep (`HbmEvictionLog`), DRAM writes what its entry
lacks before the next tier lookup, a read back keeps the DRAM copy, and a
session's next admitted round cuts that copy to what it shares;
`hbm-evictions-exclusive` instead releases the entry a read moves into HBM, so
DRAM never duplicates HBM and HBM evicting the context again writes it whole. With
`ssd_tier_writes: dram-evictions` (not vLLM) SSD is likewise DRAM's victim tier
(written back): it takes what DRAM evicts or cannot hold, as what its entry
lacks, and only refreshes an entry that already holds it; an SSD read keeps the
SSD copy and copies into DRAM only if DRAM is write-through. The admission owns them
(`admission/prefix_fetch.rs`) and looks a session up when its request reaches the
head of the queue, as vLLM's KV connector does at scheduling. If a tier holds
more than HBM does, floored to HBM's hit quantum, the request leaves the queue
while that tier's FIFO channel reads just the difference. As vLLM allocates an
async load's blocks before it starts, the read starts only if the whole request
fits in HBM (`KvStore::fits`), and holds that footprint (`PrefixKv::hold_for_read`)
until admission; otherwise admission stops. A landed request waits in its
partition's landed queue, which admission takes before the pending order (vLLM's
skipped-waiting queue); there the context goes back into HBM (`restore_prefix`)
and the hold turns into the request's reservation. A lookup at arrival would miss
every context HBM evicts while the request queues. On the pipeline head,
`prefix_tier_max_read_wait_ms` (not vLLM) starts a read only while its tier would
begin it within that bound, so a long read queue does not sit on HBM. A request
past it is passed over by default (`prefix_tier_read_wait_policy: pass-over`),
holding nothing, and admission goes on to the ones behind; `block` stops
admission at it instead, so the queue behind waits too. Both admission
lifecycles take fresh prompts from the same sources in the same order (landed
reads, overdue requests, the pending order; `admission/fresh_source.rs`), and a
passed-over request goes back to the front of the queue it came from.
`prefix_tier_loading_area` (pipeline head, not vLLM) stages loads instead:
pending -> loading -> ready -> batch. That fraction of HBM's KV capacity is a
loading area the ledger always counts as held, so admission and retained
prefixes leave it free; before each admission a loading pass walks the overdue
and pending queues in order and starts reads while the area has room (a read
larger than the area starts once it is empty; a read that does not fit stops
its queue, so none jumps it), and admission takes only requests whose context
is in HBM, passing over the rest. With an SSD behind DRAM, the same fraction of
the DRAM tier is carved out as a DRAM area: an SSD read lands there, then hops
to HBM on DRAM's channel once the HBM area has room. Under this `held` rule
(`prefix_tier_loading: held`, the default) a request holds its area until
admission, so large reads that landed and wait for admission fill it and stall
the reads behind them.
`prefix_tier_loading: bounded` derives two separate bounds instead. (1) An
in-flight area per landing tier bounds the reads in progress: HBM's is what the
fastest tier's channel reads in one batch period, DRAM's (for SSD reads) what
the SSD channel reads in it, by Little's law (bytes in flight = channel rate x
residency, and a read waits at most one period behind the channel). The batch
period is the cost model's stage time of one `max_batch_tokens` prefill
microbatch from an empty context: stages are balanced and each holds one
microbatch, so the pipeline prefills `max_batch_tokens` per period. DRAM's
area is carved out of the DRAM tier; HBM's reserves nothing: a read holds its
request's whole footprint from its start (like an unstaged read), counted
against HBM like a running request's (retained prefixes make room), and starts
only if it fits, so any landed request can be admitted and landed data cannot
deadlock admission. A read gives its area back when it lands (an SSD read's
DRAM area when its hop lands) and a loading pass starts the next reads at once;
its data stays held in HBM until admission. (2) New reads start only while the
landed, unadmitted data plus the new read fits one in-flight area (a read
larger than the area starts when nothing has landed), so landed data stays
below two areas, and while the landed,
unadmitted requests' prefill work (their new tokens plus any declared prefix
past the hit) is below `pp_size x max_batch_tokens`, what fills the pipeline.
A landed request evicted before admission goes back to pending and is read
again (`rereads`); `force_schedule_after_ms` puts overdue requests first in the
loading pass too. `prefix_tiers_<tag>.json`'s `loading` records the rule and
the sizes it derived.
A tier is sized and read in the most loaded stage's bytes per token;
`prefix_tier_balanced_load` (not vLLM) uses the stages' mean instead. A hybrid
pipeline's entry adds one recurrent state on that same GPU, in real bytes (the
stage's own, or the stages' mean), not the HBM pool's padded state blocks; its
earlier round-end states are charged the same bytes each. A victim tier merges
an entry it receives into its own for the session (the longer context with the
latest k states of both), writing only what it lacks. A shortest-prefill-first order ranks a request by what HBM or a tier
holds. With `prefix_tier_warm_start`, a session request whose declared prefix
exceeds every context the run stored for it is a pre-run conversation, read
from the slowest tier (closed-loop traces that join sessions mid-life).
`prefix_tiers_w<id>.csv` / `.json` record each admitted session request and the
tiers' totals. With `external_decode`, a round completes at its first token. The head
then restores its context with every output but the last (whose KV the next
round computes) as if a decode instance handed it back. The trace carries the
decode time in its tool waits (`trace/session_decode_wait.py`). The `pp` flow
keeps each session on the replica its first round went to.

The hybrid `chunked_prefill` worker (DP attention included) takes the same
`dram_tier_gb` / `dram_tier_writes` / `ssd_tier_gb` / `ssd_tier_writes` / `prefix_tier_warm_start` /
`prefix_tier_max_read_wait_ms` / `prefix_tier_read_wait_policy` /
`external_decode` knobs (a read-wait bound needs a tier there). Its admission reads from one tier set per attention DP rank and keeps a
session on the rank it last ran on (`ChunkedPrefillAdmission::with_sticky_sessions`);
`SessionTierWorker` (`workers/iter/session_tier_worker.rs`) wraps the shell to
write finished contexts through and run `external_decode`. There `plain` alignment also
means one state at a retained context's end, reused whole or not at all.

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
attention shard count, KV bytes per token, cost manifest — instead of a model
handle, so it names no L4 model trait and cannot select an execution adapter. It
returns the `CostBuffers`; the recipe wraps them.

Deployments and pool controllers only call those recipes:

- unified → `build_barebone_worker` / `build_hp_worker` /
  `build_chunked_prefill_worker` / `build_speculative_worker` /
  `build_qwen36_hybrid_worker` / `build_hybrid_chunked_prefill_worker`
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
chunk-end quantum (`with_chunk_end_quantum`) ports vLLM's Mamba `align` split:
non-final chunks end on recurrent-state checkpoint boundaries, and the budget
a clipped chunk leaves may start further prompts. Only the hybrid recipes set
it, from the arch's checkpoint interval, when prefix caching is on and the
worker's `prefill_chunk_alignment` is `checkpoint` (the default, vLLM's
behavior); `plain` gives `min(remaining, budget)` chunks. The unified hybrid
recipe honours `dp_placement` like the full-attention one.

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

`chunked_prefill` takes the pipeline head's scheduler options too, all off by
default and none of them vLLM, so a PP vs DP comparison can differ only in
parallelism. Each acts per attention DP rank, as each rank is one vLLM engine
with its own scheduler. `srpt` merges a rank's started prompts into its queue's
shortest-first order (`admission/srpt.rs`, the head's code). `load_budget_*`
runs a rank's iteration at `load_budget_low_tokens` while the rank's own
backlog (its queued and started prompts' tokens; there is nothing in flight
between iterations) is at most the low bound, rising linearly to
`max_batch_tokens` at the high bound. `force_schedule_after_ms` also puts
overdue started prompts first under `srpt`. With tiers,
`prefix_tier_max_read_wait_ms` and `prefix_tier_read_wait_policy` bound a
rank's reads as on the head.

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
