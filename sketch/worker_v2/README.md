# worker_v2 — interface expressiveness experiment

Purpose: stress-test the component interfaces now consolidated in
[`doc/detailed_design/L5_redesign.md`](../../doc/detailed_design/L5_redesign.md)
(the single authoritative doc; the original interfaces/refactor/module_apis drafts are
archived out of repo at `../../../L5_old_docs/`), by implementing ~20 worker compositions
sampled from the [compatibility matrix](../../doc/detailed_design/L5_worker_compose_compatibility_matrix.md).
Implementation bodies are rough; **input/output types and external function calls
are accurate** against current `simulator/src/` source (re-verified, not trusted
from the older `worker_compose` sketch).

## Layout — one folder per axis, one folder per worker family

```
worker_v2/
  shared/      shared vocab (PartitionId ⟂ AdvanceScope) + WorkerContext
  kv/          KvStore (+ IterWorkerKv, SlotPipelineKv, caps: ChunkedPrefillKv / HandoffKv)  [ONE cross-family axis]
  admission/   per-family: IterAdmission<K> (iter: LocalPrefillDecodeAdmission / ChunkedPrefillAdmission / PrefillHandoffAdmission) | SlotPipelineAdmission<K> (AFD-attn) | — (ffn)
    policy/    PendingOrderPolicy (FifoOrder; SLO later)  — iter-family only so far
  execution/  per-family: IterModelExecution (iter) | AttentionLayerExecution (AFD-attn) | FfnTaskExecution (AFD-ffn)
  workers/     concrete cadence shells (own the FSM + comm seam; integrate the axes)
    iter/          IterBatchWorker + UnifiedIterBuildEssentials + one file per build_* recipe
    pd_decode/     PullDecodeWorker + build_pd_decode_worker
    afd_attention/ AttentionBuildEssentials + shared slot pipeline + two ingress workers/builders
    afd_ffn/       BufferedFfnWorker + build_afd_ffn_worker
```

Inside `workers/`, a family is a directory because cadence is the worker-level
ownership boundary. Worker/FSM code stays in the worker's file, while every
concrete `build_*` construction recipe has a same-named file. This keeps adding a
composition from turning one family module into a builder grab bag. Repeated
allocation/capacity/sampler/cost wiring belongs in family-private build
essentials; each recipe keeps explicit imports and owns only its K/A/E/ingress choices.

## The composable shape — one per family (family = cadence)

```rust
// iter family (S0/S2/S3/S6 — whole-iteration cost)
IterBatchWorker<K, A, E>     where K: KvStore + IterWorkerKv, A: IterAdmission<K>,     E: IterModelExecution<K>
// AFD-attn family (layer-wise, 3-slot pipeline)
SlotAttentionWorker<K, A, E>  where K: KvStore + SlotPipelineKv,  A: SlotPipelineAdmission<K>, E: AttentionLayerExecution
// AFD-ffn family (double-buffer over pre-composed FfnTasks) — NO KV, NO admission
BufferedFfnWorker<E>         where E: FfnTaskExecution
```

FSM is NOT a generic axis — each shell owns its cadence. Admission takes the KV
type as a **trait parameter** so a lifecycle can require capability sub-traits on
`K` in its `impl` (M2 hard point): `LocalPrefillDecodeAdmission` requires `K: IterWorkerKv`.

**#7 (AFD attn) — the family boundary:** `KvStore` is the ONLY axis that unifies
across families — `FullAttnKv` is reused verbatim (the AFD shell just drives it via
`advance(AdvanceScope::RequestSubset{..})` instead of `advance(AdvanceScope::WholePartition)`, plus one
read view `SlotPipelineKv`). `Admission` and `IterModelExecution` are per-family trait surfaces,
not one trait: the AFD shell owns layer cadence + slot batching, so its admission has
no `form_batch`/`complete_iteration`, and its exec is `evaluate_attention_layer` (per layer) not
`evaluate_iteration` (per iter). The comm seam (comm-group registration + `submit_gather` pulls)
lives on the **shell**. Per-family ≠ per-worker: a second iter worker reuses
`IterBatchWorker`; a second AFD-attn worker reuses `SlotAttentionWorker`.

**#8 (AFD ffn) — axes are OPTIONAL per family:** the ffn worker has no KV and no
admission (L6 hands it pre-composed `FfnTask`s), so it composes as `<E>`-only. This
proves the triple isn't a fixed shape every worker fills — a family takes only the
axes it needs. `FfnTaskExecution` is the purest exec: **store-free and KV-free**, operating
only on token counts. Per-token completion bookkeeping — the one lifecycle fact the
ffn shares with the iter family — runs over `RequestRecord`'s methods
(`record_first_token` / `record_token` / `is_complete`), inlined in the Terminal like
the iter family does it (no shared wrapper), stamped on the request's sticky ATTN
owner so stage-stamping there bypasses `WorkerContext::stamp_stage`.

## Workers — 22 servers over 16 distinct types

A **server** = a (composition, config) pair; several servers share one worker TYPE
(barebone vs HP = `N`; hybrid vs hybrid-DP = `N`;
policy swap = `P` type param). The 16 distinct TYPES are each **compile-asserted** in
[`census.rs`](census.rs) against the real L6 `IterWorker` / `AfdAttnWorker` traits — `cargo
check` passing IS the proof they compose. `M` is the arch model (dense/MoE alike — MoE is
transparent to the worker, only its DP-shard count shows).

| # | server | composition (type) | builder | status |
|---|---|---|---|---|
| 1 | dense, no attn-DP | `IterBatchWorker<FullAttnKv, LocalPrefillDecodeAdmission<FifoOrder>, UnifiedIterExecution<M>>` | `build_barebone_worker` | ✅ |
| 2 | HP/DP unified (N shards) | ⟵ same type, `N` partitions + RoundRobin | `build_hp_worker` | ✅ |
| 3 | latency-scheduled (SJF) | `IterBatchWorker<FullAttnKv, LocalPrefillDecodeAdmission<ShortestJobFirst>, UnifiedIterExecution<M>>` | `build_shortest_job_worker` | ✅ |
| 4 | speculative (MTP, N=1) | `DraftVerifyWorker<FullAttnKv, DraftVerifyAdmission<FifoOrder>, DraftVerifyExecution<M,O>>` | `build_speculative_decode_worker` | ✅ |
| 5 | speculative + DP | ⟵ same S6 type, N sticky KV partitions | `build_speculative_decode_worker` | ✅ |
| 6 | chunked prefill | `IterBatchWorker<FullAttnKv, ChunkedPrefillAdmission<FifoOrder>, UnifiedIterExecution<M>>` | `build_chunked_prefill_worker` | ✅ |
| 7 | chunked + SJF | `IterBatchWorker<FullAttnKv, ChunkedPrefillAdmission<ShortestJobFirst>, UnifiedIterExecution<M>>` | (census) | ✅ |
| 8 | PD prefill (handoff) | `IterBatchWorker<FullAttnKv, PrefillHandoffAdmission, UnifiedIterExecution<M>>` | `build_pd_prefill_worker` | ✅ |
| 9 | PD decode (pull→decode) | `PullDecodeWorker<FullAttnKv, UnifiedIterExecution<M>>` | `build_pd_decode_worker` | ✅ |
| 10 | prefix-cache dense / DP | `IterBatchWorker<ModeledPrefixCacheKv, PrefixPrefillDecodeAdmission<FifoOrder>, UnifiedIterExecution<M>>` | `build_prefix_cache_worker` | ✅ |
| 11 | prefix + SJF | `IterBatchWorker<ModeledPrefixCacheKv, PrefixPrefillDecodeAdmission<ShortestJobFirst>, UnifiedIterExecution<M>>` | (census) | ✅ |
| 12 | hybrid-attention dense | `IterBatchWorker<HybridStateKv, LocalPrefillDecodeAdmission<FifoOrder>, UnifiedIterExecution<M>>` | `build_hybrid_kv_worker` | ✅ |
| 13 | hybrid-attention DP | ⟵ same type, N partitions | `build_hybrid_kv_worker` | ✅ |
| 14 | hybrid + SJF | `IterBatchWorker<HybridStateKv, LocalPrefillDecodeAdmission<ShortestJobFirst>, UnifiedIterExecution<M>>` | (census) | ✅ |
| 15 | multi-model co-serve (2) | `IterBatchWorker<ModelPartitionedKv, MultiModelAdmission<FifoOrder>, MultiModelIterExecution<M>>` | `build_multi_model_worker` | ✅ |
| 16 | multi-model co-serve (4) | ⟵ same type, 4 co-resident models | `build_multi_model_worker` | ✅ |
| 17 | multi-model + SJF | `IterBatchWorker<ModelPartitionedKv, MultiModelAdmission<ShortestJobFirst>, MultiModelIterExecution<M>>` | (census) | ✅ |
| 18 | AFD attn, dense | `SlotAttentionWorker<FullAttnKv, FreshRequestSlotAdmission, AttentionLayerExecutionAdapter<M>>` (→ `AttentionSlotPipeline`) | `build_afd_attention_worker` | ✅ |
| 19 | AFD ffn, dense | `BufferedFfnWorker<FfnSectionExecutionAdapter<M>>` | `build_afd_ffn_worker` | ✅ |
| 20 | PD-AFD decode-attn | `PullSlotAttentionWorker<FullAttnKv, AttentionLayerExecutionAdapter<M>>` (shares `AttentionSlotPipeline`) | `build_pd_afd_decode_attention_worker` | ✅ |
| 21 | PD-AFD prefill (reuse #8) | `IterBatchWorker<FullAttnKv, PrefillHandoffAdmission, UnifiedIterExecution<M>>` | `build_pd_prefill_worker` | ✅ |
| 22 | PD-AFD decode-ffn (reuse #19) | `BufferedFfnWorker<FfnSectionExecutionAdapter<M>>` | `build_afd_ffn_worker` | ✅ |

### Coverage — every axis impl exercised ≥ once

| axis | impls (each ≥ 1 server) |
|---|---|
| **Kv** | `FullAttnKv` (1–9,18,20–22) · `ModeledPrefixCacheKv` (10–11) · `HybridStateKv` (12–14) · `ModelPartitionedKv` (15–17) |
| **Admission** | `LocalPrefillDecodeAdmission` (1–5) · `ChunkedPrefillAdmission` (6–7) · `PrefillHandoffAdmission` (8,21) · `PrefixPrefillDecodeAdmission` (10–11) · `MultiModelAdmission` (15–17) · `FreshRequestSlotAdmission` (18) · *(none: 9,19,20,22)* |
| **Policy** | `FifoOrder` (most) · `ShortestJobFirst` (3,7,11,14,17) |
| **IterModelExecution** | `UnifiedIterExecution` (1–17) · `AttentionLayerExecutionAdapter` (18,20) · `FfnSectionExecutionAdapter` (19,22) · `MultiModelIterExecution` (15–17) |
| **Shell** | `IterBatchWorker` (1–17) · `PullDecodeWorker` (9) · `SlotAttentionWorker` (18) · `PullSlotAttentionWorker` (20) · `BufferedFfnWorker` (19,22) |

Capability sub-traits exercised: `IterWorkerKv` (all iter), `SlotPipelineKv` (18,20),
`HandoffKv` (8,21), `ChunkedPrefillKv` (6–7), `PrefixCacheKv` (10–11), `HybridKvView` (12–14),
`ModelSwitchKv` (15–17). Every KV impl is either `FullAttnKv` or a thin wrapper of it
(`ModeledPrefixCacheKv`/`HybridStateKv`/`ModelPartitionedKv` reuse its `Batch`/`KvAdmission` machinery verbatim
and add exactly one capability), so the real leaf accounting is shared across all four.

**#4a (PD prefill) — two interface generalizations, one new capability.** PD prefill
prefills then HANDS OFF (no local decode): the KV becomes HELD until the decode side
pulls it. Reuses `IterBatchWorker` + `FullAttnKv` + `UnifiedIterExecution`; the admission is new
(`PrefillHandoffAdmission`), and PD forced two workers/trait generalizations (both behavior-
preserving for existing workers): (a) `IterAdmission::accept_message` now takes `&mut K` — PD's
`ReleaseKv` message drops a held reservation out of the iter cycle, which a `msg`-only
accept could not touch; (b) `IterBatchWorker` now uses `A::Msg`/`A::Event` (was pinned to
`WorkerMsgCommon`) so it carries the PD-specific `PdPrefillMsg`/`PdPrefillEvent`. New
capability `HandoffKv` (hold/drop_held); `fits` counts held KV against capacity. Deviation
flagged: the doc's zero-arg `hold(req)` assumed the reservation persists in `promised`,
but `drain_ready` clears it at `form_batch`, so `hold` carries the token count (as the
real PD worker does). **#4b (PD decode) is the 4th shell** — `PullDecodeWorker` — because it has TWO async
timelines (the KV transfer pull + the decode iter) that `IterBatchWorker`'s single-compute
FSM does not model. But it REUSES `FullAttnKv` (K) + `UnifiedIterExecution` (E) unchanged: the
pull front-end (`pending_pulls` → one-in-flight `submit_transfer` → backlog-gated →
`pending_decodes`, emitting `PullComplete`) is shell-only, and the direct-admit
(`commit_resident` with no prior reserve — the request arrives already prefilled) plus
the decode bookkeeping are thin enough to inline (no admission axis, like AFD-ffn). Net:
four cadences now cover the census — `IterBatchWorker` (iter), the `AttentionSlotPipeline` slot pipeline (AFD-attn,
worn by `SlotAttentionWorker` *and* `PullSlotAttentionWorker`), `BufferedFfnWorker` (AFD-ffn), `PullDecodeWorker` (PD decode)
— and K/E are shared across all of them.

**#9 (PD-AFD decode-attn) — the slot core is SHARED, not embedded.** The three-pool
PD-for-AFD disaggregation reuses two existing builders unchanged — prefill = `build_pd_prefill_worker`
(#4a, iter-family handoff), decode-ffn = `build_afd_ffn_worker` (#8) — so the ONLY new piece is
the decode-attn worker. Its lifecycle is: receive a `Handoff` (already-prefilled request +
its held-KV endpoint) → PULL the prompt KV across the fabric → ack the prefill side → decode
via the layer-wise slot pipeline. That last step is *byte-for-byte* the AFD-attn slot pipeline,
so to obey the interfaces doc ("share the private slot core, do NOT embed one concrete worker
inside another") the pipeline + comm seam were factored out of `SlotAttentionWorker` into a shared
**`AttentionSlotPipeline<K, E>`** (a behavior-preserving refactor of #7 — `ctx` moved to the shell so the
ingress gets `&mut kv` + `&ctx` in one expression). Now BOTH shells are thin wrappers over
`AttentionSlotPipeline` differing only in INGRESS: `SlotAttentionWorker` = `AttentionSlotPipeline` + `FreshRequestSlotAdmission` (reserve-on-
admit); `PullSlotAttentionWorker` = `AttentionSlotPipeline` + `KvPullIngress` (single-in-flight `submit_transfer`, like
`PullDecodeWorker`'s front-end but acking with `AttnWorkerEvent::KvPullComplete`, the "PD-for-AFD only"
event). Two interface results: (a) the AFD-attn family is axis-OPTIONAL like AFD-ffn — this
worker composes as `<K, E>` with **no admission axis** (the pull front is shell-only); (b) the L6
`AfdAttnWorker` bound `Self::Msg: From<AttnWorkerMsg>` lets it carry a WIDER `PullAttentionWorkerMsg` (common
control protocol + `Handoff`) while the pool drives it through the common `From`. The landed
handoff needs **no new commit path** — the core's *existing* prefilled-handoff branch
(`open_iterations` → `begin_decode` at the layer-0 boundary) commits its pulled KV resident, so
the pull front-end never touches KV. Rough spot (flagged): the pull throttle is single-in-flight
only; the real 5% token-backlog gate is deferred (same load heuristic `PullDecodeWorker` carries).

**#3 (HP/DP) — barebone is the N=1 instantiation.** The crate keeps `HpUnifiedWorker`
as a separate ~460-line file; here it is the *identical* composition to #1 with
`num_partitions = N` (one `FullAttnKv` `Batch` per DP shard) + `LoadBalance::RoundRobin`.
Getting there generalized three shared pieces from single- to multi-partition, all
byte-identical at N=1: `IterModelExecution::build_iteration_input` now loops partitions (one arch group per
shard — the arch `Max`es across them for the DP wallclock); `LocalPrefillDecodeAdmission` now carries
a `LoadBalance` cursor + per-group budget and loops partitions in `form_batch` /
`complete_iteration`; `IterBatchWorker::status`/`next_wakeup` sum over partitions. `FullAttnKv`
needed NO change — it was already multi-`Batch` (the `promised` ledger keys partition).
So "+HP groups" is a config axis (N, balance), not a new worker.

**#2 (chunked prefill) — capability escalation + reuse.** Superset worker (the crate
parses `ChunkedPrefillAdmission` but its deployment bails — no reference impl; `Batch` even
notes it "does not carry the chunked-prefill queue yet"). Reuses `IterBatchWorker` + KV +
`UnifiedIterExecution` verbatim; only the admission is new. Two interface results: (a) the new
`ChunkedPrefillAdmission<P>` binds `K: IterWorkerKv + ChunkedPrefillKv` — strictly stronger than
`LocalPrefillDecodeAdmission`'s `K: IterWorkerKv` — so the M2 capability escalation is real (the
`ChunkedPrefillKv` capability is exercised for the first time: chunks land reserved→resident
via `append_prefill_chunk`, the last chunk `finish_chunked_prefill`s to decode). (b)
`IterBatchWorker` itself never mentions `ChunkedPrefillKv`; the stronger bound is satisfied purely
at the `ChunkedPrefillAdmission` impl, so the shell is untouched. All external `Batch` calls are
real (`add_kv`/`sub_kv`/`finalize_to_decode`); the chunked lifecycle is the new design.

**#3 (SJF) is a selection-policy swap; #4–5 (spec) require a new cadence.**
`FifoOrder → ShortestJobFirst` remains a `P` type-param swap with zero change to
lifecycle, KV, execution, or shell. Speculative decode is different: the S6
`DraftVerifyWorker` creates tentative proposals, evaluates a draft/target-shaped
input through `DraftVerifyExecution`, retains request-local runtime outcomes
across the compute window, then commits exactly `committed_tokens` and discards
the rejected suffix through `SpeculativeKv`. The proposal width is config; the
accepted length is not. This is additive to the four-axis design: it uses a new
per-family execution/admission surface and cadence while reusing `FullAttnKv`
through a narrow capability.

**#10–11 (prefix) / #12–14 (hybrid) / #15–17 (multi-model) — three new KV impls, each a
thin wrapper of `FullAttnKv` + one capability.** All three reuse the real `Batch` /
`KvAdmission` leaf verbatim (delegating every `KvStore`/`IterWorkerKv` method) and add
exactly one thing: **`ModeledPrefixCacheKv`** provides a partition-local `PrefixPlacementProbe` whose match
and footprint come from the same snapshot; its paired `PrefixPrefillDecodeAdmission` picks the best
feasible partition, reserves there, and keeps that placement sticky through decode.
The modeled cached region shares the request-KV capacity gate but is conservatively
charged per request (no radix/refcount ledger yet). **`HybridStateKv`** adds a fixed recurrent ledger beside
the growing full-attention `Batch`, so the per-kind `advance` (full +N, recurrent +0) falls
out for free and `HybridKvView` exposes both kinds. **`ModelPartitionedKv`** makes partition ≡
`ModelId`, adds `ModelSwitchKv`, and pairs with `MultiModelAdmission` (wider `SwitchModel` message,
like PD's wider msg) + `MultiModelIterExecution` (per-partition model, `Max`ed). All three prove the
axis split's core claim: the admission/exec constrain only on a read-view capability
(`PrefixCacheKv` / `HybridKvView` / `ModelSwitchKv`), never on the concrete KV type. Superset: none
of prefix-radix / SSM state / `ModelId` exists in-crate (verified), so the *lifecycles* are
new while every external primitive call (`Batch`, `KvAdmission::try_admit`, `CostBuffers::
run_iter`, `RequestRecord`) stays signature-accurate.

## Compile-check

Included under `cfg(test)` from `simulator/src/worker/mod.rs`. Build with the repo
env (`just`/`uv`): `uv run cargo build` (or `cargo check`). It is not wired into
any production selector; it exists only to prove the seams compose.
