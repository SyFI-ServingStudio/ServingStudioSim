---
name: compose-new-worker
description: "Use when planning a NEW VibeSim worker combination in the four-axis model (KvStore / Admission / IterModelExecution / cadence-shell) — how to dissect a proposed worker into the four axes, decide share-vs-new per axis, estimate the difficulty with the composition-difficulty rule + the shell discriminator, spot the K-as-trait-param escalation seams, and prove composability with a census assert + cargo check. Grounded in sketch/worker_v2 and doc/detailed_design/L5_redesign.md. NOT the house-conventions tidy of an existing production IterWorker (that is worker-compose-rules), and NOT cost-model/kernel work (top-add-kernel / impl-validate-kernel-cache)."
---

# Compose New Worker

How to think about, dissect, and estimate the difficulty of adding a NEW worker
combination in the four-axis redesign, then plan the change. This is a
**design-reasoning** skill: it produces a plan + a difficulty estimate + a
compile-proof recipe, not the finished worker.

Authority + ground truth:
- **`doc/detailed_design/L5_redesign.md`** — the single authoritative design (axes,
  rulings, migration). Read Parts II/III/VI before planning anything non-trivial.
- **`sketch/worker_v2/`** — the compile-checked prototype: every claim here is
  demonstrated by a composition in `census.rs`. Read the `.code-lessons`
  architecture lesson `worker-v2-four-axis` for a guided tour.
- **`doc/detailed_design/L5_worker_compose_compatibility_matrix.md`** — the sampled
  combinations + the blind-test method.

Sibling skill: `worker-compose-rules` governs the *house conventions* of an
existing production `IterWorker` file. THIS skill governs *whether and how* a new
combination composes at all. Use that one to tidy; use this one to plan.

## 1. The mental model — four axes × family(= cadence)

A worker is `⟨K, A, E⟩` bolted onto a **cadence shell**:

```
IterBatchWorker<K, A, E>     iter family      K: IterWorkerKv, A: IterAdmission<K>,     E: IterModelExecution<K>
SlotAttentionWorker<K, A, E>  AFD-attn family  K: SlotPipelineKv,  A: SlotPipelineAdmission<K>, E: AttentionLayerExecution  (wears AttentionSlotPipeline)
BufferedFfnWorker<E>         AFD-ffn family   E: FfnTaskExecution                            (no KV, no admission)
PullDecodeWorker<K, E>       PD-decode family K: IterWorkerKv, E: IterModelExecution<K>       (no admission axis)
```

Four invariants you plan against:

- **Family = cadence.** The shell owns the FSM: how many async timelines exist and
  in what order events fire. It does NOT own arithmetic (batch size, attention
  shape, tokens/step).
- **Family = directory; construction recipe = file.** Put a cadence family under
  `workers/<family>/`; keep each worker/FSM or family-private pipeline in its own
  implementation file, and every concrete `build_*` recipe in a same-named file.
  A new composition adds a builder file instead of growing a family-wide grab bag.
  Repeated construction mechanics may use family-private build essentials, but those
  must not select K/A/E, policy, or ingress. Keep those choices and explicit
  imports in the concrete builder so the recipe remains readable in isolation.
- **`KvStore` is the ONLY axis that unifies across families.** A KV impl crosses
  families by adding one *read-view* capability (`IterWorkerKv` for iter,
  `SlotPipelineKv` for AFD-attn) — same leaf, two cadences.
- **`Admission` and `IterModelExecution` are per-family trait surfaces**, not one trait.
  Axes are OPTIONAL per family (ffn has neither; PD-decode has no admission).
- **`IterAdmission<K>` / `IterModelExecution<K>` take K as a *trait parameter*.** An impl can
  demand a KV capability in its own bound (`ChunkedPrefillAdmission` requires `K: ChunkedPrefillKv`).
  This is the M2 hard point; keep it in mind for §5.

## 2. Dissect the proposal — map each requirement to an axis

Never reason about "a prefix + hybrid + PD + SLO worker" as one thing. Split the
request into per-axis requirements first.

**Step A — Name the cadence (the shell question, ask FIRST).**
Does the proposal introduce a **new async timeline** or change **event ordering**
versus an existing shell?
- **No → reuse a shell.** Most workers do. Arithmetic differences (bigger batch,
  N-token attention, N tokens/step) are NOT cadence — they live in admission/exec.
- **Yes → new shell** (the expensive case). Triggers seen so far: a cross-worker KV
  transfer that overlaps compute (PD-decode → `PullDecodeWorker`), per-layer pipeline
  lockstep (AFD-attn → `AttentionSlotPipeline`), double-buffered transfer/compute (AFD-ffn →
  `BufferedFfnWorker`). A single forward pass = one compute region = still `IterBatchWorker`.

**Step B — KV capabilities.** List the capacity/lookup facts the worker needs, each
maps to a capability sub-trait (all in `kv/mod.rs`):
prefix reuse → `PrefixCacheKv` · held-for-handoff → `HandoffKv` · partial prefill →
`ChunkedPrefillKv` · mixed full+recurrent → `HybridKvView` · tiered/offload → `TieredKvView` ·
model co-serve → `ModelSwitchKv` · AFD per-request view → `SlotPipelineKv`.
Multiple capabilities → `+` them onto the bound (free, see §3).

**Step C — Admission lifecycle.** What is the request's fresh→…→done path? Is it one
existing lifecycle (`LocalPrefillDecodeAdmission` / `ChunkedPrefillAdmission` / `PrefillHandoffAdmission` /
`PrefixPrefillDecodeAdmission` / `FreshRequestSlotAdmission` / `MultiModelAdmission`) or a **merge of two**? A merge is
the one thing that costs a new file (§3).

**Step D — Selection policy.** What admit order? FIFO / SJF / SLO-deadline. A policy
is a pure `P` type-param swap over pre-computed `AdmissionCandidate` facts — free unless it is
a genuinely new ordering algorithm.

**Step E — Exec / cost.** Does the cost differ, and does it need to READ a KV
capability? If the arch model folds the difference (opaque to the worker), reuse
`UnifiedIterExecution` / `AttentionLayerExecutionAdapter`. If the cost must read a capability view
(tiered offload bytes, per-modality state), the exec must escalate (§5).

## 3. Estimate the difficulty

The rule (see memory `worker-compose-difficulty-rule`, and L5_redesign VI.3):

> **KV capabilities are MULTIPLICATIVE —叠加免费.** Each is a thin wrapper of the one
> real leaf `FullAttnKv` + one method; AND them onto the bound, stack any number.
> **Admission lifecycles are SINGLE-SELECT — 合并才收费.** A worker has exactly one
> lifecycle; needing two lifecycles' behavior = write ONE merged file.
> **Difficulty ≈ how many lifecycles you merged.**

Cost ladder (add them up):

| ingredient | cost | why |
|---|---|---|
| new shell (new cadence) | **HIGH** | a new FSM: timelines + event ordering |
| merged lifecycle | **MODERATE** | one new `Admission`/`SlotPipelineAdmission` file |
| new KV *leaf* (not a wrapper) | LOW–MOD | rare; only if `FullAttnKv` machinery can't back it |
| KV capability add (wrapper) | ~free | thin delegate + one read-view method |
| policy swap | ~free | `P` type-param, reads only `AdmissionCandidate` |
| config knob (N partitions, decode_steps) | ~free | same TYPE, different construction |

Calibration from validated combinations (all in `census.rs`):
- **hybrid + prefix-cache = LOW.** One lifecycle (`PrefixPrefillDecodeAdmission`), hybrid rides a KV
  capability (`HybridKvView`) invisible to admission. Two wrappers, reuse workers/exec.
  The one trap this combination surfaced: two wrappers that each contribute a
  "resident but never advanced" quantity need **one footprint slot each**
  (`FullAttentionKvFootprint.cached_prefix` / `.fixed_state`), not a shared slot. A shared slot
  sums correctly but makes the wrappers silently mutually exclusive and misnames the
  quantity. Generalize: when adding a wrapper that charges capacity, check whether the
  slot it writes is already owned by another wrapper.
- **+ PD disaggregation = MODERATE.** PD is a topology/cadence split. Decode side ≈ 0
  (reuses `PullDecodeWorker`+existing K/E). Cost concentrates in ONE merged prefill
  lifecycle: prefix-admit (`PrefixCacheKv`) × PD-handoff (`HandoffKv`) → one
  `PrefixPrefillHandoff` bound `K: HandoffKv + PrefixCacheKv`. Everything else additive.
- **AFD + hybrid KV + admission = LOW.** `SlotAttentionWorker<HybridStateKv, FreshRequestSlotAdmission,
  AttentionLayerExecutionAdapter<M>>`. Only gap: `HybridStateKv` needs `impl SlotPipelineKv` (~4-line delegate
  to its inner `FullAttnKv`, which already has it) — a pure add. `FreshRequestSlotAdmission` (bound
  `K: KvStore`) and `AttentionLayerExecutionAdapter<M>` compose for free.
- **speculative decode = config, not a worker.** Same constructor as barebone plus
  `.with_decode_steps(N)`; same workers/KV/exec. Batch size + attention shape are exec/model
  arithmetic, so no new cadence.
- **tiered KV = blind-test LOW.** New wrapper + one read-view + one census line; the
  tier-aware COST needs an escalated exec (§5), not a new shell.

### 3a. KV wrapper checklist — does your `fits`/`pressure` delegate actually hold?

Wrapping `FullAttnKv` and forwarding every method is the standard cheap move, but it is
only sound when the wrapper owns no resource of its own:

> **Wrapper keeps its own resource ledger ⇒ the `fits`/`pressure` delegate is a LIE.
> Wrapper keeps only config / a threshold / routing ⇒ the delegate holds.**

- `TieredMemoryKv` — folds `fast + slow` into the inner capacity at construction, keeps only a
  read-only threshold. Delegate holds.
- `ModelPartitionedKv` — one `Batch` partition per model, keeps only routing. Delegate holds.
- `ModeledPrefixCacheKv` — keeps a hit-rate config; its whole contribution rides in the footprint.
  Delegate holds.
- `HybridStateKv` — keeps a `recurrent` ledger of REAL occupancy. Delegate did NOT hold: the
  candidate's own state reaches the gate via the footprint, but `commit_resident` drops it
  from `promised` and it never enters the inner `Batch`'s `active_kv`, so the pool
  over-admits by one state per resident request. Fix pattern: the inner already solves this
  shape for `held` (occupied, not in `active_kv`) by folding it into `group_promised` —
  expose a `fits_with_extra_occupied` seam, plain passes 0, the wrapper passes its total.

The trace to run on any new wrapper: follow one request through `footprint → fits →
reserve → commit_resident → advance → release` and ask **at each hop which ledger holds
your quantity**. A hop where it is in none is an over-admit; a hop where it is in two is an
under-admit.

### 3b. Config or mode? (before you add a constructor)

A variant that changes only a NUMBER is config; a variant that changes BEHAVIOR is a mode
and probably wants its own type. The test is mechanical:

> **Delete the knob and hardcode the default. Does the struct get simpler?**
> No → the generality lives in a FIELD, it is config. Yes → it was carrying a behavior.

`decode_steps` passes: it is read twice (`advance(_, N)` + the emit count), the body has no
`if`, and N=1 is just a parameter value, not a special case. So speculative decode is a
config of `LocalPrefillDecodeAdmission`, not a second admission.

Two naming rules follow:
- **Name the parameter, not the use case.** `with_decode_steps(N)`, not `new_spec(...)` —
  the worker is explicitly blind to *why* N > 1 (lookahead and multi-token sampling set the
  same knob), so caller vocabulary must not leak into its API.
- **Prefer a `with_*` setter to a second constructor.** Two constructors differing by one
  argument do not compose: the second knob gives you `new_spec_chunked`. Setters add.

Counter-example in the same file: `max_batch_tokens: Option<u32>` IS a real behavioral fork
(`match` → one-prefill-per-iter vs budget-filled). It is kept because it mirrors the
production worker, but it is the honest target if someone calls `LocalPrefillDecodeAdmission` "all-in-one"
— not `decode_steps`. Meanwhile N partitions (barebone vs HP/DP) is NOT a fork at all: it is
a `for partition in 0..n` that is byte-identical to the single-group code at N=1.

## 4. Share-vs-new — the per-axis verdict

Produce this table for the proposal (this IS the plan's core):

| axis | reuse? | what you write |
|---|---|---|
| **shell** | reuse unless Step A tripped | (new shell only if a new timeline) |
| **KV** | reuse `FullAttnKv` machinery | a thin wrapper + the capability method(s) |
| **Admission (lifecycle)** | reuse or **merge** | nothing, or ONE merged file |
| **Admission (policy)** | swap `P` | nothing, or one new `PendingOrderPolicy` |
| **IterModelExecution** | reuse `UnifiedIterExecution`/`AttentionLayerExecutionAdapter` | nothing, or an escalated exec (§5) |

"Reuse the machinery" is the default at every axis — the leaf accounting lives once
in `FullAttnKv` + `CostBuffers`; a new combination is almost always *new bounds over
old bodies*, not new bodies.

## 5. The escalation seams — when a capability must reach an axis

Two places a KV capability has to travel to another axis. Both are additive; know
which one you are hitting:

- **Lifecycle needs a capability** (e.g. chunked admit needs `ChunkedPrefillKv`) → put the
  bound on the `IterAdmission<K>` **impl block** (`impl<K: … + ChunkedPrefillKv> IterAdmission<K>`).
  Works today; the shell never mentions the capability.
- **Cost needs a capability** (e.g. tiered offload surcharge needs `TieredKvView`) → the
  exec must be `IterModelExecution<K>` with the bound on its impl (see `TierAwareIterExecution`:
  `impl<M, K: IterWorkerKv + TieredKvView> IterModelExecution<K>`). This is the **Root ② fix** —
  `IterModelExecution` takes K as a trait param so cost can escalate, symmetric with
  `IterAdmission<K>`. Common execs still `impl for all K: IterWorkerKv`, so they compose
  with every KV.
- **Known unclosed seam:** `AttentionLayerExecution::build_slot_input<K: SlotPipelineKv>` is still
  *method-generic*, so an AFD-attn cost CANNOT yet escalate to a capability (e.g.
  `HybridKvView`). If a proposal needs that, it is a **Class B** seam extension
  (parameterize `AttentionLayerExecution` by K, or add a capability read) — additive, same shape as
  Root ②, but flag it as real work.

## 6. When it does NOT fit — the fixability taxonomy

If an axis can't express the proposal, classify the blast radius before writing code
(L5_redesign VI). Prefer the lowest class that works:

- **A — pure add.** New wrapper / new lifecycle file / new policy; zero existing
  bodies touched. (Most new combinations. e.g. `HybridStateKv: SlotPipelineKv`.)
- **B — seam extension.** Add an associated type or a trait param + a default so old
  impls are unaffected. (e.g. escalate `AttentionLayerExecution` to `AttentionLayerExecution<K>`.)
- **C — mechanical reshape.** Bodies unchanged, signatures shift uniformly. (Root ②
  was this: `IterModelExecution` → `IterModelExecution<K>`.)
- **D — new sub-axis.** A genuinely new capability sub-trait or a new shell.
- **E — re-decide the axes.** The four-axis split itself is wrong for this. Has not
  happened; treat as a design-review escalation, not a code task.

A proposal that lands in A/B/C is "add an interface, keep the design." D is real but
bounded. E means stop and take it to the design doc.

## Workflow

1. Read `doc/detailed_design/L5_redesign.md` Parts II/III/VI and skim `census.rs` for
   the nearest existing combination. Open the `worker-v2-four-axis` code-lesson if you
   want the guided tour.
2. Run §2 Steps A–E on the proposal; write the per-axis requirement list.
3. Fill the §4 share-vs-new table; tag each "new" cell with its §6 fixability class.
4. Add up the §3 ladder → a one-word difficulty (LOW / MODERATE / HIGH) with the
   dominant cost named (usually "one new shell" or "one merged lifecycle").
5. Check the §5 escalation seams: does any capability need to reach admission or cost?
   Name the seam and whether it is already open (iter cost = open; AFD cost = Class B).
6. **Prove composability cheaply.** Write the composition type
   `Shell<K, A, E>`, add one `assert_iter::<…>()` / `assert_afd_attn::<…>()` line to
   `sketch/worker_v2/census.rs`, and run:
   ```bash
   uv run cargo check --tests   # EXIT 0 = the tuple composes against the real L6 traits
   ```
   A missing bound (e.g. `HybridStateKv: SlotPipelineKv not satisfied`) IS the additive
   worklist — the compiler points at exactly the wrapper/method to add. This is the
   plan's completion evidence; you do not need to build the real worker to prove the
   seams hold.

## Output

Lead with the difficulty verdict, then the plan. Include:

- **Difficulty**: one word (LOW/MODERATE/HIGH) + the dominant cost named.
- **Per-axis table** (§4): reuse-or-write for shell / KV / lifecycle / policy / exec,
  each "write" cell tagged with a fixability class (§6).
- **The composition type** `Shell<K, A, E>` and the census assert line that proves it.
- **Escalation notes** (§5): any capability that must reach admission/cost, and
  whether that seam is open or needs a Class B extension.
- **The additive worklist**: the wrappers/methods/files to write, smallest first —
  ideally what `cargo check --tests` names as unsatisfied bounds.
