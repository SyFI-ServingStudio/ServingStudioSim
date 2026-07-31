---
name: dev-compose-worker
description: "Use when planning or estimating a new VibeSim worker combination, or when implementing, composing, reviewing, renaming, moving, or tidying production L5 worker code under `simulator/src/worker/`: KV stores and capability traits, admission lifecycle/selection policies, execution adapters, cadence shells, family directories, `build_*_worker` recipes, and L6 wiring. Enforces the canonical four-axis ownership, share-vs-new and difficulty analysis, pending-membership rule, shell discriminator, construction/file boundaries, complete changed-file coverage, and behavior-preserving validation. Not for L4 kernel/cost-model implementation or an unrelated general file review."
---

# Compose Worker

Apply these rules to production L5 worker changes. Treat
`doc/detailed_design/L5.md` as the canonical design and
`simulator/src/worker/README.md` as the production tree guide. Old reference or
sketch code may explain behavior only when the canonical design and production
code leave a detail unclear; it never overrides them.

This skill specializes `dev-file-design-review` for L5. Use the stricter rule
when the two overlap.

## Required reading

Before editing:

1. Read `doc/detailed_design/L5.md` completely.
2. Read `simulator/src/worker/README.md`.
3. Read `simulator/src/worker/iter_worker.rs`.
4. Read the target component and concrete family shell completely.
5. Read every `build_*_worker.rs` recipe and L6 caller affected by the change.

Do not infer a new name, directory, trait surface, or ownership rule without
support from these sources. Stop for a design decision if the canonical L5
contract does not cover the proposed shape.

## 1. Classify the change before editing

Use the production equation:

```text
Worker = <KV, Admission<Selection>, Execution> × Shell(cadence)
```

Write down which owner changes:

| Owner | Owns | Must not own |
|---|---|---|
| KV | footprint, capacity facts, reserve/resident/held ledgers, sticky partition ownership, advance/release, KV sampling | selection, model input, shell phase |
| Admission lifecycle | fresh-request state transitions and gates | a shadow pending queue, KV byte arithmetic, model input |
| Selection policy | the sole not-yet-started pending membership and stable order | resident/active cadence state |
| Execution | L4 model, reusable input/cost buffers, state-to-`ArchInput` lowering, model evaluation | request ordering, transfer overlap, completion cadence |
| Shell | message protocol, FSM/cadence, grouping, transfer overlap, wakeups, completion ownership | duplicated KV/admission/accounting or model math |

Identify the cadence family separately:

- whole iteration → `workers/iter/IterBatchWorker`;
- pull plus decode → `workers/pd_decode/PullDecodeWorker`;
- layer-slot attention → `workers/afd_attention/SlotAttentionWorker` plus its
  private `AttentionSlotPipeline`;
- buffered FFN → `workers/afd_ffn/BufferedFfnWorker`.

Reuse a shell unless the change introduces a different asynchronous timeline,
event ordering, synchronization protocol, overlap model, or completion owner.
Batch arithmetic, token counts, selection order, and capacity policy do not by
themselves justify a new shell.

### Plan a new composition

Dissect a proposed worker before estimating or implementing it:

1. **Shell:** name the closest production cadence and any new timeline, ordering,
   synchronization, overlap, or completion-owner requirement.
2. **KV:** list the resource lifecycle and observation capabilities required.
3. **Admission lifecycle:** write the fresh-to-terminal state path; identify
   whether one existing lifecycle covers it or multiple behaviors must merge.
4. **Selection:** name the ordering facts and algorithm independently from the
   lifecycle.
5. **Execution:** name the L4 input/cost shape and every KV or shell fact it must
   read.
6. **Construction/L6:** name the concrete recipe, selector, and protocol caller;
   L6 may choose the recipe but must not assemble the axes.

Produce this share-vs-new table:

| Part | Reuse by default | Write only when |
|---|---|---|
| Shell | nearest cadence family | the shell discriminator above fires |
| KV | `FullAttnKv` plus existing capabilities | resource lifecycle or required observation is absent |
| Lifecycle | nearest family lifecycle | required state transitions cannot be expressed by one existing lifecycle |
| Policy | existing `PendingOrderPolicy` | ordering needs new frozen facts or a new algorithm |
| Execution | matching production adapter | input lowering or cost evaluation needs new facts/shape |
| Recipe/L6 | sibling builder and caller pattern | a new concrete composition must be selectable |

Capabilities are additive refinements; lifecycle behavior is single-select. A
combination of capabilities should normally be bounds over one KV owner. If two
lifecycle behaviors are both required, merge them once in one lifecycle rather
than stacking lifecycle objects or copying queues.

Estimate difficulty by the dominant semantic change, not the number of files:

- **LOW:** recipe/alias, policy swap, config value, or implementation of an
  already-defined capability;
- **MODERATE:** one new lifecycle, resource-owning KV leaf/wrapper, execution
  adapter, or bounded trait-seam extension;
- **HIGH:** a new cadence shell/protocol, changed completion ownership, or a
  challenge to the axis split.

Tag every new part with the smallest fixability class that describes it:

- **A — pure add:** add a leaf, policy, adapter, or recipe without changing
  existing bodies;
- **B — seam extension:** expose one missing capability through a trait bound,
  associated type, or narrow method;
- **C — mechanical reshape:** uniformly change signatures while preserving
  bodies and ownership;
- **D — new sub-axis or shell:** add a genuinely new capability family or
  cadence;
- **E — re-decide the axes:** the ownership equation cannot express the worker.

A–C extend the abstraction. D is substantial but bounded. E is a design-doc
decision: stop before implementation and make the ownership contradiction
explicit.

## 2. Enforce axis ownership

### KV

Keep `KvStore` as the common resource lifecycle:

```text
footprint → fits → reserve → commit_resident → advance → release
```

Expose family observations through the narrow capability traits
`IterWorkerKv`, `SlotPipelineKv`, and `HandoffKv`. Add a capability bound to the
concrete composition instead of adding a role enum or optional method to
`KvStore`.

Keep these invariants:

- one request stays on one KV partition for its KV lifetime;
- an AFD cadence slot is not a KV partition;
- the KV implementation is the only strict capacity/accounting source;
- input builders consume borrowed visitors/slices instead of cloning live
  membership;
- `KvStore` never constructs an L4 input.

When adding or changing a ledger, trace one request through every lifecycle hop.
At each hop name exactly one owner for the quantity. No owner means
over-admission; two owners mean double accounting.

Do not delegate `fits` or pressure accounting through a wrapper that owns an
additional occupancy ledger. Delegation is sound only for wrappers that keep
configuration/routing while placing the full resource quantity in the inner
footprint and ledgers. A resource-owning wrapper must extend the capacity seam
so its occupancy participates at every lifecycle hop.

### Admission and selection

Keep not-yet-started pending membership entirely in `PendingOrderPolicy`.
Lifecycle implementations must not maintain a second `pending` container.

Freeze selection facts in `AdmissionCandidate` at enqueue. Preserve stable tie
behavior and update membership, queued-count, and queued-KV observations
together for `push`, `pop`, `remove`, and cancellation.

Do not confuse a selection queue with active cadence state:

- fresh requests waiting to start → selection policy;
- PD landed/in-transit work → `PullDecodeWorker`;
- AFD slotted work → slot pipeline;
- FFN tasks → `BufferedFfnWorker`.

Use the family-specific admission surface. `SlotPipelineAdmission` is not an
`IterAdmission` with missing callbacks.

For future chunked prefill, keep fresh requests in the policy and at most one
partially processed `active_chunk` per partition in lifecycle state. Never pop
and push an active chunk through the policy: that rotates order and changes the
scheduling semantics.

### Execution

Keep model/input ownership in the matching sibling trait:

- `IterModelExecution<K>`;
- `AttentionLayerExecution`;
- `FfnTaskExecution`.

Build `UnifiedArchInput`, `AttnArchInput`, or `FfnArchInput` only here. Reuse
input and cost buffers. Keep transfer submission in the shell because overlap is
a cadence fact.

When another axis needs a KV fact, carry it through the narrowest production
seam:

- an iteration lifecycle places the capability bound on its
  `IterAdmission<K>` implementation;
- an iteration execution adapter places it on its
  `IterModelExecution<K>` implementation;
- a family trait that cannot express the needed fact requires a Class B seam
  extension, not a shell-side downcast or duplicated calculation.

Check method-generic family traits such as `AttentionLayerExecution` explicitly:
if the required capability cannot be named at the implementation boundary,
report that as real seam work rather than claiming the tuple already composes.

### Shell

Keep the shell concrete and behavior-readable. It owns the local fixpoint loop,
wakeup calculation, protocol messages, and terminal event timing. Do not hide
these behind a universal FSM trait or optional callbacks.

Missing capabilities should fail at the concrete recipe through trait bounds.
Do not repair a composition with runtime role checks, nullable components, or a
shared mega-trait.

## 3. Use compatibility as a semantic test

Reject a combination only for a semantic contradiction, such as a no-KV FFN
worker exposing decoder-KV behavior. “Not implemented”, “no preset”, “expensive”,
or “the current selector has another name” are not design contradictions.

An unknown variant fits when it needs:

1. one new leaf component or necessary thin adapter;
2. one declarative build recipe/registration;
3. focused contract and behavior tests;
4. no copied tick loop, pending ledger, KV ledger, or `ArchInput` lowering.

If the proposal requires copied shared machinery, identify the misplaced owner
before coding. Add a new shell only when the cadence discriminator in section 1
fires.

Classify a variant as config only when changing the value does not add a state
transition, ownership rule, event-order branch, or timeline. A behavioral fork
needs a distinct lifecycle, adapter, capability, or shell as appropriate.
Prefer parameter names that describe the mechanism rather than one use case,
and prefer composable `with_*` configuration over multiplying constructors.

## 4. Preserve construction and file boundaries

Use `workers/<family>/` as the shell-family directory.

- Put each concrete composition in its own `build_*_worker.rs`.
- Make the recipe visibly import and select KV, lifecycle, policy, execution,
  and shell types.
- Keep compatibility aliases as names for concrete generic compositions, not
  wrapper runtimes.
- Let deployments/controllers call recipes; do not make L6 assemble L5 axes.
- Keep construction off `IterWorker`.

Move repeated mechanics into a narrowly named essentials helper only when they
are truly mechanical: GPU allocation, capacity derivation/registration, sampler
creation, cost buffers, communication-group registration, and shared context.
An essentials helper must not select:

- a KV implementation;
- an admission lifecycle;
- a selection policy;
- an execution adapter;
- a shell or ingress behavior.

If a helper makes one of those choices, move that choice back into the concrete
recipe.

## 5. Make shell source read in execution order

For a non-trivial shell or private pipeline, make the module `//!` block state:

1. what the file owns and explicitly does not own;
2. the cadence/timelines it runs;
3. its accounting, membership, and completion invariants;
4. an explicit reading order.

Lay the file out:

1. module contract;
2. constants and state types;
3. worker/pipeline struct;
4. construction;
5. `IterWorker` and capability impls;
6. message handlers;
7. tick/fixpoint path;
8. helpers immediately after their first caller;
9. tests.

Order helpers by the call path, not by categories such as “all getters”.

For a shell-specific message enum, keep `enqueue` as pure dispatch and map each
variant 1:1 to `on_msg_<variant>`. Reserve `on_msg_*` for external message entry
points. A generic shell such as `IterBatchWorker` may delegate its opaque
`A::Msg` directly to admission; it must not inspect lifecycle-specific variants.

An inner protocol object may use `on_msg_*` only when the outer handler delegates
the same external message to it. Do not use `on_*` for an internal lifecycle
transition.

## 6. Apply naming-quality checks

Run every touched name through these tests:

- **Make the name earn its keep.** Keep a single-use helper when its name and doc
  encode a real concept or invariant; inline a purely mechanical expression.
- **Avoid false quantifiers.** Do not use `all_`, `any_`, or `count` when the
  value is one flag or one item.
- **Name the transition.** Prefer `begin_decode`, `commit_resident`, or
  `release_slotted_request` over `process`, `handle`, `update`, or `finalize`.
- **Qualify thin wrappers.** State whose fact is queried: `request_*`, `slot_*`,
  `partition_*`, or `kv_*`.
- **Use domain names at call sites.** Prefer `reserved_requests`,
  `classified_specs`, or `pulling_task` over vague `items`, `data`, or `state`.

Keep an old-reference symbol as a `(ref old_name)` anchor only when it materially
helps compare behavior with the still-relevant reference implementation. Remove
anchors to deleted local abstractions or superseded design; canonical L5 names
win.

For a rename, update the definition, call sites, tests, comments, docs, and
re-exports together. Use `rg` before and after, and distinguish same-spelled
methods on sibling types.

## 7. Require complete change-surface coverage

“Coverage” here means coverage of the changed design/code surface, not only test
line coverage.

Start with `git status --short`, `git diff --name-status`, and the relevant
commit/branch diff. Build a file inventory before reviewing or handing off.
Every materially changed file must appear in the review inventory. Do not hide
files under “miscellaneous” or review only the most interesting shell.

For each changed file, record:

| File | Axis/family | Behavior or ownership changed | Callers/consumers checked | Test/doc evidence |
|---|---|---|---|---|

Expand the review radius according to the owner:

- **Shared KV change:** inspect every implemented capability, all recipes using
  the store, cancellation/release, capacity registration, sampling, and both
  iteration/slot consumers.
- **Admission/policy change:** inspect every lifecycle using the policy,
  cancellation, queued status/KV, stable ties, and the absence of a shadow
  pending queue.
- **Execution change:** inspect every input builder, reusable buffer, cost-log
  path, model-layout accessor, and shell call site.
- **Shell/protocol change:** inspect message/event enums, wakeups, empty-work
  behavior, status counts, transfer overlap, terminal ownership, the L6
  controller, and the end-to-end sim action.
- **Recipe/essentials change:** inspect every sibling recipe, re-export,
  deployment factory call, GPU/KV/communication registration, logging setup,
  and constructor test.
- **Shared type or rename:** inspect all families, L6 callers, docs/README,
  lessons if they are in scope, and stale deleted paths.

For considerable worker work, explicitly cover all production kinds in the
inventory:

- barebone;
- HP unified;
- PD prefill;
- PD decode;
- AFD attention;
- AFD FFN.

Mark a kind “not affected” only after checking its composition/caller. Do not
silently omit it.

Also verify observational surfaces: `WorkerStatus`, request stages, event
ordering, KV snapshots/capacity metadata, cost logs/manifests, transfer logs, and
summary fields. An abstraction-preserving refactor may still be wrong if one of
these drifts.

## Workflow

1. Read the required sources and record the dirty-worktree boundary.
2. Classify the change by axis and cadence; for a new combination, complete the
   share-vs-new table, difficulty estimate, and A–E fixability tags.
3. Write the concrete proposed composition type and list every required trait
   bound before adding bodies.
4. Inventory the full change surface and affected production kinds.
5. Trace the behavior end to end:
   `enqueue → policy/lifecycle → KV → shell → execution → event/L6`.
6. Implement the smallest ownership-correct change.
7. Update module docs, `worker/README.md`, canonical L5, and neighboring L4/L6/L7
   docs when the contract, construction, or wiring materially changes.
8. Run the complete change-surface review from section 7; do not stop after the
   primary file builds.
9. Validate and report every skipped or unavailable check.

## Validation

Always run:

```bash
git diff --check
uv run cargo check -p simulator --lib
uv run cargo test -p simulator --lib
```

Check formatting only for changed Rust files:

```bash
rustfmt --edition 2021 --config skip_children=true --check <changed-file.rs> ...
```

Do not use `cargo fmt --all` for a scoped worker change.

Also run:

- `rg` for stale file paths, old type names, removed helpers, and incomplete
  rename call sites;
- focused tests for each changed axis and affected family;
- local Markdown-link and terminology checks when docs/README change.

For a new composition, prove the production tuple rather than adding a sketch
census. Add a focused compile-time trait assertion beside the closest shell and
a constructor test for its `build_*_worker` recipe. Treat an unsatisfied bound as
the exact additive worklist; do not bypass it with runtime role checks.

When cadence, admission, KV accounting, execution lowering, L6 wiring, or event
timing changes, compare representative unified, HP/MoE, PD, and AFD simulations
as applicable. Use the same resolved preset, workload, model JSON, and filled
profile database on both sides. Compare deterministic modeled outputs and event
semantics. Do not treat `wall_s` or derived `realtime_x` as equivalence fields.

## Output

For a planning-only request, lead with:

- **Difficulty:** LOW/MODERATE/HIGH plus the dominant cost;
- the share-vs-new table and A–E tag for every new part;
- the proposed concrete `Shell<K, A, E>` shape, omitting axes the family does not
  own;
- capability-escalation seams and the smallest additive worklist;
- compile/behavior evidence required before implementation is complete.

For an implementation or review, lead with the outcome and ownership verdict.
Include:

- the axis/family classification;
- the complete changed-file coverage table, with every material file accounted
  for;
- design and naming rules applied;
- affected production kinds and why any are not affected;
- validation commands and results;
- remaining semantic risks, skipped checks, or follow-up work.
