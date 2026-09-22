---
name: top-add-new-arch
description: >-
  Use as the top entry point when the user wants to add a whole new model
  architecture to ServingStudio Sim end to end — explore it, split its forward into kernels,
  categorize every element into L1/L2/L3/L4, build the kernels, ops, worklets,
  and model_arch, then add an independent model.work necessary-work label. The
  one criterion running through the timing layers is the
  timing-prediction wiring — where each element's timing comes from: L1 is the
  only layer that MEASURES a number (profile.db); L2/L3/L4 only COMPOSE measured
  L1 timings. Adding an arch is paired with alignment by default: unless the user
  declines, capture a real vLLM/SGLang run first and let the measured kernel
  inventory — not source-reading guesswork — decide which kernels to add, largest
  first. This is a routing/sequencing skill; it delegates each phase to a lower
  skill and does not implement leaves itself.
---

# Top Add New Arch

Top-level umbrella orchestrator for standing up a brand-new model architecture in
ServingStudio Sim end to end. You sequence lower skills and build bottom-up. Do not implement
kernels, ops, worklets, or the arch file from here — each phase routes to a
dedicated skill.

## The one criterion: where does the timing come from?

Every element you create answers a single question — is its timing **measured**
or **composed**? That answer is its layer.

- **Measured → L1 kernel.** A real GPU launch you benchmark into `profile.db`
  and look up. L1 is the *only* layer that produces a number; it is the timing
  source of truth. The `elementwise` byte-placeholder is still L1 (a measured
  memory-bound floor), just approximate.
- **Composed → L2 / L3 / L4.** Everything above L1 predicts timing by *wiring*
  measured L1 timings together — no new numbers, only composition. Which layer
  depends on the *kind* of composition:
  - **L2 op** — sum a **fixed, 0-overlap** kernel sequence, plus op-specific
    shape normalization (model shape → kernel sweep coordinates).
  - **L3 worklet** — compose ops across **one sync section**: partition
    derivation + cross-op overlap math (`sum` / `max` / `pipeline`).
  - **L4 arch** — compose worklets: rank fan-out (`max`), layer stitching
    (`Scale`), and `WorkletGroup` boundaries.

This is both the categorization rule and the through-line of the whole build:
build bottom-up, and at each element ask "measure it, or wire it from below?"

## Phase 0 — Capture the real kernel sequence first

**Run this before Phase 1 unless the user declines.** Nearly every architecture
worth adding already runs in vLLM or SGLang, so the kernels you are about to
model can be watched instead of inferred. A capture answers two questions no
amount of source reading answers reliably: **which math is really one fused
launch**, and **which launches are big enough to matter**.

It can run this early because the profile phase stands alone — it needs only the
real engine running the checkpoint, not any ServingStudio Sim code. (Only the
later `timing-predict` phase reads a `simulation.yaml` preset, and that preset
needs the arch you have not written yet.)

- **How.** Route to `operate-run-alignment` and run **only its profile phase**,
  `profile_kind: nsys`. Stop there: the label and analyze phases compare against
  a CostTree you do not have yet. They run in Phase 5, on this same capture.
- **MoE models get a second pass**, `profile_kind: token_corpus`. Routing skew
  redistributes tokens into fuller and emptier expert groups, which changes the
  grouped-GEMM cost, and the arch you are about to write needs a
  `token_corpus_file` field to consume the measurement. Discovering that after L4
  is a rewrite, not an addition. (`expert_popularity` reduces the same run to a
  per-layer marginal and is deprecated: it is only correct where the batch is
  independent, i.e. no speculative decoding.)
- **What it gives you.** The folded measured kernel sequence per engine phase —
  real launch granularity, plus each kernel's share of iteration time. That share
  is the priority order for the entire build. And Phase 5 reuses the same capture,
  so this costs one GPU profile, not two.
- **What it does not give you.** Semantics. A demangled kernel name and its
  `suggested_category` are hints, not proof — `top-align-with-framework` treats
  every apparent gap as guilty until proven. The capture tells you where the
  launch edges are and which ones are large; `dev-lookup-transformers-model`
  still tells you what each one computes.

If the user declines, or the model has no vLLM/SGLang implementation, or there is
no GPU or checkpoint: Phase 1's source-reading path still works, but mark every
granularity verdict `inferred, unverified` in the decision table and say plainly
what that costs — Phase 5 becomes the first and only place those guesses are
tested, and a fusion edge found wrong there is a rebuild of L1/L2, not a retune.

## Phase 1 — Explore & split against the capture

Route to `top-split-model-into-kernels` (which first calls `top-explore-models`
to establish the architecture), and hand it the Phase 0 capture. Output: the
per-op decision table — each op is **reuse** an existing kernel / **new dedicated
kernel** / **`elementwise` placeholder** / **fold** into a neighbor. That table is
your L1 vocabulary and your worklist for the build.

With a capture in hand the split changes character:

- **The measured sequence answers "one fused launch or several?" directly**, so
  `dev-explore-kernel` narrows to the question it is actually good at: which
  public callable to wrap and how `KernelArgs` map onto it — the wrapper plan
  `top-add-kernel` needs. Do not re-derive granularity from source when the trace
  already shows it.
- **The table carries the evidence**: two extra columns for the measured
  kernel(s) (engine phase / track / folded position) and each op's share of
  iteration time.
- **Check completeness both ways.** Every measured kernel with material duration
  lands in some row, or is named explicitly as framework plumbing (bookkeeping,
  alloc/fill/copy, launch prep, sampling). Every row has a measured counterpart,
  or states why it has none — simulator-only work, or a fold whose cost already
  sits inside its host kernel.
- **Sort the table by measured share, descending. That order is the build
  order.**

## Phase 2 — Build bottom-up (the timing wiring)

Build in strict bottom-up order; each layer only wires the layer below.

- **L1 — kernels (get the timing), largest first.** For each *new dedicated
  kernel* verdict, run `top-add-kernel` (Python profiling + Rust timing/cache).
  Reuse / elementwise / fold verdicts need no new L1. Done when `profile.db` has
  the rows and the Rust cache resolves.

  Work the Phase 1 table top-down, biggest measured share first. Everything below
  your current line already has a home — an `elementwise` placeholder or a fold —
  so the arch is complete and runnable at every point in the build; descending the
  table improves precision, not coverage.

  Not forward order. Iteration time is dominated by a handful of kernels, so error
  retires fastest at the top of the table: a placeholder standing in for a 0.5% op
  costs you 0.5%, while a mismodeled attention or expert GEMM costs you the model.
  Building embed → norm → QKV → … in forward order spends the expensive
  `top-add-kernel` budget on ops that cannot move the number.

  Stop rule: after each new kernel, re-run Phase 5's checks and watch the
  deviation drop. Promote the next placeholder only while its remaining share
  justifies it — a placeholder the comparison already shows is close is a finished
  answer, not a debt.

  The stop rule picks *which* kernels to build, not the order you build them in.
  Once several have cleared it, hand them to `top-add-kernel` as one set and let
  it fan out — see its "Several kernels at once".
- **L2 — ops (sum + normalize).** Compose kernels into named ops: atomic
  (`Op<K>`, one kernel, naming only) or compound (hand-written, multi-kernel +
  op normalization). Boundary — a fixed 0-overlap sequence; comm merges into the
  op only if it shares the op's algorithm state, else it stays a separate atomic
  op. *Boundary examples:*
  - `lm_head` = `Op<SingleGemmKernel>` — one kernel, atomic, naming only.
  - `FlashInferAttentionOp` folds the prefill / append / decode sub-kernels under
    one attention three-formula → **one** compound op.
  - MoE dispatch's inter + intra all-to-all **merge into one** op because they
    share the routing math (shared algorithm state).
  - a TP all-reduce stays its **own** `Op<AllReduceKernel>` — a parallelism
    artifact with no shared state, so it does not merge into the compute op.

  Skill `impl-compose-op`
  (contract: `simulator/src/op/README.md`; design.md is intent only).
- **L3 — worklets (partition + overlap).** Compose ops into sync-section
  worklets on a declared GPU group (`Local`/`TP`/`HP`/`EP`/…): partition
  derivation, inclusion conditionals, and cross-op overlap math. Boundary — one
  sync section (ends at a collective, or at single-GPU self-completion).
  *Boundary examples:*
  - `FfnDenseTPWorklet` sums up_gate → activation → down ops, then a terminating
    `tp_allreduce` included only when `tp_size > 1` — the sync section ends at
    that collective.
  - `FlashInferAttnHPWorklet` puts the attention compound op in a slot, derives
    the GQA head split (HP partition), and is fanned out `num_hp_groups` times by
    L4.
  - the line against L2: `RingAttnWorklet` is a worklet, **not** an op, because
    its K rounds overlap compute and comm (`pipeline`) — overlap can only be
    expressed at L3.

  Skill `impl-compose-worklet` (contract: `simulator/src/worklet/README.md`).
- **L4 — arch (fan-out + stitch).** Wire the chosen worklets into the
  `model_arch` cost file: prologue / per-layer / epilogue groups, multi-rank
  fan-out, and `WorkletGroup` boundaries. It computes no partition itself.
  Skill `impl-compose-arch` (contract: `simulator/src/arch/README.md`).

## Phase 3 — Wire (make it selectable & predictable)

The L4 step produces an arch model that *builds in isolation*; it is not yet
reachable from a config or the predictor. This phase does the cross-file
integration that turns it into a selectable, predictable arch — and paves the road
for a future worker. Route to `impl-wire-new-arch`. It connects the new arch to
the shared dispatch sites:

- **`arch/build.rs`** — a concrete builder + its predictor arm in
  `build_iter_model` / `build_attn_model` / `build_ffn_model` (one uniform seam per
  kind, each boxed). This is what the offline predictor calls; `timing_predict.rs`
  needs no per-arch edit.
- **`deployment/{unified,pd,afd}.rs`** — the worker-pairing arm, or an explicit
  `bail!` "worker not wired yet" so future worker dev is a drop-in.

Uniform rule: **one predictor arm in `arch/build.rs` + one deployment arm**, for
every arch kind.

Most sites are **compiler-forced** (exhaustive matches) — adding the selector
variant breaks the build until each arm exists, so the compiler is the checklist.
Done when timing-predict runs green on the new arch (the Phase 5 proof). This is
the point where "where does the timing come from?" becomes *observable* — the
composed L1→L4 cost is now evaluable without a worker or a DES run.

## Phase 4 — Add the independent necessary-work label

Route to `impl-add-model-work-label` after the L4 arch and its exact location
names are stable. This phase gives Optimality an independent denominator for R6
necessary-work coverage and R7 redundancy:

- derive the model's compulsory FLOPs/bytes from the true architecture and
  workload, never from the simulator's timing tree;
- compose/register the `model.work` builder and add hand-derived parameter/FLOP
  goldens;
- map stable semantic work rows to every exact non-communication CostTree
  location for each supported deployment.

The architecture evidence comes from Phase 1; the location names come from the
finished Phase 2/3 wiring. Those inputs have different roles and must not be
mixed: simulator shapes may identify where semantic work lands, but never define
how much minimum work exists.

## Phase 5 — Validate

Confirm the wired arch predicts sane timing **before** standing up a full
deployment — you do not need a workload / pools / trace to check the cost model.

- **First, the offline predictor** — route to `operate-run-timing-predict`. It
  costs the new arch's compiled `CostTree` over a handful of explicit batch shapes
  (prefill-only, decode-only, mixed) with no scheduler/clock. Pick the selector
  that matches the arch (`iter` for a whole-iteration arch; `attn` / `ffn` for an
  AFD half), give a few representative cases, and read `reports/iter_breakdown.ans`:
  every element from the Phase 1 table should appear as a leaf with a non-zero,
  plausibly-scaled timing (no missing kernel, no absurd µs). This is the tightest
  loop for catching a mis-wired L2/L3/L4 composition.
- **Then close the loop on the Phase 0 capture** — the arch now exists, so the
  alignment phases that were blocked in Phase 0 can run. Resume that experiment in
  `operate-run-alignment`: `timing-predict` → label → `analyze` (kernel-align),
  then hand the artifacts to `top-align-with-framework` for Check 1, the
  per-kernel comparison against measured. **Never re-run the GPU profile** — the
  capture is still valid; only the simulated side changed.

  This is what turns Phase 0's *importance* ranking into an *error* ranking, and
  it feeds straight back into the L1 stop rule: the largest remaining deviation
  names the next placeholder to promote. Its two structural findings map onto
  Phase 1 verdicts — a large simulated slot with no measured kernel usually means
  a **fold** verdict was wrong, and a large unmapped measured kernel usually means
  the split **missed an op**.
- **Then a real run** — once the per-building-block numbers look right, graduate to
  `operate-run-simulation`: build + `--dry-run` to confirm the expanded plan, then
  launch a small workload and sanity-check end-to-end throughput. This is also what
  the alignment run's duty-cycle and TTFT/TPOT checks need, so if you want Checks
  2–3 they follow here, not above.

## Delegation boundary

You are the umbrella, not an implementer. Issue each phase to the named skill and
verify its closeout before moving up a layer. Stop and return to the user between
phases whenever a design decision surfaces — a new args schema, a boundary call,
or a deviation from the design docs — rather than deciding it silently. The build
is complete only when every element from the Phase 1 table has a home at its
correct layer, the arch is wired (Phase 3), `model.work` support and the intended
deployment maps are complete (Phase 4), and the Phase 5 checks pass — including,
unless the user declined Phase 0, a Check 1 comparison with no missing large
kernel and every material deviation attributed.
