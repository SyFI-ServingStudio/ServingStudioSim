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
  L1 timings. This is a routing/sequencing skill; it delegates each phase to a
  lower skill and does not implement leaves itself.
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

## Phase 1 — Explore & split

Route to `top-split-model-into-kernels` (which first calls `top-explore-models`
to establish the architecture). Output: the per-op decision table — each op is
**reuse** an existing kernel / **new dedicated kernel** / **`elementwise`
placeholder** / **fold** into a neighbor. That table is your L1 vocabulary and
your worklist for the build.

## Phase 2 — Build bottom-up (the timing wiring)

Build in strict bottom-up order; each layer only wires the layer below.

- **L1 — kernels (get the timing).** For each *new dedicated kernel* verdict,
  run `top-add-kernel` (Python profiling + Rust timing/cache). Reuse /
  elementwise / fold verdicts need no new L1. Done when `profile.db` has the rows
  and the Rust cache resolves.
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
- **Then a real run** — once the per-building-block numbers look right, graduate to
  `operate-run-simulation`: build + `--dry-run` to confirm the expanded plan, then
  launch a small workload and sanity-check end-to-end throughput.

## Delegation boundary

You are the umbrella, not an implementer. Issue each phase to the named skill and
verify its closeout before moving up a layer. Stop and return to the user between
phases whenever a design decision surfaces — a new args schema, a boundary call,
or a deviation from the design docs — rather than deciding it silently. The build
is complete only when every element from the Phase 1 table has a home at its
correct layer, the arch is wired (Phase 3), `model.work` support and the intended
deployment maps are complete (Phase 4), and the Phase 5 checks pass.
