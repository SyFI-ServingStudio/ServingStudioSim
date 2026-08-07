# Invariants

These are the rules the whole stack upholds. They are what keep the layers
composable: as long as each holds, a change inside one layer cannot corrupt the
layer above or below. They are grouped by where they bite.

## Timing and numbers

- **L1 is the only measured layer.** Every number originates from a profiled row in
  `profile.db`. No layer above L1 invents a timing; it only composes L1's outputs.
- **`gpu_name` is explicit in every kernel lookup.** The GPU identity is part of the
  kernel config (baked at `resolve_config`), not a per-call input and not an ambient
  global — the cache is keyed per GPU.
- **Coverage OODs up.** A leaf that had to extrapolate, JIT-profile, or missed
  coverage flags itself, and those flags OR their way to the root, so an off-grid
  leaf anywhere is visible at the total.

## CostTree (compile once, evaluate per iteration)

- **INV-1 — the leaf count is fixed at compile.** The tree's structure does not
  depend on the request count. A variable per-request fan-out (e.g. many prefills)
  aggregates *into* a fixed slot rather than minting one slot per request.
- **INV-2 — slot index = visit order.** `eval` must push leaf metrics in the exact
  child order that `compile` minted the slots, so the evaluator cursor lines up with
  the slot buffer.
- **INV-4 — only Scale and Max touch time.** `flops` / `bytes` / `energy` always
  sum; `time` is only rescaled by a `Scale{n}` fold or overlapped by a `Max`. A
  plain `Sum` adds time; nothing else rewrites it.
- **INV-3 — `Scale{n}` folds only provably-identical subtrees.** Repeating a
  homogeneous decoder layer with `Scale{num_layers}` is valid only because every
  folded layer has bit-identical structure; a heterogeneous fan-out (uneven EP, a
  hybrid layer schedule) expands instead.
- **INV-5 — names live only in the manifest.** The hot-path flat tree and the log
  rows carry no names; the `CostManifest` sidecar holds the taxonomy, and render-only
  `Labeled` nodes are dropped from the flat tree.
- **INV-6 — one scalar out, taxonomy replayed.** Runtime aggregation yields a single
  metric per iteration; the analyzer reconstructs the per-op / per-worklet breakdown
  from the manifest after the run, not during it.

## Composition boundaries

- **An op's internal composition is fixed and zero-overlap.** The kernels inside an
  op run in a fixed sequence with no cross-kernel overlap; overlap is a worklet
  concern, not an op concern.
- **A worklet is exactly one sync section.** Cross-op composition — sum, overlap,
  pipeline — lives only at L3, and a worklet's suffix (`Local` / `TP` / …) names its
  single sync grain. Ring attention is a worklet, not an op.
- **L2 and L3 are parallelism-agnostic below L4.** Ops and kernels see only per-rank
  shapes; no op or kernel reads a `ParallelConfig`. The partition is derived once at
  L3's `resolve_config` and consumed at L4.
- **No op-slot polymorphism.** A leaf's backend is selected by a backend string in
  the kernel config that passes straight through to L1; a worklet never branches to
  pick a backend, and an op never swaps its kernel by type.
- **A worklet never sees an op's internals.** The sub-kernels and internal
  normalisation of an op are hidden from the worklet that holds it.

## Identity and naming

- **Cross-language identity holds.** `KernelSpec::KIND` (Rust) == `profile.db` table
  name == Python facade stem; the bridge derives `get_{kind}_times` from it. A
  kernel's `ArgsPayload` field set is the wire schema the Python `*Args` dataclass
  validates field-for-field.
- **Name paths thread parent to child.** A leaf's dotted name is composed as
  `{parent}.{slot}` down the tree, so `model.attn.o_proj` is assembled, never
  hard-coded at the leaf.

## Deployment, workers, and the run

- **One worker binds one arch; many workers to one arch (G1).** A worker holds
  exactly one model arch; several workers may share the same arch instance.
- **One worker = one physical GPU group (G2).** A worker maps to a single physical
  GPU group; parallelism within that group is the arch's concern, not the
  orchestrator's.
- **L5 state has one owner per axis.** KV owns resource accounting; the selection
  policy owns not-yet-started pending membership; execution owns L4 input
  lowering; the concrete shell owns cadence, overlap, and completion.
- **Request-to-KV-partition placement is sticky.** A request does not rotate
  between attention-DP partitions during its KV lifetime; cadence slots are not
  KV partitions.
- **The pool boundary is L6's only abstraction.** Routing distinguishes pool-local
  (L6a) from inter-pool (L6b); the pool is the single unit the orchestrator reasons
  about, and a deployment binds one orchestrator.
- **One sim thread, one global clock.** The simulation runs single-threaded on one
  clock via a single tick loop; there is no concurrent mutation of sim state.
- **`Flow` is the only object L7 holds.** The deployment is erased behind the `Flow`
  trait (`on_arrival` / `tick` / `cluster`) at a single `dyn` dispatch point, so
  the tick loop is deployment-agnostic.
