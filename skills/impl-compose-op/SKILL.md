---
name: impl-compose-op
description: >-
  Use when implementing an L2 op under simulator/src/op/ after
  top-split-model-into-kernels assigned an op to "reuse an existing kernel"
  (atomic) or "new compound op", and the L1 kernels it wraps already build.
  Covers the live code contract — atomic Op<K> (one Op::new line, no file) vs a
  compound op file with Config/Input/Self + build/compile/eval over the CostTree.
  Follows the code, not design.md's retired init_ops/dry_run_init_ops/lookup/
  LookupResult. NOT for L1 kernels (impl-register-kernel / impl-wire-kernel-to-rust)
  or L3 worklets (impl-compose-worklet).
---

# Impl Compose Op

You implement one L2 op under `simulator/src/op/`. `top-add-new-arch` Phase 2 (via
`top-split-model-into-kernels`) should already have fixed this op's boundary —
reuse an existing kernel as an atomic `Op<K>`, or a new compound op — and the L1
kernels it wraps must already build. Do not add kernels or worklets here.

**The code is the contract**, not `docs/detailed_design/L2/design.md`: the live
op layer uses a **CostTree** (`build` / `compile` / `eval`), not the design doc's
`init_ops` / `dry_run_init_ops` / `lookup -> LookupResult` (retired). Read the
README and a real op before writing.

## Read First

- `simulator/src/op/README.md` — atomic vs compound, the compile/eval contract.
- `simulator/src/op/attention/flashinfer.rs` — the reference compound op: `Config`,
  `Input`, `Self`, `build`, `compile`, `eval`, pure helper fns + unit tests.
- `simulator/src/timing/COST_TREE.md` + `timing/README.md` — `CostNode`,
  `CostTreeBuilder`, `Evaluator`, `LeafMetrics`, `Probe`.
- `simulator/src/timing/slot_input.rs` — the closed `SlotInput` enum a leaf input
  must join.

## Two forms

- **Atomic** (`Op<K>`): the op's input maps 1:1 to one `*KernelInput`. **No file
  here** — the worklet/arch build site adds one
  `Op::new(name, Arc::new(K::build(...)))` line (INV-2.5-1). Nothing to write in `op/`.
- **Compound** (input does not map 1:1): a hand-written `op/<family>/<name>.rs`.

## Compound op contract (mirror `flashinfer.rs`)

- `*Config` — raw op-level fields (dims, `dtype`, `fp8`, `backends`, `gpu_name`);
  it expands into each sub-kernel's `*KernelConfig` via **pure helper fns**.
- `*Input` — op-level per-call data (NOT a `*KernelInput`); partition/dispatch
  happen inside `eval`, so L3 never sees the sub-kernels (INV-3.5-3).
- `Self` — `name: String` + each sub-kernel as `Arc<*Kernel>`.
- `build(name, cfg, bridge) -> Result<Self, BuildError>` — construct each
  sub-kernel via `*Kernel::build(format!("{name}.<slot>"), subcfg, bridge)`.
- `compile(&self, builder: &mut CostTreeBuilder) -> CostNode` — mint a **fixed**
  set of leaves (independent of request count, INV-1) via
  `builder.leaf(name, kernel.kind(), kernel.describe_config())`, combined with
  `CostNode::Sum` / `Max`.
- `eval(&self, input, ev: &mut Evaluator)` — fill slots in the **exact `compile`
  child order** (INV-2). A per-request fan-out aggregates INTO one slot
  (`LeafMetrics::add` in a loop, then `ev.push(metrics, || log.into())`), never one
  slot per request.
- pure helper fns (config expansion, input collapse) — unit-tested without a bridge.

**Boundary:** keep it one op only if the sub-kernels share the op's algorithm
state (attention prefill/decode; MoE dispatch inter+intra all-to-all sharing the
routing math). Comm that is a parallelism artifact (TP all-reduce, HP gather)
stays a separate atomic `Op<K>` the worklet strings in — not merged here (INV-6.1).

## Rules

- Leaf count fixed at `compile`, request-count-independent (INV-1);
  `aggregate(Sum[...])` must reproduce the streamed slot total.
- `eval` pushes in the same order `compile` minted slots (INV-2).
- A new leaf input type must be added to `timing/slot_input.rs`, or the
  `K::Input: Into<SlotInput>` bound in `eval` fails to compile.
- Parallelism-agnostic: an op sees per-rank facts only, never `ParallelCfg`
  (INV-4.5). `gpu_name` is an explicit `*Config` field baked into each sub-kernel.
- Backend selection is config-level (`backends: Vec<&'static str>` on the kernel),
  transparent to the op.

## Tests And Smoke

- Unit-test the pure helpers (config expansion, input collapse) with **no bridge**,
  as in `flashinfer.rs`'s `#[cfg(test)] mod tests`.
- Build + test: `just test-cpu` (Rust `--lib` + mocked pytest), or
  `uv run cargo test -p simulator op::` with libpython on `LD_LIBRARY_PATH`.
- The op reaches a leaf only through a worklet/arch — a full leaf smoke is the arch
  dry-run (see `impl-compose-arch`).

## Report Back

Return: files changed; the `*Config` / `*Input` field lists; the compile leaf slots
(names + kernel kinds) in order; any new `SlotInput` variant; the
atomic-vs-compound decision + boundary justification; `cargo test` output; and the
worklet slot this op plugs into (handoff to `impl-compose-worklet`).
