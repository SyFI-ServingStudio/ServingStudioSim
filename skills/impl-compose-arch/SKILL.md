---
name: impl-compose-arch
description: >-
  Use when implementing an L4 model_arch under simulator/src/arch/ after its L3
  worklets build — the full-model wire file for one worker type that forwards
  ModelCfg into worklet configs, compiles the per-iter CostTree once, and exposes
  IterwiseUnifiedModel. Covers the live code contract: build_configs/resolve_configs/
  build + cost_tree/eval_iter + the Scale{n} layer fold + the config.rs selector.
  Follows the code, not design.md's retired dry_run/JitPlan/Describe. NOT for L3
  worklets (impl-compose-worklet).
---

# Impl Compose Arch

You implement one L4 `model_arch` under `simulator/src/arch/`, the full-model wire
for one worker type. Its L3 worklets must already build. This is the top of the
bottom-up build; after it, register the arch selector so the launcher can run it.

**Code is the contract**, not `docs/detailed_design/L4/design.md`: the live arch
uses `build_configs` / `resolve_configs` / `build` + `cost_tree` +
`IterwiseUnifiedModel` over the CostTree — NOT the design doc's `dry_run` →
`JitPlan` or `Describe` (retired; dry-run is now `build` against a dry-run bridge,
and the label render lives in `CostNode::Labeled`).

## Read First

- `simulator/src/arch/README.md` — the L4↔L5 contract, the build shape, the
  `Scale{n}` fold.
- `simulator/src/arch/llama3_dense.rs` — the reference dense-local arch (every
  member below); `arch/llama3_dense_tp.rs` for a TP variant.
- `simulator/src/arch/contract.rs` — `IterwiseUnifiedModel` / `UnifiedArchInput`.
- `simulator/src/arch/config.rs` + `model_cfg.rs` — the arch selector enum + `ModelCfg`.

## File & scope

`arch/<family>.rs` (or `<family>_{attn,ffn}.rs` for split worker types),
re-exported via `mod.rs`. You also register the selector in `arch/config.rs`
(`IterArchSel` tagged enum) so the launcher/deployment can pick it. Different
backends/algorithms = different arch files (no in-file if/match polymorphism).

## Contract (mirror `llama3_dense.rs`)

- a per-arch numeric parallel struct (e.g. `DenseParallel { gpu_name }`) — each
  arch owns the parallel dims it needs (the shared `ParallelCfg` union is retired).
- `*Configs` aggregate — one field per worklet/op `*Config` + `num_layers`.
- `*Resolved` aggregate — one field per worklet `*Resolved` (atomic ops pass their
  `*KernelConfig` straight through).
- `*Model` — `name` + the built worklets/ops + `cost_flat: Vec<FlatCostNode>` +
  `n_slots` (the CostTree compiled once at build).
- `build_configs(&ModelCfg, &Parallel) -> *Configs` — forward dims 1:1, bake
  `gpu_name`, pick each role's `backends`. **No bridge.**
- `resolve_configs(&Configs) -> *Resolved` — call each worklet
  `::resolve_config(&cfgs.<field>)`; nothing else.
- `build(name, resolved, bridge) -> Result<*Model, BuildError>` — `Op::new` /
  `*Worklet::build` each member with `format!("{name}.<slot>")`, then compile the
  tree once: `let tree = model.cost_tree(); model.cost_flat = tree.flatten();
  model.n_slots = tree.n_slots();`.
- `cost_tree(&self) -> CostTree` — build with `CostTreeBuilder`; wrap the
  homogeneous decoder layer in `CostNode::Scale { n: num_layers, child }` (leaves
  minted once, the fold supplies `×num_layers`); a `Labeled` root; `b.finish(root)`.
- `eval_into(&self, batch, ev)` — the single eval body, streaming leaves in exact
  `cost_tree` order (the cursor must fill `n_slots`).
- `impl IterwiseUnifiedModel` — `eval_iter` (fill `slots`, `CostTree::aggregate`),
  `eval_iter_with_inputs` (same via `Evaluator::with_inputs` for cost_log),
  `cost_log_manifest`, `total_kv_bytes_per_token`, `gpus_per_replica`.
- an arch-specific `total_kv_bytes_per_token(resolved)` (KV layout is arch
  knowledge — GQA vs MLA differ — so it lives here, not on `ModelCfg`).

## Rules

- `build_configs` never touches the bridge (M2); `resolve_configs` only calls each
  worklet's `resolve_config` (M3).
- Dry-run = `build` against a dry-run bridge (it tallies missing specs); there is no
  separate traversal / `JitPlan`.
- `Scale{num_layers}` mints per-layer leaves once; `eval_into` streams in exactly
  the order `cost_tree` minted them.
- `gpu_name` is the single source of truth threaded into every kernel lookup.
- Multi-rank fan-out uses `CostNode::Max` / `LeafMetrics` with overlap_factor `1.0`
  at L4 — real <1.0 overlap only lives inside an L3 worklet.

## Tests And Smoke

- Unit-test `build_configs` (dims + `gpu_name` threaded) and `resolve_configs`
  (worklet shapes baked) with **no bridge** (as `llama3_dense.rs`).
- `just test-cpu`.
- Real smoke: run the launcher dry-run so `build` runs against a dry-run bridge —
  `uv run python -m launcher <preset>.json --dry-run` (the arch must be registered
  in `config.rs` and reachable from a preset). It prints the compiled cost-tree
  manifest + the missing-spec tally.

**Next — hand off to `impl-wire-new-arch`.** This skill's proof is the no-bridge
unit tests (the arch builds in isolation). Making the arch *reachable* from a
config/predictor and a worker — the `arch/build.rs` builder + its
`build_{iter,attn,ffn}_model` predictor dispatch, and the deployment arm — is the
wiring phase. The launcher-reachable dry-run / timing-predict smoke above only
passes once that wiring exists.

## Report Back

Return: files changed (arch file + `config.rs` selector); the `*Configs` /
`*Resolved` aggregates; the `cost_tree` structure (Sum/Scale shape) + slot count;
the `IterwiseUnifiedModel` methods implemented; `cargo test` + launcher dry-run
output; and any unsupported gpu/dtype the `build` rejects.
