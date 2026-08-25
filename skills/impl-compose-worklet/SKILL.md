---
name: impl-compose-worklet
description: >-
  Use when implementing an L3 worklet under simulator/src/worklet/ after its L2
  ops build — one model-module sync section that derives the per-rank partition
  and composes ops with sum/max over the CostTree. Covers the live code contract:
  Config/Resolved/Input + resolve_config/build/compile/eval, and the group suffix
  (Local/TP/HP/EP). Follows the code, not design.md's retired
  init_ops/dry_run_init_ops/lookup/Describe. NOT for L2 ops (impl-compose-op) or
  L4 arch (impl-compose-arch).
---

# Impl Compose Worklet

You implement one L3 worklet under `simulator/src/worklet/`, a single sync section
(one GPU group, ending at a collective or single-GPU self-completion). Its L2 ops
must already build. Do not add ops or wire the arch here.

**Code is the contract**: the live worklet
uses `resolve_config` / `build` / `compile` / `eval` over the CostTree — NOT the
design doc's `init_ops` / `dry_run_init_ops` / `lookup` / `Describe` (retired; the
partition label is now folded into a `CostNode::Labeled`).

## Read First

- `simulator/src/worklet/README.md` — the worklet member table + the group suffix.
- `simulator/src/worklet/attn_block_tp.rs` — the reference TP worklet: partition
  math + `#[should_panic]` divisibility, inclusion conditional (`Option<...>`),
  `Labeled` header, all-reduce message sizing.
- `simulator/src/worklet/pre_attn_local.rs` — the smallest `Local` worklet.
- `simulator/src/timing/COST_TREE.md` — `CostNode` / `CostTreeBuilder` / `Evaluator`.

## File

`worklet/<family>_<suffix>.rs`, one per module, re-exporting its
`{Worklet, Config, Input, Resolved}` quartet through `mod.rs`. The `<suffix>` is
the mandatory sync grain: `local` / `tp` / `hp` / `ep` / `hptp` / `eptp`.

## Contract (mirror `attn_block_tp.rs`)

- `*Config` — raw global dims + parallelism degree (`tp_size`/…) + collective
  `Fabric` + per-role `backends` vecs + `gpu_name`. No partition yet.
- `*Resolved` — **pure data**: `raw_cfg: *Config` (kept for the label + KV
  accounting) + each sub-op/sub-kernel `*Config` baked + per-rank facts
  (`num_qo_heads_per_rank`, `dtype_bytes`); a conditional collective is `Option<...>`.
- `*Input` — per-call **shape only** (`batch_tokens`, attention's
  `prefill_chunk_pairs` + `decode_kv_lens`). No `gpu_name` (it rode in `*Config`).
- `resolve_config(&Config) -> Resolved` — **the one place partition math lives**:
  divisibility asserts, per-rank dim derivation, bake `gpu_name` + backends into
  each sub-config; inclusion conditional via `(tp_size > 1).then(|| ...)`. Pure —
  no bridge / GPU / `Arc`.
- `build(name, Resolved, bridge) -> Result<Self, BuildError>` — instantiate op
  slots (`Op::new(name, Arc::new(K::build(...)))` / `*CompoundOp::build`) with
  `format!("{name}.<slot>")` names; move `resolved` into `Self`.
- `compile(&self, builder) -> CostNode` — `CostNode::Sum` of each slot's
  `.compile(builder)`, wrapped in `CostNode::Labeled { label, child }` carrying the
  worklet identity + partition summary (render-only, dropped from the hot path).
- `eval(&self, input, ev)` — derive trivial shapes (`m = batch_tokens`), fill slots
  in the exact `compile` order; a conditional slot uses `if let Some(slot)`.
  Collective message sizes are computed here (all-reduce = the full
  `[tokens × hidden]` partial-sum, NOT `hidden/tp`).

## Rules

- Group suffix mandatory + closed set; divisibility asserts belong in
  `resolve_config` (V5). `hidden` is never sharded; `tp_size == 1` must degenerate
  to the `Local` shape (per-rank == full, collective slot `None`).
- `resolve_config` is pure (no bridge/GPU/Arc) (V4).
- `build`/`eval` enumerate slots in the same order as `compile` (V2 / INV-2);
  name-path threads `format!("{name}.<slot>")` matching the field (V3).
- No op-slot polymorphism — slot fields are concrete `Op<K>` / `*Op` / `Option<...>`,
  never `enum` / `Box<dyn>` (V7). Backends pass through as config strings.

Reconstruct the production dependency graph before choosing `Sum` or `Max`.
Mixed prefill/decode inputs share one common projection spine when production
does; do not concatenate two complete phase worklets and double-charge common
work. A `Max` may contain only children launched from the same source fanout and
must close at the real join/barrier before later work begins. A compound L1 slot
retains every fixed prologue/tail launch inside its public boundary.

Do not mint slots for runtime chunks, ranks, layer repetition, or capacity. If
the L1 callable owns a loop or fixed multi-launch sequence, it remains one
semantic slot. A serial-stream counterfactual changes only `Max` versus ordered
`Sum`; it preserves the same leaves, inputs, and source barriers.

Profile DB compatibility must not determine worklet semantics. When a
source-backed workload derivation changes, propagate the new physical input and
re-profile rather than adding a model-specific conversion solely to hit old
keys.

## Tests And Smoke

- Unit-test only behavior-bearing boundaries with **no bridge**: partition
  invariants, invalid divisibility, and one observable CostTree/input-order test
  when fanout/barrier structure is non-trivial. Do not duplicate field lists or
  implementation formulas in tests.
- `just test-cpu`, or `uv run cargo test -p simulator worklet::`.
- Full leaf smoke happens through the arch dry-run (`impl-compose-arch`).

## Report Back

Return: file + suffix; `*Config` / `*Resolved` / `*Input` field lists; the partition
formula + divisibility asserts; the compile slot order + which slot is conditional;
`cargo test` output; and the arch slot this worklet fills (handoff to
`impl-compose-arch`).
