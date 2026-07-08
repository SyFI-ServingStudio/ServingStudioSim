# L2 Op — naming kernels into operations

An **op** is a named composition of one or more L1 kernels with `compile` / `eval`
entry points into the [CostTree](../timing/COST_TREE.md). It is the first layer
that gives a cost leaf a *semantic name* (`model.attn.o_proj`) and, for compound
ops, the op-specific math that turns one logical operation into several kernel
calls. L2 is **parallelism-agnostic and sim-state-agnostic**: an op sees per-rank
shapes and returns metrics; it never reads `ParallelConfig` or simulation state.

This is the practical, code-matching reference; the code is the ground truth.
For the layer overview see `doc/detailed_design/L2.md`.

## Two forms of op

| Form | Where it lives | Example |
|---|---|---|
| **Atomic** — `Op<K>`, a generic single-kernel wrapper | **no file**; instantiated at L4 wiring with `Op::new(name, kernel)` | `qkv`, `o_proj`, `gate_up`, `down`, `lm_head`, `rms_norm` |
| **Compound** — hand-written struct, multi-kernel + custom cost math | its own file under `op/<family>/` | `FlashInferAttentionOp` |

Adding an atomic op needs **no code here** — it is one `Op::new` line at the
wiring point (INV-2.5-1). Only a compound op (its input does *not* map 1:1 to a
single `*KernelInput`) gets a file.

## The compile / eval contract

Both forms expose the same pair, and they must mirror each other:

- **`compile(&mut CostTreeBuilder) -> CostNode`** — mint this op's leaf slot(s).
  An atomic op is one `Leaf`; a compound op is a small `Sum`/`Max` of fixed
  leaves. The leaf carries the op's dotted `name` plus the kernel's `kind` /
  one-line `config` (for the shape render).
- **`eval(&Input, &mut Evaluator)`** — push each leaf's `LeafMetrics` in the
  **same child order** `compile` minted slots (INV-2, so the evaluator cursor
  lines up with the slot index). The metric is the kernel's best-of-N `eval`.

The leaf count is **fixed at compile** regardless of request count (INV-1):
a variable per-request fan-out aggregates *into* a fixed slot rather than minting
one slot per request (see attention below).

`eval` captures each leaf's typed input into the `SlotInput` enum **only when the
evaluator is recording** (the closure passed to `Evaluator::push` doesn't run
otherwise). The `K::Input: Into<SlotInput>` bound is the central registry: a
kernel input not listed in `timing::slot_input::log_inputs!` fails to compile here.

## Worked compound op: `FlashInferAttentionOp`

One attention call over a sim batch dispatches to two L1 kernels
(`flashinfer_attn_prefill`, `flashinfer_attn_decode`):

- **Two fixed leaves** — `prefill` and `decode` — minted regardless of how many
  requests the step carries (INV-1).
- **`prefill` is an aggregating leaf**: `eval` sums `prefill.eval(prefix_i,
  append_i)` over every `(prefix_len, append_len)` in `prefill_chunk_pairs` into
  that one slot. A prefill/chunked request *is* the prefill kernel's cache cell,
  so it passes through with no conversion.
- **`decode` collapses** all decode requests to one cell `(batch_size = count,
  total_tokens = Σ kv)`.

This **v1 cost model** (per-request prefill sum + decode collapse) is exact when
a step has ≤1 prefill request — the common continuous-batching case — and
over-estimates only when several small-q prefills are batched. See
`agent-trace/attention_cache_fidelity.md`.

## What's in the tree

- **Live:** `Op<K>` (atomic) and `FlashInferAttentionOp` (compound).
- **Stubs:** `op/comm`, `op/moe`, `op/ssm` are one-line module headers. The
  compound ops they will hold (e.g. an MoE dispatch op, comm ops, SSM) are
  specified in `doc/detailed_design/L2.md`.

## Up / down

- **Below (required):** L1 timing kernels via the `Probe` trait
  (`eval(&Input) -> LeafMetrics`, `kind`, `describe_config`). An op holds its
  kernels as `Arc<K>` and forwards.
- **Above (consumer):** L3 worklets compose ops into model-module-level units,
  calling `compile`/`eval` on each op slot. The worklet never sees an op's
  sub-kernels or its internal normalization (INV-3.5-3).

Authoring: skill `impl-compose-op` — atomic is one `Op::new` line at the wiring
site; compound scaffolds `op/<family>/<name>.rs` (mirror `attention/flashinfer.rs`).
