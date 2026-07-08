# L3 Worklet — model-module compositions

A **worklet** is one model-module-level unit of a decoder layer (the
pre-attention section, the attention block, the MLP block, …) assembled from L2
ops, with `compile` / `eval` entry points into the [CostTree](../timing/COST_TREE.md).
It is where a parallelism scheme's **per-rank partition** is derived and the
sub-op/sub-kernel configs are baked. A worklet still returns only metrics — it
owns no sim state.

This is the practical, code-matching reference; the code is the ground truth.
For the layer overview see `doc/detailed_design/L3.md`.

## The worklet shape (every file follows it)

Each worklet is a hand-written struct exposing the same members. `pre_attn_local.rs`
is the smallest example; the rest scale it up:

| Member | Role |
|---|---|
| `*Config` | raw global config + parallelism degree (`tp_size`) + collective fabric/backends. Attention configs carry a separate `kv_dtype` (KV-cache dtype, distinct from the activation `dtype`; see `attn_block_tp.rs:39`) |
| `*Resolved` | **pure data**: post-partition per-rank shapes with every sub-kernel/op config fully baked |
| `*Input` | per-call **shape only** (`batch_tokens`, and for attention the `(prefix_len, append_len)` prefill pairs + `decode_kv_lens`) |
| `resolve_config(&Config) -> Resolved` | the one place partition math lives; bakes `gpu_name` + backends into each sub-config |
| `build(name, Resolved, bridge) -> Self` | instantiate the `Op` slots (each `Op::new` over a built `*Kernel`) |
| `compile(&mut CostTreeBuilder) -> CostNode` | a `Sum` of the op slots, wrapped in a `Labeled` node carrying the worklet identity + partition annotation |
| `eval(&Input, &mut Evaluator)` | fill the slots in the **exact `compile` child order** (INV-2) |

`gpu_name` rides in `*Config` and is baked into each sub-kernel config at
`resolve_config`; `*Input` is pure shape. Backend strings pass straight through to
L1 (config-level polymorphism) — a worklet never selects a backend.

The `Labeled` wrapper is render-only: it tags the composite subtree (e.g.
`"m.attn (AttnBlockTpWorklet) [tp=4; ...]"`) for the per-worker `CostManifest`
so an analyzer can find a worklet's subtree structurally, and it is dropped from
the hot-path flat tree (INV-5).

## `Local` vs `TP` (the group suffix)

The suffix names the worklet's **sync grain** (L3 §1.5):

- **`Local`** — one GPU, self-synced, **no collective**. The three dense-layer
  sections: `PreAttnLocalWorklet` (input RMSNorm → fused QKV),
  `AttnLocalWorklet`, `PostAttnLocalWorklet`.
- **`TP`** — one tensor-parallel sync section whose boundary is the all-reduce
  that re-syncs a row-parallel output across `tp_size` ranks. `AttnBlockTpWorklet`
  (norm → col-parallel QKV → attn over per-rank heads → row-parallel o_proj →
  `tp_allreduce`) and `MlpBlockTpWorklet` (norm → col-parallel up_gate → SwiGLU →
  row-parallel down → `tp_allreduce`).

`resolve_config` does the Megatron sharding: column-parallel projections split
the output dim per rank, the activation/attention run on local shards, and the
row-parallel output's full `[tokens × hidden]` partial-sum is all-reduced.
**`hidden` is never sharded.** `tp_size == 1` degenerates to the `Local` shape:
per-rank == full and the `tp_ar` slot is `None`, so the cost matches the
single-GPU path exactly.

Two TP invariants that are easy to miss:

- **`tp <= num_kv_heads`** — both head counts must divide `tp`, and `tp` may not
  exceed `num_kv_heads`: v1 shards KV heads across ranks with no KV-head
  replication (asserted in `resolve_config`).
- **All-reduce message size** is the full `[tokens × hidden]` partial-sum, **not**
  `hidden/tp` — the row-parallel o_proj produces a complete `hidden`-wide output
  per rank that must be summed (`attn_block_tp.rs` `eval`).

## Up / down

- **Below (required):** L2 ops — atomic `Op<K>` (norm/gemm/activation/all-reduce)
  and the compound `FlashInferAttentionOp`. A worklet holds them as fields and
  calls their `compile`/`eval`.
- **Above (consumer):** L4 model/arch assembles worklets into a full model,
  repeating the homogeneous decoder layer via a CostTree `Scale{n}` fold rather
  than materializing it N times.

## Current set

`pre_attn_local`, `attn_local`, `post_attn_local` (the dense `arch::llama3_dense`
layer); `attn_block_tp`, `mlp_block_tp` (the tensor-parallel `arch::llama3_dense_tp`
layer). Each module re-exports its `{Worklet, Config, Input, Resolved}` quartet
through `mod.rs`.

## Authoring

Adding a worklet: skill `impl-compose-worklet` (scaffold
`worklet/<family>_<suffix>.rs`, mirror `attn_block_tp.rs`).
