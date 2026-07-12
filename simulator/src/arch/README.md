# L4 Arch — model assembly

A **model_arch** is the full-model wire file for one worker type. It picks an L3
worklet set, forwards the numeric `ModelCfg` + `ParallelCfg` 1:1 into each
worklet's config, assembles them into a model, and **compiles the per-iteration
CostTree once**. It is the layer that owns the parallelism layout and the
embedding→layers→lm_head structure, and it is the worker's (L5) sole view of "the
model".

This is the practical, code-matching reference; the code is the ground truth.
For the layer overview see `doc/detailed_design/L4.md`.

## The L4 ↔ L5 contract (`contract.rs`)

A worker hands the arch a per-iteration input and gets back one aggregated cost.
The arch never sees sim state; the worker never sees worklets/kernels.

- **`UnifiedArchInput`** — `groups: Vec<ArchGroupInput>` (dense local has exactly
  one) + `tokens_per_source_rank` (FFN routing view; empty for dense). An
  `ArchGroupInput` is the batch state: `batch_tokens`, the `prefill_tokens` /
  `decode_tokens` split, the `(prefix_len, append_len)` `prefill_chunk_pairs`,
  `decode_kv_lens`, `total_kv_len`. Its derived `request_count()` is the lm-head
  row count for this non-speculative input type; modes with different logits
  semantics use a different ArchInput type rather than optional future fields.
- **`IterwiseUnifiedModel`** — the iter-wise query face (one call costs the whole
  iteration):

| Method | Role |
|---|---|
| `eval_iter(batch, &mut slots, &mut scratch) -> LeafMetrics` | per-iter CostTree eval: fill `slots` with per-leaf metrics, aggregate through the caller-owned scratch buffer, return the root (`.m.time_ms` is the clock) |
| `eval_iter_with_inputs(batch, &mut slots, &mut scratch, &mut inputs)` | same, plus capture each leaf's typed `SlotInput` for cost_log (the clone is paid only here) |
| `cost_log_manifest() -> CostManifest` | the slots + flat nodes so a consumer reproduces `total_time_ms` from a row |
| `kv_bytes_per_token() -> u64` | KV footprint per token; the worker divides its memory budget by this to size its `KvPool`. TP archs report the **per-rank** (tp-head-sharded) footprint; the dense arch reports the full model's |
| `gpus_per_replica() -> u16` | GPUs one replica spans (the arch knows the real parallel extent); L5/L6 only read it |

## The build shape (every arch follows it)

```
build_configs(&ModelCfg, &ParallelCfg) -> *Configs     // raw worklet/op configs (gpu_name baked)
        │ resolve_configs
        ▼
   *Resolved                                            // per-rank shapes + sub-configs baked
        │ build(name, resolved, bridge)
        ▼
   Model (holds the worklets/ops + the compiled, flattened CostTree)
```

`build` instantiates each worklet/op (which profiles its kernels through the
`bridge`), then **compiles the CostTree once** and caches its flattened form +
slot count on the model, so the per-iter `eval_iter` only evals leaves and
aggregates through caller-owned buffers — no per-tick recompile, `String` minting,
or aggregate scratch allocation. A dry-run `bridge` turns `build` into a coverage
tally (no separate traversal).

## The compiled cost structure

```rust
// Llama3DenseModel::cost_tree()
Labeled{ "<name> [dense local, N layers]" }
  Sum( embed,
       Scale{ n=num_layers }( Sum(pre_attn, attn, post_attn) ),   // homogeneous-layer fold
       final_norm,
       lm_head )
```

The `Scale{num_layers}` fold is the economy: **one** worklet instance per type is
built and its leaves minted once; the fold supplies the `×num_layers` at
aggregate (the layers are never materialized N times). `eval_into` streams the
leaves in the exact order `cost_tree` minted slots — embed, then one layer's
pre/attn/post, then final_norm, lm_head — so the evaluator cursor stays aligned
(see [COST_TREE.md](../timing/COST_TREE.md)).

## Model dims & parallelism (`model_cfg.rs`)

- **`ModelCfg`** — the resolved numeric transformer dims (hidden, heads, layers,
  dtype, …), parallelism-agnostic. `from_json` loads a HuggingFace `config.json`;
  that is the only runtime source of dims.
- **`ParallelCfg`** — the resolved `tp_size` / `ep_size` / `num_hp_groups`
  degrees + `gpu_name`. `gpu_name` is the single source of truth threaded into
  every kernel lookup. Dense local = all degrees 1 (`ParallelCfg::local`).

## Arch selectors (`config.rs`)

Serde tagged enums, the symmetric sibling of the worker selector, **provider-first**
(selecting the tag is the only way its params appear — no global union):

- `IterArchSel` — `llama3_dense`, `llama3_dense_tp`, and
  `llama3_dp_attn_tp_ffn` are wired for their matching workers/deployments;
  `deepseek_moe` parses but `build` bails. `tp_size` is **absent only from
  `Llama3Dense`**. `Llama3DenseTp` carries `tp_size`; `Llama3DpAttnTpFfn` carries
  `attn_tp_size` / `ffn_tp_size`; `DeepseekMoe` carries `tp_size` + `ep_size`.
  `IterArchSel::model()` returns the `ModelSpec` every variant flattens.
- `AttnArchSel` / `FfnArchSel` — the AFD layer-wise contract (config-only today).
- `ModelSpec` is flattened into every arch tag (`model_config`, `num_layers` /
  `sim_num_layers`, `fp8`); `model_config` + `fp8` are `#[param(cache_key)]`
  (they change which kernels are needed). `#[derive(ParamStruct)]` /
  `#[derive(ProviderSchema)]` emit the launcher `list-params` schema.

## Current set & up/down

- **Live archs:** `llama3_dense` (Local, single GPU), `llama3_dense_tp`
  (tensor-parallel, ends each block in a `tp_allreduce`), and
  `llama3_dp_attn_tp_ffn` (attention-DP + FFN-TP, paired with `hp_unified` or PD
  decode).
- **Below (assembled):** L3 worklets + L2 atomic ops, and through the CostTree,
  L1 kernels.
- **Above (consumer):** the `deployment` layer runs the
  `build_configs → resolve_configs → build` cascade and selects the arch by tag;
  the L5 worker then holds the model as `Arc<M>` and calls `eval_iter`.

## Authoring

Adding an arch: skill `impl-compose-arch` (scaffold `arch/<family>.rs` + register
the `config.rs` selector, mirror `llama3_dense.rs`).
