# L4 Arch — model assembly

`arch/` turns model dimensions and parallel layout into typed cost-query models
for L5. It owns worklet composition, compiled CostTrees, model topology, and
wire-payload definitions. It never owns request or worker state.

The canonical layer contract is
`doc/detailed_design/L4.md`; this README is the source-tree guide.

## Read the tree in this order

1. `contract.rs` — the three L4↔L5 query contracts and their input types.
2. `model_cfg.rs` / `moe_model_cfg.rs` — dense and MoE model dimensions.
3. `config.rs` — serde selector surfaces.
4. `build.rs` — shared selector-to-concrete-model builders.
5. `llama3_*.rs` / `qwen3_*.rs` — concrete model composition.

## Contracts (`contract.rs`)

### Iter-wise

`IterwiseUnifiedModel` evaluates one complete iteration from
`UnifiedArchInput`:

- `groups: Vec<ArchGroupInput>` contains one group per independent
  attention-DP shard;
- `tokens_per_source_rank` is the FFN routing view;
- each group contains token counts, prefill chunk pairs, decode KV lengths, and
  cumulative KV occupancy.

The model fills caller-owned `slots` and `scratch` buffers and returns aggregate
`LeafMetrics`. `eval_iter_with_inputs` additionally captures typed
`SlotInput`s for `cost_log`.

Capacity/topology methods are:

- `total_kv_bytes_per_token()` — total all-layer/all-rank KV bytes for one token;
- `gpus_per_replica()` — physical extent of one replica;
- `num_attn_dp_groups()` — independent attention-DP groups;
- `num_attn_shards()` — GPUs in one attention shard.

The worker derives a partition's token capacity from:

```text
attn_kv_bytes_per_gpu × num_attn_shards
────────────────────────────────────────
       total_kv_bytes_per_token
```

### Layer-wise AFD

`AttnLayerwiseModel` consumes `AttnArchInput` and costs one attention layer.
`FfnLayerwiseModel` consumes `FfnArchInput` and costs
Bootstrap/Bridge/Terminal sections. Each side reports the full unsharded bytes
per token it sends across the attention↔FFN boundary.

The FFN section split follows the physical fusion:

```text
Bootstrap: pre(0)
Bridge(L): post(L) + pre(L+1)
Terminal:  post(last) + iteration epilogue
```

Section-aware manifests let the worker log each cost group without flattening
the layer-wise protocol into the iter-wise input type.

## Build path

Concrete builders follow:

```text
build_configs → resolve_configs → build → compiled CostTree section(s)
```

The profiling bridge is used only while building leaves. `build` compiles and
caches the stable tree shape, so evaluation performs no per-tick compilation or
scratch allocation. Homogeneous iter-wise layers use
`Scale { n: num_layers }`; AFD compiles its distinct sections once.

`build.rs` is the shared dispatch used by deployments and `timing-predict`.
Deployment-specific code validates legal arch/worker pairings but does not
reimplement model assembly.

## Dimensions and parallel layout

- `ModelCfg` loads dense-decoder dimensions.
- `MoeModelCfg` extends those facts with MoE dimensions.
- `ParallelCfg` and the Qwen3 family-specific parallel structs resolve TP, EP,
  attention-DP/HP, NVLink-domain, and GPU facts.
- `num_layers` / `sim_num_layers` truncation is applied before
  `build_configs`.
- Routing distribution is a build-time model fact, not an `ArchInput` field.

The arch reports resolved topology to L5/L6. Callers must not reconstruct GPU
extent or KV sharding from selector fields.

## Selector surface (`config.rs`)

Serde-tagged selectors are provider-first: selecting a tag reveals only that
variant's parameters.

| Contract | Wired selectors | Explicitly unsupported selectors |
|---|---|---|
| `IterArchSel` | `llama3_dense`, `llama3_dense_tp`, `llama3_dp_attn_tp_ffn`, `qwen3_moe_dp_attn_ep_ffn` | none |
| `AttnArchSel` | `qwen3_attn_tp` | `llama3_attn_tp` |
| `FfnArchSel` | `qwen3_ffn_moe` | `deepseek_ffn_moe` |

`ModelSpec` is flattened into every tag and carries `model_config`, layer
controls, and `fp8`. `ParamStruct`/`ProviderSchema` generate the launcher
schema; no second hand-maintained config union belongs here.

## Production pairings

- unified:
  - Llama3 dense/TP → `barebone`
  - Llama3 DP-attention/TP-FFN → `hp_unified`
  - Qwen3 MoE DP-attention/EP-FFN → `hp_unified`
- PD:
  - Llama3 TP prefill → Llama3 TP decode
  - Llama3 TP prefill → Llama3 DP-attention/TP-FFN decode
- AFD:
  - `qwen3_attn_tp` → `qwen3_ffn_moe`

## Authoring

Use the `impl-compose-arch` skill when adding an L4 model. Keep the concrete
model's `build_configs → resolve_configs → build` path in its family file, add
the smallest selector/build dispatch entry, and test cost consistency before
wiring a deployment.
