# L4 Arch — model assembly

`arch/` turns model dimensions and parallel layout into typed cost-query models
for L5. It owns worklet composition, compiled CostTrees, model topology, and
wire-payload definitions. It never owns request or worker state.

The canonical layer contract is
`doc/detailed_design/L4.md`; this README is the source-tree guide.

## Read the tree in this order

1. `contract.rs` — the three L4↔L5 query contracts and their input types.
2. `kv_layout.rs` — `KvCacheLayout`, the per-request KV state the contracts
   report.
3. `model_cfg.rs` / `moe_model_cfg.rs` — dense and MoE model dimensions.
4. `config.rs` — serde selector surfaces.
5. `build.rs` — shared selector-to-concrete-model builders.
6. `llama3_*.rs` / `qwen3_*.rs` / `glm52_*.rs` — concrete model composition.

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

- `kv_cache_layout()` — one request's persistent KV state as a `KvCacheLayout`
  (`kv_layout.rs`), part by part, bytes summed over one KV partition's ranks;
- `gpus_per_replica()` — physical extent of one replica;
- `num_attn_dp_groups()` — independent attention-DP groups;
- `num_attn_shards()` — GPUs in one attention shard.

`kv_cache_layout()` is required on all three KV-owning contracts
(`IterwiseUnifiedModel`, `SpeculativeUnifiedModel`, `AttnLayerwiseModel`). Each
`KvCachePart` says how it grows (`KvGrowth::PerToken { tokens_per_entry }`,
`SlidingWindow { window_tokens }`, `PerRequest`), its `bytes_per_entry`, whether
a prefix hit can resume from it (`prefix_reusable`), and what vLLM allocates on
top (`PartOverhead`: padded entry bytes, extra copies). `EngineOverhead` holds
the pool-wide terms: a large engine block and fixed device bytes. Use
`gqa_kv_bytes_per_token` for GQA layers; it counts the heads replicated when TP
exceeds `num_kv_heads`.

The worker resolves the layout into KV units against the partition budget
`attn_kv_bytes_per_gpu × num_attn_shards` (L5 `KvAccounting`); a layout of one
per-token part counts plain tokens. Contract and rationale:
`doc/detailed_design/L4.md` (KV cache layout).

### Speculative Iter-wise

`SpeculativeUnifiedModel` consumes `SpeculativeArchInput`: each decode request
carries its final verify-row KV length and query width separately. The compiled
tree includes target verification and draft passes. Offline prediction selects
`arch.speculative_iter` with the same speculative architecture selector used by
deployment; its cases contain `decode_requests: [[final_kv_len, query_len], ...]`.
`build_speculative_iter_model` constructs this contract without a worker.

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
- `MoeModelCfg` extends those facts with MoE dimensions. GLM-5.2's DSA/MoE
  dimensions live in `Glm52ModelCfg` (`glm52_model_cfg.rs`), because its
  heterogeneous 78-layer schedule and indexer dimensions are not a `MoeModelCfg`
  extension.
- `InklingModelCfg` (`inkling_model_cfg.rs`) reads Inkling's `text_config` and
  its ModelOpt NVFP4 exclusions: the sliding-window/global attention schedule,
  the dense-FFN layers, the sink experts and each MoE layer's expert precision.
- `ParallelCfg` and the family-specific parallel structs resolve TP, EP,
  attention-DP/HP, NVLink-domain, and GPU facts.
- `num_layers` / `sim_num_layers` truncation is applied before
  `build_configs`.
- Routing distribution is a build-time model fact, not an `ArchInput` field.

The arch reports resolved topology to L5/L6. Callers must not reconstruct GPU
extent or KV sharding from selector fields.

## Selector surface (`config.rs`)

Serde-tagged selectors are provider-first: selecting a tag reveals only that
variant's parameters.

| Contract | Wired selectors |
|---|---|
| `IterArchSel` | `qwen36_local`, `llama3_dense`, `llama3_dense_tp`, `llama3_dp_attn_tp_ffn`, `qwen3_moe_dp_attn_ep_ffn`, `qwen3_moe_fp8_dp_attn_ep_ffn`, `qwen3_vllm_moe_dp_attn_ep_ffn`, `deepseek_v4_vllm`, `deepseek_v4_vllm_serial_streams`, `deepseek_v41_vllm`, `deepseek_v41_vllm_serial_streams`, `glm52_vllm_dsa_moe`, `glm52_vllm_nvfp4_dsa_moe`, `glm52_vllm_nvfp4_dsa_moe_speculative`, `glm52_vllm_nvfp4_pp_dsa_moe`, `glm52_vllm_nvfp4_dp_attn_dsa_moe`, `glm53_vllm_nvfp4_dsa_moe_dflash2`, `glm52_sglang_nvfp4_tp_dsa_moe`, `glm53_flash_vllm_{fp8,nvfp4}_kda_dsa_moe`, `glm53_flash_vllm_{fp8,nvfp4}_pp_kda_dsa_moe`, `glm53_flash_vllm_{fp8,nvfp4}_dp_attn_ep_moe`, `inkling_vllm_nvfp4_rel_attn_moe`, `mimo_v2_vllm_fp8_swa_moe`, `qwen35_vllm_nvfp4_gdn_dense`, `qwen4_exp_vllm_nvfp4_gdn_qsa_moe_speculative` |
| `AttnArchSel` | `qwen3_attn_tp` |
| `FfnArchSel` | `qwen3_ffn_moe`, `qwen3_fp8_ffn_moe` |

`ModelSpec` is flattened into every tag and carries `model_config`, layer
controls, and `fp8`. `ParamStruct`/`ProviderSchema` generate the launcher
schema; no second hand-maintained config union belongs here.

## Production pairings

- unified:
  - Llama3 dense/TP → `barebone`
  - Llama3 DP-attention/TP-FFN → `hp_unified`
  - Qwen3 MoE DP-attention/EP-FFN (native BF16, native FP8, or vLLM-aligned FP8) → `hp_unified`
  - GLM-5.2 vLLM DSA/MoE → `hp_unified`
  - GLM-5.2 vLLM NVFP4 → `hp_unified`, `chunked_prefill`, or `speculative`
  - GLM-5.2/5.3 vLLM NVFP4 DP attention + EP (`glm52_vllm_nvfp4_dp_attn_dsa_moe`)
    → `hp_unified` or `chunked_prefill`, one KV partition per DP rank
  - GLM-5.2 SGLang NVFP4 pure TP → `chunked_prefill`
- pp:
  - GLM-5.2/5.3 vLLM NVFP4 pipeline stages (`glm52_vllm_nvfp4_pp_dsa_moe`) →
    pipeline stage workers; `unified` rejects it. Each stage is its own
    `IterwiseUnifiedModel` (one GPU, its own layers' KV); the whole-pipeline
    model sums the stages for `timing-predict` and shares them by `Arc`.
  - GLM-5.3-Flash vLLM FP8/NVFP4 pipeline stages
    (`glm53_flash_vllm_{fp8,nvfp4}_pp_kda_dsa_moe`) → the same stage workers.
    `layer_partition` overrides the `get_pp_indices` split (vLLM
    `VLLM_PP_LAYER_PARTITION`). Each stage labels its layer groups with the
    layers they hold, which `tools/pp-layer-balance/pp_layer_balance.py` reads
    to pick a balanced split from a finished run.
- PD:
  - Llama3 TP prefill → Llama3 TP decode
  - Llama3 TP prefill → Llama3 DP-attention/TP-FFN decode
- AFD:
  - Qwen3 BF16 attention + `qwen3_ffn_moe`
  - Qwen3 FP8 attention + `qwen3_fp8_ffn_moe`

## Authoring

Use the `impl-compose-arch` skill when adding an L4 model. Keep the concrete
model's `build_configs → resolve_configs → build` path in its family file, add
the smallest selector/build dispatch entry, and test cost consistency before
wiring a deployment.

See [`doc/architecture_compatibility.md`](../../../doc/architecture_compatibility.md)
for selector compatibility and migration rules.

## MoE routing configuration

`routing` names where a MoE arch's routed demand comes from.

`uniform` (the schema default) and seeded `random` are synthetic marginals and
reject every measured artifact. `popularity` requires `expert_popularity_file`
and loads a measured per-expert marginal; migrate older `uniform` + file configs
to it. `corpus` requires `token_corpus_file` and samples recorded per-token
routes. A missing or invalid artifact fails; the runtime never falls back to
uniform. AFD FFN selectors with no artifact field support only synthetic
routing.

Resampling a marginal draws tokens independently, and a serving batch is not
independent: under speculative decoding one request submits `draft_tokens + 1`
consecutive positions that route almost alike. A grouped GEMM is billed partly
by how many of a rank's expert groups are non-empty, which is a statement about
co-occurrence and so unanswerable from a marginal — that is what `corpus` is
for, and it only differs from `popularity` where the verify width exceeds one.

The two sources also differ in what they can say about a speculative model's
MTP layer. A corpus is layer-indexed, so the body MoE and the MTP MoE are two
slices of one file and the MTP layer's own routing is measured. A marginal has
already summed the layer axis away and was captured over the body's layers, so
the MTP layer folds that profile's layer-summed distribution -- the same
evidence at the only resolution a marginal has. One file either way; there is no
separate draft profile.

A corpus records the body's routed layers, plus one MTP slot when the capture
ran a drafter. A build that prices an MTP layer needs that slot; a build that
does not may still read a drafted capture, which is how the same traffic is
priced without speculation.
