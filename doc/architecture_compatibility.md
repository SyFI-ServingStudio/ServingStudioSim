# Architecture compatibility

Architecture selectors identify an execution graph, not only a model family.
Two selectors may share a checkpoint while representing different parallel
layouts, precision, worker contracts, or framework kernel boundaries. Those
differences determine whether both selectors remain supported.

## Compatibility rule

An unaligned architecture is retired when an aligned architecture supports the
same model, precision, worker contract, and complete parallel-layout domain. It
is retained when it covers any parallel configuration the aligned graph cannot
represent.

The generated launcher schema is fail-closed. Retired selector names are not
accepted as aliases and are not silently redirected to a different CostTree.
This keeps stored parameters, cost logs, and model-work location maps tied to
the graph that produced them.

## Current coverage

| Selector | Precision and execution graph | Parallel coverage | Why it remains |
|---|---|---|---|
| `llama3_dense` | dense local | one GPU | only local dense graph |
| `llama3_dense_tp` | dense TP | one configurable TP group | covers TP layouts |
| `llama3_dp_attn_tp_ffn` | dense split attention/FFN | `attn_tp_size` divides `ffn_tp_size`; attention DP is `ffn_tp_size / attn_tp_size` | covers distinct attention-DP layouts |
| `qwen3_moe_dp_attn_ep_ffn` | native BF16 MoE | configurable attention TP, attention DP, EP, HP, and NVLink domains | only BF16 unified graph |
| `qwen3_moe_fp8_dp_attn_ep_ffn` | native FP8 MoE | configurable attention TP, attention DP, EP, HP, and NVLink domains | covers layouts outside the aligned vLLM subset |
| `qwen3_vllm_moe_dp_attn_ep_ffn` | vLLM-aligned FP8 MoE | `attn_tp_size == ep_size` and `hp_size == 1`; no attention DP | preserves measured vLLM kernel boundaries |
| `qwen3_attn_tp` + `qwen3_ffn_moe` | layer-wise AFD BF16 | attention TP workers plus a separately replicated EP FFN pool | distinct worker and scheduling contract |
| `qwen3_attn_tp` + `qwen3_fp8_ffn_moe` | layer-wise AFD FP8 | attention TP workers plus a separately replicated EP FFN pool | distinct worker and scheduling contract |
| `deepseek_v4_vllm` | vLLM-aligned FP4 MoE | fixed H200 EP4 in one NVLink domain, with four local-attention DP groups | canonical DeepSeek V4 graph |
| `deepseek_v4_vllm_serial_streams` | the same kernels with source-level parallel regions serialized | same fixed layout as `deepseek_v4_vllm` | explicit alignment counterfactual; experimental |
| `glm52_vllm_dsa_moe` | vLLM-aligned BF16 or FP8 DSA/MoE | local TP1 attention replicated across configurable EP ranks; `nvl_num_gpu` divides `ep_size` | canonical replacement for `glm52_dsa_moe` |
| `glm52_vllm_nvfp4_dsa_moe` | vLLM-aligned NVFP4 DSA/MoE | shared TP/EP rank group, with one attention group | distinct precision and parallel topology |
| `glm52_vllm_nvfp4_dsa_moe_speculative` | vLLM-aligned NVFP4 target plus MTP draft passes | same TP/EP topology as the ordinary NVFP4 graph | distinct model and speculative-worker contract |
| `glm52_sglang_nvfp4_tp_dsa_moe` | SGLang-aligned NVFP4 DSA/MoE | pure TP with EP1; every rank owns all experts | covers a topology the vLLM graph cannot represent |
| `qwen36_local` | heterogeneous local FP8 graph | fixed TP1/EP1 | only Qwen3.6 graph |

## Retired selectors

### `glm52_dsa_moe`

`glm52_dsa_moe` and `glm52_vllm_dsa_moe` accepted the same model configs,
precision flag, EP size, NVLink-domain size, routing modes, and MTP modes. Both
reported local TP1 attention with one independent attention group per EP rank.
The vLLM graph supersedes the original graph because its leaves follow the
measured framework kernel boundaries.

Migrate only the selector name; all other fields retain their meaning:

```yaml
arch:
  type: glm52_vllm_dsa_moe
  model_config: model/config/glm52_fp8.json
  fp8: true
  ep_size: 8
  nvl_num_gpu: 8
  routing: uniform
  mtp_mode: "off"
```

Existing simulation and prediction artifacts remain readable as historical
data. Re-running their stored parameters requires changing the selector name,
and newly generated cost logs use the vLLM graph's leaf names. Consumers must
use `model/work/location_maps/glm52_vllm_dsa_moe_unified.json` for new runs.

### `llama3_attn_tp` and `deepseek_ffn_moe`

These layer-wise names were schema placeholders and never had runnable model or
deployment implementations. They have no result-preserving migration. Use the
implemented Qwen3 AFD selectors for Qwen3 workloads; a Llama 3 or DeepSeek AFD
graph must be implemented and aligned before a new selector is published.
