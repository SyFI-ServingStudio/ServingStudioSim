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
| `deepseek_v41_vllm` | vLLM-aligned MXFP4 MoE, MXFP8 dense GEMMs, Engram | fixed B200 TP4 attention with EP4 experts over the same four ranks; one attention group | canonical DeepSeek-V4.1-Flash graph |
| `deepseek_v41_vllm_serial_streams` | the same kernels with the gated side streams serialized | same fixed layout as `deepseek_v41_vllm` | explicit alignment counterfactual; experimental |
| `glm52_vllm_dsa_moe` | vLLM-aligned BF16 or FP8 DSA/MoE | local TP1 attention replicated across configurable EP ranks; `nvl_num_gpu` divides `ep_size` | canonical replacement for `glm52_dsa_moe` |
| `glm52_vllm_nvfp4_dsa_moe` | vLLM-aligned NVFP4 DSA/MoE | shared TP/EP rank group, with one attention group | distinct precision and parallel topology |
| `glm52_vllm_nvfp4_dsa_moe_speculative` | vLLM-aligned NVFP4 target plus MTP draft passes | same TP/EP topology as the ordinary NVFP4 graph | distinct model and speculative-worker contract |
| `glm52_vllm_nvfp4_pp_dsa_moe` | the vLLM NVFP4 graph's kernels under pure pipeline parallelism; GLM-5.2 and GLM-5.3 NVFP4 | `pp_size` one-GPU stages at EP1 (vLLM `get_pp_indices` split); no collectives, no MTP; deployment `pp` only | distinct parallel topology and per-stage worker contract |
| `glm52_vllm_nvfp4_dp_attn_dsa_moe` | the vLLM NVFP4 graph's kernels under DP attention + EP MoE; GLM-5.2 and GLM-5.3 NVFP4 | `ep_size` GPUs (4 or 8, one NVLink node), each its own TP1 attention rank and KV partition; routed experts sharded behind an NVFP4 all-gather and a bf16 reduce-scatter; no MTP | distinct parallel topology and MoE exchange |
| `glm53_vllm_nvfp4_dsa_moe_dflash2` | the same NVFP4 target, run without its MTP layer, plus one DFlash2 block-parallel draft pass | same TP/EP topology; the draft is TP-sharded over the same ranks | distinct proposer checkpoint and cost tree |
| `glm53_flash_vllm_fp8_kda_dsa_moe` | vLLM-fork-aligned GLM-5.3-Flash FP8 block hybrid KDA/DSA (kpool)/MoE with mHC | shared TP/EP rank group (TP4/EP4 profiled), one attention group; `barebone` worker with hybrid KV | only KDA + kpool-DSA graph |
| `glm53_flash_vllm_fp8_pp_kda_dsa_moe` | the GLM-5.3-Flash FP8 graph's kernels under pure pipeline parallelism | `pp_size` one-GPU stages at TP1/EP1 (vLLM `get_pp_indices` split); per-stage KDA state and DSA cache; no collectives, no MTP; deployment `pp` only | distinct parallel topology and per-stage hybrid cache |
| `glm53_flash_vllm_fp8_dp_attn_ep_moe` | the GLM-5.3-Flash FP8 graph's kernels at TP1 under vLLM DP attention + EP MoE | `ep_size` (4 or 8) one-GPU engines, each its own attention-DP group with its own KV/KDA state; routed experts behind an FP8 all-gatherv and a bf16 reduce-scatterv; hybrid KV worker | distinct parallel topology and collective schedule |
| `glm53_flash_vllm_nvfp4_kda_dsa_moe` | the TP/EP Flash graph for NVIDIA's ModelOpt NVFP4 checkpoint: NVFP4 routed experts (TRT-LLM fused MoE, DeepSeekV3 routing) and dense FFN (FlashInfer CuTe-DSL FP4 GEMM, swizzled-scale quant); BF16 attention, router and shared expert; FP8 KV cache | as `glm53_flash_vllm_fp8_kda_dsa_moe`; the config's `quantization_config` must be the NVFP4 one | precision differs per sublayer, so the FP8 tag would mislabel it |
| `glm53_flash_vllm_nvfp4_pp_kda_dsa_moe` | the NVFP4 Flash graph under pure pipeline parallelism | as `glm53_flash_vllm_fp8_pp_kda_dsa_moe`; deployment `pp` only | NVFP4 checkpoint, PP topology |
| `glm53_flash_vllm_nvfp4_dp_attn_ep_moe` | the NVFP4 Flash graph under vLLM DP attention + EP MoE: vLLM top-k select, NVFP4 quant, quantized all-gatherv (FP4 + scales + top-k), TRT-LLM routed MoE, BF16 reduce-scatterv | as `glm53_flash_vllm_fp8_dp_attn_ep_moe` | NVFP4 dispatch replaces the FP8 one |
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
