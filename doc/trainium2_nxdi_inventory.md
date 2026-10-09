# Llama 3.1 8B: the captured NxDI execution boundary

This inventory describes the public NxDI whole-model path, distinct from
`llama3_neuron`'s separately compiled layer composition. The aligned selector is
`llama3_nxdi`; its proposed stable location is `unified.forward`.

## Evidence and shape contract

The real checkpoint is Llama 3.1 8B, with 32 layers, hidden width 4096,
intermediate width 14336, 32 query heads, eight KV heads, head width 128,
vocabulary 128256, untied embeddings/head, and scaled Llama 3.1 RoPE. Compute,
learned weights, and KV storage are BF16. Public NxDI returns FP32 logits.

The verified public implementation is NxDI revision
`4bcdc54bf3b6ccdad490e5cd0680a3dc95270b8e`. Source anchors below refer to its
`src/neuronx_distributed_inference/` tree. Configuration and checkpoint proof
are reproducible through `tools/trainium2/validate_checkpoint.py`.

The raw native capture lives in
`logs/20261007_1_trainium2_nxdi_capture/profile/`: `system-trace.json`,
`forward-records.json`, and `event-types.json`. Ten requests produced 112
forwards: ten prefill and 102 decode. Each forward has one `nrt_execute` and
two physical-core `nc_exec_running` intervals, representing one logical LNC2
invocation. Physical core IDs in this capture are 4 and 5; logical device ID
is 0. Timing unions synchronized `timestamp_ns` intervals, rather than adding
the two cores or merging their independent `nc_timestamp_ns` clocks.

The compiled prefill graph processes a fixed 128-row bucket; decode uses one
query row and a fixed 512-token cache allocation. Logical query/context lengths
remain in the workload records and independent necessary-work label. Batch is
one, TP is one, prefix reuse is disabled, and `layer_boundary_markers` and the
optional QKV/MLP/attention NKI paths are disabled. Changing those options creates
a different execution boundary requiring separate measurement.

## Decision table

Rows are ordered by their separately measurable device-time share. The whole
forward owns all captured active device time. Its component operations have no
separately observable share in this trace; assigning them individual layer
times would invent a decomposition. Each fold below contributes zero additional
timing, because its cost is already measured in the forward invocation.

| Operation | Production source anchor | Captured phase/track/position; share | Verdict and timing home |
| --- | --- | --- | --- |
| Full stateful forward | `models/model_base.py`: model forward and `get_model_output`; public `HuggingFaceGenerationAdapter.forward` | Prefill CTE or decode TKG NEFF / logical device 0 / position 0; 100% of captured active device time | New L1 `neuron_llama_forward/nxdi_compiler`; one atomic L2 op, one local L3 worklet, one whole-model L4 boundary |
| Token embedding | `models/model_base.py`: `embed_tokens(input_ids)` | Inside the same CTE/TKG invocation; individual share unavailable | Fold into forward |
| Input RMSNorm, Q/K/V projections | `models/llama/modeling_llama.py`: `NeuronLlamaDecoderLayer.forward`; `modules/attention/attention_base.py`: `standard_causal_attention_forward` | Inside CTE/TKG, repeated through all 32 layers | Fold into forward |
| Scaled RoPE and attention masks | `models/llama/modeling_llama.py`: rotary embedding; `modules/attention/attention_base.py`: `standard_causal_attention_forward` | Inside CTE/TKG, all layers | Fold into forward |
| GQA QK, softmax, and probability-times-V | `modules/attention/attention_base.py`: `attention_context_encode` and `attention_tokengen`, including `manual_softmax` and past/active value matmuls | Inside CTE/TKG, all layers; prefill and cached decode have different HLO | Fold into forward |
| O projection and first residual addition | `NeuronLlamaDecoderLayer.forward`: `self_attn` output and residual addition | Inside CTE/TKG, all layers | Fold into forward |
| Post-attention RMSNorm | `NeuronLlamaDecoderLayer.forward`: `post_attention_layernorm` | Inside CTE/TKG, all layers | Fold into forward |
| Gate/up projections, SiLU product, down projection, second residual | `NeuronLlamaDecoderLayer.forward`: public MLP and residual addition | Inside CTE/TKG, all layers | Fold into forward |
| Persistent K/V cache reads and updates | `models/model_base.py`: `kv_mgr.get_cache` and `kv_mgr.update_cache` | Inside CTE/TKG; 64 aliased cache outputs | Fold into forward; preserve stateful cache ABI |
| Final RMSNorm | `models/model_base.py`: `self.norm(hidden_states)` | Inside CTE/TKG | Fold into forward |
| Last-position gather, output head, and FP32 logits conversion | `models/model_base.py`: `get_model_output`, `lm_head`, `logits.float()` | Inside CTE/TKG; one output position | Fold into forward |

This is a new kind because the existing GEMM, normalization, embedding, MLP,
and separately compiled decoder kinds do not describe the whole-model stateful
operation, its public callable, or its weight-separated launch boundary. Adding
their times would measure a different compilation. No collectives are issued
at TP1. There is no MoE routing or token-corpus phase for this dense model.

## Framework plumbing and completeness

The capture's remaining event types are explicitly outside the active compute
boundary:

- Input/output copies and buffers: `nrt_tensor_write`, `nrt_tensor_read`,
  `dmem_buf_copyin`, `dmem_buf_copyout`, `nrt_tensor_free`, `nrt_dma_mem_alloc`,
  and `nrt_dma_mem_dealloc`.
- Submission and synchronization: `nrt_execute`, `nrt_model_submit`,
  `kbl_exec_pre`, `kmgr_exec_core`, `kbl_exec_wait`, `kbl_exec_post`,
  `timestamp_sync_point`, and `nrt_profile_add_node_info`.
- Runtime scheduling and diagnostics: `nc_model_switch`,
  `exec_consume_gpsimd_stdio`, and `notification_consume_errors`.

These host/runtime events overlap the compute invocation and are not additional
math kernels. Host token selection and HTTP streaming belong to the real
serving workload. Their gaps must be retained in duty-cycle and end-to-end
alignment, rather than folded into measured device kernel time. Every observed
event type is accounted for above, and every cost-bearing model operation has
a home in the decision table. The native normalizer validates one complete
two-core invocation per forward and preserves logical q/K metadata.

The original capture remains the kernel-inventory evidence. A matched serving
campaign uses identical req-frontend prompt tokens for its native kernel pass
and independent unprofiled workload-metrics pass. That later measurement proves
the new server/workload adapter; it does not replace a valid capture because a
simulator label or report needs repair.

## Hardware and metric assumptions

The catalog's dense BF16 rate is 158 TFLOPS per LNC2, derived from two published
79-TFLOPS NC-v3 Tensor Engines. Its 725-GB/s HBM rate is an equal-bank analysis
estimate (2.9-TB/s chip bandwidth divided by four banks), not an independently
published or measured per-unit guarantee. Analyzer hardware floors and R6/R7
must be interpreted under that assumption. Neither value supplies kernel time.

Executed FLOPs come from concrete HLO dot dimensions. Executed bytes estimate
persistent learned-weight reads, selected embedding rows, phase-specific cache
reads/writes, and the FP32 logits output. They exclude transient on-chip
operands, activation spills, tiling reloads, and embedding-row reuse. The byte
metric is an estimate, not an HBM counter. Independent `model.work` instead
uses logical causal q/K and compulsory semantics; it never reads this HLO or
the simulator timing tree.
