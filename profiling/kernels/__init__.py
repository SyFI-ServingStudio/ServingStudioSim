"""Per-kernel modules. Each module's import triggers a side-effect
``register(KernelProfilerSpec(...))`` call that wires the kind into the shared
registry.

Add a kernel by creating ``profiling/kernels/<kind>.py`` with a ``KIND``
constant, the ``<Kind>Args`` dataclass, and a ``register(...)`` call, then add
``from . import <kind>`` to this barrel.

Symmetric with Rust ``simulator/src/timing/kernels/<kind>.rs``: one file per
kernel kind owns its Python wire format and registry presence.
"""

from profiling.kernels import (
    all_reduce,  # noqa: F401
    all_reduce_fusion,  # noqa: F401
    all_reduce_residual_rms_norm,  # noqa: F401
    batched_gemm,  # noqa: F401
    bf16_fused_moe,  # noqa: F401
    clamped_swiglu,  # noqa: F401
    deepseek_v4_fused_inv_rope_fp8_quant,  # noqa: F401
    deepseek_v4_fused_q_kv_rmsnorm,  # noqa: F401
    deepseek_v4_indexer_mqa_logits_decode,  # noqa: F401
    deepseek_v4_indexer_mqa_logits_prefill,  # noqa: F401
    deepseek_v4_indexer_q_rope_quant,  # noqa: F401
    deepseek_v4_indexer_topk_decode,  # noqa: F401
    deepseek_v4_indexer_topk_prefill,  # noqa: F401
    deepseek_v4_packed_cache_gather,  # noqa: F401
    deepseek_v4_qnorm_rope_kv_insert,  # noqa: F401
    deepseek_v4_sparse_attn_compress_store,  # noqa: F401
    deepseek_v4_sparse_mla_decode,  # noqa: F401
    deepseek_v4_sparse_mla_prefill,  # noqa: F401
    deepseek_v4_terminal_mhc_head,  # noqa: F401
    dsa_index_cache_append,  # noqa: F401
    dsa_indexer_q_rope_quant,  # noqa: F401
    dsa_mqa_logits_prefill,  # noqa: F401
    dsa_paged_mqa_logits_decode,  # noqa: F401
    dsa_persistent_topk_decode,  # noqa: F401
    dsa_sparse_index_remap,  # noqa: F401
    dsa_sparse_mla_attention,  # noqa: F401
    dsa_sparse_mla_prefill,  # noqa: F401
    dsa_topk_prefill,  # noqa: F401
    elementwise,  # noqa: F401
    flashinfer_attn_decode,  # noqa: F401
    flashinfer_attn_prefill,  # noqa: F401
    flashinfer_attn_rect,  # noqa: F401
    fp8_block_quant,  # noqa: F401
    fp8_blockscale_grouped_gemm,  # noqa: F401
    fp8_per_token_group_quant,  # noqa: F401
    gdn_causal_conv_decode,  # noqa: F401
    gdn_causal_conv_prefill,  # noqa: F401
    gdn_chunk_delta_rule,  # noqa: F401
    gdn_chunk_local_cumsum,  # noqa: F401
    gdn_chunk_output,  # noqa: F401
    gdn_chunk_recompute_w_u,  # noqa: F401
    gdn_chunk_scaled_dot_kkt,  # noqa: F401
    gdn_chunk_solve_tril,  # noqa: F401
    gdn_chunk_state_update,  # noqa: F401
    gdn_gated_rms_norm,  # noqa: F401
    gdn_prefill_post_conv,  # noqa: F401
    gdn_recurrent_decode,  # noqa: F401
    gemm_fp32_output,  # noqa: F401
    grouped_gemm,  # noqa: F401
    kv_cache_append,  # noqa: F401
    logits_topk,  # noqa: F401
    mhc_fused_post_pre_rms_norm,  # noqa: F401
    mhc_pre_rms_norm,  # noqa: F401
    mla_cache_append,  # noqa: F401
    mla_rope_quantize_fp8,  # noqa: F401
    moe_align_block_size,  # noqa: F401
    moe_alltoall,  # noqa: F401
    moe_alltoall_prepare,  # noqa: F401
    moe_ep_all_gather,  # noqa: F401
    moe_ep_reduce_scatter,  # noqa: F401
    moe_finalize_fuse_shared,  # noqa: F401
    moe_finalize_routing,  # noqa: F401
    moe_fused_topk,  # noqa: F401
    moe_sum,  # noqa: F401
    moe_topk_softplus_sqrt,  # noqa: F401
    mxfp4_marlin_moe_gemm,  # noqa: F401
    nvfp4_fused_moe,  # noqa: F401
    nvfp4_quant,  # noqa: F401
    p2p_inter,  # noqa: F401
    p2p_intra,  # noqa: F401
    residual_rms_norm,  # noqa: F401
    rms_norm,  # noqa: F401
    single_gemm,  # noqa: F401
    vllm_fused_moe,  # noqa: F401
    vllm_mla_rope,  # noqa: F401
)
