//! `worklet` (L3) — model-module-level op compositions with a CostTree
//! `compile` / `eval` pair. Each worklet is a hand-written struct of L2 op slots
//! exposing (`*Config` / `*Resolved` / `*Input` / `Self` / `resolve_config` /
//! `build` / `compile` / `eval`).
//! See doc/detailed_design/L3.md.
//!
//! Current set: the three `Local` worklets of a dense decoder layer
//! (pre-attention / attention / post-attention), used by `arch::llama3_dense`;
//! plus the two `TP` worklets (attn-block / mlp-block), used by
//! `arch::llama3_dense_tp` and `arch::llama3_dp_attn_tp_ffn`.

pub mod attn_block_tp;
pub mod attn_local;
pub mod bf16_moe_local;
pub mod deepseek_v4_attention_local;
pub mod deepseek_v4_moe_expert_compute_local;
pub mod deepseek_v4_moe_router_local;
pub mod deepseek_v4_shared_expert_local;
pub mod dflash2_context_kv_local;
pub mod dflash2_draft_layer_local;
pub mod dflash2_selector_local;
pub mod fp8_attn_block_tp;
pub mod fp8_post_attn_router_tp;
pub mod fp8_pre_attn_proj_tp;
pub mod glm52_dense_ffn_local;
mod glm52_dsa_attn_common;
pub mod glm52_moe_router_local;
pub mod glm52_mtp_head_local;
pub mod glm52_mtp_prelude_local;
pub mod glm52_shared_expert_local;
pub mod mlp_block_tp;
pub mod moe_expert_compute_local;
pub mod native_fp8_moe_router_local;
pub mod native_moe_expert_compute_local;
pub mod native_moe_router_local;
pub mod nvfp4_moe_local;
pub mod post_attn_local;
pub mod post_attn_router_tp;
pub mod pre_attn_local;
pub mod pre_attn_proj_tp;
pub mod qwen36_gated_gqa_local;
pub mod qwen36_gdn_local;
pub mod qwen36_head_local;
pub mod qwen36_moe_finalize_local;
pub mod qwen36_moe_router_local;
pub mod qwen36_shared_expert_local;
pub mod sglang_glm52_dsa_attn_local;
pub mod sglang_glm52_moe_router_local;
pub mod sglang_moe_finalize_local;
pub mod vllm_fp8_attn_block_tp;
pub mod vllm_fp8_moe_expert_compute_local;
pub mod vllm_fp8_moe_router_local;
pub mod vllm_glm52_dense_ffn_local;
pub mod vllm_glm52_dsa_attn_local;
pub mod vllm_glm52_shared_expert_local;

pub use attn_block_tp::{
    AttnBlockTpWorklet, AttnBlockTpWorkletConfig, AttnBlockTpWorkletInput,
    AttnBlockTpWorkletResolved,
};
pub use attn_local::{
    AttnLocalWorklet, AttnLocalWorkletConfig, AttnLocalWorkletInput, AttnLocalWorkletResolved,
};
pub use bf16_moe_local::{
    Bf16MoeLocalWorklet, Bf16MoeLocalWorkletConfig, Bf16MoeLocalWorkletInput,
    Bf16MoeLocalWorkletResolved,
};
pub use deepseek_v4_attention_local::{
    DeepseekV4AttentionEntry, DeepseekV4AttentionLocalWorklet,
    DeepseekV4AttentionLocalWorkletConfig, DeepseekV4AttentionLocalWorkletInput,
    DeepseekV4AttentionLocalWorkletResolved,
};
pub use deepseek_v4_moe_expert_compute_local::{
    DeepseekV4MoeExpertComputeLocalWorklet, DeepseekV4MoeExpertComputeLocalWorkletConfig,
    DeepseekV4MoeExpertComputeLocalWorkletInput, DeepseekV4MoeExpertComputeLocalWorkletResolved,
};
pub use deepseek_v4_moe_router_local::{
    DeepseekV4MoeRouterLocalWorklet, DeepseekV4MoeRouterLocalWorkletConfig,
    DeepseekV4MoeRouterLocalWorkletInput, DeepseekV4MoeRouterLocalWorkletResolved,
};
pub use deepseek_v4_shared_expert_local::{
    DeepseekV4SharedExpertLocalWorklet, DeepseekV4SharedExpertLocalWorkletConfig,
    DeepseekV4SharedExpertLocalWorkletInput, DeepseekV4SharedExpertLocalWorkletResolved,
};
pub use dflash2_context_kv_local::{
    Dflash2ContextKvLocalWorklet, Dflash2ContextKvLocalWorkletConfig,
    Dflash2ContextKvLocalWorkletInput, Dflash2ContextKvLocalWorkletResolved,
};
pub use dflash2_draft_layer_local::{
    Dflash2DraftAttnLocalWorklet, Dflash2DraftAttnLocalWorkletInput,
    Dflash2DraftAttnLocalWorkletResolved, Dflash2DraftFfnLocalWorklet,
    Dflash2DraftFfnLocalWorkletInput, Dflash2DraftFfnLocalWorkletResolved,
    Dflash2DraftLayerLocalWorkletConfig,
};
pub use dflash2_selector_local::{
    Dflash2SelectorLocalWorklet, Dflash2SelectorLocalWorkletConfig,
    Dflash2SelectorLocalWorkletInput, Dflash2SelectorLocalWorkletResolved,
};
pub use fp8_attn_block_tp::{
    Fp8AttnBlockTpWorklet, Fp8AttnBlockTpWorkletConfig, Fp8AttnBlockTpWorkletInput,
    Fp8AttnBlockTpWorkletResolved,
};
pub use fp8_post_attn_router_tp::{
    Fp8PostAttnRouterTpWorklet, Fp8PostAttnRouterTpWorkletConfig, Fp8PostAttnRouterTpWorkletInput,
    Fp8PostAttnRouterTpWorkletResolved,
};
pub use fp8_pre_attn_proj_tp::{
    Fp8PreAttnProjTpWorklet, Fp8PreAttnProjTpWorkletConfig, Fp8PreAttnProjTpWorkletInput,
    Fp8PreAttnProjTpWorkletResolved,
};
pub use glm52_dense_ffn_local::{
    Glm52DenseFfnLocalWorklet, Glm52DenseFfnLocalWorkletConfig, Glm52DenseFfnLocalWorkletInput,
    Glm52DenseFfnLocalWorkletResolved,
};
pub use glm52_moe_router_local::{
    Glm52MoeRouterLocalWorklet, Glm52MoeRouterLocalWorkletConfig, Glm52MoeRouterLocalWorkletInput,
    Glm52MoeRouterLocalWorkletResolved,
};
pub use glm52_mtp_head_local::{
    Glm52MtpHeadLocalWorklet, Glm52MtpHeadLocalWorkletConfig, Glm52MtpHeadLocalWorkletInput,
    Glm52MtpHeadLocalWorkletResolved,
};
pub use glm52_mtp_prelude_local::{
    Glm52MtpPreludeLocalWorklet, Glm52MtpPreludeLocalWorkletConfig,
    Glm52MtpPreludeLocalWorkletInput, Glm52MtpPreludeLocalWorkletResolved,
};
pub use glm52_shared_expert_local::{
    Glm52SharedExpertLocalWorklet, Glm52SharedExpertLocalWorkletConfig,
    Glm52SharedExpertLocalWorkletInput, Glm52SharedExpertLocalWorkletResolved,
};
pub use mlp_block_tp::{
    MlpBlockTpWorklet, MlpBlockTpWorkletConfig, MlpBlockTpWorkletInput, MlpBlockTpWorkletResolved,
};
pub use moe_expert_compute_local::{
    uniform_local_ppm, MoeExpertComputeLocalWorklet, MoeExpertComputeLocalWorkletConfig,
    MoeExpertComputeLocalWorkletInput, MoeExpertComputeLocalWorkletResolved,
};
pub use native_fp8_moe_router_local::{
    NativeFp8MoeRouterLocalWorklet, NativeFp8MoeRouterLocalWorkletConfig,
    NativeFp8MoeRouterLocalWorkletInput, NativeFp8MoeRouterLocalWorkletResolved,
};
pub use native_moe_expert_compute_local::{
    NativeMoeExpertComputeLocalWorklet, NativeMoeExpertComputeLocalWorkletConfig,
    NativeMoeExpertComputeLocalWorkletInput, NativeMoeExpertComputeLocalWorkletResolved,
};
pub use native_moe_router_local::{
    NativeMoeRouterLocalWorklet, NativeMoeRouterLocalWorkletConfig,
    NativeMoeRouterLocalWorkletInput, NativeMoeRouterLocalWorkletResolved,
};
pub use nvfp4_moe_local::{
    Nvfp4MoeLocalWorklet, Nvfp4MoeLocalWorkletConfig, Nvfp4MoeLocalWorkletInput,
    Nvfp4MoeLocalWorkletResolved,
};
pub use post_attn_local::{
    PostAttnLocalWorklet, PostAttnLocalWorkletConfig, PostAttnLocalWorkletInput,
    PostAttnLocalWorkletResolved,
};
pub use post_attn_router_tp::{
    PostAttnRouterTpWorklet, PostAttnRouterTpWorkletConfig, PostAttnRouterTpWorkletInput,
    PostAttnRouterTpWorkletResolved,
};
pub use pre_attn_local::{
    PreAttnLocalWorklet, PreAttnLocalWorkletConfig, PreAttnLocalWorkletInput,
    PreAttnLocalWorkletResolved,
};
pub use pre_attn_proj_tp::{
    PreAttnProjTpWorklet, PreAttnProjTpWorkletConfig, PreAttnProjTpWorkletInput,
    PreAttnProjTpWorkletResolved,
};
pub use qwen36_gated_gqa_local::{
    Qwen36GatedGqaLocalWorklet, Qwen36GatedGqaLocalWorkletConfig, Qwen36GatedGqaLocalWorkletInput,
    Qwen36GatedGqaLocalWorkletResolved,
};
pub use qwen36_gdn_local::{
    Qwen36GdnLocalWorklet, Qwen36GdnLocalWorkletConfig, Qwen36GdnLocalWorkletInput,
    Qwen36GdnLocalWorkletResolved,
};
pub use qwen36_head_local::{
    Qwen36HeadLocalWorklet, Qwen36HeadLocalWorkletConfig, Qwen36HeadLocalWorkletInput,
    Qwen36HeadLocalWorkletResolved,
};
pub use qwen36_moe_finalize_local::{
    Qwen36MoeFinalizeLocalWorklet, Qwen36MoeFinalizeLocalWorkletConfig,
    Qwen36MoeFinalizeLocalWorkletInput, Qwen36MoeFinalizeLocalWorkletResolved,
};
pub use qwen36_moe_router_local::{
    Qwen36MoeRouterLocalWorklet, Qwen36MoeRouterLocalWorkletConfig,
    Qwen36MoeRouterLocalWorkletInput, Qwen36MoeRouterLocalWorkletResolved,
};
pub use qwen36_shared_expert_local::{
    Qwen36SharedExpertLocalWorklet, Qwen36SharedExpertLocalWorkletConfig,
    Qwen36SharedExpertLocalWorkletInput, Qwen36SharedExpertLocalWorkletResolved,
};
pub use sglang_glm52_dsa_attn_local::{
    SglangGlm52DsaAttnLocalDecodeInput, SglangGlm52DsaAttnLocalWorklet,
    SglangGlm52DsaAttnLocalWorkletConfig, SglangGlm52DsaAttnLocalWorkletInput,
    SglangGlm52DsaAttnLocalWorkletResolved,
};
pub use sglang_glm52_moe_router_local::{
    SglangGlm52MoeRouterLocalWorklet, SglangGlm52MoeRouterLocalWorkletConfig,
    SglangGlm52MoeRouterLocalWorkletInput, SglangGlm52MoeRouterLocalWorkletResolved,
};
pub use sglang_moe_finalize_local::{
    SglangMoeFinalizeLocalWorklet, SglangMoeFinalizeLocalWorkletConfig,
    SglangMoeFinalizeLocalWorkletInput, SglangMoeFinalizeLocalWorkletResolved,
};
pub use vllm_fp8_attn_block_tp::{
    VllmFp8AttnBlockTpWorklet, VllmFp8AttnBlockTpWorkletConfig, VllmFp8AttnBlockTpWorkletInput,
    VllmFp8AttnBlockTpWorkletResolved,
};
pub use vllm_fp8_moe_expert_compute_local::{
    VllmFp8MoeExpertComputeLocalWorklet, VllmFp8MoeExpertComputeLocalWorkletConfig,
    VllmFp8MoeExpertComputeLocalWorkletInput, VllmFp8MoeExpertComputeLocalWorkletResolved,
};
pub use vllm_fp8_moe_router_local::{
    VllmFp8MoeRouterLocalWorklet, VllmFp8MoeRouterLocalWorkletConfig,
    VllmFp8MoeRouterLocalWorkletInput, VllmFp8MoeRouterLocalWorkletResolved,
};
pub use vllm_glm52_dense_ffn_local::{
    VllmGlm52DenseFfnLocalWorklet, VllmGlm52DenseFfnLocalWorkletConfig,
    VllmGlm52DenseFfnLocalWorkletInput, VllmGlm52DenseFfnLocalWorkletResolved,
};
pub use vllm_glm52_dsa_attn_local::{
    VllmGlm52DsaAttnLocalDecodeInput, VllmGlm52DsaAttnLocalWorklet,
    VllmGlm52DsaAttnLocalWorkletConfig, VllmGlm52DsaAttnLocalWorkletInput,
    VllmGlm52DsaAttnLocalWorkletResolved,
};
pub use vllm_glm52_shared_expert_local::{
    VllmGlm52SharedExpertLocalWorklet, VllmGlm52SharedExpertLocalWorkletConfig,
    VllmGlm52SharedExpertLocalWorkletInput, VllmGlm52SharedExpertLocalWorkletResolved,
};
