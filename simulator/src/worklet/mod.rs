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
pub mod fp8_attn_block_tp;
pub mod fp8_post_attn_router_tp;
pub mod fp8_pre_attn_proj_tp;
pub mod glm52_dense_ffn_local;
pub mod glm52_dsa_attn_local;
pub mod glm52_moe_router_local;
pub mod glm52_mtp_prelude_local;
pub mod glm52_shared_expert_local;
pub mod mlp_block_tp;
pub mod moe_expert_compute_local;
pub mod native_fp8_moe_router_local;
pub mod native_moe_expert_compute_local;
pub mod native_moe_router_local;
pub mod post_attn_local;
pub mod post_attn_router_tp;
pub mod pre_attn_local;
pub mod pre_attn_proj_tp;
pub mod vllm_fp8_attn_block_tp;
pub mod vllm_fp8_moe_expert_compute_local;
pub mod vllm_fp8_moe_router_local;

pub use attn_block_tp::{
    AttnBlockTpWorklet, AttnBlockTpWorkletConfig, AttnBlockTpWorkletInput,
    AttnBlockTpWorkletResolved,
};
pub use attn_local::{
    AttnLocalWorklet, AttnLocalWorkletConfig, AttnLocalWorkletInput, AttnLocalWorkletResolved,
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
pub use glm52_dsa_attn_local::{
    Glm52DsaAttnLocalDecodeInput, Glm52DsaAttnLocalWorklet, Glm52DsaAttnLocalWorkletConfig,
    Glm52DsaAttnLocalWorkletInput, Glm52DsaAttnLocalWorkletResolved,
};
pub use glm52_moe_router_local::{
    Glm52MoeRouterLocalWorklet, Glm52MoeRouterLocalWorkletConfig, Glm52MoeRouterLocalWorkletInput,
    Glm52MoeRouterLocalWorkletResolved,
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
