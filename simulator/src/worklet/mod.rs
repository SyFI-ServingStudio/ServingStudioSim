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
pub mod kda_block_local;
pub mod mla_block_local;
pub mod mlp_block_tp;
pub mod moe_expert_compute_local;
pub mod moe_router_local;
pub mod post_attn_local;
pub mod post_attn_router_tp;
pub mod pre_attn_local;
pub mod pre_attn_proj_tp;

pub use attn_block_tp::{
    AttnBlockTpWorklet, AttnBlockTpWorkletConfig, AttnBlockTpWorkletInput,
    AttnBlockTpWorkletResolved,
};
pub use attn_local::{
    AttnLocalWorklet, AttnLocalWorkletConfig, AttnLocalWorkletInput, AttnLocalWorkletResolved,
};
pub use kda_block_local::{
    KdaBlockLocalWorklet, KdaBlockLocalWorkletConfig, KdaBlockLocalWorkletInput,
    KdaBlockLocalWorkletResolved,
};
pub use mla_block_local::{
    MlaBlockLocalWorklet, MlaBlockLocalWorkletConfig, MlaBlockLocalWorkletInput,
    MlaBlockLocalWorkletResolved,
};
pub use mlp_block_tp::{
    MlpBlockTpWorklet, MlpBlockTpWorkletConfig, MlpBlockTpWorkletInput, MlpBlockTpWorkletResolved,
};
pub use moe_expert_compute_local::{
    uniform_local_ppm, MoeExpertComputeLocalWorklet, MoeExpertComputeLocalWorkletConfig,
    MoeExpertComputeLocalWorkletInput, MoeExpertComputeLocalWorkletResolved,
};
pub use moe_router_local::{
    MoeRouterLocalWorklet, MoeRouterLocalWorkletConfig, MoeRouterLocalWorkletInput,
    MoeRouterLocalWorkletResolved,
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
