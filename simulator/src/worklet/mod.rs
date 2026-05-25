//! `worklet` (L3) — model-module-level op compositions with a CostTree
//! `compile` / `eval` pair. Each worklet is a hand-written struct of L2 op slots
//! exposing (`*Config` / `*Resolved` / `*Input` / `Self` / `resolve_config` /
//! `build` / `compile` / `eval`).
//! See docs/detailed_design/L3/design.md.
//!
//! Current set: the three `Local` worklets of a dense decoder layer
//! (pre-attention / attention / post-attention), used by the Llama3-8B dense
//! `arch::llama3_dense` model; plus the two `TP` worklets (attn-block /
//! mlp-block) used by the tensor-parallel `arch::llama3_dense_tp` model.

pub mod attn_block_tp;
pub mod attn_local;
pub mod mlp_block_tp;
pub mod post_attn_local;
pub mod pre_attn_local;

pub use attn_block_tp::{
    AttnBlockTpWorklet, AttnBlockTpWorkletConfig, AttnBlockTpWorkletInput,
    AttnBlockTpWorkletResolved,
};
pub use attn_local::{
    AttnLocalWorklet, AttnLocalWorkletConfig, AttnLocalWorkletInput, AttnLocalWorkletResolved,
};
pub use mlp_block_tp::{
    MlpBlockTpWorklet, MlpBlockTpWorkletConfig, MlpBlockTpWorkletInput, MlpBlockTpWorkletResolved,
};
pub use post_attn_local::{
    PostAttnLocalWorklet, PostAttnLocalWorkletConfig, PostAttnLocalWorkletInput,
    PostAttnLocalWorkletResolved,
};
pub use pre_attn_local::{
    PreAttnLocalWorklet, PreAttnLocalWorkletConfig, PreAttnLocalWorkletInput,
    PreAttnLocalWorkletResolved,
};
