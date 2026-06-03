//! `arch` (L4) — per-worker-type model_arch wire files + the L4↔L5 data
//! contract. Each model_arch picks an L3 worklet set, forwards `ModelCfg` + its
//! own numeric parallel struct (`DenseParallel` / `DenseTpParallel` /
//! `DpAttnTpFfnParallel` / …) 1:1 into worklet configs, and assembles a
//! build/cost model. See docs/detailed_design/L4/design.md.

pub mod build;
pub mod contract;
pub mod llama3_dense;
pub mod llama3_dense_tp;
pub mod llama3_dp_attn_tp_ffn;
pub mod model_cfg;
pub mod moe_model_cfg;
pub mod qwen3_attn_layerwise;
pub mod qwen3_ffn_moe_layerwise;
pub mod qwen3_moe_dp_attn_ep_ffn;
pub mod config;

pub use contract::{
    ArchGroupInput, AttnArchInput, AttnLayerwiseModel, FfnArchInput, FfnLayerwiseModel,
    IterwiseUnifiedModel, UnifiedArchInput,
};
pub use llama3_dense::{DenseParallel, Llama3DenseModel};
pub use llama3_dense_tp::{DenseTpParallel, Llama3DenseTpModel};
pub use llama3_dp_attn_tp_ffn::{DpAttnTpFfnParallel, Llama3DpAttnTpFfnModel};
pub use model_cfg::ModelCfg;
pub use moe_model_cfg::MoeModelCfg;
pub use qwen3_attn_layerwise::{Qwen3AttnLayerwiseModel, Qwen3AttnParallel};
pub use qwen3_ffn_moe_layerwise::{Qwen3FfnMoeLayerwiseModel, Qwen3FfnMoeParallel};
pub use qwen3_moe_dp_attn_ep_ffn::{Qwen3MoeDpAttnEpFfnModel, Qwen3MoeParallel};
pub use config::{AttnArchSel, FfnArchSel, IterArchSel, ModelSpec, RoutingKind};
