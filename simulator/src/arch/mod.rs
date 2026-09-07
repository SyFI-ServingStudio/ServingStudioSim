//! `arch` (L4) — per-worker-type model_arch wire files + the L4↔L5 data
//! contract. Each model_arch picks an L3 worklet set, forwards `ModelCfg` + its
//! own numeric parallel struct (`DenseParallel` / `DenseTpParallel` /
//! `DpAttnTpFfnParallel` / …) 1:1 into worklet configs, and assembles a
//! build/cost model. See doc/detailed_design/L4.md.

pub mod build;
pub mod config;
pub mod contract;
pub mod deepseek_v4_vllm;
pub mod glm52_dsa_moe;
pub mod glm52_sglang_nvfp4_tp_dsa_moe;
pub mod glm52_vllm_dsa_moe;
pub mod glm52_vllm_nvfp4_dsa_moe;
pub mod llama3_dense;
pub mod llama3_dense_tp;
pub mod llama3_dp_attn_tp_ffn;
pub mod model_cfg;
pub mod moe_model_cfg;
pub mod qwen36_local;
pub mod qwen3_attn_layerwise;
pub mod qwen3_ffn_moe_layerwise;
pub mod qwen3_fp8_ffn_moe_layerwise;
pub mod qwen3_moe_dp_attn_ep_ffn;
pub mod qwen3_moe_fp8_dp_attn_ep_ffn;
pub mod qwen3_vllm_moe_dp_attn_ep_ffn;

pub use config::{AttnArchSel, FfnArchSel, IterArchSel, ModelSpec, RoutingKind};
pub use contract::{
    ArchGroupInput, AttnArchInput, AttnLayerwiseModel, FfnArchInput, FfnLayerwiseModel,
    IterwiseUnifiedModel, SpeculativeArchGroupInput, SpeculativeArchInput, SpeculativeDecodeInput,
    SpeculativeUnifiedModel, UnifiedArchInput,
};
pub use deepseek_v4_vllm::{
    DeepseekV4ModelCfg, DeepseekV4VllmConfigs, DeepseekV4VllmModel, DeepseekV4VllmParallel,
    DeepseekV4VllmResolved,
};
pub use glm52_dsa_moe::{
    Glm52DsaMoeConfigs, Glm52DsaMoeModel, Glm52DsaMoeParallel, Glm52DsaMoeResolved, Glm52ModelCfg,
    Glm52MtpMode,
};
pub use glm52_sglang_nvfp4_tp_dsa_moe::{
    Glm52SglangNvfp4TpDsaMoeConfigs, Glm52SglangNvfp4TpDsaMoeModel,
    Glm52SglangNvfp4TpDsaMoeParallel, Glm52SglangNvfp4TpDsaMoeResolved,
};
pub use glm52_vllm_dsa_moe::{
    Glm52VllmDsaMoeConfigs, Glm52VllmDsaMoeModel, Glm52VllmDsaMoeParallel, Glm52VllmDsaMoeResolved,
};
pub use glm52_vllm_nvfp4_dsa_moe::{
    Glm52VllmNvfp4DsaMoeConfigs, Glm52VllmNvfp4DsaMoeModel, Glm52VllmNvfp4DsaMoeParallel,
    Glm52VllmNvfp4DsaMoeResolved, Glm52VllmNvfp4DsaMoeSpeculativeModel,
};
pub use llama3_dense::{DenseParallel, Llama3DenseModel};
pub use llama3_dense_tp::{DenseTpParallel, Llama3DenseTpModel};
pub use llama3_dp_attn_tp_ffn::{DpAttnTpFfnParallel, Llama3DpAttnTpFfnModel};
pub use model_cfg::ModelCfg;
pub use moe_model_cfg::MoeModelCfg;
pub use qwen36_local::{
    Qwen36LocalConfigs, Qwen36LocalModel, Qwen36LocalParallel, Qwen36LocalResolved, Qwen36ModelCfg,
};
pub use qwen3_attn_layerwise::{Qwen3AttnLayerwiseModel, Qwen3AttnParallel};
pub use qwen3_ffn_moe_layerwise::{Qwen3FfnMoeLayerwiseModel, Qwen3FfnMoeParallel};
pub use qwen3_fp8_ffn_moe_layerwise::{Qwen3Fp8FfnMoeLayerwiseModel, Qwen3Fp8FfnMoeParallel};
pub use qwen3_moe_dp_attn_ep_ffn::{Qwen3MoeDpAttnEpFfnModel, Qwen3MoeParallel};
pub use qwen3_moe_fp8_dp_attn_ep_ffn::{Qwen3MoeFp8DpAttnEpFfnModel, Qwen3MoeFp8Parallel};
pub use qwen3_vllm_moe_dp_attn_ep_ffn::{Qwen3VllmMoeDpAttnEpFfnModel, Qwen3VllmMoeParallel};
