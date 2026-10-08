//! `op/attention` — compound attention ops. See doc/detailed_design/L2.md.

pub mod deepseek_v41_indexer;
pub mod deepseek_v41_mega_attn;
pub mod dsa_indexer;
pub mod dsa_sparse_mla;
pub mod flashinfer;
pub mod glm53_kpool_sparse_mla;

pub use deepseek_v41_indexer::{
    byte_rate_placeholder_shape, DeepseekV41CandidateRole, DeepseekV41IndexerOp,
    DeepseekV41IndexerOpConfig, DeepseekV41IndexerOpInput, DeepseekV41IndexerOpResolved,
};
pub use deepseek_v41_mega_attn::{
    DeepseekV41MegaAttnOp, DeepseekV41MegaAttnOpConfig, DeepseekV41MegaAttnOpInput,
};
pub use dsa_indexer::{
    DsaIndexerConfig, DsaIndexerDecodeInput, DsaIndexerInput, DsaIndexerLaunchGraph, DsaIndexerOp,
};
pub use dsa_sparse_mla::{
    DsaSparseMlaAttentionConfig, DsaSparseMlaAttentionInput, DsaSparseMlaAttentionOp,
    DsaSparseMlaExactVarlenConfig, DsaSparseMlaLaunchGraph,
};
pub use flashinfer::{FlashInferAttentionConfig, FlashInferAttentionInput, FlashInferAttentionOp};
pub use glm53_kpool_sparse_mla::{
    Glm53KpoolSparseMlaConfig, Glm53KpoolSparseMlaInput, Glm53KpoolSparseMlaOp,
    Glm53KpoolSparseMlaResolved,
};
