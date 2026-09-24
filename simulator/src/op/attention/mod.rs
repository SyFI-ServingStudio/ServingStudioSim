//! `op/attention` — compound attention ops. See doc/detailed_design/L2.md.

pub mod dsa_indexer;
pub mod dsa_sparse_mla;
pub mod flashinfer;
pub mod glm53_kpool_sparse_mla;

pub use dsa_indexer::{
    DsaIndexerConfig, DsaIndexerDecodeInput, DsaIndexerInput, DsaIndexerLaunchGraph, DsaIndexerOp,
};
pub use dsa_sparse_mla::{
    DsaSparseMlaAttentionConfig, DsaSparseMlaAttentionInput, DsaSparseMlaAttentionOp,
    DsaSparseMlaExactVarlenConfig, DsaSparseMlaLaunchGraph,
};
pub use glm53_kpool_sparse_mla::{
    Glm53KpoolSparseMlaConfig, Glm53KpoolSparseMlaInput, Glm53KpoolSparseMlaOp,
    Glm53KpoolSparseMlaResolved,
};
pub use flashinfer::{FlashInferAttentionConfig, FlashInferAttentionInput, FlashInferAttentionOp};
