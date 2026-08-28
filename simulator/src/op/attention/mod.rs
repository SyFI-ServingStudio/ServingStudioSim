//! `op/attention` — compound attention ops. See doc/detailed_design/L2.md.

pub mod dsa_indexer;
pub mod dsa_sparse_mla;
pub mod flashinfer;

pub use dsa_indexer::{DsaIndexerConfig, DsaIndexerDecodeInput, DsaIndexerInput, DsaIndexerOp};
pub use dsa_sparse_mla::{
    DsaSparseMlaAttentionConfig, DsaSparseMlaAttentionInput, DsaSparseMlaAttentionOp,
    DsaSparseMlaExactVarlenConfig,
};
pub use flashinfer::{FlashInferAttentionConfig, FlashInferAttentionInput, FlashInferAttentionOp};
