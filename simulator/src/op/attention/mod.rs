//! `op/attention` — compound attention ops. See doc/detailed_design/L2.md.

pub mod flashinfer;
pub mod mla;

pub use flashinfer::{FlashInferAttentionConfig, FlashInferAttentionInput, FlashInferAttentionOp};
pub use mla::{MlaAttentionConfig, MlaAttentionInput, MlaAttentionOp};
