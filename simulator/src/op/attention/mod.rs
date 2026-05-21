//! `op/attention` — compound attention ops. See docs/detailed_design/L2/ §3.

pub mod flashinfer;

pub use flashinfer::{FlashInferAttentionConfig, FlashInferAttentionInput, FlashInferAttentionOp};
