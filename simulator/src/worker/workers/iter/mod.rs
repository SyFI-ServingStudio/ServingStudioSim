//! Whole-iteration worker family.
//!
//! The shell owns cadence; each concrete component recipe lives in a separate
//! `build_*_worker.rs` file.

mod build_barebone_worker;
mod build_chunked_prefill_worker;
mod build_hp_worker;
mod build_hybrid_chunked_prefill_worker;
mod build_pd_prefill_worker;
mod build_qwen36_hybrid_worker;
mod build_speculative_worker;
mod iter_batch_worker;

pub(crate) use build_barebone_worker::build_barebone_worker;
pub(crate) use build_chunked_prefill_worker::build_chunked_prefill_worker;
pub(crate) use build_hp_worker::build_hp_worker;
pub(crate) use build_hybrid_chunked_prefill_worker::build_hybrid_chunked_prefill_worker;
pub(crate) use build_pd_prefill_worker::build_pd_prefill_worker;
pub(crate) use build_qwen36_hybrid_worker::build_qwen36_hybrid_worker;
pub(crate) use build_speculative_worker::build_speculative_worker;
pub use iter_batch_worker::{
    BareboneWorker, ChunkedPrefillWorker, HpUnifiedWorker, HybridChunkedPrefillWorker,
    PdPrefillWorker, Qwen36HybridWorker, SpeculativeWorker,
};
