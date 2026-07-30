//! Workers grouped by cadence family.
//!
//! Each family directory owns its worker/FSM implementation and keeps every
//! concrete `build_*` construction recipe in a separate file.

// This barrel is the construction facade; the compile census consumes the worker
// types but intentionally does not call every exported builder.
#![allow(unused_imports)]

mod afd_attention;
mod afd_ffn;
mod draft_verify;
mod iter;
mod pd_decode;

pub use afd_attention::{
    build_afd_attention_worker, build_pd_afd_decode_attention_worker, PullAttentionWorkerMsg,
    PullSlotAttentionWorker, SlotAttentionWorker,
};
pub use afd_ffn::{build_afd_ffn_worker, BufferedFfnWorker};
pub use draft_verify::{build_speculative_decode_worker, DraftVerifyWorker};
pub use iter::{
    build_barebone_worker, build_chunked_prefill_worker, build_hp_worker, build_hybrid_kv_worker,
    build_multi_model_worker, build_pd_prefill_worker, build_prefix_cache_worker,
    build_shortest_job_worker, IterBatchWorker,
};
pub use pd_decode::{build_pd_decode_worker, PullDecodeWorker};
