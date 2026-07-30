//! Whole-iteration cadence family.
//!
//! `IterBatchWorker` owns the cadence. Each concrete server recipe lives in one
//! `build_*` module so construction choices do not obscure the worker FSM.

mod build_barebone_worker;
mod build_chunked_prefill_worker;
mod build_hp_worker;
mod build_hybrid_kv_worker;
mod build_multi_model_worker;
mod build_pd_prefill_worker;
mod build_prefix_cache_worker;
mod build_shortest_job_worker;
mod iter_batch_worker;
mod unified_iter_build_essentials;

pub use build_barebone_worker::build_barebone_worker;
pub use build_chunked_prefill_worker::build_chunked_prefill_worker;
pub use build_hp_worker::build_hp_worker;
pub use build_hybrid_kv_worker::build_hybrid_kv_worker;
pub use build_multi_model_worker::build_multi_model_worker;
pub use build_pd_prefill_worker::build_pd_prefill_worker;
pub use build_prefix_cache_worker::build_prefix_cache_worker;
pub use build_shortest_job_worker::build_shortest_job_worker;
pub use iter_batch_worker::IterBatchWorker;
