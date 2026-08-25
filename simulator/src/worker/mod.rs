//! `worker` (L5) — per-worker FSM that turns admitted requests into iter cost
//! queries and drives request lifecycle. See doc/detailed_design/L5.md.
//!
//! Production workers are statically composed under `workers/<cadence-family>/`:
//! whole-iteration barebone/HP/PD-prefill, pull+decode PD-decode, slot-pipelined
//! AFD attention, and double-buffered AFD FFN. L6 sees their existing typed
//! message/event traits; it does not see the components.

pub(crate) mod admission;
pub mod config;
pub mod cost_buffers;
pub(crate) mod execution;
pub mod gpu_cluster;
pub mod iter_worker;
pub(crate) mod kv;
pub(crate) mod shared;
pub mod types;
pub(crate) mod workers;

pub use admission::{
    AdmissionCandidate, FifoOrder, LoadBalance, LongestPrefixMatch, PendingOrder, PendingOrderKind,
    PendingOrderPolicy, SessionStartOrder, ShortestJobFirst,
};
pub(crate) use config::resolve_prefix_cache_config;
pub use config::{AttnWorkerSel, BatchPolicy, FfnWorkerSel, IterWorkerSel};
pub use cost_buffers::CostBuffers;
pub use gpu_cluster::{CostSource, GpuCluster, GpuInfo, SharedGpuCluster};
pub use iter_worker::{AfdAttnWorker, AfdFfnWorker, IterWorker};
pub use kv::{PrefixCacheConfig, PrefixCacheMode, PrefixCachePolicy};
pub use types::{
    AttnWorkerEvent, AttnWorkerMsg, BatchFsmState, FfnPullSource, FfnTask, FfnTaskKind,
    FfnWorkerEvent, FfnWorkerMsg, IterCursor, IterEndState, PdDecodeEvent, PdDecodeMsg,
    PdPrefillEvent, PdPrefillMsg, TransferPlan, WorkerConfig, WorkerEventCommon, WorkerFsmState,
    WorkerMsgCommon, WorkerStatus,
};
pub(crate) use workers::afd_attention::build_afd_attention_worker;
pub use workers::afd_attention::DisaggAttnWorker;
pub(crate) use workers::afd_ffn::build_afd_ffn_worker;
pub use workers::afd_ffn::DisaggFfnWorker;
pub(crate) use workers::iter::{
    build_barebone_worker, build_chunked_prefill_worker, build_hp_worker, build_pd_prefill_worker,
    build_qwen36_hybrid_worker,
};
pub use workers::iter::{
    BareboneWorker, ChunkedPrefillWorker, HpUnifiedWorker, PdPrefillWorker, Qwen36HybridWorker,
};
pub(crate) use workers::pd_decode::build_pd_decode_worker;
pub use workers::pd_decode::PdDecodeWorker;
