//! `worker` (L5) — per-worker FSM that turns admitted requests into iter cost
//! queries and drives request lifecycle. See doc/detailed_design/L5.md.
//!
//! Current set: barebone and HP/DP unified workers, the PD prefill/decode pair,
//! the `IterWorker` trait they share, selector/config types, and the shared
//! admission primitives (§2).

pub mod admission_helpers;
pub mod config;
pub mod cost_buffers;
pub mod disagg_attn;
pub mod disagg_ffn;
pub mod gpu_cluster;
pub mod hp_unified;
pub mod iter_worker;
pub mod pd_decode;
pub mod pd_prefill;
pub mod types;
pub mod unified;

// Compile the worker-composition experiment against the real crate in test builds.
// Keeping this private and test-only lets the stress census exercise crate-private
// L5 contracts without changing production selectors or module exports.
#[cfg(test)]
#[path = "../../../sketch/worker_compose/mod.rs"]
mod worker_compose_stress;

#[cfg(test)]
#[path = "../../../sketch/worker_v2/mod.rs"]
mod worker_v2_stress;

pub use admission_helpers::{Batch, DecodeReqState, KvAdmission, KvPool, LoadBalance};
pub use config::{AttnWorkerSel, BatchPolicy, FfnWorkerSel, IterWorkerSel};
pub use cost_buffers::CostBuffers;
pub use disagg_attn::DisaggAttnWorker;
pub use disagg_ffn::DisaggFfnWorker;
pub use gpu_cluster::{CostSource, GpuCluster, GpuInfo, SharedGpuCluster};
pub use hp_unified::HpUnifiedWorker;
pub use iter_worker::{AfdAttnWorker, AfdFfnWorker, IterWorker};
pub use pd_decode::PdDecodeWorker;
pub use pd_prefill::PdPrefillWorker;
pub use types::{
    AttnWorkerEvent, AttnWorkerMsg, BatchFsmState, FfnPullSource, FfnTask, FfnTaskKind,
    FfnWorkerEvent, FfnWorkerMsg, IterCursor, IterEndState, PdDecodeEvent, PdDecodeMsg,
    PdPrefillEvent, PdPrefillMsg, TransferPlan, WorkerConfig, WorkerEventCommon, WorkerFsmState,
    WorkerMsgCommon, WorkerStatus,
};
pub use unified::BareboneWorker;
