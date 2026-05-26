//! `worker` (L5) — per-worker FSM that turns admitted requests into iter cost
//! queries and drives request lifecycle. See docs/detailed_design/L5/design.md.
//!
//! Current set: the barebone iter-wise unified worker (§3.4) + the shared
//! admission primitives (§2).

pub mod admission_helpers;
pub mod config;
pub mod unified;

pub use admission_helpers::{Batch, DecodeReqState, KvAdmission, KvPool, LoadBalance};
pub use config::{AttnWorkerSel, BatchPolicy, FfnWorkerSel, IterWorkerSel};
pub use unified::{
    BareboneWorker, BatchFsmState, IterCursor, WorkerConfig, WorkerEvent, WorkerFsmState,
    WorkerMsg, WorkerStatus,
};
