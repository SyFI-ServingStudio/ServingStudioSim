//! `worker_v2` — interface-expressiveness experiment for the L5 worker redesign.
//!
//! Implements the component interfaces from
//! `doc/detailed_design/L5_worker_compose_interfaces.md` as REAL traits and
//! samples worker compositions across the compatibility matrix to test whether
//! the seams are expressive enough. Implementations are rough, but every input/
//! output type and every external function call is accurate against current
//! source (verified, not copied from the older `worker_compose` sketch).
//!
//! Axes, one folder each:
//!   - `shared/`     — shared vocab (`PartitionId` ⟂ `AdvanceScope`) + `WorkerContext`
//!   - `kv/`         — `KvStore` + `IterWorkerKv` (+ future capability sub-traits)
//!   - `admission/`  — `IterAdmission<K>` = lifecycle × `PendingOrderPolicy`
//!   - `execution/` — `IterModelExecution` (owns Input + the ArchInput builder)
//!   - `workers/<family>/` — cadence shells grouped by family; one file per `build_*` recipe
//!
//! 22 servers over 15 distinct worker TYPES now compose (see `README.md`'s server table
//! + coverage matrix). Every axis impl is exercised ≥ once: Kv {FullAttnKv, ModeledPrefixCacheKv,
//! HybridStateKv, ModelPartitionedKv}, IterAdmission {LocalPrefillDecodeAdmission, ChunkedPrefillAdmission, PrefillHandoffAdmission,
//! PrefixPrefillDecodeAdmission, MultiModelAdmission, FreshRequestSlotAdmission}, Policy {FifoOrder, ShortestJobFirst}, IterModelExecution
//! {UnifiedIterExecution, AttentionLayerExecutionAdapter, FfnSectionExecutionAdapter, MultiModelIterExecution}, Shell {IterBatchWorker,
//! PullDecodeWorker, SlotAttentionWorker, PullSlotAttentionWorker, BufferedFfnWorker}. `census.rs` compile-asserts
//! each type against the real L6 `IterWorker` / `AfdAttnWorker` traits.
//!
//! NOT wired into production selectors; included under `cfg(test)` only, to
//! compile-check the seams.

#![allow(dead_code)]

pub mod admission;
mod census;
pub mod execution;
pub mod kv;
pub mod shared;
pub mod workers;
