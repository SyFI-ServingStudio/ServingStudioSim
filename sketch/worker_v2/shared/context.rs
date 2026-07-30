//! `WorkerContext` — component-agnostic shared state (interfaces doc §5.1 / §0).
//!
//! Only what multiple components need but none owns: identity, the shared request
//! slab, and the two lifecycle-logging flags. NOT the whole `WorkerConfig`:
//! construction-only fields are consumed in the shell's builder; policy fields
//! live on the Admission axis. External calls verified against real source
//! (`common/request.rs`, `common/id.rs`).

use crate::common::{PoolId, RequestRecord, SharedRequests, Time, WorkerId};

pub struct WorkerContext {
    pub id: WorkerId,
    /// Paired with `id` to globally identify the worker in `record_stage`.
    pub pool: PoolId,
    pub requests: SharedRequests,
    /// Mirror of `WorkerConfig.log_output_token_times`.
    pub log_output_token_times: bool,
    /// Mirror of `WorkerConfig.log_stage_transitions`.
    pub log_stage_transitions: bool,
}

impl WorkerContext {
    /// Stamp a lifecycle stage transition. `code` is `SomeStage as u16`
    /// (real `RequestRecord::record_stage(now, code, pool, worker, log)`).
    #[inline]
    pub fn stamp_stage(&self, record: &mut RequestRecord, now: Time, code: u16) {
        record.record_stage(now, code, self.pool, self.id, self.log_stage_transitions);
    }

    #[inline]
    pub fn log_tokens(&self) -> bool {
        self.log_output_token_times
    }
}
