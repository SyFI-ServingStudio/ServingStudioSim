//! `WorkerCtx` — component-agnostic shared state.
//!
//! The facts multiple components need but none owns: identity, the shared request slab,
//! and the two lifecycle-logging flags. Deliberately NOT the whole
//! `WorkerConfig`: construction-only fields (`attn_kv_bytes`, `kv_log_stride`,
//! `gpu_time_multiplier`) are consumed in `Worker::new` and dropped; policy
//! fields (`admission`, `max_batch_tokens`) live on the Admission axis. This
//! keeps `ctx` from becoming "every worker option, visible to every component"
//! (review point: config is owner-specific, not globally shared).

use crate::common::{PoolId, RequestRecord, SharedRequests, Time, WorkerId};

pub struct WorkerCtx {
    pub id: WorkerId,
    /// Paired with `id` to globally identify the worker in `record_stage`
    /// (`id` alone is only unique within a pool).
    pub pool: PoolId,
    pub requests: SharedRequests,
    /// Mirror of `WorkerConfig.log_output_token_times` (per-token ITL stamps).
    pub log_output_token_times: bool,
    /// Mirror of `WorkerConfig.log_stage_transitions` (per-request stage timeline).
    pub log_stage_transitions: bool,
}

impl WorkerCtx {
    /// Stamp a lifecycle stage transition on `record` at this worker's location.
    /// Centralizes the `(pool, id, log_stage_transitions)` triple that every
    /// `record_stage` call site in `unified.rs` repeats today.
    #[inline]
    pub(super) fn stamp_stage(&self, record: &mut RequestRecord, now: Time, code: u16) {
        record.record_stage(now, code, self.pool, self.id, self.log_stage_transitions);
    }

    #[inline]
    pub(super) fn log_tokens(&self) -> bool {
        self.log_output_token_times
    }
}
