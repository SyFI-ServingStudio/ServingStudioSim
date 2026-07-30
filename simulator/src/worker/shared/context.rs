//! Component-neutral facts shared inside one composed L5 worker.
//!
//! Construction-only fields are deliberately absent. Policy configuration belongs
//! to Admission, resource configuration belongs to KV, and model/cost state belongs
//! to Execution.

use crate::common::{PoolId, RequestRecord, SharedRequests, Time, WorkerId};

pub struct WorkerContext {
    pub(crate) id: WorkerId,
    pub(crate) pool: PoolId,
    pub(crate) requests: SharedRequests,
    pub(crate) log_output_token_times: bool,
    pub(crate) log_stage_transitions: bool,
}

impl WorkerContext {
    #[inline]
    pub(crate) fn stamp_stage(&self, record: &mut RequestRecord, now: Time, stage: u16) {
        record.record_stage(now, stage, self.pool, self.id, self.log_stage_transitions);
    }

    #[inline]
    pub(crate) fn log_tokens(&self) -> bool {
        self.log_output_token_times
    }
}
