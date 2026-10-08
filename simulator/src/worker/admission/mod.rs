//! Request lifecycle and selection axis.

use crate::common::{RequestId, Time};
use crate::worker::kv::KvStore;
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::IterBatchPlan;

mod chunked_prefill_admission;
mod decode_completion;
mod fresh_request_slot_admission;
mod local_prefill_decode_admission;
mod pipelined_chunked_prefill_admission;
mod placement;
mod policy;
mod prefill_handoff_admission;
mod token_budget;

pub use decode_completion::{
    DecodeCompletion, SingleTokenDecodeCompletion, SpeculativeDecodeCompletion,
};
pub use fresh_request_slot_admission::FreshRequestSlotAdmission;
pub use local_prefill_decode_admission::LocalPrefillDecodeAdmission;
pub use pipelined_chunked_prefill_admission::PipelinedChunkedPrefillAdmission;
pub use placement::LoadBalance;
pub(crate) use placement::PartitionLoad;
pub(crate) use policy::EnqueueSequence;
pub use policy::{
    AdmissionCandidate, FifoOrder, LongestPrefixMatch, PendingOrder, PendingOrderKind,
    PendingOrderPolicy, SessionStartOrder, ShortestJobFirst,
};
pub use prefill_handoff_admission::PrefillHandoffAdmission;
pub(crate) use token_budget::prefill_fits_budget;

pub trait IterAdmission<K: KvStore> {
    type Msg;
    type Event;

    fn accept_message(&mut self, kv_store: &mut K, msg: Self::Msg, context: &WorkerContext);
    fn form_batch(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        batch_plan: &mut IterBatchPlan,
        now: Time,
    ) -> bool;
    fn complete_iteration(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        batch_plan: &IterBatchPlan,
        events: &mut Vec<Self::Event>,
        now: Time,
    );
    fn queued_requests(&self) -> u32;
    fn cancel_pending(&mut self, request: RequestId) -> bool;
}

/// Admission surface of the pipeline-head family.
///
/// Several microbatches are in flight at once, so a request's prefill progress
/// is committed when its chunk is scheduled (`commit_microbatch`), and its
/// token and completion wait until that microbatch leaves the last stage
/// (`complete_microbatch`). A running request has at most one microbatch in
/// flight. The shell keeps one ticket per in-flight microbatch.
pub trait MicrobatchAdmission<K: KvStore> {
    /// What completion needs to know about one formed microbatch.
    type Ticket;

    fn accept_request(&mut self, kv_store: &mut K, request: RequestId, context: &WorkerContext);
    /// Schedule the next microbatch's prefill chunks into KV and its decodes
    /// into `batch_plan`. `false` when nothing can run.
    fn form_microbatch(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        batch_plan: &mut IterBatchPlan,
        now: Time,
    ) -> bool;
    /// Called after execution lowered the microbatch: commit its prefill
    /// progress so the next microbatch can carry each request's following chunk.
    fn commit_microbatch(&mut self, kv_store: &mut K, now: Time) -> Self::Ticket;
    /// The microbatch left the last stage at `at`; push the requests it finished.
    fn complete_microbatch(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        ticket: Self::Ticket,
        completed: &mut Vec<RequestId>,
        at: Time,
    );
    fn queued_requests(&self) -> u32;
}

pub trait SlotPipelineAdmission<K: KvStore> {
    fn enqueue_fresh_request(&mut self, kv_store: &K, request: RequestId, context: &WorkerContext);
    fn reserve_fitting_requests<'a>(
        &'a mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        now: Time,
    ) -> &'a [RequestId];
    fn cancel_pending(&mut self, request: RequestId) -> bool;
    fn queued_kv_tokens(&self) -> u64;
    fn queued_requests(&self) -> u32;
}
pub use chunked_prefill_admission::ChunkedPrefillAdmission;
