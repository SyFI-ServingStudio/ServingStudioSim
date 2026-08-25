//! Request lifecycle and selection axis.

use crate::common::{RequestId, Time};
use crate::worker::kv::KvStore;
use crate::worker::shared::context::WorkerContext;

mod chunked_prefill_admission;
mod fresh_request_slot_admission;
mod local_prefill_decode_admission;
mod placement;
mod policy;
mod prefill_handoff_admission;
mod token_budget;

pub use fresh_request_slot_admission::FreshRequestSlotAdmission;
pub use local_prefill_decode_admission::LocalPrefillDecodeAdmission;
pub use placement::LoadBalance;
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
    fn form_batch(&mut self, kv_store: &mut K, context: &WorkerContext, now: Time) -> bool;
    fn complete_iteration(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        events: &mut Vec<Self::Event>,
        now: Time,
    );
    fn queued_requests(&self) -> u32;
    fn cancel_pending(&mut self, request: RequestId) -> bool;
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
