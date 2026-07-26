//! Admission axis — lifecycle × selection (interfaces doc §3).
//!
//! `IterAdmission<K>` takes the KV type as a trait parameter (NOT a method-generic),
//! so a lifecycle impl can require capability sub-traits on `K` in its `impl`
//! block (`LocalPrefillDecodeAdmission` requires `K: IterWorkerKv`;
//! `PrefillHandoffAdmission` would require `K: HandoffKv`). This is the M2 hard
//! point resolution: a method-generic
//! `form_batch<K>` could not add a stronger bound than the trait declares.

use crate::common::{RequestId, Time};

use super::execution::DraftVerifyResult;
use super::kv::{KvStore, SpeculativeKv};
use super::shared::context::WorkerContext;

mod chunked_prefill_admission;
mod draft_verify_admission;
mod fresh_request_slot_admission;
mod local_prefill_decode_admission;
mod multi_model_admission;
pub mod policy;
mod prefill_handoff_admission;
mod prefix_prefill_decode_admission;

pub use chunked_prefill_admission::ChunkedPrefillAdmission;
pub use draft_verify_admission::DraftVerifyAdmission;
pub use fresh_request_slot_admission::FreshRequestSlotAdmission;
pub use local_prefill_decode_admission::LocalPrefillDecodeAdmission;
pub use multi_model_admission::MultiModelAdmission;
pub use prefill_handoff_admission::PrefillHandoffAdmission;
pub use prefix_prefill_decode_admission::PrefixPrefillDecodeAdmission;

pub trait IterAdmission<K: KvStore> {
    type Msg;
    type Event;

    /// Handle an incoming message. Usually enqueues + stamps `Pending`, but the
    /// message may also touch KV out of the iter cycle — e.g. PD's `ReleaseKv`
    /// dropping a held reservation (`kv_store.drop_held`) — so `accept_message` gets
    /// `&mut K` (ref `accept`).
    /// Lifecycles with no such message simply ignore it.
    fn accept_message(&mut self, kv_store: &mut K, msg: Self::Msg, context: &WorkerContext);

    /// Token gate + selection policy + `kv_store.fits` + `kv_store.reserve` → this iter's work.
    /// Returns whether this iter has any work. Stamps `Prefill` on admit.
    fn form_batch(&mut self, kv_store: &mut K, context: &WorkerContext, now: Time) -> bool;

    /// Iter-end: record tokens, drive Decode/Done, commit/advance/release via KV
    /// (ref `on_iter_complete`).
    fn complete_iteration(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        events: &mut Vec<Self::Event>,
        now: Time,
    );

    fn queued_requests(&self) -> u32;

    /// Cancellation's pending half (container composes with `kv_store.release_external`).
    fn cancel_pending(&mut self, req: RequestId) -> bool;
}

/// Admission surface paired with the S6 draft/verify cadence.
///
/// Fresh-request selection remains an admission concern, but iteration
/// completion consumes the execution's per-request outcome instead of assuming
/// that every live decode advanced by the same scalar.
pub trait DraftVerifyAdmissionLifecycle<K: SpeculativeKv> {
    type Msg;
    type Event;

    fn accept_message(&mut self, kv_store: &mut K, msg: Self::Msg, context: &WorkerContext);

    fn form_batch(&mut self, kv_store: &mut K, context: &WorkerContext, now: Time) -> bool;

    fn complete_draft_verify_iteration(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        result: DraftVerifyResult,
        events: &mut Vec<Self::Event>,
        now: Time,
    );

    fn queued_requests(&self) -> u32;
    fn cancel_pending(&mut self, request: RequestId) -> bool;
}

/// AFD-attn family admission (interfaces doc §3, per-family surface — NOT the
/// same trait as `IterAdmission`). Two levels: L1 `enqueue_fresh_request`
/// enqueues fresh (no KV); L2 `reserve_fitting_requests` is the KV-gated reserve
/// that returns admitted ids for the SHELL to place into pipeline slots. There
/// is no `form_batch`/`complete_iteration`
/// here: the AFD shell owns the layer cadence + slot batching, and completion
/// lives in the paired FFN worker, not here. Its only KV writes are reserve (via
/// `kv_store.reserve`) and release (via `kv_store.release`); commit→resident fires at a layer
/// boundary and is driven by the shell (`kv_store.commit_resident`).
pub trait SlotPipelineAdmission<K: KvStore> {
    /// Level-1: enqueue a fresh request; stamps `Pending`. No KV yet.
    fn enqueue_fresh_request(&mut self, req: RequestId, context: &WorkerContext);

    /// Level-2: reserve the FULL footprint for as many head-of-line requests as
    /// KV fits; returns admitted ids in admission order for the shell to slot.
    fn reserve_fitting_requests(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        now: Time,
    ) -> Vec<RequestId>;

    /// Cancellation: drop from the pending queue if still queued, else release its
    /// live KV. Returns whether the request was known to this admission.
    fn cancel_or_release_request(&mut self, kv_store: &mut K, req: RequestId) -> bool;

    /// Token demand of queued fresh requests that have not reserved KV yet
    /// (ref `reserved_kv`; feeds L6 `estimated_peak_kv`).
    fn queued_kv_tokens(&self) -> u64;

    fn queued_requests(&self) -> u32;
}
