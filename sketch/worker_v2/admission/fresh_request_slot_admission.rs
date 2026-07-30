//! `FreshRequestSlotAdmission` — the AFD-attn lifecycle (A "reserve-on-admit,
//! no per-iter
//! batch selection"). Owns a FIFO pending queue of fresh requests; L2
//! `reserve_fitting_requests` reserves the full footprint (prompt+decode) for as many
//! head-of-line requests as KV fits and hands the admitted ids to the shell.
//!
//! No `PendingOrderPolicy` here in v1 (AFD admits FIFO); a policy-parameterized
//! variant would slot in the same way `LocalPrefillDecodeAdmission<P>` does.
//! Stage codes use the
//! AFD lifecycle vocab `AfdStage` (Pending on enqueue, Prefill on reserve). External
//! calls verified against real source: `RequestStore::mark_admitted`, `RequestRecord`
//! field reads, `AfdStage` (disagg_attn.rs uses the same two stamps).

use std::collections::VecDeque;

use crate::common::{AfdStage, RequestId, Time};

use super::super::kv::KvStore;
use super::super::shared::context::WorkerContext;
use super::SlotPipelineAdmission;

pub struct FreshRequestSlotAdmission {
    /// FIFO of fresh requests, carrying their queued `(prompt, decode)` demand so
    /// `queued_kv_tokens` is a running sum without re-reading the store.
    pending: VecDeque<(RequestId, u32, u32)>,
}

impl FreshRequestSlotAdmission {
    pub fn new() -> Self {
        Self {
            pending: VecDeque::new(),
        }
    }
}

impl<K: KvStore> SlotPipelineAdmission<K> for FreshRequestSlotAdmission {
    fn enqueue_fresh_request(&mut self, req: RequestId, context: &WorkerContext) {
        let mut store = context.requests.borrow_mut();
        let record = &mut store[req];
        let (prompt, decode, arrival) = (record.prompt_len, record.decode_len, record.arrival_time);
        context.stamp_stage(record, arrival, AfdStage::Pending as u16);
        drop(store);
        self.pending.push_back((req, prompt, decode));
    }

    fn reserve_fitting_requests(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        now: Time,
    ) -> Vec<RequestId> {
        let mut admitted: Vec<RequestId> = Vec::new();
        // FIFO head-of-line: stop at the first request KV cannot fit.
        while let Some(&(req, prompt, decode)) = self.pending.front() {
            let footprint = kv_store.footprint(req, prompt, decode);
            if !kv_store.fits(0, &footprint) {
                break;
            }
            kv_store.reserve(req, 0, footprint);
            self.pending.pop_front();

            let mut store = context.requests.borrow_mut();
            store.mark_admitted(req);
            let record = &mut store[req];
            record.active_chunk_len = prompt;
            record.prefix_kv = 0;
            context.stamp_stage(record, now, AfdStage::Prefill as u16);

            admitted.push(req);
        }
        admitted
    }

    fn cancel_or_release_request(&mut self, kv_store: &mut K, req: RequestId) -> bool {
        if let Some(position) = self
            .pending
            .iter()
            .position(|&(pending, _, _)| pending == req)
        {
            self.pending.remove(position);
            return true;
        }
        // Already reserved/resident → drop its KV (batch recomputes current_kv).
        kv_store.release(req, 0);
        true
    }

    #[inline]
    fn queued_kv_tokens(&self) -> u64 {
        self.pending
            .iter()
            .map(|&(_, prompt, decode)| u64::from(prompt + decode))
            .sum()
    }

    #[inline]
    fn queued_requests(&self) -> u32 {
        self.pending.len() as u32
    }
}
