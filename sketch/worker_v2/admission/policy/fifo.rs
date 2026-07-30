//! `FifoOrder` — the throughput baseline selection policy (AP0).
//!
//! Arrival order, so the queue IS the order: a `VecDeque` whose head is the answer, with
//! O(1) push/peek/pop. That is exactly what the production `worker/unified.rs` does with
//! `pending_prefills.front()` — the sketch now matches its cost, not just its behavior.
//!
//! Kept as a real `PendingOrderPolicy` impl (never special-cased in a lifecycle)
//! so the seam stays exercised: swapping to
//! `ShortestJobFirst`/`EarliestDeadline` changes only a type param.

use std::collections::VecDeque;

use crate::common::RequestId;

use super::{AdmissionCandidate, PendingOrderPolicy};

#[derive(Default)]
pub struct FifoOrder {
    queue: VecDeque<AdmissionCandidate>,
}

impl FifoOrder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PendingOrderPolicy for FifoOrder {
    type Context = ();

    #[inline]
    fn push(&mut self, candidate: AdmissionCandidate, _ctx: &mut ()) {
        self.queue.push_back(candidate);
    }

    #[inline]
    fn peek(&self) -> Option<AdmissionCandidate> {
        self.queue.front().copied()
    }

    #[inline]
    fn pop(&mut self, _ctx: &mut ()) -> Option<AdmissionCandidate> {
        self.queue.pop_front()
    }

    fn remove(&mut self, req: RequestId) -> bool {
        match self
            .queue
            .iter()
            .position(|candidate| candidate.request == req)
        {
            Some(position) => {
                self.queue.remove(position);
                true
            }
            None => false,
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.queue.len()
    }
}
