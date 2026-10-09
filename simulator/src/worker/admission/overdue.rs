//! Force-schedule a request that has waited too long (not vLLM).
//!
//! A shortest-first order (SPF, srpt) can leave a long prompt waiting for as
//! long as shorter work keeps arriving: under a saturated closed loop some
//! waited hours. [`OverdueQueue`] bounds that. Once a queued request has waited
//! `after` since its arrival, admission takes it out of the pending order into
//! the partition's overdue queue, which admission serves, oldest arrival first,
//! before the pending order and with no shortest-first bound. A pipeline head
//! with srpt also runs overdue started prompts before the others
//! (`pipelined_chunked_prefill_admission.rs`).

use std::collections::VecDeque;

use crate::common::{RequestId, Time};

use super::{AdmissionCandidate, PendingOrderPolicy};

pub(crate) struct OverdueQueue {
    after: Time,
    /// Per partition: queued requests in arrival order. A request that left the
    /// pending order meanwhile is skipped when it reaches the front.
    arrivals: Vec<VecDeque<(Time, RequestId)>>,
    /// Per partition: requests taken out of the pending order, oldest first.
    overdue: Vec<VecDeque<AdmissionCandidate>>,
    /// Prefill tokens the overdue requests carry, before any prefix hit.
    overdue_prompt_tokens: u64,
}

impl OverdueQueue {
    pub(crate) fn new(after_ms: f64) -> Self {
        assert!(after_ms > 0.0, "force_schedule_after_ms must be positive");
        Self {
            after: Time::from_ms(after_ms),
            arrivals: Vec::new(),
            overdue: Vec::new(),
            overdue_prompt_tokens: 0,
        }
    }

    fn ensure(&mut self, partition: usize) {
        if self.arrivals.len() <= partition {
            self.arrivals.resize_with(partition + 1, VecDeque::new);
            self.overdue.resize_with(partition + 1, VecDeque::new);
        }
    }

    /// Record a request entering `partition`'s pending order. Arrivals reach a
    /// worker in time order.
    pub(crate) fn arrived(&mut self, partition: u16, request: RequestId, arrival: Time) {
        let partition = usize::from(partition);
        self.ensure(partition);
        debug_assert!(self.arrivals[partition]
            .back()
            .map_or(true, |&(last, _)| last <= arrival));
        self.arrivals[partition].push_back((arrival, request));
    }

    /// Whether a request that arrived at `arrival` has waited past the bound.
    pub(crate) fn is_overdue(&self, arrival: Time, now: Time) -> bool {
        now.0.saturating_sub(arrival.0) >= self.after.0
    }

    /// Move every request of `partition` that has waited past the bound out of
    /// `policy` into the overdue queue.
    pub(crate) fn promote<P: PendingOrderPolicy>(
        &mut self,
        partition: u16,
        policy: &mut P,
        now: Time,
    ) {
        let partition = usize::from(partition);
        self.ensure(partition);
        while let Some(&(arrival, request)) = self.arrivals[partition].front() {
            if !self.is_overdue(arrival, now) {
                break;
            }
            self.arrivals[partition].pop_front();
            if let Some(candidate) = policy.remove(request) {
                self.overdue_prompt_tokens += u64::from(candidate.fresh_prompt_tokens);
                self.overdue[partition].push_back(candidate);
            }
        }
    }

    pub(crate) fn peek(&self, partition: u16) -> Option<AdmissionCandidate> {
        self.overdue
            .get(usize::from(partition))
            .and_then(|queue| queue.front().copied())
    }

    pub(crate) fn pop(&mut self, partition: u16) -> Option<AdmissionCandidate> {
        let candidate = self.overdue.get_mut(usize::from(partition))?.pop_front()?;
        self.overdue_prompt_tokens -= u64::from(candidate.fresh_prompt_tokens);
        Some(candidate)
    }

    /// Put a request [`Self::pop`] took back at the front of its queue.
    pub(crate) fn push_front(&mut self, partition: u16, candidate: AdmissionCandidate) {
        let partition = usize::from(partition);
        self.ensure(partition);
        self.overdue_prompt_tokens += u64::from(candidate.fresh_prompt_tokens);
        self.overdue[partition].push_front(candidate);
    }

    /// Drop `request` from the overdue queue, if it is there.
    pub(crate) fn remove(&mut self, request: RequestId) -> bool {
        for queue in &mut self.overdue {
            if let Some(index) = queue.iter().position(|c| c.request_id == request) {
                let candidate = queue.remove(index).expect("index is in range");
                self.overdue_prompt_tokens -= u64::from(candidate.fresh_prompt_tokens);
                return true;
            }
        }
        false
    }

    pub(crate) fn len(&self) -> usize {
        self.overdue.iter().map(VecDeque::len).sum()
    }

    pub(crate) fn prompt_tokens(&self) -> u64 {
        self.overdue_prompt_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::SessionInput;
    use crate::worker::admission::{EnqueueSequence, FifoOrder};

    #[test]
    fn promote_moves_only_requests_past_the_bound_oldest_first() {
        let mut sequence = EnqueueSequence::default();
        let mut policy = FifoOrder::default();
        let mut overdue = OverdueQueue::new(10.0);
        for (id, arrival_ms) in [(0, 0.0), (1, 4.0), (2, 8.0)] {
            let candidate = sequence.freeze(
                RequestId(id),
                100,
                1,
                SessionInput::Standalone,
                Time::ZERO,
                0,
            );
            policy.push(candidate, &mut ());
            overdue.arrived(0, RequestId(id), Time::from_ms(arrival_ms));
        }
        // Request 0 was admitted the normal way before it became overdue.
        policy.remove(RequestId(0));
        overdue.promote(0, &mut policy, Time::from_ms(15.0));
        assert_eq!(overdue.pop(0).map(|c| c.request_id), Some(RequestId(1)));
        assert_eq!(overdue.pop(0), None, "request 2 has waited 7 ms");
        assert!(policy.contains(RequestId(2)));
        overdue.promote(0, &mut policy, Time::from_ms(18.0));
        assert_eq!(overdue.prompt_tokens(), 100);
        assert!(overdue.remove(RequestId(2)));
        assert_eq!((overdue.len(), overdue.prompt_tokens()), (0, 0));
    }
}
