//! FIFO pending-request selection.

use std::collections::VecDeque;

use crate::common::RequestId;

use super::{AdmissionCandidate, PendingOrderPolicy};

#[derive(Default)]
pub struct FifoOrder {
    queue: VecDeque<AdmissionCandidate>,
    queued_kv_tokens: u64,
}

impl FifoOrder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PendingOrderPolicy for FifoOrder {
    type Context = ();

    #[inline]
    fn push(&mut self, candidate: AdmissionCandidate, _context: &mut Self::Context) {
        self.queued_kv_tokens += candidate.queued_kv_tokens();
        self.queue.push_back(candidate);
    }

    #[inline]
    fn peek(&self) -> Option<AdmissionCandidate> {
        self.queue.front().copied()
    }

    #[inline]
    fn pop(&mut self, _context: &mut Self::Context) -> Option<AdmissionCandidate> {
        let candidate = self.queue.pop_front()?;
        self.queued_kv_tokens -= candidate.queued_kv_tokens();
        Some(candidate)
    }

    fn remove(&mut self, request: RequestId) -> Option<AdmissionCandidate> {
        let position = self
            .queue
            .iter()
            .position(|candidate| candidate.request == request)?;
        let candidate = self
            .queue
            .remove(position)
            .expect("position came from the same FIFO");
        self.queued_kv_tokens -= candidate.queued_kv_tokens();
        Some(candidate)
    }

    #[inline]
    fn contains(&self, request: RequestId) -> bool {
        self.queue
            .iter()
            .any(|candidate| candidate.request == request)
    }

    #[inline]
    fn len(&self) -> usize {
        self.queue.len()
    }

    #[inline]
    fn queued_kv_tokens(&self) -> u64 {
        self.queued_kv_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::PrefixInput;

    fn candidate(request: u32, prompt: u32, decode: u32) -> AdmissionCandidate {
        AdmissionCandidate {
            request: RequestId(request),
            enqueue_sequence: u64::from(request),
            prompt,
            decode,
            prefix: PrefixInput::None,
            deadline: None,
            matched_tokens: 0,
        }
    }

    #[test]
    fn preserves_arrival_order_and_tracks_queued_kv() {
        let mut policy = FifoOrder::new();
        policy.push(candidate(0, 10, 2), &mut ());
        policy.push(candidate(1, 4, 1), &mut ());

        assert_eq!(policy.peek().unwrap().request, RequestId(0));
        assert_eq!(policy.queued_kv_tokens(), 17);
        assert_eq!(policy.pop(&mut ()).unwrap().request, RequestId(0));
        assert_eq!(policy.queued_kv_tokens(), 5);
    }

    #[test]
    fn remove_updates_membership_and_queued_kv() {
        let mut policy = FifoOrder::new();
        policy.push(candidate(0, 10, 2), &mut ());
        policy.push(candidate(1, 4, 1), &mut ());

        assert_eq!(policy.remove(RequestId(1)).unwrap().request, RequestId(1));
        assert!(!policy.contains(RequestId(1)));
        assert_eq!(policy.queued_kv_tokens(), 12);
    }
}
