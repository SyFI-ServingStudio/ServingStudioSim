//! Incremental shortest-job-first pending-request selection.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::common::RequestId;

use super::{AdmissionCandidate, PendingOrderPolicy};

struct ShortestWorkFirst(AdmissionCandidate);

impl ShortestWorkFirst {
    #[inline]
    fn work(&self) -> u64 {
        self.0.queued_kv_tokens()
    }
}

impl Ord for ShortestWorkFirst {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .work()
            .cmp(&self.work())
            .then_with(|| other.0.enqueue_sequence.cmp(&self.0.enqueue_sequence))
    }
}

impl PartialOrd for ShortestWorkFirst {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for ShortestWorkFirst {}

impl PartialEq for ShortestWorkFirst {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

#[derive(Default)]
pub struct ShortestJobFirst {
    queue: BinaryHeap<ShortestWorkFirst>,
    queued_kv_tokens: u64,
}

impl ShortestJobFirst {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl PendingOrderPolicy for ShortestJobFirst {
    type Context = ();

    fn push(&mut self, candidate: AdmissionCandidate, _context: &mut Self::Context) {
        self.queued_kv_tokens += candidate.queued_kv_tokens();
        self.queue.push(ShortestWorkFirst(candidate));
    }

    fn peek(&self) -> Option<AdmissionCandidate> {
        self.queue.peek().map(|entry| entry.0)
    }

    fn pop(&mut self, _context: &mut Self::Context) -> Option<AdmissionCandidate> {
        let candidate = self.queue.pop()?.0;
        self.queued_kv_tokens -= candidate.queued_kv_tokens();
        Some(candidate)
    }

    /// `BinaryHeap` has no keyed removal. Cancellation is a rare control path;
    /// rebuilding here keeps the per-iteration `peek`/`pop` path incremental.
    fn remove(&mut self, request: RequestId) -> Option<AdmissionCandidate> {
        let mut removed = None;
        let retained: Vec<ShortestWorkFirst> = std::mem::take(&mut self.queue)
            .into_vec()
            .into_iter()
            .filter_map(|entry| {
                if entry.0.request_id == request {
                    removed = Some(entry.0);
                    None
                } else {
                    Some(entry)
                }
            })
            .collect();
        self.queue = BinaryHeap::from(retained);
        if let Some(candidate) = removed {
            self.queued_kv_tokens -= candidate.queued_kv_tokens();
        }
        removed
    }

    fn contains(&self, request: RequestId) -> bool {
        self.queue
            .iter()
            .any(|candidate| candidate.0.request_id == request)
    }

    fn len(&self) -> usize {
        self.queue.len()
    }

    fn queued_kv_tokens(&self) -> u64 {
        self.queued_kv_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{SessionInput, Time};

    fn candidate(
        request_id: u32,
        enqueue_sequence: u64,
        fresh_prompt_tokens: u32,
        remaining_output_tokens: u32,
    ) -> AdmissionCandidate {
        AdmissionCandidate {
            request_id: RequestId(request_id),
            enqueue_sequence,
            fresh_prompt_tokens,
            remaining_output_tokens,
            session_input: SessionInput::Standalone,
            conversation_start_time: Time::ZERO,
            resident_prefix_tokens: 0,
        }
    }

    #[test]
    fn selects_the_smallest_total_job() {
        let mut policy = ShortestJobFirst::new();
        policy.push(candidate(0, 0, 10, 2), &mut ());
        policy.push(candidate(1, 1, 4, 1), &mut ());

        assert_eq!(policy.peek().unwrap().request_id, RequestId(1));
        assert_eq!(policy.pop(&mut ()).unwrap().request_id, RequestId(1));
        assert_eq!(policy.queued_kv_tokens(), 12);
    }

    #[test]
    fn equal_work_uses_monotonic_enqueue_sequence() {
        let mut policy = ShortestJobFirst::new();
        policy.push(candidate(0, 7, 4, 1), &mut ());
        policy.push(candidate(1, 8, 3, 2), &mut ());

        assert_eq!(policy.pop(&mut ()).unwrap().request_id, RequestId(0));
    }

    #[test]
    fn remove_updates_membership_and_queued_kv() {
        let mut policy = ShortestJobFirst::new();
        policy.push(candidate(0, 0, 10, 2), &mut ());
        policy.push(candidate(1, 1, 4, 1), &mut ());

        assert_eq!(
            policy.remove(RequestId(0)).unwrap().request_id,
            RequestId(0)
        );
        assert!(!policy.contains(RequestId(0)));
        assert_eq!(policy.queued_kv_tokens(), 5);
    }
}
