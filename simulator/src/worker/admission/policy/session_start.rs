//! Oldest-session-first pending-request selection.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::common::RequestId;

use super::{AdmissionCandidate, PendingOrderPolicy};

/// Reverse the natural heap order so the earliest session/arrival wins.
struct EarliestSessionStart(AdmissionCandidate);

impl Ord for EarliestSessionStart {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .conversation_start_time
            .cmp(&self.0.conversation_start_time)
            .then_with(|| other.0.enqueue_sequence.cmp(&self.0.enqueue_sequence))
    }
}

impl PartialOrd for EarliestSessionStart {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for EarliestSessionStart {}

impl PartialEq for EarliestSessionStart {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

/// Prefer requests belonging to the conversation that started earliest.
/// Standalone requests use their own arrival as a one-request session start.
#[derive(Default)]
pub struct SessionStartOrder {
    queue: BinaryHeap<EarliestSessionStart>,
    queued_kv_tokens: u64,
}

impl SessionStartOrder {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PendingOrderPolicy for SessionStartOrder {
    type Context = ();

    fn push(&mut self, candidate: AdmissionCandidate, _context: &mut Self::Context) {
        self.queued_kv_tokens += candidate.queued_kv_tokens();
        self.queue.push(EarliestSessionStart(candidate));
    }

    fn peek(&self) -> Option<AdmissionCandidate> {
        self.queue.peek().map(|entry| entry.0)
    }

    fn pop(&mut self, _context: &mut Self::Context) -> Option<AdmissionCandidate> {
        let candidate = self.queue.pop()?.0;
        self.queued_kv_tokens -= candidate.queued_kv_tokens();
        Some(candidate)
    }

    /// Cancellation is not on the per-iteration hot path. Rebuilding the heap
    /// keeps normal selection incremental and preserves one owner of membership.
    fn remove(&mut self, request: RequestId) -> Option<AdmissionCandidate> {
        let mut removed = None;
        let retained = std::mem::take(&mut self.queue)
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
            .collect::<Vec<_>>();
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
        conversation_start_time: Time,
        fresh_prompt_tokens: u32,
        remaining_output_tokens: u32,
    ) -> AdmissionCandidate {
        AdmissionCandidate {
            request_id: RequestId(request_id),
            enqueue_sequence,
            fresh_prompt_tokens,
            remaining_output_tokens,
            session_input: SessionInput::Standalone,
            conversation_start_time,
        }
    }

    #[test]
    fn later_round_of_older_session_ranks_ahead_of_newer_session() {
        let mut policy = SessionStartOrder::new();
        policy.push(candidate(0, 0, Time::from_ms_u64(20), 10, 2), &mut ());
        policy.push(candidate(1, 1, Time::from_ms_u64(5), 4, 1), &mut ());

        assert_eq!(policy.pop(&mut ()).unwrap().request_id, RequestId(1));
    }

    #[test]
    fn equal_session_start_uses_monotonic_enqueue_sequence() {
        let mut policy = SessionStartOrder::new();
        policy.push(candidate(0, 7, Time::from_ms_u64(5), 4, 1), &mut ());
        policy.push(candidate(1, 8, Time::from_ms_u64(5), 4, 1), &mut ());

        assert_eq!(policy.pop(&mut ()).unwrap().request_id, RequestId(0));
    }

    #[test]
    fn remove_updates_membership_and_queued_kv() {
        let mut policy = SessionStartOrder::new();
        policy.push(candidate(0, 0, Time::ZERO, 10, 2), &mut ());
        policy.push(candidate(1, 1, Time::ZERO, 4, 1), &mut ());

        assert_eq!(
            policy.remove(RequestId(0)).unwrap().request_id,
            RequestId(0)
        );
        assert!(!policy.contains(RequestId(0)));
        assert_eq!(policy.queued_kv_tokens(), 5);
    }
}
