//! Incremental shortest-job-first pending-request selection, keyed on the
//! request's whole KV footprint or on just the prefill it still has to compute.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::common::RequestId;

use super::{AdmissionCandidate, PendingOrderPolicy};

/// What "job size" means for [`ShortestJobFirst`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum JobSize {
    /// Prompt, declared prefix and remaining output: the KV the request holds.
    #[default]
    KvTokens,
    /// Prefill tokens to compute: the fresh prompt plus whatever part of the
    /// declared prefix was not resident at enqueue. This is the work that
    /// stands between the request and its first token.
    PrefillTokens,
}

impl JobSize {
    fn of(self, candidate: &AdmissionCandidate) -> u64 {
        match self {
            Self::KvTokens => candidate.queued_kv_tokens(),
            Self::PrefillTokens => {
                let declared = candidate.session_input.declared_prefix_tokens();
                u64::from(candidate.fresh_prompt_tokens)
                    + u64::from(declared.saturating_sub(candidate.resident_prefix_tokens))
            }
        }
    }
}

struct ShortestWorkFirst {
    candidate: AdmissionCandidate,
    work: u64,
}

impl ShortestWorkFirst {
    #[inline]
    fn work(&self) -> u64 {
        self.work
    }
}

impl Ord for ShortestWorkFirst {
    fn cmp(&self, other: &Self) -> Ordering {
        other.work().cmp(&self.work()).then_with(|| {
            other
                .candidate
                .enqueue_sequence
                .cmp(&self.candidate.enqueue_sequence)
        })
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
    queued_prompt_tokens: u64,
    size: JobSize,
}

impl ShortestJobFirst {
    pub fn new() -> Self {
        Self::default()
    }

    /// Order by `size` instead of the KV footprint.
    pub fn by(size: JobSize) -> Self {
        Self {
            size,
            ..Self::default()
        }
    }
}

impl PendingOrderPolicy for ShortestJobFirst {
    type Context = ();

    fn push(&mut self, candidate: AdmissionCandidate, _context: &mut Self::Context) {
        self.queued_kv_tokens += candidate.queued_kv_tokens();
        self.queued_prompt_tokens += u64::from(candidate.fresh_prompt_tokens);
        let work = self.size.of(&candidate);
        self.queue.push(ShortestWorkFirst { candidate, work });
    }

    fn peek(&self) -> Option<AdmissionCandidate> {
        self.queue.peek().map(|entry| entry.candidate)
    }

    fn pop(&mut self, _context: &mut Self::Context) -> Option<AdmissionCandidate> {
        let candidate = self.queue.pop()?.candidate;
        self.queued_kv_tokens -= candidate.queued_kv_tokens();
        self.queued_prompt_tokens -= u64::from(candidate.fresh_prompt_tokens);
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
                if entry.candidate.request_id == request {
                    removed = Some(entry.candidate);
                    None
                } else {
                    Some(entry)
                }
            })
            .collect();
        self.queue = BinaryHeap::from(retained);
        if let Some(candidate) = removed {
            self.queued_kv_tokens -= candidate.queued_kv_tokens();
            self.queued_prompt_tokens -= u64::from(candidate.fresh_prompt_tokens);
        }
        removed
    }

    fn contains(&self, request: RequestId) -> bool {
        self.queue
            .iter()
            .any(|entry| entry.candidate.request_id == request)
    }

    fn len(&self) -> usize {
        self.queue.len()
    }

    fn queued_kv_tokens(&self) -> u64 {
        self.queued_kv_tokens
    }

    fn queued_prompt_tokens(&self) -> u64 {
        self.queued_prompt_tokens
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
            retracted: false,
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
    fn prefill_size_ignores_a_resident_prefix_and_counts_a_missing_one() {
        let mut policy = ShortestJobFirst::by(JobSize::PrefillTokens);
        // 900 fresh tokens behind a fully resident 100k prefix: 900 to compute.
        let mut cached = candidate(0, 0, 900, 1);
        cached.session_input = SessionInput::PinnedPrefix {
            prefix_tokens: 100_000,
        };
        cached.resident_prefix_tokens = 100_000;
        // 500 fresh tokens whose 1000-token prefix is gone: 1500 to compute.
        let mut evicted = candidate(1, 1, 500, 1);
        evicted.session_input = SessionInput::PinnedPrefix {
            prefix_tokens: 1_000,
        };
        policy.push(evicted, &mut ());
        policy.push(cached, &mut ());

        assert_eq!(policy.pop(&mut ()).unwrap().request_id, RequestId(0));
        assert_eq!(policy.pop(&mut ()).unwrap().request_id, RequestId(1));
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
        assert_eq!(policy.queued_prompt_tokens(), 4);
    }
}
