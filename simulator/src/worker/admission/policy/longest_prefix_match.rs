//! Most-prefix-matched-first pending-request selection.
//!
//! Unlike the other policies, the ranking key here is not an enqueue-time fact:
//! how much of a request's declared prefix is actually resident decays with
//! every eviction. Re-scoring the whole pending set each time the lifecycle
//! forms a batch would be O(n) per iteration, which this hot path cannot pay.
//!
//! Instead the heap holds a cached key that is always an **upper bound** on the
//! live value, and only the head is ever reconciled:
//!
//! * the seed is the exact resident length measured at enqueue
//!   (`AdmissionCandidate::resident_prefix_tokens`) — deliberately not the
//!   declared length, which would over-state every request whose prefix has
//!   already been evicted and make the first reconciliation walk the queue;
//! * a queued request's resident prefix only ever shrinks — it grows only when
//!   its own session returns KV, which happened before this request was even
//!   enqueued.
//!
//! So every element's cached key dominates its live value, and the head's cached
//! key dominates every other cached key. When the head's live value equals its
//! cached key, the head is a true maximum and no other element needs to be
//! looked at. When it is lower, the head is re-inserted at the corrected key and
//! the check repeats. Each correction strictly lowers one key, so the work is
//! O(1) amortized — one hash lookup in the common case.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::common::RequestId;

use super::{AdmissionCandidate, PendingOrderPolicy};

/// A candidate plus its cached upper bound on resident prefix tokens.
struct MostPrefixMatched {
    /// Upper bound on the live resident prefix. Lowered in place by
    /// `refresh_head`; never raised.
    matched_prefix_tokens: u32,
    candidate: AdmissionCandidate,
}

impl Ord for MostPrefixMatched {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: more matched prefix wins; the earlier arrival breaks ties.
        self.matched_prefix_tokens
            .cmp(&other.matched_prefix_tokens)
            .then_with(|| {
                other
                    .candidate
                    .enqueue_sequence
                    .cmp(&self.candidate.enqueue_sequence)
            })
    }
}

impl PartialOrd for MostPrefixMatched {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for MostPrefixMatched {}

impl PartialEq for MostPrefixMatched {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

/// Prefer the request that would reuse the most already-resident prefix KV,
/// so a batch recomputes as few evicted tokens as possible.
#[derive(Default)]
pub struct LongestPrefixMatch {
    queue: BinaryHeap<MostPrefixMatched>,
    queued_kv_tokens: u64,
}

impl LongestPrefixMatch {
    pub fn new() -> Self {
        Self::default()
    }
}

impl PendingOrderPolicy for LongestPrefixMatch {
    type Context = ();

    fn push(&mut self, candidate: AdmissionCandidate, _context: &mut Self::Context) {
        self.queued_kv_tokens += candidate.queued_kv_tokens();
        self.queue.push(MostPrefixMatched {
            matched_prefix_tokens: candidate.resident_prefix_tokens,
            candidate,
        });
    }

    /// Reconcile the head against live KV state. See the module comment for why
    /// touching only the head is exact.
    fn refresh_head(&mut self, matched_prefix_tokens: &mut dyn FnMut(AdmissionCandidate) -> u32) {
        while let Some(head) = self.queue.peek() {
            let live = matched_prefix_tokens(head.candidate);
            if live >= head.matched_prefix_tokens {
                // Not an over-estimate, so the head is a true maximum. (A live
                // value above the cached one cannot demote the head either: it
                // already dominates every other cached key.)
                break;
            }
            let mut stale = self.queue.pop().expect("peeked a head");
            stale.matched_prefix_tokens = live;
            self.queue.push(stale);
        }
    }

    fn peek(&self) -> Option<AdmissionCandidate> {
        self.queue.peek().map(|entry| entry.candidate)
    }

    fn pop(&mut self, _context: &mut Self::Context) -> Option<AdmissionCandidate> {
        let candidate = self.queue.pop()?.candidate;
        self.queued_kv_tokens -= candidate.queued_kv_tokens();
        Some(candidate)
    }

    /// Cancellation is not on the per-iteration hot path; rebuilding the heap
    /// keeps normal selection incremental and preserves one owner of membership.
    fn remove(&mut self, request: RequestId) -> Option<AdmissionCandidate> {
        let mut removed = None;
        let retained = std::mem::take(&mut self.queue)
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
            .any(|entry| entry.candidate.request_id == request)
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
    use std::collections::HashMap;

    use super::*;
    use crate::common::{SessionInput, Time};

    /// `declared` and `resident` are supplied separately on purpose: the whole
    /// point of the policy is that they differ once anything has been evicted.
    fn candidate(
        request_id: u32,
        enqueue_sequence: u64,
        session_id: u32,
        declared_prefix_tokens: u32,
        resident_prefix_tokens: u32,
    ) -> AdmissionCandidate {
        AdmissionCandidate {
            request_id: RequestId(request_id),
            enqueue_sequence,
            fresh_prompt_tokens: 16,
            remaining_output_tokens: 8,
            session_input: SessionInput::Session {
                session_id,
                session_start_time: Time::ZERO,
                declared_prefix_tokens,
            },
            conversation_start_time: Time::ZERO,
            resident_prefix_tokens,
        }
    }

    /// A scorer standing in for the prefix cache, counting how often it is asked.
    struct Resident {
        by_session: HashMap<u32, u32>,
        lookups: u32,
    }

    impl Resident {
        fn score(&mut self, candidate: AdmissionCandidate) -> u32 {
            self.lookups += 1;
            let SessionInput::Session { session_id, .. } = candidate.session_input else {
                return 0;
            };
            self.by_session.get(&session_id).copied().unwrap_or(0)
        }
    }

    fn resident(entries: &[(u32, u32)]) -> Resident {
        Resident {
            by_session: entries.iter().copied().collect(),
            lookups: 0,
        }
    }

    #[test]
    fn ranking_follows_resident_prefix_not_the_larger_declared_one() {
        let mut policy = LongestPrefixMatch::new();
        // Request 1 declares the largest prefix but holds none of it; request 2
        // declares less and holds all of it. Ranking on `declared` would put 1
        // first and recompute 100k tokens.
        policy.push(candidate(1, 0, 10, 100_000, 0), &mut ());
        policy.push(candidate(2, 1, 20, 40_000, 40_000), &mut ());

        assert_eq!(policy.peek().expect("a head").request_id, RequestId(2));
    }

    #[test]
    fn an_intact_head_costs_one_lookup_regardless_of_queue_depth() {
        let mut policy = LongestPrefixMatch::new();
        policy.push(candidate(1, 0, 10, 90_000, 90_000), &mut ());
        for index in 1..64u32 {
            policy.push(
                candidate(index + 1, u64::from(index) + 1, 20 + index, 1_000, 1_000),
                &mut (),
            );
        }

        let mut cache = resident(&[(10, 90_000)]);
        policy.refresh_head(&mut |c| cache.score(c));

        assert_eq!(policy.peek().expect("a head").request_id, RequestId(1));
        // The head's cached key was already exact, so nothing else was scored:
        // reconciliation is O(1), not O(queue depth).
        assert_eq!(cache.lookups, 1);
    }

    #[test]
    fn eviction_while_queued_demotes_the_head_and_promotes_the_next_best() {
        let mut policy = LongestPrefixMatch::new();
        policy.push(candidate(1, 0, 10, 90_000, 90_000), &mut ());
        policy.push(candidate(2, 1, 20, 50_000, 50_000), &mut ());
        assert_eq!(policy.peek().expect("a head").request_id, RequestId(1));

        // Session 10 is evicted while both requests are still queued, so the
        // enqueue-time key on request 1 is now stale-high.
        let mut after = resident(&[(10, 0), (20, 50_000)]);
        policy.refresh_head(&mut |c| after.score(c));

        assert_eq!(policy.peek().expect("a head").request_id, RequestId(2));
        // Request 1 demoted, request 2 confirmed — two lookups, not a re-scan.
        assert_eq!(after.lookups, 2);
    }

    #[test]
    fn equal_matches_fall_back_to_arrival_order() {
        let mut policy = LongestPrefixMatch::new();
        policy.push(candidate(7, 5, 10, 30_000, 30_000), &mut ());
        policy.push(candidate(3, 2, 20, 30_000, 30_000), &mut ());

        let mut cache = resident(&[(10, 30_000), (20, 30_000)]);
        policy.refresh_head(&mut |c| cache.score(c));

        assert_eq!(policy.peek().expect("a head").request_id, RequestId(3));
    }

    #[test]
    fn removing_a_queued_request_releases_its_queued_kv_tokens() {
        let mut policy = LongestPrefixMatch::new();
        let first = candidate(1, 0, 10, 90_000, 90_000);
        policy.push(first, &mut ());
        policy.push(candidate(2, 1, 20, 50_000, 50_000), &mut ());
        let total = policy.queued_kv_tokens();

        let removed = policy.remove(RequestId(1)).expect("request 1 was queued");

        assert_eq!(removed.request_id, RequestId(1));
        assert!(!policy.contains(RequestId(1)));
        assert_eq!(policy.len(), 1);
        assert_eq!(policy.queued_kv_tokens(), total - first.queued_kv_tokens());
    }
}
