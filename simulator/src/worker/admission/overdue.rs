//! Force-schedule a request that has waited too long (not vLLM).
//!
//! A shortest-first order (SPF, srpt) can leave a long prompt waiting for as
//! long as shorter work keeps arriving: under a saturated closed loop some
//! waited hours. [`OverdueQueue`] bounds that. Once a queued request has waited
//! `after` since its arrival, admission takes it out of the pending order into
//! the partition's overdue queue, which admission serves, oldest arrival first,
//! before the pending order and with no shortest-first bound. A pipeline head
//! or chunked-prefill DP rank with srpt also runs overdue started prompts
//! before the others (`srpt.rs`).

use std::collections::{HashSet, VecDeque};

use crate::common::{RequestId, Time};

use super::{AdmissionCandidate, PendingOrderPolicy};

pub(crate) struct OverdueQueue {
    after: Time,
    /// Per partition: queued requests in arrival order. A request that left the
    /// pending order meanwhile is skipped when it reaches the front.
    arrivals: Vec<VecDeque<(Time, RequestId)>>,
    /// Per partition: requests taken out of the pending order, oldest first.
    overdue: Vec<VecDeque<AdmissionCandidate>>,
    /// Per partition: prefill tokens the overdue requests carry, before any
    /// prefix hit.
    overdue_prompt_tokens: Vec<u64>,
    /// Per partition, with [`Self::with_hits_first`]: overdue hits, oldest
    /// first; `overdue` then holds the other overdue requests.
    overdue_hits: Vec<VecDeque<AdmissionCandidate>>,
    hits_first: bool,
    /// With [`Self::with_compute_tracking`]: the requests in `overdue`, and
    /// per partition the prefill tokens they compute (prompt less the prefix
    /// held when they became overdue).
    tracked: Option<HashSet<RequestId>>,
    overdue_compute_tokens: Vec<u64>,
}

impl OverdueQueue {
    pub(crate) fn new(after_ms: f64) -> Self {
        assert!(after_ms > 0.0, "force_schedule_after_ms must be positive");
        Self {
            after: Time::from_ms(after_ms),
            arrivals: Vec::new(),
            overdue: Vec::new(),
            overdue_prompt_tokens: Vec::new(),
            overdue_hits: Vec::new(),
            hits_first: false,
            tracked: None,
            overdue_compute_tokens: Vec::new(),
        }
    }

    /// Track which requests wait in the oldest-first overdue queue and the
    /// prefill tokens they will compute ([`Self::compute_tokens`]).
    pub(crate) fn with_compute_tracking(mut self) -> Self {
        self.tracked = Some(HashSet::new());
        self
    }

    /// Prefill tokens `partition`'s oldest-first overdue requests compute.
    pub(crate) fn compute_tokens(&self, partition: u16) -> u64 {
        self.overdue_compute_tokens
            .get(usize::from(partition))
            .copied()
            .unwrap_or(0)
    }

    fn track_in(&mut self, partition: usize, candidate: &AdmissionCandidate) {
        if let Some(tracked) = self.tracked.as_mut() {
            tracked.insert(candidate.request_id);
            self.overdue_compute_tokens[partition] += u64::from(
                candidate
                    .fresh_prompt_tokens
                    .saturating_sub(candidate.resident_prefix_tokens),
            );
        }
    }

    fn track_out(&mut self, partition: usize, candidate: &AdmissionCandidate) {
        if let Some(tracked) = self.tracked.as_mut() {
            tracked.remove(&candidate.request_id);
            self.overdue_compute_tokens[partition] -= u64::from(
                candidate
                    .fresh_prompt_tokens
                    .saturating_sub(candidate.resident_prefix_tokens),
            );
        }
    }

    /// Whether `request` waits in the oldest-first overdue queue (tracked).
    pub(crate) fn is_tracked(&self, request: RequestId) -> bool {
        self.tracked
            .as_ref()
            .is_some_and(|tracked| tracked.contains(&request))
    }

    /// Split the overdue requests by cost class: a request whose session's
    /// context is held (HBM or a tier), so it computes only its new tokens,
    /// is a hit and goes before every request that computes its context
    /// (`srpt.rs`, [`Self::hits_first`]). Not vLLM.
    pub(crate) fn with_hits_first(mut self) -> Self {
        self.hits_first = true;
        self
    }

    pub(crate) fn hits_first(&self) -> bool {
        self.hits_first
    }

    /// Turn the hits-first split on or off from now on; requests already in
    /// the hits queue keep going first.
    pub(crate) fn set_hits_first(&mut self, on: bool) {
        self.hits_first = on;
    }

    fn ensure(&mut self, partition: usize) {
        if self.arrivals.len() <= partition {
            self.arrivals.resize_with(partition + 1, VecDeque::new);
            self.overdue.resize_with(partition + 1, VecDeque::new);
            self.overdue_hits.resize_with(partition + 1, VecDeque::new);
            self.overdue_prompt_tokens.resize(partition + 1, 0);
            self.overdue_compute_tokens.resize(partition + 1, 0);
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
        self.promote_classified(partition, policy, now, |_| 0);
    }

    /// [`Self::promote`], given the prefix tokens `held` finds in HBM or a
    /// tier for a request: with hits first, a request with some goes in the
    /// hits queue; with compute tracking, they are what it does not compute.
    pub(crate) fn promote_classified<P: PendingOrderPolicy>(
        &mut self,
        partition: u16,
        policy: &mut P,
        now: Time,
        mut held: impl FnMut(&AdmissionCandidate) -> u32,
    ) {
        let partition = usize::from(partition);
        self.ensure(partition);
        while let Some(&(arrival, request)) = self.arrivals[partition].front() {
            if !self.is_overdue(arrival, now) {
                break;
            }
            self.arrivals[partition].pop_front();
            if let Some(mut candidate) = policy.remove(request) {
                self.overdue_prompt_tokens[partition] += u64::from(candidate.fresh_prompt_tokens);
                if self.tracked.is_some() {
                    candidate.resident_prefix_tokens = held(&candidate);
                }
                if self.hits_first && held(&candidate) > 0 {
                    self.overdue_hits[partition].push_back(candidate);
                } else {
                    self.track_in(partition, &candidate);
                    self.overdue[partition].push_back(candidate);
                }
            }
        }
    }

    /// The next overdue request, hits first, and whether it is a hit;
    /// `hits_only` offers hits alone.
    pub(crate) fn peek(
        &self,
        partition: u16,
        hits_only: bool,
    ) -> Option<(AdmissionCandidate, bool)> {
        let partition = usize::from(partition);
        if let Some(&candidate) = self.overdue_hits.get(partition).and_then(VecDeque::front) {
            return Some((candidate, true));
        }
        if hits_only {
            return None;
        }
        self.overdue
            .get(partition)
            .and_then(|queue| queue.front().copied())
            .map(|candidate| (candidate, false))
    }

    /// Take the request [`Self::peek`] offered.
    pub(crate) fn pop(&mut self, partition: u16) -> Option<AdmissionCandidate> {
        let index = usize::from(partition);
        let hits = self
            .overdue_hits
            .get_mut(index)
            .filter(|queue| !queue.is_empty());
        let candidate = match hits {
            Some(queue) => queue.pop_front()?,
            None => {
                let candidate = self.overdue.get_mut(index)?.pop_front()?;
                self.track_out(index, &candidate);
                candidate
            }
        };
        self.overdue_prompt_tokens[usize::from(partition)] -=
            u64::from(candidate.fresh_prompt_tokens);
        Some(candidate)
    }

    /// Put a request [`Self::pop`] took back at the front of its queue.
    pub(crate) fn push_front(&mut self, partition: u16, candidate: AdmissionCandidate, hit: bool) {
        let partition = usize::from(partition);
        self.ensure(partition);
        self.overdue_prompt_tokens[partition] += u64::from(candidate.fresh_prompt_tokens);
        if hit {
            self.overdue_hits[partition].push_front(candidate);
        } else {
            self.track_in(partition, &candidate);
            self.overdue[partition].push_front(candidate);
        }
    }

    /// Drop `request` from the overdue queue, if it is there.
    pub(crate) fn remove(&mut self, request: RequestId) -> bool {
        for hits in [false, true] {
            for partition in 0..self.overdue.len() {
                let queue = if hits {
                    &mut self.overdue_hits[partition]
                } else {
                    &mut self.overdue[partition]
                };
                if let Some(index) = queue.iter().position(|c| c.request_id == request) {
                    let candidate = queue.remove(index).expect("index is in range");
                    self.overdue_prompt_tokens[partition] -=
                        u64::from(candidate.fresh_prompt_tokens);
                    if !hits {
                        self.track_out(partition, &candidate);
                    }
                    return true;
                }
            }
        }
        false
    }

    pub(crate) fn len(&self) -> usize {
        self.overdue
            .iter()
            .chain(&self.overdue_hits)
            .map(VecDeque::len)
            .sum()
    }

    pub(crate) fn prompt_tokens(&self) -> u64 {
        self.overdue_prompt_tokens.iter().sum()
    }

    /// Prefill tokens `partition`'s overdue requests carry.
    pub(crate) fn partition_prompt_tokens(&self, partition: u16) -> u64 {
        self.overdue_prompt_tokens
            .get(usize::from(partition))
            .copied()
            .unwrap_or(0)
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

    fn overdue_with(hits_first: bool) -> (OverdueQueue, FifoOrder) {
        let mut sequence = EnqueueSequence::default();
        let mut policy = FifoOrder::default();
        let mut overdue = OverdueQueue::new(10.0);
        if hits_first {
            overdue = overdue.with_hits_first();
        }
        for id in 0..4 {
            let candidate = sequence.freeze(
                RequestId(id),
                100,
                1,
                SessionInput::Standalone,
                Time::ZERO,
                0,
            );
            policy.push(candidate, &mut ());
            overdue.arrived(0, RequestId(id), Time::from_ms(f64::from(id)));
        }
        // Odd requests are hits.
        overdue.promote_classified(0, &mut policy, Time::from_ms(20.0), |c| c.request_id.0 % 2);
        (overdue, policy)
    }

    #[test]
    fn hits_first_serves_overdue_hits_before_misses_each_oldest_first() {
        let (mut overdue, _) = overdue_with(true);
        assert_eq!(
            overdue.peek(0, true).map(|(c, hit)| (c.request_id, hit)),
            Some((RequestId(1), true))
        );
        let order: Vec<_> = std::iter::from_fn(|| overdue.pop(0).map(|c| c.request_id.0)).collect();
        assert_eq!(order, [1, 3, 0, 2]);
        assert_eq!(overdue.prompt_tokens(), 0);
    }

    #[test]
    fn hits_only_peek_offers_nothing_once_hits_are_gone_and_push_front_restores_class() {
        let (mut overdue, _) = overdue_with(true);
        let first = overdue.pop(0).unwrap();
        let second = overdue.pop(0).unwrap();
        assert_eq!(overdue.peek(0, true), None);
        assert_eq!(
            overdue.peek(0, false).map(|(c, hit)| (c.request_id, hit)),
            Some((RequestId(0), false))
        );
        overdue.push_front(0, second, true);
        overdue.push_front(0, first, true);
        assert_eq!(
            overdue.peek(0, true).map(|(c, _)| c.request_id),
            Some(RequestId(1))
        );
        assert!(overdue.remove(RequestId(3)));
        assert!(overdue.remove(RequestId(0)));
        assert_eq!((overdue.len(), overdue.prompt_tokens()), (2, 200));
    }

    #[test]
    fn without_hits_first_the_classifier_is_ignored() {
        let (mut overdue, _) = overdue_with(false);
        assert!(!overdue.hits_first());
        assert_eq!(overdue.peek(0, true), None);
        let order: Vec<_> = std::iter::from_fn(|| overdue.pop(0).map(|c| c.request_id.0)).collect();
        assert_eq!(order, [0, 1, 2, 3]);
    }

    #[test]
    fn compute_tracking_counts_what_oldest_first_requests_compute() {
        let mut sequence = EnqueueSequence::default();
        let mut policy = FifoOrder::default();
        let mut overdue = OverdueQueue::new(10.0).with_compute_tracking();
        for id in 0..3 {
            let candidate = sequence.freeze(
                RequestId(id),
                100,
                1,
                SessionInput::Standalone,
                Time::ZERO,
                0,
            );
            policy.push(candidate, &mut ());
            overdue.arrived(0, RequestId(id), Time::ZERO);
        }
        // Request 1 holds 80 of its 100 prompt tokens.
        overdue.promote_classified(0, &mut policy, Time::from_ms(10.0), |c| {
            if c.request_id == RequestId(1) {
                80
            } else {
                0
            }
        });
        assert_eq!(overdue.compute_tokens(0), 220);
        assert!(overdue.is_tracked(RequestId(2)));
        assert!(overdue.remove(RequestId(2)));
        assert!(!overdue.is_tracked(RequestId(2)));
        assert_eq!(overdue.compute_tokens(0), 120);
        let first = overdue.pop(0).unwrap();
        assert_eq!(overdue.compute_tokens(0), 20);
        overdue.push_front(0, first, false);
        assert_eq!(overdue.compute_tokens(0), 120);
        let order: Vec<_> = std::iter::from_fn(|| overdue.pop(0).map(|c| c.request_id.0)).collect();
        assert_eq!(order, [0, 1]);
        assert_eq!((overdue.compute_tokens(0), overdue.prompt_tokens()), (0, 0));
    }
}
