//! Force-schedule bound over any pending order (not vLLM).
//!
//! A shortest-first order can leave a long prompt waiting for as long as
//! shorter work keeps arriving. [`Deadline`] bounds that: once a queued request
//! has waited `after` since its arrival, [`Deadline::promote`] takes it out of
//! the inner order into an overdue queue, and the wrapper offers overdue
//! requests, oldest arrival first, before the inner order's head.
//!
//! No policy sees time, so the lifecycle calls `promote(now)` at the start of
//! each formation; `PendingOrderPolicy` itself is unchanged. Without a bound the
//! wrapper forwards every call to the inner order.
//!
//! Only requests entered through [`Deadline::push_arrived`] carry an arrival and
//! can become overdue; a request pushed back through the trait (a retraction)
//! joins the inner order untracked. Arrival entries of requests that left the
//! queue meanwhile are skipped when they reach the front.

use std::collections::VecDeque;

use crate::common::{RequestId, Time};

use super::{AdmissionCandidate, PendingOrderPolicy};

pub struct Deadline<P> {
    inner: P,
    /// How long a request may wait before it is served first; `None`: no bound.
    after: Option<Time>,
    /// Requests entered with an arrival, in arrival order.
    arrivals: VecDeque<(Time, RequestId)>,
    /// Requests taken out of `inner`, oldest arrival first.
    overdue: VecDeque<AdmissionCandidate>,
    overdue_kv_tokens: u64,
    overdue_prompt_tokens: u64,
}

impl<P> Deadline<P> {
    /// `inner` with no bound.
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            after: None,
            arrivals: VecDeque::new(),
            overdue: VecDeque::new(),
            overdue_kv_tokens: 0,
            overdue_prompt_tokens: 0,
        }
    }

    /// Serve a request that has waited `after` since arrival before `inner`'s head.
    pub fn with_bound(mut self, after: Time) -> Self {
        assert!(after > Time::ZERO, "force-schedule bound must be positive");
        self.after = Some(after);
        self
    }

    /// Whether a request that arrived at `arrival` has waited past the bound.
    pub fn is_overdue(&self, arrival: Time, now: Time) -> bool {
        self.after
            .is_some_and(|after| now.0.saturating_sub(arrival.0) >= after.0)
    }

    /// Whether [`PendingOrderPolicy::peek`] offers an overdue request.
    pub fn head_is_overdue(&self) -> bool {
        !self.overdue.is_empty()
    }
}

impl<P: PendingOrderPolicy> Deadline<P> {
    /// Enqueue a fresh request that arrived at `arrival`. Arrivals reach a
    /// worker in time order.
    pub fn push_arrived(
        &mut self,
        candidate: AdmissionCandidate,
        arrival: Time,
        context: &mut P::Context,
    ) {
        if self.after.is_some() {
            debug_assert!(self
                .arrivals
                .back()
                .is_none_or(|&(last, _)| last <= arrival));
            self.arrivals.push_back((arrival, candidate.request_id));
        }
        self.inner.push(candidate, context);
    }

    /// Move every request that has waited past the bound by `now` out of the
    /// inner order into the overdue queue.
    pub fn promote(&mut self, now: Time) {
        while let Some(&(arrival, request)) = self.arrivals.front() {
            if !self.is_overdue(arrival, now) {
                break;
            }
            self.arrivals.pop_front();
            if let Some(candidate) = self.inner.remove(request) {
                self.overdue_kv_tokens += candidate.queued_kv_tokens();
                self.overdue_prompt_tokens += u64::from(candidate.fresh_prompt_tokens);
                self.overdue.push_back(candidate);
            }
        }
    }

    fn take_overdue(&mut self, index: usize) -> AdmissionCandidate {
        let candidate = self.overdue.remove(index).expect("index is in range");
        self.overdue_kv_tokens -= candidate.queued_kv_tokens();
        self.overdue_prompt_tokens -= u64::from(candidate.fresh_prompt_tokens);
        candidate
    }
}

impl<P: PendingOrderPolicy> PendingOrderPolicy for Deadline<P> {
    type Context = P::Context;

    fn push(&mut self, candidate: AdmissionCandidate, context: &mut Self::Context) {
        self.inner.push(candidate, context);
    }

    fn push_front(&mut self, candidate: AdmissionCandidate, context: &mut Self::Context) {
        self.inner.push_front(candidate, context);
    }

    /// The inner head is read only when no request is overdue.
    fn refresh_head(&mut self, resident_prefix_tokens: &mut dyn FnMut(AdmissionCandidate) -> u32) {
        if self.overdue.is_empty() {
            self.inner.refresh_head(resident_prefix_tokens);
        }
    }

    fn peek(&self) -> Option<AdmissionCandidate> {
        match self.overdue.front() {
            Some(&candidate) => Some(candidate),
            None => self.inner.peek(),
        }
    }

    fn pop(&mut self, context: &mut Self::Context) -> Option<AdmissionCandidate> {
        if self.overdue.is_empty() {
            self.inner.pop(context)
        } else {
            Some(self.take_overdue(0))
        }
    }

    fn remove(&mut self, request: RequestId) -> Option<AdmissionCandidate> {
        match self
            .overdue
            .iter()
            .position(|candidate| candidate.request_id == request)
        {
            Some(index) => Some(self.take_overdue(index)),
            None => self.inner.remove(request),
        }
    }

    fn contains(&self, request: RequestId) -> bool {
        self.overdue
            .iter()
            .any(|candidate| candidate.request_id == request)
            || self.inner.contains(request)
    }

    fn len(&self) -> usize {
        self.overdue.len() + self.inner.len()
    }

    fn queued_kv_tokens(&self) -> u64 {
        self.overdue_kv_tokens + self.inner.queued_kv_tokens()
    }

    fn queued_prompt_tokens(&self) -> u64 {
        self.overdue_prompt_tokens + self.inner.queued_prompt_tokens()
    }
}

#[cfg(test)]
mod tests {
    use super::super::{EnqueueSequence, FifoOrder, JobSize, ShortestJobFirst};
    use super::*;
    use crate::common::SessionInput;

    fn enqueue<P: PendingOrderPolicy<Context = ()>>(
        deadline: &mut Deadline<P>,
        sequence: &mut EnqueueSequence,
        requests: &[(u32, u32, f64)],
    ) {
        for &(id, prompt_tokens, arrival_ms) in requests {
            let candidate = sequence.freeze(
                RequestId(id),
                prompt_tokens,
                1,
                SessionInput::Standalone,
                Time::ZERO,
                0,
            );
            deadline.push_arrived(candidate, Time::from_ms(arrival_ms), &mut ());
        }
    }

    #[test]
    fn promote_moves_only_requests_past_the_bound_oldest_first() {
        let mut sequence = EnqueueSequence::default();
        let mut deadline = Deadline::new(FifoOrder::new()).with_bound(Time::from_ms(10.0));
        enqueue(
            &mut deadline,
            &mut sequence,
            &[(0, 100, 0.0), (1, 100, 4.0), (2, 100, 8.0)],
        );
        // Request 0 was admitted the normal way before it became overdue.
        deadline.pop(&mut ());
        deadline.promote(Time::from_ms(15.0));
        assert!(deadline.head_is_overdue());
        assert_eq!(
            deadline.pop(&mut ()).map(|c| c.request_id),
            Some(RequestId(1))
        );
        assert!(!deadline.head_is_overdue(), "request 2 has waited 7 ms");
        deadline.promote(Time::from_ms(18.0));
        assert!(deadline.head_is_overdue());
        assert_eq!(deadline.queued_prompt_tokens(), 100);
        assert_eq!(
            deadline.remove(RequestId(2)).map(|c| c.request_id),
            Some(RequestId(2))
        );
        assert_eq!(
            (
                deadline.len(),
                deadline.queued_prompt_tokens(),
                deadline.queued_kv_tokens()
            ),
            (0, 0, 0)
        );
    }

    #[test]
    fn an_overdue_request_goes_before_a_shorter_one_that_arrived_later() {
        let mut sequence = EnqueueSequence::default();
        let mut deadline = Deadline::new(ShortestJobFirst::by(JobSize::PrefillTokens))
            .with_bound(Time::from_ms(10.0));
        enqueue(
            &mut deadline,
            &mut sequence,
            &[(0, 5_000, 0.0), (1, 100, 1.0), (2, 200, 2.0)],
        );
        deadline.promote(Time::from_ms(5.0));
        assert_eq!(deadline.peek().map(|c| c.request_id), Some(RequestId(1)));
        deadline.promote(Time::from_ms(10.0));
        let order: Vec<_> = std::iter::from_fn(|| deadline.pop(&mut ()))
            .map(|c| c.request_id.0)
            .collect();
        assert_eq!(order, [0, 1, 2]);
    }

    #[test]
    fn without_a_bound_nothing_becomes_overdue() {
        let mut sequence = EnqueueSequence::default();
        let mut deadline = Deadline::new(ShortestJobFirst::by(JobSize::PrefillTokens));
        enqueue(
            &mut deadline,
            &mut sequence,
            &[(0, 5_000, 0.0), (1, 100, 1.0)],
        );
        deadline.promote(Time::from_ms(1e9));
        assert!(!deadline.is_overdue(Time::ZERO, Time::from_ms(1e9)));
        assert_eq!(deadline.peek().map(|c| c.request_id), Some(RequestId(1)));
    }
}
