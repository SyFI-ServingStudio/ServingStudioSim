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
//! Ranked mode ([`Deadline::with_rank`], not vLLM) replaces both orders with a
//! [`PrefillRank`] (see `prefill_rank`): queued requests sit in ordered maps
//! under keys that do not change with time, `promote(now)` moves only the
//! requests that turned late since the last call, and the head is the better of
//! two map heads, so every operation is O(log n) in the queue length. Nothing
//! is overdue in this mode; the head is offered as the overdue (miss) head.
//!
//! Only requests entered through [`Deadline::push_arrived`] carry an arrival and
//! can become overdue; a request pushed back through the trait (a retraction)
//! joins the inner order untracked. Arrival entries of requests that left the
//! queue meanwhile are skipped when they reach the front.
//!
//! [`HitsFirst`] splits the overdue requests by cost class. A request whose
//! session's context is held (HBM or a tier), the `held` count `promote` is
//! given, computes only its new tokens: a hit. Hits wait in their own queue,
//! offered before the other overdue requests, each class oldest first. The
//! class is fixed at promotion. A lifecycle that splits its started prompts the
//! same way reads [`Deadline::splits_by_class`] and takes each class on its own
//! through [`Deadline::peek_head`] (`QueueAccess`). Under
//! [`HitsFirst::WhenAtRisk`] the split is off until the lifecycle turns it on
//! (`set_hits_first`): it does while serving the overdue misses oldest first
//! would evict a queued request's held context, a KV fact the lifecycle asks the
//! store for. For that the wrapper keeps the prefill tokens the overdue misses
//! compute ([`Deadline::at_risk_horizon`]) and each session's latest queued
//! request ([`Deadline::queued_in_session`]).

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use crate::common::{RequestId, SessionInput, Time};

use crate::worker::admission::prefill_rank::{PrefillRank, Rank};

use super::{AdmissionCandidate, PendingOrderPolicy};

/// How overdue requests are ordered by cost class. Not vLLM.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum HitsFirst {
    /// Oldest arrival first.
    #[default]
    Off,
    /// Hits first, then the other overdue requests, each oldest first.
    Always,
    /// `Always` only while the lifecycle finds a held context at risk.
    WhenAtRisk,
}

/// The force-schedule bound and the class order of the requests past it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ForceSchedule {
    pub after: Time,
    pub hits_first: HitsFirst,
}

/// Which queued requests [`Deadline::peek_head`] may offer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueAccess {
    /// Overdue requests (hits first), then the inner order.
    All,
    /// Overdue requests alone, hits first.
    Overdue,
    /// Overdue hits alone.
    OverdueHits,
    /// The inner order alone.
    Pending,
    /// The inner order's hits alone (frozen `resident_prefix_tokens > 0` after
    /// the head refresh); the misses it passes are set aside, keeping their
    /// places once [`Deadline::restore_set_aside`] puts them back.
    PendingHits,
}

impl QueueAccess {
    /// Whether the access reaches beyond the overdue queue.
    pub fn takes_pending(self) -> bool {
        !matches!(self, Self::Overdue | Self::OverdueHits)
    }
}

/// Which queue [`Deadline::peek_head`] found its candidate in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Queued {
    /// The overdue queue; `hit`: its hits queue.
    Overdue { hit: bool },
    /// The inner order.
    Pending,
}

/// Map key: (time-free key, enqueue sequence).
type Slot = (u64, u64);

/// A queued request in ranked mode.
struct RankedItem {
    candidate: AdmissionCandidate,
    arrival: Time,
    size: u32,
    /// Its slots in `Ranked::waiting` and `Ranked::onset` until it turns
    /// late, then in `Ranked::late`.
    waiting: Option<Slot>,
    onset: Option<Slot>,
    late: Option<Slot>,
}

/// Ranked-mode state.
struct Ranked {
    rank: PrefillRank,
    /// Arrival and prefill size of every request entered with an arrival; read
    /// for started prompts too, so entries outlive the queue.
    entries: HashMap<RequestId, (Time, u32)>,
    /// The queue, by request.
    queued: HashMap<RequestId, RankedItem>,
    /// Not (yet) late, by size.
    waiting: BTreeMap<Slot, RequestId>,
    /// `waiting`'s requests by the instant they turn late.
    onset: BTreeMap<Slot, RequestId>,
    /// Late requests by deadline.
    late: BTreeMap<Slot, RequestId>,
    /// The last `promote`'s time, at which the late head is ranked.
    now: Time,
}

fn ns(time: Time) -> u64 {
    time.as_ns()
}

impl Ranked {
    fn new(rank: PrefillRank) -> Self {
        Self {
            rank,
            entries: HashMap::new(),
            queued: HashMap::new(),
            waiting: BTreeMap::new(),
            onset: BTreeMap::new(),
            late: BTreeMap::new(),
            now: Time::ZERO,
        }
    }

    fn rank_of(&self, item: &RankedItem) -> Rank {
        self.rank.rank(item.size, item.arrival, item.size, self.now)
    }

    /// The head of the queue and its rank: the smaller index key of the two
    /// tier heads (the late one on a tie).
    fn head(&self) -> Option<(RequestId, Rank)> {
        let of =
            |(_, &request): (&Slot, &RequestId)| (request, self.rank_of(&self.queued[&request]));
        let waiting = self.waiting.first_key_value().map(of);
        let late = self.late.first_key_value().map(of);
        match (waiting, late) {
            (Some(w), Some(l)) => Some(if w.1.key < l.1.key { w } else { l }),
            (w, l) => l.or(w),
        }
    }

    fn insert(&mut self, candidate: AdmissionCandidate) {
        let size = prefill_size(&candidate);
        let (arrival, size) = *self
            .entries
            .entry(candidate.request_id)
            .or_insert((self.now, size));
        let request = candidate.request_id;
        let seq = candidate.enqueue_sequence;
        let mut item = RankedItem {
            candidate,
            arrival,
            size,
            waiting: None,
            onset: None,
            late: None,
        };
        let PrefillRank::Square(cost) = self.rank;
        if cost.late_at(size, arrival) <= self.now {
            let slot = (ns(cost.due(size, arrival)), seq);
            item.late = Some(slot);
            self.late.insert(slot, request);
        } else {
            let waiting = (u64::from(size), seq);
            let onset = (ns(cost.late_at(size, arrival)), seq);
            item.waiting = Some(waiting);
            item.onset = Some(onset);
            self.waiting.insert(waiting, request);
            self.onset.insert(onset, request);
        }
        self.queued.insert(request, item);
    }

    fn take(&mut self, request: RequestId) -> Option<AdmissionCandidate> {
        let item = self.queued.remove(&request)?;
        if let Some(slot) = item.waiting {
            self.waiting.remove(&slot);
        }
        if let Some(slot) = item.onset {
            self.onset.remove(&slot);
        }
        if let Some(slot) = item.late {
            self.late.remove(&slot);
        }
        Some(item.candidate)
    }

    /// Move the requests that turned late by `now` to the late tier.
    fn advance(&mut self, now: Time) {
        self.now = self.now.max(now);
        let PrefillRank::Square(cost) = self.rank;
        while let Some((&slot, &request)) = self.onset.first_key_value() {
            if slot.0 > ns(self.now) {
                break;
            }
            self.onset.remove(&slot);
            let item = self
                .queued
                .get_mut(&request)
                .expect("onset entries are queued");
            item.onset = None;
            if let Some(waiting) = item.waiting.take() {
                self.waiting.remove(&waiting);
            }
            let late = (ns(cost.due(item.size, item.arrival)), slot.1);
            item.late = Some(late);
            self.late.insert(late, request);
        }
    }
}

/// Prefill tokens to compute for a queued candidate (as `JobSize::PrefillTokens`).
fn prefill_size(candidate: &AdmissionCandidate) -> u32 {
    let declared = candidate.session_input.declared_prefix_tokens();
    candidate
        .fresh_prompt_tokens
        .saturating_add(declared.saturating_sub(candidate.resident_prefix_tokens))
}

pub struct Deadline<P> {
    inner: P,
    /// How long a request may wait before it is served first; `None`: no bound.
    after: Option<Time>,
    /// Requests entered with an arrival, in arrival order.
    arrivals: VecDeque<(Time, RequestId)>,
    /// Requests taken out of `inner`, oldest arrival first: every overdue
    /// request, or with the class split the misses.
    overdue: VecDeque<AdmissionCandidate>,
    /// Overdue hits, oldest arrival first.
    overdue_hits: VecDeque<AdmissionCandidate>,
    overdue_kv_tokens: u64,
    overdue_prompt_tokens: u64,
    hits_first: HitsFirst,
    /// Whether promotion splits by class now.
    split: bool,
    /// Under [`HitsFirst::WhenAtRisk`].
    at_risk: Option<AtRiskIndex>,
    /// Inner requests a [`QueueAccess::PendingHits`] pass went past; empty
    /// between passes.
    set_aside: Vec<AdmissionCandidate>,
    /// Ranked mode: the queue, in place of `inner` and the bound.
    ranked: Option<Ranked>,
}

/// What the at-risk test reads from the queue.
#[derive(Default)]
struct AtRiskIndex {
    /// The requests in `overdue`.
    misses: HashSet<RequestId>,
    /// Prefill tokens they compute.
    compute_tokens: u64,
    /// Each session's latest request entered with an arrival.
    latest: HashMap<u32, AdmissionCandidate>,
}

impl AtRiskIndex {
    /// Prefill tokens an overdue miss computes: its fresh tokens plus the
    /// part of its declared prefix not held at promotion.
    fn compute_tokens_of(candidate: &AdmissionCandidate) -> u64 {
        u64::from(candidate.fresh_prompt_tokens)
            + u64::from(
                candidate
                    .session_input
                    .declared_prefix_tokens()
                    .saturating_sub(candidate.resident_prefix_tokens),
            )
    }

    fn track_in(&mut self, candidate: &AdmissionCandidate) {
        self.misses.insert(candidate.request_id);
        self.compute_tokens += Self::compute_tokens_of(candidate);
    }

    fn track_out(&mut self, candidate: &AdmissionCandidate) {
        self.misses.remove(&candidate.request_id);
        self.compute_tokens -= Self::compute_tokens_of(candidate);
    }
}

impl<P> Deadline<P> {
    /// `inner` with no bound.
    pub fn new(inner: P) -> Self {
        Self {
            inner,
            after: None,
            arrivals: VecDeque::new(),
            overdue: VecDeque::new(),
            overdue_hits: VecDeque::new(),
            overdue_kv_tokens: 0,
            overdue_prompt_tokens: 0,
            hits_first: HitsFirst::Off,
            split: false,
            at_risk: None,
            set_aside: Vec::new(),
            ranked: None,
        }
    }

    /// Rank every request by `rank` instead of `inner`'s order and the bound.
    pub fn with_rank(mut self, rank: PrefillRank) -> Self {
        self.ranked = Some(Ranked::new(rank));
        self
    }

    /// Whether the rank orders the queue.
    pub fn rank_active(&self) -> bool {
        self.ranked.is_some()
    }

    /// The rank of request `request` with `remaining` prefill tokens at `now`;
    /// `None` outside ranked mode or for a request entered without an arrival.
    pub fn rank_of(&self, request: RequestId, remaining: u32, now: Time) -> Option<Rank> {
        let ranked = self.ranked.as_ref()?;
        let &(arrival, size) = ranked.entries.get(&request)?;
        Some(ranked.rank.rank(size, arrival, remaining, now))
    }

    /// Ranked mode: the rank of [`PendingOrderPolicy::peek`]'s request as of
    /// the last `promote`.
    pub fn head_rank(&self) -> Option<Rank> {
        self.ranked.as_ref()?.head().map(|(_, rank)| rank)
    }

    /// Ranked SquareCost mode: whether a queued request is in the late tier
    /// as of the last `promote` (O(1)). Started prompts have left the tier.
    pub fn has_late(&self) -> bool {
        self.ranked
            .as_ref()
            .is_some_and(|ranked| !ranked.late.is_empty())
    }

    /// Ranked mode with the pad rule on.
    pub fn pads(&self) -> bool {
        self.ranked
            .as_ref()
            .is_some_and(|ranked| ranked.rank.pads())
    }

    /// Serve a request that has waited `after` since arrival before `inner`'s head.
    pub fn with_bound(mut self, after: Time) -> Self {
        assert!(after > Time::ZERO, "force-schedule bound must be positive");
        self.after = Some(after);
        self
    }

    /// [`Self::with_bound`] and [`Self::with_hits_first`] of `schedule`.
    pub fn with_force_schedule(self, schedule: ForceSchedule) -> Self {
        self.with_bound(schedule.after)
            .with_hits_first(schedule.hits_first)
    }

    /// Split the overdue requests by cost class as `hits_first` says.
    pub fn with_hits_first(mut self, hits_first: HitsFirst) -> Self {
        assert!(
            hits_first == HitsFirst::Off || self.after.is_some(),
            "hits first needs the force-schedule bound"
        );
        self.hits_first = hits_first;
        self.split = hits_first == HitsFirst::Always;
        self.at_risk = (hits_first == HitsFirst::WhenAtRisk).then(AtRiskIndex::default);
        self
    }

    /// Whether a request that arrived at `arrival` has waited past the bound.
    pub fn is_overdue(&self, arrival: Time, now: Time) -> bool {
        self.after
            .is_some_and(|after| now.0.saturating_sub(arrival.0) >= after.0)
    }

    /// Whether [`PendingOrderPolicy::peek`] offers an overdue request.
    pub fn head_is_overdue(&self) -> bool {
        !self.rank_active() && !(self.overdue.is_empty() && self.overdue_hits.is_empty())
    }

    /// Whether requests are now taken in two cost classes.
    pub fn splits_by_class(&self) -> bool {
        self.split
    }

    /// Under [`HitsFirst::WhenAtRisk`]: turn the split on or off from now on;
    /// requests already in the hits queue keep going first.
    pub fn set_hits_first(&mut self, on: bool) {
        debug_assert_eq!(self.hits_first, HitsFirst::WhenAtRisk);
        self.split = on;
    }

    /// Under [`HitsFirst::WhenAtRisk`]: the prefill tokens the overdue misses
    /// compute, oldest first ahead of everything else.
    pub fn at_risk_horizon(&self) -> Option<u64> {
        self.at_risk.as_ref().map(|index| index.compute_tokens)
    }

    /// The overdue head, hits first; `hits_only` offers hits alone.
    fn overdue_head(&self, hits_only: bool) -> Option<(AdmissionCandidate, bool)> {
        if let Some(ranked) = &self.ranked {
            let head = ranked.head().filter(|_| !hits_only)?;
            return Some((ranked.queued[&head.0].candidate, false));
        }
        if let Some(&candidate) = self.overdue_hits.front() {
            return Some((candidate, true));
        }
        if hits_only {
            return None;
        }
        self.overdue.front().map(|&candidate| (candidate, false))
    }

    fn push_back_overdue(&mut self, candidate: AdmissionCandidate, hit: bool) {
        self.overdue_kv_tokens += candidate.queued_kv_tokens();
        self.overdue_prompt_tokens += u64::from(candidate.fresh_prompt_tokens);
        if hit {
            self.overdue_hits.push_back(candidate);
        } else {
            if let Some(index) = self.at_risk.as_mut() {
                index.track_in(&candidate);
            }
            self.overdue.push_back(candidate);
        }
    }

    /// Ranked mode: enqueue `candidate`.
    fn push_ranked(&mut self, candidate: AdmissionCandidate) {
        self.overdue_kv_tokens += candidate.queued_kv_tokens();
        self.overdue_prompt_tokens += u64::from(candidate.fresh_prompt_tokens);
        self.ranked.as_mut().expect("ranked mode").insert(candidate);
    }

    /// Ranked mode: dequeue `request`.
    fn take_ranked(&mut self, request: RequestId) -> Option<AdmissionCandidate> {
        let candidate = self.ranked.as_mut().expect("ranked mode").take(request)?;
        self.overdue_kv_tokens -= candidate.queued_kv_tokens();
        self.overdue_prompt_tokens -= u64::from(candidate.fresh_prompt_tokens);
        Some(candidate)
    }

    /// Put an overdue request taken out this pass back at the front of its
    /// class's queue.
    pub fn push_front_overdue(&mut self, candidate: AdmissionCandidate, hit: bool) {
        if self.ranked.is_some() {
            // Ranked mode: back under its own key.
            self.push_ranked(candidate);
            return;
        }
        self.overdue_kv_tokens += candidate.queued_kv_tokens();
        self.overdue_prompt_tokens += u64::from(candidate.fresh_prompt_tokens);
        if hit {
            self.overdue_hits.push_front(candidate);
        } else {
            if let Some(index) = self.at_risk.as_mut() {
                index.track_in(&candidate);
            }
            self.overdue.push_front(candidate);
        }
    }

    fn take_overdue(&mut self, hit: bool, index: usize) -> AdmissionCandidate {
        if let Some((request, _)) = self.ranked.as_ref().and_then(Ranked::head) {
            debug_assert!(!hit && index == 0, "ranked mode takes its head");
            return self.take_ranked(request).expect("the head is queued");
        }
        let candidate = if hit {
            self.overdue_hits.remove(index)
        } else {
            self.overdue.remove(index)
        }
        .expect("index is in range");
        if !hit {
            if let Some(at_risk) = self.at_risk.as_mut() {
                at_risk.track_out(&candidate);
            }
        }
        self.overdue_kv_tokens -= candidate.queued_kv_tokens();
        self.overdue_prompt_tokens -= u64::from(candidate.fresh_prompt_tokens);
        candidate
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
        if let Some(ranked) = self.ranked.as_mut() {
            ranked
                .entries
                .insert(candidate.request_id, (arrival, prefill_size(&candidate)));
            self.push_ranked(candidate);
            return;
        }
        if self.after.is_some() {
            debug_assert!(self
                .arrivals
                .back()
                .is_none_or(|&(last, _)| last <= arrival));
            self.arrivals.push_back((arrival, candidate.request_id));
        }
        if let (Some(index), SessionInput::Session { session_id, .. }) =
            (self.at_risk.as_mut(), candidate.session_input)
        {
            index.latest.insert(session_id, candidate);
        }
        self.inner.push(candidate, context);
    }

    /// Move every request that has waited past the bound by `now` out of the
    /// inner order into the overdue queue. `held` gives the prefix tokens HBM
    /// or a tier holds for a request; it is read only to split by class or,
    /// under [`HitsFirst::WhenAtRisk`], to count what a miss computes (and it
    /// then replaces the request's frozen `resident_prefix_tokens`).
    pub fn promote(&mut self, now: Time, mut held: impl FnMut(AdmissionCandidate) -> u32) {
        if let Some(ranked) = self.ranked.as_mut() {
            ranked.advance(now);
            return;
        }
        while let Some(&(arrival, request)) = self.arrivals.front() {
            if !self.is_overdue(arrival, now) {
                break;
            }
            self.arrivals.pop_front();
            if let Some(mut candidate) = self.inner.remove(request) {
                if self.at_risk.is_some() {
                    candidate.resident_prefix_tokens = held(candidate);
                }
                let hit = self.split && held(candidate) > 0;
                self.push_back_overdue(candidate, hit);
            }
        }
    }

    /// Under [`HitsFirst::WhenAtRisk`]: `session_id`'s latest request entered
    /// with an arrival, if it still waits in the inner order or among the
    /// overdue misses.
    pub fn queued_in_session(&self, session_id: u32) -> Option<AdmissionCandidate> {
        let index = self.at_risk.as_ref()?;
        let candidate = *index.latest.get(&session_id)?;
        (self.inner.contains(candidate.request_id) || index.misses.contains(&candidate.request_id))
            .then_some(candidate)
    }

    /// The next candidate `access` offers. `resident_prefix_tokens` refreshes
    /// the inner head (`PendingOrderPolicy::refresh_head`) before it is read.
    pub fn peek_head(
        &mut self,
        access: QueueAccess,
        context: &mut P::Context,
        resident_prefix_tokens: &mut dyn FnMut(AdmissionCandidate) -> u32,
    ) -> Option<(AdmissionCandidate, Queued)> {
        if !matches!(access, QueueAccess::Pending | QueueAccess::PendingHits) {
            if let Some((candidate, hit)) = self.overdue_head(access == QueueAccess::OverdueHits) {
                return Some((candidate, Queued::Overdue { hit }));
            }
        }
        if !access.takes_pending() {
            return None;
        }
        self.inner.refresh_head(resident_prefix_tokens);
        loop {
            let candidate = self.inner.peek()?;
            if access != QueueAccess::PendingHits || candidate.resident_prefix_tokens > 0 {
                return Some((candidate, Queued::Pending));
            }
            self.inner.pop(context);
            self.set_aside.push(candidate);
            self.inner.refresh_head(resident_prefix_tokens);
        }
    }

    /// Take the candidate [`Self::peek_head`] offered out of `from`.
    pub fn take_head(&mut self, from: Queued, context: &mut P::Context) -> AdmissionCandidate {
        match from {
            Queued::Overdue { hit } => self.take_overdue(hit, 0),
            Queued::Pending => self.inner.pop(context).expect("a peeked head"),
        }
    }

    /// Put a candidate taken this pass back at the front of `from`.
    pub fn return_head(
        &mut self,
        candidate: AdmissionCandidate,
        from: Queued,
        context: &mut P::Context,
    ) {
        match from {
            Queued::Overdue { hit } => self.push_front_overdue(candidate, hit),
            Queued::Pending => self.inner.push_front(candidate, context),
        }
    }

    /// Put the requests a [`QueueAccess::PendingHits`] pass went past back at
    /// the front of the inner order, in their order.
    pub fn restore_set_aside(&mut self, context: &mut P::Context) {
        for candidate in std::mem::take(&mut self.set_aside).into_iter().rev() {
            self.inner.push_front(candidate, context);
        }
    }
}

impl<P: PendingOrderPolicy> PendingOrderPolicy for Deadline<P> {
    type Context = P::Context;

    fn push(&mut self, candidate: AdmissionCandidate, context: &mut Self::Context) {
        if self.rank_active() {
            self.push_ranked(candidate);
        } else {
            self.inner.push(candidate, context);
        }
    }

    fn push_front(&mut self, candidate: AdmissionCandidate, context: &mut Self::Context) {
        if self.rank_active() {
            self.push_ranked(candidate);
        } else {
            self.inner.push_front(candidate, context);
        }
    }

    /// The inner head is read only when no request is overdue.
    fn refresh_head(&mut self, resident_prefix_tokens: &mut dyn FnMut(AdmissionCandidate) -> u32) {
        if !self.head_is_overdue() {
            self.inner.refresh_head(resident_prefix_tokens);
        }
    }

    fn peek(&self) -> Option<AdmissionCandidate> {
        match self.overdue_head(false) {
            Some((candidate, _)) => Some(candidate),
            None => self.inner.peek(),
        }
    }

    fn pop(&mut self, context: &mut Self::Context) -> Option<AdmissionCandidate> {
        match self.overdue_head(false) {
            Some((_, hit)) => Some(self.take_overdue(hit, 0)),
            None => self.inner.pop(context),
        }
    }

    fn remove(&mut self, request: RequestId) -> Option<AdmissionCandidate> {
        if self.rank_active() {
            return self.take_ranked(request);
        }
        for hit in [false, true] {
            let queue = if hit {
                &self.overdue_hits
            } else {
                &self.overdue
            };
            if let Some(index) = queue
                .iter()
                .position(|candidate| candidate.request_id == request)
            {
                return Some(self.take_overdue(hit, index));
            }
        }
        self.inner.remove(request)
    }

    fn contains(&self, request: RequestId) -> bool {
        if let Some(ranked) = &self.ranked {
            return ranked.queued.contains_key(&request);
        }
        self.overdue
            .iter()
            .chain(&self.overdue_hits)
            .any(|candidate| candidate.request_id == request)
            || self.inner.contains(request)
    }

    fn len(&self) -> usize {
        self.ranked.as_ref().map_or(0, |ranked| ranked.queued.len())
            + self.overdue.len()
            + self.overdue_hits.len()
            + self.inner.len()
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
        deadline.promote(Time::from_ms(15.0), |_| 0);
        assert!(deadline.head_is_overdue());
        assert_eq!(
            deadline.pop(&mut ()).map(|c| c.request_id),
            Some(RequestId(1))
        );
        assert!(!deadline.head_is_overdue(), "request 2 has waited 7 ms");
        deadline.promote(Time::from_ms(18.0), |_| 0);
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
        deadline.promote(Time::from_ms(5.0), |_| 0);
        assert_eq!(deadline.peek().map(|c| c.request_id), Some(RequestId(1)));
        deadline.promote(Time::from_ms(10.0), |_| 0);
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
        deadline.promote(Time::from_ms(1e9), |_| 0);
        assert!(!deadline.is_overdue(Time::ZERO, Time::from_ms(1e9)));
        assert_eq!(deadline.peek().map(|c| c.request_id), Some(RequestId(1)));
    }

    /// Four 100-token requests arriving at 0..3 ms, all overdue at 20 ms; odd
    /// ones hold `held` prefix tokens.
    fn split_overdue(hits_first: HitsFirst, held: u32) -> Deadline<FifoOrder> {
        let mut sequence = EnqueueSequence::default();
        let mut deadline = Deadline::new(FifoOrder::new())
            .with_bound(Time::from_ms(10.0))
            .with_hits_first(hits_first);
        enqueue(
            &mut deadline,
            &mut sequence,
            &[(0, 100, 0.0), (1, 100, 1.0), (2, 100, 2.0), (3, 100, 3.0)],
        );
        deadline.promote(Time::from_ms(20.0), |c| (c.request_id.0 % 2) * held);
        deadline
    }

    fn drain(deadline: &mut Deadline<FifoOrder>) -> Vec<u32> {
        std::iter::from_fn(|| deadline.pop(&mut ()))
            .map(|c| c.request_id.0)
            .collect()
    }

    #[test]
    fn hits_first_serves_overdue_hits_before_misses_each_oldest_first() {
        let mut deadline = split_overdue(HitsFirst::Always, 50);
        assert!(deadline.splits_by_class());
        assert_eq!(drain(&mut deadline), [1, 3, 0, 2]);
        assert_eq!(deadline.queued_prompt_tokens(), 0);
        // Without the split the held counts are not read.
        let mut deadline = split_overdue(HitsFirst::Off, 50);
        assert_eq!(drain(&mut deadline), [0, 1, 2, 3]);
    }

    #[test]
    fn a_hits_only_pass_offers_nothing_once_hits_are_gone_and_return_restores_the_class() {
        let mut deadline = split_overdue(HitsFirst::Always, 50);
        let hits_only = |deadline: &mut Deadline<FifoOrder>| {
            deadline
                .peek_head(QueueAccess::OverdueHits, &mut (), &mut |_| 0)
                .map(|(c, queued)| (c.request_id.0, queued))
        };
        assert_eq!(
            hits_only(&mut deadline),
            Some((1, Queued::Overdue { hit: true }))
        );
        let first = deadline.take_head(Queued::Overdue { hit: true }, &mut ());
        let second = deadline.take_head(Queued::Overdue { hit: true }, &mut ());
        assert_eq!(hits_only(&mut deadline), None);
        assert_eq!(
            deadline
                .peek_head(QueueAccess::Overdue, &mut (), &mut |_| 0)
                .map(|(c, queued)| (c.request_id.0, queued)),
            Some((0, Queued::Overdue { hit: false }))
        );
        deadline.return_head(second, Queued::Overdue { hit: true }, &mut ());
        deadline.return_head(first, Queued::Overdue { hit: true }, &mut ());
        assert_eq!(hits_only(&mut deadline).map(|(id, _)| id), Some(1));
        assert!(deadline.remove(RequestId(3)).is_some());
        assert!(deadline.remove(RequestId(0)).is_some());
        assert_eq!((deadline.len(), deadline.queued_prompt_tokens()), (2, 200));
    }

    #[test]
    fn a_pending_hits_pass_sets_misses_aside_and_restores_their_places() {
        let mut sequence = EnqueueSequence::default();
        let mut deadline = Deadline::new(FifoOrder::new());
        for (id, held) in [(0, 0), (1, 0), (2, 30), (3, 0)] {
            let candidate = sequence.freeze(
                RequestId(id),
                100,
                1,
                SessionInput::Standalone,
                Time::ZERO,
                held,
            );
            deadline.push(candidate, &mut ());
        }
        let (candidate, queued) = deadline
            .peek_head(QueueAccess::PendingHits, &mut (), &mut |_| 0)
            .unwrap();
        assert_eq!(
            (candidate.request_id, queued),
            (RequestId(2), Queued::Pending)
        );
        deadline.take_head(queued, &mut ());
        assert_eq!(
            deadline.peek_head(QueueAccess::PendingHits, &mut (), &mut |_| 0),
            None
        );
        deadline.restore_set_aside(&mut ());
        assert_eq!(drain(&mut deadline), [0, 1, 3]);
    }

    #[test]
    fn when_at_risk_counts_what_the_overdue_misses_compute_and_finds_queued_sessions() {
        let mut sequence = EnqueueSequence::default();
        let mut deadline = Deadline::new(FifoOrder::new())
            .with_bound(Time::from_ms(10.0))
            .with_hits_first(HitsFirst::WhenAtRisk);
        assert!(!deadline.splits_by_class());
        for id in 0..4 {
            let candidate = sequence.freeze(
                RequestId(id),
                100,
                1,
                SessionInput::Session {
                    session_id: id % 3,
                    session_start_time: Time::ZERO,
                    declared_prefix_tokens: 80,
                },
                Time::ZERO,
                0,
            );
            let arrival = Time::from_ms(if id == 3 { 5.0 } else { 0.0 });
            deadline.push_arrived(candidate, arrival, &mut ());
        }
        // Each request computes its 100 fresh tokens and the part of its
        // 80-token prefix not held: request 1 holds all 80.
        deadline.promote(Time::from_ms(10.0), |c| {
            if c.request_id == RequestId(1) {
                80
            } else {
                0
            }
        });
        assert_eq!(deadline.at_risk_horizon(), Some(460));
        // Session 0's latest request (3) is still pending; session 2's (2)
        // is an overdue miss.
        let queued = |deadline: &Deadline<FifoOrder>, session| {
            deadline.queued_in_session(session).map(|c| c.request_id.0)
        };
        assert_eq!(queued(&deadline, 0), Some(3));
        assert_eq!(queued(&deadline, 2), Some(2));
        assert!(deadline.remove(RequestId(2)).is_some());
        assert_eq!(queued(&deadline, 2), None);
        assert_eq!(deadline.at_risk_horizon(), Some(280));
        let first = deadline.pop(&mut ()).unwrap();
        assert_eq!(deadline.at_risk_horizon(), Some(100));
        deadline.push_front_overdue(first, false);
        assert_eq!(deadline.at_risk_horizon(), Some(280));
        // Turned on, the split classifies later promotions only.
        deadline.set_hits_first(true);
        deadline.promote(Time::from_ms(15.0), |_| 80);
        assert_eq!(drain(&mut deadline), [3, 0, 1]);
        assert_eq!(deadline.at_risk_horizon(), Some(0));
    }

    #[test]
    fn ranked_mode_serves_a_late_long_request_before_a_shorter_on_time_one() {
        use crate::worker::admission::prefill_rank::{PrefillRank, SquareCost};
        let rank = PrefillRank::Square(SquareCost {
            lambda_per_s: 1.0,
            d0_s: 10.0,
            k: 3.0,
            tokens_per_s: 100_000.0,
            pad: false,
        });
        let mut sequence = EnqueueSequence::default();
        let mut deadline =
            Deadline::new(ShortestJobFirst::by(JobSize::PrefillTokens)).with_rank(rank);
        enqueue(
            &mut deadline,
            &mut sequence,
            &[(0, 600_000, 0.0), (1, 100_000, 17_000.0)],
        );
        // At 1 s both are on time: shortest first.
        deadline.promote(Time::from_ms(1_000.0), |_| 0);
        assert_eq!(deadline.peek().map(|c| c.request_id), Some(RequestId(1)));
        assert!(!deadline.head_is_overdue());
        // At 17 s request 0 projects 5 s past tau = 18 s: weight 11, key 54_546.
        deadline.promote(Time::from_ms(17_000.0), |_| 0);
        assert_eq!(deadline.head_rank().map(|r| r.key), Some(54_546));
        assert_eq!(
            deadline.pop(&mut ()).map(|c| c.request_id),
            Some(RequestId(0))
        );
        assert_eq!(deadline.queued_prompt_tokens(), 100_000);
        assert!(deadline
            .rank_of(RequestId(0), 300_000, Time::from_ms(20_000.0))
            .is_some());
    }

    #[test]
    fn late_requests_go_by_deadline_and_the_tier_heads_compare_by_index() {
        use crate::worker::admission::prefill_rank::{PrefillRank, SquareCost};
        let rank = PrefillRank::Square(SquareCost {
            lambda_per_s: 1.0,
            d0_s: 10.0,
            k: 3.0,
            tokens_per_s: 100_000.0,
            pad: false,
        });
        let mut sequence = EnqueueSequence::default();
        let mut deadline =
            Deadline::new(ShortestJobFirst::by(JobSize::PrefillTokens)).with_rank(rank);
        // 0: 600k at 0 s (late from 12 s, due 18 s); 1: 1k at 1 s (late from
        // 10.99 s, due 11 s); 2: 50k at 30 s (on time until 39.5 s).
        enqueue(
            &mut deadline,
            &mut sequence,
            &[
                (0, 600_000, 0.0),
                (1, 1_000, 1_000.0),
                (2, 50_000, 30_000.0),
            ],
        );
        deadline.promote(Time::from_ms(30_000.0), |_| 0);
        // Both 0 and 1 are late: 1 is due first. Its index key (1k / (1 + 2 *
        // 19.01)) = 26 beats 2's 50k.
        assert_eq!(deadline.peek().map(|c| c.request_id), Some(RequestId(1)));
        assert_eq!(
            deadline.pop(&mut ()).map(|c| c.request_id),
            Some(RequestId(1))
        );
        // 0 heads the late tier: weight 1 + 2 * (30 + 6 - 18) = 37, key
        // 16_217 < 50_000.
        assert_eq!(deadline.head_rank().map(|r| r.key), Some(16_217));
        assert_eq!(
            deadline.pop(&mut ()).map(|c| c.request_id),
            Some(RequestId(0))
        );
        assert_eq!(
            (deadline.len(), deadline.queued_prompt_tokens()),
            (1, 50_000)
        );
        assert!(deadline.contains(RequestId(2)));
        assert_eq!(
            deadline.remove(RequestId(2)).map(|c| c.request_id),
            Some(RequestId(2))
        );
        assert_eq!((deadline.len(), deadline.queued_prompt_tokens()), (0, 0));
    }

    #[test]
    fn has_late_reports_a_non_empty_late_tier_as_of_the_last_promote() {
        use crate::worker::admission::prefill_rank::{PrefillRank, SquareCost};
        let rank = PrefillRank::Square(SquareCost {
            lambda_per_s: 10.0,
            d0_s: 10.0,
            k: 3.0,
            tokens_per_s: 100_000.0,
            pad: false,
        });
        let mut sequence = EnqueueSequence::default();
        let mut deadline =
            Deadline::new(ShortestJobFirst::by(JobSize::PrefillTokens)).with_rank(rank);
        assert!(!deadline.has_late());
        // 0: 1k at 0 s, late from 9.99 s; 1: 50k at 0 s, late from 9.5 s.
        enqueue(
            &mut deadline,
            &mut sequence,
            &[(0, 1_000, 0.0), (1, 50_000, 0.0)],
        );
        deadline.promote(Time::from_ms(9_000.0), |_| 0);
        assert!(!deadline.has_late());
        deadline.promote(Time::from_ms(9_600.0), |_| 0);
        assert!(deadline.has_late());
        // 0 is on time and shorter, so it goes first; 1 keeps the tier full.
        assert_eq!(
            deadline.pop(&mut ()).map(|c| c.request_id),
            Some(RequestId(0))
        );
        assert!(deadline.has_late());
        assert_eq!(
            deadline.pop(&mut ()).map(|c| c.request_id),
            Some(RequestId(1))
        );
        assert!(!deadline.has_late());
        // Outside ranked mode there is no late tier.
        let mut fifo = Deadline::new(FifoOrder::new());
        enqueue(&mut fifo, &mut sequence, &[(2, 1_000, 0.0)]);
        fifo.promote(Time::from_ms(60_000.0), |_| 0);
        assert!(!fifo.has_late());
    }
}
