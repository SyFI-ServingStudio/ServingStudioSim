//! Order in which a chunked-prefill lifecycle fills one partition's prefill
//! budget from its started prompts and its pending queue.
//!
//! vLLM schedules every started prompt's next chunk, in start order, before it
//! looks at the waiting queue ([`StartedOrder::StartOrder`]), so a short prompt
//! that arrives behind a long one waits for all of its chunks. Under
//! [`StartedOrder::ShortestRemaining`] (srpt, not vLLM) a queued prompt with no
//! more tokens left than a started one goes first, and the reverse. The pending
//! order must offer its shortest prompt first (`shortest-prefill-first`); srpt
//! only merges it with the started prompts, sorted by remaining tokens.
//! Requests past the force-schedule bound (`Deadline`) go before all: queued
//! ones, then started ones by arrival.
//!
//! One partition at a time. The pipeline head (one partition) and the
//! whole-iteration chunked-prefill lifecycle (one per attention DP rank, each
//! rank an engine with its own scheduler) supply the primitives through
//! [`PrefillFill`] and share [`PrefillFill::fill_prefill`].

use crate::common::Time;
use crate::worker::kv::ChunkedPrefillKv;
use crate::worker::shared::context::WorkerContext;

use super::AdmissionCandidate;

/// Started-prompt order of a chunked-prefill lifecycle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StartedOrder {
    /// Started prompts in start order, before any queued prompt (vLLM).
    #[default]
    StartOrder,
    /// Shortest remaining prefill first across started and queued prompts.
    ShortestRemaining,
}

/// Which queued prompts one fresh-admission pass may take.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FreshGate {
    /// Overdue requests only.
    OverdueOnly,
    /// Overdue requests, then the pending order.
    Any,
    /// Overdue requests, then pending prompts with at most this many prefill
    /// tokens left; the first longer one stops the pass.
    NoLongerThan(u32),
}

impl FreshGate {
    /// Whether the pass may take a queued head with `remaining` prefill tokens.
    pub(super) fn admits(self, overdue: bool, remaining: u32) -> bool {
        match self {
            Self::OverdueOnly => overdue,
            Self::Any => true,
            Self::NoLongerThan(bound) => overdue || remaining <= bound,
        }
    }
}

/// How a chunked-prefill lifecycle fills one partition's prefill budget.
pub(super) trait PrefillFill<K: ChunkedPrefillKv> {
    /// Per-formation admission state the lifecycle threads through its
    /// fresh-prompt pass (whatever gates admission beyond budget and KV).
    type Step;

    /// Whether a request that arrived at `arrival` is past `partition`'s
    /// force-schedule bound at `now`.
    fn is_overdue(&self, partition: u16, arrival: Time, now: Time) -> bool;

    /// `partition`'s started prompts, in start order.
    fn started_prefills_mut(&mut self, partition: u16) -> &mut Vec<AdmissionCandidate>;

    /// Admit queued prompts into `budget` as `gate` allows. An admitted prompt
    /// that does not finish joins the started list. Returns the budget taken.
    #[allow(clippy::too_many_arguments)]
    fn admit_fresh_prompts(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        partition: u16,
        step: &mut Self::Step,
        budget: u32,
        now: Time,
        gate: FreshGate,
    ) -> u32;

    /// Schedule `candidate`'s next chunk within `budget`; returns the chunk's
    /// tokens and the budget it took (`(0, 0)`: it cannot run now).
    fn schedule_started_chunk(
        &mut self,
        kv_store: &mut K,
        partition: u16,
        candidate: AdmissionCandidate,
        budget: u32,
    ) -> (u32, u32);

    /// Fill `partition`'s prefill `budget` in `order`.
    #[allow(clippy::too_many_arguments)]
    fn fill_prefill(
        &mut self,
        order: StartedOrder,
        kv_store: &mut K,
        context: &WorkerContext,
        partition: u16,
        step: &mut Self::Step,
        mut budget: u32,
        now: Time,
    ) {
        let remaining_of = |kv_store: &K, candidate: &AdmissionCandidate| {
            kv_store
                .resolved_prefill_context(candidate.request_id)
                .remaining_prefill_tokens()
        };
        let started = std::mem::take(self.started_prefills_mut(partition));
        let mut still_started = Vec::with_capacity(started.len());
        match order {
            StartedOrder::StartOrder => {
                // One that cannot run keeps its place.
                for candidate in started {
                    let remaining = remaining_of(kv_store, &candidate);
                    let (chunk_tokens, charged) =
                        self.schedule_started_chunk(kv_store, partition, candidate, budget);
                    budget -= charged;
                    if chunk_tokens < remaining {
                        still_started.push(candidate);
                    }
                }
                *self.started_prefills_mut(partition) = still_started;
                self.admit_fresh_prompts(
                    kv_store,
                    context,
                    partition,
                    step,
                    budget,
                    now,
                    FreshGate::Any,
                );
            }
            StartedOrder::ShortestRemaining => {
                let mut keyed: Vec<(bool, u64, AdmissionCandidate)> = {
                    let store = context.requests.borrow();
                    started
                        .into_iter()
                        .map(|candidate| {
                            let arrival = store[candidate.request_id].request.core.arrival_time;
                            if self.is_overdue(partition, arrival, now) {
                                (false, arrival.0, candidate)
                            } else {
                                (
                                    true,
                                    u64::from(remaining_of(kv_store, &candidate)),
                                    candidate,
                                )
                            }
                        })
                        .collect()
                };
                keyed.sort_by_key(|&(not_overdue, key, _)| (not_overdue, key));
                if budget > 0 {
                    budget -= self.admit_fresh_prompts(
                        kv_store,
                        context,
                        partition,
                        step,
                        budget,
                        now,
                        FreshGate::OverdueOnly,
                    );
                }
                for (not_overdue, _, candidate) in keyed {
                    let remaining = remaining_of(kv_store, &candidate);
                    if not_overdue && budget > 0 {
                        budget -= self.admit_fresh_prompts(
                            kv_store,
                            context,
                            partition,
                            step,
                            budget,
                            now,
                            FreshGate::NoLongerThan(remaining),
                        );
                    }
                    let (chunk_tokens, charged) =
                        self.schedule_started_chunk(kv_store, partition, candidate, budget);
                    budget -= charged;
                    if chunk_tokens < remaining {
                        still_started.push(candidate);
                    }
                }
                if budget > 0 {
                    self.admit_fresh_prompts(
                        kv_store,
                        context,
                        partition,
                        step,
                        budget,
                        now,
                        FreshGate::Any,
                    );
                }
                // Prompts admitted now and not finished join after the ones
                // already started.
                let admitted_now = self.started_prefills_mut(partition);
                still_started.append(admitted_now);
                *admitted_now = still_started;
            }
        }
    }
}
