//! Shortest remaining prefill first (srpt) across started and queued prompts,
//! not vLLM.
//!
//! vLLM schedules every started prompt's next chunk before it looks at the
//! waiting queue, so a short prompt that arrives behind a long one waits for all
//! of its chunks. Under srpt a queued prompt with no more tokens left than a
//! started one goes first, and the reverse. The pending order must offer its
//! shortest prompt first (`shortest-prefill-first`); srpt only merges it with
//! the started prompts, sorted by remaining tokens. Requests past the
//! force-schedule bound (`overdue.rs`) go before all: queued ones, then started
//! ones by arrival.
//!
//! One partition at a time. The pipeline head (one partition) and the
//! whole-iteration chunked-prefill lifecycle (one per attention DP rank, each
//! rank an engine with its own scheduler) supply the primitives through
//! [`SrptFill`] and share [`SrptFill::form_prefill_srpt`].

use crate::common::Time;
use crate::worker::kv::ChunkedPrefillKv;
use crate::worker::shared::context::WorkerContext;

use super::overdue::OverdueQueue;
use super::AdmissionCandidate;

/// How a chunked-prefill lifecycle fills one partition's prefill budget.
pub(super) trait SrptFill<K: ChunkedPrefillKv> {
    /// Per-step admission state the lifecycle threads through its fresh-prompt
    /// pass (whatever gates admission beyond budget and KV).
    type Step;

    /// The force-schedule bound, if any.
    fn force_bound(&self) -> Option<&OverdueQueue>;

    /// `partition`'s started prompts, in start order.
    fn started_prefills_mut(&mut self, partition: u16) -> &mut Vec<AdmissionCandidate>;

    /// Admit fresh prompts into `budget` (landed reads, overdue requests, then
    /// the pending order), stopping at the first pending one with more than
    /// `no_longer_than` prefill tokens (`overdue_only`: no pending ones). An
    /// admitted prompt that does not finish joins the started list. Returns
    /// the budget the admitted prompts took.
    #[allow(clippy::too_many_arguments)]
    fn admit_fresh_prompts(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        partition: u16,
        step: &mut Self::Step,
        budget: u32,
        now: Time,
        no_longer_than: Option<u32>,
        overdue_only: bool,
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

    /// Fill `budget` shortest remaining prefill first: before each started
    /// prompt, queued prompts with no more tokens left than it go first.
    /// Overdue requests, queued then started (oldest first), go before all.
    fn form_prefill_srpt(
        &mut self,
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
        let started_prefills = std::mem::take(self.started_prefills_mut(partition));
        let mut started: Vec<(bool, u64, AdmissionCandidate)> = {
            let force_bound = self.force_bound();
            let store = context.requests.borrow();
            started_prefills
                .into_iter()
                .map(|candidate| {
                    let arrival = store[candidate.request_id].request.core.arrival_time;
                    let overdue = force_bound.is_some_and(|bound| bound.is_overdue(arrival, now));
                    let key = if overdue {
                        arrival.0
                    } else {
                        u64::from(remaining_of(kv_store, &candidate))
                    };
                    (!overdue, key, candidate)
                })
                .collect()
        };
        started.sort_by_key(|&(not_overdue, key, _)| (not_overdue, key));
        if self.force_bound().is_some() && budget > 0 {
            budget -= self
                .admit_fresh_prompts(kv_store, context, partition, step, budget, now, None, true);
        }
        let mut still_started = Vec::with_capacity(started.len());
        for (not_overdue, _, candidate) in started {
            let remaining = remaining_of(kv_store, &candidate);
            if not_overdue && budget > 0 {
                budget -= self.admit_fresh_prompts(
                    kv_store,
                    context,
                    partition,
                    step,
                    budget,
                    now,
                    Some(remaining),
                    false,
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
            self.admit_fresh_prompts(kv_store, context, partition, step, budget, now, None, false);
        }
        // Prompts admitted now and not finished join after the ones already started.
        let admitted_now = self.started_prefills_mut(partition);
        still_started.append(admitted_now);
        *admitted_now = still_started;
    }
}
