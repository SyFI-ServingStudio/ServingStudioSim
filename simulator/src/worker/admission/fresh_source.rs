//! Where a chunked-prefill lifecycle finds the fresh prompt it admits next.
//!
//! Shared by the whole-iteration lifecycle (`chunked_prefill_admission.rs`, one
//! queue set per attention DP rank) and the pipeline head
//! (`pipelined_chunked_prefill_admission.rs`). A partition offers, in order:
//! requests whose tier read has landed (they hold their HBM blocks, vLLM's
//! skipped-waiting queue), requests past the force-schedule bound
//! (`overdue.rs`), then the pending order. A request passed over for its tier
//! read wait (`prefix_fetch.rs`) leaves its source for the rest of the pass and
//! goes back where it was afterwards. In staged mode a loading pass walks the
//! overdue and pending queues first and starts their reads
//! ([`FreshQueues::stage_loads`]).

use crate::common::Time;
use crate::worker::kv::ChunkedPrefillKv;

use super::overdue::OverdueQueue;
use super::prefix_fetch::PrefixFetch;
use super::{AdmissionCandidate, PendingOrderPolicy};

/// Which requests an admission pass may take from the overdue queue.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OverdueAccess {
    /// Overdue requests alone, hits first.
    Only,
    /// Overdue hits alone (`OverdueQueue::with_hits_first`).
    OnlyHits,
    /// Landed reads, overdue requests, then the pending order.
    Include,
    /// Landed reads and the pending order, not the overdue queue.
    Skip,
    /// Landed reads and the pending order's hits (a request that finds its
    /// context held: it computes only its new tokens), not the overdue queue;
    /// the pending requests it passes keep their places.
    SkipHitsOnly,
}

/// Which queue [`FreshQueues::peek`] found its candidate in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FreshSource {
    Landed,
    /// The overdue queue; `true`: its hits queue.
    Overdue(bool),
    Policy,
}

/// One partition's fresh-prompt queues, borrowed from the lifecycle for one
/// admission pass.
pub(super) struct FreshQueues<'a, P: PendingOrderPolicy> {
    pub(super) partition: u16,
    pub(super) policy: &'a mut P,
    pub(super) policy_context: &'a mut P::Context,
    pub(super) prefix_fetch: &'a mut Option<PrefixFetch>,
    pub(super) overdue: &'a mut Option<OverdueQueue>,
    /// Pending requests [`OverdueAccess::SkipHitsOnly`] passed, in order.
    pub(super) set_aside: Vec<AdmissionCandidate>,
}

impl<P: PendingOrderPolicy> FreshQueues<'_, P> {
    /// The next candidate: a landed read, else an overdue request, else the
    /// pending order's head, re-ranked by what HBM or a tier now holds.
    /// `access` says whether overdue requests are offered, alone, or not.
    pub(super) fn peek<K: ChunkedPrefillKv>(
        &mut self,
        kv_store: &K,
        access: OverdueAccess,
    ) -> Option<(FreshSource, AdmissionCandidate)> {
        let partition = self.partition;
        let landed = self
            .prefix_fetch
            .as_ref()
            .filter(|_| !matches!(access, OverdueAccess::Only | OverdueAccess::OnlyHits))
            .and_then(|fetch| fetch.peek_landed(partition));
        if let Some(candidate) = landed {
            return Some((FreshSource::Landed, candidate));
        }
        if let Some((candidate, hit)) = self
            .overdue
            .as_ref()
            .filter(|_| !matches!(access, OverdueAccess::Skip | OverdueAccess::SkipHitsOnly))
            .and_then(|overdue| overdue.peek(partition, access == OverdueAccess::OnlyHits))
        {
            return Some((FreshSource::Overdue(hit), candidate));
        }
        if matches!(access, OverdueAccess::Only | OverdueAccess::OnlyHits) {
            return None;
        }
        let fetch = &mut *self.prefix_fetch;
        self.policy.refresh_head(&mut |candidate| {
            let hbm = kv_store
                .resident_prefix_tokens(candidate.fresh_prompt_tokens, candidate.session_input);
            fetch.as_mut().map_or(hbm, |fetch| {
                fetch.rank_tokens(partition, candidate.session_input, hbm)
            })
        });
        loop {
            let candidate = self.policy.peek()?;
            if access != OverdueAccess::SkipHitsOnly || candidate.resident_prefix_tokens > 0 {
                return Some((FreshSource::Policy, candidate));
            }
            self.policy.pop(self.policy_context);
            self.set_aside.push(candidate);
            let fetch = &mut *self.prefix_fetch;
            self.policy.refresh_head(&mut |candidate| {
                let hbm = kv_store
                    .resident_prefix_tokens(candidate.fresh_prompt_tokens, candidate.session_input);
                fetch.as_mut().map_or(hbm, |fetch| {
                    fetch.rank_tokens(partition, candidate.session_input, hbm)
                })
            });
        }
    }

    /// Take `candidate` out of the queue [`Self::peek`] found it in.
    pub(super) fn take(&mut self, source: FreshSource, candidate: AdmissionCandidate) {
        match source {
            FreshSource::Landed => self
                .prefix_fetch
                .as_mut()
                .expect("a landed request implies prefix fetch")
                .pop_landed(self.partition),
            FreshSource::Overdue(_) => {
                let popped = self
                    .overdue
                    .as_mut()
                    .expect("an overdue request implies the bound")
                    .pop(self.partition);
                debug_assert_eq!(popped, Some(candidate));
            }
            FreshSource::Policy => {
                let popped = self.policy.pop(self.policy_context);
                debug_assert_eq!(popped, Some(candidate));
            }
        }
    }

    /// Staged mode: walk the overdue queue, then the pending order, in
    /// admission order, starting the read of every request a tier holds
    /// beyond HBM until both loading queues stop; the rest go back in order.
    pub(super) fn stage_loads<K: ChunkedPrefillKv>(&mut self, kv_store: &mut K, now: Time) {
        let partition = self.partition;
        let Some(fetch) = self.prefix_fetch.as_mut().filter(|f| f.is_staged()) else {
            return;
        };
        let mut gate = fetch.load_gate(partition);
        let mut kept = Vec::new();
        while !gate.closed() {
            let Some((candidate, hit)) =
                self.overdue.as_ref().and_then(|o| o.peek(partition, false))
            else {
                break;
            };
            self.overdue.as_mut().expect("peeked").pop(partition);
            if !fetch.stage(kv_store, partition, candidate, &mut gate, now) {
                kept.push((FreshSource::Overdue(hit), candidate));
            }
        }
        while !gate.closed() {
            self.policy.refresh_head(&mut |candidate| {
                let hbm = kv_store
                    .resident_prefix_tokens(candidate.fresh_prompt_tokens, candidate.session_input);
                fetch.rank_tokens(partition, candidate.session_input, hbm)
            });
            let Some(candidate) = self.policy.pop(self.policy_context) else {
                break;
            };
            if !fetch.stage(kv_store, partition, candidate, &mut gate, now) {
                kept.push((FreshSource::Policy, candidate));
            }
        }
        self.return_passed_over(kept);
    }

    /// Put the requests passed over this pass back at the front of the queues
    /// they came from, in their order.
    pub(super) fn return_passed_over(
        &mut self,
        passed_over: Vec<(FreshSource, AdmissionCandidate)>,
    ) {
        // Newest first, so each queue gets its order back.
        for candidate in std::mem::take(&mut self.set_aside).into_iter().rev() {
            self.policy.push_front(candidate, self.policy_context);
        }
        for (source, candidate) in passed_over.into_iter().rev() {
            match source {
                FreshSource::Policy => self.policy.push_front(candidate, self.policy_context),
                FreshSource::Overdue(hit) => self
                    .overdue
                    .as_mut()
                    .expect("an overdue request implies the bound")
                    .push_front(self.partition, candidate, hit),
                // Staged loading: a landed context evicted before admission
                // goes back to pending, to be read again.
                FreshSource::Landed => self.policy.push_front(candidate, self.policy_context),
            }
        }
    }
}
