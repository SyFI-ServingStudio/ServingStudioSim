//! Where a chunked-prefill lifecycle finds the fresh prompt it admits next.
//!
//! Shared by the whole-iteration lifecycle (`chunked_prefill_admission.rs`, one
//! queue set per attention DP rank) and the pipeline head
//! (`pipelined_chunked_prefill_admission.rs`). A partition offers, in order:
//! requests whose tier read has landed (they hold their HBM blocks, vLLM's
//! skipped-waiting queue), requests past the force-schedule bound
//! (`overdue.rs`), then the pending order. A request passed over for its tier
//! read wait (`prefix_fetch.rs`) leaves its source for the rest of the pass and
//! goes back where it was afterwards.

use crate::worker::kv::ChunkedPrefillKv;

use super::overdue::OverdueQueue;
use super::prefix_fetch::PrefixFetch;
use super::{AdmissionCandidate, PendingOrderPolicy};

/// Which queue [`FreshQueues::peek`] found its candidate in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FreshSource {
    Landed,
    Overdue,
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
}

impl<P: PendingOrderPolicy> FreshQueues<'_, P> {
    /// The next candidate: a landed read, else an overdue request, else the
    /// pending order's head, re-ranked by what HBM or a tier now holds.
    /// `overdue_only` offers overdue requests alone.
    pub(super) fn peek<K: ChunkedPrefillKv>(
        &mut self,
        kv_store: &K,
        overdue_only: bool,
    ) -> Option<(FreshSource, AdmissionCandidate)> {
        let partition = self.partition;
        let landed = self
            .prefix_fetch
            .as_ref()
            .filter(|_| !overdue_only)
            .and_then(|fetch| fetch.peek_landed(partition));
        if let Some(candidate) = landed {
            return Some((FreshSource::Landed, candidate));
        }
        if let Some(candidate) = self
            .overdue
            .as_ref()
            .and_then(|overdue| overdue.peek(partition))
        {
            return Some((FreshSource::Overdue, candidate));
        }
        if overdue_only {
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
        self.policy
            .peek()
            .map(|candidate| (FreshSource::Policy, candidate))
    }

    /// Take `candidate` out of the queue [`Self::peek`] found it in.
    pub(super) fn take(&mut self, source: FreshSource, candidate: AdmissionCandidate) {
        match source {
            FreshSource::Landed => self
                .prefix_fetch
                .as_mut()
                .expect("a landed request implies prefix fetch")
                .pop_landed(self.partition),
            FreshSource::Overdue => {
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

    /// Put the requests passed over this pass back at the front of the queues
    /// they came from, in their order.
    pub(super) fn return_passed_over(
        &mut self,
        passed_over: Vec<(FreshSource, AdmissionCandidate)>,
    ) {
        // Newest first, so each queue gets its order back.
        for (source, candidate) in passed_over.into_iter().rev() {
            match source {
                FreshSource::Policy => self.policy.push_front(candidate, self.policy_context),
                FreshSource::Overdue => self
                    .overdue
                    .as_mut()
                    .expect("an overdue request implies the bound")
                    .push_front(self.partition, candidate),
                FreshSource::Landed => unreachable!("a landed read is never passed over"),
            }
        }
    }
}
