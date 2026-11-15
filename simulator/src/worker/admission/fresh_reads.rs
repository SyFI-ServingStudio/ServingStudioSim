//! Host-tier reads of fresh requests, shared by the chunked lifecycles: the
//! pipeline head (one partition) and the whole-iteration chunked prefill (one
//! per attention DP rank).
//!
//! With host tiers behind HBM (`TieredPrefixKv`), a fresh request goes pending
//! -> reading -> landed -> started, as under vLLM's KV connector: when
//! admission reaches a request whose context its partition's tiers hold beyond
//! HBM, its read starts if the whole request fits in HBM, holding that
//! footprint, and the request leaves its queue (`WAITING_FOR_REMOTE_KVS`)
//! while other requests go ahead; one that does not fit stops admission. Once
//! the read lands the request is offered before every queue of its partition,
//! in landing order, and its context goes back into HBM just before admission
//! resolves its prefix; one evicted again before admission is read again. A
//! `ReadWait` bound passes over or stops at a request whose tier would begin
//! its read too late (`read_wait.rs`).
//!
//! Under staged loading (a loading area, not vLLM) reads start in a loading
//! pass before each fresh-admission pass instead: it walks the overdue
//! requests, then the pending order, and starts each read the KV store's
//! loading area has room for. A read that does not fit stops its landing
//! queue (HBM's, or DRAM's for an SSD read that hops on) for the rest of the
//! pass, so no later read jumps it; requests that start no read keep their
//! places. Admission passes over a request still needing a read, holding
//! nothing, and a landed one whose context HBM evicted again goes back to the
//! pending order to be read again.
//!
//! The lifecycle owns its partitions' queues (`Deadline`) and hands each call
//! the one partition it works on; this module owns only the requests out of
//! those queues and when their reads start. Every request ranks by
//! [`TieredPrefixKv::reusable_prefix_tokens`], which is HBM's resident prefix
//! without tiers.

use std::collections::{HashMap, VecDeque};

use crate::common::{RequestId, Time};
use crate::worker::kv::{LoadingQueue, TieredPrefixKv};
use crate::worker::shared::advance_scope::PartitionId;

use super::policy::{QueueAccess, Queued};
use super::read_wait::{ReadWait, ReadWaitPolicy};
use super::{AdmissionCandidate, Deadline, PendingOrderPolicy};

/// Where a fresh-admission pass found its candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FreshSource {
    Landed,
    Queued(Queued),
}

/// What admission does with the candidate at the head of a pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AtHead {
    /// Its context is in HBM: admit it as any request.
    Admit,
    /// It left the head: its read started, or the pass went past it.
    Moved,
    /// Admission stops here this pass.
    Stop,
}

/// A request whose read has landed.
#[derive(Clone, Copy, Debug)]
struct Landed {
    candidate: AdmissionCandidate,
    /// Its context is back in HBM.
    restored: bool,
}

#[derive(Debug)]
pub(super) struct FreshReads {
    /// Fresh requests out of their queue while their host-tier read runs.
    reading: HashMap<RequestId, (PartitionId, AdmissionCandidate)>,
    /// Per partition: requests whose read has landed, in landing order,
    /// offered before every queue.
    landed: Vec<VecDeque<Landed>>,
    /// When a read may start; `None`: whenever it fits (vLLM).
    read_wait: Option<ReadWait>,
}

impl FreshReads {
    pub(super) fn new(partitions: usize) -> Self {
        Self {
            reading: HashMap::new(),
            landed: (0..partitions).map(|_| VecDeque::new()).collect(),
            read_wait: None,
        }
    }

    /// Start a read only within `read_wait` of its tier's channel.
    pub(super) fn set_read_wait(&mut self, read_wait: ReadWait) {
        assert!(
            read_wait.max > Time::ZERO,
            "a read-wait bound must be positive"
        );
        self.read_wait = Some(read_wait);
    }

    /// Requests out of their queue while their read runs.
    pub(super) fn reading(&self) -> usize {
        self.reading.len()
    }

    /// Requests reading or landed and not yet admitted.
    pub(super) fn len(&self) -> usize {
        self.reading.len() + self.landed.iter().map(VecDeque::len).sum::<usize>()
    }

    /// The next fresh candidate of `partition`: a landed read, then what
    /// `queues` offers of the overdue requests and the pending order, whose
    /// head is re-ranked by what HBM or a tier now holds. Landed reads come
    /// only with the pending order.
    pub(super) fn peek<P: PendingOrderPolicy, K: TieredPrefixKv>(
        &self,
        partition: PartitionId,
        policy: &mut Deadline<P>,
        policy_context: &mut P::Context,
        kv_store: &K,
        queues: QueueAccess,
    ) -> Option<(FreshSource, AdmissionCandidate)> {
        if queues.takes_pending() {
            if let Some(landed) = self.landed[usize::from(partition)].front() {
                return Some((FreshSource::Landed, landed.candidate));
            }
        }
        policy
            .peek_head(queues, policy_context, &mut |candidate| {
                kv_store.reusable_prefix_tokens(
                    partition,
                    candidate.fresh_prompt_tokens,
                    candidate.session_input,
                )
            })
            .map(|(candidate, queued)| (FreshSource::Queued(queued), candidate))
    }

    /// Take the candidate [`Self::peek`] returned out of its source.
    pub(super) fn take<P: PendingOrderPolicy>(
        &mut self,
        partition: PartitionId,
        policy: &mut Deadline<P>,
        policy_context: &mut P::Context,
        source: FreshSource,
        candidate: AdmissionCandidate,
    ) {
        match source {
            FreshSource::Landed => {
                let landed = self.landed[usize::from(partition)].pop_front();
                debug_assert_eq!(landed.map(|landed| landed.candidate), Some(candidate));
            }
            FreshSource::Queued(queued) => {
                let taken = policy.take_head(queued, policy_context);
                debug_assert_eq!(taken, candidate);
            }
        }
    }

    /// Put the candidates passed over this pass back at the front of the
    /// queues they came from, in their order, after the pending requests a
    /// hits-only pass went past. A landed one, whose context HBM evicted
    /// again, goes back to the pending order.
    pub(super) fn return_passed_over<P: PendingOrderPolicy>(
        policy: &mut Deadline<P>,
        policy_context: &mut P::Context,
        passed_over: Vec<(FreshSource, AdmissionCandidate)>,
    ) {
        policy.restore_set_aside(policy_context);
        for (source, candidate) in passed_over.into_iter().rev() {
            let queued = match source {
                FreshSource::Landed => Queued::Pending,
                FreshSource::Queued(queued) => queued,
            };
            policy.return_head(candidate, queued, policy_context);
        }
    }

    /// Staged loading: start `partition`'s reads of the overdue requests,
    /// then of the pending order, in order, while the loading area each lands
    /// in has room (module docs). The pass ends once both landing queues have
    /// stopped.
    pub(super) fn start_staged_reads<P: PendingOrderPolicy, K: TieredPrefixKv>(
        &mut self,
        partition: PartitionId,
        policy: &mut Deadline<P>,
        policy_context: &mut P::Context,
        kv_store: &mut K,
        now: Time,
    ) {
        let (mut hbm_stopped, mut dram_stopped) = (false, false);
        let mut kept = Vec::new();
        while !(hbm_stopped && dram_stopped) {
            let Some((candidate, queued)) =
                policy.peek_head(QueueAccess::All, policy_context, &mut |candidate| {
                    kv_store.reusable_prefix_tokens(
                        partition,
                        candidate.fresh_prompt_tokens,
                        candidate.session_input,
                    )
                })
            else {
                break;
            };
            policy.take_head(queued, policy_context);
            let source = FreshSource::Queued(queued);
            let Some(read) = kv_store.plan_tier_read(
                partition,
                candidate.request_id,
                candidate.fresh_prompt_tokens,
                candidate.remaining_output_tokens,
                candidate.session_input,
                now,
            ) else {
                kept.push((source, candidate));
                continue;
            };
            let stopped = match read.staged.expect("a staged read").queue {
                LoadingQueue::Hbm => &mut hbm_stopped,
                LoadingQueue::Dram => &mut dram_stopped,
            };
            if *stopped || !kv_store.has_loading_room(partition, &read) {
                *stopped = true;
                kept.push((source, candidate));
                continue;
            }
            kv_store.start_tier_read(
                partition,
                candidate.request_id,
                candidate.session_input,
                read,
                now,
            );
            self.reading
                .insert(candidate.request_id, (partition, candidate));
        }
        Self::return_passed_over(policy, policy_context, kept);
    }

    /// Move the reads landed by `now` from reading to their partition's
    /// landed queue; returns the latest landing.
    pub(super) fn land<K: TieredPrefixKv>(&mut self, kv_store: &mut K, now: Time) -> Option<Time> {
        let mut latest = None;
        for (request, ready) in kv_store.take_landed_reads(now) {
            let (partition, candidate) = self
                .reading
                .remove(&request)
                .expect("a landed read belongs to a reading request");
            self.landed[usize::from(partition)].push_back(Landed {
                candidate,
                restored: false,
            });
            latest = Some(ready);
        }
        latest
    }

    /// Decide what admission does with the candidate [`Self::peek`] returned
    /// (module docs). A landed candidate's context goes back into HBM; one
    /// whose partition's tiers hold more than HBM starts its read or is
    /// passed over (`passed_over` takes it), or stops admission.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn at_head<P: PendingOrderPolicy, K: TieredPrefixKv>(
        &mut self,
        partition: PartitionId,
        policy: &mut Deadline<P>,
        policy_context: &mut P::Context,
        kv_store: &mut K,
        source: FreshSource,
        candidate: AdmissionCandidate,
        passed_over: &mut Vec<(FreshSource, AdmissionCandidate)>,
        now: Time,
    ) -> AtHead {
        let landed = &mut self.landed[usize::from(partition)];
        if source == FreshSource::Landed && !landed[0].restored {
            // Its blocks were held since the read started: the context
            // fills them and the admission that follows takes them over.
            kv_store.restore_landed_read(partition, candidate.request_id, now);
            landed[0].restored = true;
            return AtHead::Admit;
        }
        let Some(read) = kv_store.plan_tier_read(
            partition,
            candidate.request_id,
            candidate.fresh_prompt_tokens,
            candidate.remaining_output_tokens,
            candidate.session_input,
            now,
        ) else {
            return AtHead::Admit;
        };
        if kv_store.stages_tier_reads() {
            // Not landed: a loading pass starts its read.
            self.take(partition, policy, policy_context, source, candidate);
            passed_over.push((source, candidate));
            return AtHead::Moved;
        }
        if let Some(read_wait) = self
            .read_wait
            .filter(|read_wait| read_wait.exceeded(read.queue_wait))
        {
            match read_wait.policy {
                ReadWaitPolicy::PassOver => {
                    self.take(partition, policy, policy_context, source, candidate);
                    passed_over.push((source, candidate));
                    return AtHead::Moved;
                }
                ReadWaitPolicy::Block => return AtHead::Stop,
            }
        }
        // The whole request must fit before its read starts (vLLM's waiting
        // loop stops at an async load that cannot allocate).
        if !kv_store.fits(partition, &read.hold) {
            return AtHead::Stop;
        }
        kv_store.start_tier_read(
            partition,
            candidate.request_id,
            candidate.session_input,
            read,
            now,
        );
        self.take(partition, policy, policy_context, source, candidate);
        self.reading
            .insert(candidate.request_id, (partition, candidate));
        AtHead::Moved
    }

    /// Whether serving `partition`'s overdue misses oldest first would evict
    /// a queued request's held context. Those requests compute `horizon`
    /// prefill tokens ahead of everything else, and each computed context is
    /// written to the tiers (write-through as it finishes, or into the victim
    /// tier as HBM makes room for it), so `horizon` written tokens is the
    /// horizon. A context is at risk if it evicts it from every tier holding
    /// it, its session's queued request resumes from it, and HBM does not
    /// hold it for a victim tier to take.
    pub(super) fn held_context_at_risk<P: PendingOrderPolicy, K: TieredPrefixKv>(
        partition: PartitionId,
        policy: &Deadline<P>,
        kv_store: &K,
        horizon: u64,
    ) -> bool {
        let victim_tier = kv_store.tiers_take_hbm_evictions(partition);
        kv_store
            .tier_sessions_at_risk(partition, horizon)
            .into_iter()
            .any(|session_id| {
                let Some(candidate) = policy.queued_in_session(session_id) else {
                    return false;
                };
                let (fresh, session_input) =
                    (candidate.fresh_prompt_tokens, candidate.session_input);
                let hbm = kv_store.resident_prefix_tokens(fresh, session_input);
                !(hbm > 0 && victim_tier)
                    && kv_store.reusable_prefix_tokens(partition, fresh, session_input) > hbm
            })
    }
}
