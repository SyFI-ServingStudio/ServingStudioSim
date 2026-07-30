//! `LocalPrefillDecodeAdmission<P>` — the fresh→prefill→decode→completion
//! lifecycle (A1) with a swappable selection policy `P`, generalized over N KV
//! partitions (DP shards).
//! Owns the pending queue + soft per-group token budget + the `LoadBalance` cursor
//! that routes fresh prefills across groups.
//!
//! ONE admission covers both the single-group barebone worker (`num_partitions = 1`,
//! `LoadBalance::Single`) and the multi-group HP/DP worker (`num_partitions = N`,
//! `LoadBalance::RoundRobin`) — the real crate keeps these as two separate worker
//! files, and collapsing them here is exactly the reuse the refactor targets. The
//! per-partition loop is byte-identical to the single-group code at N=1.
//!
//! Requires `K: IterWorkerKv` on the impl (the M2 capability bound). Speaks only
//! tokens + opaque `K::Footprint`; never matches a concrete KV type. Stage vocab is
//! `UnifiedStage`. External calls verified against real source: `prefill_fits_budget`,
//! `LoadBalance::choose`, `RequestStore::mark_admitted`,
//! `RequestRecord::{record_token,record_first_token,is_complete}`.

use crate::common::{RequestId, Time, UnifiedStage};
use crate::worker::admission_helpers::{prefill_fits_budget, LoadBalance};
use crate::worker::types::{WorkerEventCommon, WorkerMsgCommon};

use super::super::kv::{IterWorkerKv, KvStore};
use super::super::shared::advance_scope::{AdvanceScope, PartitionId};
use super::super::shared::context::WorkerContext;
use super::policy::{AdmissionCandidate, PendingOrderPolicy};
use super::IterAdmission;

pub struct LocalPrefillDecodeAdmission<P: PendingOrderPolicy> {
    /// The pending queue lives INSIDE the policy (it owns the ordering structure), so
    /// `form_batch` only ever touches the head — no per-iteration backlog walk.
    policy: P,
    policy_ctx: P::Context,
    /// Distributed out of `WorkerConfig.max_batch_tokens` (soft per-group budget).
    max_batch_tokens: Option<u32>,
    /// Fresh-prefill placement across DP shards (`Single` for barebone).
    balance: LoadBalance,
}

impl<P: PendingOrderPolicy> LocalPrefillDecodeAdmission<P> {
    pub fn new(
        policy: P,
        policy_ctx: P::Context,
        max_batch_tokens: Option<u32>,
        balance: LoadBalance,
    ) -> Self {
        Self {
            policy,
            policy_ctx,
            max_batch_tokens,
            balance,
        }
    }

    /// Reserve + store seed + `Prefill` stamp for one admitted request into `partition_id`.
    fn admit_store<K: KvStore>(
        &self,
        kv_store: &mut K,
        context: &WorkerContext,
        candidate: &AdmissionCandidate,
        partition_id: PartitionId,
        now: Time,
    ) {
        let footprint = kv_store.footprint(candidate.request, candidate.prompt, candidate.decode);
        kv_store.reserve(candidate.request, partition_id, footprint);
        let mut store = context.requests.borrow_mut();
        store.mark_admitted(candidate.request);
        let record = &mut store[candidate.request];
        record.active_chunk_len = candidate.prompt;
        record.prefix_kv = 0;
        context.stamp_stage(record, now, UnifiedStage::Prefill as u16);
    }
}

impl<P: PendingOrderPolicy, K: KvStore + IterWorkerKv> IterAdmission<K>
    for LocalPrefillDecodeAdmission<P>
{
    type Msg = WorkerMsgCommon;
    type Event = WorkerEventCommon;

    /// Freezes the candidate facts ONCE, at arrival, and hands them to the policy's
    /// queue. `prompt`/`decode`/`arrival_seq` never change afterwards, so nothing has to
    /// re-read the store per pending request per iteration.
    fn accept_message(&mut self, _kv: &mut K, msg: WorkerMsgCommon, context: &WorkerContext) {
        let WorkerMsgCommon::Request(rid) = msg;
        let candidate = {
            let mut store = context.requests.borrow_mut();
            let record = &mut store[rid];
            let arrival = record.arrival_time;
            context.stamp_stage(record, arrival, UnifiedStage::Pending as u16);
            AdmissionCandidate {
                request: rid,
                arrival_seq: arrival.0,
                prompt: record.prompt_len,
                decode: record.decode_len,
                deadline: None,
                matched_tokens: 0,
            }
        };
        self.policy.push(candidate, &mut self.policy_ctx);
    }

    fn form_batch(&mut self, kv_store: &mut K, context: &WorkerContext, now: Time) -> bool {
        let num_partitions = kv_store.num_partitions();
        let had_live_decode =
            (0..num_partitions as u16).any(|partition| kv_store.has_live_decode(partition));

        match self.max_batch_tokens {
            None => {
                // One fresh prefill per iter, into a balance-chosen group. Only the
                // queue HEAD is ever examined — O(1), whatever the backlog depth.
                if let Some(candidate) = self.policy.peek() {
                    let partition_id = self.balance.choose(num_partitions) as u16;
                    let footprint =
                        kv_store.footprint(candidate.request, candidate.prompt, candidate.decode);
                    if kv_store.fits(partition_id, &footprint) {
                        self.policy.pop(&mut self.policy_ctx);
                        self.admit_store(kv_store, context, &candidate, partition_id, now);
                    }
                    // Declined: the head stays queued (peek did not dequeue it).
                }
            }
            Some(budget) => {
                // Per-group budget: each shard reserves its own live decodes then
                // fills its own remainder. RR picks each candidate's target group;
                // stop at the first head that cannot be taken (budget or KV gate) —
                // the same stop-at-first-failure rule as before, but now the tail is
                // never ranked, only the heads actually consumed.
                let decode_tokens: Vec<u32> = (0..num_partitions as u16)
                    .map(|partition| kv_store.live_decode_count(partition))
                    .collect();
                let mut admitted_tokens = vec![0u32; num_partitions];
                while let Some(candidate) = self.policy.peek() {
                    let partition_id = self.balance.choose(num_partitions) as u16;
                    let partition_index = partition_id as usize;
                    if !prefill_fits_budget(
                        budget,
                        decode_tokens[partition_index],
                        admitted_tokens[partition_index],
                        candidate.prompt,
                    ) {
                        break;
                    }
                    let footprint =
                        kv_store.footprint(candidate.request, candidate.prompt, candidate.decode);
                    if !kv_store.fits(partition_id, &footprint) {
                        break;
                    }
                    self.policy.pop(&mut self.policy_ctx);
                    self.admit_store(kv_store, context, &candidate, partition_id, now);
                    admitted_tokens[partition_index] += candidate.prompt;
                }
            }
        }

        kv_store.drain_ready();
        had_live_decode
            || (0..num_partitions as u16).any(|partition| kv_store.has_prefill_admit(partition))
    }

    fn complete_iteration(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        events: &mut Vec<WorkerEventCommon>,
        now: Time,
    ) {
        let num_partitions = kv_store.num_partitions();

        for partition in 0..num_partitions as u16 {
            let mut completed: Vec<RequestId> = Vec::new();

            // (a) live decodes each produced one token this iter.
            {
                let log_tokens = context.log_tokens();
                let members = kv_store.decode_members(partition);
                let mut store = context.requests.borrow_mut();
                for (rid, _current_kv) in members {
                    let record = &mut store[rid];
                    record.record_token(now, log_tokens);
                    if record.is_complete() {
                        context.stamp_stage(record, now, UnifiedStage::Done as u16);
                        completed.push(rid);
                    }
                }
            }

            // (b) bump KV for every live decode in this partition.
            kv_store.advance(AdvanceScope::WholePartition(partition), 1);

            // (c) resolved prefills → first token; complete now or enter decode.
            let mut to_finalize: Vec<(RequestId, u64, u32)> = Vec::new();
            {
                let log_tokens = context.log_tokens();
                let prefills = kv_store.prefill_admits(partition);
                let mut store = context.requests.borrow_mut();
                for rid in prefills {
                    let record = &mut store[rid];
                    record.prefill_processed = record.prompt_len;
                    record.record_first_token(now, log_tokens);
                    if record.is_complete() {
                        context.stamp_stage(record, now, UnifiedStage::Done as u16);
                        completed.push(rid);
                    } else {
                        let kv_len = u64::from(record.prompt_len + record.prefix_kv);
                        let remaining = record.decode_len.saturating_sub(record.tokens_emitted);
                        context.stamp_stage(record, now, UnifiedStage::Decode as u16);
                        to_finalize.push((rid, kv_len, remaining));
                    }
                }
            }
            for (rid, kv_len, remaining) in to_finalize {
                kv_store.commit_resident(rid, partition, kv_len, remaining);
            }
            kv_store.clear_prefill_admits(partition);

            // (d) emit + release completed requests.
            for rid in completed {
                kv_store.release(rid, partition);
                events.push(WorkerEventCommon::RequestComplete {
                    worker: context.id,
                    req: rid,
                });
            }

            kv_store.sample_submit(partition, now);
        }
    }

    #[inline]
    fn queued_requests(&self) -> u32 {
        self.policy.len() as u32
    }

    #[inline]
    fn cancel_pending(&mut self, rid: RequestId) -> bool {
        self.policy.remove(rid)
    }
}
