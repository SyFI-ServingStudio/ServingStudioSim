//! `ChunkedPrefillAdmission<P>` — the chunked-prefill lifecycle (matrix A
//! "ChunkedPrefill",
//! iter family). A prefill is reserved at its FULL footprint on first touch, then
//! its prompt lands into resident KV a `max_batch_tokens`-bounded chunk per iter;
//! the last chunk transitions it to decode. Swappable selection policy `P` orders
//! the pending prefills.
//!
//! Impl bound is `K: IterWorkerKv + ChunkedPrefillKv` (ref
//! `IterBatchKv + KvChunked`) — STRICTLY stronger than
//! `LocalPrefillDecodeAdmission`'s `K: IterWorkerKv` (interfaces doc §0:
//! "Admission = ChunkedPrefill bound K: ChunkedPrefillKv"). This is the M2
//! capability escalation in action: the lifecycle
//! that needs partial prefill realization declares it in its bound, and a layout that
//! cannot chunk simply does not compose with it. Reuses `IterBatchWorker` + `FullAttnKv` +
//! `UnifiedIterExecution` verbatim — only the admission is new
//! (`UnifiedIterExecution::build_iteration_input`
//! already reads `active_chunk_len`, so it is chunk-aware unchanged).
//!
//! `BatchPolicy` (Mix / SeparatePrefillPriority[NoInterleave]) is carried but only
//! Mix is faithfully modeled in v1; the Separate variants would gate whether decode
//! and prefill share an iteration (a batch-composition heuristic, not an interface
//! question). External calls verified: `prefill_fits_budget` is NOT reused (its whole-
//! prefill budget differs from a chunk cap); `RequestRecord` field reads,
//! `mark_admitted`, `record_first_token`/`record_token`/`is_complete`.

use crate::common::{RequestId, Time, UnifiedStage};
use crate::worker::config::BatchPolicy;
use crate::worker::types::{WorkerEventCommon, WorkerMsgCommon};

use super::super::kv::{ChunkedPrefillKv, IterWorkerKv, KvStore};
use super::super::shared::advance_scope::AdvanceScope;
use super::super::shared::context::WorkerContext;
use super::policy::{AdmissionCandidate, PendingOrderPolicy};
use super::IterAdmission;

pub struct ChunkedPrefillAdmission<P: PendingOrderPolicy> {
    /// The sole prefill whose prompt was only partially consumed by an earlier
    /// iteration. A partition may batch several complete prefills in one iteration,
    /// but the token-budget boundary can leave at most one cross-iteration chunk.
    active_chunk: Option<AdmissionCandidate>,
    /// Fresh, not-yet-started prefills. The policy remains their single membership
    /// owner and decides which request starts after `active_chunk` has priority.
    policy: P,
    policy_ctx: P::Context,
    /// Hard per-iter token cap that chunks prefills to fit (NOT the soft
    /// whole-prefill budget of `LocalPrefillDecodeAdmission`).
    max_batch_tokens: u32,
    batch_policy: BatchPolicy,
}

impl<P: PendingOrderPolicy> ChunkedPrefillAdmission<P> {
    pub fn new(
        policy: P,
        policy_ctx: P::Context,
        max_batch_tokens: u32,
        batch_policy: BatchPolicy,
    ) -> Self {
        Self {
            active_chunk: None,
            policy,
            policy_ctx,
            max_batch_tokens,
            batch_policy,
        }
    }
}

impl<P: PendingOrderPolicy, K: KvStore + IterWorkerKv + ChunkedPrefillKv> IterAdmission<K>
    for ChunkedPrefillAdmission<P>
{
    type Msg = WorkerMsgCommon;
    type Event = WorkerEventCommon;

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
        let had_decode = kv_store.has_live_decode(0);
        let decode_tokens = kv_store.live_decode_count(0);
        // Mix: the chunk budget is what remains after this iter's live decodes take
        // 1 token each. (Separate* would instead run prefill-only / decode-only iters
        // — v1 models Mix; the field is carried for that follow-up.)
        let _ = self.batch_policy;
        let prefill_budget = self.max_batch_tokens.saturating_sub(decode_tokens);

        let mut admitted_tokens = 0u32;
        // The one cross-iteration chunk is already fully reserved and cannot be
        // preempted by a newly arrived, higher-ranked policy candidate.
        if prefill_budget > 0 {
            if let Some(candidate) = self.active_chunk.take() {
                let rid = candidate.request;
                let processed = context.requests.borrow()[rid].prefill_processed;
                let remaining_prompt = candidate.prompt.saturating_sub(processed);
                let chunk = remaining_prompt.min(prefill_budget);
                {
                    let mut store = context.requests.borrow_mut();
                    let record = &mut store[rid];
                    record.prefix_kv = processed;
                    record.active_chunk_len = chunk;
                }
                kv_store.append_prefill_chunk(0, rid, chunk);
                admitted_tokens += chunk;
                if chunk < remaining_prompt {
                    self.active_chunk = Some(candidate);
                }
            }
        }

        // Fill the rest of this iteration with fresh prefills in policy order. Several
        // may finish in full; only the final budget-crossing request becomes the sole
        // continuation. A declined head stays in the policy rather than being skipped.
        while self.active_chunk.is_none() && admitted_tokens < prefill_budget {
            let Some(candidate) = self.policy.peek() else {
                break;
            };
            let footprint =
                kv_store.footprint(candidate.request, candidate.prompt, candidate.decode);
            if !kv_store.fits(0, &footprint) {
                break;
            }

            self.policy.pop(&mut self.policy_ctx);
            kv_store.reserve(candidate.request, 0, footprint);
            let mut store = context.requests.borrow_mut();
            store.mark_admitted(candidate.request);
            let record = &mut store[candidate.request];
            context.stamp_stage(record, now, UnifiedStage::Prefill as u16);
            record.prefix_kv = 0;
            let chunk = candidate.prompt.min(prefill_budget - admitted_tokens);
            record.active_chunk_len = chunk;
            drop(store);

            kv_store.append_prefill_chunk(0, candidate.request, chunk);
            admitted_tokens += chunk;
            if chunk < candidate.prompt {
                self.active_chunk = Some(candidate);
            }
        }

        had_decode || kv_store.has_prefill_admit(0)
    }

    fn complete_iteration(
        &mut self,
        kv_store: &mut K,
        context: &WorkerContext,
        events: &mut Vec<WorkerEventCommon>,
        now: Time,
    ) {
        let mut completed: Vec<RequestId> = Vec::new();

        // (a) live decodes each produced one token this iter.
        {
            let log_tokens = context.log_tokens();
            let members = kv_store.decode_members(0);
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

        // (b) bump KV for every live decode.
        kv_store.advance(AdvanceScope::WholePartition(0), 1);

        // (c) advance this iteration's prefills. Several may finish in full, while
        // at most the final token-budget-crossing request remains as `active_chunk`.
        let mut to_finish: Vec<(RequestId, u64, u32)> = Vec::new();
        {
            let log_tokens = context.log_tokens();
            let admits = kv_store.prefill_admits(0);
            let mut store = context.requests.borrow_mut();
            for rid in admits {
                let record = &mut store[rid];
                record.prefill_processed += record.active_chunk_len;
                if record.prefill_processed < record.prompt_len {
                    continue; // still chunking; stays pending, no first token yet
                }
                // Last chunk: prompt fully resident → first token, enter decode.
                record.record_first_token(now, log_tokens);
                let remaining = record.decode_len.saturating_sub(record.tokens_emitted);
                let complete = record.is_complete();
                context.stamp_stage(
                    record,
                    now,
                    if complete {
                        UnifiedStage::Done
                    } else {
                        UnifiedStage::Decode
                    } as u16,
                );
                // ALWAYS finish into the decode set (unlike barebone, chunks already
                // made the prompt resident, so a decode_len==0 prefill must round-trip
                // through the decode set to have its resident KV released below —
                // `kv_store.release` recomputes `current_kv` from that set).
                to_finish.push((rid, u64::from(record.prompt_len), remaining));
                if complete {
                    completed.push(rid);
                }
            }
        }
        for (rid, initial_kv, remaining) in to_finish {
            kv_store.finish_chunked_prefill(0, rid, initial_kv, remaining);
        }
        kv_store.clear_prefill_admits(0);

        // (d) release + emit completed (decodes and zero-decode prefills alike).
        for rid in completed {
            kv_store.release(rid, 0);
            events.push(WorkerEventCommon::RequestComplete {
                worker: context.id,
                req: rid,
            });
        }

        kv_store.sample_submit(0, now);
    }

    /// Only not-yet-started arrivals are queued. `active_chunk` is already admitted
    /// and is counted by the KV axis as active.
    #[inline]
    fn queued_requests(&self) -> u32 {
        self.policy.len() as u32
    }

    fn cancel_pending(&mut self, rid: RequestId) -> bool {
        if self.policy.remove(rid) {
            return true;
        }

        // An active chunk is admitted, so report `false` and let
        // `IterBatchWorker` run
        // `kv_store.release_external`; only clear this lifecycle's continuation pointer.
        if self
            .active_chunk
            .is_some_and(|candidate| candidate.request == rid)
        {
            self.active_chunk = None;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use crate::common::{PoolId, RequestId, Time, WorkerId};
    use crate::test_helpers::shared_with;
    use crate::worker::admission_helpers::KvAdmission;

    use super::super::super::kv::{FullAttnKv, IterWorkerKv};
    use super::super::super::shared::context::WorkerContext;
    use super::super::policy::{FifoOrder, ShortestJobFirst};
    use super::super::IterAdmission;
    use super::*;

    fn test_ctx(requests: &[(u32, u32, u32)]) -> WorkerContext {
        WorkerContext {
            id: WorkerId(0),
            pool: PoolId(0),
            requests: shared_with(requests),
            log_output_token_times: false,
            log_stage_transitions: false,
        }
    }

    fn test_kv() -> FullAttnKv {
        FullAttnKv::new(1, 1_000, KvAdmission::Strict, None)
    }

    #[test]
    fn one_iteration_can_admit_many_prefills_but_leave_only_one_active_chunk() {
        let context = test_ctx(&[(0, 3, 2), (1, 3, 2), (2, 5, 2), (3, 1, 2)]);
        let mut kv_store = test_kv();
        let mut admission = ChunkedPrefillAdmission::new(FifoOrder::new(), (), 8, BatchPolicy::Mix);
        for request in 0..4 {
            admission.accept_message(
                &mut kv_store,
                WorkerMsgCommon::Request(RequestId(request)),
                &context,
            );
        }

        assert!(admission.form_batch(&mut kv_store, &context, Time::ZERO));

        assert_eq!(
            kv_store.prefill_admits(0),
            vec![RequestId(0), RequestId(1), RequestId(2)]
        );
        assert_eq!(
            admission.active_chunk.map(|candidate| candidate.request),
            Some(RequestId(2))
        );
        assert_eq!(admission.policy.len(), 1);
        let store = context.requests.borrow();
        assert_eq!(store[RequestId(0)].active_chunk_len, 3);
        assert_eq!(store[RequestId(1)].active_chunk_len, 3);
        assert_eq!(store[RequestId(2)].active_chunk_len, 2);
    }

    #[test]
    fn active_chunk_is_not_preempted_by_a_new_shorter_job() {
        let context = test_ctx(&[(0, 10, 2), (1, 1, 2)]);
        let mut kv_store = test_kv();
        let mut admission =
            ChunkedPrefillAdmission::new(ShortestJobFirst::new(), (), 4, BatchPolicy::Mix);
        admission.accept_message(
            &mut kv_store,
            WorkerMsgCommon::Request(RequestId(0)),
            &context,
        );
        assert!(admission.form_batch(&mut kv_store, &context, Time::ZERO));
        admission.complete_iteration(&mut kv_store, &context, &mut Vec::new(), Time::from_ms(1.0));

        admission.accept_message(
            &mut kv_store,
            WorkerMsgCommon::Request(RequestId(1)),
            &context,
        );
        assert!(admission.form_batch(&mut kv_store, &context, Time::from_ms(1.0)));

        assert_eq!(kv_store.prefill_admits(0), vec![RequestId(0)]);
        assert_eq!(
            admission.active_chunk.map(|candidate| candidate.request),
            Some(RequestId(0))
        );
        assert_eq!(admission.policy.len(), 1);
    }
}
