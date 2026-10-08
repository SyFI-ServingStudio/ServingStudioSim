//! `SessionTierWorker` — the hybrid chunked-prefill worker with DRAM/SSD prefix
//! tiers behind each attention DP rank's HBM cache, and decode run elsewhere.
//!
//! It wraps the whole-iteration shell and acts only between its messages:
//!
//! - **Tier reads.** The admission owns the tiers (`admission/prefix_fetch.rs`):
//!   when a request reaches the head of its rank's queue and that rank's tiers
//!   hold more of its context than HBM does, it leaves the queue for a FIFO
//!   read of the difference and returns once the read lands. Sessions stick to
//!   the rank they last ran on. This worker wakes the shell when a read lands.
//! - **Completion.** The finished context is written through to its rank's
//!   tiers. With `external_decode`, a request completes at its first token and
//!   its context, every output but the last, is restored into HBM as a decode
//!   instance would hand it back.
//!
//! The pipeline head (`workers/pipeline`) does the same with one tier set per
//! pipeline. Without tiers or external decode it is the bare shell.

use std::collections::HashMap;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{RequestId, Time, WorkerId};
use crate::worker::iter_worker::IterWorker;
use crate::worker::kv::PrefixKv;
use crate::worker::types::WorkerStatus;
use crate::worker::{WorkerEventCommon, WorkerMsgCommon};

use super::iter_batch_worker::HybridChunkedPrefillShell;

pub struct SessionTierWorker<M: IterwiseUnifiedModel> {
    inner: HybridChunkedPrefillShell<M>,
    /// Whether the admission reads from DRAM/SSD tiers (finished contexts are
    /// then written through).
    tiers: bool,
    /// With `external_decode`: each in-flight request's original target
    /// outputs, while it runs with a target of one.
    external_outputs: Option<HashMap<RequestId, u32>>,
}

impl<M: IterwiseUnifiedModel> SessionTierWorker<M> {
    pub(super) fn new(
        inner: HybridChunkedPrefillShell<M>,
        tiers: bool,
        external_decode: bool,
    ) -> Self {
        Self {
            inner,
            tiers,
            external_outputs: external_decode.then(HashMap::new),
        }
    }

    #[cfg(test)]
    pub(super) fn shell_mut(&mut self) -> &mut HybridChunkedPrefillShell<M> {
        &mut self.inner
    }

    fn on_request(&mut self, request: RequestId) {
        if let Some(outputs) = &mut self.external_outputs {
            let requests = self.inner.requests().clone();
            let mut store = requests.borrow_mut();
            let definition = &mut store[request].request.definition;
            outputs.insert(request, definition.target_output_tokens);
            definition.target_output_tokens = 1;
        }
        self.inner.enqueue(WorkerMsgCommon::Request(request));
    }

    fn on_request_complete(&mut self, request: RequestId, at: Time) {
        let external_outputs = self
            .external_outputs
            .as_mut()
            .and_then(|outputs| outputs.remove(&request));
        if !self.tiers && external_outputs.is_none_or(|outputs| outputs <= 1) {
            return;
        }
        let (session_input, context_tokens) = {
            let store = self.inner.requests().borrow();
            let definition = &store[request].request.definition;
            let outputs = external_outputs.unwrap_or(definition.target_output_tokens);
            // The last output token has no KV yet: the next round computes it.
            (
                definition.session,
                u64::from(definition.session.declared_prefix_tokens())
                    + u64::from(definition.prompt_tokens)
                    + u64::from(outputs.saturating_sub(1)),
            )
        };
        let Some(session_id) = session_input.session_id() else {
            return;
        };
        let (kv_store, admission) = self.inner.kv_and_admission();
        let rank = admission
            .session_partition(session_id)
            .expect("a session worker with tiers or external decode keeps sessions sticky");
        if external_outputs.is_some_and(|outputs| outputs > 1) {
            kv_store.restore_prefix(request, rank, session_id, context_tokens, at);
        }
        if self.tiers {
            admission.store_session_context(rank, session_id, context_tokens);
        }
    }
}

impl<M: IterwiseUnifiedModel> IterWorker for SessionTierWorker<M> {
    type Msg = WorkerMsgCommon;
    type Event = WorkerEventCommon;

    fn id(&self) -> WorkerId {
        self.inner.id()
    }

    fn enqueue(&mut self, msg: Self::Msg) {
        let WorkerMsgCommon::Request(request) = msg;
        self.on_request(request);
    }

    fn tick(&mut self, now: Time, events: &mut Vec<Self::Event>) -> Option<Time> {
        let first = events.len();
        let wakeup = self.inner.tick(now, events);
        let completed: Vec<RequestId> = events[first..]
            .iter()
            .map(|event| {
                let WorkerEventCommon::RequestComplete { req, .. } = event;
                *req
            })
            .collect();
        for request in completed {
            self.on_request_complete(request, now);
        }
        let (_, admission) = self.inner.kv_and_admission();
        let next_read = admission.next_prefix_read();
        [wakeup, next_read].into_iter().flatten().min()
    }

    fn status(&self) -> WorkerStatus {
        let mut status = self.inner.status();
        status.queued_requests += self.inner.admission().reading_requests();
        status
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::common::{PoolId, SessionInput, SharedRequests};
    use crate::test_helpers::{shared_with, test_cluster, FakeModel};
    use crate::worker::kv::PrefixTierSpec;
    use crate::worker::types::WorkerConfig;
    use crate::worker::workers::iter::build_hybrid_chunked_prefill_worker;

    fn as_session(
        store: &SharedRequests,
        id: u32,
        session_id: u32,
        declared: u32,
        arrival_ms: f64,
    ) {
        let mut store = store.borrow_mut();
        let request = &mut store[RequestId(id)].request;
        request.definition.session = SessionInput::Session {
            session_id,
            session_start_time: Time::ZERO,
            declared_prefix_tokens: declared,
        };
        request.core.arrival_time = Time::from_ms(arrival_ms);
    }

    /// 1 B/token, 120 tokens of HBM per rank; DRAM holds 1000 tokens and reads
    /// 1e5 B/s, so 50 tokens take 0.5 ms.
    fn worker(
        store: SharedRequests,
        dp_groups: u16,
        external_decode: bool,
    ) -> SessionTierWorker<FakeModel> {
        build_hybrid_chunked_prefill_worker(
            WorkerId(0),
            "main",
            Arc::new(FakeModel { ms: 1.0, dp_groups }),
            store,
            WorkerConfig {
                max_batch_tokens: Some(256),
                attn_kv_bytes: 120,
                prefix_tiers: [
                    Some(PrefixTierSpec {
                        name: "dram",
                        capacity_gb_per_gpu: 1e-6,
                        read_gb_per_s_per_gpu: 1e-4,
                    }),
                    None,
                ],
                external_decode,
                ..WorkerConfig::default()
            },
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        )
    }

    fn run_until(
        worker: &mut SessionTierWorker<FakeModel>,
        from: Time,
        until: Time,
        events: &mut Vec<WorkerEventCommon>,
    ) {
        let mut now = from;
        while now <= until {
            let Some(next) = worker.tick(now, events) else {
                break;
            };
            now = next.max(now + Time::from_ms(0.1));
        }
    }

    fn hit(store: &SharedRequests, id: u32) -> Option<u32> {
        store.borrow()[RequestId(id)]
            .telemetry
            .prefix_cache_hit_tokens
    }

    #[test]
    fn a_rank_reads_an_evicted_session_back_from_its_dram() {
        // Session 7 round 0 (50 tokens), a 100-token prompt that evicts it from
        // HBM, then round 1 declaring 51 tokens, 50 of them with KV in DRAM.
        let store = shared_with(&[(0, 50, 1), (1, 100, 1), (2, 4, 1)]);
        as_session(&store, 0, 7, 0, 0.0);
        as_session(&store, 2, 7, 51, 10.0);
        let mut worker = worker(store.clone(), 1, false);
        let mut events = Vec::new();
        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        run_until(&mut worker, Time::ZERO, Time::from_ms(4.0), &mut events);
        worker.enqueue(WorkerMsgCommon::Request(RequestId(1)));
        run_until(
            &mut worker,
            Time::from_ms(4.0),
            Time::from_ms(9.0),
            &mut events,
        );
        assert_eq!(events.len(), 2);

        worker.enqueue(WorkerMsgCommon::Request(RequestId(2)));
        assert_eq!(worker.status().queued_requests, 1);
        assert_eq!(
            worker.tick(Time::from_ms(10.0), &mut events),
            Some(Time::from_ms(10.5))
        );
        run_until(
            &mut worker,
            Time::from_ms(10.5),
            Time::from_ms(20.0),
            &mut events,
        );
        assert_eq!(events.len(), 3);
        assert_eq!(hit(&store, 2), Some(50));
    }

    #[test]
    fn decode_elsewhere_on_a_dp_rank_keeps_the_session_on_it() {
        // Round 0 of session 7 has 5 target outputs and lands on rank 0; a
        // standalone prompt takes rank 1; round 1 finds the 54 tokens with KV
        // (all but the last output) back on rank 0.
        let store = shared_with(&[(0, 50, 5), (1, 8, 1), (2, 4, 1)]);
        as_session(&store, 0, 7, 0, 0.0);
        as_session(&store, 2, 7, 55, 10.0);
        let mut worker = worker(store.clone(), 2, true);
        let mut events = Vec::new();
        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        worker.enqueue(WorkerMsgCommon::Request(RequestId(1)));
        run_until(&mut worker, Time::ZERO, Time::from_ms(9.0), &mut events);
        assert_eq!(events.len(), 2);
        assert_eq!(
            store.borrow()[RequestId(0)].progress.output_tokens_emitted,
            1
        );
        worker.enqueue(WorkerMsgCommon::Request(RequestId(2)));
        run_until(
            &mut worker,
            Time::from_ms(10.0),
            Time::from_ms(20.0),
            &mut events,
        );
        assert_eq!(events.len(), 3);
        assert_eq!(hit(&store, 2), Some(54));
    }
}
