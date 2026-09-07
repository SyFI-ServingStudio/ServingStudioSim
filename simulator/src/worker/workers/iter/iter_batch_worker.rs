//! `IterBatchWorker` — the whole-iteration L5 shell.
//!
//! The shell owns cadence and cross-axis sequencing. KV, Admission, and Execution
//! are statically composed; the FSM itself is deliberately not an axis.

use crate::common::{RequestId, Time, WorkerId};
use crate::worker::admission::{
    ChunkedPrefillAdmission, IterAdmission, LocalPrefillDecodeAdmission, PendingOrder,
    PrefillHandoffAdmission, SpeculativeDecodeCompletion,
};
use crate::worker::execution::{
    IterModelExecution, SpeculativeIterExecution, UnifiedIterExecution,
};
use crate::worker::iter_worker::IterWorker;
use crate::worker::kv::{FullAttnKv, HybridGdnKv, IterWorkerKv};
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::{
    BatchFsmState, IterBatchPlan, IterCursor, WorkerFsmState, WorkerStatus,
};

struct IterationFsm {
    worker_state: WorkerFsmState,
    batch_state: BatchFsmState,
    iteration: u32,
}

impl IterationFsm {
    fn new() -> Self {
        Self {
            worker_state: WorkerFsmState::Idle,
            batch_state: BatchFsmState {
                cursor: IterCursor::Done,
                compute_end: Time::ZERO,
            },
            iteration: 0,
        }
    }
}

pub struct IterBatchWorker<K, A, E>
where
    K: IterWorkerKv,
    A: IterAdmission<K>,
    E: IterModelExecution<K>,
{
    context: WorkerContext,
    kv_store: K,
    admission: A,
    execution: E,
    input: E::Input,
    batch_plan: IterBatchPlan,
    iteration_fsm: IterationFsm,
}

/// Compatibility name retained for L6 factories and deployment selectors.
///
/// The pending order is `PendingOrder` (a runtime dispatcher) rather than a
/// concrete policy so the queue discipline stays a preset knob without fanning
/// this alias — and every alias below it — out by one type per policy. The
/// policy axis is still static: see `pending_policy_is_a_composition_axis_for_iter_lifecycles`.
pub type BareboneWorker<M> =
    IterBatchWorker<FullAttnKv, LocalPrefillDecodeAdmission<PendingOrder>, UnifiedIterExecution<M>>;
/// Multi-partition HP/DP recipe uses the same whole-iteration shell.
pub type HpUnifiedWorker<M> = BareboneWorker<M>;
pub type ChunkedPrefillWorker<M> =
    IterBatchWorker<FullAttnKv, ChunkedPrefillAdmission<PendingOrder>, UnifiedIterExecution<M>>;
/// `ChunkedPrefillWorker` with a speculating decode engine. Two axes change
/// together because they are two halves of one fact: the lifecycle's decode
/// completion retires an accepted chain instead of one token, and the execution
/// adapter submits the matching verify width to the model. KV, policy, and shell
/// are unchanged — speculation is not a cadence.
pub type SpeculativeWorker<M> = IterBatchWorker<
    FullAttnKv,
    ChunkedPrefillAdmission<PendingOrder, SpeculativeDecodeCompletion>,
    SpeculativeIterExecution<M>,
>;
/// Barebone on every axis but KV: a hybrid arch's per-request recurrent state
/// shares the attention capacity with its per-token KV. Only the store differs.
pub type Qwen36HybridWorker<M> = IterBatchWorker<
    HybridGdnKv,
    LocalPrefillDecodeAdmission<PendingOrder>,
    UnifiedIterExecution<M>,
>;
pub type PdPrefillWorker<M> =
    IterBatchWorker<FullAttnKv, PrefillHandoffAdmission<PendingOrder>, UnifiedIterExecution<M>>;

impl<K, A, E> IterBatchWorker<K, A, E>
where
    K: IterWorkerKv,
    A: IterAdmission<K>,
    E: IterModelExecution<K>,
{
    pub(super) fn from_components(
        context: WorkerContext,
        kv_store: K,
        admission: A,
        execution: E,
    ) -> Self {
        Self {
            context,
            kv_store,
            admission,
            execution,
            input: Default::default(),
            batch_plan: IterBatchPlan::default(),
            iteration_fsm: IterationFsm::new(),
        }
    }

    pub fn enqueue(&mut self, msg: A::Msg) {
        self.admission
            .accept_message(&mut self.kv_store, msg, &self.context);
    }

    pub fn tick(&mut self, now: Time, events: &mut Vec<A::Event>) -> Option<Time> {
        use IterCursor::{Computing, Done, NotStarted};
        use WorkerFsmState::{Active, Idle};

        loop {
            match self.iteration_fsm.worker_state {
                Idle => {
                    if !self.form_batch(now) {
                        break;
                    }
                    self.iteration_fsm.worker_state = Active;
                    self.iteration_fsm.batch_state.cursor = NotStarted;
                    self.iteration_fsm.batch_state.compute_end = Time::ZERO;
                }
                Active => match self.iteration_fsm.batch_state.cursor {
                    NotStarted => {
                        let compute_end = self.start_iteration(now);
                        self.iteration_fsm.batch_state.cursor = Computing;
                        self.iteration_fsm.batch_state.compute_end = compute_end;
                        break;
                    }
                    Computing if now < self.iteration_fsm.batch_state.compute_end => break,
                    Computing => {
                        self.iteration_fsm.batch_state.cursor = Done;
                    }
                    Done => {
                        self.admission.complete_iteration(
                            &mut self.kv_store,
                            &self.context,
                            &self.batch_plan,
                            events,
                            now,
                        );
                        self.iteration_fsm.worker_state = Idle;
                    }
                },
            }
        }

        self.next_wakeup(now)
    }

    fn form_batch(&mut self, now: Time) -> bool {
        if !self
            .admission
            .form_batch(&mut self.kv_store, &self.context, &mut self.batch_plan, now)
        {
            return false;
        }
        self.iteration_fsm.iteration += 1;
        true
    }

    fn start_iteration(&mut self, now: Time) -> Time {
        self.execution.build_iteration_input(
            &self.kv_store,
            &self.context.requests,
            &self.batch_plan,
            &mut self.input,
        );
        let cost = self.execution.evaluate_iteration(
            &self.input,
            self.iteration_fsm.iteration as u64,
            now,
        );
        now + cost
    }

    fn next_wakeup(&self, now: Time) -> Option<Time> {
        match self.iteration_fsm.worker_state {
            WorkerFsmState::Idle => {
                let status = self.status();
                (status.queued_requests > 0 || status.active_requests > 0).then_some(now)
            }
            WorkerFsmState::Active => match self.iteration_fsm.batch_state.cursor {
                IterCursor::NotStarted | IterCursor::Done => Some(now),
                IterCursor::Computing => Some(self.iteration_fsm.batch_state.compute_end),
            },
        }
    }

    pub fn status(&self) -> WorkerStatus {
        let active_requests = (0..self.kv_store.num_partitions() as u16)
            .map(|partition| self.kv_store.status_active(partition))
            .sum();
        WorkerStatus {
            queued_requests: self.admission.queued_requests(),
            active_requests,
        }
    }

    pub fn release_request(&mut self, request: RequestId, current_kv: u64) -> Option<u16> {
        if self.admission.cancel_pending(request) {
            return None;
        }
        self.kv_store.release_external(request, current_kv)
    }

    pub fn id(&self) -> WorkerId {
        self.context.id
    }
}

impl<K, A, E> IterWorker for IterBatchWorker<K, A, E>
where
    K: IterWorkerKv,
    A: IterAdmission<K>,
    E: IterModelExecution<K>,
{
    type Msg = A::Msg;
    type Event = A::Event;

    fn id(&self) -> WorkerId {
        self.context.id
    }

    fn enqueue(&mut self, msg: Self::Msg) {
        IterBatchWorker::enqueue(self, msg);
    }

    fn tick(&mut self, now: Time, events: &mut Vec<Self::Event>) -> Option<Time> {
        IterBatchWorker::tick(self, now, events)
    }

    fn status(&self) -> WorkerStatus {
        IterBatchWorker::status(self)
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use std::sync::Arc;

    use super::*;
    use crate::arch::contract::IterwiseUnifiedModel;
    use crate::common::{PoolId, SessionInput, SharedRequests, UnifiedStage};
    use crate::test_helpers::{shared_with, test_cluster, FakeModel};
    use crate::worker::admission::{FifoOrder, ShortestJobFirst};
    use crate::worker::config::BoundedFutureKvAdmissionConfig;
    use crate::worker::kv::{PrefixCacheConfig, PrefixKv};
    use crate::worker::types::{WorkerConfig, WorkerEventCommon, WorkerMsgCommon};
    use crate::worker::workers::iter::{
        build_barebone_worker, build_chunked_prefill_worker, build_hp_worker,
    };
    use crate::worker::{BatchPolicy, DecodeRetractionPolicy, KvAdmissionConfig};

    fn assert_iter_worker<W: IterWorker>() {}

    #[test]
    fn pending_policy_is_a_composition_axis_for_iter_lifecycles() {
        assert_iter_worker::<
            IterBatchWorker<
                FullAttnKv,
                LocalPrefillDecodeAdmission<FifoOrder>,
                UnifiedIterExecution<FakeModel>,
            >,
        >();
        assert_iter_worker::<
            IterBatchWorker<
                FullAttnKv,
                LocalPrefillDecodeAdmission<ShortestJobFirst>,
                UnifiedIterExecution<FakeModel>,
            >,
        >();
        assert_iter_worker::<
            IterBatchWorker<
                FullAttnKv,
                PrefillHandoffAdmission<FifoOrder>,
                UnifiedIterExecution<FakeModel>,
            >,
        >();
        assert_iter_worker::<
            IterBatchWorker<
                FullAttnKv,
                PrefillHandoffAdmission<ShortestJobFirst>,
                UnifiedIterExecution<FakeModel>,
            >,
        >();
    }

    fn run_to_quiescence<M: IterwiseUnifiedModel>(
        worker: &mut BareboneWorker<M>,
        max_steps: u64,
    ) -> Vec<WorkerEventCommon> {
        let mut events = Vec::new();
        for step in 0..max_steps {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        events
    }

    #[test]
    fn single_request_prefill_then_decode_completes() {
        let store = shared_with(&[(0, 16, 3)]);
        let mut worker = build_barebone_worker(
            WorkerId(0),
            "main",
            Arc::new(FakeModel::for_ms(1.0)),
            Rc::clone(&store),
            WorkerConfig::default(),
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        );
        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));

        let events = run_to_quiescence(&mut worker, 50);
        assert_eq!(
            events,
            vec![WorkerEventCommon::RequestComplete {
                worker: WorkerId(0),
                req: RequestId(0),
            }]
        );
        let store = store.borrow();
        let record = &store[RequestId(0)];
        assert!(record.lifecycle.completed);
        assert_eq!(record.progress.output_tokens_emitted, 3);
        assert!(record.telemetry.first_output_time.is_some());
    }

    #[test]
    fn production_admission_prefers_the_oldest_session_start() {
        let store = shared_with(&[(0, 8, 2), (1, 8, 2)]);
        {
            let mut requests = store.borrow_mut();
            requests[RequestId(0)].request.definition.session = SessionInput::Session {
                session_id: 10,
                session_start_time: Time::from_ms_u64(10),
                declared_prefix_tokens: 0,
            };
            requests[RequestId(1)].request.definition.session = SessionInput::Session {
                session_id: 1,
                session_start_time: Time::from_ms_u64(1),
                declared_prefix_tokens: 0,
            };
        }
        let mut worker = worker_with(Rc::clone(&store), config_with_budget(8));

        // The older session arrives at this worker second, but starts first.
        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        worker.enqueue(WorkerMsgCommon::Request(RequestId(1)));
        worker.tick(Time::from_ms_u64(20), &mut Vec::new());

        let requests = store.borrow();
        assert!(!requests[RequestId(0)].lifecycle.admitted);
        assert!(requests[RequestId(1)].lifecycle.admitted);
    }

    #[test]
    fn three_requests_all_complete() {
        let store = shared_with(&[(0, 8, 2), (1, 8, 2), (2, 8, 2)]);
        let mut worker = build_barebone_worker(
            WorkerId(0),
            "main",
            Arc::new(FakeModel::for_ms(1.0)),
            Rc::clone(&store),
            WorkerConfig::default(),
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        );
        for request in [0, 1, 2] {
            worker.enqueue(WorkerMsgCommon::Request(RequestId(request)));
        }

        let events = run_to_quiescence(&mut worker, 200);
        assert_eq!(events.len(), 3);
        let store = store.borrow();
        for request in [0, 1, 2] {
            assert!(
                store[RequestId(request)].lifecycle.completed,
                "req {request} should complete"
            );
        }
    }

    #[test]
    fn completed_session_kv_is_reused_without_exceeding_total_attention_capacity() {
        let store = shared_with(&[(0, 20, 2), (1, 30, 2)]);
        {
            let mut requests = store.borrow_mut();
            for request in [RequestId(0), RequestId(1)] {
                requests[request].request.definition.session = SessionInput::Session {
                    session_id: 7,
                    session_start_time: Time::ZERO,
                    declared_prefix_tokens: 100,
                };
            }
        }
        let config = WorkerConfig {
            attn_kv_bytes: 1_000,
            ..WorkerConfig::default()
        };
        let mut worker = worker_with(Rc::clone(&store), config);
        let mut events = Vec::new();

        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        for step in 0..100 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(
            store.borrow()[RequestId(0)]
                .progress
                .prefill_tokens_processed,
            120,
            "a cold session recomputes its declared prefix"
        );
        assert_eq!(
            store.borrow()[RequestId(0)]
                .telemetry
                .prefix_cache_hit_tokens,
            Some(0),
            "the first request records a resolved cache miss"
        );

        worker.enqueue(WorkerMsgCommon::Request(RequestId(1)));
        for step in 100..200 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(
            store.borrow()[RequestId(1)]
                .progress
                .prefill_tokens_processed,
            30,
            "the same session consumes retained KV and computes only fresh prompt tokens"
        );
        assert_eq!(
            store.borrow()[RequestId(1)]
                .telemetry
                .prefix_cache_hit_tokens,
            Some(100),
            "the second request records the worker-local retained-prefix hit"
        );
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn disabled_prefix_cache_is_a_no_reuse_baseline() {
        let store = shared_with(&[(0, 20, 2), (1, 30, 2)]);
        {
            let mut requests = store.borrow_mut();
            for request in [RequestId(0), RequestId(1)] {
                requests[request].request.definition.session = SessionInput::Session {
                    session_id: 7,
                    session_start_time: Time::ZERO,
                    declared_prefix_tokens: 100,
                };
            }
        }
        let config = WorkerConfig {
            attn_kv_bytes: 1_000,
            prefix_cache: PrefixCacheConfig::Disabled,
            ..WorkerConfig::default()
        };
        let mut worker = worker_with(Rc::clone(&store), config);
        let mut events = Vec::new();

        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        for step in 0..100 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        worker.enqueue(WorkerMsgCommon::Request(RequestId(1)));
        for step in 100..200 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }

        assert_eq!(
            store.borrow()[RequestId(1)]
                .progress
                .prefill_tokens_processed,
            130,
            "disabled mode recomputes the declared prefix for every request"
        );
        assert_eq!(events.len(), 2);
    }

    fn config_with_budget(max_batch_tokens: u32) -> WorkerConfig {
        WorkerConfig {
            max_batch_tokens: Some(max_batch_tokens),
            ..WorkerConfig::default()
        }
    }

    fn worker_with(store: SharedRequests, config: WorkerConfig) -> BareboneWorker<FakeModel> {
        build_barebone_worker(
            WorkerId(0),
            "main",
            Arc::new(FakeModel::for_ms(1.0)),
            store,
            config,
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        )
    }

    #[test]
    fn budget_admits_multiple_prefills_in_one_iter() {
        let store = shared_with(&[(0, 8, 0), (1, 8, 0), (2, 8, 0), (3, 8, 0)]);
        let mut worker = worker_with(store, config_with_budget(20));
        for request in [0, 1, 2, 3] {
            worker.enqueue(WorkerMsgCommon::Request(RequestId(request)));
        }

        worker.form_batch(Time::ZERO);
        assert_eq!(worker.kv_store.prefill_admits(0).count(), 2);
        assert_eq!(worker.admission.queued_requests(), 2);
    }

    #[test]
    fn none_budget_admits_every_fitting_prefill() {
        let store = shared_with(&[(0, 8, 0), (1, 8, 0), (2, 8, 0)]);
        let mut worker = worker_with(store, WorkerConfig::default());
        for request in [0, 1, 2] {
            worker.enqueue(WorkerMsgCommon::Request(RequestId(request)));
        }

        worker.form_batch(Time::ZERO);
        assert_eq!(worker.kv_store.prefill_admits(0).count(), 3);
        assert_eq!(worker.admission.queued_requests(), 0);
    }

    #[test]
    fn budget_force_admits_single_overlong_prefill() {
        let store = shared_with(&[(0, 10, 0), (1, 10, 0)]);
        let mut worker = worker_with(store, config_with_budget(4));
        for request in [0, 1] {
            worker.enqueue(WorkerMsgCommon::Request(RequestId(request)));
        }

        worker.form_batch(Time::ZERO);
        assert_eq!(worker.kv_store.prefill_admits(0).count(), 1);
        assert_eq!(worker.admission.queued_requests(), 1);
    }

    fn hp_worker(
        store: SharedRequests,
        dp_groups: u16,
        config: WorkerConfig,
    ) -> HpUnifiedWorker<FakeModel> {
        build_hp_worker(
            WorkerId(0),
            "main",
            Arc::new(FakeModel { ms: 1.0, dp_groups }),
            store,
            config,
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        )
    }

    #[test]
    fn chunked_prefill_exposes_two_real_8192_token_iterations() {
        let store = shared_with(&[(0, 16_384, 2)]);
        let mut worker = build_chunked_prefill_worker(
            WorkerId(0),
            "main",
            Arc::new(FakeModel::for_ms(1.0)),
            Rc::clone(&store),
            WorkerConfig {
                max_batch_tokens: Some(8_192),
                ..WorkerConfig::default()
            },
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        );
        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));

        assert!(worker.form_batch(Time::ZERO));
        let mut first_input = Default::default();
        worker.execution.build_iteration_input(
            &worker.kv_store,
            &worker.context.requests,
            &worker.batch_plan,
            &mut first_input,
        );
        assert_eq!(first_input.groups[0].prefill_chunk_pairs, [(0, 8_192)]);
        worker.admission.complete_iteration(
            &mut worker.kv_store,
            &worker.context,
            &worker.batch_plan,
            &mut Vec::new(),
            Time::from_ms(1.0),
        );
        assert_eq!(
            store.borrow()[RequestId(0)]
                .progress
                .prefill_tokens_processed,
            8_192
        );
        assert!(store.borrow()[RequestId(0)]
            .telemetry
            .first_output_time
            .is_none());

        assert!(worker.form_batch(Time::from_ms(1.0)));
        let mut second_input = Default::default();
        worker.execution.build_iteration_input(
            &worker.kv_store,
            &worker.context.requests,
            &worker.batch_plan,
            &mut second_input,
        );
        assert_eq!(second_input.groups[0].prefill_chunk_pairs, [(8_192, 8_192)]);
        worker.admission.complete_iteration(
            &mut worker.kv_store,
            &worker.context,
            &worker.batch_plan,
            &mut Vec::new(),
            Time::from_ms(2.0),
        );
        assert_eq!(
            store.borrow()[RequestId(0)]
                .progress
                .prefill_tokens_processed,
            16_384
        );
        assert!(store.borrow()[RequestId(0)]
            .telemetry
            .first_output_time
            .is_some());
    }

    fn chunked_worker(
        store: SharedRequests,
        batch_policy: BatchPolicy,
        max_batch_tokens: u32,
    ) -> ChunkedPrefillWorker<FakeModel> {
        build_chunked_prefill_worker(
            WorkerId(0),
            "main",
            Arc::new(FakeModel::for_ms(1.0)),
            store,
            WorkerConfig {
                max_batch_tokens: Some(max_batch_tokens),
                batch_policy,
                ..WorkerConfig::default()
            },
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        )
    }

    fn bounded_future_config(max_batch_tokens: u32, attn_kv_tokens: u64) -> WorkerConfig {
        WorkerConfig {
            attn_kv_bytes: attn_kv_tokens,
            max_batch_tokens: Some(max_batch_tokens),
            batch_policy: BatchPolicy::SeparatePrefillPriority,
            kv_admission: KvAdmissionConfig::BoundedFuture(BoundedFutureKvAdmissionConfig {
                page_size: 1,
                max_future_tokens: 4,
                initial_new_token_ratio: 0.7,
                minimum_new_token_ratio: 0.098,
                new_token_ratio_decay_steps: 600,
                retract_decode_steps: 2,
                retraction_policy: DecodeRetractionPolicy::Length,
            }),
            ..WorkerConfig::default()
        }
    }

    fn form_chunked_input(
        worker: &mut ChunkedPrefillWorker<FakeModel>,
        now: Time,
    ) -> crate::arch::UnifiedArchInput {
        assert!(worker.form_batch(now));
        let mut input = Default::default();
        worker.execution.build_iteration_input(
            &worker.kv_store,
            &worker.context.requests,
            &worker.batch_plan,
            &mut input,
        );
        input
    }

    fn complete_chunked_iteration(worker: &mut ChunkedPrefillWorker<FakeModel>, now: Time) {
        worker.admission.complete_iteration(
            &mut worker.kv_store,
            &worker.context,
            &worker.batch_plan,
            &mut Vec::new(),
            now,
        );
    }

    #[test]
    fn separate_prefill_priority_suspends_and_then_resumes_resident_decode() {
        let store = shared_with(&[(0, 4, 5), (1, 16, 1)]);
        let mut worker = chunked_worker(Rc::clone(&store), BatchPolicy::SeparatePrefillPriority, 8);

        // Establish request 0 as a resident decode before request 1 arrives.
        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        let first = form_chunked_input(&mut worker, Time::ZERO);
        assert_eq!(first.groups[0].prefill_chunk_pairs, [(0, 4)]);
        assert_eq!(first.groups[0].decode_tokens, 0);
        complete_chunked_iteration(&mut worker, Time::from_ms(1.0));
        let output_before_prefill = store.borrow()[RequestId(0)].progress.output_tokens_emitted;
        assert_eq!(output_before_prefill, 1);

        worker.enqueue(WorkerMsgCommon::Request(RequestId(1)));
        for (step, prefix) in [(2, 0), (3, 8)] {
            let input = form_chunked_input(&mut worker, Time::from_ms(step as f64));
            assert_eq!(input.groups[0].prefill_chunk_pairs, [(prefix, 8)]);
            assert_eq!(
                input.groups[0].decode_tokens, 0,
                "a SGLang-style prefill iteration excludes resident decode"
            );
            complete_chunked_iteration(&mut worker, Time::from_ms((step + 1) as f64));
            assert_eq!(
                store.borrow()[RequestId(0)].progress.output_tokens_emitted,
                output_before_prefill,
                "excluded decode emits no token while retaining its lifecycle state"
            );
        }

        let decode = form_chunked_input(&mut worker, Time::from_ms(4.0));
        assert!(decode.groups[0].prefill_chunk_pairs.is_empty());
        assert_eq!(decode.groups[0].decode_tokens, 1);
        complete_chunked_iteration(&mut worker, Time::from_ms(5.0));
        assert_eq!(
            store.borrow()[RequestId(0)].progress.output_tokens_emitted,
            output_before_prefill + 1,
            "resident decode resumes when no prefill batch can run"
        );
    }

    #[test]
    fn mix_preserves_shared_prefill_decode_budget_and_progress() {
        let store = shared_with(&[(0, 4, 5), (1, 16, 1)]);
        let mut worker = chunked_worker(Rc::clone(&store), BatchPolicy::Mix, 8);

        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        let _ = form_chunked_input(&mut worker, Time::ZERO);
        complete_chunked_iteration(&mut worker, Time::from_ms(1.0));
        let output_before_mix = store.borrow()[RequestId(0)].progress.output_tokens_emitted;

        worker.enqueue(WorkerMsgCommon::Request(RequestId(1)));
        let mixed = form_chunked_input(&mut worker, Time::from_ms(2.0));
        assert_eq!(mixed.groups[0].prefill_chunk_pairs, [(0, 7)]);
        assert_eq!(mixed.groups[0].decode_tokens, 1);
        assert_eq!(mixed.groups[0].batch_tokens, 8);
        complete_chunked_iteration(&mut worker, Time::from_ms(3.0));
        assert_eq!(
            store.borrow()[RequestId(0)].progress.output_tokens_emitted,
            output_before_mix + 1
        );
    }

    #[test]
    fn bounded_future_retraction_requeues_reprefills_and_preserves_ttft() {
        // Each request needs 23 KV tokens by completion and therefore fits by
        // itself in 24 tokens. The source-style bounded estimate admits both,
        // then physical decode growth forces the later equal-length request to
        // retract. This exercises the real worker FSM rather than a KV helper.
        let store = shared_with(&[(0, 4, 20), (1, 4, 20)]);
        let mut worker = build_chunked_prefill_worker(
            WorkerId(0),
            "main",
            Arc::new(FakeModel::for_ms(1.0)),
            Rc::clone(&store),
            bounded_future_config(64, 24),
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        );
        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        worker.enqueue(WorkerMsgCommon::Request(RequestId(1)));

        let mut events = Vec::new();
        for step in 0..100 {
            worker.tick(Time::from_ms(step as f64), &mut events);
            if events.len() == 2 && worker.status().active_requests == 0 {
                break;
            }
        }

        assert_eq!(events.len(), 2, "both requests must drain after requeue");
        let requests = store.borrow();
        let kept = &requests[RequestId(0)];
        let retracted = &requests[RequestId(1)];
        assert!(kept.lifecycle.completed && retracted.lifecycle.completed);
        assert_eq!(kept.progress.output_tokens_emitted, 20);
        assert_eq!(retracted.progress.output_tokens_emitted, 20);
        assert_eq!(kept.telemetry.retraction_count, 0);
        assert_eq!(retracted.telemetry.retraction_count, 1);
        assert_eq!(
            retracted.telemetry.first_output_time,
            Some(Time::from_ms(1.0))
        );
        assert_eq!(retracted.telemetry.reprocessed_prefills.len(), 1);
        let episode = retracted.telemetry.reprocessed_prefills[0];
        assert_eq!(episode.output_tokens_before, 9);
        assert_eq!(episode.prefix_cache_hit_tokens, 0);
        assert_eq!(episode.prefill_tokens_processed, 13);
        assert!(episode.completed);
        assert_eq!(
            retracted
                .lifecycle
                .stage_log
                .iter()
                .map(|event| event.code)
                .collect::<Vec<_>>(),
            vec![
                UnifiedStage::Pending as u16,
                UnifiedStage::Prefill as u16,
                UnifiedStage::Decode as u16,
                UnifiedStage::Pending as u16,
                UnifiedStage::Prefill as u16,
                UnifiedStage::Decode as u16,
                UnifiedStage::Done as u16,
            ],
            "retraction must be an observable decode→queue→prefill→decode episode"
        );
    }

    #[test]
    fn bounded_future_ratio_resets_after_the_worker_becomes_fully_idle() {
        let store = shared_with(&[(0, 4, 3)]);
        let mut config = bounded_future_config(64, 24);
        config.kv_admission = KvAdmissionConfig::BoundedFuture(BoundedFutureKvAdmissionConfig {
            new_token_ratio_decay_steps: 2,
            ..match config.kv_admission {
                KvAdmissionConfig::BoundedFuture(config) => config,
                KvAdmissionConfig::FullFootprint => unreachable!(),
            }
        });
        let mut worker = build_chunked_prefill_worker(
            WorkerId(0),
            "main",
            Arc::new(FakeModel::for_ms(1.0)),
            Rc::clone(&store),
            config,
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        );
        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));

        let mut events = Vec::new();
        worker.tick(Time::ZERO, &mut events);
        worker.tick(Time::from_ms(1.0), &mut events);
        assert!(
            worker.admission.current_new_token_ratio() < 0.7,
            "the first successful decode-capacity check must decay the ratio"
        );
        for step in 2..10 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }

        assert_eq!(events.len(), 1);
        assert_eq!(worker.status().active_requests, 0);
        assert_eq!(worker.status().queued_requests, 0);
        assert_eq!(worker.admission.current_new_token_ratio(), 0.7);
    }

    #[test]
    #[should_panic(
        expected = "bounded-future KV admission is validated only for one attention partition"
    )]
    fn bounded_future_rejects_unaudited_multi_partition_composition() {
        let _ = build_chunked_prefill_worker(
            WorkerId(0),
            "main",
            Arc::new(FakeModel {
                ms: 1.0,
                dp_groups: 2,
            }),
            shared_with(&[]),
            bounded_future_config(64, 24),
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        );
    }

    #[test]
    fn hp_builds_one_partition_per_dp_group() {
        let worker = hp_worker(shared_with(&[]), 2, WorkerConfig::default());
        assert_eq!(worker.kv_store.num_partitions(), 2);
    }

    #[test]
    fn hp_round_robin_spreads_prefills_across_partitions() {
        let store = shared_with(&[(0, 4, 50), (1, 4, 50)]);
        let mut worker = hp_worker(store, 2, WorkerConfig::default());
        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        worker.enqueue(WorkerMsgCommon::Request(RequestId(1)));
        let mut events = Vec::new();
        for step in 0..20 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }

        let first_partition = worker.kv_store.decode_members(0).count();
        let second_partition = worker.kv_store.decode_members(1).count();
        assert_eq!((first_partition, second_partition), (1, 1));
    }

    #[test]
    fn hp_same_session_returns_to_its_retained_prefix_partition() {
        for max_batch_tokens in [None, Some(256)] {
            let store = shared_with(&[(0, 20, 1), (1, 30, 1)]);
            {
                let mut requests = store.borrow_mut();
                for request in [RequestId(0), RequestId(1)] {
                    requests[request].request.definition.session = SessionInput::Session {
                        session_id: 7,
                        session_start_time: Time::ZERO,
                        declared_prefix_tokens: 100,
                    };
                }
            }
            let config = WorkerConfig {
                attn_kv_bytes: 1_000,
                max_batch_tokens,
                ..WorkerConfig::default()
            };
            let mut worker = hp_worker(Rc::clone(&store), 2, config);

            worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
            let events = run_to_quiescence(&mut worker, 100);
            assert_eq!(events.len(), 1);
            assert_eq!(
                store.borrow()[RequestId(0)]
                    .progress
                    .prefill_tokens_processed,
                120
            );

            // The cold request advanced RR to partition 1. A cache-unaware
            // placement would therefore send this request away from its prefix.
            worker.enqueue(WorkerMsgCommon::Request(RequestId(1)));
            assert!(worker.form_batch(Time::from_ms(100.0)));
            assert_eq!(worker.kv_store.prefill_admits(0).count(), 1);
            assert_eq!(worker.kv_store.prefill_admits(1).count(), 0);
            assert_eq!(worker.kv_store.prefill_tokens_to_compute(RequestId(1)), 30);
        }
    }

    #[test]
    fn hp_input_has_one_group_per_partition() {
        let worker = hp_worker(shared_with(&[]), 3, WorkerConfig::default());
        let mut batch_plan = IterBatchPlan::default();
        batch_plan.reset_decode_participation(3, true);
        let mut input = Default::default();
        worker.execution.build_iteration_input(
            &worker.kv_store,
            &worker.context.requests,
            &batch_plan,
            &mut input,
        );
        assert_eq!(input.groups.len(), 3);
    }

    #[test]
    fn hp_all_arrivals_complete() {
        let store = shared_with(&[(0, 4, 2), (1, 4, 2), (2, 4, 2)]);
        let mut worker = hp_worker(store, 2, WorkerConfig::default());
        for request in 0..3 {
            worker.enqueue(WorkerMsgCommon::Request(RequestId(request)));
        }

        let mut completed = Vec::new();
        let mut events = Vec::new();
        for step in 0..500 {
            worker.tick(Time::from_ms(step as f64), &mut events);
            for event in events.drain(..) {
                let WorkerEventCommon::RequestComplete { req, .. } = event;
                completed.push(req);
            }
        }
        completed.sort_by_key(|request| request.0);
        assert_eq!(completed, (0..3).map(RequestId).collect::<Vec<_>>());
    }

    #[test]
    fn hp_budget_is_independent_per_partition() {
        let store = shared_with(&[
            (0, 8, 0),
            (1, 8, 0),
            (2, 8, 0),
            (3, 8, 0),
            (4, 8, 0),
            (5, 8, 0),
        ]);
        let config = WorkerConfig {
            max_batch_tokens: Some(16),
            ..WorkerConfig::default()
        };
        let mut worker = hp_worker(store, 2, config);
        for request in 0..6 {
            worker.enqueue(WorkerMsgCommon::Request(RequestId(request)));
        }

        worker.form_batch(Time::ZERO);
        assert_eq!(worker.kv_store.prefill_admits(0).count(), 2);
        assert_eq!(worker.kv_store.prefill_admits(1).count(), 2);
        assert_eq!(worker.admission.queued_requests(), 2);
    }
}
