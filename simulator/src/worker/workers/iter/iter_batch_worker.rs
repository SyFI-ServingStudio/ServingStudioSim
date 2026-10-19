//! `IterBatchWorker` — the whole-iteration L5 shell.
//!
//! The shell owns cadence and cross-axis sequencing. KV, Admission, and Execution
//! are statically composed; the FSM itself is deliberately not an axis.
//!
//! Two planning cadences share the one iteration loop:
//!
//! - **Synchronous (default).** Every message reaches admission when it is
//!   enqueued, and step N+1 is formed when step N completes from everything
//!   enqueued so far.
//! - **Async one-step lookahead (opt-in, [`IterBatchWorker::with_async_lookahead`]).**
//!   An engine with async scheduling plans step N+1 a fixed offset `w` after
//!   step N starts, while N executes, treating N's scheduled work as done. Its
//!   input queue is drained only at that plan moment. The shell models this
//!   with [`AsyncLookahead`]: a message enqueued while a step runs waits in the
//!   shell's engine inbox until the next plan moment, so a request that arrives
//!   after `start + w` misses step N+1. Step N+1 still starts when N finishes
//!   (or at the plan moment if N was shorter than `w`), and is formed after N's
//!   completion, which is exactly "N's work treated as complete". An idle engine
//!   polls its queue, so with no step running a message is visible at once, and
//!   a planned-empty step makes the inbox visible immediately.
//!
//!   The offset can differ for a step that started from an idle engine (the
//!   first step of a burst, or one after a planned-empty step): with no
//!   predecessor in flight, the real engine plans the successor right after
//!   dispatching it, which the measured stock vLLM Neuron first-step offset
//!   reflects, rather than after a predecessor's output. The recipe passes it
//!   separately; absent, every step uses the same offset.
//!
//! The inbox is transport state, not pending membership: admission and its
//! selection policy never see an inbox message until the plan moment delivers
//! it, so the policy remains the sole owner of not-yet-started requests.

use crate::common::{RequestId, Time, WorkerId};
use crate::worker::admission::{
    ChunkedPrefillAdmission, IterAdmission, LocalPrefillDecodeAdmission, PendingOrder,
    PrefillHandoffAdmission, SpeculativeDecodeCompletion,
};
use crate::worker::execution::{
    IterModelExecution, SpeculativeIterExecution, UnifiedIterExecution,
};
use crate::worker::iter_worker::IterWorker;
use crate::worker::kv::{IterWorkerKv, KvCacheStore};
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

/// Async-scheduling ingress: the engine input queue and the next plan moment.
///
/// Present only when the recipe opts in. `next_plan_at` is set when a step
/// starts and cleared at the plan moment, when the inbox is delivered to
/// admission. While it is set, the step's successor has not been planned yet.
struct AsyncLookahead<M> {
    /// Plan moment offset after a step starts.
    offset: Time,
    /// Plan moment offset after a step that started from an idle engine.
    offset_from_idle: Time,
    /// No step has started since the engine last found nothing to run: the
    /// next step starts from idle.
    engine_idle: bool,
    /// Messages that reached the engine while a step ran and that no plan
    /// moment has delivered yet, in arrival order.
    inbox: Vec<M>,
    next_plan_at: Option<Time>,
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
    /// `None` is the synchronous cadence.
    lookahead: Option<AsyncLookahead<A::Msg>>,
}

/// Compatibility name retained for L6 factories and deployment selectors.
///
/// The pending order is `PendingOrder` (a runtime dispatcher) rather than a
/// concrete policy so the queue discipline stays a preset knob without fanning
/// this alias — and every alias below it — out by one type per policy. The
/// policy axis is still static: see `pending_policy_is_a_composition_axis_for_iter_lifecycles`.
pub type BareboneWorker<M> = IterBatchWorker<
    KvCacheStore,
    LocalPrefillDecodeAdmission<PendingOrder>,
    UnifiedIterExecution<M>,
>;
/// Multi-partition HP/DP recipe uses the same whole-iteration shell.
pub type HpUnifiedWorker<M> = BareboneWorker<M>;
pub type ChunkedPrefillWorker<M> =
    IterBatchWorker<KvCacheStore, ChunkedPrefillAdmission<PendingOrder>, UnifiedIterExecution<M>>;
/// `ChunkedPrefillWorker` with a speculating decode engine. Two axes change
/// together because they are two halves of one fact: the lifecycle's decode
/// completion retires an accepted chain instead of one token, and the execution
/// adapter submits the matching verify width to the model. KV, policy, and shell
/// are unchanged — speculation is not a cadence.
pub type SpeculativeWorker<M> = IterBatchWorker<
    KvCacheStore,
    ChunkedPrefillAdmission<PendingOrder, SpeculativeDecodeCompletion>,
    SpeculativeIterExecution<M>,
>;
pub type PdPrefillWorker<M> =
    IterBatchWorker<KvCacheStore, PrefillHandoffAdmission<PendingOrder>, UnifiedIterExecution<M>>;

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
            lookahead: None,
        }
    }

    /// Select async one-step lookahead with plan offset `offset` (`None` keeps
    /// the synchronous cadence), and `offset_from_idle` after a step that
    /// started from an idle engine (`None` uses `offset`). See the module
    /// documentation.
    pub(super) fn with_async_lookahead(
        mut self,
        offset: Option<Time>,
        offset_from_idle: Option<Time>,
    ) -> Self {
        self.lookahead = offset.map(|offset| AsyncLookahead {
            offset,
            offset_from_idle: offset_from_idle.unwrap_or(offset),
            engine_idle: true,
            inbox: Vec::new(),
            next_plan_at: None,
        });
        self
    }

    pub fn enqueue(&mut self, msg: A::Msg) {
        match &mut self.lookahead {
            Some(lookahead) if self.iteration_fsm.worker_state == WorkerFsmState::Active => {
                lookahead.inbox.push(msg);
            }
            _ => self
                .admission
                .accept_message(&mut self.kv_store, msg, &self.context),
        }
    }

    pub fn tick(&mut self, now: Time, events: &mut Vec<A::Event>) -> Option<Time> {
        use IterCursor::{Computing, Done, NotStarted};
        use WorkerFsmState::{Active, Idle};

        loop {
            self.plan_if_due(now);
            match self.iteration_fsm.worker_state {
                Idle => {
                    if self.awaiting_plan() {
                        break;
                    }
                    if !self.form_batch(now) {
                        // An engine whose planned step is empty goes back to
                        // polling its input queue, so the inbox is visible now.
                        self.mark_engine_idle();
                        if self.deliver_inbox() {
                            continue;
                        }
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

    /// At the plan moment, the engine drains its input queue: deliver the inbox.
    fn plan_if_due(&mut self, now: Time) {
        let Some(lookahead) = &mut self.lookahead else {
            return;
        };
        if lookahead.next_plan_at.is_some_and(|plan_at| plan_at <= now) {
            lookahead.next_plan_at = None;
            self.deliver_inbox();
        }
    }

    /// The previous step finished before its successor's plan moment.
    fn awaiting_plan(&self) -> bool {
        self.lookahead
            .as_ref()
            .is_some_and(|lookahead| lookahead.next_plan_at.is_some())
    }

    /// The engine found nothing to run: its next step starts from idle.
    fn mark_engine_idle(&mut self) {
        if let Some(lookahead) = &mut self.lookahead {
            lookahead.engine_idle = true;
        }
    }

    /// Hand every inbox message to admission in arrival order. Returns whether
    /// any message was delivered.
    fn deliver_inbox(&mut self) -> bool {
        let Some(lookahead) = &mut self.lookahead else {
            return false;
        };
        if lookahead.inbox.is_empty() {
            return false;
        }
        for msg in lookahead.inbox.drain(..) {
            self.admission
                .accept_message(&mut self.kv_store, msg, &self.context);
        }
        true
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
        if let Some(lookahead) = &mut self.lookahead {
            let offset = if lookahead.engine_idle {
                lookahead.offset_from_idle
            } else {
                lookahead.offset
            };
            lookahead.engine_idle = false;
            lookahead.next_plan_at = Some(now + offset);
        }
        now + cost
    }

    fn next_wakeup(&self, now: Time) -> Option<Time> {
        let next_plan_at = self
            .lookahead
            .as_ref()
            .and_then(|lookahead| lookahead.next_plan_at);
        match self.iteration_fsm.worker_state {
            WorkerFsmState::Idle => {
                if next_plan_at.is_some() {
                    return next_plan_at;
                }
                let status = self.status();
                (status.queued_requests > 0 || status.active_requests > 0).then_some(now)
            }
            WorkerFsmState::Active => match self.iteration_fsm.batch_state.cursor {
                IterCursor::NotStarted | IterCursor::Done => Some(now),
                IterCursor::Computing => {
                    let compute_end = self.iteration_fsm.batch_state.compute_end;
                    Some(next_plan_at.map_or(compute_end, |plan_at| plan_at.min(compute_end)))
                }
            },
        }
    }

    /// Queued counts both admission's pending requests and async-lookahead
    /// inbox messages not yet delivered to it.
    pub fn status(&self) -> WorkerStatus {
        let active_requests = (0..self.kv_store.num_partitions() as u16)
            .map(|partition| self.kv_store.status_active(partition))
            .sum();
        let inbox = self
            .lookahead
            .as_ref()
            .map_or(0, |lookahead| lookahead.inbox.len() as u32);
        WorkerStatus {
            queued_requests: self.admission.queued_requests() + inbox,
            active_requests,
        }
    }

    /// Cancel a pending request or release a resident one. A message still in
    /// the async-lookahead inbox is invisible to admission and is not found;
    /// no L6 caller cancels an iteration worker's request today.
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
    use crate::worker::config::{BoundedFutureKvAdmissionConfig, DpPlacement};
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
                KvCacheStore,
                LocalPrefillDecodeAdmission<FifoOrder>,
                UnifiedIterExecution<FakeModel>,
            >,
        >();
        assert_iter_worker::<
            IterBatchWorker<
                KvCacheStore,
                LocalPrefillDecodeAdmission<ShortestJobFirst>,
                UnifiedIterExecution<FakeModel>,
            >,
        >();
        assert_iter_worker::<
            IterBatchWorker<
                KvCacheStore,
                PrefillHandoffAdmission<FifoOrder>,
                UnifiedIterExecution<FakeModel>,
            >,
        >();
        assert_iter_worker::<
            IterBatchWorker<
                KvCacheStore,
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
            Arc::new(FakeModel::with_dp_groups(1.0, dp_groups)),
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

    #[test]
    fn chunked_prefill_computes_only_the_fresh_prompt_after_a_pinned_prefix() {
        let store = shared_with(&[(0, 12_000, 2), (1, 12_000, 2)]);
        store.borrow_mut()[RequestId(0)].request.definition.session = SessionInput::PinnedPrefix {
            prefix_tokens: 30_000,
        };
        // Room for the pinned request's 30_000 + 12_000 + 2 context, not for a
        // second request beside it: the prefix counts toward capacity.
        let mut worker = build_chunked_prefill_worker(
            WorkerId(0),
            "main",
            Arc::new(FakeModel::for_ms(1.0)),
            Rc::clone(&store),
            WorkerConfig {
                max_batch_tokens: Some(8_192),
                attn_kv_bytes: 50_000,
                prefix_cache: PrefixCacheConfig::Disabled,
                ..WorkerConfig::default()
            },
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        );
        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        worker.enqueue(WorkerMsgCommon::Request(RequestId(1)));

        let mut chunk_pairs = Vec::new();
        for step in 0..2 {
            assert!(worker.form_batch(Time::from_ms(step as f64)));
            let mut input = Default::default();
            worker.execution.build_iteration_input(
                &worker.kv_store,
                &worker.context.requests,
                &worker.batch_plan,
                &mut input,
            );
            chunk_pairs.extend(input.groups[0].prefill_chunk_pairs.iter().copied());
            worker.admission.complete_iteration(
                &mut worker.kv_store,
                &worker.context,
                &worker.batch_plan,
                &mut Vec::new(),
                Time::from_ms(step as f64 + 1.0),
            );
        }
        // Attention sees the pinned prefix as context; only 12_000 tokens run.
        assert_eq!(chunk_pairs, [(30_000, 8_192), (38_192, 3_808)]);
        assert!(!store.borrow()[RequestId(1)].lifecycle.admitted);
        let requests = store.borrow();
        let record = &requests[RequestId(0)];
        assert_eq!(record.progress.prefill_tokens_processed, 12_000);
        assert_eq!(record.telemetry.prefix_cache_hit_tokens, Some(30_000));
        assert!(record.telemetry.first_output_time.is_some());
    }

    /// A chunked-prefill worker over a layout with a one-byte recurrent state
    /// on a `checkpoint_block`-token engine block, or plain per-token KV.
    fn hybrid_chunked_worker(
        store: SharedRequests,
        checkpoint_block: Option<u32>,
    ) -> ChunkedPrefillWorker<FakeModel> {
        hybrid_chunked_worker_aligned(store, checkpoint_block, None)
    }

    fn hybrid_chunked_worker_aligned(
        store: SharedRequests,
        checkpoint_block: Option<u32>,
        prefill_chunk_alignment: Option<crate::worker::config::PrefillChunkAlignment>,
    ) -> ChunkedPrefillWorker<FakeModel> {
        use crate::arch::kv_layout::{EngineOverhead, KvCacheLayout, KvCachePart};
        let kv_layout = checkpoint_block.map(|block| {
            KvCacheLayout::new()
                .with_part(KvCachePart::per_token("kv", 1))
                .with_part(KvCachePart::per_request("state", 1))
                .with_engine(EngineOverhead {
                    block_tokens: Some(block),
                    fixed_device_bytes: 0,
                })
        });
        build_chunked_prefill_worker(
            WorkerId(0),
            "main",
            Arc::new(FakeModel {
                kv_layout,
                ..FakeModel::for_ms(1.0)
            }),
            store,
            WorkerConfig {
                max_batch_tokens: Some(10),
                prefill_chunk_alignment,
                ..WorkerConfig::default()
            },
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        )
    }

    fn hybrid_chunk_pairs(
        worker: &mut ChunkedPrefillWorker<FakeModel>,
        now: Time,
    ) -> Vec<(u32, u32)> {
        assert!(worker.form_batch(now));
        let mut input: crate::arch::UnifiedArchInput = Default::default();
        worker.execution.build_iteration_input(
            &worker.kv_store,
            &worker.context.requests,
            &worker.batch_plan,
            &mut input,
        );
        worker.admission.complete_iteration(
            &mut worker.kv_store,
            &worker.context,
            &worker.batch_plan,
            &mut Vec::new(),
            now,
        );
        input.groups[0].prefill_chunk_pairs.clone()
    }

    #[test]
    fn checkpoint_aligned_chunks_leave_budget_for_a_second_prompt() {
        let store = shared_with(&[(0, 20, 2), (1, 2, 1)]);
        let mut worker = hybrid_chunked_worker(Rc::clone(&store), Some(4));
        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        worker.enqueue(WorkerMsgCommon::Request(RequestId(1)));

        // 10 tokens of budget reach context 10; the chunk ends on boundary 8
        // and the short prompt runs to completion in the two left over.
        assert_eq!(
            hybrid_chunk_pairs(&mut worker, Time::ZERO),
            [(0, 8), (0, 2)]
        );
        assert!(store.borrow()[RequestId(1)]
            .telemetry
            .first_output_time
            .is_some());
        assert_eq!(
            hybrid_chunk_pairs(&mut worker, Time::from_ms(1.0)),
            [(8, 8)]
        );
        // The final chunk crosses no boundary and is not clipped.
        assert_eq!(
            hybrid_chunk_pairs(&mut worker, Time::from_ms(2.0)),
            [(16, 4)]
        );
        assert!(store.borrow()[RequestId(0)]
            .telemetry
            .first_output_time
            .is_some());
    }

    #[test]
    fn plain_alignment_ignores_the_checkpoint_interval() {
        let store = shared_with(&[(0, 20, 2), (1, 2, 1)]);
        let mut worker = hybrid_chunked_worker_aligned(
            Rc::clone(&store),
            Some(4),
            Some(crate::worker::config::PrefillChunkAlignment::Plain),
        );
        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        worker.enqueue(WorkerMsgCommon::Request(RequestId(1)));

        // Same interval as the checkpoint-aligned case, but the chunk spends
        // the whole budget instead of stopping on boundary 8.
        assert_eq!(hybrid_chunk_pairs(&mut worker, Time::ZERO), [(0, 10)]);
        assert_eq!(
            hybrid_chunk_pairs(&mut worker, Time::from_ms(1.0)),
            [(10, 10)]
        );
    }

    #[test]
    fn without_a_checkpoint_interval_a_partial_chunk_spends_the_whole_budget() {
        let store = shared_with(&[(0, 20, 2), (1, 2, 1)]);
        let mut worker = hybrid_chunked_worker(Rc::clone(&store), None);
        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        worker.enqueue(WorkerMsgCommon::Request(RequestId(1)));

        assert_eq!(hybrid_chunk_pairs(&mut worker, Time::ZERO), [(0, 10)]);
        assert_eq!(
            hybrid_chunk_pairs(&mut worker, Time::from_ms(1.0)),
            [(10, 10)]
        );
        assert_eq!(
            hybrid_chunk_pairs(&mut worker, Time::from_ms(2.0)),
            [(0, 2)]
        );
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

    fn limited_chunked_worker(
        store: SharedRequests,
        max_prefill_requests_per_iteration: Option<u32>,
        max_resident_requests_per_partition: Option<u32>,
        max_batch_tokens: u32,
    ) -> ChunkedPrefillWorker<FakeModel> {
        build_chunked_prefill_worker(
            WorkerId(0),
            "main",
            Arc::new(FakeModel::for_ms(1.0)),
            store,
            WorkerConfig {
                attn_kv_bytes: 64,
                max_batch_tokens: Some(max_batch_tokens),
                max_prefill_requests_per_iteration,
                max_resident_requests_per_partition,
                batch_policy: BatchPolicy::SeparatePrefillPriority,
                prefix_cache: PrefixCacheConfig::Disabled,
                ..WorkerConfig::default()
            },
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        )
    }

    fn prefill_admits(worker: &ChunkedPrefillWorker<FakeModel>) -> Vec<RequestId> {
        let mut admits = Vec::new();
        worker
            .kv_store
            .visit_prefill_admits(0, |request| admits.push(request));
        admits
    }

    #[test]
    fn request_limits_preserve_fifo_suspend_decode_and_reopen_after_release() {
        let store = shared_with(&[(0, 4, 3), (1, 4, 3), (2, 4, 2)]);
        let mut worker = limited_chunked_worker(store.clone(), Some(1), Some(2), 32);
        for id in 0..3 {
            worker.enqueue(WorkerMsgCommon::Request(RequestId(id)));
        }
        // One prompt per iteration, in FIFO order, though the budget fits all.
        for id in 0..2 {
            let input = form_chunked_input(&mut worker, Time::from_ms(id as f64));
            assert_eq!(input.groups[0].prefill_chunk_pairs, [(0, 4)]);
            assert_eq!(input.groups[0].decode_tokens, 0);
            assert_eq!(prefill_admits(&worker), [RequestId(id)]);
            complete_chunked_iteration(&mut worker, Time::from_ms((id + 1) as f64));
        }
        assert_eq!(
            store.borrow()[RequestId(0)].progress.output_tokens_emitted,
            1
        );
        // Two residents fill the cap: the third waits while both decode.
        assert!(!store.borrow()[RequestId(2)].lifecycle.admitted);
        for step in 2..4 {
            let input = form_chunked_input(&mut worker, Time::from_ms(step as f64));
            assert!(input.groups[0].prefill_chunk_pairs.is_empty());
            assert_eq!(input.groups[0].decode_tokens, 2);
            complete_chunked_iteration(&mut worker, Time::from_ms((step + 1) as f64));
        }
        assert_eq!(worker.status().active_requests, 0);
        let input = form_chunked_input(&mut worker, Time::from_ms(4.0));
        assert_eq!(input.groups[0].prefill_chunk_pairs, [(0, 4)]);
        assert!(store.borrow()[RequestId(2)].lifecycle.admitted);
        complete_chunked_iteration(&mut worker, Time::from_ms(5.0));
        let _ = form_chunked_input(&mut worker, Time::from_ms(5.0));
        complete_chunked_iteration(&mut worker, Time::from_ms(6.0));
        assert_eq!(worker.status().active_requests, 0);
        assert_eq!(worker.status().queued_requests, 0);
        let entire_pool = worker.kv_store.footprint(RequestId(99), 63, 1);
        assert!(
            worker.kv_store.fits(0, &entire_pool),
            "completion releases the entire KV reservation"
        );
    }

    #[test]
    fn resident_limit_counts_output_one_and_completed_scheduled_prefills() {
        let store = shared_with(&[(0, 2, 1), (1, 2, 1), (2, 2, 1)]);
        let mut worker = limited_chunked_worker(store, None, Some(2), 32);
        for id in 0..3 {
            worker.enqueue(WorkerMsgCommon::Request(RequestId(id)));
        }
        let input = form_chunked_input(&mut worker, Time::ZERO);
        assert_eq!(input.groups[0].prefill_chunk_pairs.len(), 2);
        assert_eq!(worker.status().queued_requests, 1);
        complete_chunked_iteration(&mut worker, Time::from_ms(1.0));
        let input = form_chunked_input(&mut worker, Time::from_ms(1.0));
        assert_eq!(input.groups[0].prefill_chunk_pairs.len(), 1);

        let store = shared_with(&[(0, 12, 1), (1, 2, 1)]);
        let mut worker = limited_chunked_worker(store, None, Some(1), 8);
        for id in 0..2 {
            worker.enqueue(WorkerMsgCommon::Request(RequestId(id)));
        }
        let input = form_chunked_input(&mut worker, Time::ZERO);
        assert_eq!(input.groups[0].prefill_chunk_pairs, [(0, 8)]);
        complete_chunked_iteration(&mut worker, Time::from_ms(1.0));
        // Finishing a started prompt still occupies the resident slot for this iteration.
        let input = form_chunked_input(&mut worker, Time::from_ms(1.0));
        assert_eq!(input.groups[0].prefill_chunk_pairs, [(8, 4)]);
        assert_eq!(worker.status().queued_requests, 1);
        complete_chunked_iteration(&mut worker, Time::from_ms(2.0));
        let input = form_chunked_input(&mut worker, Time::from_ms(2.0));
        assert_eq!(input.groups[0].prefill_chunk_pairs, [(0, 2)]);
    }

    #[test]
    fn absent_request_limits_keep_multiple_prefills_per_iteration() {
        let store = shared_with(&[(0, 2, 1), (1, 2, 1), (2, 2, 1)]);
        let mut worker = limited_chunked_worker(store, None, None, 32);
        for id in 0..3 {
            worker.enqueue(WorkerMsgCommon::Request(RequestId(id)));
        }
        let input = form_chunked_input(&mut worker, Time::ZERO);
        assert_eq!(input.groups[0].prefill_chunk_pairs.len(), 3);
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
    fn fcfs_retraction_resumes_the_most_recent_preemption_first() {
        // Three requests fit the bounded estimate, then decode growth retracts
        // request 2 and later request 1. SGLang's length policy requeues each
        // behind the waiting set, so 2 resumes first; vLLM's FCFS prepends each
        // preempted request, so 1, preempted last, resumes first.
        for (policy, first_resumed) in [
            (DecodeRetractionPolicy::Length, 2),
            (DecodeRetractionPolicy::Fcfs, 1),
        ] {
            let store = shared_with(&[(0, 4, 20), (1, 4, 20), (2, 4, 20)]);
            let mut config = bounded_future_config(64, 24);
            config.pending_order = crate::worker::admission::PendingOrderKind::Fifo;
            let KvAdmissionConfig::BoundedFuture(bounded) = &mut config.kv_admission else {
                unreachable!("bounded_future_config builds bounded-future admission")
            };
            bounded.retraction_policy = policy;
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
            for id in 0..3 {
                worker.enqueue(WorkerMsgCommon::Request(RequestId(id)));
            }
            for step in 0..200 {
                worker.tick(Time::from_ms(step as f64), &mut Vec::new());
            }

            let requests = store.borrow();
            let resumed_at = |id: u32| {
                let record = &requests[RequestId(id)];
                assert!(record.lifecycle.completed, "{policy:?}: request {id}");
                assert_eq!(
                    record.telemetry.retraction_count, 1,
                    "{policy:?}: request {id}"
                );
                // The second Prefill stage is the re-prefill after retraction.
                record
                    .lifecycle
                    .stage_log
                    .iter()
                    .filter(|event| event.code == UnifiedStage::Prefill as u16)
                    .nth(1)
                    .expect("a retracted request re-prefills")
                    .time
            };
            let other = 3 - first_resumed;
            assert!(
                resumed_at(first_resumed) < resumed_at(other),
                "{policy:?}: request {first_resumed} must resume before request {other}"
            );
        }
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
            Arc::new(FakeModel::with_dp_groups(1.0, 2)),
            shared_with(&[]),
            bounded_future_config(64, 24),
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        );
    }

    /// Two partitions; request 0 decodes for a long time on partition 0 while
    /// request 1 finishes on partition 1. Round-robin then sends request 2
    /// back to busy partition 0; vLLM's least-loaded balancer picks idle 1.
    fn third_request_partition(dp_placement: DpPlacement) -> (usize, usize) {
        let store = shared_with(&[(0, 4, 500), (1, 4, 1), (2, 4, 1)]);
        let mut worker = build_chunked_prefill_worker(
            WorkerId(0),
            "main",
            Arc::new(FakeModel::with_dp_groups(1.0, 2)),
            store,
            WorkerConfig {
                max_batch_tokens: Some(64),
                dp_placement,
                ..WorkerConfig::default()
            },
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        );
        let mut events = Vec::new();
        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        worker.enqueue(WorkerMsgCommon::Request(RequestId(1)));
        for step in 0..20 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(events.len(), 1, "only request 1 has finished");
        assert_eq!(worker.kv_store.decode_members(0).count(), 1);
        assert_eq!(worker.kv_store.decode_members(1).count(), 0);

        worker.enqueue(WorkerMsgCommon::Request(RequestId(2)));
        assert!(worker.form_batch(Time::from_ms(20.0)));
        (
            worker.kv_store.prefill_admits(0).count(),
            worker.kv_store.prefill_admits(1).count(),
        )
    }

    #[test]
    fn round_robin_dp_placement_ignores_partition_load() {
        assert_eq!(third_request_partition(DpPlacement::RoundRobin), (1, 0));
    }

    #[test]
    fn vllm_least_loaded_dp_placement_picks_the_idle_partition() {
        assert_eq!(
            third_request_partition(DpPlacement::VllmLeastLoaded),
            (0, 1)
        );
    }

    /// Stock-Neuron-shaped chunked worker: one prefill per iteration,
    /// separate-prefill-priority, FIFO, 1 ms iterations.
    fn stock_shaped_worker(
        store: SharedRequests,
        async_lookahead: Option<Time>,
    ) -> ChunkedPrefillWorker<FakeModel> {
        stock_shaped_worker_from_idle(store, async_lookahead, None)
    }

    /// [`stock_shaped_worker`] with a separate plan offset after a step that
    /// started from an idle engine.
    fn stock_shaped_worker_from_idle(
        store: SharedRequests,
        async_lookahead: Option<Time>,
        async_lookahead_from_idle: Option<Time>,
    ) -> ChunkedPrefillWorker<FakeModel> {
        build_chunked_prefill_worker(
            WorkerId(0),
            "main",
            Arc::new(FakeModel::for_ms(1.0)),
            store,
            WorkerConfig {
                attn_kv_bytes: 256,
                max_batch_tokens: Some(64),
                max_prefill_requests_per_iteration: Some(1),
                max_resident_requests_per_partition: Some(16),
                batch_policy: BatchPolicy::SeparatePrefillPriority,
                pending_order: crate::worker::PendingOrderKind::Fifo,
                prefix_cache: PrefixCacheConfig::Disabled,
                async_lookahead,
                async_lookahead_from_idle,
                ..WorkerConfig::default()
            },
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        )
    }

    /// Drive `worker` on a 0.1 ms clock, enqueuing each `(time_us, request)`
    /// at the first tick at or after its time, and return each iteration's
    /// start time (µs) and kind: `P` prefill, `D` decode.
    fn iteration_kinds(
        worker: &mut ChunkedPrefillWorker<FakeModel>,
        arrivals: &[(u64, u32)],
        until_us: u64,
    ) -> Vec<(u64, char)> {
        let mut kinds = Vec::new();
        let mut events = Vec::new();
        let mut next_arrival = 0;
        for now_us in (0..=until_us).step_by(100) {
            while next_arrival < arrivals.len() && arrivals[next_arrival].0 <= now_us {
                worker.enqueue(WorkerMsgCommon::Request(RequestId(
                    arrivals[next_arrival].1,
                )));
                next_arrival += 1;
            }
            let before = worker.iteration_fsm.iteration;
            worker.tick(Time::from_us(now_us), &mut events);
            if worker.iteration_fsm.iteration != before {
                let prefill = !worker.input.groups[0].prefill_chunk_pairs.is_empty();
                kinds.push((now_us, if prefill { 'P' } else { 'D' }));
            }
        }
        kinds
    }

    fn kind_string(kinds: &[(u64, char)]) -> String {
        kinds.iter().map(|(_, kind)| kind).collect()
    }

    #[test]
    fn async_lookahead_runs_a_solo_decode_when_the_second_request_misses_the_plan() {
        // Step 0 starts at 0 and plans step 1 at 0.3 ms. Request 1 reaches the
        // engine at 0.5 ms, after that plan, so step 1 decodes request 0 alone.
        let store = shared_with(&[(0, 4, 3), (1, 4, 3)]);
        let mut worker = stock_shaped_worker(Rc::clone(&store), Some(Time::from_us(300)));
        let kinds = iteration_kinds(&mut worker, &[(0, 0), (500, 1)], 10_000);
        assert_eq!(kind_string(&kinds), "PDPDD");
        assert_eq!(kinds[2].0, 2_000, "request 1 prefills in step 2");
        assert!(store.borrow()[RequestId(0)].lifecycle.completed);
        assert!(store.borrow()[RequestId(1)].lifecycle.completed);
    }

    #[test]
    fn async_lookahead_prefills_back_to_back_when_the_second_request_beats_the_plan() {
        let store = shared_with(&[(0, 4, 3), (1, 4, 3)]);
        let mut worker = stock_shaped_worker(store, Some(Time::from_us(300)));
        let kinds = iteration_kinds(&mut worker, &[(0, 0), (200, 1)], 10_000);
        assert_eq!(kind_string(&kinds), "PPDD");
    }

    #[test]
    fn synchronous_cadence_sees_every_arrival_before_the_step_ends() {
        // Default: the same late arrival is visible when step 0 completes.
        let store = shared_with(&[(0, 4, 3), (1, 4, 3)]);
        let mut worker = stock_shaped_worker(store, None);
        let kinds = iteration_kinds(&mut worker, &[(0, 0), (500, 1)], 10_000);
        assert_eq!(kind_string(&kinds), "PPDD");
    }

    #[test]
    fn async_lookahead_inbox_counts_as_queued_until_the_next_plan() {
        let store = shared_with(&[(0, 4, 3), (1, 4, 3)]);
        let mut worker = stock_shaped_worker(store, Some(Time::from_us(300)));
        let mut events = Vec::new();
        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        assert_eq!(
            worker.tick(Time::ZERO, &mut events),
            Some(Time::from_us(300))
        );
        assert_eq!(
            worker.tick(Time::from_us(300), &mut events),
            Some(Time::from_ms(1.0))
        );
        worker.enqueue(WorkerMsgCommon::Request(RequestId(1)));
        assert_eq!(worker.status().queued_requests, 1);
        let admission_queued = |worker: &ChunkedPrefillWorker<FakeModel>| {
            IterAdmission::<KvCacheStore>::queued_requests(&worker.admission)
        };
        assert_eq!(admission_queued(&worker), 0, "admission has not seen it");
        // Step 1 (a solo decode) starts at 1 ms and plans step 2 at 1.3 ms.
        assert_eq!(
            worker.tick(Time::from_ms(1.0), &mut events),
            Some(Time::from_us(1_300))
        );
        worker.tick(Time::from_us(1_300), &mut events);
        assert_eq!(admission_queued(&worker), 1);
    }

    #[test]
    fn async_lookahead_makes_the_inbox_visible_when_the_planned_step_is_empty() {
        // Request 0 finishes with its prefill, so the step planned at 0.3 ms
        // is empty: the engine polls its queue and starts request 1 at 1 ms.
        let store = shared_with(&[(0, 4, 1), (1, 4, 2)]);
        let mut worker = stock_shaped_worker(store, Some(Time::from_us(300)));
        let kinds = iteration_kinds(&mut worker, &[(0, 0), (500, 1)], 10_000);
        assert_eq!(kinds, [(0, 'P'), (1_000, 'P'), (2_000, 'D')]);
    }

    #[test]
    fn async_lookahead_longer_than_a_step_waits_for_the_plan_moment() {
        // Step 0 ends at 1 ms but step 1 is planned only at 1.5 ms, so a
        // request arriving at 1.2 ms is still visible to step 1.
        let store = shared_with(&[(0, 4, 3), (1, 4, 3)]);
        let mut worker = stock_shaped_worker(store, Some(Time::from_us(1_500)));
        let kinds = iteration_kinds(&mut worker, &[(0, 0), (1_200, 1)], 10_000);
        assert_eq!(kinds[..2], [(0, 'P'), (1_500, 'P')]);
    }

    #[test]
    fn async_lookahead_from_idle_sets_only_the_plan_after_an_idle_start() {
        // Same late arrival (0.5 ms) as the solo-decode case, but the step
        // that started from idle plans its successor at 0.6 ms, so request 1
        // makes step 1.
        let store = shared_with(&[(0, 4, 3), (1, 4, 3)]);
        let mut worker = stock_shaped_worker_from_idle(
            store,
            Some(Time::from_us(300)),
            Some(Time::from_us(600)),
        );
        let kinds = iteration_kinds(&mut worker, &[(0, 0), (500, 1)], 10_000);
        assert_eq!(kind_string(&kinds), "PPDD");

        // Steps that follow a running predecessor keep the steady offset.
        let store = shared_with(&[(0, 4, 3)]);
        let mut worker = stock_shaped_worker_from_idle(
            store,
            Some(Time::from_us(300)),
            Some(Time::from_us(600)),
        );
        let mut events = Vec::new();
        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        assert_eq!(
            worker.tick(Time::ZERO, &mut events),
            Some(Time::from_us(600)),
            "step 0 started from idle"
        );
        worker.tick(Time::from_us(600), &mut events);
        assert_eq!(
            worker.tick(Time::from_ms(1.0), &mut events),
            Some(Time::from_us(1_300)),
            "step 1 follows step 0"
        );
    }

    #[test]
    fn async_lookahead_from_idle_applies_again_after_a_planned_empty_step() {
        // Request 0 finishes with its prefill, so the step planned at 0.6 ms
        // is empty and the engine goes idle; request 1 then starts from idle.
        let store = shared_with(&[(0, 4, 1), (1, 4, 3)]);
        let mut worker = stock_shaped_worker_from_idle(
            store,
            Some(Time::from_us(300)),
            Some(Time::from_us(600)),
        );
        let mut events = Vec::new();
        worker.enqueue(WorkerMsgCommon::Request(RequestId(0)));
        worker.tick(Time::ZERO, &mut events);
        worker.tick(Time::from_us(600), &mut events);
        worker.enqueue(WorkerMsgCommon::Request(RequestId(1)));
        assert_eq!(
            worker.tick(Time::from_ms(1.0), &mut events),
            Some(Time::from_us(1_600))
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
