//! `IterBatchWorker` — the whole-iteration L5 shell.
//!
//! The shell owns cadence and cross-axis sequencing. KV, Admission, and Execution
//! are statically composed; the FSM itself is deliberately not an axis.

use crate::common::{RequestId, Time, WorkerId};
use crate::worker::admission::{
    FifoOrder, IterAdmission, LocalPrefillDecodeAdmission, PrefillHandoffAdmission,
};
use crate::worker::execution::{IterModelExecution, UnifiedIterExecution};
use crate::worker::iter_worker::IterWorker;
use crate::worker::kv::{FullAttnKv, IterWorkerKv};
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::{BatchFsmState, IterCursor, WorkerFsmState, WorkerStatus};

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
    iteration_fsm: IterationFsm,
}

/// Compatibility name retained for L6 factories and deployment selectors.
pub type BareboneWorker<M> =
    IterBatchWorker<FullAttnKv, LocalPrefillDecodeAdmission<FifoOrder>, UnifiedIterExecution<M>>;
/// Multi-partition HP/DP recipe uses the same whole-iteration shell.
pub type HpUnifiedWorker<M> = BareboneWorker<M>;
pub type PdPrefillWorker<M> =
    IterBatchWorker<FullAttnKv, PrefillHandoffAdmission<FifoOrder>, UnifiedIterExecution<M>>;

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
            .form_batch(&mut self.kv_store, &self.context, now)
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
    use crate::common::{PoolId, SharedRequests};
    use crate::test_helpers::{shared_with, test_cluster, FakeModel};
    use crate::worker::admission::ShortestJobFirst;
    use crate::worker::types::{WorkerConfig, WorkerEventCommon, WorkerMsgCommon};
    use crate::worker::workers::iter::{build_barebone_worker, build_hp_worker};

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
        assert!(record.completed);
        assert_eq!(record.tokens_emitted, 3);
        assert!(record.first_token_time.is_some());
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
                store[RequestId(request)].completed,
                "req {request} should complete"
            );
        }
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
    fn none_budget_admits_one_prefill_per_iter() {
        let store = shared_with(&[(0, 8, 0), (1, 8, 0), (2, 8, 0)]);
        let mut worker = worker_with(store, WorkerConfig::default());
        for request in [0, 1, 2] {
            worker.enqueue(WorkerMsgCommon::Request(RequestId(request)));
        }

        worker.form_batch(Time::ZERO);
        assert_eq!(worker.kv_store.prefill_admits(0).count(), 1);
        assert_eq!(worker.admission.queued_requests(), 2);
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
    fn hp_input_has_one_group_per_partition() {
        let worker = hp_worker(shared_with(&[]), 3, WorkerConfig::default());
        let mut input = Default::default();
        worker.execution.build_iteration_input(
            &worker.kv_store,
            &worker.context.requests,
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
