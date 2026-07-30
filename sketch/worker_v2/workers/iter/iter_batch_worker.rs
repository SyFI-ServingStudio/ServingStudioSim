//! `IterBatchWorker<K, A, E>` — the S0 whole-iteration shell + the L6-facing surface.
//!
//! Generic over the three composable axes; owns the concrete iter FSM (data, not
//! a component) and the cross-axis sequencing. `build_barebone_worker` is the concrete
//! wiring for the dense / no-attn-DP composition, reproducing the real
//! `BareboneWorker::new` construction choreography (unified.rs:82) verbatim.

use crate::common::{RequestId, Time, WorkerId};
use crate::worker::iter_worker::IterWorker;
use crate::worker::types::{BatchFsmState, IterCursor, WorkerFsmState, WorkerStatus};

use super::super::super::admission::IterAdmission;
use super::super::super::execution::IterModelExecution;
use super::super::super::kv::{IterWorkerKv, KvStore};
use super::super::super::shared::context::WorkerContext;

/// Private execution state for this cadence. Deliberately data, not a component.
struct IterationFsm {
    worker_state: WorkerFsmState,
    batch_state: BatchFsmState,
    iter_counter: u32,
}

impl IterationFsm {
    fn new() -> Self {
        Self {
            worker_state: WorkerFsmState::Idle,
            batch_state: BatchFsmState {
                cursor: IterCursor::Done,
                compute_end: Time::ZERO,
            },
            iter_counter: 0,
        }
    }
}

pub struct IterBatchWorker<K, A, E>
where
    K: KvStore + IterWorkerKv,
    A: IterAdmission<K>,
    E: IterModelExecution<K>,
{
    context: WorkerContext,
    kv_store: K,
    admission: A,
    execution: E,
    input: <E as IterModelExecution<K>>::Input,
    iteration_fsm: IterationFsm,
}

impl<K, A, E> IterBatchWorker<K, A, E>
where
    K: KvStore + IterWorkerKv,
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
                    if !self
                        .admission
                        .form_batch(&mut self.kv_store, &self.context, now)
                    {
                        break;
                    }
                    self.iteration_fsm.iter_counter += 1;
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

    /// Build this model's input (IterModelExecution owns the builder, all partitions) and arm
    /// the iter end.
    fn start_iteration(&mut self, now: Time) -> Time {
        self.execution.build_iteration_input(
            &self.kv_store,
            &self.context.requests,
            &mut self.input,
        );
        let cost = self.execution.evaluate_iteration(
            &self.input,
            self.iteration_fsm.iter_counter as u64,
            now,
        );
        now + cost
    }

    fn next_wakeup(&self, now: Time) -> Option<Time> {
        match self.iteration_fsm.worker_state {
            WorkerFsmState::Idle => {
                let has_active_requests = (0..self.kv_store.num_partitions() as u16)
                    .any(|partition| self.kv_store.status_active(partition) > 0);
                let has_work = self.admission.queued_requests() > 0 || has_active_requests;
                has_work.then_some(now)
            }
            WorkerFsmState::Active => match self.iteration_fsm.batch_state.cursor {
                IterCursor::NotStarted | IterCursor::Done => Some(now),
                IterCursor::Computing => Some(self.iteration_fsm.batch_state.compute_end),
            },
        }
    }

    /// Cross-axis read: queued (IterAdmission) + active (KV, summed over partitions).
    pub fn status(&self) -> WorkerStatus {
        let active_requests = (0..self.kv_store.num_partitions() as u16)
            .map(|partition| self.kv_store.status_active(partition))
            .sum();
        WorkerStatus {
            queued_requests: self.admission.queued_requests(),
            active_requests,
        }
    }

    /// Cross-axis cancellation: pending half (IterAdmission) then admitted half (KV).
    pub fn cancel_request(&mut self, rid: RequestId, current_kv: u64) -> Option<u16> {
        if self.admission.cancel_pending(rid) {
            return None;
        }
        self.kv_store.release_external(rid, current_kv)
    }

    pub fn id(&self) -> WorkerId {
        self.context.id
    }
}

impl<K, A, E> IterWorker for IterBatchWorker<K, A, E>
where
    K: KvStore + IterWorkerKv,
    A: IterAdmission<K>,
    E: IterModelExecution<K>,
{
    type Msg = A::Msg;
    type Event = A::Event;

    fn id(&self) -> WorkerId {
        self.context.id
    }
    fn enqueue(&mut self, msg: Self::Msg) {
        IterBatchWorker::enqueue(self, msg)
    }
    fn tick(&mut self, now: Time, events: &mut Vec<Self::Event>) -> Option<Time> {
        IterBatchWorker::tick(self, now, events)
    }
    fn status(&self) -> WorkerStatus {
        IterBatchWorker::status(self)
    }
}
