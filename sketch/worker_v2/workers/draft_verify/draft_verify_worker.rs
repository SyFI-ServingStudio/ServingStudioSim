//! `DraftVerifyWorker` — the S6 speculative cadence.
//!
//! Pipeline: admit/form mixed prefill+decode input → create tentative draft
//! proposals → evaluate draft+target verification → retain per-request outcomes
//! until compute end → commit accepted state and discard rejected state.

use crate::common::{RequestId, Time, WorkerId};
use crate::worker::iter_worker::IterWorker;
use crate::worker::types::{BatchFsmState, IterCursor, WorkerFsmState, WorkerStatus};

use super::super::super::admission::DraftVerifyAdmissionLifecycle;
use super::super::super::execution::{
    DraftVerifyModelExecution, DraftVerifyResult, SpeculativeProposal,
};
use super::super::super::kv::{KvStore, SpeculativeKv};
use super::super::super::shared::context::WorkerContext;

struct DraftVerifyFsm {
    worker_state: WorkerFsmState,
    batch_state: BatchFsmState,
    iteration: u32,
    result: Option<DraftVerifyResult>,
}

impl DraftVerifyFsm {
    fn new() -> Self {
        Self {
            worker_state: WorkerFsmState::Idle,
            batch_state: BatchFsmState {
                cursor: IterCursor::Done,
                compute_end: Time::ZERO,
            },
            iteration: 0,
            result: None,
        }
    }
}

pub struct DraftVerifyWorker<K, A, E>
where
    K: SpeculativeKv,
    A: DraftVerifyAdmissionLifecycle<K>,
    E: DraftVerifyModelExecution<K>,
{
    context: WorkerContext,
    kv_store: K,
    admission: A,
    execution: E,
    input: E::Input,
    proposals: Vec<SpeculativeProposal>,
    fsm: DraftVerifyFsm,
}

impl<K, A, E> DraftVerifyWorker<K, A, E>
where
    K: SpeculativeKv,
    A: DraftVerifyAdmissionLifecycle<K>,
    E: DraftVerifyModelExecution<K>,
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
            proposals: Vec::new(),
            fsm: DraftVerifyFsm::new(),
        }
    }

    pub fn enqueue(&mut self, message: A::Msg) {
        self.admission
            .accept_message(&mut self.kv_store, message, &self.context);
    }

    pub fn tick(&mut self, now: Time, events: &mut Vec<A::Event>) -> Option<Time> {
        use IterCursor::{Computing, Done, NotStarted};
        use WorkerFsmState::{Active, Idle};

        loop {
            match self.fsm.worker_state {
                Idle => {
                    if !self
                        .admission
                        .form_batch(&mut self.kv_store, &self.context, now)
                    {
                        break;
                    }
                    self.fsm.iteration += 1;
                    self.fsm.worker_state = Active;
                    self.fsm.batch_state.cursor = NotStarted;
                }
                Active => match self.fsm.batch_state.cursor {
                    NotStarted => {
                        self.start_draft_verify(now);
                        self.fsm.batch_state.cursor = Computing;
                        break;
                    }
                    Computing if now < self.fsm.batch_state.compute_end => break,
                    Computing => self.fsm.batch_state.cursor = Done,
                    Done => {
                        let result = self
                            .fsm
                            .result
                            .take()
                            .expect("active draft/verify iteration must retain its result");
                        self.admission.complete_draft_verify_iteration(
                            &mut self.kv_store,
                            &self.context,
                            result,
                            events,
                            now,
                        );
                        self.fsm.worker_state = Idle;
                    }
                },
            }
        }
        self.next_wakeup(now)
    }

    fn start_draft_verify(&mut self, now: Time) {
        self.execution.build_draft_verify_input(
            &self.kv_store,
            &self.context.requests,
            &mut self.input,
        );
        E::collect_proposals(&self.input, &mut self.proposals);
        for proposal in &self.proposals {
            self.kv_store.begin_proposal(
                proposal.request,
                proposal.partition,
                proposal.proposal_tokens,
            );
        }
        let result = self.execution.evaluate_draft_verify_iteration(
            &self.input,
            self.fsm.iteration as u64,
            now,
        );
        self.fsm.batch_state.compute_end = now + result.duration;
        self.fsm.result = Some(result);
    }

    fn next_wakeup(&self, now: Time) -> Option<Time> {
        match self.fsm.worker_state {
            WorkerFsmState::Idle => {
                let has_active_requests = (0..self.kv_store.num_partitions() as u16)
                    .any(|partition| self.kv_store.status_active(partition) > 0);
                (self.admission.queued_requests() > 0 || has_active_requests).then_some(now)
            }
            WorkerFsmState::Active => match self.fsm.batch_state.cursor {
                IterCursor::NotStarted | IterCursor::Done => Some(now),
                IterCursor::Computing => Some(self.fsm.batch_state.compute_end),
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

    pub fn cancel_request(&mut self, request: RequestId, current_kv: u64) -> Option<u16> {
        if self.admission.cancel_pending(request) {
            return None;
        }
        self.kv_store.release_external(request, current_kv)
    }

    pub fn id(&self) -> WorkerId {
        self.context.id
    }
}

impl<K, A, E> IterWorker for DraftVerifyWorker<K, A, E>
where
    K: SpeculativeKv,
    A: DraftVerifyAdmissionLifecycle<K>,
    E: DraftVerifyModelExecution<K>,
{
    type Msg = A::Msg;
    type Event = A::Event;

    fn id(&self) -> WorkerId {
        self.context.id
    }

    fn enqueue(&mut self, message: Self::Msg) {
        DraftVerifyWorker::enqueue(self, message)
    }

    fn tick(&mut self, now: Time, events: &mut Vec<Self::Event>) -> Option<Time> {
        DraftVerifyWorker::tick(self, now, events)
    }

    fn status(&self) -> WorkerStatus {
        DraftVerifyWorker::status(self)
    }
}
