//! `PullDecodeWorker<K, E>` — the decode half of a PD deployment.
//!
//! The shell owns two independent timelines: a serialized KV-pull pipeline and
//! a whole-iteration decode FSM. It does not own KV membership or model input
//! shape: `K` owns resident/request-partition accounting, while `E` builds and
//! evaluates the opaque iteration input. A landed handoff waits in
//! `pending_decodes`; at most one transfer is in flight, and landed plus
//! in-transit tokens share one pull-backlog budget. Completion acknowledges the
//! prefill worker only after KV has landed.
//!
//! Reading order: types → construction → message handling (the `IterWorker`
//! entry points) → the pull/decode tick loop → helpers in call order → tests.

use std::collections::VecDeque;

use crate::common::{PdStage, RequestId, Time, WorkerId};
use crate::worker::admission::LoadBalance;
use crate::worker::execution::{IterModelExecution, UnifiedIterExecution};
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::iter_worker::IterWorker;
use crate::worker::kv::{FullAttnKv, IterWorkerKv};
use crate::worker::shared::advance_scope::AdvanceScope;
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::{
    BatchFsmState, IterCursor, PdDecodeEvent, PdDecodeMsg, TransferPlan, WorkerFsmState,
    WorkerStatus,
};

/// Pull backlog ceiling as a fraction of one attention partition's token capacity.
pub(super) const PULL_BUDGET_FRACTION: f64 = 0.05;

#[derive(Clone, Copy, Debug)]
struct InFlightPull {
    request: RequestId,
    pull_end: Time,
    prefill_worker: WorkerId,
    tokens: u64,
}

struct PullPipeline {
    /// KV is resident locally but has not entered a decode partition.
    pending_decodes: VecDeque<RequestId>,
    /// O(1) token sum paired with every `pending_decodes` push/pop.
    pending_decode_tokens: u64,
    /// Handoffs not yet submitted; these tokens do not occupy local KV.
    pending_pulls: VecDeque<(RequestId, TransferPlan)>,
    /// The one serialized cluster transfer currently in flight.
    in_flight: Option<InFlightPull>,
    pull_budget_tokens: u64,
    cluster: SharedGpuCluster,
    receive_group_id: u16,
    kv_bytes_per_token: u64,
}

struct DecodeIterationFsm {
    worker_state: WorkerFsmState,
    batch_state: BatchFsmState,
    iteration: u32,
}

pub struct PullDecodeWorker<K, E>
where
    K: IterWorkerKv,
    E: IterModelExecution<K>,
{
    context: WorkerContext,
    kv_store: K,
    execution: E,
    input: E::Input,
    balance: LoadBalance,
    pull_pipeline: PullPipeline,
    decode_fsm: DecodeIterationFsm,
}

/// Compatibility name used by the existing L6 PD factory.
pub type PdDecodeWorker<M> = PullDecodeWorker<FullAttnKv, UnifiedIterExecution<M>>;

impl<K, E> PullDecodeWorker<K, E>
where
    K: IterWorkerKv,
    E: IterModelExecution<K>,
{
    #[allow(clippy::too_many_arguments)]
    pub(super) fn from_components(
        context: WorkerContext,
        kv_store: K,
        execution: E,
        balance: LoadBalance,
        cluster: SharedGpuCluster,
        receive_group_id: u16,
        pull_budget_tokens: u64,
        kv_bytes_per_token: u64,
    ) -> Self {
        Self {
            context,
            kv_store,
            execution,
            input: Default::default(),
            balance,
            pull_pipeline: PullPipeline {
                pending_decodes: VecDeque::new(),
                pending_decode_tokens: 0,
                pending_pulls: VecDeque::new(),
                in_flight: None,
                pull_budget_tokens,
                cluster,
                receive_group_id,
                kv_bytes_per_token,
            },
            decode_fsm: DecodeIterationFsm {
                worker_state: WorkerFsmState::Idle,
                batch_state: BatchFsmState {
                    cursor: IterCursor::Done,
                    compute_end: Time::ZERO,
                },
                iteration: 0,
            },
        }
    }
}

impl<K, E> IterWorker for PullDecodeWorker<K, E>
where
    K: IterWorkerKv,
    E: IterModelExecution<K>,
{
    type Msg = PdDecodeMsg;
    type Event = PdDecodeEvent;

    fn id(&self) -> WorkerId {
        self.context.id
    }

    fn enqueue(&mut self, msg: Self::Msg) {
        match msg {
            PdDecodeMsg::Request(request) => self.on_msg_request(request),
            PdDecodeMsg::Handoff {
                req,
                send_gid,
                tokens,
                prefill_worker,
            } => self.on_msg_handoff(req, send_gid, tokens, prefill_worker),
        }
    }

    fn tick(&mut self, now: Time, events: &mut Vec<Self::Event>) -> Option<Time> {
        self.tick_inner(now, events)
    }

    fn status(&self) -> WorkerStatus {
        let active_requests = (0..self.kv_store.num_partitions() as u16)
            .map(|partition| self.kv_store.live_decode_count(partition))
            .sum();
        let queued_requests = self.pull_pipeline.pending_decodes.len()
            + self.pull_pipeline.pending_pulls.len()
            + usize::from(self.pull_pipeline.in_flight.is_some());
        WorkerStatus {
            queued_requests: queued_requests as u32,
            active_requests,
        }
    }
}

impl<K, E> PullDecodeWorker<K, E>
where
    K: IterWorkerKv,
    E: IterModelExecution<K>,
{
    /// Direct-admit test path: the request's prompt KV is already resident.
    fn on_msg_request(&mut self, request: RequestId) {
        let tokens = {
            let store = self.context.requests.borrow();
            let record = &store[request];
            record.request.definition.prompt_tokens as u64
        };
        self.pull_pipeline.pending_decodes.push_back(request);
        self.pull_pipeline.pending_decode_tokens += tokens;

        let mut store = self.context.requests.borrow_mut();
        let arrival = store[request].request.core.arrival_time;
        self.context
            .stamp_stage(&mut store[request], arrival, PdStage::PendingDecode as u16);
    }

    /// Queue a prefill handoff for serialized transfer submission.
    fn on_msg_handoff(
        &mut self,
        request: RequestId,
        send_group_id: u16,
        tokens: u64,
        prefill_worker: WorkerId,
    ) {
        self.pull_pipeline.pending_pulls.push_back((
            request,
            TransferPlan {
                send_gid: send_group_id,
                recv_gid: self.pull_pipeline.receive_group_id,
                tokens,
                prefill_worker,
            },
        ));
    }

    fn tick_inner(&mut self, now: Time, events: &mut Vec<PdDecodeEvent>) -> Option<Time> {
        self.advance_pulls(now, events);
        use IterCursor::{Computing, Done, NotStarted};
        use WorkerFsmState::{Active, Idle};

        loop {
            let should_continue = match self.decode_fsm.worker_state {
                Idle => self.tick_idle(now),
                Active => match self.decode_fsm.batch_state.cursor {
                    NotStarted => self.tick_not_started(now),
                    Computing => self.tick_computing(now),
                    Done => self.tick_done(now, events),
                },
            };
            if !should_continue {
                break;
            }
        }
        self.next_wakeup(now)
    }

    /// Promote a landed pull, then submit the next handoff if the backlog permits.
    fn advance_pulls(&mut self, now: Time, events: &mut Vec<PdDecodeEvent>) {
        loop {
            if let Some(pull) = self.pull_pipeline.in_flight {
                if now < pull.pull_end {
                    return;
                }
                self.pull_pipeline.pending_decodes.push_back(pull.request);
                self.pull_pipeline.pending_decode_tokens += pull.tokens;
                self.pull_pipeline.in_flight = None;

                {
                    let mut store = self.context.requests.borrow_mut();
                    self.context.stamp_stage(
                        &mut store[pull.request],
                        now,
                        PdStage::PendingDecode as u16,
                    );
                }
                events.push(PdDecodeEvent::PullComplete {
                    worker: self.context.id,
                    req: pull.request,
                    prefill_worker: pull.prefill_worker,
                });
            }

            let Some(&(_, ref next_transfer)) = self.pull_pipeline.pending_pulls.front() else {
                return;
            };
            let head_tokens = next_transfer.tokens;
            let backlog_tokens = self.pull_backlog_tokens();
            let fits_normally =
                backlog_tokens + head_tokens <= self.pull_pipeline.pull_budget_tokens;
            let single_request_exception =
                backlog_tokens == 0 && head_tokens > self.pull_pipeline.pull_budget_tokens;
            if !fits_normally && !single_request_exception {
                return;
            }

            let (request, transfer) = self.pull_pipeline.pending_pulls.pop_front().unwrap();
            let bytes = transfer
                .tokens
                .saturating_mul(self.pull_pipeline.kv_bytes_per_token);
            let pull_end = self.pull_pipeline.cluster.borrow_mut().submit_transfer(
                now,
                transfer.send_gid,
                transfer.recv_gid,
                bytes,
                "pd_kv_pull",
                "",
            );
            self.pull_pipeline.in_flight = Some(InFlightPull {
                request,
                pull_end,
                prefill_worker: transfer.prefill_worker,
                tokens: transfer.tokens,
            });

            let mut store = self.context.requests.borrow_mut();
            self.context
                .stamp_stage(&mut store[request], now, PdStage::Transfer as u16);
        }
    }

    #[inline]
    fn pull_backlog_tokens(&self) -> u64 {
        self.pull_pipeline.pending_decode_tokens
            + self
                .pull_pipeline
                .in_flight
                .map(|pull| pull.tokens)
                .unwrap_or(0)
    }

    fn tick_idle(&mut self, now: Time) -> bool {
        if !self.form_batch(now) {
            return false;
        }
        self.decode_fsm.worker_state = WorkerFsmState::Active;
        self.decode_fsm.batch_state.cursor = IterCursor::NotStarted;
        self.decode_fsm.batch_state.compute_end = Time::ZERO;
        true
    }

    /// Admit at most one landed request directly into its chosen decode partition.
    fn form_batch(&mut self, now: Time) -> bool {
        let num_partitions = self.kv_store.num_partitions();
        let had_decode =
            (0..num_partitions as u16).any(|partition| self.kv_store.has_live_decode(partition));

        if let Some(&request) = self.pull_pipeline.pending_decodes.front() {
            let (prompt_kv, remaining) = {
                let store = self.context.requests.borrow();
                let record = &store[request];
                (
                    record.request.definition.prompt_tokens as u64,
                    record
                        .request
                        .definition
                        .target_output_tokens
                        .saturating_sub(record.progress.output_tokens_emitted),
                )
            };
            let partition = self.balance.choose(num_partitions) as u16;
            let footprint = self
                .kv_store
                .footprint(request, prompt_kv as u32, remaining);
            if remaining == 0 || self.kv_store.fits(partition, &footprint) {
                self.pull_pipeline.pending_decodes.pop_front();
                self.pull_pipeline.pending_decode_tokens = self
                    .pull_pipeline
                    .pending_decode_tokens
                    .saturating_sub(prompt_kv);
                // `reserve` makes the KV partition sticky; `commit_resident`
                // immediately clears the transient promise and enters decode.
                self.kv_store.reserve(request, partition, footprint);
                self.kv_store
                    .commit_resident(request, partition, prompt_kv, remaining);

                let mut store = self.context.requests.borrow_mut();
                self.context
                    .stamp_stage(&mut store[request], now, PdStage::Decode as u16);
            }
        }

        let still_decoding =
            (0..num_partitions as u16).any(|partition| self.kv_store.has_live_decode(partition));
        if !had_decode && !still_decoding {
            return false;
        }
        self.decode_fsm.iteration += 1;
        true
    }

    fn tick_not_started(&mut self, now: Time) -> bool {
        let compute_end = self.start_iteration(now);
        self.decode_fsm.batch_state.cursor = IterCursor::Computing;
        self.decode_fsm.batch_state.compute_end = compute_end;
        false
    }

    fn start_iteration(&mut self, now: Time) -> Time {
        self.execution.build_iteration_input(
            &self.kv_store,
            &self.context.requests,
            &mut self.input,
        );
        let cost =
            self.execution
                .evaluate_iteration(&self.input, self.decode_fsm.iteration as u64, now);
        now + cost
    }

    fn tick_computing(&mut self, now: Time) -> bool {
        if now < self.decode_fsm.batch_state.compute_end {
            return false;
        }
        self.decode_fsm.batch_state.cursor = IterCursor::Done;
        true
    }

    fn tick_done(&mut self, now: Time, events: &mut Vec<PdDecodeEvent>) -> bool {
        self.complete_iteration(now, events);
        self.decode_fsm.worker_state = WorkerFsmState::Idle;
        true
    }

    fn complete_iteration(&mut self, now: Time, events: &mut Vec<PdDecodeEvent>) {
        for partition in 0..self.kv_store.num_partitions() as u16 {
            let mut completed = Vec::new();
            {
                let mut store = self.context.requests.borrow_mut();
                self.kv_store.visit_decode_members(partition, |request, _| {
                    let record = &mut store[request];
                    record.record_token(now, self.context.log_tokens());
                    if record.is_complete() {
                        self.context.stamp_stage(record, now, PdStage::Done as u16);
                        completed.push(request);
                    }
                });
            }

            self.kv_store
                .advance(AdvanceScope::WholePartition(partition), 1);

            for request in completed {
                self.kv_store.release(request, partition);
                events.push(PdDecodeEvent::RequestComplete {
                    worker: self.context.id,
                    req: request,
                });
            }
            self.kv_store.sample_submit(partition, now);
        }
    }

    fn next_wakeup(&self, now: Time) -> Option<Time> {
        let pull_wakeup = self.pull_pipeline.in_flight.map(|pull| pull.pull_end);
        let compute_wakeup = match self.decode_fsm.worker_state {
            WorkerFsmState::Idle => {
                if self.pull_pipeline.pending_decodes.is_empty() {
                    None
                } else {
                    Some(now)
                }
            }
            WorkerFsmState::Active => match self.decode_fsm.batch_state.cursor {
                IterCursor::NotStarted | IterCursor::Done => Some(now),
                IterCursor::Computing => Some(self.decode_fsm.batch_state.compute_end),
            },
        };

        match (pull_wakeup, compute_wakeup) {
            (Some(pull), Some(compute)) => Some(pull.min(compute)),
            (Some(wakeup), None) | (None, Some(wakeup)) => Some(wakeup),
            (None, None) => None,
        }
    }

    /// Remove a request from the landed backlog or its sticky KV partition.
    pub fn release_request(&mut self, request: RequestId, current_kv: u64) -> Option<u16> {
        if let Some(position) = self
            .pull_pipeline
            .pending_decodes
            .iter()
            .position(|&pending| pending == request)
        {
            self.pull_pipeline.pending_decodes.remove(position);
            let tokens = {
                let store = self.context.requests.borrow();
                let record = &store[request];
                record.request.definition.prompt_tokens as u64
            };
            self.pull_pipeline.pending_decode_tokens = self
                .pull_pipeline
                .pending_decode_tokens
                .saturating_sub(tokens);
            return None;
        }
        self.kv_store.release_external(request, current_kv)
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use std::sync::Arc;

    use super::*;
    use crate::common::{PoolId, SharedRequests};
    use crate::test_helpers::{prefilled_store, test_cluster, FakeModel};
    use crate::worker::gpu_cluster::SharedGpuCluster;
    use crate::worker::types::WorkerConfig;
    use crate::worker::workers::pd_decode::build_pd_decode_worker;

    fn worker(store: SharedRequests) -> PdDecodeWorker<FakeModel> {
        worker_with_partitions(store, 1)
    }

    fn worker_with_partitions(
        store: SharedRequests,
        num_partitions: u16,
    ) -> PdDecodeWorker<FakeModel> {
        build_pd_decode_worker(
            WorkerId(0),
            "decode",
            Arc::new(FakeModel {
                ms: 1.0,
                dp_groups: num_partitions,
            }),
            store,
            WorkerConfig::default(),
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        )
    }

    #[test]
    fn handed_off_request_decodes_to_completion() {
        let store = prefilled_store(&[(0, 16, 3)]);
        let mut worker = worker(Rc::clone(&store));
        worker.enqueue(PdDecodeMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        for step in 0..50u64 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(
            events,
            vec![PdDecodeEvent::RequestComplete {
                worker: WorkerId(0),
                req: RequestId(0),
            }]
        );
        let store = store.borrow();
        let record = &store[RequestId(0)];
        assert!(record.lifecycle.completed);
        assert_eq!(record.progress.output_tokens_emitted, 3);
    }

    #[test]
    fn multiple_handed_off_requests_all_complete() {
        let store = prefilled_store(&[(0, 8, 2), (1, 8, 2), (2, 8, 2)]);
        let mut worker = worker(Rc::clone(&store));
        for request in 0..3 {
            worker.enqueue(PdDecodeMsg::Request(RequestId(request)));
        }
        let mut events = Vec::new();
        for step in 0..200u64 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(events.len(), 3);
        let store = store.borrow();
        for request in 0..3 {
            assert!(
                store[RequestId(request)].lifecycle.completed,
                "request {request} should complete"
            );
        }
    }

    #[test]
    fn builds_one_decode_partition_per_dp_shard() {
        let worker = worker_with_partitions(prefilled_store(&[]), 2);
        assert_eq!(worker.kv_store.num_partitions(), 2);
    }

    #[test]
    fn decode_round_robins_handoffs_across_partitions() {
        let store = prefilled_store(&[(0, 4, 50), (1, 4, 50)]);
        let mut worker = worker_with_partitions(store, 2);
        worker.enqueue(PdDecodeMsg::Request(RequestId(0)));
        worker.enqueue(PdDecodeMsg::Request(RequestId(1)));
        let mut events = Vec::new();
        for step in 0..10u64 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(
            (
                worker.kv_store.live_decode_count(0),
                worker.kv_store.live_decode_count(1),
            ),
            (1, 1)
        );
    }

    fn register_test_sender(cluster: &SharedGpuCluster) -> u16 {
        let mut cluster = cluster.borrow_mut();
        cluster.allocate(99, 99, 1, "sender-gpu", "prefill");
        cluster.register_comm_group(0, 1, "prefill", 99)
    }

    #[test]
    fn pull_backlog_gate_holds_second_handoff_until_first_drains() {
        let store = prefilled_store(&[(0, 40, 1), (1, 20, 1)]);
        let cluster = test_cluster();
        let sender_group_id = register_test_sender(&cluster);
        let mut worker = build_pd_decode_worker(
            WorkerId(0),
            "decode",
            Arc::new(FakeModel {
                ms: 1.0,
                dp_groups: 1,
            }),
            Rc::clone(&store),
            WorkerConfig {
                attn_kv_bytes: 1000,
                ..WorkerConfig::default()
            },
            None,
            PoolId(0),
            "test-gpu",
            Rc::clone(&cluster),
        );
        for request in [0u32, 1] {
            worker.enqueue(PdDecodeMsg::Handoff {
                req: RequestId(request),
                send_gid: sender_group_id,
                tokens: if request == 0 { 40 } else { 20 },
                prefill_worker: WorkerId(99),
            });
        }
        let mut events = Vec::new();
        worker.tick(Time::ZERO, &mut events);
        assert!(worker
            .pull_pipeline
            .in_flight
            .is_some_and(|pull| pull.request == RequestId(0)));
        assert_eq!(worker.pull_pipeline.pending_pulls.len(), 1);
    }

    #[test]
    fn pull_backlog_allows_one_oversized_request_when_empty() {
        let store = prefilled_store(&[(0, 200, 1)]);
        let cluster = test_cluster();
        let sender_group_id = register_test_sender(&cluster);
        let mut worker = build_pd_decode_worker(
            WorkerId(0),
            "decode",
            Arc::new(FakeModel {
                ms: 1.0,
                dp_groups: 1,
            }),
            Rc::clone(&store),
            WorkerConfig {
                attn_kv_bytes: 1000,
                ..WorkerConfig::default()
            },
            None,
            PoolId(0),
            "test-gpu",
            Rc::clone(&cluster),
        );
        worker.enqueue(PdDecodeMsg::Handoff {
            req: RequestId(0),
            send_gid: sender_group_id,
            tokens: 200,
            prefill_worker: WorkerId(99),
        });
        let mut events = Vec::new();
        worker.tick(Time::ZERO, &mut events);
        assert!(worker
            .pull_pipeline
            .in_flight
            .is_some_and(|pull| pull.request == RequestId(0)));
        assert!(worker.pull_pipeline.pending_pulls.is_empty());
    }

    #[test]
    fn partitioned_decode_requests_all_complete() {
        let store = prefilled_store(&[(0, 8, 2), (1, 8, 2), (2, 8, 2), (3, 8, 2)]);
        let mut worker = worker_with_partitions(Rc::clone(&store), 2);
        for request in 0..4u32 {
            worker.enqueue(PdDecodeMsg::Request(RequestId(request)));
        }
        let mut events = Vec::new();
        for step in 0..200u64 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(events.len(), 4);
        let store = store.borrow();
        for request in 0..4u32 {
            assert!(
                store[RequestId(request)].lifecycle.completed,
                "request {request} should complete"
            );
        }
    }
}
