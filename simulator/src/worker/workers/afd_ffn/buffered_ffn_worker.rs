//! `BufferedFfnWorker<E>` — production AFD-FFN shell.
//!
//! This family has no KV store and no admission lifecycle: L6 sends complete
//! `FfnTask`s. The shell overlaps at most one attention→FFN pull with at most one
//! FFN compute, queues excess tasks, consumes controller-aggregated task token
//! counts, and owns Terminal token/completion bookkeeping. `E` alone owns DP
//! input construction and model-section cost evaluation.
//!
//! Reading order: types → construction → `IterWorker` message dispatch → message
//! handler → double-buffer tick loop → pull/compute/completion helpers → wakeup.

use std::collections::VecDeque;

use crate::common::{AfdStage, Time, WorkerId};
use crate::worker::execution::{FfnSectionExecutionAdapter, FfnTaskExecution};
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::iter_worker::IterWorker;
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::{FfnTask, FfnTaskKind, FfnWorkerEvent, FfnWorkerMsg, WorkerStatus};

struct PullingTask {
    task: FfnTask,
    pull_end: Time,
}

struct ComputingTask {
    task: FfnTask,
    compute_end: Time,
}

pub struct BufferedFfnWorker<E: FfnTaskExecution> {
    context: WorkerContext,
    execution: E,
    input: E::Input,
    cluster: SharedGpuCluster,
    communication_group_id: u16,
    incoming: VecDeque<FfnTask>,
    pulling_task: Option<PullingTask>,
    computing_task: Option<ComputingTask>,
    iteration_by_slot: Vec<u64>,
}

/// Compatibility name retained at the L6 FFN pool surface.
pub type DisaggFfnWorker<M> = BufferedFfnWorker<FfnSectionExecutionAdapter<M>>;

impl<E: FfnTaskExecution> BufferedFfnWorker<E> {
    pub(super) fn from_components(
        context: WorkerContext,
        execution: E,
        cluster: SharedGpuCluster,
        communication_group_id: u16,
    ) -> Self {
        Self {
            context,
            execution,
            input: Default::default(),
            cluster,
            communication_group_id,
            incoming: VecDeque::new(),
            pulling_task: None,
            computing_task: None,
            iteration_by_slot: Vec::new(),
        }
    }
}

impl<E: FfnTaskExecution> IterWorker for BufferedFfnWorker<E> {
    type Msg = FfnWorkerMsg;
    type Event = FfnWorkerEvent;

    fn id(&self) -> WorkerId {
        self.context.id
    }

    fn enqueue(&mut self, msg: Self::Msg) {
        match msg {
            FfnWorkerMsg::Task(task) => self.on_msg_task(task),
        }
    }

    fn tick(&mut self, now: Time, events: &mut Vec<Self::Event>) -> Option<Time> {
        self.tick_inner(now, events)
    }

    fn status(&self) -> WorkerStatus {
        let queued = self.incoming.len() + usize::from(self.pulling_task.is_some());
        #[allow(
            clippy::cast_possible_truncation,
            reason = "queued request count is bounded by the simulated request population, far under u32::MAX"
        )]
        WorkerStatus {
            queued_requests: queued as u32,
            active_requests: u32::from(self.computing_task.is_some()),
        }
    }
}

impl<E: FfnTaskExecution> BufferedFfnWorker<E> {
    fn on_msg_task(&mut self, task: FfnTask) {
        self.incoming.push_back(task);
    }

    fn tick_inner(&mut self, now: Time, events: &mut Vec<FfnWorkerEvent>) -> Option<Time> {
        loop {
            let mut progressed = false;

            if self
                .computing_task
                .as_ref()
                .is_some_and(|task| now >= task.compute_end)
            {
                let task = self.computing_task.take().unwrap();
                self.complete_task(task.task, now, events);
                progressed = true;
            }

            if self.computing_task.is_none()
                && self
                    .pulling_task
                    .as_ref()
                    .is_some_and(|task| now >= task.pull_end)
            {
                let task = self.pulling_task.take().unwrap();
                self.computing_task = Some(self.start_compute(task.task, now));
                progressed = true;
            }

            if self.pulling_task.is_none() {
                if let Some(task) = self.incoming.pop_front() {
                    self.pulling_task = Some(self.start_pull(task, now));
                    progressed = true;
                }
            }

            if !progressed {
                break;
            }
        }
        self.next_wakeup(now)
    }

    fn start_pull(&mut self, task: FfnTask, now: Time) -> PullingTask {
        let sources: Vec<(u16, u64)> = task
            .pull_sources
            .iter()
            .filter(|source| source.bytes > 0)
            .map(|source| (source.send_gid, source.bytes))
            .collect();
        let pull_end = if sources.is_empty() {
            now
        } else {
            self.cluster.borrow_mut().submit_gather(
                now,
                &sources,
                self.communication_group_id,
                "afd_ffn_pull",
                "",
            )
        };
        PullingTask { task, pull_end }
    }

    fn start_compute(&mut self, task: FfnTask, now: Time) -> ComputingTask {
        let slot = task.slot as usize;
        if matches!(task.kind, FfnTaskKind::Bootstrap) {
            if slot >= self.iteration_by_slot.len() {
                self.iteration_by_slot.resize(slot + 1, 0);
            }
            self.iteration_by_slot[slot] += 1;
        }
        let iteration = self.iteration_by_slot.get(slot).copied().unwrap_or(0);
        let duration = if task.reqs.is_empty() {
            Time::ZERO
        } else {
            self.execution
                .build_task_input(task.tokens, &mut self.input);
            self.execution
                .evaluate_ffn_task(task.kind, task.slot, iteration, &self.input, now)
        };
        ComputingTask {
            task,
            compute_end: now + duration,
        }
    }

    fn complete_task(&mut self, task: FfnTask, now: Time, events: &mut Vec<FfnWorkerEvent>) {
        match task.kind {
            FfnTaskKind::Terminal => {
                let mut completed = Vec::new();
                {
                    let mut store = self.context.requests.borrow_mut();
                    for &request in &task.reqs {
                        let record = &mut store[request];
                        let owner = (
                            record.lifecycle.current_stage.pool,
                            record.lifecycle.current_stage.worker,
                        );
                        if record.progress.output_tokens_emitted == 0 {
                            record.record_first_token(now, self.context.log_output_token_times);
                            record.record_stage(
                                now,
                                AfdStage::Decode as u16,
                                owner.0,
                                owner.1,
                                self.context.log_stage_transitions,
                            );
                        } else {
                            record.record_token(now, self.context.log_output_token_times);
                        }
                        if record.is_complete() {
                            record.record_stage(
                                now,
                                AfdStage::Done as u16,
                                owner.0,
                                owner.1,
                                self.context.log_stage_transitions,
                            );
                            completed.push(request);
                        }
                    }
                }
                events.push(FfnWorkerEvent::IterComplete {
                    worker: self.context.id,
                    slot: task.slot,
                    reqs: task.reqs,
                    completed,
                });
            }
            FfnTaskKind::Bootstrap | FfnTaskKind::Bridge { .. } => {
                let out_bytes = self.execution.ffn_to_attn_bytes_per_token() * task.tokens;
                events.push(FfnWorkerEvent::SectionReady {
                    worker: self.context.id,
                    slot: task.slot,
                    kind: task.kind,
                    reqs: task.reqs,
                    out_send_gid: self.communication_group_id,
                    out_bytes,
                });
            }
        }
    }

    fn next_wakeup(&self, now: Time) -> Option<Time> {
        let mut next_wakeup = None;
        if let Some(task) = &self.computing_task {
            if task.compute_end > now {
                next_wakeup = Some(task.compute_end);
            }
        }
        if let Some(task) = &self.pulling_task {
            if task.pull_end > now {
                next_wakeup = Some(
                    next_wakeup.map_or(task.pull_end, |current: Time| current.min(task.pull_end)),
                );
            }
        }
        next_wakeup
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use std::sync::Arc;

    use super::*;
    use crate::common::{PoolId, RequestId, SharedRequests};
    use crate::test_helpers::{shared_with, test_cluster, FakeFfn};
    use crate::worker::build_afd_ffn_worker;
    use crate::worker::gpu_cluster::SharedGpuCluster;
    use crate::worker::types::{FfnPullSource, WorkerConfig};

    fn worker(store: SharedRequests, cluster: SharedGpuCluster) -> DisaggFfnWorker<FakeFfn> {
        build_afd_ffn_worker(
            WorkerId(0),
            Arc::new(FakeFfn { ms: 1.0, layers: 2 }),
            store,
            WorkerConfig::default(),
            PoolId(1),
            "test-gpu",
            cluster,
            None,
            "ffn",
        )
    }

    fn register_sender(cluster: &SharedGpuCluster) -> u16 {
        let mut cluster = cluster.borrow_mut();
        cluster.allocate(99, 99, 1, "attn-gpu", "attn");
        cluster.register_comm_group(0, 1, "attn", 99)
    }

    #[test]
    fn bootstrap_computes_without_pull_and_emits_qkv() {
        let store = shared_with(&[(0, 8, 4)]);
        let mut worker = worker(Rc::clone(&store), test_cluster());
        worker.enqueue(FfnWorkerMsg::Task(FfnTask {
            kind: FfnTaskKind::Bootstrap,
            slot: 0,
            reqs: vec![RequestId(0)],
            pull_sources: Vec::new(),
            tokens: 8,
        }));
        let mut events = Vec::new();
        for step in 0..10 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(
            events,
            vec![FfnWorkerEvent::SectionReady {
                worker: WorkerId(0),
                slot: 0,
                kind: FfnTaskKind::Bootstrap,
                reqs: vec![RequestId(0)],
                out_send_gid: worker.communication_group_id,
                out_bytes: 16,
            }]
        );
    }

    #[test]
    fn terminal_records_token_on_attention_owner() {
        let store = shared_with(&[(0, 8, 1)]);
        let attention_pool = PoolId(7);
        let attention_worker = WorkerId(3);
        store.borrow_mut()[RequestId(0)].record_stage(
            Time::ZERO,
            AfdStage::Prefill as u16,
            attention_pool,
            attention_worker,
            true,
        );
        let mut worker = worker(Rc::clone(&store), test_cluster());
        worker.context.log_stage_transitions = true;
        worker.enqueue(FfnWorkerMsg::Task(FfnTask {
            kind: FfnTaskKind::Terminal,
            slot: 0,
            reqs: vec![RequestId(0)],
            pull_sources: Vec::new(),
            tokens: 8,
        }));
        let mut events = Vec::new();
        for step in 0..10 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        let store = store.borrow();
        assert!(store[RequestId(0)].lifecycle.completed);
        assert!(store[RequestId(0)]
            .lifecycle
            .stage_log
            .iter()
            .all(|stage| { (stage.pool, stage.worker) == (attention_pool, attention_worker) }));
    }

    #[test]
    fn empty_terminal_finishes_in_same_tick() {
        let mut worker = worker(shared_with(&[]), test_cluster());
        worker.enqueue(FfnWorkerMsg::Task(FfnTask {
            kind: FfnTaskKind::Terminal,
            slot: 0,
            reqs: Vec::new(),
            pull_sources: Vec::new(),
            tokens: 0,
        }));
        let mut events = Vec::new();
        assert_eq!(worker.tick(Time::ZERO, &mut events), None);
        assert_eq!(
            events,
            vec![FfnWorkerEvent::IterComplete {
                worker: WorkerId(0),
                slot: 0,
                reqs: Vec::new(),
                completed: Vec::new(),
            }]
        );
    }

    #[test]
    fn next_pull_overlaps_current_compute() {
        let store = shared_with(&[(0, 8, 4), (1, 8, 4)]);
        let cluster = test_cluster();
        let sender_group_id = register_sender(&cluster);
        let mut worker = worker(Rc::clone(&store), Rc::clone(&cluster));
        worker.enqueue(FfnWorkerMsg::Task(FfnTask {
            kind: FfnTaskKind::Bootstrap,
            slot: 0,
            reqs: vec![RequestId(0)],
            pull_sources: Vec::new(),
            tokens: 8,
        }));
        worker.enqueue(FfnWorkerMsg::Task(FfnTask {
            kind: FfnTaskKind::Bridge { upstream: 0 },
            slot: 1,
            reqs: vec![RequestId(1)],
            pull_sources: vec![FfnPullSource {
                send_gid: sender_group_id,
                bytes: 4096,
            }],
            tokens: 8,
        }));
        let mut events = Vec::new();
        assert!(worker.tick(Time::ZERO, &mut events).is_some());
        assert!(worker.computing_task.is_some());
        assert!(worker.pulling_task.is_some());
    }
}
