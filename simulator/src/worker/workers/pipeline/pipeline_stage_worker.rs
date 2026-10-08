//! `PipelineStageWorker<E>` — a follower stage (1..N-1) of a pipeline replica.
//!
//! Owns: this stage's activation pull and compute timelines. Does not own: KV,
//! admission, or request state (the head owns them), or routing (L6). It stamps
//! no request stage, so every request stays located on its head.
//!
//! Cadence. Microbatches arrive in formation order and run FIFO. The stage
//! double-buffers: it pulls the next microbatch's activations while computing
//! the current one, so one pull can start only when the previous microbatch has
//! left the receive buffer for compute. All times are exact (computed from the
//! previous stage's `ready_at`), not rounded to the simulator tick:
//!
//! ```text
//! pull_start(k)    = max(ready_at(k), compute_start(k-1))
//! compute_start(k) = max(pull_end(k), compute_end(k-1))
//! ```
//!
//! A microbatch is reported (`StageDone`) once `now` reaches its compute end.
//!
//! Reading order: state types → struct → construction → `IterWorker` → message
//! handler → tick (schedule arrivals, report finished) → tests.

use std::collections::VecDeque;

use crate::common::{Time, WorkerId};
use crate::worker::execution::PipelineStageExecution;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::iter_worker::IterWorker;
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::{
    PipelineMicrobatch, PipelineStageEvent, PipelineStageMsg, WorkerStatus,
};

struct ScheduledMicrobatch {
    microbatch: PipelineMicrobatch,
    compute_end: Time,
}

pub struct PipelineStageWorker<E: PipelineStageExecution> {
    context: WorkerContext,
    execution: E,
    cluster: SharedGpuCluster,
    communication_group_id: u16,
    incoming: VecDeque<PipelineMicrobatch>,
    scheduled: VecDeque<ScheduledMicrobatch>,
    /// Start and end of the last scheduled compute.
    last_compute_start: Time,
    last_compute_end: Time,
}

impl<E: PipelineStageExecution> PipelineStageWorker<E> {
    pub(super) fn from_components(
        context: WorkerContext,
        execution: E,
        cluster: SharedGpuCluster,
        communication_group_id: u16,
    ) -> Self {
        Self {
            context,
            execution,
            cluster,
            communication_group_id,
            incoming: VecDeque::new(),
            scheduled: VecDeque::new(),
            last_compute_start: Time::ZERO,
            last_compute_end: Time::ZERO,
        }
    }
}

impl<E: PipelineStageExecution> IterWorker for PipelineStageWorker<E> {
    type Msg = PipelineStageMsg;
    type Event = PipelineStageEvent;

    fn id(&self) -> WorkerId {
        self.context.id
    }

    fn enqueue(&mut self, msg: Self::Msg) {
        match msg {
            PipelineStageMsg::Microbatch(microbatch) => self.on_msg_microbatch(microbatch),
        }
    }

    fn tick(&mut self, now: Time, events: &mut Vec<Self::Event>) -> Option<Time> {
        self.tick_inner(now, events)
    }

    fn status(&self) -> WorkerStatus {
        let started = !self.scheduled.is_empty();
        WorkerStatus {
            queued_requests: (self.incoming.len() + self.scheduled.len() - usize::from(started))
                as u32,
            active_requests: u32::from(started),
        }
    }
}

impl<E: PipelineStageExecution> PipelineStageWorker<E> {
    fn on_msg_microbatch(&mut self, microbatch: PipelineMicrobatch) {
        self.incoming.push_back(microbatch);
    }

    fn tick_inner(&mut self, now: Time, events: &mut Vec<PipelineStageEvent>) -> Option<Time> {
        while let Some(microbatch) = self.incoming.pop_front() {
            let scheduled = self.schedule_microbatch(microbatch);
            self.scheduled.push_back(scheduled);
        }
        while self
            .scheduled
            .front()
            .is_some_and(|scheduled| now >= scheduled.compute_end)
        {
            let ScheduledMicrobatch {
                mut microbatch,
                compute_end,
            } = self.scheduled.pop_front().unwrap();
            microbatch.send_gid = self.communication_group_id;
            microbatch.ready_at = compute_end;
            events.push(PipelineStageEvent::StageDone {
                worker: self.context.id,
                microbatch,
            });
        }
        self.scheduled
            .front()
            .map(|scheduled| scheduled.compute_end)
    }

    /// Fix this microbatch's pull and compute windows behind the previous one.
    fn schedule_microbatch(&mut self, microbatch: PipelineMicrobatch) -> ScheduledMicrobatch {
        let pull_start = microbatch.ready_at.max(self.last_compute_start);
        let pull_end = self.cluster.borrow_mut().submit_transfer(
            pull_start,
            microbatch.send_gid,
            self.communication_group_id,
            microbatch.activation_bytes,
            "pp_activation",
            "",
        );
        let compute_start = pull_end.max(self.last_compute_end);
        let cost = self
            .execution
            .evaluate_stage(&microbatch.input, microbatch.id, compute_start);
        self.last_compute_start = compute_start;
        self.last_compute_end = compute_start + cost;
        ScheduledMicrobatch {
            microbatch,
            compute_end: self.last_compute_end,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use std::sync::Arc;

    use super::*;
    use crate::arch::contract::{ArchGroupInput, UnifiedArchInput};
    use crate::common::PoolId;
    use crate::test_helpers::{shared_with, test_cluster, FakeModel};
    use crate::worker::types::WorkerConfig;
    use crate::worker::workers::pipeline::{build_pipeline_stage_worker, PipelineStage};

    fn assert_iter_worker<W: IterWorker>() {}

    #[test]
    fn the_follower_recipe_is_an_iter_worker() {
        assert_iter_worker::<PipelineStage<FakeModel>>();
    }

    fn stage(cluster: SharedGpuCluster) -> PipelineStage<FakeModel> {
        build_pipeline_stage_worker(
            WorkerId(1),
            "stage",
            Arc::new(FakeModel::for_ms(1.0)),
            shared_with(&[]),
            WorkerConfig::default(),
            None,
            PoolId(0),
            "test-gpu",
            cluster,
        )
    }

    fn microbatch(id: u64, send_gid: u16, ready_ms: f64) -> PipelineMicrobatch {
        let mut group = ArchGroupInput::default();
        group.prefill_chunk_pairs.push((0, 4));
        group.prefill_tokens = 4;
        group.batch_tokens = 4;
        PipelineMicrobatch {
            id,
            input: Rc::new(UnifiedArchInput {
                groups: vec![group],
                tokens_per_source_rank: Vec::new(),
            }),
            activation_bytes: 0,
            send_gid,
            ready_at: Time::from_ms(ready_ms),
        }
    }

    fn done(events: &[PipelineStageEvent]) -> Vec<(u64, u16, Time)> {
        events
            .iter()
            .map(|PipelineStageEvent::StageDone { microbatch, .. }| {
                (microbatch.id, microbatch.send_gid, microbatch.ready_at)
            })
            .collect()
    }

    #[test]
    fn microbatches_run_fifo_at_exact_times_and_report_this_stage_as_source() {
        let cluster = test_cluster();
        let source = {
            let mut cluster = cluster.borrow_mut();
            cluster.allocate(0, 0, 1, "test-gpu", "stage");
            cluster.register_comm_group(0, 1, "stage", 0)
        };
        let mut worker = stage(Rc::clone(&cluster));
        let own_gid = worker.communication_group_id;
        // The second arrives while the first computes, so it queues behind it.
        worker.enqueue(PipelineStageMsg::Microbatch(microbatch(1, source, 0.25)));
        worker.enqueue(PipelineStageMsg::Microbatch(microbatch(2, source, 0.5)));
        let mut events = Vec::new();
        assert_eq!(
            worker.tick(Time::from_ms(0.5), &mut events),
            Some(Time::from_ms(1.25))
        );
        assert!(events.is_empty());
        assert_eq!(worker.status().active_requests, 1);
        assert_eq!(worker.status().queued_requests, 1);
        assert_eq!(
            worker.tick(Time::from_ms(1.3), &mut events),
            Some(Time::from_ms(2.25))
        );
        assert_eq!(worker.tick(Time::from_ms(2.3), &mut events), None);
        assert_eq!(
            done(&events),
            vec![
                (1, own_gid, Time::from_ms(1.25)),
                (2, own_gid, Time::from_ms(2.25)),
            ]
        );
    }
}
