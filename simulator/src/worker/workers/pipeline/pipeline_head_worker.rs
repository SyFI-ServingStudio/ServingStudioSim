//! `PipelineHeadWorker<K, A, E>` — stage 0 of a pipeline-parallel replica.
//!
//! Owns: request admission and KV for the whole pipeline, microbatch formation,
//! stage 0's compute timeline, the in-flight window, and request completion.
//! Does not own: later stages' compute or the activation transfers between
//! stages (each follower pulls its own input), or routing (L6).
//!
//! Cadence. Stage 0 runs one microbatch at a time. It forms the next one when it
//! is idle and fewer than `pipeline_depth` microbatches are in flight, so at most
//! one microbatch per stage exists, as in vLLM's batch queue. When stage 0
//! finishes, the shell emits `MicrobatchLaunched` and L6 hands the microbatch to
//! stage 1. When the last stage finishes, L6 sends `MicrobatchExit` back here.
//!
//! Invariants:
//! - `in_flight` holds one ticket per formed microbatch, in formation order,
//!   including the one stage 0 is computing. Exits arrive in the same order,
//!   because every stage is FIFO.
//! - KV progress is committed at formation. Token emission, the Done stage, and
//!   KV release happen at exit, stamped with the exact exit time.
//! - A head blocked on the in-flight window or on KV has no wakeup: only an exit
//!   or a new request can unblock it, and L6 ticks it on every message.
//!
//! Reading order: state types → struct → construction → `IterWorker` → message
//! handlers → tick (exits, stage-0 completion, formation) → wakeup → tests.

use std::collections::VecDeque;
use std::rc::Rc;

use crate::arch::contract::UnifiedArchInput;
use crate::common::{RequestId, Time, WorkerId};
use crate::worker::admission::MicrobatchAdmission;
use crate::worker::execution::IterModelExecution;
use crate::worker::iter_worker::IterWorker;
use crate::worker::kv::IterWorkerKv;
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::{
    IterBatchPlan, PipelineHeadEvent, PipelineHeadMsg, PipelineMicrobatch, WorkerStatus,
};

/// Activation size and depth of the pipeline this head feeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PipelineLayout {
    /// Number of stages, and so the maximum number of microbatches in flight.
    pub depth: u16,
    /// KV bytes per token on the stage holding the most KV. Every stage keeps
    /// the same blocks, so the stage with the most bytes per token bounds the
    /// pipeline's token capacity.
    pub kv_bytes_per_token: u64,
    /// Bytes one token's activations occupy between two stages.
    pub activation_bytes_per_token: u64,
}

struct ComputingMicrobatch {
    microbatch: PipelineMicrobatch,
    compute_end: Time,
}

pub struct PipelineHeadWorker<K, A, E>
where
    K: IterWorkerKv,
    A: MicrobatchAdmission<K>,
    E: IterModelExecution<K, Input = UnifiedArchInput>,
{
    context: WorkerContext,
    kv_store: K,
    admission: A,
    execution: E,
    layout: PipelineLayout,
    /// Communication group the next stage pulls stage 0's output from.
    send_gid: u16,
    /// Pipeline heads never run decode; the plan says so to input lowering.
    batch_plan: IterBatchPlan,
    next_microbatch: u64,
    computing: Option<ComputingMicrobatch>,
    in_flight: VecDeque<(u64, A::Ticket)>,
    /// Exits received since the last tick.
    exits: Vec<(u64, Time)>,
    completed: Vec<RequestId>,
}

impl<K, A, E> PipelineHeadWorker<K, A, E>
where
    K: IterWorkerKv,
    A: MicrobatchAdmission<K>,
    E: IterModelExecution<K, Input = UnifiedArchInput>,
{
    pub(super) fn from_components(
        context: WorkerContext,
        kv_store: K,
        admission: A,
        execution: E,
        layout: PipelineLayout,
        send_gid: u16,
    ) -> Self {
        assert!(layout.depth > 0, "a pipeline needs at least one stage");
        assert_eq!(
            kv_store.num_partitions(),
            1,
            "a pipeline head owns one attention partition"
        );
        let mut batch_plan = IterBatchPlan::default();
        batch_plan.reset_decode_participation(1, false);
        Self {
            context,
            kv_store,
            admission,
            execution,
            layout,
            send_gid,
            batch_plan,
            next_microbatch: 1,
            computing: None,
            in_flight: VecDeque::new(),
            exits: Vec::new(),
            completed: Vec::new(),
        }
    }

    pub fn send_gid(&self) -> u16 {
        self.send_gid
    }
}

impl<K, A, E> IterWorker for PipelineHeadWorker<K, A, E>
where
    K: IterWorkerKv,
    A: MicrobatchAdmission<K>,
    E: IterModelExecution<K, Input = UnifiedArchInput>,
{
    type Msg = PipelineHeadMsg;
    type Event = PipelineHeadEvent;

    fn id(&self) -> WorkerId {
        self.context.id
    }

    fn enqueue(&mut self, msg: Self::Msg) {
        match msg {
            PipelineHeadMsg::Request(request) => self.on_msg_request(request),
            PipelineHeadMsg::MicrobatchExit { microbatch, at } => {
                self.on_msg_microbatch_exit(microbatch, at)
            }
        }
    }

    fn tick(&mut self, now: Time, events: &mut Vec<Self::Event>) -> Option<Time> {
        self.tick_inner(now, events)
    }

    fn status(&self) -> WorkerStatus {
        WorkerStatus {
            queued_requests: self.admission.queued_requests(),
            active_requests: self.kv_store.status_active(0),
        }
    }
}

impl<K, A, E> PipelineHeadWorker<K, A, E>
where
    K: IterWorkerKv,
    A: MicrobatchAdmission<K>,
    E: IterModelExecution<K, Input = UnifiedArchInput>,
{
    fn on_msg_request(&mut self, request: RequestId) {
        self.admission
            .accept_request(&mut self.kv_store, request, &self.context);
    }

    fn on_msg_microbatch_exit(&mut self, microbatch: u64, at: Time) {
        self.exits.push((microbatch, at));
    }

    fn tick_inner(&mut self, now: Time, events: &mut Vec<PipelineHeadEvent>) -> Option<Time> {
        // Exits first: they release KV the next formation may need, and their
        // times precede `now`, so KV samples stay in time order.
        self.complete_exited_microbatches(events);
        loop {
            let mut progressed = false;
            if self
                .computing
                .as_ref()
                .is_some_and(|computing| now >= computing.compute_end)
            {
                let ComputingMicrobatch {
                    mut microbatch,
                    compute_end,
                } = self.computing.take().unwrap();
                microbatch.ready_at = compute_end;
                events.push(PipelineHeadEvent::MicrobatchLaunched {
                    worker: self.context.id,
                    microbatch,
                });
                progressed = true;
            }
            if self.computing.is_none()
                && self.in_flight.len() < usize::from(self.layout.depth)
                && self
                    .admission
                    .form_microbatch(&mut self.kv_store, &self.context, now)
            {
                self.computing = Some(self.launch_microbatch(now));
                progressed = true;
            }
            if !progressed {
                break;
            }
        }
        self.next_wakeup()
    }

    fn complete_exited_microbatches(&mut self, events: &mut Vec<PipelineHeadEvent>) {
        for (microbatch, at) in std::mem::take(&mut self.exits) {
            let (oldest, ticket) = self
                .in_flight
                .pop_front()
                .expect("microbatch exit without one in flight");
            assert_eq!(
                oldest, microbatch,
                "microbatches must leave the pipeline in formation order"
            );
            self.admission.complete_microbatch(
                &mut self.kv_store,
                &self.context,
                ticket,
                &mut self.completed,
                at,
            );
            events.extend(
                self.completed
                    .drain(..)
                    .map(|req| PipelineHeadEvent::RequestComplete {
                        worker: self.context.id,
                        req,
                    }),
            );
        }
    }

    /// Lower the scheduled chunks, commit their progress, and start stage 0.
    fn launch_microbatch(&mut self, now: Time) -> ComputingMicrobatch {
        let mut input = UnifiedArchInput::default();
        self.execution.build_iteration_input(
            &self.kv_store,
            &self.context.requests,
            &self.batch_plan,
            &mut input,
        );
        let id = self.next_microbatch;
        self.next_microbatch += 1;
        let ticket = self.admission.commit_microbatch(&mut self.kv_store, now);
        self.in_flight.push_back((id, ticket));
        let tokens: u64 = input
            .groups
            .iter()
            .map(|group| u64::from(group.batch_tokens))
            .sum();
        let cost = self.execution.evaluate_iteration(&input, id, now);
        ComputingMicrobatch {
            microbatch: PipelineMicrobatch {
                id,
                input: Rc::new(input),
                activation_bytes: tokens * self.layout.activation_bytes_per_token,
                send_gid: self.send_gid,
                ready_at: now + cost,
            },
            compute_end: now + cost,
        }
    }

    fn next_wakeup(&self) -> Option<Time> {
        self.computing
            .as_ref()
            .map(|computing| computing.compute_end)
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use std::sync::Arc;

    use super::*;
    use crate::common::{PoolId, SessionInput, SharedRequests, UnifiedStage};
    use crate::test_helpers::{shared_with, test_cluster, FakeModel};
    use crate::worker::admission::{FifoOrder, PipelinedPrefillAdmission};
    use crate::worker::execution::UnifiedIterExecution;
    use crate::worker::kv::FullAttnKv;
    use crate::worker::types::WorkerConfig;
    use crate::worker::workers::pipeline::{build_pipeline_head_worker, PipelineHead};

    fn assert_iter_worker<W: IterWorker>() {}

    #[test]
    fn pending_policy_is_a_composition_axis_for_the_pipeline_head() {
        assert_iter_worker::<
            PipelineHeadWorker<
                FullAttnKv,
                PipelinedPrefillAdmission<FifoOrder>,
                UnifiedIterExecution<FakeModel>,
            >,
        >();
    }

    const LAYOUT: PipelineLayout = PipelineLayout {
        depth: 2,
        kv_bytes_per_token: 2,
        activation_bytes_per_token: 4,
    };

    fn head(
        store: SharedRequests,
        max_batch_tokens: u32,
        attn_kv_bytes: u64,
    ) -> PipelineHead<FakeModel> {
        build_pipeline_head_worker(
            WorkerId(0),
            "stage",
            Arc::new(FakeModel::for_ms(1.0)),
            LAYOUT,
            store,
            WorkerConfig {
                max_batch_tokens: Some(max_batch_tokens),
                attn_kv_bytes,
                ..WorkerConfig::default()
            },
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        )
    }

    fn launched(events: &[PipelineHeadEvent]) -> Vec<(u64, u64, Time)> {
        events
            .iter()
            .filter_map(|event| match event {
                PipelineHeadEvent::MicrobatchLaunched { microbatch, .. } => Some((
                    microbatch.id,
                    microbatch.input.groups[0].prefill_tokens as u64,
                    microbatch.ready_at,
                )),
                PipelineHeadEvent::RequestComplete { .. } => None,
            })
            .collect()
    }

    fn completed(events: &[PipelineHeadEvent]) -> Vec<RequestId> {
        events
            .iter()
            .filter_map(|event| match event {
                PipelineHeadEvent::RequestComplete { req, .. } => Some(*req),
                PipelineHeadEvent::MicrobatchLaunched { .. } => None,
            })
            .collect()
    }

    #[test]
    fn a_long_prompt_is_sliced_across_consecutive_microbatches_up_to_the_depth() {
        let store = shared_with(&[(0, 10, 1)]);
        let mut worker = head(Rc::clone(&store), 4, 1_000);
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        for step in 0..10 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        // Two chunks of 4 are in flight; the third waits for an exit even though
        // stage 0 is idle and the prompt has 2 tokens left.
        assert_eq!(
            launched(&events),
            vec![(1, 4, Time::from_ms(1.0)), (2, 4, Time::from_ms(2.0))]
        );
        assert_eq!(worker.in_flight.len(), 2);
        assert_eq!(worker.tick(Time::from_ms(10.0), &mut events), None);
        assert_eq!(
            store.borrow()[RequestId(0)]
                .progress
                .prefill_tokens_processed,
            0
        );

        worker.enqueue(PipelineHeadMsg::MicrobatchExit {
            microbatch: 1,
            at: Time::from_ms(9.5),
        });
        events.clear();
        for step in 10..20 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(launched(&events), vec![(3, 2, Time::from_ms(11.0))]);
        assert_eq!(
            store.borrow()[RequestId(0)]
                .progress
                .prefill_tokens_processed,
            4
        );
        assert!(completed(&events).is_empty());
    }

    #[test]
    fn first_token_is_stamped_at_last_stage_exit_and_releases_kv() {
        let store = shared_with(&[(0, 4, 1)]);
        let mut worker = head(Rc::clone(&store), 8, 1_000);
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        worker.tick(Time::ZERO, &mut events);
        assert_eq!(worker.status().active_requests, 1);
        worker.tick(Time::from_ms(1.0), &mut events);
        assert_eq!(launched(&events), vec![(1, 4, Time::from_ms(1.0))]);

        worker.enqueue(PipelineHeadMsg::MicrobatchExit {
            microbatch: 1,
            at: Time::from_ms(2.25),
        });
        events.clear();
        assert_eq!(worker.tick(Time::from_ms(2.3), &mut events), None);
        assert_eq!(completed(&events), vec![RequestId(0)]);
        let store = store.borrow();
        let record = &store[RequestId(0)];
        assert_eq!(
            record.telemetry.first_output_time,
            Some(Time::from_ms(2.25))
        );
        assert_eq!(record.progress.prefill_tokens_processed, 4);
        assert_eq!(
            record.lifecycle.current_stage.code,
            UnifiedStage::Done as u16
        );
        drop(store);
        assert_eq!(worker.status().active_requests, 0);
    }

    #[test]
    fn kv_capacity_counts_in_flight_requests_until_they_exit() {
        // 20 tokens of capacity: one 12-token prompt fits, a second does not
        // until the first leaves the pipeline.
        let store = shared_with(&[(0, 12, 1), (1, 12, 1)]);
        let mut worker = head(Rc::clone(&store), 16, 40);
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        worker.enqueue(PipelineHeadMsg::Request(RequestId(1)));
        let mut events = Vec::new();
        for step in 0..5 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(launched(&events), vec![(1, 12, Time::from_ms(1.0))]);
        assert_eq!(worker.status().queued_requests, 1);

        worker.enqueue(PipelineHeadMsg::MicrobatchExit {
            microbatch: 1,
            at: Time::from_ms(4.5),
        });
        events.clear();
        for step in 5..7 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(completed(&events), vec![RequestId(0)]);
        assert_eq!(launched(&events), vec![(2, 12, Time::from_ms(6.0))]);
    }

    #[test]
    fn a_pinned_prefix_is_context_for_every_chunk_but_never_computed() {
        // 2 KV bytes per token: 120 tokens of capacity. The pinned request needs
        // 100 + 10 + 1 of them, so the 12-token standalone prompt queued behind
        // it waits for its exit.
        let store = shared_with(&[(0, 10, 1), (1, 12, 1)]);
        store.borrow_mut()[RequestId(0)].request.definition.session =
            SessionInput::PinnedPrefix { prefix_tokens: 100 };
        let mut worker = head(Rc::clone(&store), 4, 240);
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        worker.enqueue(PipelineHeadMsg::Request(RequestId(1)));
        let mut events = Vec::new();
        let mut chunk_pairs = Vec::new();
        let collect = |events: &[PipelineHeadEvent], chunk_pairs: &mut Vec<(u32, u32)>| {
            for event in events {
                if let PipelineHeadEvent::MicrobatchLaunched { microbatch, .. } = event {
                    chunk_pairs.extend(microbatch.input.groups[0].prefill_chunk_pairs.iter());
                }
            }
        };
        for step in 0..5 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        for (microbatch, at) in [(1, 5.0), (2, 6.0)] {
            worker.enqueue(PipelineHeadMsg::MicrobatchExit {
                microbatch,
                at: Time::from_ms(at),
            });
            worker.tick(Time::from_ms(at), &mut events);
        }
        collect(&events, &mut chunk_pairs);
        assert_eq!(chunk_pairs, [(100, 4), (104, 4), (108, 2)]);
        assert!(!store.borrow()[RequestId(1)].lifecycle.admitted);

        events.clear();
        worker.enqueue(PipelineHeadMsg::MicrobatchExit {
            microbatch: 3,
            at: Time::from_ms(8.0),
        });
        worker.tick(Time::from_ms(8.0), &mut events);
        assert_eq!(completed(&events), vec![RequestId(0)]);
        assert!(store.borrow()[RequestId(1)].lifecycle.admitted);
        let requests = store.borrow();
        let record = &requests[RequestId(0)];
        assert_eq!(record.progress.prefill_tokens_processed, 10);
        assert_eq!(record.telemetry.prefix_cache_hit_tokens, Some(100));
    }

    #[test]
    #[should_panic(expected = "models prefill only")]
    fn a_decode_request_is_rejected_at_enqueue() {
        let store = shared_with(&[(0, 4, 2)]);
        let mut worker = head(store, 8, 1_000);
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
    }
}
