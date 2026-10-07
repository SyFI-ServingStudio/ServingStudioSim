//! `PipelineHeadWorker<K, A, E>` — stage 0 of a pipeline-parallel replica.
//!
//! Owns: request admission and KV for the whole pipeline, microbatch formation,
//! stage 0's compute timeline, the in-flight window, and request completion.
//! Does not own: later stages' compute or the activation transfers between
//! stages (each follower pulls its own input), or routing (L6).
//!
//! Cadence. Stage 0 runs one microbatch at a time. It forms the next one when it
//! is idle and fewer than `pipeline_depth` microbatches are in flight, so at most
//! one microbatch per stage exists, as in vLLM's batch queue. It starts that
//! microbatch at the exact time the state allowing it arose (stage 0's finish,
//! an exit, or a request's arrival tick), so stage 0, like the followers, is
//! not rounded to the simulator tick. A microbatch may mix
//! prefill chunks and decode steps; admission writes which resident decodes it
//! carries into the batch plan. When stage 0 finishes, the shell emits
//! `MicrobatchLaunched` and L6 hands the microbatch to stage 1. When the last
//! stage finishes, L6 sends `MicrobatchExit` back here.
//!
//! Invariants:
//! - `in_flight` holds one ticket per formed microbatch, in formation order,
//!   including the one stage 0 is computing. Exits arrive in the same order,
//!   because every stage is FIFO.
//! - Prefill progress is committed at formation. Token emission, decode KV
//!   advance, the Done stage, and KV release happen at exit, stamped with the
//!   exact exit time. A request has at most one microbatch in flight once its
//!   prompt is fully scheduled.
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
    /// The decodes the microbatch being formed carries, for input lowering.
    batch_plan: IterBatchPlan,
    next_microbatch: u64,
    computing: Option<ComputingMicrobatch>,
    in_flight: VecDeque<(u64, A::Ticket)>,
    /// Exits received since the last tick.
    exits: Vec<(u64, Time)>,
    /// Whether a request arrived since the last tick.
    request_arrived: bool,
    /// Latest time anything the next formation depends on changed: stage 0
    /// finishing, a microbatch exiting, or a request arriving (at the tick that
    /// delivered it). Nothing changes between this and `now`, so the next
    /// microbatch forms and starts here, not on the tick grid.
    state_changed_at: Time,
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
            request_arrived: false,
            state_changed_at: Time::ZERO,
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
        self.request_arrived = true;
    }

    fn on_msg_microbatch_exit(&mut self, microbatch: u64, at: Time) {
        self.exits.push((microbatch, at));
    }

    fn tick_inner(&mut self, now: Time, events: &mut Vec<PipelineHeadEvent>) -> Option<Time> {
        if std::mem::take(&mut self.request_arrived) {
            self.state_changed_at = self.state_changed_at.max(now);
        }
        // Exits first: they release KV the next formation may need, and their
        // times precede the formation's, so KV samples stay in time order.
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
                self.state_changed_at = self.state_changed_at.max(compute_end);
                events.push(PipelineHeadEvent::MicrobatchLaunched {
                    worker: self.context.id,
                    microbatch,
                });
                progressed = true;
            }
            let start = self.state_changed_at;
            debug_assert!(start <= now, "formation state changed after now");
            if self.computing.is_none()
                && self.in_flight.len() < usize::from(self.layout.depth)
                && self.admission.form_microbatch(
                    &mut self.kv_store,
                    &self.context,
                    &mut self.batch_plan,
                    start,
                )
            {
                self.computing = Some(self.launch_microbatch(start));
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
            self.state_changed_at = self.state_changed_at.max(at);
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

    /// Lower the scheduled chunks and decodes, commit them, and start stage 0
    /// at `start`.
    fn launch_microbatch(&mut self, start: Time) -> ComputingMicrobatch {
        let mut input = UnifiedArchInput::default();
        self.execution.build_iteration_input(
            &self.kv_store,
            &self.context.requests,
            &self.batch_plan,
            &mut input,
        );
        let id = self.next_microbatch;
        self.next_microbatch += 1;
        let ticket = self.admission.commit_microbatch(&mut self.kv_store, start);
        self.in_flight.push_back((id, ticket));
        let tokens: u64 = input
            .groups
            .iter()
            .map(|group| u64::from(group.batch_tokens))
            .sum();
        let cost = self.execution.evaluate_iteration(&input, id, start);
        ComputingMicrobatch {
            microbatch: PipelineMicrobatch {
                id,
                input: Rc::new(input),
                activation_bytes: tokens * self.layout.activation_bytes_per_token,
                send_gid: self.send_gid,
                ready_at: start + cost,
            },
            compute_end: start + cost,
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
    use crate::worker::admission::{FifoOrder, PipelinedChunkedPrefillAdmission};
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
                PipelinedChunkedPrefillAdmission<FifoOrder>,
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
        head_with(
            store,
            WorkerConfig {
                max_batch_tokens: Some(max_batch_tokens),
                attn_kv_bytes,
                ..WorkerConfig::default()
            },
        )
    }

    fn head_with(store: SharedRequests, config: WorkerConfig) -> PipelineHead<FakeModel> {
        build_pipeline_head_worker(
            WorkerId(0),
            "stage",
            Arc::new(FakeModel::for_ms(1.0)),
            LAYOUT,
            store,
            config,
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
        // The exit at 9.5 frees the window; stage 0 starts then, off the tick
        // grid, and finishes at 10.5.
        assert_eq!(launched(&events), vec![(3, 2, Time::from_ms(10.5))]);
        assert_eq!(
            store.borrow()[RequestId(0)]
                .progress
                .prefill_tokens_processed,
            4
        );
        assert!(completed(&events).is_empty());
    }

    #[test]
    fn stage0_starts_the_next_microbatch_when_it_finishes_not_on_the_tick() {
        let store = shared_with(&[(0, 8, 1)]);
        let mut worker = head(Rc::clone(&store), 4, 1_000);
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        worker.tick(Time::ZERO, &mut events);
        // Stage 0 finishes the first chunk at 1.0 ms; the head sees it at 1.3.
        assert_eq!(
            worker.tick(Time::from_ms(1.3), &mut events),
            Some(Time::from_ms(2.0))
        );
        assert_eq!(
            launched(&events),
            vec![(1, 4, Time::from_ms(1.0))],
            "the second chunk started at 1.0 and finishes at 2.0"
        );
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
        // The freed KV admits the second prompt at the exit, not at the tick.
        assert_eq!(launched(&events), vec![(2, 12, Time::from_ms(5.5))]);
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

    fn hybrid_head(
        store: SharedRequests,
        max_batch_tokens: u32,
        attn_kv_bytes: u64,
        block_aligned_chunks: bool,
    ) -> crate::worker::workers::pipeline::HybridPipelineHead<FakeModel> {
        use crate::worker::kv::{PrefixCacheConfig, PrefixCachePolicy};
        use crate::worker::workers::pipeline::{
            build_hybrid_pipeline_head_worker, PipelineHybridState,
        };
        build_hybrid_pipeline_head_worker(
            WorkerId(0),
            "stage",
            Arc::new(FakeModel::for_ms(1.0)),
            PipelineLayout { depth: 3, ..LAYOUT },
            // 4-token blocks of 2 x 4 = 8 bytes; 2 fixed blocks per request.
            PipelineHybridState {
                block_tokens: 4,
                state_blocks_per_request: 2,
                block_aligned_chunks,
            },
            store,
            WorkerConfig {
                max_batch_tokens: Some(max_batch_tokens),
                attn_kv_bytes,
                prefix_cache: PrefixCacheConfig::Opportunistic {
                    policy: PrefixCachePolicy::Lru,
                    max_retained_bytes: None,
                },
                ..WorkerConfig::default()
            },
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        )
    }

    #[test]
    fn a_hybrid_head_sizes_one_block_pool_and_charges_fixed_state() {
        use crate::worker::workers::pipeline::PipelineHybridState;
        let hybrid = PipelineHybridState {
            block_tokens: 4,
            state_blocks_per_request: 2,
            block_aligned_chunks: true,
        };
        // 80 bytes / 8 per block = 10 blocks, less vLLM's null block.
        assert_eq!(hybrid.capacity_tokens(&LAYOUT, 80), 36);
        assert_eq!(hybrid.state_tokens_per_request(), 8);

        // Each 12-token prompt costs 12 + 8: one fits in 36 tokens, two do not.
        let store = shared_with(&[(0, 12, 1), (1, 12, 1)]);
        let mut worker = hybrid_head(Rc::clone(&store), 16, 80, true);
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        worker.enqueue(PipelineHeadMsg::Request(RequestId(1)));
        let mut events = Vec::new();
        for step in 0..3 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(launched(&events), vec![(1, 12, Time::from_ms(1.0))]);
        assert_eq!(worker.status().queued_requests, 1);
    }

    #[test]
    fn a_hybrid_head_ends_non_final_chunks_on_checkpoint_boundaries() {
        // Budget 6, blocks of 4: the first chunk floors to 4, the second stops
        // at the prompt's last boundary (8), the tail runs alone.
        assert_eq!(hybrid_chunks(true), [4, 4, 2]);
    }

    #[test]
    fn a_hybrid_head_decodes_and_frees_its_state_blocks_for_the_next_request() {
        // 36 tokens of blocks; each request needs 12 + 3 tokens plus 8 of fixed
        // state, so the second starts only once the first has decoded and left.
        let store = shared_with(&[(0, 12, 3), (1, 12, 3)]);
        let mut worker = hybrid_head(Rc::clone(&store), 16, 80, true);
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        worker.enqueue(PipelineHeadMsg::Request(RequestId(1)));
        let mut events = Vec::new();
        let mut done = Vec::new();
        let mut first_launch = None;
        let mut now = 0.0;
        while done.len() < 2 && now < 100.0 {
            events.clear();
            worker.tick(Time::from_ms(now), &mut events);
            for (microbatch, prefill, _) in launched(&events) {
                if prefill == 12 && first_launch.is_some() {
                    assert_eq!(done, [RequestId(0)], "the second prompt waits for blocks");
                }
                first_launch.get_or_insert(microbatch);
                worker.enqueue(PipelineHeadMsg::MicrobatchExit {
                    microbatch,
                    at: Time::from_ms(now + 1.0),
                });
            }
            done.extend(completed(&events));
            now += 1.0;
        }
        assert_eq!(done, [RequestId(0), RequestId(1)]);
        for id in [RequestId(0), RequestId(1)] {
            assert_eq!(store.borrow()[id].progress.output_tokens_emitted, 3);
        }
        assert_eq!(worker.status().active_requests, 0);
    }

    #[test]
    fn a_hybrid_head_without_block_alignment_chunks_plainly() {
        // Same prompt and budget: each chunk is min(remaining, budget).
        assert_eq!(hybrid_chunks(false), [6, 4]);
    }

    /// Chunk sizes a hybrid head launches for one 10-token prompt at budget 6
    /// with 4-token blocks.
    fn hybrid_chunks(block_aligned_chunks: bool) -> Vec<u64> {
        let store = shared_with(&[(0, 10, 1)]);
        let mut worker = hybrid_head(Rc::clone(&store), 6, 1_000, block_aligned_chunks);
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        for step in 0..6 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        launched(&events)
            .into_iter()
            .map(|(_, tokens, _)| tokens)
            .collect()
    }

    fn launched_decode_kv(events: &[PipelineHeadEvent]) -> Vec<(u64, Vec<u32>)> {
        events
            .iter()
            .filter_map(|event| match event {
                PipelineHeadEvent::MicrobatchLaunched { microbatch, .. } => Some((
                    microbatch.id,
                    microbatch.input.groups[0].decode_kv_lens.clone(),
                )),
                PipelineHeadEvent::RequestComplete { .. } => None,
            })
            .collect()
    }

    #[test]
    fn a_decode_step_waits_for_the_previous_step_to_leave_the_pipeline() {
        // Prompt 4, three output tokens: the first comes with the prompt's exit,
        // each later one needs the previous token, so no two of this request's
        // steps are ever in flight together even though the depth allows two.
        let store = shared_with(&[(0, 4, 3)]);
        let mut worker = head(Rc::clone(&store), 8, 1_000);
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        worker.tick(Time::ZERO, &mut events);
        worker.tick(Time::from_ms(1.0), &mut events);
        assert_eq!(launched(&events), vec![(1, 4, Time::from_ms(1.0))]);
        assert_eq!(worker.tick(Time::from_ms(1.5), &mut events), None);

        let mut decode_kv = Vec::new();
        for (microbatch, exit_ms) in [(1, 2.0), (2, 4.0), (3, 6.0)] {
            worker.enqueue(PipelineHeadMsg::MicrobatchExit {
                microbatch,
                at: Time::from_ms(exit_ms),
            });
            events.clear();
            worker.tick(Time::from_ms(exit_ms), &mut events);
            worker.tick(Time::from_ms(exit_ms + 1.0), &mut events);
            decode_kv.extend(launched_decode_kv(&events));
            assert!(
                worker.in_flight.len() <= 1,
                "one request, one step in flight"
            );
            if microbatch == 3 {
                assert_eq!(completed(&events), vec![RequestId(0)]);
            } else {
                assert!(completed(&events).is_empty());
            }
        }
        // Two decode steps, each over one more resident token than the last.
        assert_eq!(decode_kv.len(), 2);
        assert_eq!(decode_kv[1].1[0], decode_kv[0].1[0] + 1);

        let requests = store.borrow();
        let record = &requests[RequestId(0)];
        assert_eq!(record.progress.output_tokens_emitted, 3);
        assert_eq!(record.telemetry.first_output_time, Some(Time::from_ms(2.0)));
        assert_eq!(
            record.lifecycle.current_stage.code,
            UnifiedStage::Done as u16
        );
        drop(requests);
        assert_eq!(worker.status().active_requests, 0);
    }

    #[test]
    fn requests_ready_at_different_exits_decode_in_different_microbatches() {
        // A 2-token budget puts each prompt in its own microbatch, so the two
        // requests become ready at different exits and then alternate: each
        // decode microbatch carries only the request whose step returned.
        let store = shared_with(&[(0, 2, 3), (1, 2, 3)]);
        let mut worker = head(Rc::clone(&store), 2, 1_000);
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        worker.enqueue(PipelineHeadMsg::Request(RequestId(1)));
        let mut events = Vec::new();
        worker.tick(Time::ZERO, &mut events);
        worker.tick(Time::from_ms(1.0), &mut events);
        worker.tick(Time::from_ms(2.0), &mut events);
        assert_eq!(worker.in_flight.len(), 2);

        let mut decode_rows = Vec::new();
        let mut done = Vec::new();
        let mut now = 2.0;
        for microbatch in 1..=6 {
            now += 1.0;
            worker.enqueue(PipelineHeadMsg::MicrobatchExit {
                microbatch,
                at: Time::from_ms(now),
            });
            events.clear();
            worker.tick(Time::from_ms(now), &mut events);
            decode_rows.extend(
                launched_decode_kv(&events)
                    .into_iter()
                    .map(|(_, kv)| kv.len()),
            );
            done.extend(completed(&events));
            if microbatch < 6 {
                // Each exit is followed by the head's next launch.
                worker.tick(Time::from_ms(now + 1.0), &mut events);
                decode_rows.extend(
                    launched_decode_kv(&events)
                        .into_iter()
                        .map(|(_, kv)| kv.len()),
                );
            }
        }
        assert!(decode_rows.iter().all(|&rows| rows <= 1), "{decode_rows:?}");
        assert_eq!(done, vec![RequestId(0), RequestId(1)]);
        let requests = store.borrow();
        for request in [RequestId(0), RequestId(1)] {
            assert_eq!(requests[request].progress.output_tokens_emitted, 3);
        }
    }

    /// Run four 1-token prompts with four outputs each through a depth-2 head
    /// whose microbatches exit 2 ms after launch. Returns each decode
    /// microbatch's request count and the completion order.
    fn decode_microbatch_sizes(balance: bool) -> (Vec<usize>, Vec<RequestId>) {
        let store = shared_with(&[(0, 1, 4), (1, 1, 4), (2, 1, 4), (3, 1, 4)]);
        let mut worker = head_with(
            Rc::clone(&store),
            WorkerConfig {
                max_batch_tokens: Some(8),
                attn_kv_bytes: 1_000,
                balance_decode_microbatches: balance,
                ..WorkerConfig::default()
            },
        );
        for id in 0..4 {
            worker.enqueue(PipelineHeadMsg::Request(RequestId(id)));
        }
        let (mut sizes, mut done, mut exits) = (Vec::new(), Vec::new(), Vec::new());
        let mut events = Vec::new();
        for step in 0..200 {
            let now = Time::from_ms(f64::from(step) * 0.5);
            exits.retain(|&(microbatch, at): &(u64, Time)| {
                if at > now {
                    return true;
                }
                worker.enqueue(PipelineHeadMsg::MicrobatchExit { microbatch, at });
                false
            });
            events.clear();
            worker.tick(now, &mut events);
            for event in &events {
                if let PipelineHeadEvent::MicrobatchLaunched { microbatch, .. } = event {
                    exits.push((microbatch.id, microbatch.ready_at + Time::from_ms(2.0)));
                    let rows = microbatch.input.groups[0].decode_kv_lens.len();
                    if rows > 0 {
                        sizes.push(rows);
                    }
                }
            }
            done.extend(completed(&events));
        }
        (sizes, done)
    }

    #[test]
    fn balanced_decodes_split_running_requests_across_the_depth() {
        // Greedy (vLLM): the four requests become ready at the same exit and
        // ride one microbatch for every step.
        let (greedy, greedy_done) = decode_microbatch_sizes(false);
        assert!(greedy.iter().all(|&rows| rows == 4), "{greedy:?}");
        // Balanced: ceil(4 / 2) = 2 per microbatch, so both in-flight slots
        // carry half the running requests and stay split until completion.
        let (balanced, balanced_done) = decode_microbatch_sizes(true);
        assert!(balanced.iter().all(|&rows| rows == 2), "{balanced:?}");
        assert_eq!(balanced.iter().sum::<usize>(), greedy.iter().sum::<usize>());
        assert_eq!(greedy_done.len(), 4);
        assert_eq!(balanced_done.len(), 4);
    }

    /// A four-stage head with `max_batch_tokens` 2048 and the given sizing.
    fn depth4_head(
        store: SharedRequests,
        microbatch_sizing: crate::worker::config::MicrobatchSizing,
    ) -> PipelineHead<FakeModel> {
        build_pipeline_head_worker(
            WorkerId(0),
            "stage",
            Arc::new(FakeModel::for_ms(1.0)),
            PipelineLayout { depth: 4, ..LAYOUT },
            store,
            WorkerConfig {
                max_batch_tokens: Some(2048),
                attn_kv_bytes: 100_000,
                microbatch_sizing,
                ..WorkerConfig::default()
            },
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        )
    }

    /// Prefill tokens of each launched microbatch, in order.
    fn launched_tokens(events: &[PipelineHeadEvent]) -> Vec<u64> {
        launched(events)
            .into_iter()
            .map(|(_, tokens, _)| tokens)
            .collect()
    }

    #[test]
    fn greedy_sizing_fills_each_microbatch_to_the_cap() {
        use crate::worker::config::MicrobatchSizing;
        let store = shared_with(&[(0, 3000, 1)]);
        let mut worker = depth4_head(Rc::clone(&store), MicrobatchSizing::Greedy);
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        for step in 0..6 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(launched_tokens(&events), vec![2048, 952]);
    }

    #[test]
    fn even_sizing_slices_one_prompt_into_depth_equal_pieces() {
        use crate::worker::config::MicrobatchSizing;
        let store = shared_with(&[(0, 3000, 1)]);
        let mut worker = depth4_head(Rc::clone(&store), MicrobatchSizing::Even { min_tokens: 0 });
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        for step in 0..6 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        // In-flight tokens count toward the round, so each slice stays P / 4
        // instead of shrinking as P/4, 3P/16, ...
        assert_eq!(launched_tokens(&events), vec![750, 750, 750, 750]);
    }

    #[test]
    fn even_sizing_counts_in_flight_tokens_when_new_work_arrives() {
        use crate::worker::config::MicrobatchSizing;
        let store = shared_with(&[(0, 400, 1), (1, 1200, 1)]);
        let mut worker = depth4_head(Rc::clone(&store), MicrobatchSizing::Even { min_tokens: 0 });
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        worker.tick(Time::ZERO, &mut events);
        worker.tick(Time::from_ms(1.0), &mut events);
        // Two slices of 100 are in flight and 200 tokens pending when 1200 more
        // arrive: the round is 1600, so every later slice is 400.
        worker.enqueue(PipelineHeadMsg::Request(RequestId(1)));
        for step in 2..8 {
            worker.tick(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(launched_tokens(&events), vec![100, 100, 400, 400]);
        assert_eq!(worker.in_flight.len(), 4);
    }

    #[test]
    fn even_sizing_floor_takes_a_short_prompt_whole_and_never_waits() {
        use crate::worker::config::MicrobatchSizing;
        let store = shared_with(&[(0, 100, 1), (1, 10, 1)]);
        let mut worker = depth4_head(
            Rc::clone(&store),
            MicrobatchSizing::Even { min_tokens: 1024 },
        );
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        worker.tick(Time::ZERO, &mut events);
        // A later 10-token prompt starts as soon as stage 0 is free, below the
        // floor and with work in flight: the floor sizes, it never holds.
        worker.enqueue(PipelineHeadMsg::Request(RequestId(1)));
        worker.tick(Time::from_ms(0.5), &mut events);
        worker.tick(Time::from_ms(2.0), &mut events);
        assert_eq!(
            launched(&events),
            vec![(1, 100, Time::from_ms(1.0)), (2, 10, Time::from_ms(2.0))]
        );
    }
}
