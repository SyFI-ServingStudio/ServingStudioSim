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
use crate::worker::kv::{IterWorkerKv, TierBytes};
use crate::worker::shared::context::WorkerContext;
use crate::worker::types::{
    IterBatchPlan, PipelineHeadEvent, PipelineHeadMsg, PipelineMicrobatch, WorkerStatus,
};

/// Activation size and depth of the pipeline this head feeds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PipelineLayout {
    /// Number of stages, and so the maximum number of microbatches in flight.
    pub depth: u16,
    /// Bytes one token's activations occupy between two stages.
    pub activation_bytes_per_token: u64,
    /// What each GPU stores in and reads from a host tier
    /// ([`TierBytes::of_stages`]).
    pub tier_bytes: TierBytes,
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
        // A landed read can be admitted from its landing on.
        if let Some(landed) = self.admission.land_reads(&mut self.kv_store, now) {
            self.state_changed_at = self.state_changed_at.max(landed);
        }
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
        let compute = self
            .computing
            .as_ref()
            .map(|computing| computing.compute_end);
        let landing = self.admission.next_read_landing(&self.kv_store);
        match (compute, landing) {
            (Some(compute), Some(landing)) => Some(compute.min(landing)),
            (compute, landing) => compute.or(landing),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;
    use std::sync::Arc;

    use super::*;
    use crate::common::{PoolId, SessionInput, SharedRequests, UnifiedStage};
    use crate::test_helpers::{shared_with, test_cluster, FakeModel};
    use crate::worker::admission::{
        FifoOrder, ForceSchedule, HitsFirst, LoadBudget, MicrobatchBudget, PendingOrderKind,
        PipelinedChunkedPrefillAdmission, PrefillSplit, ReadWait, ReadWaitPolicy, StartedOrder,
    };
    use crate::worker::execution::UnifiedIterExecution;
    use crate::worker::kv::{KvCacheStore, PrefixKv, TieredPrefixKv};
    use crate::worker::types::WorkerConfig;
    use crate::worker::workers::pipeline::{build_pipeline_head_worker, PipelineHead};

    fn assert_iter_worker<W: IterWorker>() {}

    #[test]
    fn pending_policy_is_a_composition_axis_for_the_pipeline_head() {
        assert_iter_worker::<
            PipelineHeadWorker<
                KvCacheStore,
                PipelinedChunkedPrefillAdmission<FifoOrder>,
                UnifiedIterExecution<FakeModel>,
            >,
        >();
    }

    const LAYOUT: PipelineLayout = PipelineLayout {
        depth: 2,
        activation_bytes_per_token: 4,
        tier_bytes: TierBytes {
            per_token: 2,
            state: 0,
        },
    };

    /// Two KV bytes per token on the most constrained stage.
    fn kv_layout() -> crate::arch::kv_layout::KvCacheLayout {
        crate::arch::kv_layout::KvCacheLayout::new()
            .with_part(crate::arch::kv_layout::KvCachePart::per_token("kv", 2))
    }

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
            &kv_layout(),
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

    /// A hybrid head over 2-byte tokens on 4-token blocks, with a 6-byte
    /// recurrent state per request padded to 8 units and one null block.
    fn hybrid_head(
        store: SharedRequests,
        max_batch_tokens: u32,
        attn_kv_bytes: u64,
        block_aligned_chunks: bool,
    ) -> PipelineHead<FakeModel> {
        hybrid_head_charging(
            store,
            max_batch_tokens,
            attn_kv_bytes,
            block_aligned_chunks,
            Vec::new(),
        )
    }

    /// [`hybrid_head`] with `kv_part_padding`.
    fn hybrid_head_charging(
        store: SharedRequests,
        max_batch_tokens: u32,
        attn_kv_bytes: u64,
        block_aligned_chunks: bool,
        kv_part_padding: Vec<crate::worker::kv::PartPadding>,
    ) -> PipelineHead<FakeModel> {
        use crate::arch::kv_layout::{EngineOverhead, KvCachePart};
        use crate::worker::config::PrefillChunkAlignment;
        use crate::worker::kv::{PrefixCacheConfig, PrefixCachePolicy};
        let kv_cache_layout = kv_layout()
            .with_part(KvCachePart::per_request("state", 6).padded_to(16))
            .with_engine(EngineOverhead {
                block_tokens: Some(4),
                fixed_device_bytes: 8,
            });
        build_pipeline_head_worker(
            WorkerId(0),
            "stage",
            Arc::new(FakeModel::for_ms(1.0)),
            PipelineLayout { depth: 3, ..LAYOUT },
            &kv_cache_layout,
            store,
            WorkerConfig {
                max_batch_tokens: Some(max_batch_tokens),
                attn_kv_bytes,
                prefix_cache: PrefixCacheConfig::Opportunistic {
                    policy: PrefixCachePolicy::Lru,
                    max_retained_bytes: None,
                },
                prefill_chunk_alignment: Some(if block_aligned_chunks {
                    PrefillChunkAlignment::Checkpoint
                } else {
                    PrefillChunkAlignment::Plain
                }),
                kv_part_padding,
                ..WorkerConfig::default()
            },
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        )
    }

    #[test]
    fn a_hybrid_head_sizes_one_pool_and_charges_the_state() {
        // (80 - 8 null-block bytes) / 2 = 36 units. A 12-token prompt holds
        // 12 tokens of KV plus its 8-unit state once (align mode's previous-step
        // copy is not held): 20 units, so one fits and two do not.
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
    fn a_hybrid_head_can_charge_state_by_its_real_bytes() {
        // (84 - 8) / 2 = 38 units. Padded, a 12-token prompt reserves at
        // least 12 + 8 and only one fits; unpadded, the state is its 6 bytes,
        // 3 units, and both fit.
        for (charge, launches) in [("state=padded", 1), ("state=unpadded", 2)] {
            let store = shared_with(&[(0, 12, 1), (1, 12, 1)]);
            let padding = vec![charge.parse().unwrap()];
            let mut worker = hybrid_head_charging(Rc::clone(&store), 16, 84, true, padding);
            worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
            worker.enqueue(PipelineHeadMsg::Request(RequestId(1)));
            let mut events = Vec::new();
            for step in 0..3 {
                worker.tick(Time::from_ms(step as f64), &mut events);
            }
            assert_eq!(launched(&events).len(), launches, "{charge}");
        }
    }

    #[test]
    fn a_hybrid_head_ends_non_final_chunks_on_checkpoint_boundaries() {
        // Budget 6, blocks of 4: the first chunk floors to 4, the second stops
        // at the prompt's last boundary (8), the tail runs alone.
        assert_eq!(hybrid_chunks(true), [4, 4, 2]);
    }

    #[test]
    fn a_hybrid_head_decodes_and_frees_its_state_blocks_for_the_next_request() {
        // 36 units; each request reaches 15 tokens (16 in whole blocks) plus
        // two 8-unit states, so the second starts only once the first has
        // decoded and left.
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

    // ── Microbatch budget, srpt and force-schedule ───────────────────────────

    /// A four-stage head with `max_batch_tokens` 2048 over `config`.
    fn depth4_head(store: SharedRequests, config: WorkerConfig) -> PipelineHead<FakeModel> {
        build_pipeline_head_worker(
            WorkerId(0),
            "stage",
            Arc::new(FakeModel::for_ms(1.0)),
            PipelineLayout { depth: 4, ..LAYOUT },
            &kv_layout(),
            store,
            WorkerConfig {
                max_batch_tokens: Some(2048),
                attn_kv_bytes: 1_000_000,
                ..config
            },
            None,
            PoolId(0),
            "test-gpu",
            test_cluster(),
        )
    }

    fn with_budget(budget: MicrobatchBudget) -> WorkerConfig {
        WorkerConfig {
            microbatch_budget: budget,
            ..WorkerConfig::default()
        }
    }

    const EVEN: MicrobatchBudget = MicrobatchBudget {
        split: PrefillSplit::Even { min_tokens: 0 },
        long_prefill_token_threshold: None,
        load: None,
    };

    /// Prefill tokens of each launched microbatch, in order.
    fn launched_tokens(events: &[PipelineHeadEvent]) -> Vec<u64> {
        launched(events)
            .into_iter()
            .map(|(_, tokens, _)| tokens)
            .collect()
    }

    /// Each launched microbatch's prefill chunk lengths, sorted.
    fn launched_chunk_lengths(events: &[PipelineHeadEvent]) -> Vec<Vec<u32>> {
        events
            .iter()
            .filter_map(|event| match event {
                PipelineHeadEvent::MicrobatchLaunched { microbatch, .. } => {
                    let mut lengths: Vec<u32> = microbatch.input.groups[0]
                        .prefill_chunk_pairs
                        .iter()
                        .map(|&(_, length)| length)
                        .collect();
                    lengths.sort_unstable();
                    Some(lengths)
                }
                PipelineHeadEvent::RequestComplete { .. } => None,
            })
            .collect()
    }

    fn run_one_prompt(prompt_tokens: u32, config: WorkerConfig, steps: u32) -> Vec<u64> {
        let store = shared_with(&[(0, prompt_tokens, 1)]);
        let mut worker = depth4_head(store, config);
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        for step in 0..steps {
            worker.tick(Time::from_ms(f64::from(step)), &mut events);
        }
        launched_tokens(&events)
    }

    #[test]
    fn greedy_sizing_fills_each_microbatch_to_the_cap() {
        assert_eq!(
            run_one_prompt(3000, WorkerConfig::default(), 6),
            vec![2048, 952]
        );
    }

    #[test]
    fn even_sizing_slices_one_prompt_into_depth_equal_pieces() {
        // In-flight tokens count toward the round, so each slice stays P / 4
        // instead of shrinking as P/4, 3P/16, ...
        assert_eq!(
            run_one_prompt(3000, with_budget(EVEN), 6),
            vec![750, 750, 750, 750]
        );
    }

    #[test]
    fn even_sizing_counts_in_flight_tokens_when_new_work_arrives() {
        let store = shared_with(&[(0, 400, 1), (1, 1200, 1)]);
        let mut worker = depth4_head(Rc::clone(&store), with_budget(EVEN));
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        worker.tick(Time::ZERO, &mut events);
        worker.tick(Time::from_ms(1.0), &mut events);
        // Two slices of 100 are in flight and 200 tokens pending when 1200 more
        // arrive: the round is 1600, so every later slice is 400.
        worker.enqueue(PipelineHeadMsg::Request(RequestId(1)));
        for step in 2..8 {
            worker.tick(Time::from_ms(f64::from(step)), &mut events);
        }
        assert_eq!(launched_tokens(&events), vec![100, 100, 400, 400]);
        assert_eq!(worker.in_flight.len(), 4);
    }

    #[test]
    fn even_sizing_floor_takes_a_short_prompt_whole_and_never_waits() {
        let store = shared_with(&[(0, 100, 1), (1, 10, 1)]);
        let mut worker = depth4_head(
            Rc::clone(&store),
            with_budget(MicrobatchBudget {
                split: PrefillSplit::Even { min_tokens: 1024 },
                ..MicrobatchBudget::default()
            }),
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

    #[test]
    fn long_prefill_threshold_leaves_budget_for_a_short_prompt_behind_a_long_one() {
        let store = shared_with(&[(0, 3000, 1), (1, 100, 1)]);
        let mut worker = depth4_head(
            Rc::clone(&store),
            with_budget(MicrobatchBudget {
                long_prefill_token_threshold: Some(512),
                ..MicrobatchBudget::default()
            }),
        );
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        worker.enqueue(PipelineHeadMsg::Request(RequestId(1)));
        let mut events = Vec::new();
        for step in 0..6 {
            worker.tick(Time::from_ms(f64::from(step)), &mut events);
        }
        // Without the cap the long prompt takes all 2048 and the short one waits
        // for the next microbatch; with it both start in the first.
        assert_eq!(launched_tokens(&events), vec![612, 512, 512, 512]);
    }

    #[test]
    fn long_prefill_threshold_caps_the_even_target_per_request() {
        // The even target is 2000 (8000 / 4); one request may take only 1000.
        let budget = MicrobatchBudget {
            long_prefill_token_threshold: Some(1000),
            ..EVEN
        };
        assert_eq!(
            run_one_prompt(8000, with_budget(budget), 6),
            vec![1000, 1000, 1000, 1000]
        );
    }

    #[test]
    fn a_load_budget_follows_queued_and_in_flight_tokens() {
        let budget = MicrobatchBudget {
            load: Some(LoadBudget {
                low_tokens: 512,
                backlog_lo_tokens: 1000,
                backlog_hi_tokens: 5000,
            }),
            ..MicrobatchBudget::default()
        };
        // Backlog 3000 tokens (queued, then started plus in flight) is halfway
        // from 1000 to 5000: the budget sits halfway from 512 to 2048.
        assert_eq!(
            run_one_prompt(3000, with_budget(budget), 5),
            vec![1280, 1280, 440]
        );
    }

    fn srpt_config() -> WorkerConfig {
        WorkerConfig {
            pending_order: PendingOrderKind::ShortestPrefillFirst,
            started_order: StartedOrder::ShortestRemaining,
            ..WorkerConfig::default()
        }
    }

    #[test]
    fn srpt_orders_started_and_queued_prompts_by_remaining_tokens() {
        let store = shared_with(&[(0, 2156, 1), (1, 10_000, 1), (2, 500, 1)]);
        let mut worker = depth4_head(store, srpt_config());
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        worker.tick(Time::ZERO, &mut events);
        worker.enqueue(PipelineHeadMsg::Request(RequestId(1)));
        worker.enqueue(PipelineHeadMsg::Request(RequestId(2)));
        for step in 1..4 {
            worker.tick(Time::from_ms(f64::from(step)), &mut events);
        }
        // Prompt 0's 108-token tail beats the queued 500, which beats the
        // queued 10000; that one takes the rest and then runs alone.
        assert_eq!(
            launched_chunk_lengths(&events),
            vec![vec![2048], vec![108, 500, 1440], vec![2048]]
        );
    }

    /// A 10k-token prompt queued at 0 ms behind one 2048-token prompt arriving
    /// every ms, each filling a microbatch under shortest-prefill-first and
    /// srpt; returns the long prompt's prefill tokens processed by 12 ms.
    fn starved_prompt_progress(force_schedule_after: Option<Time>) -> u32 {
        let mut requests = vec![(0, 10_000, 1)];
        requests.extend((1..=12).map(|id| (id, 2048, 1)));
        let store = shared_with(&requests);
        for id in 1..=12 {
            store.borrow_mut()[RequestId(id)].request.core.arrival_time =
                Time::from_ms(f64::from(id - 1));
        }
        let mut worker = depth4_head(
            Rc::clone(&store),
            WorkerConfig {
                force_schedule: force_schedule_after.map(|after| ForceSchedule {
                    after,
                    hits_first: HitsFirst::Off,
                }),
                ..srpt_config()
            },
        );
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        let mut exits: Vec<(u64, Time)> = Vec::new();
        for step in 0..12_u32 {
            let start = Time::from_ms(f64::from(step));
            let end = Time::from_ms(f64::from(step + 1));
            worker.enqueue(PipelineHeadMsg::Request(RequestId(step + 1)));
            let mut now = start;
            loop {
                exits.retain(|&(microbatch, at)| {
                    let due = at <= now;
                    if due {
                        worker.enqueue(PipelineHeadMsg::MicrobatchExit { microbatch, at });
                    }
                    !due
                });
                let mut events = Vec::new();
                let wakeup = worker.tick(now, &mut events);
                for (id, _, ready) in launched(&events) {
                    exits.push((id, ready + Time::from_ms(1.0)));
                }
                let next = [wakeup, exits.iter().map(|&(_, at)| at).min()]
                    .into_iter()
                    .flatten()
                    .min();
                match next {
                    Some(next) if next > now && next < end => now = next,
                    _ => break,
                }
            }
        }
        let progress = store.borrow()[RequestId(0)]
            .progress
            .prefill_tokens_processed;
        progress
    }

    #[test]
    fn a_request_waiting_past_the_force_bound_runs_before_shorter_ones() {
        // Shortest-prefill-first and srpt alone: every microbatch goes to a
        // newer, shorter prompt.
        assert_eq!(starved_prompt_progress(None), 0);
        // Overdue at 5 ms, it takes whole microbatches from then on: at least
        // three have left the pipeline by 12 ms.
        let progress = starved_prompt_progress(Some(Time::from_ms(5.0)));
        assert!(progress >= 3 * 2048, "progress {progress}");
    }

    // ── Prefix tiers ─────────────────────────────────────────────────────────

    fn as_session(
        store: &SharedRequests,
        id: u32,
        session_id: u32,
        declared: u32,
        arrival_ms: f64,
    ) {
        let mut store = store.borrow_mut();
        let request = &mut store[RequestId(id)].request;
        request.definition.session = SessionInput::Session {
            session_id,
            session_start_time: Time::ZERO,
            declared_prefix_tokens: declared,
        };
        request.core.arrival_time = Time::from_ms(arrival_ms);
    }

    /// Tick at every wakeup and stage-1 exit (1 ms after stage 0) up to `until`.
    fn run_until(
        worker: &mut PipelineHead<FakeModel>,
        from: Time,
        until: Time,
        events: &mut Vec<PipelineHeadEvent>,
    ) {
        let mut exits: Vec<(u64, Time)> = Vec::new();
        let mut now = from;
        while now <= until {
            let mut fresh = Vec::new();
            let wakeup = worker.tick(now, &mut fresh);
            for (id, _, ready) in launched(&fresh) {
                exits.push((id, ready + Time::from_ms(1.0)));
            }
            events.extend(fresh);
            let next_exit = exits.iter().map(|&(_, at)| at).min();
            let next = [wakeup, next_exit].into_iter().flatten().min();
            let Some(next) = next else { break };
            if let Some(index) = exits.iter().position(|&(_, at)| at == next) {
                let (microbatch, at) = exits.remove(index);
                worker.enqueue(PipelineHeadMsg::MicrobatchExit { microbatch, at });
            }
            now = next.max(now);
        }
    }

    /// 2 B/token, 120 tokens of HBM, 64-token microbatches. DRAM: 500 tokens
    /// read at 1e5 B/s, so 50 tokens take 1 ms.
    fn tiered_config() -> WorkerConfig {
        WorkerConfig {
            max_batch_tokens: Some(64),
            attn_kv_bytes: 240,
            prefix_tiers: vec![crate::worker::kv::TierSpec {
                name: "dram",
                capacity_gb: 1e-6,
                read_gb_per_s: 1e-4,
                writes: crate::worker::kv::TierWrites::EveryContext,
            }],
            ..WorkerConfig::default()
        }
    }

    fn hit_tokens(store: &SharedRequests, id: u32) -> Option<u32> {
        store.borrow()[RequestId(id)]
            .telemetry
            .prefix_cache_hit_tokens
    }

    #[test]
    fn decode_elsewhere_completes_at_the_first_token_and_retains_the_outputs() {
        // Round 0 has 5 target outputs; round 1 declares all 55 tokens and
        // finds the 54 with KV in HBM (the last output's KV is round 1's to
        // compute).
        let store = shared_with(&[(0, 50, 5), (1, 4, 5)]);
        as_session(&store, 0, 7, 0, 0.0);
        as_session(&store, 1, 7, 55, 10.0);
        let mut worker = head_with(
            Rc::clone(&store),
            WorkerConfig {
                external_decode: true,
                ..tiered_config()
            },
        );
        let mut events = Vec::new();
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        run_until(&mut worker, Time::ZERO, Time::from_ms(9.0), &mut events);
        assert_eq!(completed(&events), vec![RequestId(0)]);
        worker.enqueue(PipelineHeadMsg::Request(RequestId(1)));
        run_until(
            &mut worker,
            Time::from_ms(10.0),
            Time::from_ms(20.0),
            &mut events,
        );
        assert_eq!(completed(&events), vec![RequestId(0), RequestId(1)]);
        assert_eq!(
            store.borrow()[RequestId(0)].progress.output_tokens_emitted,
            1
        );
        assert_eq!(hit_tokens(&store, 1), Some(54));
    }

    #[test]
    fn a_session_evicted_from_hbm_reads_its_context_back_from_dram_first() {
        // Session 7 round 0 (50 tokens), then a 100-token prompt that evicts it
        // from HBM, then round 1 declaring the 51-token context, whose first 50
        // have KV in DRAM (the output's KV is round 1's to compute).
        let store = shared_with(&[(0, 50, 1), (1, 100, 1), (2, 4, 1)]);
        as_session(&store, 0, 7, 0, 0.0);
        as_session(&store, 2, 7, 51, 10.0);
        let mut worker = head_with(Rc::clone(&store), tiered_config());
        let mut events = Vec::new();
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        run_until(&mut worker, Time::ZERO, Time::from_ms(3.0), &mut events);
        worker.enqueue(PipelineHeadMsg::Request(RequestId(1)));
        run_until(
            &mut worker,
            Time::from_ms(3.0),
            Time::from_ms(9.0),
            &mut events,
        );
        assert_eq!(completed(&events), vec![RequestId(0), RequestId(1)]);

        events.clear();
        worker.enqueue(PipelineHeadMsg::Request(RequestId(2)));
        // The read starts at 10 ms and lands 1 ms later; nothing launches
        // before it.
        assert_eq!(
            worker.tick(Time::from_ms(10.0), &mut events),
            Some(Time::from_ms(11.0))
        );
        assert_eq!(worker.status().queued_requests, 1);
        assert!(events.is_empty());
        run_until(
            &mut worker,
            Time::from_ms(11.0),
            Time::from_ms(20.0),
            &mut events,
        );
        let pairs: Vec<(u32, u32)> = events
            .iter()
            .filter_map(|event| match event {
                PipelineHeadEvent::MicrobatchLaunched { microbatch, .. } => {
                    Some(microbatch.input.groups[0].prefill_chunk_pairs.clone())
                }
                PipelineHeadEvent::RequestComplete { .. } => None,
            })
            .flatten()
            .collect();
        assert_eq!(pairs, [(50, 5)]);
        assert_eq!(hit_tokens(&store, 2), Some(50));
        assert!(store.borrow()[RequestId(2)].lifecycle.completed);
    }

    #[test]
    fn a_context_evicted_while_queued_is_read_when_admission_reaches_it() {
        // Round 1 of session 7 arrives while HBM still holds round 0's 50
        // tokens, behind a 100-token prompt of session 8 whose admission evicts
        // them. The read starts when round 1 reaches the head, not at arrival,
        // so it still finds its context; HBM alone recomputes it.
        for (tiers, hit) in [(true, 50), (false, 0)] {
            let store = shared_with(&[(0, 50, 1), (1, 100, 1), (2, 4, 1)]);
            as_session(&store, 0, 7, 0, 0.0);
            as_session(&store, 1, 8, 0, 4.0);
            as_session(&store, 2, 7, 51, 4.0);
            let config = if tiers {
                tiered_config()
            } else {
                WorkerConfig {
                    max_batch_tokens: Some(64),
                    attn_kv_bytes: 240,
                    ..WorkerConfig::default()
                }
            };
            let mut worker = head_with(Rc::clone(&store), config);
            let mut events = Vec::new();
            worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
            run_until(&mut worker, Time::ZERO, Time::from_ms(3.0), &mut events);
            let round1 = store.borrow()[RequestId(2)].request.definition.session;
            assert_eq!(worker.kv_store.resident_prefix_tokens(4, round1), 50);
            worker.enqueue(PipelineHeadMsg::Request(RequestId(1)));
            worker.enqueue(PipelineHeadMsg::Request(RequestId(2)));
            run_until(
                &mut worker,
                Time::from_ms(4.0),
                Time::from_ms(60.0),
                &mut events,
            );
            assert_eq!(
                completed(&events),
                vec![RequestId(0), RequestId(1), RequestId(2)]
            );
            assert_eq!(hit_tokens(&store, 2), Some(hit), "tiers={tiers}");
        }
    }

    #[test]
    fn a_read_waits_until_its_request_fits_in_hbm_and_holds_it() {
        // HBM holds 120 tokens. A 60-token prompt runs while session 7 (70
        // pre-run tokens in the tier, 4 fresh) reaches the head with budget
        // left: its 74 tokens do not fit beside the prompt, so its read starts
        // only once the prompt leaves, as vLLM allocates an async load's
        // blocks before starting it.
        let store = shared_with(&[(0, 60, 1), (1, 4, 1)]);
        as_session(&store, 0, 8, 0, 0.0);
        as_session(&store, 1, 7, 70, 0.0);
        let mut worker = head_with(
            Rc::clone(&store),
            WorkerConfig {
                prefix_tier_warm_start: true,
                ..tiered_config()
            },
        );
        let mut events = Vec::new();
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        worker.enqueue(PipelineHeadMsg::Request(RequestId(1)));
        run_until(&mut worker, Time::ZERO, Time::from_ms(60.0), &mut events);
        assert_eq!(completed(&events), vec![RequestId(0), RequestId(1)]);
        assert_eq!(hit_tokens(&store, 1), Some(70));
        // The prompt's microbatch leaves 1 ms after launch; the 70-token read
        // (1.4 ms) starts then, and session 7 launches once it lands.
        let launches = launched(&events);
        assert_eq!(launches.len(), 2, "{launches:?}");
        assert!(
            launches[1].2 >= launches[0].2 + Time::from_ms(2.4),
            "{launches:?}"
        );
    }

    #[test]
    fn a_landed_read_is_admitted_before_a_queue_head_that_cannot_start_its_read() {
        // Shortest prefill first, 120 tokens of HBM. Session 7 (40 pre-run
        // tokens, 4 fresh) starts its read and holds 45 tokens. Session 9 (80
        // pre-run, 2 fresh) then ranks first but cannot start its read beside
        // that hold. Session 7's landed read goes first, as vLLM schedules
        // loaded requests before its waiting queue; otherwise neither moves.
        let store = shared_with(&[(0, 4, 1), (1, 2, 1)]);
        as_session(&store, 0, 7, 40, 0.0);
        as_session(&store, 1, 9, 80, 0.1);
        let mut worker = head_with(
            Rc::clone(&store),
            WorkerConfig {
                pending_order: PendingOrderKind::ShortestPrefillFirst,
                prefix_tier_warm_start: true,
                ..tiered_config()
            },
        );
        let mut events = Vec::new();
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        run_until(&mut worker, Time::ZERO, Time::from_ms(0.1), &mut events);
        worker.enqueue(PipelineHeadMsg::Request(RequestId(1)));
        run_until(
            &mut worker,
            Time::from_ms(0.1),
            Time::from_ms(60.0),
            &mut events,
        );
        assert_eq!(completed(&events), vec![RequestId(0), RequestId(1)]);
        assert_eq!(hit_tokens(&store, 0), Some(40));
        assert_eq!(hit_tokens(&store, 1), Some(80));
    }

    /// Session 7's 40-token pre-run read keeps the DRAM channel busy until
    /// 0.8 ms; session 9 (20 pre-run tokens) reaches the head at 0.1 ms with
    /// `behind` (a 8-token prompt needing no read) queued after it.
    fn read_wait_head(
        read_wait: Option<ReadWait>,
        behind: bool,
    ) -> (
        SharedRequests,
        PipelineHead<FakeModel>,
        Vec<PipelineHeadEvent>,
    ) {
        let store = shared_with(&[(0, 4, 1), (1, 2, 1), (2, 8, 1)]);
        as_session(&store, 0, 7, 40, 0.0);
        as_session(&store, 1, 9, 20, 0.1);
        let mut worker = head_with(
            Rc::clone(&store),
            WorkerConfig {
                prefix_tier_warm_start: true,
                prefix_tier_read_wait: read_wait,
                ..tiered_config()
            },
        );
        let mut events = Vec::new();
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        run_until(&mut worker, Time::ZERO, Time::from_ms(0.1), &mut events);
        worker.enqueue(PipelineHeadMsg::Request(RequestId(1)));
        if behind {
            worker.enqueue(PipelineHeadMsg::Request(RequestId(2)));
        }
        (store, worker, events)
    }

    fn bound(ms: f64, policy: ReadWaitPolicy) -> Option<ReadWait> {
        (ms > 0.0).then(|| ReadWait {
            max: Time::from_ms(ms),
            policy,
        })
    }

    #[test]
    fn a_read_waits_in_its_queue_while_its_tier_is_past_the_read_wait_bound() {
        // Unbounded, session 9's read queues behind session 7's (and holds
        // HBM); with a 0.5 ms bound it waits in its queue until the channel
        // drains. Both land the same: the channel is FIFO.
        for (bound_ms, reading_at_0_2) in [(0.0, 2), (0.5, 1)] {
            let (store, mut worker, mut events) =
                read_wait_head(bound(bound_ms, ReadWaitPolicy::PassOver), false);
            run_until(
                &mut worker,
                Time::from_ms(0.1),
                Time::from_ms(0.2),
                &mut events,
            );
            assert_eq!(
                worker.admission.reading_requests(),
                reading_at_0_2,
                "bound {bound_ms}"
            );
            run_until(
                &mut worker,
                Time::from_ms(0.2),
                Time::from_ms(60.0),
                &mut events,
            );
            assert_eq!(completed(&events), vec![RequestId(0), RequestId(1)]);
            assert_eq!(hit_tokens(&store, 0), Some(40), "bound {bound_ms}");
            assert_eq!(hit_tokens(&store, 1), Some(20), "bound {bound_ms}");
        }
    }

    #[test]
    fn a_request_past_the_read_wait_bound_is_passed_over_for_the_ones_behind() {
        // Unbounded, session 9's read queues and request 2 follows it; with a
        // 0.5 ms bound session 9 is passed over rather than stopping admission.
        // Either way request 2 launches at 0.1 ms (1 ms stage), before the
        // channel drains.
        for bound_ms in [0.0, 0.5] {
            let (store, mut worker, mut events) =
                read_wait_head(bound(bound_ms, ReadWaitPolicy::PassOver), true);
            run_until(
                &mut worker,
                Time::from_ms(0.1),
                Time::from_ms(60.0),
                &mut events,
            );
            assert_eq!(
                launched(&events)[0],
                (1, 8, Time::from_ms(1.1)),
                "bound {bound_ms}"
            );
            assert_eq!(
                completed(&events),
                vec![RequestId(2), RequestId(0), RequestId(1)],
                "bound {bound_ms}"
            );
            assert_eq!(hit_tokens(&store, 1), Some(20), "bound {bound_ms}");
        }
    }

    #[test]
    fn the_block_policy_stops_admission_at_a_request_past_the_read_wait_bound() {
        // With a 0.5 ms bound: passed over, request 2 runs at 0.1 ms; blocked,
        // it waits behind session 9 until the channel drains and session 7's
        // read lands at 0.8 ms.
        let first_token_of_request_2 = |policy| {
            let (store, mut worker, mut events) = read_wait_head(bound(0.5, policy), true);
            run_until(
                &mut worker,
                Time::from_ms(0.1),
                Time::from_ms(60.0),
                &mut events,
            );
            assert_eq!(completed(&events).len(), 3, "{policy:?}");
            assert_eq!(hit_tokens(&store, 1), Some(20), "{policy:?}");
            let first_token = store.borrow()[RequestId(2)].telemetry.first_output_time;
            first_token.expect("request 2 finishes")
        };
        assert_eq!(
            first_token_of_request_2(ReadWaitPolicy::PassOver),
            Time::from_ms(2.1)
        );
        assert_eq!(
            first_token_of_request_2(ReadWaitPolicy::Block),
            Time::from_ms(2.8)
        );
    }

    #[test]
    fn warm_start_reads_a_pre_run_context_from_the_slowest_tier() {
        // A session's first request in the run declares 40 tokens: with warm
        // start they come from the tier (0.8 ms), without it they are computed.
        for (warm_start, hit, read_ms) in [(true, 40, 0.8), (false, 0, 0.0)] {
            let store = shared_with(&[(0, 4, 1)]);
            as_session(&store, 0, 7, 40, 0.0);
            let mut worker = head_with(
                Rc::clone(&store),
                WorkerConfig {
                    prefix_tier_warm_start: warm_start,
                    ..tiered_config()
                },
            );
            let mut events = Vec::new();
            worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
            run_until(&mut worker, Time::ZERO, Time::from_ms(20.0), &mut events);
            assert_eq!(completed(&events), vec![RequestId(0)]);
            assert_eq!(hit_tokens(&store, 0), Some(hit));
            // Stage 0 starts once the read lands.
            let launches = launched(&events);
            assert!(
                launches[0].2 > Time::from_ms(read_ms),
                "warm_start={warm_start}: {launches:?}"
            );
        }
    }

    /// 40% of HBM's 120 tokens is the loading area: 48 tokens.
    fn staged_config(prefix_tiers: Vec<crate::worker::kv::TierSpec>) -> WorkerConfig {
        WorkerConfig {
            prefix_tiers,
            prefix_tier_warm_start: true,
            prefix_tier_loading_area: 0.4,
            ..tiered_config()
        }
    }

    #[test]
    fn staged_reads_start_in_order_and_a_later_one_never_jumps_the_head() {
        // Pre-run contexts of 40, 30 and 5 tokens in DRAM. Session 7's read
        // takes 40 of the 48-token area; session 8's does not fit beside it,
        // and session 9's would but may not jump it. Both wait in the queue,
        // passed over: the plain prompt behind them runs at once.
        let store = shared_with(&[(0, 4, 1), (1, 4, 1), (2, 4, 1), (3, 8, 1)]);
        as_session(&store, 0, 7, 40, 0.0);
        as_session(&store, 1, 8, 30, 0.0);
        as_session(&store, 2, 9, 5, 0.0);
        let mut worker = head_with(
            Rc::clone(&store),
            staged_config(tiered_config().prefix_tiers),
        );
        for id in 0..4 {
            worker.enqueue(PipelineHeadMsg::Request(RequestId(id)));
        }
        let mut events = Vec::new();
        worker.tick(Time::ZERO, &mut events);
        assert_eq!(worker.admission.reading_requests(), 1);
        run_until(&mut worker, Time::ZERO, Time::from_ms(60.0), &mut events);
        assert_eq!(launched(&events)[0], (1, 8, Time::from_ms(1.0)));
        assert_eq!(completed(&events).len(), 4);
        for (id, hit) in [(0, 40), (1, 30), (2, 5)] {
            assert_eq!(hit_tokens(&store, id), Some(hit));
        }
    }

    #[test]
    fn a_staged_ssd_read_lands_in_dram_and_hops_on_to_hbm() {
        // Session 7's 40 pre-run tokens are in SSD (5 tokens per ms): they
        // reach DRAM's loading area at 8 ms and hop on to HBM on DRAM's
        // channel (50 tokens per ms) by 8.8 ms; then the request launches.
        let store = shared_with(&[(0, 4, 1)]);
        as_session(&store, 0, 7, 40, 0.0);
        let mut tiers = tiered_config().prefix_tiers;
        tiers.push(crate::worker::kv::TierSpec {
            name: "ssd",
            capacity_gb: 1e-5,
            read_gb_per_s: 1e-5,
            writes: crate::worker::kv::TierWrites::EveryContext,
        });
        let mut worker = head_with(Rc::clone(&store), staged_config(tiers));
        worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
        let mut events = Vec::new();
        assert_eq!(
            worker.tick(Time::ZERO, &mut events),
            Some(Time::from_ms(8.0))
        );
        assert_eq!(
            worker.tick(Time::from_ms(8.0), &mut events),
            Some(Time::from_ms(8.8))
        );
        assert!(events.is_empty());
        run_until(
            &mut worker,
            Time::from_ms(8.8),
            Time::from_ms(60.0),
            &mut events,
        );
        assert_eq!(launched(&events), [(1, 4, Time::from_ms(9.8))]);
        assert_eq!(hit_tokens(&store, 0), Some(40));
    }

    // ── Force-schedule cost classes ──────────────────────────────────────────

    fn hits_first(hits_first: HitsFirst, after_ms: f64) -> Option<ForceSchedule> {
        Some(ForceSchedule {
            after: Time::from_ms(after_ms),
            hits_first,
        })
    }

    #[test]
    fn hits_first_admits_an_overdue_hit_before_an_older_overdue_miss() {
        // HBM holds 120 tokens and session 7's 50. A 100-token prompt of
        // session 8 (arrived 4 ms) and session 7's round 1 (5 ms) are both
        // overdue when the head first looks at 10 ms. Oldest first admits the
        // prompt, whose footprint evicts session 7, so round 1 recomputes;
        // hits first admits round 1 on its context first. Gated on an at-risk
        // tier context, hits first stays off without tiers: oldest first.
        for (order, hit) in [
            (HitsFirst::Off, 0),
            (HitsFirst::Always, 50),
            (HitsFirst::WhenAtRisk, 0),
        ] {
            let store = shared_with(&[(0, 50, 1), (1, 100, 1), (2, 4, 1)]);
            as_session(&store, 0, 7, 0, 0.0);
            as_session(&store, 1, 8, 0, 4.0);
            as_session(&store, 2, 7, 51, 5.0);
            let mut worker = head_with(
                Rc::clone(&store),
                WorkerConfig {
                    max_batch_tokens: Some(64),
                    attn_kv_bytes: 240,
                    force_schedule: hits_first(order, 2.0),
                    ..srpt_config()
                },
            );
            let mut events = Vec::new();
            worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
            run_until(&mut worker, Time::ZERO, Time::from_ms(3.0), &mut events);
            worker.enqueue(PipelineHeadMsg::Request(RequestId(1)));
            worker.enqueue(PipelineHeadMsg::Request(RequestId(2)));
            run_until(
                &mut worker,
                Time::from_ms(10.0),
                Time::from_ms(80.0),
                &mut events,
            );
            assert_eq!(completed(&events).len(), 3, "{order:?}");
            assert_eq!(hit_tokens(&store, 2), Some(hit), "{order:?}");
        }
    }

    #[test]
    fn hits_first_when_at_risk_splits_only_while_the_overdue_misses_would_evict_a_held_context() {
        // Session 7's 50 tokens and session 8's 100 are in DRAM; HBM (120
        // tokens) holds only session 8's. At 10 ms a 100-token prompt of
        // session 9 (arrived 4 ms) is overdue and session 7's round 1 (9 ms)
        // is not. Oldest first runs the prompt, whose footprint leaves no room
        // for round 1's read; its context is then written through DRAM. In a
        // 150-token DRAM that write evicts session 7, so round 1 recomputes:
        // the context is at risk and the gated split reads it first. In a
        // 500-token DRAM it is not at risk: the gate keeps oldest first and
        // round 1 still finds its context after the prompt.
        for (dram_tokens, order, hit, round1_first) in [
            (150, HitsFirst::Off, 0, false),
            (150, HitsFirst::Always, 50, true),
            (150, HitsFirst::WhenAtRisk, 50, true),
            (500, HitsFirst::Always, 50, true),
            (500, HitsFirst::WhenAtRisk, 50, false),
        ] {
            let store = shared_with(&[(0, 50, 1), (1, 100, 1), (2, 100, 1), (3, 4, 1)]);
            as_session(&store, 0, 7, 0, 0.0);
            as_session(&store, 1, 8, 0, 3.0);
            as_session(&store, 2, 9, 0, 4.0);
            as_session(&store, 3, 7, 51, 9.0);
            let mut config = tiered_config();
            config.prefix_tiers[0].capacity_gb = f64::from(2 * dram_tokens) * 1e-9;
            let mut worker = head_with(
                Rc::clone(&store),
                WorkerConfig {
                    force_schedule: hits_first(order, 5.0),
                    ..WorkerConfig {
                        pending_order: PendingOrderKind::ShortestPrefillFirst,
                        started_order: StartedOrder::ShortestRemaining,
                        ..config
                    }
                },
            );
            let mut events = Vec::new();
            worker.enqueue(PipelineHeadMsg::Request(RequestId(0)));
            run_until(&mut worker, Time::ZERO, Time::from_ms(2.0), &mut events);
            worker.enqueue(PipelineHeadMsg::Request(RequestId(1)));
            run_until(
                &mut worker,
                Time::from_ms(3.0),
                Time::from_ms(9.0),
                &mut events,
            );
            assert_eq!(completed(&events), vec![RequestId(0), RequestId(1)]);
            let round1 = store.borrow()[RequestId(3)].request.definition.session;
            assert_eq!(worker.kv_store.resident_prefix_tokens(4, round1), 0);
            assert_eq!(worker.kv_store.reusable_prefix_tokens(0, 4, round1), 50);
            worker.enqueue(PipelineHeadMsg::Request(RequestId(2)));
            worker.enqueue(PipelineHeadMsg::Request(RequestId(3)));
            run_until(
                &mut worker,
                Time::from_ms(10.0),
                Time::from_ms(80.0),
                &mut events,
            );
            let label = format!("DRAM {dram_tokens}, {order:?}");
            let done = completed(&events);
            assert_eq!(done.len(), 4, "{label}");
            assert_eq!(hit_tokens(&store, 3), Some(hit), "{label}");
            let position = |id| done.iter().position(|&request| request == RequestId(id));
            assert_eq!(position(3) < position(2), round1_first, "{label}: {done:?}");
        }
    }
}
