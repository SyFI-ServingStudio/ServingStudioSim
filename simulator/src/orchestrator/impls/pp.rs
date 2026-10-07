//! `pp` — pipeline-parallel deployment. One `stage` pool whose workers are the
//! stages of `replicas` independent pipelines. Worker `r * depth + s` is stage
//! `s` of replica `r`; stage 0 is a [`PipelineHeadWorker`]-style head that owns
//! admission and KV, and stages `1..depth` are followers that only pull and
//! compute.
//!
//! L6a (`PpStagePoolController`) places arrivals on heads (least-queued or
//! round-robin), hands each finished microbatch to the next stage of its own
//! replica, and turns the last stage's `StageDone` into `MicrobatchExit` for the
//! head. L6b (`PpFlow`) is the object L7 calls.
//!
//! A stage hand-off costs no simulator tick: `tick` repeats the pool sweep at the
//! same `now` until no worker emits an event, so a microbatch that finishes on
//! one stage is already pulling on the next. Followers settle before heads tick,
//! so a head forms only after every exit up to `now` has reached it. Every routed message makes its
//! receiver due immediately, even one with a later compute wakeup: a head that
//! is computing must still complete an exited microbatch now, so L7 sees the
//! request finish at the right tick.
//!
//! Activation pulls use the shared [`GpuCluster`] with the NVLink `p2p_intra`
//! curve, since a replica's stages share one NVLink domain.
//!
//! [`PipelineHeadWorker`]: crate::worker::workers::pipeline::PipelineHeadWorker

use crate::common::{PoolId, Request, RequestId, SharedRequests, Time, WorkerId};
use crate::worker::{
    CostSource, GpuCluster, IterWorker, PipelineHeadEvent, PipelineHeadMsg, PipelineStageEvent,
    PipelineStageMsg, SharedGpuCluster,
};

use super::super::{Flow, OrchAction};
use super::simple_dp::DpPlacementPolicy;

/// The single pool of a `pp` deployment.
pub const PP_STAGE_POOL: PoolId = PoolId(0);

/// Same sentinel as `simple_dp`: a quiescent worker.
const NO_WAKEUP_TIME: Time = Time::from_ns(u64::MAX);

/// Config for the `stage` pool.
pub struct PpStagePoolConfig {
    pub replicas: u16,
    /// Stages per replica.
    pub depth: u16,
    pub placement: DpPlacementPolicy,
}

struct PpReplica<H, S> {
    head: H,
    head_wakeup: Time,
    /// Stage `s` (1-based) is `followers[s - 1]`.
    followers: Vec<S>,
    follower_wakeups: Vec<Time>,
}

// ── L6a: the stage pool ──────────────────────────────────────────────────────

pub struct PpStagePoolController<H, S>
where
    H: IterWorker<Msg = PipelineHeadMsg, Event = PipelineHeadEvent>,
    S: IterWorker<Msg = PipelineStageMsg, Event = PipelineStageEvent>,
{
    depth: u16,
    replicas: Vec<PpReplica<H, S>>,
    placement: DpPlacementPolicy,
    rr_next: usize,
    head_events: Vec<PipelineHeadEvent>,
    stage_events: Vec<PipelineStageEvent>,
}

impl<H, S> PpStagePoolController<H, S>
where
    H: IterWorker<Msg = PipelineHeadMsg, Event = PipelineHeadEvent>,
    S: IterWorker<Msg = PipelineStageMsg, Event = PipelineStageEvent>,
{
    /// Build each replica's stages in order, so a replica's GPUs are contiguous.
    /// `build_head(id)` builds stage 0; `build_follower(id, stage)` builds
    /// stage `stage >= 1`.
    pub fn new(
        cfg: &PpStagePoolConfig,
        mut build_head: impl FnMut(WorkerId) -> H,
        mut build_follower: impl FnMut(WorkerId, u16) -> S,
    ) -> Self {
        assert!(cfg.replicas > 0, "pp needs at least one replica");
        assert!(cfg.depth > 0, "pp needs at least one stage");
        let replicas = (0..cfg.replicas)
            .map(|replica| {
                let base = replica * cfg.depth;
                let head = build_head(WorkerId(base));
                let followers: Vec<S> = (1..cfg.depth)
                    .map(|stage| build_follower(WorkerId(base + stage), stage))
                    .collect();
                PpReplica {
                    head,
                    head_wakeup: NO_WAKEUP_TIME,
                    follower_wakeups: vec![NO_WAKEUP_TIME; followers.len()],
                    followers,
                }
            })
            .collect();
        Self {
            depth: cfg.depth,
            replicas,
            placement: cfg.placement,
            rr_next: 0,
            head_events: Vec::new(),
            stage_events: Vec::new(),
        }
    }

    pub fn admit(&mut self, request: RequestId) {
        let replica = match self.placement {
            DpPlacementPolicy::LeastQueued => self.least_queued_replica(),
            DpPlacementPolicy::RoundRobin => {
                let replica = self.rr_next;
                self.rr_next = (self.rr_next + 1) % self.replicas.len();
                replica
            }
        };
        self.route_to_head(replica, PipelineHeadMsg::Request(request));
    }

    fn least_queued_replica(&self) -> usize {
        self.replicas
            .iter()
            .enumerate()
            .min_by_key(|(replica, pipeline)| {
                let status = pipeline.head.status();
                (status.queued_requests, status.active_requests, *replica)
            })
            .map(|(replica, _)| replica)
            .expect("pp has at least one replica")
    }

    /// Sweep due workers and route their hand-offs until none emits an event.
    /// Completed requests are appended to `completed`.
    ///
    /// Followers run to their own fixpoint before any head ticks, so every
    /// microbatch that has left its last stage by `now` reaches its head before
    /// the head forms the next one: a returning decode joins that formation,
    /// and the exit's KV release is sampled before it.
    pub fn tick_collect(&mut self, now: Time, completed: &mut Vec<RequestId>) {
        loop {
            self.settle_followers(now);
            let mut head_events = std::mem::take(&mut self.head_events);
            for pipeline in &mut self.replicas {
                if pipeline.head_wakeup <= now {
                    pipeline.head_wakeup = pipeline
                        .head
                        .tick(now, &mut head_events)
                        .unwrap_or(NO_WAKEUP_TIME);
                }
            }
            let quiescent = head_events.is_empty();
            for event in head_events.drain(..) {
                self.on_head_event(event, completed);
            }
            self.head_events = head_events;
            if quiescent {
                break;
            }
        }
    }

    /// Tick due followers and route their hand-offs until none emits an event.
    fn settle_followers(&mut self, now: Time) {
        loop {
            let mut stage_events = std::mem::take(&mut self.stage_events);
            for pipeline in &mut self.replicas {
                for (wakeup, follower) in pipeline
                    .follower_wakeups
                    .iter_mut()
                    .zip(pipeline.followers.iter_mut())
                {
                    if *wakeup <= now {
                        *wakeup = follower
                            .tick(now, &mut stage_events)
                            .unwrap_or(NO_WAKEUP_TIME);
                    }
                }
            }
            let quiescent = stage_events.is_empty();
            for event in stage_events.drain(..) {
                self.on_stage_event(event);
            }
            self.stage_events = stage_events;
            if quiescent {
                break;
            }
        }
    }

    fn on_head_event(&mut self, event: PipelineHeadEvent, completed: &mut Vec<RequestId>) {
        match event {
            PipelineHeadEvent::RequestComplete { req, .. } => completed.push(req),
            PipelineHeadEvent::MicrobatchLaunched { worker, microbatch } => {
                let (replica, stage) = self.locate(worker);
                debug_assert_eq!(stage, 0);
                if self.depth == 1 {
                    self.route_to_head(
                        replica,
                        PipelineHeadMsg::MicrobatchExit {
                            microbatch: microbatch.id,
                            at: microbatch.ready_at,
                        },
                    );
                } else {
                    self.route_to_follower(replica, 1, PipelineStageMsg::Microbatch(microbatch));
                }
            }
        }
    }

    fn on_stage_event(&mut self, event: PipelineStageEvent) {
        let PipelineStageEvent::StageDone { worker, microbatch } = event;
        let (replica, stage) = self.locate(worker);
        if stage + 1 == self.depth {
            self.route_to_head(
                replica,
                PipelineHeadMsg::MicrobatchExit {
                    microbatch: microbatch.id,
                    at: microbatch.ready_at,
                },
            );
        } else {
            self.route_to_follower(replica, stage + 1, PipelineStageMsg::Microbatch(microbatch));
        }
    }

    /// `(replica, stage)` of a worker id.
    fn locate(&self, worker: WorkerId) -> (usize, u16) {
        (usize::from(worker.0 / self.depth), worker.0 % self.depth)
    }

    fn route_to_head(&mut self, replica: usize, msg: PipelineHeadMsg) {
        let pipeline = &mut self.replicas[replica];
        pipeline.head.enqueue(msg);
        pipeline.head_wakeup = Time::ZERO;
    }

    fn route_to_follower(&mut self, replica: usize, stage: u16, msg: PipelineStageMsg) {
        let pipeline = &mut self.replicas[replica];
        let index = usize::from(stage) - 1;
        pipeline.followers[index].enqueue(msg);
        pipeline.follower_wakeups[index] = Time::ZERO;
    }
}

// ── L6b: deployment flow ─────────────────────────────────────────────────────

pub struct PpFlow<H, S>
where
    H: IterWorker<Msg = PipelineHeadMsg, Event = PipelineHeadEvent>,
    S: IterWorker<Msg = PipelineStageMsg, Event = PipelineStageEvent>,
{
    requests: SharedRequests,
    stage_pool: PpStagePoolController<H, S>,
    cluster: SharedGpuCluster,
    completed: Vec<RequestId>,
}

impl<H, S> PpFlow<H, S>
where
    H: IterWorker<Msg = PipelineHeadMsg, Event = PipelineHeadEvent>,
    S: IterWorker<Msg = PipelineStageMsg, Event = PipelineStageEvent>,
{
    /// Build the shared cluster with the activation-transfer `cost`, then the
    /// stage pool. The builders receive the cluster so each worker allocates its
    /// own GPU and communication group in it.
    pub fn new(
        requests: SharedRequests,
        cfg: &PpStagePoolConfig,
        cost: CostSource,
        mut build_head: impl FnMut(WorkerId, &SharedGpuCluster) -> H,
        mut build_follower: impl FnMut(WorkerId, u16, &SharedGpuCluster) -> S,
    ) -> Self {
        let cluster: SharedGpuCluster =
            std::rc::Rc::new(std::cell::RefCell::new(GpuCluster::new(cost)));
        let stage_pool = PpStagePoolController::new(
            cfg,
            |id| build_head(id, &cluster),
            |id, stage| build_follower(id, stage, &cluster),
        );
        Self {
            requests,
            stage_pool,
            cluster,
            completed: Vec::new(),
        }
    }
}

impl<H, S> Flow for PpFlow<H, S>
where
    H: IterWorker<Msg = PipelineHeadMsg, Event = PipelineHeadEvent>,
    S: IterWorker<Msg = PipelineStageMsg, Event = PipelineStageEvent>,
{
    fn on_arrival(&mut self, req: Request) {
        let id = req.core.id;
        self.requests.borrow_mut().insert(req);
        self.stage_pool.admit(id);
    }

    fn tick(&mut self, now: Time) -> Vec<OrchAction> {
        self.stage_pool.tick_collect(now, &mut self.completed);
        self.completed
            .drain(..)
            .map(|req| OrchAction::Complete { req })
            .collect()
    }

    fn cluster(&self) -> &SharedGpuCluster {
        &self.cluster
    }
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    use super::*;
    use crate::common::RequestStore;
    use crate::test_helpers::{text_request, FakeModel};
    use crate::worker::{
        build_pipeline_head_worker, build_pipeline_stage_worker, PipelineHead, PipelineLayout,
        PipelineStage, WorkerConfig,
    };

    const STAGE_MS: f64 = 1.0;

    fn build_flow(
        replicas: u16,
        depth: u16,
        max_batch_tokens: u32,
    ) -> (
        PpFlow<PipelineHead<FakeModel>, PipelineStage<FakeModel>>,
        SharedRequests,
    ) {
        let store: SharedRequests = Rc::new(RefCell::new(RequestStore::new()));
        let model = Arc::new(FakeModel::for_ms(STAGE_MS));
        let config = WorkerConfig {
            max_batch_tokens: Some(max_batch_tokens),
            ..WorkerConfig::default()
        };
        let layout = PipelineLayout {
            depth,
            kv_bytes_per_token: 1,
            activation_bytes_per_token: 0,
        };
        let flow = PpFlow::new(
            Rc::clone(&store),
            &PpStagePoolConfig {
                replicas,
                depth,
                placement: DpPlacementPolicy::LeastQueued,
            },
            CostSource::analytic(100.0),
            |id, cluster| {
                build_pipeline_head_worker(
                    id,
                    "stage",
                    Arc::clone(&model),
                    layout,
                    Rc::clone(&store),
                    config,
                    None,
                    PP_STAGE_POOL,
                    "test-gpu",
                    Rc::clone(cluster),
                )
            },
            |id, _stage, cluster| {
                build_pipeline_stage_worker(
                    id,
                    "stage",
                    Arc::clone(&model),
                    Rc::clone(&store),
                    config,
                    None,
                    PP_STAGE_POOL,
                    "test-gpu",
                    Rc::clone(cluster),
                )
            },
        );
        (flow, store)
    }

    fn run(
        flow: &mut PpFlow<PipelineHead<FakeModel>, PipelineStage<FakeModel>>,
        steps: u64,
        step_ms: f64,
    ) -> Vec<(RequestId, Time)> {
        let mut completed = Vec::new();
        for step in 0..steps {
            let now = Time::from_ms(step as f64 * step_ms);
            for OrchAction::Complete { req } in flow.tick(now) {
                completed.push((req, now));
            }
        }
        completed
    }

    #[test]
    fn single_request_ttft_is_the_sum_of_stage_times() {
        let (mut flow, store) = build_flow(1, 4, 64);
        flow.on_arrival(text_request(RequestId(0), 16, 1, Time::ZERO));
        // 0.1 ms ticks, like the L7 driver. Hand-offs cost no tick, so the
        // request finishes at exactly 4 stage times and L7 sees it on that tick.
        let completed = run(&mut flow, 100, 0.1);
        assert_eq!(completed, vec![(RequestId(0), Time::from_ms(4.0))]);
        let first_output = store.borrow()[RequestId(0)].telemetry.first_output_time;
        assert_eq!(first_output, Some(Time::from_ms(4.0 * STAGE_MS)));
    }

    #[test]
    fn an_exit_on_the_formation_tick_joins_that_microbatch() {
        // A's prefill leaves stage 1 at 2 ms, the tick B arrives on. The head
        // must see that exit before it forms, so A's decode rides with B's
        // prefill in the microbatch starting at 2 ms instead of waiting for
        // stage 0 to finish it at 3 ms.
        let (mut flow, _store) = build_flow(1, 2, 64);
        flow.on_arrival(text_request(RequestId(0), 4, 2, Time::ZERO));
        let mut completed = Vec::new();
        for step in 0..10 {
            let now = Time::from_ms(step as f64);
            if step == 2 {
                flow.on_arrival(text_request(RequestId(1), 4, 1, now));
            }
            for OrchAction::Complete { req } in flow.tick(now) {
                completed.push((req, now));
            }
        }
        assert_eq!(
            completed,
            vec![
                (RequestId(0), Time::from_ms(4.0)),
                (RequestId(1), Time::from_ms(4.0)),
            ]
        );
    }

    #[test]
    fn a_long_prompt_fills_the_pipeline_one_chunk_per_stage() {
        // 64 tokens at 16 per microbatch: four chunks, one per stage time.
        let (mut flow, store) = build_flow(1, 4, 16);
        flow.on_arrival(text_request(RequestId(0), 64, 1, Time::ZERO));
        let completed = run(&mut flow, 100, 0.1);
        // Chunk k leaves stage 0 at k+1 ms and the pipeline at k+4 ms.
        assert_eq!(completed, vec![(RequestId(0), Time::from_ms(7.0))]);
        assert_eq!(
            store.borrow()[RequestId(0)]
                .progress
                .prefill_tokens_processed,
            64
        );
    }

    #[test]
    fn saturated_throughput_is_one_microbatch_per_stage_time() {
        let (mut flow, _store) = build_flow(1, 4, 16);
        for id in 0..8 {
            flow.on_arrival(text_request(RequestId(id), 16, 1, Time::ZERO));
        }
        let completed = run(&mut flow, 200, 0.1);
        let times: Vec<Time> = completed.into_iter().map(|(_, at)| at).collect();
        let expected: Vec<Time> = (4..12).map(|ms| Time::from_ms(ms as f64)).collect();
        assert_eq!(times, expected);
    }

    #[test]
    fn replicas_are_independent_pipelines_with_contiguous_gpus() {
        let (mut flow, _store) = build_flow(2, 2, 16);
        {
            let cluster = flow.cluster().borrow();
            let owners: Vec<u16> = cluster.gpus.iter().map(|gpu| gpu.worker_id).collect();
            assert_eq!(owners, vec![0, 1, 2, 3]);
        }
        flow.on_arrival(text_request(RequestId(0), 16, 1, Time::ZERO));
        flow.on_arrival(text_request(RequestId(1), 16, 1, Time::ZERO));
        let completed = run(&mut flow, 50, 0.1);
        // Least-queued placement sends one to each replica; both finish at 2 ms.
        assert_eq!(
            completed,
            vec![
                (RequestId(0), Time::from_ms(2.0)),
                (RequestId(1), Time::from_ms(2.0)),
            ]
        );
    }
}
