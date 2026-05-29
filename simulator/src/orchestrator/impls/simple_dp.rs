//! `simple_dp` — a reusable DP pool controller plus the single-pool unified flow.
//! L6a (`SimpleDpPoolController`) + L6b (`SimpleDpFlow`) live in one file because
//! the deployment is tiny, but stay two structs (L6 design.md §「simple DP」).

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, Request, RequestId, SharedRequests, Time};
use crate::worker::{CostSource, GpuCluster, IterWorker, SharedGpuCluster, WorkerEvent, WorkerMsg};

use super::super::{Flow, OrchAction, PoolEvent, UnifiedWorkerFactory};

/// Sentinel for a quiescent worker. `Option<Time>` would add a tag; the simulator
/// clock is a `u64` newtype, so this keeps the hot wakeup array dense while still
/// carrying a typed timestamp.
const NO_WAKEUP_TIME: Time = Time::from_ns(u64::MAX);

// ── Configs & policy (simple_dp-specific; L6 design.md §「simple DP」) ──────────

/// Worker placement within a DP pool.
#[derive(Clone, Copy, Debug)]
pub enum DpPlacementPolicy {
    LeastQueued,
    RoundRobin,
}

/// Config for one DP pool — pure orchestration (which workers, how to place).
/// The GPU facts each worker spans live on the [`UnifiedWorkerFactory`] (the
/// worker-stamping infra) and on the worker's own model, not here.
pub struct SimpleDpPoolConfig {
    pub pool: PoolId,
    pub num_workers: u16,
    pub placement: DpPlacementPolicy,
}

/// Config for the whole simple_dp deployment.
pub struct SimpleDpConfig {
    pub dp_pool: SimpleDpPoolConfig,
}

// ── L6a: pool-local orchestration ─────────────────────────────────────────────

pub struct SimpleDpPoolController<M: IterwiseUnifiedModel, W: IterWorker> {
    pool: PoolId,
    workers: Vec<W>,
    /// Hot wakeup filter: `NO_WAKEUP_TIME` means quiescent; any real timestamp is
    /// the earliest sim time at which the worker may advance. The L7 driver still
    /// calls the flow at its configured tick cadence, so this wakes on the first
    /// outer tick whose `now >= wakeup_time`.
    worker_wakeup_times: Vec<Time>,
    placement: DpPlacementPolicy,
    rr_next: usize,
    _model: std::marker::PhantomData<M>,
}

impl<M: IterwiseUnifiedModel, W: IterWorker> SimpleDpPoolController<M, W> {
    // ── Construction ──────────────────────────────────────────────────────────
    /// Build the pool's workers, each handed the shared `cluster` so it can
    /// self-register its GPU block. Sharing one cluster across pools keeps GPU
    /// ids globally unique — a multi-pool deployment threads the same cluster
    /// into every pool's `new`, so ids continue (`allocate` appends from the
    /// current length) instead of every pool restarting at 0.
    pub fn new(
        cfg: &SimpleDpPoolConfig,
        factory: &UnifiedWorkerFactory<M, W>,
        cluster: &SharedGpuCluster,
    ) -> Self {
        assert!(cfg.num_workers > 0, "simple_dp needs at least one worker");
        let workers: Vec<W> = (0..cfg.num_workers)
            .map(|i| factory.build(i, cfg.pool, cluster))
            .collect();
        Self {
            pool: cfg.pool,
            worker_wakeup_times: vec![NO_WAKEUP_TIME; workers.len()],
            workers,
            placement: cfg.placement,
            rr_next: 0,
            _model: std::marker::PhantomData,
        }
    }

    // ── Outward API (called by L6b) ───────────────────────────────────────────
    pub fn admit(&mut self, rid: RequestId) {
        self.admit_msg(WorkerMsg::Request(rid));
    }

    /// Pick a worker via the pool's placement policy and enqueue `msg` into it.
    /// Deployment-agnostic primitive: the unified flow admits a plain `Request`;
    /// a PD flow uses this to enqueue a `Handoff` (whose content does not depend
    /// on the chosen worker — the receiver fills in its own destination block).
    pub fn admit_msg(&mut self, msg: WorkerMsg) {
        let idx = self.choose_worker_idx();
        self.workers[idx].enqueue(msg);
        // An idle worker has no scheduled wakeup; new work must make it due
        // immediately. A computing worker already has a `compute_end` wakeup, so
        // keep that instead of forcing an early poll.
        if self.worker_wakeup_times[idx] == NO_WAKEUP_TIME {
            self.worker_wakeup_times[idx] = Time::ZERO;
        }
    }

    fn choose_worker_idx(&mut self) -> usize {
        match self.placement {
            DpPlacementPolicy::LeastQueued => self.choose_least_queued_idx(),
            DpPlacementPolicy::RoundRobin => self.choose_round_robin_idx(),
        }
    }

    fn choose_least_queued_idx(&self) -> usize {
        self.workers
            .iter()
            .enumerate()
            .min_by_key(|(idx, w)| {
                let s = w.status();
                (s.queued_requests, s.active_requests, *idx)
            })
            .map(|(idx, _)| idx)
            .expect("simple_dp has at least one worker")
    }

    fn choose_round_robin_idx(&mut self) -> usize {
        let idx = self.rr_next;
        self.rr_next = (self.rr_next + 1) % self.workers.len();
        idx
    }

    pub fn pool(&self) -> PoolId {
        self.pool
    }

    // ── Tick driving ──────────────────────────────────────────────────────────
    /// Sweep worker wakeup times and tick only due workers, each pushing its
    /// self-tagged `WorkerEvent`s into the caller's sink. One sweep — no separate
    /// drain pass; the worker already stamps its id so the pool needs no
    /// per-worker attribution loop.
    pub fn tick_collect(&mut self, now: Time, events: &mut Vec<WorkerEvent>) {
        debug_assert_eq!(self.worker_wakeup_times.len(), self.workers.len());
        for (wakeup_time, worker) in self
            .worker_wakeup_times
            .iter_mut()
            .zip(self.workers.iter_mut())
        {
            if *wakeup_time <= now {
                *wakeup_time = worker.tick(now, events).unwrap_or(NO_WAKEUP_TIME);
            }
        }
    }
}

/// Tag a worker event with its pool to form the L6b [`PoolEvent`]. The worker
/// already carries its own id; the caller supplies the pool context for the
/// per-pool event sink being converted.
pub(crate) fn to_pool_event(pool: PoolId, event: WorkerEvent) -> PoolEvent {
    match event {
        WorkerEvent::RequestComplete { worker, req } => {
            PoolEvent::RequestComplete { pool, worker, req }
        }
        WorkerEvent::PrefillDone { worker, req, send_spec } => {
            PoolEvent::PrefillDone { pool, worker, req, send_spec }
        }
    }
}

// ── L6b: deployment flow (the object L7 calls) ────────────────────────────────

pub struct SimpleDpFlow<M: IterwiseUnifiedModel, W: IterWorker> {
    requests: SharedRequests,
    dp_pool: SimpleDpPoolController<M, W>,
    /// Shared run-level GPU cluster (registry + transfer oracle), built here and
    /// threaded into the pool's construction so workers self-register and (PD
    /// only) keep a handle for runtime transfers. simple_dp has one pool today,
    /// but the ownership shape generalizes to multi-pool (allocate into the
    /// same cluster, ids continue).
    cluster: SharedGpuCluster,
    /// Reused per-tick event sink — workers push `WorkerEvent`s into it during
    /// `tick_collect`, then it is drained here and cleared for the next tick.
    events: Vec<WorkerEvent>,
}

impl<M: IterwiseUnifiedModel, W: IterWorker> SimpleDpFlow<M, W> {
    pub fn new(cfg: SimpleDpConfig, factory: UnifiedWorkerFactory<M, W>) -> Self {
        let requests = std::rc::Rc::clone(&factory.requests);
        // simple_dp has no inter-worker transfers, but the cluster is still the
        // GPU registry — wire a sentinel `CostSource` whose `submit_transfer`
        // would return ~zero if ever called (it isn't: only PD decode workers
        // call it, and there are none here).
        let cluster: SharedGpuCluster = std::rc::Rc::new(std::cell::RefCell::new(
            GpuCluster::new(CostSource::analytic(f64::INFINITY)),
        ));
        let dp_pool = SimpleDpPoolController::new(&cfg.dp_pool, &factory, &cluster);
        Self {
            requests,
            dp_pool,
            cluster,
            events: Vec::new(),
        }
    }

    fn on_pool_event(&mut self, event: PoolEvent) -> Vec<OrchAction> {
        match event {
            PoolEvent::RequestComplete { req, .. } => vec![OrchAction::Complete { req }],
            // A single-pool unified deployment never produces a prefill handoff.
            PoolEvent::PrefillDone { .. } => vec![],
        }
    }
}

impl<M: IterwiseUnifiedModel, W: IterWorker> Flow for SimpleDpFlow<M, W> {
    fn on_arrival(&mut self, req: Request) {
        let rid = req.id;
        self.requests.borrow_mut().insert(&req);
        self.dp_pool.admit(rid);
    }

    fn tick(&mut self, now: Time) -> Vec<OrchAction> {
        let mut events = std::mem::take(&mut self.events);
        events.clear();
        self.dp_pool.tick_collect(now, &mut events);
        let pool = self.dp_pool.pool();
        let mut actions = Vec::new();
        for ev in events.drain(..) {
            actions.extend(self.on_pool_event(to_pool_event(pool, ev)));
        }
        self.events = events;
        actions
    }

    fn cluster(&self) -> &SharedGpuCluster {
        &self.cluster
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::UnifiedArchInput;
    use crate::common::RequestStore;
    use crate::timing::cache::interp::{CoverageFlags, Metrics4};
    use crate::timing::LeafMetrics;
    use crate::worker::{BareboneWorker, WorkerConfig};
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    struct FakeModel {
        ms: f64,
    }
    impl IterwiseUnifiedModel for FakeModel {
        fn eval_iter(
            &self,
            _b: &UnifiedArchInput,
            slots: &mut Vec<LeafMetrics>,
            _scratch: &mut Vec<LeafMetrics>,
        ) -> LeafMetrics {
            slots.clear();
            LeafMetrics {
                m: Metrics4 {
                    time_ms: self.ms as f32,
                    flops: 0.0,
                    bytes: 0.0,
                    energy_j: 0.0,
                },
                coverage: CoverageFlags::EMPTY,
            }
        }
        fn kv_bytes_per_token(&self) -> u64 {
            1
        }
        fn gpus_per_replica(&self) -> u16 {
            1
        }
    }

    fn build_flow(
        num_workers: u16,
        placement: DpPlacementPolicy,
    ) -> (SimpleDpFlow<FakeModel, BareboneWorker<FakeModel>>, SharedRequests) {
        let store: SharedRequests = Rc::new(RefCell::new(RequestStore::new()));
        let factory = UnifiedWorkerFactory::new(
            Arc::new(FakeModel { ms: 1.0 }),
            Rc::clone(&store),
            WorkerConfig::default(),
            None,
            "test-gpu".to_string(),
            "main",
            BareboneWorker::<FakeModel>::new,
        );
        let cfg = SimpleDpConfig {
            dp_pool: SimpleDpPoolConfig {
                pool: PoolId(0),
                num_workers,
                placement,
            },
        };
        (SimpleDpFlow::new(cfg, factory), store)
    }

    #[test]
    fn all_arrivals_complete() {
        let (mut flow, _store) = build_flow(2, DpPlacementPolicy::RoundRobin);
        for id in 0..5u32 {
            flow.on_arrival(Request::new(RequestId(id), 8, 2, Time::ZERO));
        }
        let mut completed = Vec::new();
        for step in 0..500u64 {
            for a in flow.tick(Time::from_ms(step as f64)) {
                let OrchAction::Complete { req } = a;
                completed.push(req);
            }
        }
        completed.sort_by_key(|r| r.0);
        assert_eq!(
            completed,
            (0..5).map(RequestId).collect::<Vec<_>>(),
            "every arrival should complete exactly once"
        );
    }

    #[test]
    fn round_robin_spreads_across_workers() {
        // 4 arrivals over 2 workers, RoundRobin → 2 each. We can't see worker
        // internals directly, but all must still complete (smoke for placement).
        let (mut flow, _store) = build_flow(2, DpPlacementPolicy::RoundRobin);
        for id in 0..4u32 {
            flow.on_arrival(Request::new(RequestId(id), 4, 1, Time::ZERO));
        }
        let mut n = 0;
        for step in 0..200u64 {
            n += flow.tick(Time::from_ms(step as f64)).len();
        }
        assert_eq!(n, 4);
    }
}
