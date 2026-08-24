//! `simple_dp` — a reusable DP pool controller plus the single-pool unified flow.
//! L6a (`SimpleDpPoolController`) + L6b (`SimpleDpFlow`) live in one file because
//! the deployment is tiny, but stay two structs (L6 design.md §「simple DP」).

use crate::common::{PoolId, Request, RequestId, SharedRequests, Time, WorkerId};
use crate::worker::{CostSource, GpuCluster, IterWorker, SharedGpuCluster, WorkerEventCommon};

use super::super::{Flow, OrchAction, WorkerFactory};

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
/// The GPU facts each worker spans live behind its [`WorkerFactory`] and on the
/// worker's paired model/arch contract, not here.
pub struct SimpleDpPoolConfig {
    pub pool: PoolId,
    pub num_workers: u16,
    pub placement: DpPlacementPolicy,
}

/// Config for the whole `simple_dp` deployment.
pub struct SimpleDpConfig {
    pub dp_pool: SimpleDpPoolConfig,
}

// ── L6a: pool-local orchestration ─────────────────────────────────────────────

pub struct SimpleDpPoolController<W: IterWorker> {
    pool: PoolId,
    workers: Vec<W>,
    /// Hot wakeup filter: `NO_WAKEUP_TIME` means quiescent; any real timestamp is
    /// the earliest sim time at which the worker may advance. The L7 driver still
    /// calls the flow at its configured tick cadence, so this wakes on the first
    /// outer tick whose `now >= wakeup_time`.
    worker_wakeup_times: Vec<Time>,
    placement: DpPlacementPolicy,
    rr_next: usize,
}

impl<W: IterWorker> SimpleDpPoolController<W> {
    // ── Construction ──────────────────────────────────────────────────────────
    /// Build the pool's workers, each handed the shared `cluster` so it can
    /// self-register its GPU block. Sharing one cluster across pools keeps GPU
    /// ids globally unique — a multi-pool deployment threads the same cluster
    /// into every pool's `new`, so ids continue (`allocate` appends from the
    /// current length) instead of every pool restarting at 0.
    pub fn new<F>(cfg: &SimpleDpPoolConfig, factory: &F, cluster: &SharedGpuCluster) -> Self
    where
        F: WorkerFactory<W>,
    {
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
        }
    }

    // ── Outward API (called by L6b) ───────────────────────────────────────────

    /// Pick a worker via the pool's placement policy and enqueue `msg` into it.
    /// Deployment-agnostic primitive over `W::Msg`: the unified flow admits a
    /// plain `Request` (via [`Self::admit`] convenience); a PD flow uses this
    /// to enqueue a `Handoff` whose content does not depend on the chosen
    /// worker — the receiver fills in its own destination block.
    pub fn admit_msg(&mut self, msg: W::Msg) {
        let idx = self.choose_worker_idx();
        self.workers[idx].enqueue(msg);
        // An idle worker has no scheduled wakeup; new work must make it due
        // immediately. A computing worker already has a `compute_end` wakeup, so
        // keep that instead of forcing an early poll.
        if self.worker_wakeup_times[idx] == NO_WAKEUP_TIME {
            self.worker_wakeup_times[idx] = Time::ZERO;
        }
    }

    /// Route `msg` to a *specific* worker (bypasses placement). Used for
    /// targeted acks like PD's `ReleaseKv`, where the message must reach the
    /// exact worker that holds the addressed request's KV. Panics if
    /// `worker_id` is outside this pool — silently dropping the ack would
    /// leak the held reservation forever, so a bad id is treated as a bug.
    pub fn route_msg_to(&mut self, worker_id: WorkerId, msg: W::Msg) {
        let idx = worker_id.0 as usize;
        self.workers[idx].enqueue(msg);
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

    #[must_use]
    pub fn pool(&self) -> PoolId {
        self.pool
    }

    // ── Tick driving ──────────────────────────────────────────────────────────
    /// Sweep worker wakeup times and tick only due workers, each pushing its
    /// self-tagged events into the caller's sink. One sweep — no separate drain
    /// pass; the worker already stamps its id so the pool needs no per-worker
    /// attribution loop. The sink type is `Vec<W::Event>` so each pool gets its
    /// own role-specific event stream (a barebone pool's sink takes
    /// `WorkerEventCommon`, a PD prefill pool's sink takes `PdPrefillEvent`,
    /// etc.).
    pub fn tick_collect(&mut self, now: Time, events: &mut Vec<W::Event>) {
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

// Convenience: every Msg enum impls `From<RequestId>` so the universal
// "admit a request" entry point can stay one method. Decoupled into its own
// `impl` block so workers whose Msg does not (or cannot) carry Request — none
// today, but the bound keeps the trait surface minimal — still compile.
impl<W: IterWorker> SimpleDpPoolController<W>
where
    W::Msg: From<RequestId>,
{
    /// Admit a fresh request (the universal entry) — wraps in the chosen
    /// worker's `W::Msg::from(rid)`. PD-side admits via `admit_msg` directly.
    pub fn admit(&mut self, rid: RequestId) {
        self.admit_msg(W::Msg::from(rid));
    }
}

// ── L6b: deployment flow (the object L7 calls) ────────────────────────────────

pub struct SimpleDpFlow<W: IterWorker<Event = WorkerEventCommon>> {
    requests: SharedRequests,
    dp_pool: SimpleDpPoolController<W>,
    /// Shared run-level GPU cluster (registry + transfer oracle), built here and
    /// threaded into the pool's construction so workers self-register and (PD
    /// only) keep a handle for runtime transfers. `simple_dp` has one pool today,
    /// but the ownership shape generalizes to multi-pool (allocate into the
    /// same cluster, ids continue).
    cluster: SharedGpuCluster,
    /// Reused per-tick event sink — workers push `WorkerEventCommon`s (a unified
    /// deployment's workers only emit `RequestComplete`) into it during
    /// `tick_collect`, then it is drained here and cleared for the next tick.
    events: Vec<WorkerEventCommon>,
}

impl<W> SimpleDpFlow<W>
where
    W: IterWorker<Event = WorkerEventCommon>,
{
    pub fn new<F>(cfg: SimpleDpConfig, factory: F) -> Self
    where
        F: WorkerFactory<W>,
    {
        let requests = std::rc::Rc::clone(factory.requests());
        // simple_dp has no inter-worker transfers, but the cluster is still the
        // GPU registry — wire a sentinel `CostSource` whose `submit_transfer`
        // would return ~zero if ever called (it isn't: only PD decode workers
        // call it, and there are none here).
        let cluster: SharedGpuCluster = std::rc::Rc::new(std::cell::RefCell::new(GpuCluster::new(
            CostSource::analytic(f64::INFINITY),
        )));
        let dp_pool = SimpleDpPoolController::new(&cfg.dp_pool, &factory, &cluster);
        Self {
            requests,
            dp_pool,
            cluster,
            events: Vec::new(),
        }
    }
}

impl<W> Flow for SimpleDpFlow<W>
where
    W: IterWorker<Event = WorkerEventCommon>,
    W::Msg: From<RequestId>,
{
    fn on_arrival(&mut self, req: Request) {
        let rid = req.core.id;
        self.requests.borrow_mut().insert(req);
        self.dp_pool.admit(rid);
    }

    fn tick(&mut self, now: Time) -> Vec<OrchAction> {
        let mut events = std::mem::take(&mut self.events);
        events.clear();
        self.dp_pool.tick_collect(now, &mut events);
        let mut actions = Vec::new();
        for ev in events.drain(..) {
            let WorkerEventCommon::RequestComplete { req, .. } = ev;
            actions.push(OrchAction::Complete { req });
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
    use crate::common::RequestStore;
    use crate::orchestrator::UnifiedWorkerFactory;
    use crate::test_helpers::{text_request, FakeModel};
    use crate::worker::{build_barebone_worker, BareboneWorker, WorkerConfig};
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    fn build_flow(
        num_workers: u16,
        placement: DpPlacementPolicy,
    ) -> (SimpleDpFlow<BareboneWorker<FakeModel>>, SharedRequests) {
        let store: SharedRequests = Rc::new(RefCell::new(RequestStore::new()));
        let factory = UnifiedWorkerFactory::new(
            Arc::new(FakeModel::for_ms(1.0)),
            Rc::clone(&store),
            WorkerConfig::default(),
            None,
            "test-gpu".to_string(),
            "main",
            build_barebone_worker::<FakeModel>,
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
            flow.on_arrival(text_request(RequestId(id), 8, 2, Time::ZERO));
        }
        let mut completed = Vec::new();
        for step in 0..500u64 {
            #[allow(
                clippy::cast_precision_loss,
                reason = "step is a tick counter bounded by the loop range, far under f64's exact integer range"
            )]
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
            flow.on_arrival(text_request(RequestId(id), 4, 1, Time::ZERO));
        }
        let mut n = 0;
        for step in 0..200u64 {
            #[allow(
                clippy::cast_precision_loss,
                reason = "step is a tick counter bounded by the loop range, far under f64's exact integer range"
            )]
            let now = Time::from_ms(step as f64);
            n += flow.tick(now).len();
        }
        assert_eq!(n, 4);
    }
}
