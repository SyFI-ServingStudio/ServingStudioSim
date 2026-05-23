//! `simple_dp` — one DP pool of identical unified workers. L6a
//! (`SimpleDpPoolController`) + L6b (`SimpleDpFlow`) live in one file because the
//! deployment is tiny, but stay two structs (L6 design.md §「simple DP」).

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, Request, RequestId, SharedRequests, Time};
use crate::worker::{BareboneWorker, WorkerEvent, WorkerMsg};

use super::super::{Flow, OrchAction, PoolEvent, UnifiedWorkerFactory};

// ── Configs & policy (simple_dp-specific; L6 design.md §「simple DP」) ──────────

/// Worker placement within a DP pool.
#[derive(Clone, Copy, Debug)]
pub enum DpPlacementPolicy {
    LeastQueued,
    RoundRobin,
}

/// Config for one DP pool.
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

pub struct SimpleDpPoolController<M: IterwiseUnifiedModel> {
    pool: PoolId,
    workers: Vec<BareboneWorker<M>>,
    placement: DpPlacementPolicy,
    rr_next: usize,
}

impl<M: IterwiseUnifiedModel> SimpleDpPoolController<M> {
    // ── Construction ──────────────────────────────────────────────────────────
    pub fn new(cfg: &SimpleDpPoolConfig, factory: &UnifiedWorkerFactory<M>) -> Self {
        let workers: Vec<BareboneWorker<M>> =
            (0..cfg.num_workers).map(|i| factory.build(i)).collect();
        assert!(!workers.is_empty(), "simple_dp needs at least one worker");
        Self {
            pool: cfg.pool,
            workers,
            placement: cfg.placement,
            rr_next: 0,
        }
    }

    // ── Outward API (called by L6b) ───────────────────────────────────────────
    pub fn admit(&mut self, rid: RequestId) {
        let idx = self.choose_worker_idx();
        self.workers[idx].enqueue(WorkerMsg::Request(rid));
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

    // ── Tick driving ──────────────────────────────────────────────────────────
    pub fn tick_workers(&mut self, now: Time) {
        for w in &mut self.workers {
            w.tick(now);
        }
    }

    // ── Event handling (WorkerEvent → PoolEvent) ──────────────────────────────
    pub fn drain_events(&mut self) -> Vec<PoolEvent> {
        let mut out = Vec::new();
        let pool = self.pool;
        for w in &mut self.workers {
            let worker_id = w.id;
            for event in w.drain_events() {
                match event {
                    WorkerEvent::RequestComplete { req } => {
                        out.push(PoolEvent::RequestComplete {
                            pool,
                            worker: worker_id,
                            req,
                        });
                    }
                }
            }
        }
        out
    }
}

// ── L6b: deployment flow (the object L7 calls) ────────────────────────────────

pub struct SimpleDpFlow<M: IterwiseUnifiedModel> {
    requests: SharedRequests,
    dp_pool: SimpleDpPoolController<M>,
}

impl<M: IterwiseUnifiedModel> SimpleDpFlow<M> {
    pub fn new(cfg: SimpleDpConfig, factory: UnifiedWorkerFactory<M>) -> Self {
        let requests = std::rc::Rc::clone(&factory.requests);
        let dp_pool = SimpleDpPoolController::new(&cfg.dp_pool, &factory);
        Self { requests, dp_pool }
    }

    fn on_pool_event(&mut self, event: PoolEvent) -> Vec<OrchAction> {
        match event {
            PoolEvent::RequestComplete { req, .. } => vec![OrchAction::Complete { req }],
        }
    }
}

impl<M: IterwiseUnifiedModel> Flow for SimpleDpFlow<M> {
    fn on_arrival(&mut self, req: Request) {
        let rid = req.id;
        self.requests.borrow_mut().insert(&req);
        self.dp_pool.admit(rid);
    }

    fn tick(&mut self, now: Time) -> Vec<OrchAction> {
        self.dp_pool.tick_workers(now);
        let mut actions = Vec::new();
        for event in self.dp_pool.drain_events() {
            actions.extend(self.on_pool_event(event));
        }
        actions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::UnifiedArchInput;
    use crate::common::RequestStore;
    use crate::timing::cache::interp::{CoverageFlags, Metrics4};
    use crate::timing::LeafMetrics;
    use crate::worker::WorkerConfig;
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    struct FakeModel {
        ms: f64,
    }
    impl IterwiseUnifiedModel for FakeModel {
        fn cost_whole_iter_metrics(&self, _b: &UnifiedArchInput) -> LeafMetrics {
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
    }

    fn build_flow(num_workers: u16, placement: DpPlacementPolicy) -> (SimpleDpFlow<FakeModel>, SharedRequests) {
        let store: SharedRequests = Rc::new(RefCell::new(RequestStore::new()));
        let factory = UnifiedWorkerFactory::new(
            Arc::new(FakeModel { ms: 1.0 }),
            Rc::clone(&store),
            WorkerConfig::default(),
            None,
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
