//! `pd` — prefill/decode-disaggregated deployment. Two DP pools of unified-style
//! workers: a prefill pool (whose workers only prefill, then hand off) and a
//! decode pool (whose workers admit already-prefilled requests straight into
//! decode). L6b routes the handoff: a prefill worker emits `PrefillDone`, this
//! flow enqueues the request into the decode pool (L6 design.md §「PD handoff」).
//!
//! Copied from `simple_dp` (build-first): each pool is its own
//! `SimpleDpPoolController`, generic over its (model, worker) pair — the prefill
//! and decode pools may run different archs/workers. v1 ignores the prefill→decode
//! KV transfer cost; the handoff is a control-plane event only, and the shared
//! `RequestStore` carries the request's prefill state across pools.

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, Request, RequestId, SharedRequests, Time};
use crate::worker::IterWorker;

use super::super::{Flow, GpuInventory, OrchAction, PoolEvent, UnifiedWorkerFactory};
use super::simple_dp::{SimpleDpPoolConfig, SimpleDpPoolController};

// ── L6b: PD deployment flow (the object L7 calls) ──────────────────────────────

/// Prefill + decode pools. Generic over each pool's (model, worker) pair so the
/// two halves can carry distinct archs/workers (`MP`/`WP` prefill, `MD`/`WD`
/// decode). The run-level `inventory` aggregates GPUs from both pools with
/// globally-unique ids (prefill pool first, then decode).
pub struct PdFlow<MP, WP, MD, WD>
where
    MP: IterwiseUnifiedModel,
    WP: IterWorker,
    MD: IterwiseUnifiedModel,
    WD: IterWorker,
{
    requests: SharedRequests,
    prefill_pool: SimpleDpPoolController<MP, WP>,
    decode_pool: SimpleDpPoolController<MD, WD>,
    inventory: GpuInventory,
}

impl<MP, WP, MD, WD> PdFlow<MP, WP, MD, WD>
where
    MP: IterwiseUnifiedModel,
    WP: IterWorker,
    MD: IterwiseUnifiedModel,
    WD: IterWorker,
{
    /// Build both pools into one run-level inventory. `prefill_cfg`/`decode_cfg`
    /// carry distinct `PoolId`s so each pool's GPUs are tagged correctly; the
    /// caller threads the same shared store into both factories.
    pub fn new(
        prefill_cfg: &SimpleDpPoolConfig,
        prefill_factory: &UnifiedWorkerFactory<MP, WP>,
        decode_cfg: &SimpleDpPoolConfig,
        decode_factory: &UnifiedWorkerFactory<MD, WD>,
    ) -> Self {
        let requests = std::rc::Rc::clone(&prefill_factory.requests);
        let mut inventory = GpuInventory::default();
        // Allocate prefill GPUs first, then decode — ids continue across pools.
        let prefill_pool = SimpleDpPoolController::new(prefill_cfg, prefill_factory, &mut inventory);
        let decode_pool = SimpleDpPoolController::new(decode_cfg, decode_factory, &mut inventory);
        Self {
            requests,
            prefill_pool,
            decode_pool,
            inventory,
        }
    }

    fn admit_to_decode(&mut self, req: RequestId) {
        self.decode_pool.admit(req);
    }
}

impl<MP, WP, MD, WD> Flow for PdFlow<MP, WP, MD, WD>
where
    MP: IterwiseUnifiedModel,
    WP: IterWorker,
    MD: IterwiseUnifiedModel,
    WD: IterWorker,
{
    fn on_arrival(&mut self, req: Request) {
        let rid = req.id;
        self.requests.borrow_mut().insert(&req);
        self.prefill_pool.admit(rid);
    }

    fn tick(&mut self, now: Time) -> Vec<OrchAction> {
        let mut actions = Vec::new();

        // Producer first (L6 INV-11): tick + drain the prefill pool. A finished
        // prefill either completes immediately (single-token request) or hands off
        // to the decode pool this same tick (so the decode pool can admit it below).
        self.prefill_pool.tick_workers(now);
        let mut handoffs = Vec::new();
        for event in self.prefill_pool.drain_events() {
            match event {
                PoolEvent::PrefillDone { req, .. } => handoffs.push(req),
                PoolEvent::RequestComplete { req, .. } => {
                    actions.push(OrchAction::Complete { req })
                }
            }
        }
        for req in handoffs {
            self.admit_to_decode(req);
        }

        // Consumer: tick + drain the decode pool. It only ever completes.
        self.decode_pool.tick_workers(now);
        for event in self.decode_pool.drain_events() {
            match event {
                PoolEvent::RequestComplete { req, .. } => {
                    actions.push(OrchAction::Complete { req })
                }
                // A decode pool never hands off.
                PoolEvent::PrefillDone { .. } => {}
            }
        }

        actions
    }

    fn inventory(&self) -> &GpuInventory {
        &self.inventory
    }
}

/// Convenience: the two pool ids a PD deployment uses (prefill = 0, decode = 1).
pub const PD_PREFILL_POOL: PoolId = PoolId(0);
pub const PD_DECODE_POOL: PoolId = PoolId(1);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::UnifiedArchInput;
    use crate::common::RequestStore;
    use crate::orchestrator::DpPlacementPolicy;
    use crate::timing::cache::interp::{CoverageFlags, Metrics4};
    use crate::timing::LeafMetrics;
    use crate::worker::{PdDecodeWorker, PdPrefillWorker, WorkerConfig};
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    struct FakeModel {
        ms: f64,
    }
    impl IterwiseUnifiedModel for FakeModel {
        fn eval_iter(&self, _b: &UnifiedArchInput, slots: &mut Vec<LeafMetrics>) -> LeafMetrics {
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

    fn build_flow() -> (
        PdFlow<FakeModel, PdPrefillWorker<FakeModel>, FakeModel, PdDecodeWorker<FakeModel>>,
        SharedRequests,
    ) {
        let store: SharedRequests = Rc::new(RefCell::new(RequestStore::new()));
        let prefill_factory = UnifiedWorkerFactory::new(
            Arc::new(FakeModel { ms: 1.0 }),
            Rc::clone(&store),
            WorkerConfig::default(),
            None,
            "test-gpu".to_string(),
            1,
            PdPrefillWorker::<FakeModel>::new,
        );
        let decode_factory = UnifiedWorkerFactory::new(
            Arc::new(FakeModel { ms: 1.0 }),
            Rc::clone(&store),
            WorkerConfig::default(),
            None,
            "test-gpu".to_string(),
            1,
            PdDecodeWorker::<FakeModel>::new,
        );
        let prefill_cfg = SimpleDpPoolConfig {
            pool: PD_PREFILL_POOL,
            num_workers: 1,
            placement: DpPlacementPolicy::RoundRobin,
        };
        let decode_cfg = SimpleDpPoolConfig {
            pool: PD_DECODE_POOL,
            num_workers: 1,
            placement: DpPlacementPolicy::RoundRobin,
        };
        let flow = PdFlow::new(&prefill_cfg, &prefill_factory, &decode_cfg, &decode_factory);
        (flow, store)
    }

    #[test]
    fn request_prefills_then_decodes_to_completion() {
        let (mut flow, store) = build_flow();
        flow.on_arrival(Request::new(RequestId(0), 16, 3, Time::ZERO));
        let mut completed = Vec::new();
        for step in 0..500u64 {
            for a in flow.tick(Time::from_ms(step as f64)) {
                let OrchAction::Complete { req } = a;
                completed.push(req);
            }
        }
        assert_eq!(completed, vec![RequestId(0)]);
        let s = store.borrow();
        let r = &s[RequestId(0)];
        assert!(r.completed);
        assert_eq!(r.tokens_emitted, 3, "1 prefill token + 2 decode tokens");
    }

    #[test]
    fn single_token_request_completes_at_prefill() {
        let (mut flow, _store) = build_flow();
        flow.on_arrival(Request::new(RequestId(0), 16, 1, Time::ZERO));
        let mut completed = Vec::new();
        for step in 0..100u64 {
            for a in flow.tick(Time::from_ms(step as f64)) {
                let OrchAction::Complete { req } = a;
                completed.push(req);
            }
        }
        assert_eq!(completed, vec![RequestId(0)]);
    }

    #[test]
    fn inventory_aggregates_both_pools() {
        let (flow, _store) = build_flow();
        // 1 prefill worker + 1 decode worker, 1 gpu each → 2 gpus, dense 0..2 ids.
        assert_eq!(flow.inventory().num_gpus(), 2);
        let ids: Vec<u16> = flow.inventory().gpus.iter().map(|g| g.id).collect();
        assert_eq!(ids, vec![0, 1]);
    }
}
