//! `pd` — prefill/decode-disaggregated deployment. Two DP pools of unified-style
//! workers: a prefill pool (whose workers only prefill, then hand off) and a
//! decode pool (whose workers pull the handed-off KV, then decode). L6b routes
//! the handoff: a prefill worker emits `PrefillDone` carrying its KV layout
//! (`send_gid`, `kv_tokens`); this flow forwards that to a placement-chosen
//! decode worker as a `WorkerMsg::Handoff`, which the decode worker assembles
//! into a full `TransferPlan` by combining the sender side with its own pre-
//! resolved destination block (L6 design.md §「PD handoff」).
//!
//! Workers own their own GPU id ranges (self-allocated from the shared cluster
//! at construction), so there is no flow-side endpoint lookup table — `tick`
//! simply forwards the sender side and the decode worker fills in its own
//! destination side.
//!
//! The KV transfer cost is modeled via the shared [`GpuCluster`] (the merged
//! GPU registry + send/recv stream contention oracle, fed by the `p2p_inter`
//! curve). One cluster is built here and threaded into both pool controllers;
//! workers register their GPU blocks in it at construction, and PD decode
//! workers keep a handle for runtime `submit_transfer`. All PD-specific
//! orchestration lives here; the reused `SimpleDpPoolController` stays
//! deployment-agnostic (the generic `admit_msg` primitive routes a `Handoff`,
//! and the shared `RequestStore` carries the request's prefill state across
//! pools).

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, Request, RequestId, SharedRequests, Time};
use crate::worker::{
    CostSource, GpuCluster, IterWorker, PdDecodeEvent, PdDecodeMsg, PdPrefillEvent, PdPrefillMsg,
    SharedGpuCluster,
};

use super::super::{Flow, OrchAction, UnifiedWorkerFactory};
use super::simple_dp::{SimpleDpPoolConfig, SimpleDpPoolController};

// ── L6b: PD deployment flow (the object L7 calls) ──────────────────────────────

/// Prefill + decode pools. Generic over each pool's (model, worker) pair so the
/// two halves can carry distinct archs/workers (`MP`/`WP` prefill, `MD`/`WD`
/// decode). The shared cluster aggregates GPUs from both pools with globally-
/// unique ids (prefill pool first, then decode) and doubles as the transfer
/// timing oracle.
pub struct PdFlow<MP, WP, MD, WD>
where
    MP: IterwiseUnifiedModel,
    WP: IterWorker<Msg = PdPrefillMsg, Event = PdPrefillEvent>,
    MD: IterwiseUnifiedModel,
    WD: IterWorker<Msg = PdDecodeMsg, Event = PdDecodeEvent>,
{
    requests: SharedRequests,
    prefill_pool: SimpleDpPoolController<MP, WP>,
    decode_pool: SimpleDpPoolController<MD, WD>,
    /// Shared run-level GPU cluster — both the registry (workers `allocate`
    /// their own block into it at construction) and the transfer oracle (PD
    /// decode workers retain a handle for `submit_transfer`). One object, two
    /// roles; see `worker::gpu_cluster`.
    cluster: SharedGpuCluster,
    /// Reused per-tick event sinks, typed per pool (`PdPrefillEvent` for the
    /// producer side, `PdDecodeEvent` for the consumer). Drained + cleared
    /// each tick; the typed sinks make "decode never emits PrefillDone" a
    /// compile-time fact rather than a silent ignore arm.
    prefill_events: Vec<PdPrefillEvent>,
    decode_events: Vec<PdDecodeEvent>,
}

impl<MP, WP, MD, WD> PdFlow<MP, WP, MD, WD>
where
    MP: IterwiseUnifiedModel,
    WP: IterWorker<Msg = PdPrefillMsg, Event = PdPrefillEvent>,
    MD: IterwiseUnifiedModel,
    WD: IterWorker<Msg = PdDecodeMsg, Event = PdDecodeEvent>,
{
    /// Build the shared `GpuCluster` with the production transfer cost, then
    /// both pools (their workers self-register their GPU blocks in the cluster
    /// at construction). `cost` is the cluster's per-link transfer-cost source
    /// (the `p2p_inter` kernel in production, an analytic curve in tests).
    pub fn new(
        prefill_cfg: &SimpleDpPoolConfig,
        prefill_factory: &UnifiedWorkerFactory<MP, WP>,
        decode_cfg: &SimpleDpPoolConfig,
        decode_factory: &UnifiedWorkerFactory<MD, WD>,
        cost: CostSource,
    ) -> Self {
        let requests = std::rc::Rc::clone(&prefill_factory.requests);
        let cluster: SharedGpuCluster =
            std::rc::Rc::new(std::cell::RefCell::new(GpuCluster::new(cost)));
        // Allocate prefill GPUs first, then decode — ids continue across pools.
        let prefill_pool = SimpleDpPoolController::new(prefill_cfg, prefill_factory, &cluster);
        let decode_pool = SimpleDpPoolController::new(decode_cfg, decode_factory, &cluster);
        Self {
            requests,
            prefill_pool,
            decode_pool,
            cluster,
            prefill_events: Vec::new(),
            decode_events: Vec::new(),
        }
    }
}

impl<MP, WP, MD, WD> Flow for PdFlow<MP, WP, MD, WD>
where
    MP: IterwiseUnifiedModel,
    WP: IterWorker<Msg = PdPrefillMsg, Event = PdPrefillEvent>,
    MD: IterwiseUnifiedModel,
    WD: IterWorker<Msg = PdDecodeMsg, Event = PdDecodeEvent>,
    WP::Msg: From<RequestId>,
{
    fn on_arrival(&mut self, req: Request) {
        let rid = req.id;
        self.requests.borrow_mut().insert(&req);
        self.prefill_pool.admit(rid);
    }

    fn tick(&mut self, now: Time) -> Vec<OrchAction> {
        let mut actions = Vec::new();

        // Producer first (L6 INV-11): tick + collect the prefill pool. A finished
        // prefill either completes immediately (single-token request) or hands off
        // to the decode pool this same tick (so the decode pool can admit it below).
        let mut prefill_events = std::mem::take(&mut self.prefill_events);
        prefill_events.clear();
        self.prefill_pool.tick_collect(now, &mut prefill_events);
        // Each prefill event is either a request completion (single-token) or
        // a PrefillDone handoff. Decode-side acks (PullComplete) are absent at
        // compile time — `PdPrefillEvent` simply has no such variant.
        for ev in prefill_events.drain(..) {
            match ev {
                PdPrefillEvent::RequestComplete { req, .. } => {
                    actions.push(OrchAction::Complete { req });
                }
                PdPrefillEvent::PrefillDone { worker, req, send_gid, kv_tokens } => {
                    // Carry the prefill worker id through so the decode side can
                    // later ack it (`ReleaseKv`) and let it drop the held KV
                    // reservation. The send-side comm group alone is not enough
                    // — multiple prefill workers may share an arch.
                    self.decode_pool.admit_msg(PdDecodeMsg::Handoff {
                        req,
                        send_gid,
                        tokens: kv_tokens,
                        prefill_worker: worker,
                    });
                }
            }
        }
        self.prefill_events = prefill_events;

        // Consumer: tick + collect the decode pool. It completes requests and
        // emits PD pull acks (PullComplete) when KV lands at its workers.
        let mut decode_events = std::mem::take(&mut self.decode_events);
        decode_events.clear();
        self.decode_pool.tick_collect(now, &mut decode_events);
        for ev in decode_events.drain(..) {
            match ev {
                PdDecodeEvent::RequestComplete { req, .. } => {
                    actions.push(OrchAction::Complete { req });
                }
                // KV has fully landed at decode — tell the originating prefill
                // worker to drop its held reservation. Targeted route (by
                // `prefill_worker` id), *not* placement-chosen: only that
                // worker tracks this request's held KV.
                PdDecodeEvent::PullComplete { req, prefill_worker, .. } => {
                    self.prefill_pool
                        .route_msg_to(prefill_worker, PdPrefillMsg::ReleaseKv { req });
                }
            }
        }
        self.decode_events = decode_events;

        actions
    }

    fn cluster(&self) -> &SharedGpuCluster {
        &self.cluster
    }
}

/// Convenience: the two pool ids a PD deployment uses (prefill = 0, decode = 1).
pub const PD_PREFILL_POOL: PoolId = PoolId(0);
pub const PD_DECODE_POOL: PoolId = PoolId(1);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{RequestId, RequestStore};
    use crate::orchestrator::DpPlacementPolicy;
    use crate::test_helpers::FakeModel;
    use crate::worker::{PdDecodeWorker, PdPrefillWorker, WorkerConfig};
    use std::cell::RefCell;
    use std::rc::Rc;
    use std::sync::Arc;

    fn build_flow() -> (
        PdFlow<FakeModel, PdPrefillWorker<FakeModel>, FakeModel, PdDecodeWorker<FakeModel>>,
        SharedRequests,
    ) {
        let store: SharedRequests = Rc::new(RefCell::new(RequestStore::new()));
        let prefill_factory = UnifiedWorkerFactory::new(
            Arc::new(FakeModel::for_ms(1.0)),
            Rc::clone(&store),
            WorkerConfig::default(),
            None,
            "test-gpu".to_string(),
            "prefill",
            PdPrefillWorker::<FakeModel>::new,
        );
        let decode_factory = UnifiedWorkerFactory::new(
            Arc::new(FakeModel::for_ms(1.0)),
            Rc::clone(&store),
            WorkerConfig::default(),
            None,
            "test-gpu".to_string(),
            "decode",
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
        // FakeModel: 1 gpu/replica, 1 attn dp group → num_attn_shards = 1.
        // Analytic cost so the transfer path is exercised without a real kernel.
        let flow = PdFlow::new(
            &prefill_cfg,
            &prefill_factory,
            &decode_cfg,
            &decode_factory,
            CostSource::analytic(100.0),
        );
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
    fn cluster_aggregates_both_pools() {
        let (flow, _store) = build_flow();
        // 1 prefill worker + 1 decode worker, 1 gpu each → 2 gpus, dense 0..2 ids.
        let cluster = flow.cluster().borrow();
        assert_eq!(cluster.num_gpus(), 2);
        let ids: Vec<u16> = cluster.gpus.iter().map(|g| g.id).collect();
        assert_eq!(ids, vec![0, 1]);
    }
}
