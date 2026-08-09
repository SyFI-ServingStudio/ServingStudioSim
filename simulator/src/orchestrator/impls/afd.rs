//! `afd` — the AFD (attention-FFN disaggregation) deployment flow (L6b, ref
//! `tick_loop`). It owns two **dedicated** pool controllers — an
//! [`AfdAttnPoolController`] (data-parallel attn shards + the cross-attn
//! aggregator) and an [`AfdFfnPoolController`] (one aggregated ffn replica) — over
//! one shared [`GpuCluster`](crate::worker::GpuCluster) (GPU registry + attn↔ffn
//! transfer oracle). Neither reuses `SimpleDpPoolController`: AFD's defining work
//! is the aggregation the attn controller holds.
//!
//! **Per-tick order — ffn first, then attn** (ref `tick_loop`): the ffn pool ticks
//! and produces this tick's QKV (`SectionReady`) and tokens (`IterComplete`); the
//! attn controller consumes those (scatter the QKV back to participants, release
//! finished requests) BEFORE the attn workers tick, so the scattered notifications
//! are in hand when they advance; the attn workers then tick and emit `IterStart` /
//! `AttnLayerOutputsReady`, which the attn controller aggregates into next tick's
//! ffn tasks. The aggregate→ffn hop is a one-tick relay (the task is submitted at
//! the end of tick `T` and the ffn pulls/computes it from tick `T+1`); the scatter
//! ffn→attn lands within the same tick. This is the only inter-pool coupling — no
//! shared notification queues, just the two event vectors the flow threads between
//! the controllers each tick.

use crate::common::{PoolId, Request, RequestId, SharedRequests, Time};
use crate::worker::{
    AfdAttnWorker, AfdFfnWorker, AttnWorkerEvent, AttnWorkerMsg, FfnTask, FfnWorkerEvent,
    SharedGpuCluster,
};

use super::super::{Flow, OrchAction};
use super::afd_attn_pool::AfdAttnPoolController;
use super::afd_ffn_pool::AfdFfnPoolController;

/// The attn pool's GPU/pool id (allocated first, so its GPU block ids come first).
pub const AFD_ATTN_POOL: PoolId = PoolId(0);
/// The ffn pool's GPU/pool id.
pub const AFD_FFN_POOL: PoolId = PoolId(1);

pub struct AfdFlow<WA, WF>
where
    WA: AfdAttnWorker,
    WA::Msg: From<AttnWorkerMsg>,
    WF: AfdFfnWorker,
{
    requests: SharedRequests,
    attn: AfdAttnPoolController<WA>,
    ffn: AfdFfnPoolController<WF>,
    /// Shared GPU registry + attn↔ffn transfer oracle (both pools' workers
    /// self-registered into it at construction; L7 serializes its `gpus`).
    cluster: SharedGpuCluster,
    // Reused per-tick buffers (cleared each tick; kept to avoid per-tick allocation).
    ffn_events: Vec<FfnWorkerEvent>,
    attn_events: Vec<AttnWorkerEvent>,
    tasks: Vec<FfnTask>,
    completed: Vec<RequestId>,
}

impl<WA, WF> AfdFlow<WA, WF>
where
    WA: AfdAttnWorker,
    WA::Msg: From<AttnWorkerMsg>,
    WF: AfdFfnWorker,
{
    /// Assemble the flow from the two already-built controllers and the shared
    /// cluster they were built into. The caller (the deployment) creates the cluster
    /// first, builds the attn pool then the ffn pool into it (so GPU ids continue
    /// across pools), and hands all three here.
    pub fn new(
        requests: SharedRequests,
        attn: AfdAttnPoolController<WA>,
        ffn: AfdFfnPoolController<WF>,
        cluster: SharedGpuCluster,
    ) -> Self {
        Self {
            requests,
            attn,
            ffn,
            cluster,
            ffn_events: Vec::new(),
            attn_events: Vec::new(),
            tasks: Vec::new(),
            completed: Vec::new(),
        }
    }
}

impl<WA, WF> Flow for AfdFlow<WA, WF>
where
    WA: AfdAttnWorker,
    WA::Msg: From<AttnWorkerMsg>,
    WF: AfdFfnWorker,
{
    fn on_arrival(&mut self, req: Request) {
        let rid = req.core.id;
        self.requests.borrow_mut().insert(req);
        self.attn.admit(rid);
    }

    fn tick(&mut self, now: Time) -> Vec<OrchAction> {
        // Take the reused buffers out so the controller calls below can borrow
        // `self.attn` / `self.ffn` mutably without aliasing the buffer fields.
        let mut ffn_events = std::mem::take(&mut self.ffn_events);
        let mut attn_events = std::mem::take(&mut self.attn_events);
        let mut tasks = std::mem::take(&mut self.tasks);
        let mut completed = std::mem::take(&mut self.completed);
        ffn_events.clear();
        attn_events.clear();
        tasks.clear();
        completed.clear();

        // FFN first (ref tick order): produce this tick's QKV + tokens.
        self.ffn.tick_collect(now, &mut ffn_events);
        // Scatter QKV back to attn participants + release/complete finished requests.
        self.attn.apply_ffn_events(&ffn_events, &mut completed, now);
        // Attn workers run: consume the scattered notifications / releases, emit
        // IterStart / AttnLayerOutputsReady.
        self.attn.tick_collect(now, &mut attn_events);
        // Aggregate this tick's attn events into next tick's ffn work.
        self.attn.aggregate(&attn_events, &mut tasks);
        for task in tasks.drain(..) {
            self.ffn.submit(task);
        }

        let actions = completed
            .iter()
            .map(|&req| OrchAction::Complete { req })
            .collect();

        self.ffn_events = ffn_events;
        self.attn_events = attn_events;
        self.tasks = tasks;
        self.completed = completed;
        actions
    }

    fn cluster(&self) -> &SharedGpuCluster {
        &self.cluster
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{RequestId, RequestStore, SessionInput};
    use crate::test_helpers::{test_cluster, text_request, FakeAttn, FakeFfn};
    use crate::worker::{DisaggAttnWorker, WorkerConfig};
    use std::cell::RefCell;
    use std::sync::Arc;

    /// An empty store — the flow's `on_arrival` inserts each request's facts itself,
    /// so the test must NOT pre-insert (the store asserts dense, in-order ids).
    fn empty_store() -> SharedRequests {
        std::rc::Rc::new(RefCell::new(RequestStore::new()))
    }

    /// Build an AFD flow with `attn_workers` attn shards + 1 ffn replica over a
    /// shared test cluster, all on the 2-layer fakes.
    fn flow(
        attn_workers: u16,
        store: SharedRequests,
    ) -> AfdFlow<DisaggAttnWorker<FakeAttn>, crate::worker::DisaggFfnWorker<FakeFfn>> {
        flow_with_attn_config(attn_workers, store, WorkerConfig::default())
    }

    fn flow_with_attn_config(
        attn_workers: u16,
        store: SharedRequests,
        attn_config: WorkerConfig,
    ) -> AfdFlow<DisaggAttnWorker<FakeAttn>, crate::worker::DisaggFfnWorker<FakeFfn>> {
        let cluster = test_cluster();
        let attn = AfdAttnPoolController::new(
            attn_workers,
            Arc::new(FakeAttn { ms: 1.0, layers: 2 }),
            std::rc::Rc::clone(&store),
            attn_config,
            AFD_ATTN_POOL,
            "attn-gpu",
            &cluster,
            None,
        );
        let ffn = AfdFfnPoolController::new(
            1,
            Arc::new(FakeFfn { ms: 1.0, layers: 2 }),
            std::rc::Rc::clone(&store),
            WorkerConfig::default(),
            AFD_FFN_POOL,
            "ffn-gpu",
            &cluster,
            None,
        );
        AfdFlow::new(store, attn, ffn, cluster)
    }

    fn drive<WA, WF>(flow: &mut AfdFlow<WA, WF>, steps: u64) -> Vec<RequestId>
    where
        WA: AfdAttnWorker,
        WA::Msg: From<AttnWorkerMsg>,
        WF: AfdFfnWorker,
    {
        let mut completed = Vec::new();
        for step in 0..steps {
            for a in flow.tick(Time::from_ms(step as f64)) {
                let OrchAction::Complete { req } = a;
                completed.push(req);
            }
        }
        completed
    }

    /// End-to-end smoke: one fresh request (prompt 8, decode 3) walks prolog → layers
    /// → terminal for each of its 3 token iterations and completes exactly once with
    /// `tokens_emitted == 3`; its KV is released afterward.
    #[test]
    fn single_request_prefill_then_decode_completes() {
        let store = empty_store();
        let mut f = flow(1, std::rc::Rc::clone(&store));
        f.on_arrival(text_request(RequestId(0), 8, 3, Time::ZERO));
        let completed = drive(&mut f, 3000);
        assert_eq!(completed, vec![RequestId(0)], "exactly one completion");
        assert_eq!(
            store.borrow()[RequestId(0)].progress.output_tokens_emitted,
            3
        );
        assert!(store.borrow()[RequestId(0)].lifecycle.completed);
    }

    /// Aggregation end-to-end: two requests placed on two attn shards (KV locality)
    /// both complete — the ffn folds both shards' same-slot work into one batch each
    /// iteration, and the per-layer barrier keeps the shards in lockstep without
    /// deadlock.
    #[test]
    fn two_shards_aggregate_and_both_complete() {
        let store = empty_store();
        let mut f = flow(2, std::rc::Rc::clone(&store));
        f.on_arrival(text_request(RequestId(0), 8, 2, Time::ZERO));
        f.on_arrival(text_request(RequestId(1), 8, 2, Time::ZERO));
        let mut completed = drive(&mut f, 4000);
        completed.sort_by_key(|r| r.0);
        assert_eq!(
            completed,
            vec![RequestId(0), RequestId(1)],
            "both complete once"
        );
        assert_eq!(
            store.borrow()[RequestId(0)].progress.output_tokens_emitted,
            2
        );
        assert_eq!(
            store.borrow()[RequestId(1)].progress.output_tokens_emitted,
            2
        );
    }

    #[test]
    fn afd_reuses_completed_session_kv_for_the_next_request() {
        let store = empty_store();
        let attn_config = WorkerConfig {
            attn_kv_bytes: 1_000,
            ..WorkerConfig::default()
        };
        let mut flow = flow_with_attn_config(1, std::rc::Rc::clone(&store), attn_config);

        let mut first = text_request(RequestId(0), 8, 2, Time::ZERO);
        first.definition.session = SessionInput::Session {
            session_id: 7,
            session_start_time: Time::ZERO,
            declared_prefix_tokens: 12,
        };
        flow.on_arrival(first);
        let mut completed = Vec::new();
        for step in 0..3_000 {
            for action in flow.tick(Time::from_ms(step as f64)) {
                let OrchAction::Complete { req } = action;
                completed.push(req);
            }
        }
        assert_eq!(completed, vec![RequestId(0)]);
        assert_eq!(
            store.borrow()[RequestId(0)]
                .progress
                .prefill_tokens_processed,
            20
        );
        assert_eq!(
            store.borrow()[RequestId(0)]
                .telemetry
                .prefix_cache_hit_tokens,
            Some(0)
        );

        let mut second = text_request(RequestId(1), 5, 2, Time::from_ms(3_000.0));
        second.definition.session = SessionInput::Session {
            session_id: 7,
            session_start_time: Time::ZERO,
            declared_prefix_tokens: 12,
        };
        flow.on_arrival(second);
        for step in 3_000..6_000 {
            for action in flow.tick(Time::from_ms(step as f64)) {
                let OrchAction::Complete { req } = action;
                completed.push(req);
            }
        }

        assert_eq!(completed, vec![RequestId(0), RequestId(1)]);
        assert_eq!(
            store.borrow()[RequestId(1)]
                .progress
                .prefill_tokens_processed,
            5
        );
        assert_eq!(
            store.borrow()[RequestId(1)]
                .telemetry
                .prefix_cache_hit_tokens,
            Some(12)
        );
    }

    /// Many requests staggered in time still all complete (no deadlock when slots
    /// fill, drain, and re-form across iterations).
    #[test]
    fn staggered_arrivals_all_complete() {
        let store = empty_store();
        let mut f = flow(2, std::rc::Rc::clone(&store));
        // Stagger arrivals over the first ticks so slots fill while others are mid-flight.
        let mut completed = Vec::new();
        for step in 0..6000u64 {
            if step < 6 {
                let id = step as u32;
                f.on_arrival(text_request(
                    RequestId(id),
                    8,
                    2,
                    Time::from_ms(step as f64),
                ));
            }
            for a in f.tick(Time::from_ms(step as f64)) {
                let OrchAction::Complete { req } = a;
                completed.push(req);
            }
        }
        completed.sort_by_key(|r| r.0);
        assert_eq!(
            completed,
            (0..6).map(RequestId).collect::<Vec<_>>(),
            "all six complete"
        );
    }
}
