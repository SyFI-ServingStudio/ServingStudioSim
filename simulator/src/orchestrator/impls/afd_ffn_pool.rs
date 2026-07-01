//! `afd_ffn_pool` — the ffn half of the AFD flow (L6a, ref `FfnScheduler`'s
//! worker-driving half). A thin pool over the [`DisaggFfnWorker`](crate::worker::DisaggFfnWorker)s:
//! the **attn** controller already did the cross-attn aggregation (it holds the
//! per-layer flush barrier), so this pool just routes each pre-composed
//! [`FfnTask`] to a worker, ticks the workers, and surfaces their
//! [`FfnWorkerEvent`]s back to the flow (which hands them to the attn controller
//! for the QKV scatter / completion).
//!
//! v1 runs a single ffn replica — one fused batch per slot — but the worker set is
//! a `Vec` so a future EP fan-out drops in without reshaping the controller. A
//! `FfnTask` is independent (the ffn side has no KV / no per-task state — its only
//! locality is the double-buffer pull/compute overlap), so tasks round-robin
//! across workers; with one worker that is just "always worker 0".
//!
//! This controller does NOT reuse `SimpleDpPoolController` (per the AFD plan): that
//! one routes a universal `Request` by placement and has no notion of a composed
//! task; here the unit of work is a `FfnTask` the attn controller built, so the
//! pool's surface is `submit(task)` + `tick_collect`, not `admit(rid)`.

use crate::arch::contract::FfnLayerwiseModel;
use crate::common::{PoolId, SharedRequests, Time};
use crate::worker::{
    DisaggFfnWorker, FfnTask, FfnWorkerEvent, FfnWorkerMsg, IterWorker, SharedGpuCluster,
    WorkerConfig,
};

/// Sentinel for a quiescent worker (mirrors `simple_dp`): `NO_WAKEUP_TIME` means
/// the worker has no scheduled advance; any real timestamp is the earliest sim
/// time it may progress.
const NO_WAKEUP_TIME: Time = Time::from_ns(u64::MAX);

pub struct AfdFfnPoolController<M: FfnLayerwiseModel> {
    workers: Vec<DisaggFfnWorker<M>>,
    /// Hot wakeup filter, parallel to `workers` (same scheme as `simple_dp`).
    worker_wakeup_times: Vec<Time>,
    /// Round-robin cursor for `submit` (tasks are independent; any worker can take one).
    rr_next: usize,
}

impl<M: FfnLayerwiseModel> AfdFfnPoolController<M> {
    // ── Construction ──────────────────────────────────────────────────────────
    /// Build the ffn pool's `num_workers` workers, each handed the shared `cluster`
    /// so it self-registers its GPU block + comm group, plus the run's `cost_log_dir`
    /// (the worker tags its rows `pool_tag = "ffn"`). Builds them directly rather than
    /// through `UnifiedWorkerFactory` (the disagg ffn worker's `new` is bespoke).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_workers: u16,
        model: std::sync::Arc<M>,
        requests: SharedRequests,
        config: WorkerConfig,
        pool: PoolId,
        gpu_name: &str,
        cluster: &SharedGpuCluster,
        cost_log_dir: Option<std::path::PathBuf>,
    ) -> Self {
        assert!(num_workers > 0, "afd ffn pool needs at least one worker");
        let workers: Vec<DisaggFfnWorker<M>> = (0..num_workers)
            .map(|i| {
                DisaggFfnWorker::new(
                    crate::common::WorkerId(i),
                    std::sync::Arc::clone(&model),
                    std::rc::Rc::clone(&requests),
                    config,
                    pool,
                    gpu_name,
                    std::rc::Rc::clone(cluster),
                    cost_log_dir.clone(),
                    "ffn",
                )
            })
            .collect();
        Self {
            worker_wakeup_times: vec![NO_WAKEUP_TIME; workers.len()],
            workers,
            rr_next: 0,
        }
    }

    // ── Outward API (called by the flow) ──────────────────────────────────────

    /// Route one aggregated task to a worker (round-robin) and wake it. A queued
    /// task makes an idle worker due immediately; a busy worker keeps its existing
    /// `compute_end` / `pull_end` wakeup (don't force an early poll).
    pub fn submit(&mut self, task: FfnTask) {
        let idx = self.rr_next;
        self.rr_next = (self.rr_next + 1) % self.workers.len();
        self.workers[idx].enqueue(FfnWorkerMsg::Task(task));
        if self.worker_wakeup_times[idx] == NO_WAKEUP_TIME {
            self.worker_wakeup_times[idx] = Time::ZERO;
        }
    }

    /// Sweep wakeup times and tick only due workers, each pushing its self-tagged
    /// events into the caller's sink (same one-sweep shape as `simple_dp`).
    pub fn tick_collect(&mut self, now: Time, events: &mut Vec<FfnWorkerEvent>) {
        for (wakeup, worker) in self
            .worker_wakeup_times
            .iter_mut()
            .zip(self.workers.iter_mut())
        {
            if *wakeup <= now {
                *wakeup = worker.tick(now, events).unwrap_or(NO_WAKEUP_TIME);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{PoolId, RequestId};
    use crate::test_helpers::{shared_with, test_cluster, FakeFfn};
    use crate::worker::{FfnTaskKind, FfnWorkerEvent};
    use std::sync::Arc;

    fn pool(store: SharedRequests) -> AfdFfnPoolController<FakeFfn> {
        AfdFfnPoolController::new(
            1,
            Arc::new(FakeFfn { ms: 1.0, layers: 2 }),
            store,
            WorkerConfig::default(),
            PoolId(1),
            "test-gpu",
            &test_cluster(),
            None,
        )
    }

    /// A submitted Bootstrap task computes (local input, no pull) and surfaces a
    /// `SectionReady` — the pool is a faithful pass-through to the worker.
    #[test]
    fn submit_routes_task_and_surfaces_section_ready() {
        let store = shared_with(&[(0, 8, 4)]);
        let mut p = pool(store);
        p.submit(FfnTask {
            kind: FfnTaskKind::Bootstrap,
            slot: 0,
            reqs: vec![RequestId(0)],
            pull_sources: Vec::new(),
        });
        let mut events = Vec::new();
        for step in 0..10u64 {
            p.tick_collect(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(events.len(), 1);
        assert!(matches!(
            events[0],
            FfnWorkerEvent::SectionReady {
                slot: 0,
                kind: FfnTaskKind::Bootstrap,
                ..
            }
        ));
    }

    /// A Terminal task surfaces an `IterComplete` (the worker owns token emission).
    #[test]
    fn terminal_surfaces_iter_complete() {
        let store = shared_with(&[(0, 8, 1)]);
        let mut p = pool(store);
        p.submit(FfnTask {
            kind: FfnTaskKind::Terminal,
            slot: 2,
            reqs: vec![RequestId(0)],
            pull_sources: Vec::new(),
        });
        let mut events = Vec::new();
        for step in 0..10u64 {
            p.tick_collect(Time::from_ms(step as f64), &mut events);
        }
        assert_eq!(
            events,
            vec![FfnWorkerEvent::IterComplete {
                worker: crate::common::WorkerId(0),
                slot: 2,
                reqs: vec![RequestId(0)],
                completed: vec![RequestId(0)],
            }]
        );
    }
}
