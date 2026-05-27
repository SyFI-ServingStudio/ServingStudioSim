//! Shared L6 vocabulary (`OrchAction` / `PoolEvent`) + the worker-stamping infra
//! (`UnifiedWorkerFactory`) used across deployment impls. Kept out of `mod.rs` so
//! the module root holds only the `Flow` contract and the module wiring. See L6
//! design.md §Organization rule (`common.rs # optional shared vocabulary`).

use std::path::PathBuf;
use std::sync::Arc;

use serde::Serialize;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, RequestId, SharedRequests, WorkerId};
use crate::worker::{IterWorker, WorkerConfig};

/// Deployment-level action returned to L7 each tick. Barebone only completes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrchAction {
    Complete { req: RequestId },
}

/// Pool-level event (L6a → L6b). One per worker-level transition a pool surfaces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PoolEvent {
    RequestComplete {
        pool: PoolId,
        worker: WorkerId,
        req: RequestId,
    },
    /// A PD prefill pool finished a request's prefill — L6b hands it off to the
    /// decode pool. Only a prefill pool surfaces this.
    PrefillDone {
        pool: PoolId,
        worker: WorkerId,
        req: RequestId,
    },
}

/// One physical GPU and who owns it. The run's GPU facts are a flat list of these
/// (see [`GpuInventory`]); `pool` + `worker_id` make the worker→gpu and pool→gpu
/// groupings derivable without a second table.
#[derive(Clone, Debug, Serialize)]
pub struct GpuInfo {
    pub id: u16,
    pub name: String,
    pub pool: u16,
    pub worker_id: u16,
}

/// The GPUs a run modeled — a *reporting* artifact (serialized to
/// `raw/run_meta.json`), not yet a timing oracle like ref's stream-serialized
/// `GpuCluster`. L6 assembles it as it builds workers ([`GpuInventory::allocate`],
/// ref's `allocate(n)` shape); the GPU count per worker is the L4 parallel-dim
/// product, threaded in by the deployment.
#[derive(Clone, Debug, Default, Serialize)]
pub struct GpuInventory {
    pub gpus: Vec<GpuInfo>,
}

impl GpuInventory {
    pub fn num_gpus(&self) -> usize {
        self.gpus.len()
    }

    /// Register `n` contiguous-id GPUs to `(pool, worker)`, all sharing `name`.
    /// Ids continue from the current length, so calling once per worker yields a
    /// dense `0..total` id space.
    pub fn allocate(&mut self, pool: u16, worker_id: u16, n: u16, name: &str) {
        let base = self.gpus.len() as u16;
        for offset in 0..n {
            self.gpus.push(GpuInfo {
                id: base + offset,
                name: name.to_string(),
                pool,
                worker_id,
            });
        }
    }
}

/// Builds identical unified workers for a DP pool, each sharing the one
/// `SharedRequests` handle and an `Arc` of the model. The worker sizes its own
/// `KvPool` from `worker_config.attn_kv_bytes`. (L7 will generalize this into a
/// trait; for now a concrete generic struct is enough.)
/// Constructor signature shared by every iter-wise worker (`BareboneWorker::new`,
/// `HpUnifiedWorker::new`, …). The factory is handed the chosen worker's `new` as
/// a plain function pointer, so it can stamp the concrete `W` without `W::new`
/// living on the [`IterWorker`] trait.
pub type WorkerBuildFn<M, W> =
    fn(WorkerId, Arc<M>, SharedRequests, WorkerConfig, Option<PathBuf>) -> W;

pub struct UnifiedWorkerFactory<M: IterwiseUnifiedModel, W: IterWorker> {
    pub model: Arc<M>,
    pub requests: SharedRequests,
    pub worker_config: WorkerConfig,
    /// Run log dir, handed to each worker for its `cost_log` writer. `Some`
    /// enables per-iteration cost logging; `None` disables it.
    pub log_dir: Option<PathBuf>,
    /// GPU facts threaded down from the deployment (which reads them off the L4
    /// parallel layout): the GPU type every worker runs on, and how many GPUs one
    /// worker (model replica) spans. The factory does not derive these — it only
    /// carries them so L6 can assemble the run's [`GpuInventory`].
    pub gpu_name: String,
    pub gpus_per_worker: u16,
    /// The concrete worker's `new`, supplied by the deployment after it picks the
    /// (arch, worker) pair.
    build_fn: WorkerBuildFn<M, W>,
}

impl<M: IterwiseUnifiedModel, W: IterWorker> UnifiedWorkerFactory<M, W> {
    pub fn new(
        model: Arc<M>,
        requests: SharedRequests,
        worker_config: WorkerConfig,
        log_dir: Option<PathBuf>,
        gpu_name: String,
        gpus_per_worker: u16,
        build_fn: WorkerBuildFn<M, W>,
    ) -> Self {
        Self {
            model,
            requests,
            worker_config,
            log_dir,
            gpu_name,
            gpus_per_worker,
            build_fn,
        }
    }

    pub fn build(&self, idx: u16) -> W {
        (self.build_fn)(
            WorkerId(idx),
            Arc::clone(&self.model),
            std::rc::Rc::clone(&self.requests),
            self.worker_config,
            self.log_dir.clone(),
        )
    }
}
