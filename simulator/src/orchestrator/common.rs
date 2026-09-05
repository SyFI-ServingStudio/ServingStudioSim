//! Shared L6 vocabulary (`OrchAction` / `PoolEvent`) + the worker-stamping infra
//! (`UnifiedWorkerFactory`) used across deployment impls. Kept out of `mod.rs` so
//! the module root holds only the `Flow` contract and the module wiring. See L6
//! design.md §Organization rule (`common.rs # optional shared vocabulary`).

use std::path::PathBuf;
use std::sync::Arc;

use crate::common::{PoolId, RequestId, SharedRequests, WorkerId};
use crate::worker::{IterWorker, SharedGpuCluster, WorkerConfig};

// Re-export so call sites that still import `crate::orchestrator::{GpuInfo,
// GpuCluster}` keep working — the canonical home is `worker::gpu_cluster`,
// which owns the merged GPU registry / transfer timing oracle.
pub use crate::worker::{GpuCluster, GpuInfo};

/// Deployment-level action returned to L7 each tick. Current flows only surface
/// request completion to L7.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrchAction {
    Complete { req: RequestId },
}

/// Builder signature shared by every iter-wise worker. The factory is handed one
/// family-local `build_*_worker` function, so concrete composition stays out of
/// both L6 and the [`IterWorker`] trait. `pool_tag` is the deployment-set pool name
/// ("main" / "prefill" / "decode" / …) — it disambiguates `cost_log` filenames
/// across pools, since `WorkerId` is per-pool (every pool starts at 0).
///
/// The cluster handle is threaded through last: every worker self-allocates its
/// owned GPU block via `cluster.borrow_mut().allocate(pool.0, id.0,
/// model.gpus_per_replica(), gpu_name)` at construction. PD workers additionally
/// keep the handle for runtime `submit_transfer`; non-PD workers drop it after
/// `allocate`. The deployment fills `gpu_name` from its arch/L4 facts; `pool` is
/// the L6 pool id this worker belongs to.
pub type WorkerBuildFn<M, W> = fn(
    WorkerId,
    &'static str,
    Arc<M>,
    SharedRequests,
    WorkerConfig,
    Option<PathBuf>,
    PoolId,
    &str, // gpu_name
    SharedGpuCluster,
) -> W;

/// The construction surface consumed by an L6 pool.
///
/// L6 owns routing and worker instances; it does not need to know whether a
/// factory carries one iter-wise model, several resident models, or additional
/// runtime policy. Keeping that distinction behind this trait lets specialized
/// L5 workers reuse the normal pool/flow path without teaching L6 their model
/// contract.
pub trait WorkerFactory<W: IterWorker> {
    fn requests(&self) -> &SharedRequests;
    fn build(&self, idx: u16, pool: PoolId, cluster: &SharedGpuCluster) -> W;
}

/// Builds identical unified workers for a DP pool, each sharing the one
/// `SharedRequests` handle and an `Arc` of the model. The worker sizes its own
/// KV partition capacity from `worker_config.attn_kv_bytes`. Specialized
/// constructors use their own factory structs and implement [`WorkerFactory`]
/// directly.
///
/// `M` carries no L4 trait bound. The factory only clones the handle and hands
/// it to `build_fn`, which already states the model contract it needs, and
/// naming one contract here would exclude the others: the speculative recipe's
/// model implements `SpeculativeUnifiedModel` instead of `IterwiseUnifiedModel`.
pub struct UnifiedWorkerFactory<M, W: IterWorker> {
    pub model: Arc<M>,
    pub requests: SharedRequests,
    pub worker_config: WorkerConfig,
    /// Run log dir, handed to each worker for its `cost_log` writer. `Some`
    /// enables per-iteration cost logging; `None` disables it.
    pub log_dir: Option<PathBuf>,
    /// GPU type every worker stamped by this factory runs on. Threaded down from
    /// the deployment (which reads it off the L4 parallel layout). The per-worker
    /// GPU **count** is not a factory fact — it's `model.gpus_per_replica()`,
    /// which the worker reads off the model at allocate time.
    pub gpu_name: String,
    /// Pool name handed to every worker so each `cost_log` file is unique
    /// across pools (`worker_<pool_tag>_<id>.parquet`).
    pub pool_tag: &'static str,
    /// The concrete worker recipe, supplied after deployment selects the
    /// `(arch, worker)` pair.
    build_fn: WorkerBuildFn<M, W>,
}

impl<M, W: IterWorker> UnifiedWorkerFactory<M, W> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Arc<M>,
        requests: SharedRequests,
        worker_config: WorkerConfig,
        log_dir: Option<PathBuf>,
        gpu_name: String,
        pool_tag: &'static str,
        build_fn: WorkerBuildFn<M, W>,
    ) -> Self {
        Self {
            model,
            requests,
            worker_config,
            log_dir,
            gpu_name,
            pool_tag,
            build_fn,
        }
    }

    /// Stamp the `idx`th worker for `pool`, threading the shared `cluster` so the
    /// worker can self-register its GPU block (and, for PD workers, keep the
    /// handle for runtime transfers).
    pub fn build(&self, idx: u16, pool: PoolId, cluster: &SharedGpuCluster) -> W {
        (self.build_fn)(
            WorkerId(idx),
            self.pool_tag,
            Arc::clone(&self.model),
            std::rc::Rc::clone(&self.requests),
            self.worker_config,
            self.log_dir.clone(),
            pool,
            &self.gpu_name,
            std::rc::Rc::clone(cluster),
        )
    }
}

impl<M, W: IterWorker> WorkerFactory<W> for UnifiedWorkerFactory<M, W> {
    fn requests(&self) -> &SharedRequests {
        &self.requests
    }

    fn build(&self, idx: u16, pool: PoolId, cluster: &SharedGpuCluster) -> W {
        UnifiedWorkerFactory::build(self, idx, pool, cluster)
    }
}
