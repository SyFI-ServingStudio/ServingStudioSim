//! Shared L6 vocabulary (`OrchAction` / `PoolEvent`) + the worker-stamping infra
//! (`UnifiedWorkerFactory`) used across deployment impls. Kept out of `mod.rs` so
//! the module root holds only the `Flow` contract and the module wiring. See L6
//! design.md §Organization rule (`common.rs # optional shared vocabulary`).

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, RequestId, SharedRequests, WorkerId};
use crate::worker::{BareboneWorker, WorkerConfig};

/// Deployment-level action returned to L7 each tick. Barebone only completes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrchAction {
    Complete { req: RequestId },
}

/// Pool-level event (L6a → L6b). One per worker-level completion in simple_dp.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PoolEvent {
    RequestComplete {
        pool: PoolId,
        worker: WorkerId,
        req: RequestId,
    },
}

/// Builds identical unified workers for a DP pool, each sharing the one
/// `SharedRequests` handle and an `Arc` of the model. The worker sizes its own
/// `KvPool` from `worker_config.attn_kv_bytes`. (L7 will generalize this into a
/// trait; for now a concrete generic struct is enough.)
pub struct UnifiedWorkerFactory<M: IterwiseUnifiedModel> {
    pub model: Arc<M>,
    pub requests: SharedRequests,
    pub worker_config: WorkerConfig,
    /// Run log dir, handed to each worker for its `cost_log` writer (only used
    /// when `worker_config.cost_log` is set). `None` disables cost logging.
    pub log_dir: Option<PathBuf>,
}

impl<M: IterwiseUnifiedModel> UnifiedWorkerFactory<M> {
    pub fn new(
        model: Arc<M>,
        requests: SharedRequests,
        worker_config: WorkerConfig,
        log_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            model,
            requests,
            worker_config,
            log_dir,
        }
    }

    pub fn build(&self, idx: u16) -> BareboneWorker<M> {
        let cost_log_dir = if self.worker_config.cost_log {
            self.log_dir.clone()
        } else {
            None
        };
        BareboneWorker::new(
            WorkerId(idx),
            Arc::clone(&self.model),
            std::rc::Rc::clone(&self.requests),
            self.worker_config,
            cost_log_dir,
        )
    }
}
