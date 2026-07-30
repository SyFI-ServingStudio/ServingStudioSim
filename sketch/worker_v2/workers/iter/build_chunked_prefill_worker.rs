use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::worker::config::BatchPolicy;
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::types::WorkerConfig;

use super::super::super::admission::policy::FifoOrder;
use super::super::super::admission::ChunkedPrefillAdmission;
use super::super::super::execution::UnifiedIterExecution;
use super::super::super::kv::FullAttnKv;
use super::iter_batch_worker::IterBatchWorker;
use super::unified_iter_build_essentials::prepare_unified_iter_build_essentials;

/// Chunked prefill (matrix A "ChunkedPrefillAdmission", iter family). Same `IterBatchWorker` shell,
/// same `FullAttnKv` (now also `impl ChunkedPrefillKv`), same `UnifiedIterExecution` (its `build_iteration_input`
/// reads `active_chunk_len`, so it renders chunks unchanged) — ONLY the admission
/// swaps to `ChunkedPrefillAdmission<FifoOrder>`. Proves "per-family ≠ per-worker": a second iter
/// worker reuses the shell + KV + execution and writes just a new admission. `max_batch_tokens`
/// (hard cap) + `batch_policy` come from the `IterWorkerSel::ChunkedPrefillAdmission` selector.
#[allow(clippy::too_many_arguments)]
pub fn build_chunked_prefill_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
    max_batch_tokens: u32,
    batch_policy: BatchPolicy,
) -> IterBatchWorker<FullAttnKv, ChunkedPrefillAdmission<FifoOrder>, UnifiedIterExecution<M>> {
    let essentials = prepare_unified_iter_build_essentials(
        id,
        pool_tag,
        model,
        requests,
        &config,
        cost_log_dir,
        pool,
        gpu_name,
        &cluster,
        1,
    );
    let kv_store = FullAttnKv::new(
        1,
        essentials.kv_capacity,
        config.admission,
        essentials.sampler,
    );
    let admission =
        ChunkedPrefillAdmission::new(FifoOrder::new(), (), max_batch_tokens, batch_policy);

    IterBatchWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        essentials.execution,
    )
}
