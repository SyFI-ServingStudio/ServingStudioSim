use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::types::WorkerConfig;

use super::super::super::admission::policy::FifoOrder;
use super::super::super::admission::LocalPrefillDecodeAdmission;
use super::super::super::execution::UnifiedIterExecution;
use super::super::super::kv::HybridStateKv;
use super::iter_batch_worker::IterBatchWorker;
use super::unified_iter_build_essentials::prepare_unified_iter_build_essentials;

/// Worker #13 — hybrid-attention dense (F4, iter family). SAME shell + `LocalPrefillDecodeAdmission` +
/// `UnifiedIterExecution` (the mixed-layer cost is the hybrid model's internal business); the KV
/// is `HybridStateKv` (full-attention `Batch` that grows + a fixed recurrent ledger that does
/// not). `recurrent_state_tokens` models the per-request SSM/conv state size; `N`
/// partitions makes it the hybrid-DP worker too.
#[allow(clippy::too_many_arguments)]
pub fn build_hybrid_kv_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
    recurrent_state_tokens: u64,
    num_partitions: usize,
) -> IterBatchWorker<HybridStateKv, LocalPrefillDecodeAdmission<FifoOrder>, UnifiedIterExecution<M>>
{
    let num_partitions = num_partitions.max(1);
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
        num_partitions,
    );

    let kv_store = HybridStateKv::new(
        num_partitions,
        essentials.kv_capacity,
        config.admission,
        essentials.sampler,
        recurrent_state_tokens,
    );
    let admission = LocalPrefillDecodeAdmission::new(
        FifoOrder::new(),
        (),
        config.max_batch_tokens,
        config.balance,
    );

    IterBatchWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        essentials.execution,
    )
}
