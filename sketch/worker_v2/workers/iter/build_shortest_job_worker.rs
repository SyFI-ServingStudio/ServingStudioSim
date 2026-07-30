use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::types::WorkerConfig;

use super::super::super::admission::policy::ShortestJobFirst;
use super::super::super::admission::LocalPrefillDecodeAdmission;
use super::super::super::execution::UnifiedIterExecution;
use super::super::super::kv::FullAttnKv;
use super::iter_batch_worker::IterBatchWorker;
use super::unified_iter_build_essentials::prepare_unified_iter_build_essentials;

/// Worker #11 — latency-scheduled dense (matrix AP class, iter family). SAME shell +
/// KV + execution + `LocalPrefillDecodeAdmission` lifecycle as barebone; the ONLY change is the policy
/// type param `FifoOrder → ShortestJobFirst` — proving the selection sub-axis is a pure swap
/// (the lifecycle, KV, and shell never mention which policy is in play).
#[allow(clippy::too_many_arguments)]
pub fn build_shortest_job_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> IterBatchWorker<
    FullAttnKv,
    LocalPrefillDecodeAdmission<ShortestJobFirst>,
    UnifiedIterExecution<M>,
> {
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
    let admission = LocalPrefillDecodeAdmission::new(
        ShortestJobFirst::new(),
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
