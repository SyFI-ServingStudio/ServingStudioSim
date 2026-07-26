use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::types::WorkerConfig;

use super::super::super::admission::policy::FifoOrder;
use super::super::super::admission::LocalPrefillDecodeAdmission;
use super::super::super::execution::UnifiedIterExecution;
use super::super::super::kv::FullAttnKv;
use super::iter_batch_worker::IterBatchWorker;
use super::unified_iter_build_essentials::prepare_unified_iter_build_essentials;

/// Worker #3 — HP/DP unified (matrix "+HP groups", iter family). SAME `IterBatchWorker`
/// shell + `LocalPrefillDecodeAdmission` admission + `UnifiedIterExecution` as barebone — the ONLY
/// differences are `num_partitions = N` (one `FullAttnKv` `Batch` per DP shard) and
/// `LoadBalance::RoundRobin` placement. The real crate keeps `HpUnifiedWorker` as a
/// separate ~460-line file; here barebone is literally the `N = 1` instantiation of
/// this same composition. `N = model.num_attn_dp_groups()`.
#[allow(clippy::too_many_arguments)]
pub fn build_hp_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> IterBatchWorker<FullAttnKv, LocalPrefillDecodeAdmission<FifoOrder>, UnifiedIterExecution<M>> {
    let num_groups = model.num_attn_dp_groups().max(1) as usize;
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
        num_groups,
    );
    let kv_store = FullAttnKv::new(
        num_groups,
        essentials.kv_capacity,
        config.admission,
        essentials.sampler,
    );

    // RoundRobin across the N shards (barebone uses Single); comes from the selector.
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
