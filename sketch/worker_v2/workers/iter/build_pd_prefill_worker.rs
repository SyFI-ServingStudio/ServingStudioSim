use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::types::WorkerConfig;

use super::super::super::admission::PrefillHandoffAdmission;
use super::super::super::execution::UnifiedIterExecution;
use super::super::super::kv::FullAttnKv;
use super::iter_batch_worker::IterBatchWorker;
use super::unified_iter_build_essentials::prepare_unified_iter_build_essentials;

/// Worker #4 (prefill half) — PD prefill (matrix A "PrefillHandoffAdmission", iter family).
/// SAME `IterBatchWorker` shell + `FullAttnKv` + `UnifiedIterExecution`; the admission is
/// `PrefillHandoffAdmission` (Msg = `PdPrefillMsg`, Event = `PdPrefillEvent`). The shell's
/// generic `A::Msg`/`A::Event` carry the PD-specific traffic (`ReleaseKv` /
/// `PrefillDone`) with no shell change. Registers the KV send comm group (the decode
/// side pulls from it). Paired with the PD decode worker (separate shell).
#[allow(clippy::too_many_arguments)]
pub fn build_pd_prefill_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> IterBatchWorker<FullAttnKv, PrefillHandoffAdmission, UnifiedIterExecution<M>> {
    let communication_group_size = model.num_attn_shards().max(1);
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
    // PD-prefill's only essentials delta: expose the allocated attention shard
    // group as a KV-send endpoint for the decode-side pull.
    let send_group_id = cluster.borrow_mut().register_comm_group(
        essentials.allocation_base,
        communication_group_size,
        pool_tag,
        id.0,
    );
    let kv_store = FullAttnKv::new(
        1,
        essentials.kv_capacity,
        config.admission,
        essentials.sampler,
    );
    let admission = PrefillHandoffAdmission::new(send_group_id);

    IterBatchWorker::from_components(
        essentials.context,
        kv_store,
        admission,
        essentials.execution,
    )
}
