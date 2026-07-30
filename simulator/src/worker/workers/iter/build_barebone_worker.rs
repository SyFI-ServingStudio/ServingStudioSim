//! Dense, single-partition whole-iteration worker recipe.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::IterwiseUnifiedModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::worker::admission::{FifoOrder, LocalPrefillDecodeAdmission};
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::kv::FullAttnKv;
use crate::worker::types::WorkerConfig;

use super::iter_batch_worker::{BareboneWorker, IterBatchWorker};
use crate::worker::workers::unified_iter_build_essentials::{
    full_attention_token_capacity, prepare_unified_iter_build_essentials,
};

/// Selects the production barebone composition without changing the L6-facing
/// nine-argument constructor contract.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_barebone_worker<M: IterwiseUnifiedModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> BareboneWorker<M> {
    let kv_capacity = full_attention_token_capacity(model.as_ref(), &config);
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
        kv_capacity,
        1,
    );
    let kv_store = FullAttnKv::new(1, essentials.kv_capacity, essentials.sampler);
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
