//! Production AFD-attention composition recipe.

use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::AttnLayerwiseModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::worker::admission::{FreshRequestSlotAdmission, SessionStartOrder};
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::types::WorkerConfig;

use super::attention_build_essentials::prepare_attention_build_essentials;
use super::slot_attention_worker::{DisaggAttnWorker, SlotAttentionWorker};

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_afd_attention_worker<M: AttnLayerwiseModel>(
    id: WorkerId,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
    cost_log_dir: Option<PathBuf>,
    pool_tag: &'static str,
) -> DisaggAttnWorker<M> {
    let essentials = prepare_attention_build_essentials(
        id,
        model,
        requests,
        config,
        pool,
        gpu_name,
        cluster,
        cost_log_dir,
        pool_tag,
    );
    SlotAttentionWorker::from_components(
        essentials.context,
        essentials.kv_store,
        FreshRequestSlotAdmission::new(SessionStartOrder::new(), ()),
        essentials.execution,
        essentials.cluster,
        essentials.receive_group_id,
    )
}
