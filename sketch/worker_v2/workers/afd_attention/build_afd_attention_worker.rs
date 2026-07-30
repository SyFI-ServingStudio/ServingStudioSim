use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::AttnLayerwiseModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::types::WorkerConfig;

use super::super::super::admission::FreshRequestSlotAdmission;
use super::super::super::execution::AttentionLayerExecutionAdapter;
use super::super::super::kv::FullAttnKv;
use super::attention_build_essentials::prepare_attention_build_essentials;
use super::slot_attention_worker::SlotAttentionWorker;

/// Worker #7 — AFD attn, dense (the old `DisaggAttnWorker`). Wires the tuple
/// `⟨FullAttnKv, FreshRequestSlotAdmission, AttentionLayerExecutionAdapter<M>⟩`; the 9-arg signature matches
/// `DisaggAttnWorker::new` so the L6 factory is untouched.
#[allow(clippy::too_many_arguments)]
pub fn build_afd_attention_worker<M: AttnLayerwiseModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> SlotAttentionWorker<FullAttnKv, FreshRequestSlotAdmission, AttentionLayerExecutionAdapter<M>> {
    let essentials = prepare_attention_build_essentials(
        id,
        pool_tag,
        model,
        requests,
        config,
        cost_log_dir,
        pool,
        gpu_name,
        cluster,
    );

    let admission = FreshRequestSlotAdmission::new();
    SlotAttentionWorker::from_components(
        essentials.context,
        essentials.kv_store,
        admission,
        essentials.execution,
        essentials.cluster,
        essentials.receive_group_id,
        essentials.num_layers,
    )
}
