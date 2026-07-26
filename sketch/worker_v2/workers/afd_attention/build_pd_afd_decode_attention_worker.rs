use std::path::PathBuf;
use std::sync::Arc;

use crate::arch::contract::AttnLayerwiseModel;
use crate::common::{PoolId, SharedRequests, WorkerId};
use crate::worker::gpu_cluster::SharedGpuCluster;
use crate::worker::types::WorkerConfig;

use super::super::super::execution::AttentionLayerExecutionAdapter;
use super::super::super::kv::FullAttnKv;
use super::attention_build_essentials::prepare_attention_build_essentials;
use super::pull_slot_attention_worker::PullSlotAttentionWorker;

/// Worker (PD-AFD decode-attn) — the only NEW piece of the three-pool PD-for-AFD
/// disaggregation. Wires `⟨FullAttnKv, —, AttentionLayerExecutionAdapter<M>⟩` + the KV-pull front-end.
/// Registers the recv comm group (the prefill side's held KV lands here). Same 9-arg
/// shape as `build_afd_attention_worker`, so an L6 factory swaps it in by builder alone.
#[allow(clippy::too_many_arguments)]
pub fn build_pd_afd_decode_attention_worker<M: AttnLayerwiseModel>(
    id: WorkerId,
    pool_tag: &'static str,
    model: Arc<M>,
    requests: SharedRequests,
    config: WorkerConfig,
    cost_log_dir: Option<PathBuf>,
    pool: PoolId,
    gpu_name: &str,
    cluster: SharedGpuCluster,
) -> PullSlotAttentionWorker<FullAttnKv, AttentionLayerExecutionAdapter<M>> {
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

    PullSlotAttentionWorker::from_components(
        essentials.context,
        essentials.kv_store,
        essentials.execution,
        essentials.cluster,
        essentials.receive_group_id,
        essentials.num_layers,
        essentials.kv_bytes_per_token,
    )
}
