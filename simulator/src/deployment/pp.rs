//! `pp` deployment — pipeline parallelism. One `stage` pool; each replica is one
//! pipeline whose workers are its stages, one GPU each. Stage 0 admits requests
//! and owns the pipeline's KV; later stages pull activations over NVLink and
//! compute their layers.
//!
//! The only wired pair is `glm52_vllm_nvfp4_pp_dsa_moe` + `pipeline_chunked_prefill`.
//! The arch builds every stage's model once; the stage models are shared across
//! replicas.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, ensure, Context};

use crate::arch::build as arch_build;
use crate::arch::contract::IterwiseUnifiedModel;
use crate::arch::IterArchSel;
use crate::common::{Fabric, SharedRequests};
use crate::deployment::config::PpConfig;
use crate::orchestrator::{
    DpPlacementPolicy, Flow, PlacementPolicy, PpFlow, PpStagePoolConfig, PP_STAGE_POOL,
};
use crate::timing::kernels::{P2pIntraKernel, P2pIntraKernelConfig};
use crate::timing::PerfApiBridge;
use crate::worker::{
    build_pipeline_head_worker, build_pipeline_stage_worker, resolve_prefix_cache_config,
    CostSource, IterWorkerSel, PendingOrderKind, PipelineLayout, PrefixCacheMode,
    PrefixCachePolicy, WorkerConfig,
};

use super::Deployment;

pub struct PpDeployment;

/// Every stage's dotted-leaf prefix (e.g. `pp.embedding`).
const MODEL_NAME: &str = "pp";
/// Pool tag on every stage's GPU and cost log.
const STAGE_POOL_TAG: &str = "stage";

impl Deployment for PpDeployment {
    const NAME: &'static str = "pp";
    type Config = PpConfig;

    fn build(
        cfg: &PpConfig,
        bridge: &PerfApiBridge,
        store: SharedRequests,
    ) -> anyhow::Result<Box<dyn Flow>> {
        let pool = &cfg.pools.stage;
        ensure!(
            pool.groups.len() == 1,
            "pp: only a single homogeneous group is supported (got {})",
            pool.groups.len()
        );
        let g = &pool.groups[0];
        let worker_config = worker_config(cfg, &g.worker)?;
        let gpu_name = g.gpu.clone();
        let log_dir = Some(cfg.io.log_dir.clone());
        let placement = match pool.placement {
            PlacementPolicy::LeastQueued => DpPlacementPolicy::LeastQueued,
            PlacementPolicy::RoundRobin => DpPlacementPolicy::RoundRobin,
        };
        let model_spec = g.arch.model();
        let _scope =
            bridge.with_backend_overrides(STAGE_POOL_TAG, cfg.backends.get(STAGE_POOL_TAG));

        match &g.arch {
            IterArchSel::Glm52VllmNvfp4PpDsaMoe {
                pp_size,
                max_model_len,
                routing,
                routing_seed,
                expert_popularity_file,
                token_corpus_file,
                ..
            } => {
                let pipeline = arch_build::glm52_vllm_nvfp4_pp_dsa_moe(
                    model_spec,
                    *pp_size,
                    *max_model_len,
                    *routing,
                    *routing_seed,
                    expert_popularity_file.as_deref(),
                    token_corpus_file.as_deref(),
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?;
                let layout = PipelineLayout {
                    depth: pipeline.pp_size(),
                    kv_bytes_per_token: pipeline.pipeline_kv_bytes_per_token(),
                    activation_bytes_per_token: pipeline.activation_bytes_per_token(),
                };
                Ok(assemble_pp_flow(
                    pipeline.stages(),
                    layout,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name.clone(),
                    PpStagePoolConfig {
                        replicas: g.replicas,
                        depth: layout.depth,
                        placement,
                    },
                    build_activation_cost(&gpu_name, bridge)?,
                ))
            }
            other => bail!("pp: arch {other:?} has no pipeline-parallel stage model"),
        }
    }
}

fn worker_config(cfg: &PpConfig, worker: &IterWorkerSel) -> anyhow::Result<WorkerConfig> {
    let IterWorkerSel::PipelineChunkedPrefill {
        attn_gpu_memory_gb,
        max_batch_tokens,
        gpu_time_multiplier,
        balance_decode_microbatches,
    } = worker
    else {
        bail!("pp: the stage pool requires worker `pipeline_chunked_prefill`, got {worker:?}");
    };
    // Same defaults as unified `chunked_prefill`: FIFO and opportunistic reuse.
    let prefix_cache = resolve_prefix_cache_config(
        "pp",
        PrefixCacheMode::Opportunistic,
        PrefixCachePolicy::Lru,
        None,
        *attn_gpu_memory_gb,
    )?;
    Ok(WorkerConfig {
        attn_kv_bytes: (*attn_gpu_memory_gb * 1e9) as u64,
        log_output_token_times: cfg.io.log_output_token_times,
        log_stage_transitions: cfg.io.log_stage_transitions,
        kv_log_stride: cfg.io.kv_log_stride,
        gpu_time_multiplier: *gpu_time_multiplier,
        max_batch_tokens: Some(*max_batch_tokens),
        pending_order: PendingOrderKind::Fifo,
        prefix_cache,
        balance_decode_microbatches: *balance_decode_microbatches,
        ..WorkerConfig::default()
    })
}

/// Stages of one pipeline share an NVLink domain, so activations move on the
/// profiled `p2p_intra` curve.
fn build_activation_cost(gpu_name: &str, bridge: &PerfApiBridge) -> anyhow::Result<CostSource> {
    let kernel = P2pIntraKernel::build(
        "pp_activation_transfer".to_string(),
        P2pIntraKernelConfig {
            backends: vec!["nccl"],
            gpu_name: gpu_name.to_string(),
            fabric: Fabric::Nvlink,
        },
        bridge,
    )
    .with_context(|| format!("building p2p_intra kernel for PP activations on {gpu_name}"))?;
    Ok(CostSource::IntraKernel(kernel))
}

#[allow(clippy::too_many_arguments)]
fn assemble_pp_flow<M: IterwiseUnifiedModel>(
    stages: &[Arc<M>],
    layout: PipelineLayout,
    store: SharedRequests,
    worker_config: WorkerConfig,
    log_dir: Option<PathBuf>,
    gpu_name: String,
    pool_config: PpStagePoolConfig,
    cost: CostSource,
) -> Box<dyn Flow> {
    assert_eq!(stages.len(), usize::from(layout.depth));
    let head_model = Arc::clone(&stages[0]);
    Box::new(PpFlow::new(
        std::rc::Rc::clone(&store),
        &pool_config,
        cost,
        |id, cluster| {
            build_pipeline_head_worker(
                id,
                STAGE_POOL_TAG,
                Arc::clone(&head_model),
                layout,
                std::rc::Rc::clone(&store),
                worker_config,
                log_dir.clone(),
                PP_STAGE_POOL,
                &gpu_name,
                std::rc::Rc::clone(cluster),
            )
        },
        |id, stage, cluster| {
            build_pipeline_stage_worker(
                id,
                STAGE_POOL_TAG,
                Arc::clone(&stages[usize::from(stage)]),
                std::rc::Rc::clone(&store),
                worker_config,
                log_dir.clone(),
                PP_STAGE_POOL,
                &gpu_name,
                std::rc::Rc::clone(cluster),
            )
        },
    ))
}
