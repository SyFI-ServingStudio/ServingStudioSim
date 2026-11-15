//! `pp` deployment — pipeline parallelism. One `stage` pool; each replica is one
//! pipeline whose workers are its stages, one GPU each. Stage 0 admits requests
//! and owns the pipeline's KV; later stages pull activations over NVLink and
//! compute their layers.
//!
//! Wired pairs: `glm52_vllm_nvfp4_pp_dsa_moe` and the hybrid
//! `glm53_flash_vllm_fp8_pp_kda_dsa_moe`, each with `pipeline_chunked_prefill`.
//! The arch builds every stage's model once; the stage models are shared across
//! replicas. The head sizes the pipeline's KV from the arch's
//! `pipeline_kv_cache_layout`: the most constrained stage's cache.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, ensure, Context};

use crate::arch::build as arch_build;
use crate::arch::contract::IterwiseUnifiedModel;
use crate::arch::kv_layout::KvCacheLayout;
use crate::arch::IterArchSel;
use crate::common::{Fabric, SharedRequests};
use crate::deployment::config::PpConfig;
use crate::orchestrator::{
    Flow, PlacementPolicy, PpFlow, PpPlacement, PpStagePoolConfig, PP_STAGE_POOL,
};
use crate::timing::kernels::{P2pIntraKernel, P2pIntraKernelConfig};
use crate::timing::PerfApiBridge;
use crate::worker::kv::TierBytes;
use crate::worker::{
    build_pipeline_head_worker, build_pipeline_stage_worker, resolve_prefix_cache_config,
    CostSource, IterWorker, IterWorkerSel, PipelineHeadEvent, PipelineHeadMsg, PipelineLayout,
    PrefixCacheMode, PrefixCachePolicy, WorkerConfig,
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
        let worker_config = worker_config(cfg, &g.worker, g.arch.max_model_len().ok())?;
        let gpu_name = g.gpu.clone();
        let log_dir = Some(cfg.io.log_dir.clone());
        let placement = match pool.placement {
            PlacementPolicy::LeastQueued => PpPlacement::LeastQueued,
            PlacementPolicy::RoundRobin => PpPlacement::RoundRobin,
            PlacementPolicy::LeastWorkAhead => PpPlacement::LeastWorkAhead,
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
                    activation_bytes_per_token: pipeline.activation_bytes_per_token(),
                    tier_bytes: tier_bytes(
                        pipeline
                            .stages()
                            .iter()
                            .map(|stage| stage.kv_cache_layout()),
                        &worker_config,
                    )?,
                };
                let kv_cache_layout = pipeline.pipeline_kv_cache_layout();
                let head_model = Arc::clone(&pipeline.stages()[0]);
                let head_store = std::rc::Rc::clone(&store);
                let head_log_dir = log_dir.clone();
                let head_gpu = gpu_name.clone();
                let head_config = worker_config.clone();
                Ok(assemble_pp_flow(
                    pipeline.stages(),
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
                    move |id, cluster| {
                        build_pipeline_head_worker(
                            id,
                            STAGE_POOL_TAG,
                            Arc::clone(&head_model),
                            layout,
                            &kv_cache_layout,
                            std::rc::Rc::clone(&head_store),
                            head_config.clone(),
                            head_log_dir.clone(),
                            PP_STAGE_POOL,
                            &head_gpu,
                            std::rc::Rc::clone(cluster),
                        )
                    },
                ))
            }
            IterArchSel::Glm53FlashVllmFp8PpKdaDsaMoe {
                pp_size,
                max_model_len,
                routing,
                routing_seed,
                expert_popularity_file,
                token_corpus_file,
                cudagraph_capture_sizes,
                layer_partition,
                ..
            }
            | IterArchSel::Glm53FlashVllmNvfp4PpKdaDsaMoe {
                pp_size,
                max_model_len,
                routing,
                routing_seed,
                expert_popularity_file,
                token_corpus_file,
                cudagraph_capture_sizes,
                layer_partition,
                ..
            } => {
                let pipeline = arch_build::glm53_flash_vllm_fp8_pp_kda_dsa_moe(
                    model_spec,
                    arch_build::glm53_flash_quant(&g.arch),
                    *pp_size,
                    *max_model_len,
                    *routing,
                    *routing_seed,
                    expert_popularity_file.as_deref(),
                    token_corpus_file.as_deref(),
                    cudagraph_capture_sizes,
                    layer_partition,
                    arch_build::glm53_flash_kernel_path(&g.arch)?,
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?;
                let layout = PipelineLayout {
                    depth: pipeline.pp_size(),
                    activation_bytes_per_token: pipeline.activation_bytes_per_token(),
                    tier_bytes: tier_bytes(
                        pipeline
                            .stages()
                            .iter()
                            .map(|stage| stage.kv_cache_layout()),
                        &worker_config,
                    )?,
                };
                let kv_cache_layout = pipeline.pipeline_kv_cache_layout();
                let head_model = Arc::clone(&pipeline.stages()[0]);
                let head_store = std::rc::Rc::clone(&store);
                let head_log_dir = log_dir.clone();
                let head_gpu = gpu_name.clone();
                let head_config = worker_config.clone();
                Ok(assemble_pp_flow(
                    pipeline.stages(),
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
                    move |id, cluster| {
                        build_pipeline_head_worker(
                            id,
                            STAGE_POOL_TAG,
                            Arc::clone(&head_model),
                            layout,
                            &kv_cache_layout,
                            std::rc::Rc::clone(&head_store),
                            head_config.clone(),
                            head_log_dir.clone(),
                            PP_STAGE_POOL,
                            &head_gpu,
                            std::rc::Rc::clone(cluster),
                        )
                    },
                ))
            }
            other => bail!("pp: arch {other:?} has no pipeline-parallel stage model"),
        }
    }
}

fn worker_config(
    cfg: &PpConfig,
    worker: &IterWorkerSel,
    max_model_len: Option<u32>,
) -> anyhow::Result<WorkerConfig> {
    let IterWorkerSel::PipelineChunkedPrefill {
        attn_gpu_memory_gb,
        max_batch_tokens,
        gpu_time_multiplier,
        balance_decode_microbatches,
        prefill_chunk_alignment,
        pending_order,
        external_decode,
        prefix_tiers,
        ..
    } = worker
    else {
        bail!("pp: the stage pool requires worker `pipeline_chunked_prefill`, got {worker:?}");
    };
    // Same default as unified `chunked_prefill`: opportunistic reuse.
    let prefix_cache = resolve_prefix_cache_config(
        "pp",
        PrefixCacheMode::Opportunistic,
        PrefixCachePolicy::Lru,
        None,
        *attn_gpu_memory_gb,
    )?;
    let mut config = WorkerConfig {
        attn_kv_bytes: (*attn_gpu_memory_gb * 1e9) as u64,
        log_output_token_times: cfg.io.log_output_token_times,
        log_stage_transitions: cfg.io.log_stage_transitions,
        kv_log_stride: cfg.io.kv_log_stride,
        gpu_time_multiplier: *gpu_time_multiplier,
        max_batch_tokens: Some(*max_batch_tokens),
        pending_order: *pending_order,
        prefix_cache,
        balance_decode_microbatches: *balance_decode_microbatches,
        prefill_chunk_alignment: *prefill_chunk_alignment,
        external_decode: *external_decode,
        max_model_len,
        ..WorkerConfig::default()
    };
    worker.kv_accounting().apply("pp", &mut config)?;
    worker.apply_chunked_scheduling("pp", &mut config)?;
    prefix_tiers.apply("pp", &mut config)?;
    Ok(config)
}

/// What each GPU stores in and reads from a host tier, from every stage's
/// layout: the first stage with the most KV bytes per token, or with
/// `prefix_tier_balanced_load` the stages' means. A loading area needs a
/// recurrent state in the tiers.
fn tier_bytes(
    stages: impl Iterator<Item = KvCacheLayout>,
    config: &WorkerConfig,
) -> anyhow::Result<TierBytes> {
    let stages: Vec<TierBytes> = stages.map(|layout| TierBytes::of_layout(&layout)).collect();
    let bytes = TierBytes::of_stages(&stages, config.prefix_tier_balanced_load);
    ensure!(
        config.prefix_tier_loading_area == 0.0 || bytes.state > 0,
        "pp: prefix_tier_loading_area needs an arch with recurrent state; this one is full \
         attention"
    );
    Ok(bytes)
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

/// One pipeline flow: `build_head` makes each replica's stage-0 head (its KV
/// store is the arch's choice); every later stage is a follower.
#[allow(clippy::too_many_arguments)]
fn assemble_pp_flow<M, H>(
    stages: &[Arc<M>],
    store: SharedRequests,
    worker_config: WorkerConfig,
    log_dir: Option<PathBuf>,
    gpu_name: String,
    pool_config: PpStagePoolConfig,
    cost: CostSource,
    build_head: impl FnMut(crate::common::WorkerId, &crate::worker::SharedGpuCluster) -> H,
) -> Box<dyn Flow>
where
    M: IterwiseUnifiedModel + 'static,
    H: IterWorker<Msg = PipelineHeadMsg, Event = PipelineHeadEvent> + 'static,
{
    assert_eq!(stages.len(), usize::from(pool_config.depth));
    Box::new(PpFlow::new(
        std::rc::Rc::clone(&store),
        &pool_config,
        cost,
        build_head,
        |id, stage, cluster| {
            build_pipeline_stage_worker(
                id,
                STAGE_POOL_TAG,
                Arc::clone(&stages[usize::from(stage)]),
                std::rc::Rc::clone(&store),
                worker_config.clone(),
                log_dir.clone(),
                PP_STAGE_POOL,
                &gpu_name,
                std::rc::Rc::clone(cluster),
            )
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::kv_layout::KvCachePart;

    #[test]
    fn a_loading_area_needs_an_arch_with_recurrent_state() {
        let full_attention = || KvCacheLayout::new().with_part(KvCachePart::per_token("mla", 545));
        let hybrid = || full_attention().with_part(KvCachePart::per_request("kda", 1_000));
        let staged = WorkerConfig {
            prefix_tier_loading_area: 0.1,
            ..WorkerConfig::default()
        };
        assert!(tier_bytes([full_attention()].into_iter(), &staged).is_err());
        assert!(tier_bytes([full_attention()].into_iter(), &WorkerConfig::default()).is_ok());
        assert_eq!(
            tier_bytes([hybrid()].into_iter(), &staged).unwrap().state,
            1_000
        );
    }
}
