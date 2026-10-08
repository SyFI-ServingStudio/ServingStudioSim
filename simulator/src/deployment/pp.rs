//! `pp` deployment — pipeline parallelism. One `stage` pool; each replica is one
//! pipeline whose workers are its stages, one GPU each. Stage 0 admits requests
//! and owns the pipeline's KV; later stages pull activations over NVLink and
//! compute their layers.
//!
//! Wired pairs: `glm52_vllm_nvfp4_pp_dsa_moe` and the hybrid
//! `glm53_flash_vllm_fp8_pp_kda_dsa_moe`, each with `pipeline_chunked_prefill`.
//! The arch builds every stage's model once; the stage models are shared across
//! replicas. The hybrid head charges KV and recurrent state against one block
//! pool (`PipelineHybridState`).

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, ensure, Context};

use crate::arch::build as arch_build;
use crate::arch::contract::IterwiseUnifiedModel;
use crate::arch::IterArchSel;
use crate::common::{Fabric, SharedRequests};
use crate::deployment::config::PpConfig;
use crate::orchestrator::{
    Flow, PlacementPolicy, PpFlow, PpPlacement, PpStagePoolConfig, PP_STAGE_POOL,
};
use crate::timing::kernels::{P2pIntraKernel, P2pIntraKernelConfig};
use crate::timing::PerfApiBridge;
use crate::worker::{
    build_hybrid_pipeline_head_worker, build_pipeline_head_worker, build_pipeline_stage_worker,
    resolve_microbatch_sizing, resolve_prefix_cache_config, CostSource, IterWorker, IterWorkerSel,
    PipelineHeadEvent, PipelineHeadMsg, PipelineHybridState, PipelineLayout, PipelineLoadBudget,
    PrefillChunkAlignment, PrefixCacheMode, PrefixCachePolicy, WorkerConfig,
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
                    kv_bytes_per_token: pipeline.pipeline_kv_bytes_per_token(),
                    activation_bytes_per_token: pipeline.activation_bytes_per_token(),
                };
                let head_model = Arc::clone(&pipeline.stages()[0]);
                let head_store = std::rc::Rc::clone(&store);
                let head_log_dir = log_dir.clone();
                let head_gpu = gpu_name.clone();
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
                            std::rc::Rc::clone(&head_store),
                            worker_config,
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
                    kv_bytes_per_token: pipeline.pipeline_kv_bytes_per_token(),
                    activation_bytes_per_token: pipeline.activation_bytes_per_token(),
                };
                let hybrid = PipelineHybridState {
                    block_tokens: pipeline.block_tokens(),
                    state_blocks_per_request: pipeline.state_blocks_per_request(),
                    align_mode: matches!(
                        g.worker,
                        IterWorkerSel::PipelineChunkedPrefill {
                            prefill_chunk_alignment: PrefillChunkAlignment::Checkpoint,
                            ..
                        }
                    ),
                };
                let head_model = Arc::clone(&pipeline.stages()[0]);
                let head_store = std::rc::Rc::clone(&store);
                let head_log_dir = log_dir.clone();
                let head_gpu = gpu_name.clone();
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
                        build_hybrid_pipeline_head_worker(
                            id,
                            STAGE_POOL_TAG,
                            Arc::clone(&head_model),
                            layout,
                            hybrid,
                            std::rc::Rc::clone(&head_store),
                            worker_config,
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

fn worker_config(cfg: &PpConfig, worker: &IterWorkerSel) -> anyhow::Result<WorkerConfig> {
    let IterWorkerSel::PipelineChunkedPrefill {
        attn_gpu_memory_gb,
        max_batch_tokens,
        gpu_time_multiplier,
        balance_decode_microbatches,
        microbatch_split,
        min_microbatch_tokens,
        long_prefill_token_threshold,
        pending_order,
        load_budget_low_tokens,
        load_budget_backlog_lo_tokens,
        load_budget_backlog_hi_tokens,
        srpt,
        dram_tier_gb,
        dram_tier_gb_per_s,
        ssd_tier_gb,
        ssd_tier_gb_per_s,
        external_decode,
        ..
    } = worker
    else {
        bail!("pp: the stage pool requires worker `pipeline_chunked_prefill`, got {worker:?}");
    };
    let prefix_tiers = crate::worker::config::prefix_tier_specs(
        "pp",
        *dram_tier_gb,
        *dram_tier_gb_per_s,
        *ssd_tier_gb,
        *ssd_tier_gb_per_s,
    )?;
    // Same defaults as unified `chunked_prefill`: FIFO and opportunistic reuse.
    let prefix_cache = resolve_prefix_cache_config(
        "pp",
        PrefixCacheMode::Opportunistic,
        PrefixCachePolicy::Lru,
        None,
        *attn_gpu_memory_gb,
    )?;
    let microbatch_sizing =
        resolve_microbatch_sizing(*microbatch_split, *min_microbatch_tokens, *max_batch_tokens)
            .context("pp: pipeline_chunked_prefill")?;
    ensure!(
        *load_budget_low_tokens <= *max_batch_tokens,
        "pp: load_budget_low_tokens ({load_budget_low_tokens}) must be <= max_batch_tokens \
         ({max_batch_tokens})"
    );
    if *load_budget_low_tokens > 0 {
        ensure!(
            load_budget_backlog_lo_tokens < load_budget_backlog_hi_tokens,
            "pp: load_budget_low_tokens needs load_budget_backlog_lo_tokens < \
             load_budget_backlog_hi_tokens, got {load_budget_backlog_lo_tokens} and \
             {load_budget_backlog_hi_tokens}"
        );
    } else {
        ensure!(
            *load_budget_backlog_lo_tokens == 0 && *load_budget_backlog_hi_tokens == 0,
            "pp: load_budget_backlog_*_tokens need load_budget_low_tokens"
        );
    }
    Ok(WorkerConfig {
        attn_kv_bytes: (*attn_gpu_memory_gb * 1e9) as u64,
        log_output_token_times: cfg.io.log_output_token_times,
        log_stage_transitions: cfg.io.log_stage_transitions,
        kv_log_stride: cfg.io.kv_log_stride,
        gpu_time_multiplier: *gpu_time_multiplier,
        max_batch_tokens: Some(*max_batch_tokens),
        pending_order: *pending_order,
        prefix_cache,
        balance_decode_microbatches: *balance_decode_microbatches,
        microbatch_sizing,
        long_prefill_token_threshold: (*long_prefill_token_threshold > 0)
            .then_some(*long_prefill_token_threshold),
        load_budget: (*load_budget_low_tokens > 0).then_some(PipelineLoadBudget {
            low_tokens: *load_budget_low_tokens,
            backlog_lo_tokens: *load_budget_backlog_lo_tokens,
            backlog_hi_tokens: *load_budget_backlog_hi_tokens,
        }),
        srpt: *srpt,
        prefix_tiers,
        external_decode: *external_decode,
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
                worker_config,
                log_dir.clone(),
                PP_STAGE_POOL,
                &gpu_name,
                std::rc::Rc::clone(cluster),
            )
        },
    ))
}
