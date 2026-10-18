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
    PipelineHeadEvent, PipelineHeadMsg, PipelineHybridState, PipelineLayout, PrefillChunkAlignment,
    PrefixCacheMode, PrefixCachePolicy, WorkerConfig,
};
use crate::worker::config::PrefixTierLoading;

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
                    tier_kv_bytes_per_token: tier_kv_bytes_per_token(
                        &g.worker,
                        pipeline.pipeline_kv_bytes_per_token(),
                        pipeline.total_kv_bytes_per_token(),
                        pipeline.pp_size(),
                    ),
                    tier_state_bytes_per_request: 0,
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
                    tier_kv_bytes_per_token: tier_kv_bytes_per_token(
                        &g.worker,
                        pipeline.pipeline_kv_bytes_per_token(),
                        pipeline.total_kv_bytes_per_token(),
                        pipeline.pp_size(),
                    ),
                    tier_state_bytes_per_request: tier_state_bytes_per_request(
                        &g.worker,
                        &pipeline
                            .stages()
                            .iter()
                            .map(|s| {
                                (
                                    s.total_kv_bytes_per_token(),
                                    s.recurrent_state_bytes_per_request(),
                                )
                            })
                            .collect::<Vec<_>>(),
                    ),
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

/// Bytes per token each stage reads from a tier: the most loaded stage's, or
/// with `prefix_tier_balanced_load` the stages' mean (rounded up).
fn tier_kv_bytes_per_token(
    worker: &IterWorkerSel,
    most_loaded: u64,
    total: u64,
    depth: u16,
) -> u64 {
    match worker {
        IterWorkerSel::PipelineChunkedPrefill {
            prefix_tier_balanced_load: true,
            ..
        } => total.div_ceil(u64::from(depth.max(1))),
        _ => most_loaded,
    }
}

/// Recurrent-state bytes each GPU reads from a tier per load, given each
/// stage's `(KV bytes per token, state bytes per request)`: with
/// `prefix_tier_balanced_load` the stages' mean (rounded up), otherwise the
/// state of the stage [`tier_kv_bytes_per_token`] reads for (the first with the
/// most KV bytes per token).
fn tier_state_bytes_per_request(worker: &IterWorkerSel, stages: &[(u64, u64)]) -> u64 {
    match worker {
        IterWorkerSel::PipelineChunkedPrefill {
            prefix_tier_balanced_load: true,
            ..
        } => stages
            .iter()
            .map(|&(_, state)| state)
            .sum::<u64>()
            .div_ceil(stages.len().max(1) as u64),
        _ => stages
            .iter()
            .fold(None::<(u64, u64)>, |best, &(kv, state)| match best {
                Some((best_kv, _)) if best_kv >= kv => best,
                _ => Some((kv, state)),
            })
            .map_or(0, |(_, state)| state),
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
        dram_tier_writes,
        ssd_tier_writes,
        prefix_tier_warm_start,
        prefix_tier_max_read_wait_ms,
        prefix_tier_read_wait_policy,
        prefix_tier_loading_area,
        prefix_tier_loading,
        force_schedule_after_ms,
        external_decode,
        recurrent_state_checkpoints,
        recurrent_state_anchors,
        recurrent_state_anchor_stride,
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
        *dram_tier_writes,
        *ssd_tier_writes,
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
    let load_budget = crate::worker::config::resolve_load_budget(
        "pp",
        *load_budget_low_tokens,
        *load_budget_backlog_lo_tokens,
        *load_budget_backlog_hi_tokens,
        *max_batch_tokens,
    )?;
    ensure!(
        *recurrent_state_checkpoints >= 1,
        "pp: recurrent_state_checkpoints must be >= 1"
    );
    ensure!(
        *recurrent_state_anchors == 0
            || (*recurrent_state_checkpoints == 1 && *recurrent_state_anchor_stride >= 1),
        "pp: recurrent_state_anchors needs recurrent_state_checkpoints: 1 and \
         recurrent_state_anchor_stride >= 1"
    );
    ensure!(
        (0.0..0.5).contains(prefix_tier_loading_area),
        "pp: prefix_tier_loading_area must be in [0, 0.5)"
    );
    ensure!(
        *prefix_tier_loading_area == 0.0 || prefix_tiers.iter().any(Option::is_some),
        "pp: prefix_tier_loading_area needs a prefix tier"
    );
    ensure!(
        *prefix_tier_loading == PrefixTierLoading::Held
            || (prefix_tiers.iter().any(Option::is_some) && *prefix_tier_loading_area == 0.0),
        "pp: prefix_tier_loading: bounded needs a prefix tier and prefix_tier_loading_area: 0 \
         (its areas are derived)"
    );
    crate::worker::config::check_read_wait(
        "pp",
        *prefix_tier_max_read_wait_ms,
        *prefix_tier_read_wait_policy,
    )?;
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
        load_budget,
        srpt: *srpt,
        prefix_tiers,
        prefix_tier_warm_start: *prefix_tier_warm_start,
        prefix_tier_max_read_wait_ms: *prefix_tier_max_read_wait_ms,
        prefix_tier_read_wait_policy: *prefix_tier_read_wait_policy,
        prefix_tier_loading_area: *prefix_tier_loading_area,
        prefix_tier_loading: *prefix_tier_loading,
        force_schedule_after_ms: *force_schedule_after_ms,
        external_decode: *external_decode,
        recurrent_state_checkpoints: *recurrent_state_checkpoints,
        recurrent_state_anchors: *recurrent_state_anchors,
        recurrent_state_anchor_stride: *recurrent_state_anchor_stride,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn pipeline_worker(extra: &str) -> IterWorkerSel {
        serde_yaml::from_str(&format!(
            "type: pipeline_chunked_prefill\nattn_gpu_memory_gb: 80.0\nmax_batch_tokens: 8192\n{extra}"
        ))
        .expect("parse pipeline worker")
    }

    #[test]
    fn tiers_read_the_most_loaded_stage_unless_loads_are_balanced() {
        // PP8 of GLM-5.3-Flash: 11 DSA layers of 545 B, at most 2 per stage.
        let (most_loaded, total) = (2 * 545, 11 * 545);
        assert_eq!(
            tier_kv_bytes_per_token(&pipeline_worker(""), most_loaded, total, 8),
            1090
        );
        assert_eq!(
            tier_kv_bytes_per_token(
                &pipeline_worker("prefix_tier_balanced_load: true\n"),
                most_loaded,
                total,
                8
            ),
            750,
            "5995 B over 8 stages, rounded up"
        );
    }

    #[test]
    fn tiers_read_one_state_per_context_on_the_same_gpu_as_the_kv() {
        // (KV bytes per token, state bytes per request) per stage.
        let stages = [(545, 30), (1090, 20), (1090, 50), (0, 7)];
        assert_eq!(
            tier_state_bytes_per_request(&pipeline_worker(""), &stages),
            20,
            "the first stage with the most KV bytes per token"
        );
        assert_eq!(
            tier_state_bytes_per_request(
                &pipeline_worker("prefix_tier_balanced_load: true\n"),
                &stages
            ),
            27,
            "107 B over 4 stages, rounded up"
        );
        assert_eq!(tier_state_bytes_per_request(&pipeline_worker(""), &[]), 0);
    }
}
