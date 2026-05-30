//! `unified` deployment — one worker type runs the whole model per iteration
//! (co-located attention + FFN), in one homogeneous `main` pool.
//!
//! `build` reads the structured `UnifiedConfig`: a single pool (`main`) with one
//! homogeneous group. The arch is selected by its explicit tag (NO `tp_size`
//! dispatch — provider-first, new-interface-design §4). Wired arms are
//! `llama3_dense` + `barebone`, `llama3_dense_tp` + `barebone`,
//! `llama3_dp_attn_tp_ffn` + `hp_unified`, and `qwen3_moe_dp_attn_ep_ffn` +
//! `hp_unified`. Each arm monomorphizes its concrete model/worker pair and
//! erases to `Box<dyn Flow>` — the single `dyn` point (the cost path is
//! `dyn`-free, L4 §4.1).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, ensure, Context};

use crate::arch::contract::IterwiseUnifiedModel;
use crate::arch::model_cfg::ModelCfg;
use crate::arch::moe_model_cfg::MoeModelCfg;
use crate::arch::{
    llama3_dense, llama3_dense_tp, llama3_dp_attn_tp_ffn, qwen3_moe_dp_attn_ep_ffn, DenseParallel,
    DenseTpParallel, DpAttnTpFfnParallel, IterArchSel, Qwen3MoeParallel,
};
use crate::timing::routing::RoutingDistribution;
use crate::common::{PoolId, SharedRequests};
use crate::deployment::UnifiedConfig;
use crate::orchestrator::common::WorkerBuildFn;
use crate::orchestrator::{
    DpPlacementPolicy, Flow, PlacementPolicy, SimpleDpConfig, SimpleDpFlow, SimpleDpPoolConfig,
    UnifiedWorkerFactory,
};
use crate::timing::PerfApiBridge;
use crate::worker::{BareboneWorker, HpUnifiedWorker, IterWorker, IterWorkerSel, WorkerConfig};

use super::Deployment;

pub struct UnifiedDeployment;

impl Deployment for UnifiedDeployment {
    const NAME: &'static str = "unified";
    type Config = UnifiedConfig;

    fn build(
        cfg: &UnifiedConfig,
        bridge: &PerfApiBridge,
        store: SharedRequests,
    ) -> anyhow::Result<Box<dyn Flow>> {
        // Single homogeneous group only (heterogeneous `groups` is parse-only).
        let pool = &cfg.pools.main;
        ensure!(
            pool.groups.len() == 1,
            "unified: only a single homogeneous group is supported (got {})",
            pool.groups.len()
        );
        let g = &pool.groups[0];

        // Model dims load from JSON (arch-independent, §4); `sim_num_layers` /
        // `num_layers` truncation must apply BEFORE build_configs. Loaded per-arm
        // because dense and MoE archs deserialize different field sets — dense
        // archs read `ModelCfg`, the MoE arch reads `MoeModelCfg`.
        let model_spec = g.arch.model();

        // L5 worker env: `attn_kv_bytes` sizes the worker's KvPool, and
        // `log_output_token_times` controls request_slo detail logging. The
        // worker *type* is matched against the arch in the arms below.
        // chunked_prefill is not wired yet.
        let attn_gpu_memory_gb = match &g.worker {
            IterWorkerSel::Barebone { attn_gpu_memory_gb }
            | IterWorkerSel::HpUnified { attn_gpu_memory_gb } => *attn_gpu_memory_gb,
            IterWorkerSel::ChunkedPrefill { .. } => {
                bail!("unified: chunked_prefill worker not wired yet")
            }
            IterWorkerSel::PdPrefill { .. } | IterWorkerSel::PdDecode { .. } => {
                bail!("unified: pd_prefill / pd_decode workers belong to the `pd` deployment")
            }
        };
        let worker_config = WorkerConfig {
            attn_kv_bytes: (attn_gpu_memory_gb * 1e9) as u64,
            log_output_token_times: cfg.io.log_output_token_times,
            ..WorkerConfig::default()
        };

        let dp_cfg = SimpleDpConfig {
            dp_pool: SimpleDpPoolConfig {
                pool: PoolId(0),
                num_workers: g.replicas,
                placement: placement_into(pool.placement),
            },
        };
        let log_dir: Option<PathBuf> = Some(cfg.io.log_dir.clone());
        let gpu_name = g.gpu.clone();

        // Arch selected by explicit tag; each arm builds its own concrete model
        // type, validates the paired worker tag, and erases via assemble_flow.
        // dense / dense_tp run on the single-group barebone worker; the DP-attn
        // arch runs on the multi-group hp_unified worker (one Batch per DP shard).
        match &g.arch {
            IterArchSel::Llama3Dense { .. } => {
                ensure_barebone(&g.worker)?;
                let model_cfg = load_dense_model_cfg(model_spec)?;
                let parallel = DenseParallel {
                    gpu_name: gpu_name.clone(),
                };
                let resolved = llama3_dense::resolve_configs(&llama3_dense::build_configs(
                    &model_cfg, &parallel,
                ));
                let model = Arc::new(
                    llama3_dense::build("unified".to_string(), resolved, bridge)
                        .context("building Llama3-dense model (often a missing profile.db row)")?,
                );
                Ok(assemble_flow(
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    BareboneWorker::new,
                ))
            }
            IterArchSel::Llama3DenseTp { tp_size, .. } => {
                ensure_barebone(&g.worker)?;
                let model_cfg = load_dense_model_cfg(model_spec)?;
                let parallel = DenseTpParallel {
                    tp_size: *tp_size,
                    gpu_name: gpu_name.clone(),
                };
                let resolved = llama3_dense_tp::resolve_configs(&llama3_dense_tp::build_configs(
                    &model_cfg, &parallel,
                ));
                let model = Arc::new(
                    llama3_dense_tp::build("unified".to_string(), resolved, bridge).context(
                        "building Llama3-dense-TP model (often a missing profile.db row)",
                    )?,
                );
                Ok(assemble_flow(
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    BareboneWorker::new,
                ))
            }
            IterArchSel::Llama3DpAttnTpFfn {
                attn_tp_size,
                ffn_tp_size,
                ..
            } => {
                ensure_hp_unified(&g.worker)?;
                let model_cfg = load_dense_model_cfg(model_spec)?;
                let parallel = DpAttnTpFfnParallel {
                    attn_tp_size: *attn_tp_size,
                    ffn_tp_size: *ffn_tp_size,
                    gpu_name: gpu_name.clone(),
                };
                let resolved = llama3_dp_attn_tp_ffn::resolve_configs(
                    &llama3_dp_attn_tp_ffn::build_configs(&model_cfg, &parallel),
                );
                let model = Arc::new(
                    llama3_dp_attn_tp_ffn::build("unified".to_string(), resolved, bridge).context(
                        "building Llama3 DP-attn TP-ffn model (often a missing profile.db row)",
                    )?,
                );
                Ok(assemble_flow(
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    HpUnifiedWorker::new,
                ))
            }
            IterArchSel::Qwen3MoeDpAttnEpFfn {
                attn_tp_size,
                ep_size,
                hp_size,
                nvl_num_gpu,
                routing_profile,
                ..
            } => {
                ensure_hp_unified(&g.worker)?;
                let model_cfg = load_moe_model_cfg(model_spec)?;
                let routing = resolve_routing(routing_profile.as_slice(), model_cfg.num_experts)?;
                let parallel = Qwen3MoeParallel {
                    attn_tp_size: *attn_tp_size,
                    ep_size: *ep_size,
                    hp_size: *hp_size,
                    nvl_num_gpu: *nvl_num_gpu,
                    gpu_name: gpu_name.clone(),
                };
                let resolved = qwen3_moe_dp_attn_ep_ffn::resolve_configs(
                    &qwen3_moe_dp_attn_ep_ffn::build_configs(&model_cfg, &parallel, &routing),
                );
                let model = Arc::new(
                    qwen3_moe_dp_attn_ep_ffn::build("unified".to_string(), resolved, bridge)
                        .context(
                            "building Qwen3-MoE DP-attn EP-ffn model \
                             (often a missing profile.db row)",
                        )?,
                );
                Ok(assemble_flow(
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    HpUnifiedWorker::new,
                ))
            }
        }
    }
}

/// Load the dense `ModelCfg` from JSON and apply the `sim_num_layers` /
/// `num_layers` override that truncates the layer count BEFORE build_configs.
fn load_dense_model_cfg(model_spec: &crate::arch::ModelSpec) -> anyhow::Result<ModelCfg> {
    let mut model_cfg = ModelCfg::from_json(Path::new(&model_spec.model_config))?;
    if let Some(n) = model_spec.sim_num_layers.or(model_spec.num_layers) {
        model_cfg.num_layers = n;
    }
    Ok(model_cfg)
}

/// Load the MoE `MoeModelCfg` from JSON and apply the layer-count override.
/// Separate loader because MoE configs include `num_experts` /
/// `num_experts_per_tok` / `moe_intermediate_size` that the dense parser does
/// not deserialize.
fn load_moe_model_cfg(model_spec: &crate::arch::ModelSpec) -> anyhow::Result<MoeModelCfg> {
    let mut model_cfg = MoeModelCfg::from_json(Path::new(&model_spec.model_config))?;
    if let Some(n) = model_spec.sim_num_layers.or(model_spec.num_layers) {
        model_cfg.num_layers = n;
    }
    Ok(model_cfg)
}

/// Resolve the MoE routing distribution from the optional selector profile.
/// Empty slice → uniform over `num_experts` (the YAML default); non-empty
/// → must have length `num_experts` (the L2 MoE op needs one ppm slot per
/// expert).
fn resolve_routing(profile: &[f32], num_experts: u32) -> anyhow::Result<RoutingDistribution> {
    if profile.is_empty() {
        return Ok(RoutingDistribution::uniform(num_experts));
    }
    ensure!(
        profile.len() == num_experts as usize,
        "routing_profile has {} weights but model has {} experts",
        profile.len(),
        num_experts,
    );
    Ok(RoutingDistribution::from_profile(profile))
}

/// The dense / dense_tp archs run on the single-group barebone worker.
fn ensure_barebone(worker: &IterWorkerSel) -> anyhow::Result<()> {
    match worker {
        IterWorkerSel::Barebone { .. } => Ok(()),
        other => bail!("unified: this arch requires worker `barebone`, got {other:?}"),
    }
}

/// The DP-attention arch runs on the multi-group hp_unified worker.
fn ensure_hp_unified(worker: &IterWorkerSel) -> anyhow::Result<()> {
    match worker {
        IterWorkerSel::HpUnified { .. } => Ok(()),
        other => {
            bail!("unified: llama3_dp_attn_tp_ffn requires worker `hp_unified`, got {other:?}")
        }
    }
}

/// Wrap a built iter-wise model in a `simple_dp` flow. Generic over the concrete
/// model `M` and worker `W`; `build_fn` is the chosen worker's `new`. The returned
/// `Box<dyn Flow>` is the only `dyn` erasure point.
fn assemble_flow<M: IterwiseUnifiedModel, W: IterWorker + 'static>(
    model: Arc<M>,
    store: SharedRequests,
    worker_config: WorkerConfig,
    log_dir: Option<PathBuf>,
    gpu_name: String,
    dp_cfg: SimpleDpConfig,
    build_fn: WorkerBuildFn<M, W>,
) -> Box<dyn Flow> {
    let factory = UnifiedWorkerFactory::new(
        model,
        store,
        worker_config,
        log_dir,
        gpu_name,
        "main",
        build_fn,
    );
    Box::new(SimpleDpFlow::new(dp_cfg, factory))
}

fn placement_into(p: PlacementPolicy) -> DpPlacementPolicy {
    match p {
        PlacementPolicy::LeastQueued => DpPlacementPolicy::LeastQueued,
        PlacementPolicy::RoundRobin => DpPlacementPolicy::RoundRobin,
    }
}
