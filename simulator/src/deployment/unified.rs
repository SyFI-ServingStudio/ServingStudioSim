//! `unified` deployment — one worker type runs the whole model per iteration
//! (co-located attention + FFN). First milestone target (Llama3-8B dense, local
//! or TP, single pool).
//!
//! `build` reads the structured `UnifiedConfig`: a single pool (`main`) with one
//! homogeneous group. The arch is selected by its explicit tag (NO `tp_size`
//! dispatch — provider-first, new-interface-design §4); `tp_size` exists only on
//! the `llama3_dense_tp` tag. The two archs are distinct model types `M`, so each
//! match arm monomorphizes `assemble_flow::<M>` and erases to `Box<dyn Flow>` —
//! the single `dyn` point (the cost path is `dyn`-free, L4 §4.1).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, ensure, Context};

use crate::arch::contract::IterwiseUnifiedModel;
use crate::arch::model_cfg::ModelCfg;
use crate::arch::{llama3_dense, llama3_dense_tp, DenseParallel, DenseTpParallel, IterArchSel};
use crate::common::{PoolId, SharedRequests};
use crate::deployment::UnifiedConfig;
use crate::orchestrator::{
    DpPlacementPolicy, Flow, PlacementPolicy, SimpleDpConfig, SimpleDpFlow, SimpleDpPoolConfig,
    UnifiedWorkerFactory,
};
use crate::timing::PerfApiBridge;
use crate::worker::{IterWorkerSel, WorkerConfig};

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
        // `num_layers` truncation must apply BEFORE build_configs.
        let model_spec = g.arch.model();
        let mut model_cfg = ModelCfg::from_json(Path::new(&model_spec.model_config))?;
        if let Some(n) = model_spec.sim_num_layers.or(model_spec.num_layers) {
            model_cfg.num_layers = n;
        }

        // L5 worker env: only `attn_kv_bytes` has a sink today (the barebone
        // worker sizes its own KvPool). chunked_prefill is not wired yet.
        let attn_gpu_memory_gb = match &g.worker {
            IterWorkerSel::Barebone { attn_gpu_memory_gb } => *attn_gpu_memory_gb,
            IterWorkerSel::ChunkedPrefill { .. } => {
                bail!("unified: chunked_prefill worker not wired yet")
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
        // type and erases via assemble_flow.
        match &g.arch {
            IterArchSel::Llama3Dense { .. } => {
                let parallel = DenseParallel {
                    gpu_name: gpu_name.clone(),
                };
                let resolved =
                    llama3_dense::resolve_configs(&llama3_dense::build_configs(&model_cfg, &parallel));
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
                ))
            }
            IterArchSel::Llama3DenseTp { tp_size, .. } => {
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
                ))
            }
            IterArchSel::DeepseekMoe { .. } => bail!("unified: deepseek_moe arch not wired yet"),
        }
    }
}

/// Wrap a built iter-wise model in a `simple_dp` flow. Generic over the concrete
/// model `M`; the returned `Box<dyn Flow>` is the only `dyn` erasure point.
fn assemble_flow<M: IterwiseUnifiedModel>(
    model: Arc<M>,
    store: SharedRequests,
    worker_config: WorkerConfig,
    log_dir: Option<PathBuf>,
    gpu_name: String,
    dp_cfg: SimpleDpConfig,
) -> Box<dyn Flow> {
    let gpus_per_worker = model.gpus_per_replica();
    let factory =
        UnifiedWorkerFactory::new(model, store, worker_config, log_dir, gpu_name, gpus_per_worker);
    Box::new(SimpleDpFlow::new(dp_cfg, factory))
}

fn placement_into(p: PlacementPolicy) -> DpPlacementPolicy {
    match p {
        PlacementPolicy::LeastQueued => DpPlacementPolicy::LeastQueued,
        PlacementPolicy::RoundRobin => DpPlacementPolicy::RoundRobin,
    }
}
