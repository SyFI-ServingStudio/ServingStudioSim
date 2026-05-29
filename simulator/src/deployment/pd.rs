//! `pd` deployment — prefill/decode disaggregation. Two pools: a prefill pool of
//! `PdPrefillWorker`s and a decode pool of `PdDecodeWorker`s, wired into a
//! [`PdFlow`]. A prefill worker prefills then hands off (`PrefillDone`); the flow
//! resolves physical GPU endpoints, builds a `TransferPlan`, and the decode pool
//! pulls the KV across the shared `GpuCluster` before decoding to completion.
//!
//! The KV transfer cost comes from the `p2p_inter` L1 kernel (per-link bandwidth
//! profiled vs message size); `build_transfer_cost` wires it. Wired pairings are
//! `llama3_dense_tp` → `llama3_dense_tp` and `llama3_dense_tp` →
//! `llama3_dp_attn_tp_ffn`; other pairings bail until a concrete experiment
//! needs them.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, ensure, Context};

use crate::arch::contract::IterwiseUnifiedModel;
use crate::arch::model_cfg::ModelCfg;
use crate::arch::{
    llama3_dense_tp, llama3_dp_attn_tp_ffn, DenseTpParallel, DpAttnTpFfnParallel, IterArchSel,
};
use crate::common::{Fabric, SharedRequests};
use crate::deployment::config::PdConfig;
use crate::orchestrator::common::WorkerBuildFn;
use crate::orchestrator::config::{GroupSpec, PoolSpec};
use crate::orchestrator::{
    DpPlacementPolicy, Flow, PdFlow, PlacementPolicy, SimpleDpPoolConfig, UnifiedWorkerFactory,
    PD_DECODE_POOL, PD_PREFILL_POOL,
};
use crate::timing::bridge::DType;
use crate::timing::kernels::{P2pInterKernel, P2pInterKernelConfig};
use crate::timing::PerfApiBridge;
use crate::worker::{CostSource, IterWorkerSel, PdDecodeWorker, PdPrefillWorker, WorkerConfig};

use super::Deployment;

pub struct PdDeployment;

impl Deployment for PdDeployment {
    const NAME: &'static str = "pd";
    type Config = PdConfig;

    fn build(
        cfg: &PdConfig,
        bridge: &PerfApiBridge,
        store: SharedRequests,
    ) -> anyhow::Result<Box<dyn Flow>> {
        let pg = single_group(&cfg.pools.prefill, "prefill")?;
        let dg = single_group(&cfg.pools.decode, "decode")?;

        ensure_pd_prefill(&pg.worker)?;
        ensure_pd_decode(&dg.worker)?;

        // Prefill and decode must point at the same model config path for the
        // handed-off KV to be meaningful. The arch match below decides which
        // parallel layouts are supported; it is the only cross-arch gate here.
        ensure!(
            pg.arch.model().model_config == dg.arch.model().model_config,
            "pd: prefill and decode pools must share a model_config \
             (prefill={:?}, decode={:?})",
            pg.arch.model().model_config,
            dg.arch.model().model_config,
        );

        let prefill_cfg = pool_cfg(PD_PREFILL_POOL, pg.replicas, cfg.pools.prefill.placement);
        let decode_cfg = pool_cfg(PD_DECODE_POOL, dg.replicas, cfg.pools.decode.placement);

        let prefill_wc = worker_config(&pg.worker, &cfg.io.log_dir, cfg.io.log_output_token_times);
        let decode_wc = worker_config(&dg.worker, &cfg.io.log_dir, cfg.io.log_output_token_times);
        let log_dir: Option<PathBuf> = Some(cfg.io.log_dir.clone());

        // The KV-transfer cost source: the profiled inter-node p2p curve, keyed
        // on the decode GPU (receiver). Built once here (the bridge lives at L7)
        // and handed to the flow's shared `GpuCluster`.
        let cost = build_transfer_cost(&dg.gpu, bridge)?;

        // Only the pairs actually exercised are wired (add a new arm when a new
        // pairing is needed). Each arm builds both pools' concrete models (own
        // dims/TP) then assembles the flow. Today: tp→tp and tp→dp_attn.
        match (&pg.arch, &dg.arch) {
            (IterArchSel::Llama3DenseTp { .. }, IterArchSel::Llama3DenseTp { .. }) => {
                let prefill_model = build_dense_tp(pg, bridge)?;
                let decode_model = build_dense_tp(dg, bridge)?;
                Ok(assemble_pd_flow(
                    prefill_model,
                    decode_model,
                    store,
                    prefill_wc,
                    decode_wc,
                    log_dir,
                    pg.gpu.clone(),
                    dg.gpu.clone(),
                    prefill_cfg,
                    decode_cfg,
                    cost,
                ))
            }
            // Cross-arch PD: TP prefill hands off to a DP-attention decode. Same
            // llama3 weights, only the parallel layout differs (the KV is logically
            // one tensor, re-sharded across the handoff), so it is a valid pair.
            (IterArchSel::Llama3DenseTp { .. }, IterArchSel::Llama3DpAttnTpFfn { .. }) => {
                let prefill_model = build_dense_tp(pg, bridge)?;
                let decode_model = build_dp_attn_tp_ffn(dg, bridge)?;
                Ok(assemble_pd_flow(
                    prefill_model,
                    decode_model,
                    store,
                    prefill_wc,
                    decode_wc,
                    log_dir,
                    pg.gpu.clone(),
                    dg.gpu.clone(),
                    prefill_cfg,
                    decode_cfg,
                    cost,
                ))
            }
            (p, d) => bail!(
                "pd: unsupported prefill/decode arch pairing \
                 (got prefill={p:?}, decode={d:?}); wired pairs: \
                 dense_tp→dense_tp, dense_tp→dp_attn_tp_ffn"
            ),
        }
    }
}

// ── per-pool helpers ───────────────────────────────────────────────────────────

fn single_group<'a>(
    pool: &'a PoolSpec<IterArchSel, IterWorkerSel>,
    role: &str,
) -> anyhow::Result<&'a GroupSpec<IterArchSel, IterWorkerSel>> {
    ensure!(
        pool.groups.len() == 1,
        "pd: {role} pool supports a single homogeneous group (got {})",
        pool.groups.len()
    );
    Ok(&pool.groups[0])
}

fn ensure_pd_prefill(worker: &IterWorkerSel) -> anyhow::Result<()> {
    match worker {
        IterWorkerSel::PdPrefill { .. } => Ok(()),
        other => bail!("pd: prefill pool requires worker `pd_prefill`, got {other:?}"),
    }
}

fn ensure_pd_decode(worker: &IterWorkerSel) -> anyhow::Result<()> {
    match worker {
        IterWorkerSel::PdDecode { .. } => Ok(()),
        other => bail!("pd: decode pool requires worker `pd_decode`, got {other:?}"),
    }
}

fn worker_config(
    worker: &IterWorkerSel,
    _log_dir: &Path,
    log_output_token_times: bool,
) -> WorkerConfig {
    let attn_gpu_memory_gb = match worker {
        IterWorkerSel::PdPrefill { attn_gpu_memory_gb }
        | IterWorkerSel::PdDecode { attn_gpu_memory_gb } => *attn_gpu_memory_gb,
        // ensure_* gates the worker tag before this is reached.
        _ => 80.0,
    };
    WorkerConfig {
        attn_kv_bytes: (attn_gpu_memory_gb * 1e9) as u64,
        log_output_token_times,
        ..WorkerConfig::default()
    }
}

fn pool_cfg(
    pool: crate::common::PoolId,
    replicas: u16,
    placement: PlacementPolicy,
) -> SimpleDpPoolConfig {
    SimpleDpPoolConfig {
        pool,
        num_workers: replicas,
        placement: placement_into(placement),
    }
}

fn load_model_cfg(g: &GroupSpec<IterArchSel, IterWorkerSel>) -> anyhow::Result<ModelCfg> {
    let model_spec = g.arch.model();
    let mut model_cfg = ModelCfg::from_json(Path::new(&model_spec.model_config))?;
    if let Some(n) = model_spec.sim_num_layers.or(model_spec.num_layers) {
        model_cfg.num_layers = n;
    }
    Ok(model_cfg)
}

fn build_dense_tp(
    g: &GroupSpec<IterArchSel, IterWorkerSel>,
    bridge: &PerfApiBridge,
) -> anyhow::Result<Arc<impl IterwiseUnifiedModel>> {
    let model_cfg = load_model_cfg(g)?;
    let IterArchSel::Llama3DenseTp { tp_size, .. } = &g.arch else {
        unreachable!("build_dense_tp called on non-dense_tp arch");
    };
    let parallel = DenseTpParallel {
        tp_size: *tp_size,
        gpu_name: g.gpu.clone(),
    };
    let resolved =
        llama3_dense_tp::resolve_configs(&llama3_dense_tp::build_configs(&model_cfg, &parallel));
    Ok(Arc::new(
        llama3_dense_tp::build("pd".to_string(), resolved, bridge)
            .context("building Llama3-dense-TP model (often a missing profile.db row)")?,
    ))
}

fn build_dp_attn_tp_ffn(
    g: &GroupSpec<IterArchSel, IterWorkerSel>,
    bridge: &PerfApiBridge,
) -> anyhow::Result<Arc<impl IterwiseUnifiedModel>> {
    let model_cfg = load_model_cfg(g)?;
    let IterArchSel::Llama3DpAttnTpFfn {
        attn_tp_size,
        ffn_tp_size,
        ..
    } = &g.arch
    else {
        unreachable!("build_dp_attn_tp_ffn called on non-dp-attn arch");
    };
    let parallel = DpAttnTpFfnParallel {
        attn_tp_size: *attn_tp_size,
        ffn_tp_size: *ffn_tp_size,
        gpu_name: g.gpu.clone(),
    };
    let resolved = llama3_dp_attn_tp_ffn::resolve_configs(&llama3_dp_attn_tp_ffn::build_configs(
        &model_cfg, &parallel,
    ));
    Ok(Arc::new(
        llama3_dp_attn_tp_ffn::build("pd".to_string(), resolved, bridge)
            .context("building Llama3 DP-attn TP-ffn model (often a missing profile.db row)")?,
    ))
}

/// Assemble a [`PdFlow`] from two built models. Generic over each pool's concrete
/// model type; the prefill pool stamps `PdPrefillWorker`, the decode pool
/// `PdDecodeWorker`. The returned `Box<dyn Flow>` is the only `dyn` erasure point.
#[allow(clippy::too_many_arguments)]
fn assemble_pd_flow<MP, MD>(
    prefill_model: Arc<MP>,
    decode_model: Arc<MD>,
    store: SharedRequests,
    prefill_wc: WorkerConfig,
    decode_wc: WorkerConfig,
    log_dir: Option<PathBuf>,
    prefill_gpu_name: String,
    decode_gpu_name: String,
    prefill_cfg: SimpleDpPoolConfig,
    decode_cfg: SimpleDpPoolConfig,
    cost: CostSource,
) -> Box<dyn Flow>
where
    MP: IterwiseUnifiedModel + 'static,
    MD: IterwiseUnifiedModel + 'static,
{
    let prefill_factory: UnifiedWorkerFactory<MP, PdPrefillWorker<MP>> = UnifiedWorkerFactory::new(
        prefill_model,
        std::rc::Rc::clone(&store),
        prefill_wc,
        log_dir.clone(),
        prefill_gpu_name,
        "prefill",
        PdPrefillWorker::<MP>::new as WorkerBuildFn<MP, PdPrefillWorker<MP>>,
    );
    let decode_factory: UnifiedWorkerFactory<MD, PdDecodeWorker<MD>> = UnifiedWorkerFactory::new(
        decode_model,
        store,
        decode_wc,
        log_dir,
        decode_gpu_name,
        "decode",
        PdDecodeWorker::<MD>::new as WorkerBuildFn<MD, PdDecodeWorker<MD>>,
    );
    Box::new(PdFlow::new(
        &prefill_cfg,
        &prefill_factory,
        &decode_cfg,
        &decode_factory,
        cost,
    ))
}

/// Build the PD KV-transfer cost source: the profiled `p2p_inter` curve for the
/// receiver GPU. v1 assumes a cross-node (Infiniband) handoff over the `nccl`
/// backend and a bf16 element dtype — the curve is keyed by message-size bytes,
/// so dtype only selects the profiled table. Fabric/dtype become config when PD
/// placement grows fabric awareness.
fn build_transfer_cost(gpu_name: &str, bridge: &PerfApiBridge) -> anyhow::Result<CostSource> {
    let kernel = P2pInterKernel::build(
        "pd_kv_transfer".to_string(),
        P2pInterKernelConfig {
            backends: vec!["nccl"],
            gpu_name: gpu_name.to_string(),
            fabric: Fabric::Infiniband,
            dtype: DType::Bf16,
        },
        bridge,
    )
    .with_context(|| format!("building p2p_inter kernel for PD KV transfer on {gpu_name}"))?;
    Ok(CostSource::Kernel(kernel))
}

fn placement_into(p: PlacementPolicy) -> DpPlacementPolicy {
    match p {
        PlacementPolicy::LeastQueued => DpPlacementPolicy::LeastQueued,
        PlacementPolicy::RoundRobin => DpPlacementPolicy::RoundRobin,
    }
}
