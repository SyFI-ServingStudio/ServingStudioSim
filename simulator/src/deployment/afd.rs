//! `afd` deployment — attention-FFN disaggregation. Two pools: a data-parallel
//! pool of [`DisaggAttnWorker`](crate::worker::DisaggAttnWorker)s (one DP shard
//! each) and a single aggregated [`DisaggFfnWorker`](crate::worker::DisaggFfnWorker)
//! replica, wired into an [`AfdFlow`]. The attn pool's controller aggregates every
//! shard's same-slot output into one big ffn batch (the defining AFD mechanism)
//! and holds the per-layer flush barrier that keeps the shards in lockstep.
//!
//! The attn↔ffn QKV transfer cost reuses the profiled `p2p_inter` curve (same as
//! PD's KV handoff); `build_transfer_cost` wires it, keyed on the ffn (aggregation
//! receiver) GPU. v1 uses one curve for both directions — a faithful per-direction
//! split is a later calibration item. Wired pairing is `qwen3_attn` → `qwen3_ffn_moe`;
//! the two sides' `attn_tp_size` may differ (they shard different work and the
//! handoff is TP-agnostic — see the `assemble` note); other pairings bail until an
//! experiment needs them.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{bail, ensure, Context};

use crate::arch::build as arch_build;
use crate::arch::{AttnArchSel, FfnArchSel};
use crate::common::{Fabric, SharedRequests};
use crate::deployment::config::AfdConfig;
use crate::orchestrator::config::{GroupSpec, PoolSpec};
use crate::orchestrator::{
    AfdAttnPoolController, AfdFfnPoolController, AfdFlow, Flow, AFD_ATTN_POOL, AFD_FFN_POOL,
};
use crate::timing::bridge::DType;
use crate::timing::kernels::{P2pInterKernel, P2pInterKernelConfig};
use crate::timing::PerfApiBridge;
use crate::worker::{
    AttnWorkerSel, CostSource, FfnWorkerSel, GpuCluster, SharedGpuCluster, WorkerConfig,
};

use super::Deployment;

pub struct AfdDeployment;

/// Both pools' models share this dotted-leaf prefix (e.g. `afd.embedding`).
const MODEL_NAME: &str = "afd";

impl Deployment for AfdDeployment {
    const NAME: &'static str = "afd";
    type Config = AfdConfig;

    fn build(
        cfg: &AfdConfig,
        bridge: &PerfApiBridge,
        store: SharedRequests,
    ) -> anyhow::Result<Box<dyn Flow>> {
        let ag = single_group(&cfg.pools.attn, "attn")?;
        let fg = single_group(&cfg.pools.ffn, "ffn")?;

        ensure_disagg_attn(&ag.worker)?;
        ensure_disagg_ffn(&fg.worker)?;

        // Topology: ≥1 attn DP shard and ≥1 ffn replica. Multiple ffn replicas are
        // data-parallel: the attn controller's aggregated per-layer `FfnTask`s
        // round-robin across them (see `AfdFfnPoolController`), so concurrent slots'
        // layers run on different replicas. Each replica is itself a full EP-`ep_size`
        // group. NOTE: the round-robin cursor is slot-agnostic, so it can break a
        // worker's per-slot double-buffer pull/compute locality — the overlap model is
        // optimistic under scatter until the route is made slot-affine / least-queued.
        ensure!(ag.replicas >= 1, "afd: attn pool needs ≥1 replica (got {})", ag.replicas);
        ensure!(fg.replicas >= 1, "afd: ffn pool needs ≥1 replica (got {})", fg.replicas);
        // Attn and ffn must share the same model_config — the QKV they exchange is
        // one logical tensor split across the attn/ffn boundary.
        ensure!(
            ag.arch.model().model_config == fg.arch.model().model_config,
            "afd: attn and ffn pools must share a model_config (attn={:?}, ffn={:?})",
            ag.arch.model().model_config,
            fg.arch.model().model_config,
        );

        let attn_wc = worker_config(attn_gpu_memory_gb(&ag.worker), cfg.io.log_output_token_times, cfg.io.kv_log_stride);
        let ffn_wc = worker_config(80.0, cfg.io.log_output_token_times, cfg.io.kv_log_stride); // ffn has no KV

        // attn↔ffn transfer cost: profiled p2p curve, keyed on the ffn GPU (the
        // aggregation receiver). Built once here (the bridge lives at L7).
        let cost = build_transfer_cost(&fg.gpu, bridge)?;

        // Only this arch pairing is wired (others bail below). The attn and ffn
        // `attn_tp_size` are INDEPENDENT — they shard different work (attn: the
        // attention core; ffn: the qkv / o_proj projections), and the cross-pool
        // handoff that connects them is TP-agnostic: `ffn_to_attn_bytes_per_token` /
        // `attn_to_ffn_bytes_per_token` are full un-sharded wire sizes (model dims
        // only, not `attn_tp`), and `GpuCluster::submit_transfer` already prices a
        // send/recv group of differing link counts (`bytes / side_count` per leg,
        // slower side bounds). So a tp=2 ffn projecting QKV and scattering to a tp=4
        // attn pool is a valid, costable topology — each pool models its own
        // sharding. `attn_replicas` (physical attn shard count) is likewise free.
        match (&ag.arch, &fg.arch) {
            (
                AttnArchSel::Qwen3AttnTp {
                    model: am,
                    attn_tp_size: a_tp,
                },
                FfnArchSel::Qwen3FfnMoe {
                    model: fm,
                    attn_tp_size: f_tp,
                    ep_size,
                    nvl_num_gpu,
                    routing,
                    routing_seed,
                },
            ) => {
                let attn_model =
                    Arc::new(arch_build::qwen3_attn(am, *a_tp, &ag.gpu, MODEL_NAME, bridge)?);
                let ffn_model = Arc::new(arch_build::qwen3_ffn_moe(
                    fm,
                    *f_tp,
                    *ep_size,
                    *nvl_num_gpu,
                    *routing,
                    *routing_seed,
                    &fg.gpu,
                    MODEL_NAME,
                    bridge,
                )?);
                // The 3-slot ring needs ≥1 layer; (unlike the plan's ≥3, this M2
                // design does not assume a slot↔layer bijection — slots are
                // independent micro-batches addressed by index, so a small num_layers
                // is fine. Real Qwen3-MoE has dozens.)
                ensure!(
                    crate::arch::contract::AttnLayerwiseModel::num_layers(&*attn_model) >= 1,
                    "afd: model must have ≥1 layer"
                );
                Ok(assemble_afd_flow(
                    attn_model,
                    ffn_model,
                    store,
                    attn_wc,
                    ffn_wc,
                    ag.gpu.clone(),
                    fg.gpu.clone(),
                    ag.replicas,
                    fg.replicas,
                    cost,
                    // Per-building-block cost_log goes under the run's log_dir, same
                    // as a colocated run (the workers tag rows `attn` / `ffn`).
                    Some(cfg.io.log_dir.clone()),
                ))
            }
            (a, f) => bail!(
                "afd: unsupported attn/ffn arch pairing (got attn={a:?}, ffn={f:?}); \
                 wired pair: qwen3_attn→qwen3_ffn_moe"
            ),
        }
    }
}

// ── per-pool helpers ───────────────────────────────────────────────────────────

fn single_group<'a, A, W>(
    pool: &'a PoolSpec<A, W>,
    role: &str,
) -> anyhow::Result<&'a GroupSpec<A, W>> {
    ensure!(
        pool.groups.len() == 1,
        "afd: {role} pool supports a single homogeneous group (got {})",
        pool.groups.len()
    );
    Ok(&pool.groups[0])
}

fn ensure_disagg_attn(worker: &AttnWorkerSel) -> anyhow::Result<()> {
    match worker {
        AttnWorkerSel::DisaggAttn { .. } => Ok(()),
    }
}

fn ensure_disagg_ffn(worker: &FfnWorkerSel) -> anyhow::Result<()> {
    match worker {
        FfnWorkerSel::DisaggFfn {} => Ok(()),
    }
}

fn attn_gpu_memory_gb(worker: &AttnWorkerSel) -> f64 {
    match worker {
        AttnWorkerSel::DisaggAttn { attn_gpu_memory_gb } => *attn_gpu_memory_gb,
    }
}

fn worker_config(
    attn_gpu_memory_gb: f64,
    log_output_token_times: bool,
    kv_log_stride: u32,
) -> WorkerConfig {
    WorkerConfig {
        attn_kv_bytes: (attn_gpu_memory_gb * 1e9) as u64,
        log_output_token_times,
        kv_log_stride,
        ..WorkerConfig::default()
    }
}

/// Assemble an [`AfdFlow`] from the two built models. Creates the shared cluster,
/// builds the attn pool then the ffn pool into it (so GPU ids continue across
/// pools), and wires the flow. The returned `Box<dyn Flow>` is the only `dyn`
/// erasure point.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn assemble_afd_flow<MA, MF>(
    attn_model: Arc<MA>,
    ffn_model: Arc<MF>,
    store: SharedRequests,
    attn_wc: WorkerConfig,
    ffn_wc: WorkerConfig,
    attn_gpu_name: String,
    ffn_gpu_name: String,
    attn_replicas: u16,
    ffn_replicas: u16,
    cost: CostSource,
    cost_log_dir: Option<std::path::PathBuf>,
) -> Box<dyn Flow>
where
    MA: crate::arch::contract::AttnLayerwiseModel,
    MF: crate::arch::contract::FfnLayerwiseModel,
{
    let cluster: SharedGpuCluster = Rc::new(RefCell::new(GpuCluster::new(cost)));
    // Kept for the post-registration `attach_logger` — `cost_log_dir` is moved
    // into the ffn pool below.
    let net_log_dir = cost_log_dir.clone();
    let attn = AfdAttnPoolController::new(
        attn_replicas,
        attn_model,
        Rc::clone(&store),
        attn_wc,
        AFD_ATTN_POOL,
        &attn_gpu_name,
        &cluster,
        cost_log_dir.clone(),
    );
    let ffn = AfdFfnPoolController::new(
        ffn_replicas,
        ffn_model,
        Rc::clone(&store),
        ffn_wc,
        AFD_FFN_POOL,
        &ffn_gpu_name,
        &cluster,
        cost_log_dir,
    );
    // Both pools have now self-registered their comm groups (with owner identity);
    // attach the `gpu_cluster` log so every attn↔ffn transfer is recorded. Runs
    // without a log dir leave the cluster logger-less.
    if let Some(dir) = net_log_dir.as_deref() {
        cluster.borrow_mut().attach_logger(dir);
    }
    Box::new(AfdFlow::new(store, attn, ffn, cluster))
}

/// Build the AFD attn↔ffn transfer cost source: the profiled `p2p_inter` curve for
/// the receiver GPU (same kernel PD uses for its KV handoff). v1 assumes a
/// cross-node (Infiniband) transfer with a bf16 element dtype — the curve is keyed
/// by message-size bytes, so dtype only selects the profiled table. Lists nccl +
/// nvshmem for parity with the collective ops (best-of-N picks the faster); note
/// p2p_inter is an analytical, backend-agnostic curve, so both resolve alike here.
fn build_transfer_cost(gpu_name: &str, bridge: &PerfApiBridge) -> anyhow::Result<CostSource> {
    let kernel = P2pInterKernel::build(
        "afd_qkv_transfer".to_string(),
        P2pInterKernelConfig {
            backends: vec!["nccl", "nvshmem"],
            gpu_name: gpu_name.to_string(),
            fabric: Fabric::Infiniband,
            dtype: DType::Bf16,
        },
        bridge,
    )
    .with_context(|| format!("building p2p_inter kernel for AFD QKV transfer on {gpu_name}"))?;
    Ok(CostSource::Kernel(kernel))
}
