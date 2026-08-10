//! `unified` deployment — one worker type runs the whole model per iteration
//! (co-located attention + FFN), in one homogeneous `main` pool.
//!
//! `build` reads the structured `UnifiedConfig`: a single pool (`main`) with one
//! homogeneous group. The arch is selected by its explicit tag (NO `tp_size`
//! dispatch — provider-first, new-interface-design §4). Wired arms are
//! `llama3_dense` + `barebone`, `llama3_dense_tp` + `barebone`,
//! `llama3_dp_attn_tp_ffn` + `hp_unified`, and `qwen3_moe_dp_attn_ep_ffn` +
//! `hp_unified`, and `glm52_dsa_moe` + `hp_unified`. GLM's TP1 local attention
//! uses one independent KV/input partition per EP rank; the existing HP shell
//! and mutable request/KV lifecycle are unchanged. Each wired arm
//! monomorphizes its concrete model/worker pair and
//! erases to `Box<dyn Flow>` — the single `dyn` point (the cost path is
//! `dyn`-free, L4 §4.1).

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, ensure};

use crate::arch::build as arch_build;
use crate::arch::contract::IterwiseUnifiedModel;
use crate::arch::IterArchSel;
use crate::common::{PoolId, SharedRequests};
use crate::deployment::UnifiedConfig;
use crate::orchestrator::common::WorkerBuildFn;
use crate::orchestrator::{
    DpPlacementPolicy, Flow, PlacementPolicy, SimpleDpConfig, SimpleDpFlow, SimpleDpPoolConfig,
    UnifiedWorkerFactory,
};
use crate::timing::PerfApiBridge;
use crate::worker::{
    build_barebone_worker, build_hp_worker, resolve_prefix_cache_config, IterWorker, IterWorkerSel,
    WorkerConfig,
};

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

        // L5 worker env: `attn_kv_bytes` sizes the worker's KV partitions, and
        // `log_output_token_times` controls request_slo detail logging. The
        // worker *type* is matched against the arch in the arms below.
        // chunked_prefill is not wired yet.
        let (
            attn_gpu_memory_gb,
            gpu_time_multiplier,
            max_batch_tokens,
            pending_order,
            prefix_cache_mode,
            prefix_cache_policy,
            prefix_cache_max_gpu_memory_gb,
        ) = match &g.worker {
            IterWorkerSel::Barebone {
                attn_gpu_memory_gb,
                gpu_time_multiplier,
                max_batch_tokens,
                pending_order,
                prefix_cache_mode,
                prefix_cache_policy,
                prefix_cache_max_gpu_memory_gb,
            }
            | IterWorkerSel::HpUnified {
                attn_gpu_memory_gb,
                gpu_time_multiplier,
                max_batch_tokens,
                pending_order,
                prefix_cache_mode,
                prefix_cache_policy,
                prefix_cache_max_gpu_memory_gb,
            } => (
                *attn_gpu_memory_gb,
                *gpu_time_multiplier,
                *max_batch_tokens,
                *pending_order,
                *prefix_cache_mode,
                *prefix_cache_policy,
                *prefix_cache_max_gpu_memory_gb,
            ),
            IterWorkerSel::ChunkedPrefill { .. } => {
                bail!("unified: chunked_prefill worker not wired yet")
            }
            IterWorkerSel::PdPrefill { .. } | IterWorkerSel::PdDecode { .. } => {
                bail!("unified: pd_prefill / pd_decode workers belong to the `pd` deployment")
            }
        };
        let prefix_cache = resolve_prefix_cache_config(
            "unified",
            prefix_cache_mode,
            prefix_cache_policy,
            prefix_cache_max_gpu_memory_gb,
            attn_gpu_memory_gb,
        )?;
        let worker_config = WorkerConfig {
            attn_kv_bytes: (attn_gpu_memory_gb * 1e9) as u64,
            log_output_token_times: cfg.io.log_output_token_times,
            log_stage_transitions: cfg.io.log_stage_transitions,
            kv_log_stride: cfg.io.kv_log_stride,
            gpu_time_multiplier,
            max_batch_tokens,
            pending_order,
            prefix_cache,
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

        // Scope the single pool `main` over the whole arch match: one call
        // activates its backend overrides (run) and tags it for the enumerate walk
        // (emit). Every arm's `arch_build::*` threads the same `bridge`; the guard
        // restores both on drop (end of `build`, or an early `?`).
        let _scope = bridge.with_backend_overrides("main", cfg.backends.get("main"));

        // Arch selected by explicit tag; each arm builds its concrete model via
        // the shared `arch::build` builders (the same ones the `pd` deployment and
        // the offline `timing-predict` path use), validates the paired worker
        // tag, and erases via assemble_flow. dense / dense_tp run on the
        // single-group barebone worker; the DP-attn / MoE archs run on the
        // multi-group hp_unified worker (one KV partition state per DP shard).
        match &g.arch {
            IterArchSel::Llama3Dense { .. } => {
                ensure_barebone(&g.worker)?;
                let model = Arc::new(arch_build::dense(
                    model_spec, &gpu_name, MODEL_NAME, bridge,
                )?);
                Ok(assemble_flow(
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    build_barebone_worker,
                ))
            }
            IterArchSel::Llama3DenseTp { tp_size, .. } => {
                ensure_barebone(&g.worker)?;
                let model = Arc::new(arch_build::dense_tp(
                    model_spec, *tp_size, &gpu_name, MODEL_NAME, bridge,
                )?);
                Ok(assemble_flow(
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    build_barebone_worker,
                ))
            }
            IterArchSel::Llama3DpAttnTpFfn {
                attn_tp_size,
                ffn_tp_size,
                ..
            } => {
                ensure_hp_unified(&g.worker)?;
                let model = Arc::new(arch_build::dp_attn_tp_ffn(
                    model_spec,
                    *attn_tp_size,
                    *ffn_tp_size,
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?);
                Ok(assemble_flow(
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    build_hp_worker,
                ))
            }
            IterArchSel::Qwen3MoeDpAttnEpFfn {
                attn_tp_size,
                ep_size,
                hp_size,
                nvl_num_gpu,
                routing,
                routing_seed,
                expert_popularity_file,
                ..
            } => {
                ensure_hp_unified(&g.worker)?;
                let model = Arc::new(arch_build::qwen3_moe(
                    model_spec,
                    *attn_tp_size,
                    *ep_size,
                    *hp_size,
                    *nvl_num_gpu,
                    *routing,
                    *routing_seed,
                    expert_popularity_file.as_deref(),
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?);
                Ok(assemble_flow(
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    build_hp_worker,
                ))
            }
            IterArchSel::Qwen3MoeFp8DpAttnEpFfn {
                attn_tp_size,
                ep_size,
                hp_size,
                nvl_num_gpu,
                routing,
                routing_seed,
                expert_popularity_file,
                ..
            } => {
                ensure_hp_unified(&g.worker)?;
                let model = Arc::new(arch_build::qwen3_moe_fp8(
                    model_spec,
                    *attn_tp_size,
                    *ep_size,
                    *hp_size,
                    *nvl_num_gpu,
                    *routing,
                    *routing_seed,
                    expert_popularity_file.as_deref(),
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?);
                Ok(assemble_flow(
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    build_hp_worker,
                ))
            }
            IterArchSel::Qwen3VllmMoeDpAttnEpFfn {
                attn_tp_size,
                ep_size,
                hp_size,
                nvl_num_gpu,
                routing,
                routing_seed,
                expert_popularity_file,
                ..
            } => {
                ensure_hp_unified(&g.worker)?;
                let model = Arc::new(arch_build::qwen3_vllm_moe(
                    model_spec,
                    *attn_tp_size,
                    *ep_size,
                    *hp_size,
                    *nvl_num_gpu,
                    *routing,
                    *routing_seed,
                    expert_popularity_file.as_deref(),
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?);
                Ok(assemble_flow(
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    build_hp_worker,
                ))
            }
            IterArchSel::Glm52VllmDsaMoe {
                ep_size,
                nvl_num_gpu,
                routing,
                routing_seed,
                mtp_mode,
                expert_popularity_file,
                ..
            } => {
                ensure_hp_unified(&g.worker)?;
                let model = Arc::new(arch_build::glm52_vllm_dsa_moe(
                    model_spec,
                    *ep_size,
                    *nvl_num_gpu,
                    *routing,
                    *routing_seed,
                    *mtp_mode,
                    expert_popularity_file.as_deref(),
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?);
                Ok(assemble_flow(
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    build_hp_worker,
                ))
            }
            IterArchSel::Glm52DsaMoe {
                ep_size,
                nvl_num_gpu,
                routing,
                routing_seed,
                mtp_mode,
                expert_popularity_file,
                ..
            } => {
                ensure_hp_unified(&g.worker)?;
                let model = Arc::new(arch_build::glm52_dsa_moe(
                    model_spec,
                    *ep_size,
                    *nvl_num_gpu,
                    *routing,
                    *routing_seed,
                    *mtp_mode,
                    expert_popularity_file.as_deref(),
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?);
                Ok(assemble_flow(
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    build_hp_worker,
                ))
            }
        }
    }
}

/// The model's dotted-leaf prefix for this deployment (e.g. `unified.embedding`).
const MODEL_NAME: &str = "unified";

/// The dense / dense_tp archs run on the single-group barebone worker.
fn ensure_barebone(worker: &IterWorkerSel) -> anyhow::Result<()> {
    match worker {
        IterWorkerSel::Barebone { .. } => Ok(()),
        other => bail!("unified: this arch requires worker `barebone`, got {other:?}"),
    }
}

/// DP-attention and MoE archs run on the multi-group hp_unified worker.
fn ensure_hp_unified(worker: &IterWorkerSel) -> anyhow::Result<()> {
    match worker {
        IterWorkerSel::HpUnified { .. } => Ok(()),
        other => {
            bail!("unified: this DP-attention/MoE arch requires worker `hp_unified`, got {other:?}")
        }
    }
}

/// Wrap a built iter-wise model in a `simple_dp` flow. Generic over the concrete
/// model `M` and worker `W`; `build_fn` is the chosen worker's `new`. The returned
/// `Box<dyn Flow>` is the only `dyn` erasure point.
fn assemble_flow<M, W>(
    model: Arc<M>,
    store: SharedRequests,
    worker_config: WorkerConfig,
    log_dir: Option<PathBuf>,
    gpu_name: String,
    dp_cfg: SimpleDpConfig,
    build_fn: WorkerBuildFn<M, W>,
) -> Box<dyn Flow>
where
    M: IterwiseUnifiedModel,
    W: IterWorker<Event = crate::worker::WorkerEventCommon> + 'static,
    W::Msg: From<crate::common::RequestId>,
{
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::Glm52DsaMoeModel;
    use crate::common::RequestId;
    use crate::worker::{HpUnifiedWorker, WorkerEventCommon};

    fn hp_worker() -> IterWorkerSel {
        IterWorkerSel::HpUnified {
            attn_gpu_memory_gb: 80.0,
            gpu_time_multiplier: 1.0,
            max_batch_tokens: None,
            pending_order: crate::worker::PendingOrderKind::SessionStart,
            prefix_cache_mode: crate::worker::PrefixCacheMode::Opportunistic,
            prefix_cache_policy: crate::worker::PrefixCachePolicy::Lru,
            prefix_cache_max_gpu_memory_gb: None,
        }
    }

    fn barebone_worker() -> IterWorkerSel {
        IterWorkerSel::Barebone {
            attn_gpu_memory_gb: 80.0,
            gpu_time_multiplier: 1.0,
            max_batch_tokens: None,
            pending_order: crate::worker::PendingOrderKind::SessionStart,
            prefix_cache_mode: crate::worker::PrefixCacheMode::Opportunistic,
            prefix_cache_policy: crate::worker::PrefixCachePolicy::Lru,
            prefix_cache_max_gpu_memory_gb: None,
        }
    }

    fn assert_iter_worker_contract<W>()
    where
        W: IterWorker<Event = WorkerEventCommon> + 'static,
        W::Msg: From<RequestId>,
    {
    }

    #[test]
    fn glm52_hp_unified_pair_satisfies_the_iter_worker_contract() {
        assert_iter_worker_contract::<HpUnifiedWorker<Glm52DsaMoeModel>>();
        ensure_hp_unified(&hp_worker()).expect("GLM accepts hp_unified");
    }

    #[test]
    fn glm52_rejects_non_hp_iter_workers_with_an_actionable_error() {
        let error = ensure_hp_unified(&barebone_worker())
            .unwrap_err()
            .to_string();
        assert!(error.contains("DP-attention/MoE arch requires worker `hp_unified`"));
        assert!(error.contains("Barebone"));

        let pd_prefill = IterWorkerSel::PdPrefill {
            attn_gpu_memory_gb: 80.0,
            gpu_time_multiplier: 1.0,
            prefix_cache_mode: crate::worker::PrefixCacheMode::Opportunistic,
            prefix_cache_policy: crate::worker::PrefixCachePolicy::Lru,
            prefix_cache_max_gpu_memory_gb: None,
        };
        let error = ensure_hp_unified(&pd_prefill).unwrap_err().to_string();
        assert!(error.contains("worker `hp_unified`"));
        assert!(error.contains("PdPrefill"));
    }

    #[test]
    fn glm52_unified_config_preserves_ep8_hp_pairing() {
        let yaml = r#"
deployment: unified
workload: { trace_files: ["trace/smoke.csv"], trace_kind: text_generation, arrival_mode: trace_timed, session_dependency: independent, duration_ms: 1000.0, run_to_end: true, request_rate: 1.0 }
io: { log_dir: "logs/test", log_level: info, quiet: true, force_cache_build: false, log_output_token_times: false }
pools:
  main:
    placement: least-queued
    groups:
      - gpu: "NVIDIA H200"
        replicas: 1
        arch:
          type: glm52_dsa_moe
          model_config: model/config/glm52.json
          fp8: false
          ep_size: 8
          nvl_num_gpu: 8
          routing: uniform
          mtp_mode: off
        worker:
          type: hp_unified
          attn_gpu_memory_gb: 80.0
          gpu_time_multiplier: 1.0
"#;
        let cfg: crate::deployment::RunConfig =
            serde_yaml::from_str(yaml).expect("GLM hp_unified config parses");
        let crate::deployment::RunConfig::Unified(cfg) = cfg else {
            panic!("expected unified config")
        };
        let group = &cfg.pools.main.groups[0];
        let IterArchSel::Glm52DsaMoe {
            ep_size,
            nvl_num_gpu,
            mtp_mode,
            ..
        } = &group.arch
        else {
            panic!("expected GLM arch")
        };
        assert_eq!((*ep_size, *nvl_num_gpu), (8, 8));
        assert_eq!(*mtp_mode, crate::arch::Glm52MtpMode::Off);
        ensure_hp_unified(&group.worker).expect("GLM hp_unified pairing is accepted");
    }
}
