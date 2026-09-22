//! `unified` deployment — one worker type runs the whole model per iteration
//! (co-located attention + FFN), in one homogeneous `main` pool.
//!
//! `build` reads the structured `UnifiedConfig`: a single pool (`main`) with one
//! homogeneous group. The arch is selected by its explicit tag (NO `tp_size`
//! dispatch; see the L4/L6 design). Wired arms are
//! `llama3_dense` + either `barebone` or `chunked_prefill`, `llama3_dense_tp` +
//! either of the same two,
//! `llama3_dp_attn_tp_ffn` + `hp_unified`, and `qwen3_moe_dp_attn_ep_ffn` +
//! `hp_unified`, `glm52_vllm_dsa_moe` + `hp_unified`, and
//! `glm52_vllm_nvfp4_dsa_moe` + either `hp_unified` or `chunked_prefill`,
//! `glm52_vllm_nvfp4_dsa_moe_speculative` + `speculative`, and
//! `glm52_sglang_nvfp4_tp_dsa_moe` + `chunked_prefill`. GLM's TP1 local
//! attention uses one independent KV/input partition per EP rank; the worker
//! shells and mutable request/KV lifecycle are unchanged. Each wired arm
//! monomorphizes its concrete model/worker pair and
//! erases to `Box<dyn Flow>` — the single `dyn` point (the cost path is
//! `dyn`-free, L4 §4.1).
//!
//! The speculative pair is the one arm whose model does not implement
//! `IterwiseUnifiedModel` at all — it implements `SpeculativeUnifiedModel`
//! instead — which is why neither `assemble_flow` nor `UnifiedWorkerFactory`
//! names an L4 contract on `M`. The recipe each arm passes already does.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{bail, ensure};

use crate::arch::build as arch_build;
use crate::arch::contract::IterwiseUnifiedModel;
use crate::arch::{Glm52MtpMode, IterArchSel};
use crate::common::{PoolId, SharedRequests, Time};
use crate::deployment::UnifiedConfig;
use crate::orchestrator::common::WorkerBuildFn;
use crate::orchestrator::{
    DpPlacementPolicy, Flow, MigrationPolicySel, MigrationTrigger, PlacementPolicy, PoolSpec,
    SimpleDpConfig, SimpleDpFlow, SimpleDpPoolConfig, TrainingConfig, TrainingSel,
    UnifiedWorkerFactory,
};
use crate::timing::PerfApiBridge;
use crate::worker::{
    build_barebone_worker, build_chunked_prefill_worker, build_hp_worker,
    build_qwen36_hybrid_worker, build_speculative_worker, resolve_prefix_cache_config, BatchPolicy,
    IterWorkerSel, KvAdmissionConfig, MigratableWorker, PendingOrderKind, PrefixCacheMode,
    PrefixCachePolicy, TrainChunkCost, WorkerConfig, WorkerMsgCommon,
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
        let (
            attn_gpu_memory_gb,
            gpu_time_multiplier,
            max_batch_tokens,
            pending_order,
            prefix_cache_mode,
            prefix_cache_policy,
            prefix_cache_max_gpu_memory_gb,
            batch_policy,
            kv_admission,
        ) = match &g.worker {
            IterWorkerSel::Barebone {
                attn_gpu_memory_gb,
                gpu_time_multiplier,
                max_batch_tokens,
                pending_order,
                prefix_cache_mode,
                prefix_cache_policy,
                prefix_cache_max_gpu_memory_gb,
                // Hybrid-only; read separately so the two arms keep the same
                // bindings. See `ssm_checkpoint_interval_tokens` below.
                ..
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
                BatchPolicy::Mix,
                KvAdmissionConfig::FullFootprint,
            ),
            IterWorkerSel::ChunkedPrefill {
                attn_gpu_memory_gb,
                max_batch_tokens,
                batch_policy,
                kv_admission,
                gpu_time_multiplier,
            } => {
                let kv_admission = kv_admission.resolve()?;
                ensure!(
                    !matches!(kv_admission, KvAdmissionConfig::BoundedFuture(_))
                        || *batch_policy == BatchPolicy::SeparatePrefillPriority,
                    "bounded-future KV admission currently requires \
                     batch_policy=separate-prefill-priority"
                );
                (
                    *attn_gpu_memory_gb,
                    *gpu_time_multiplier,
                    Some(*max_batch_tokens),
                    PendingOrderKind::Fifo,
                    PrefixCacheMode::Opportunistic,
                    PrefixCachePolicy::Lru,
                    None,
                    *batch_policy,
                    kv_admission,
                )
            }
            IterWorkerSel::Speculative {
                attn_gpu_memory_gb,
                max_batch_tokens,
                batch_policy,
                gpu_time_multiplier,
                // Read by the two helpers below so this arm keeps the same
                // bindings as its siblings. See `ssm_checkpoint_interval_tokens`.
                ..
            } => (
                *attn_gpu_memory_gb,
                *gpu_time_multiplier,
                Some(*max_batch_tokens),
                PendingOrderKind::Fifo,
                PrefixCacheMode::Opportunistic,
                PrefixCachePolicy::Lru,
                None,
                *batch_policy,
                // Not a selector field: bounded-future predicts page crossings
                // from single-token advance, which an accepted chain skips.
                KvAdmissionConfig::FullFootprint,
            ),
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
            batch_policy,
            kv_admission,
            prefix_cache,
            ssm_checkpoint_interval_tokens: ssm_checkpoint_interval_tokens(&g.worker),
            speculative_draft_tokens: speculative_draft_tokens(&g.worker),
            speculative_acceptance_seed: speculative_acceptance_seed(&g.worker),
            ..WorkerConfig::default()
        };

        let log_dir: Option<PathBuf> = Some(cfg.io.log_dir.clone());
        let dp_cfg = SimpleDpConfig {
            dp_pool: SimpleDpPoolConfig {
                pool: PoolId(0),
                num_workers: g.replicas,
                placement: placement_into(pool.placement),
            },
            migration: migration_into(pool),
            training: training_into(pool)?,
            log_dir: log_dir.clone(),
        };
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
            // Barebone cadence, hybrid KV: 30 GDN layers of per-request
            // recurrent state and 10 GQA layers of per-token KV live in one
            // attention budget, so only the KV axis differs from the arms below.
            IterArchSel::Qwen36Local {
                routing,
                routing_seed,
                expert_popularity_file,
                ..
            } => {
                ensure_barebone(&g.worker)?;
                let model = Arc::new(arch_build::qwen36_local(
                    model_spec,
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
                    build_qwen36_hybrid_worker,
                ))
            }
            IterArchSel::Llama3Dense { .. } => {
                ensure_barebone_or_chunked("llama3_dense", &g.worker)?;
                let model = Arc::new(arch_build::dense(
                    model_spec, &gpu_name, MODEL_NAME, bridge,
                )?);
                assemble_barebone_or_chunked_flow(
                    "llama3_dense",
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    &g.worker,
                )
            }
            IterArchSel::Llama3DenseTp { tp_size, .. } => {
                ensure_barebone_or_chunked("llama3_dense_tp", &g.worker)?;
                let model = Arc::new(arch_build::dense_tp(
                    model_spec, *tp_size, &gpu_name, MODEL_NAME, bridge,
                )?);
                assemble_barebone_or_chunked_flow(
                    "llama3_dense_tp",
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    &g.worker,
                )
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
            arch @ (IterArchSel::DeepseekV4Vllm {
                routing,
                routing_seed,
                expert_popularity_file,
                ..
            }
            | IterArchSel::DeepseekV4VllmSerialStreams {
                routing,
                routing_seed,
                expert_popularity_file,
                ..
            }) => {
                ensure_hp_or_chunked_worker("DeepSeek-V4", &g.worker)?;
                let serialize_streams =
                    matches!(arch, IterArchSel::DeepseekV4VllmSerialStreams { .. });
                let model = Arc::new(arch_build::deepseek_v4_vllm(
                    model_spec,
                    *routing,
                    *routing_seed,
                    expert_popularity_file.as_deref(),
                    serialize_streams,
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?);
                assemble_hp_or_chunked_flow(
                    "DeepSeek-V4",
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    &g.worker,
                )
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
            IterArchSel::Glm52VllmNvfp4DsaMoe {
                ep_size,
                nvl_num_gpu,
                max_model_len,
                routing,
                routing_seed,
                mtp_mode,
                expert_popularity_file,
                ..
            } => {
                ensure_hp_or_chunked_worker("GLM-4.5 NVFP4", &g.worker)?;
                let model = Arc::new(arch_build::glm52_vllm_nvfp4_dsa_moe(
                    model_spec,
                    *ep_size,
                    *nvl_num_gpu,
                    *max_model_len,
                    *routing,
                    *routing_seed,
                    *mtp_mode,
                    expert_popularity_file.as_deref(),
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?);
                assemble_hp_or_chunked_flow(
                    "GLM-4.5 NVFP4",
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    &g.worker,
                )
            }
            IterArchSel::Glm52VllmNvfp4DsaMoeSpeculative {
                ep_size,
                nvl_num_gpu,
                max_model_len,
                routing,
                routing_seed,
                mtp_mode,
                draft_tokens,
                expert_popularity_file,
                draft_expert_popularity_file,
                ..
            } => {
                ensure_speculative(&g.worker, *draft_tokens)?;
                ensure!(
                    *mtp_mode != Glm52MtpMode::Off,
                    "unified: a speculative GLM must run its MTP layer; mtp_mode=off \
                     leaves it with nothing to draft with"
                );
                let model = Arc::new(arch_build::glm52_vllm_nvfp4_dsa_moe_speculative(
                    model_spec,
                    *ep_size,
                    *nvl_num_gpu,
                    *max_model_len,
                    *routing,
                    *routing_seed,
                    *mtp_mode,
                    expert_popularity_file.as_deref(),
                    draft_expert_popularity_file.as_deref(),
                    *draft_tokens,
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
                    build_speculative_worker,
                ))
            }
            IterArchSel::Glm52SglangNvfp4TpDsaMoe {
                tp_size,
                max_model_len,
                routing,
                routing_seed,
                mtp_mode,
                expert_popularity_file,
                ..
            } => {
                ensure_hp_or_chunked_worker("GLM-5.2 SGLang NVFP4 pure TP", &g.worker)?;
                let model = Arc::new(arch_build::glm52_sglang_nvfp4_tp_dsa_moe(
                    model_spec,
                    *tp_size,
                    *max_model_len,
                    *routing,
                    *routing_seed,
                    *mtp_mode,
                    expert_popularity_file.as_deref(),
                    &gpu_name,
                    MODEL_NAME,
                    bridge,
                )?);
                assemble_hp_or_chunked_flow(
                    "GLM-5.2 SGLang NVFP4 pure TP",
                    model,
                    store,
                    worker_config,
                    log_dir,
                    gpu_name,
                    dp_cfg,
                    &g.worker,
                )
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

/// The hybrid KV recipe's optional snapshot-interval override. Only `barebone`
/// carries it; every other selector leaves the arch's own alignment in force.
fn ssm_checkpoint_interval_tokens(worker: &IterWorkerSel) -> Option<u32> {
    match worker {
        IterWorkerSel::Barebone {
            ssm_checkpoint_interval_tokens,
            ..
        } => *ssm_checkpoint_interval_tokens,
        _ => None,
    }
}

/// Draft candidate count, carried by the `speculative` selector only. Every other
/// worker leaves it at `0`, which is what marks it as not speculating.
fn speculative_draft_tokens(worker: &IterWorkerSel) -> u32 {
    match worker {
        IterWorkerSel::Speculative { draft_tokens, .. } => *draft_tokens,
        _ => 0,
    }
}

/// Which deterministic acceptance stream the speculative worker draws from.
fn speculative_acceptance_seed(worker: &IterWorkerSel) -> Option<u64> {
    match worker {
        IterWorkerSel::Speculative {
            acceptance_seed, ..
        } => *acceptance_seed,
        _ => None,
    }
}

/// The speculative GLM arch pairs only with the speculative worker: it builds a
/// different model type, so no other recipe can even name it.
fn ensure_speculative(worker: &IterWorkerSel, arch_draft_tokens: u32) -> anyhow::Result<()> {
    match worker {
        IterWorkerSel::Speculative { draft_tokens, .. } => {
            ensure!(
                *draft_tokens == arch_draft_tokens,
                "unified: worker draft_tokens={draft_tokens} must match the arch's \
                 draft_tokens={arch_draft_tokens} — the width selects a profiled \
                 kernel shape, so the two cannot differ"
            );
            Ok(())
        }
        other => {
            bail!("unified: this speculative arch requires worker `speculative`, got {other:?}")
        }
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

/// The dense llama3 archs accept `barebone` or `chunked_prefill`. They differ
/// only on the admission axis — same `FullAttnKv`, same `UnifiedIterExecution`
/// — so the pairing is a policy choice, not a type constraint, and a study that
/// needs chunking (or migration, which only `chunked_prefill` can take over)
/// should not have to change arch.
fn ensure_barebone_or_chunked(arch_name: &str, worker: &IterWorkerSel) -> anyhow::Result<()> {
    match worker {
        IterWorkerSel::Barebone { .. } | IterWorkerSel::ChunkedPrefill { .. } => Ok(()),
        other => bail!(
            "unified: {arch_name} requires worker `barebone` or `chunked_prefill`, got {other:?}"
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn assemble_barebone_or_chunked_flow<M>(
    arch_name: &str,
    model: Arc<M>,
    store: SharedRequests,
    worker_config: WorkerConfig,
    log_dir: Option<PathBuf>,
    gpu_name: String,
    dp_cfg: SimpleDpConfig,
    worker: &IterWorkerSel,
) -> anyhow::Result<Box<dyn Flow>>
where
    M: IterwiseUnifiedModel,
{
    match worker {
        IterWorkerSel::Barebone { .. } => Ok(assemble_flow(
            model,
            store,
            worker_config,
            log_dir,
            gpu_name,
            dp_cfg,
            build_barebone_worker,
        )),
        IterWorkerSel::ChunkedPrefill { .. } => Ok(assemble_flow(
            model,
            store,
            worker_config,
            log_dir,
            gpu_name,
            dp_cfg,
            build_chunked_prefill_worker,
        )),
        other => bail!("unified: unsupported {arch_name} worker {other:?}"),
    }
}

/// Whole-iteration arches with a captured chunked-prefill runtime may use the
/// ordinary HP admission recipe or the dedicated hard-cap/chunking recipe.
fn ensure_hp_or_chunked_worker(arch_name: &str, worker: &IterWorkerSel) -> anyhow::Result<()> {
    match worker {
        IterWorkerSel::HpUnified { .. } | IterWorkerSel::ChunkedPrefill { .. } => Ok(()),
        other => bail!(
            "unified: {arch_name} requires worker `hp_unified` or `chunked_prefill`, got {other:?}"
        ),
    }
}

fn assemble_hp_or_chunked_flow<M>(
    arch_name: &str,
    model: Arc<M>,
    store: SharedRequests,
    worker_config: WorkerConfig,
    log_dir: Option<PathBuf>,
    gpu_name: String,
    dp_cfg: SimpleDpConfig,
    worker: &IterWorkerSel,
) -> anyhow::Result<Box<dyn Flow>>
where
    M: IterwiseUnifiedModel,
{
    match worker {
        IterWorkerSel::HpUnified { .. } => Ok(assemble_flow(
            model,
            store,
            worker_config,
            log_dir,
            gpu_name,
            dp_cfg,
            build_hp_worker,
        )),
        IterWorkerSel::ChunkedPrefill { .. } => Ok(assemble_flow(
            model,
            store,
            worker_config,
            log_dir,
            gpu_name,
            dp_cfg,
            build_chunked_prefill_worker,
        )),
        other => bail!("unified: unsupported {arch_name} worker {other:?}"),
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
    // No L4 bound on `M`: `build_fn` already states the model contract its
    // recipe needs, and the speculative recipe's model implements
    // `SpeculativeUnifiedModel` rather than `IterwiseUnifiedModel`.
    // `MigratableWorker` is what lets this flow carry a migration hook. Every
    // worker a unified deployment can build satisfies it; the families that
    // cannot be drained (PD, AFD) never reach this function.
    W: MigratableWorker<Event = crate::worker::WorkerEventCommon> + 'static,
    W::Msg: From<crate::common::RequestId> + From<WorkerMsgCommon>,
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

/// Build the pool's migration hook, or `None` for the default `off`.
///
/// `off` deliberately yields no policy object at all: the flow then skips the
/// whole hook, so a preset that does not ask for migration runs the same tick
/// path it ran before migration existed.
fn migration_into<A, W>(pool: &PoolSpec<A, W>) -> Option<MigrationTrigger> {
    match pool.migration {
        MigrationPolicySel::Off => None,
        MigrationPolicySel::ActiveBatchBelow => Some(MigrationTrigger::active_batch_below(
            pool.migration_threshold,
            Time::from_ms(pool.migration_cooldown_ms),
        )),
        MigrationPolicySel::TrainGroupSamplesBelow => {
            Some(MigrationTrigger::train_group_samples_below(
                pool.migration_threshold,
                pool.migration_group_size,
                pool.migration_workers_per_train_group,
                Time::from_ms(pool.migration_group_latency_ms),
            ))
        }
    }
}

/// Build the pool's training side, or `None` for the default `off` — which, as
/// with `migration_into`, means no object at all rather than an idle one.
///
/// The rate is required: a training pool with no cost would train the whole
/// rollout in an instant and report a finish time that looks like a result.
fn training_into<A, W>(pool: &PoolSpec<A, W>) -> anyhow::Result<Option<TrainingConfig>> {
    match pool.training {
        TrainingSel::Off => Ok(None),
        TrainingSel::StreamingWorkSteal => {
            ensure!(
                pool.train_tokens_per_s > 0.0,
                "training is on but train_tokens_per_s is {}; the chunk cost is a \
                 calibration and has no default worth using",
                pool.train_tokens_per_s,
            );
            Ok(Some(TrainingConfig {
                workers_per_block: pool.migration_workers_per_train_group,
                group_size: pool.migration_group_size,
                cost: TrainChunkCost {
                    tokens_per_s: pool.train_tokens_per_s,
                    overhead: Time::from_ms(pool.train_chunk_overhead_ms),
                },
                bulk_grab: pool.train_groups_per_grab,
                tail_threshold: pool.train_tail_threshold,
                expected_groups: pool.train_expected_groups,
            }))
        }
    }
}

fn placement_into(p: PlacementPolicy) -> DpPlacementPolicy {
    match p {
        PlacementPolicy::LeastQueued => DpPlacementPolicy::LeastQueued,
        PlacementPolicy::RoundRobin => DpPlacementPolicy::RoundRobin,
        PlacementPolicy::TraceDirected => DpPlacementPolicy::TraceDirected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::{
        Glm52VllmDsaMoeModel, Glm52VllmNvfp4DsaMoeModel, Glm52VllmNvfp4DsaMoeSpeculativeModel,
        Llama3DenseModel, Qwen36LocalModel,
    };
    use crate::common::RequestId;
    use crate::worker::{BareboneWorker, ChunkedPrefillWorker, HpUnifiedWorker, WorkerEventCommon};

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
            ssm_checkpoint_interval_tokens: None,
        }
    }

    fn chunked_prefill_worker() -> IterWorkerSel {
        IterWorkerSel::ChunkedPrefill {
            attn_gpu_memory_gb: 120.0,
            max_batch_tokens: 2048,
            batch_policy: BatchPolicy::Mix,
            kv_admission: crate::worker::config::KvAdmissionSpec::default(),
            gpu_time_multiplier: 1.0,
        }
    }

    /// The contract `assemble_flow` demands: the `IterWorker` surface plus
    /// drainability, since a unified flow can carry a migration hook.
    fn assert_iter_worker_contract<W>()
    where
        W: MigratableWorker<Event = WorkerEventCommon> + 'static,
        W::Msg: From<RequestId> + From<WorkerMsgCommon>,
    {
    }

    #[test]
    fn glm52_hp_unified_pair_satisfies_the_iter_worker_contract() {
        assert_iter_worker_contract::<HpUnifiedWorker<Glm52VllmDsaMoeModel>>();
        ensure_hp_unified(&hp_worker()).expect("GLM accepts hp_unified");
    }

    #[test]
    fn glm52_nvfp4_chunked_prefill_pair_satisfies_the_iter_worker_contract() {
        assert_iter_worker_contract::<ChunkedPrefillWorker<Glm52VllmNvfp4DsaMoeModel>>();
        ensure_hp_or_chunked_worker("GLM-4.5 NVFP4", &chunked_prefill_worker())
            .expect("GLM NVFP4 accepts the dedicated chunked-prefill worker");
    }

    /// The dense llama3 archs take either admission policy. `chunked_prefill`
    /// is what a migrating pool needs, since only it can re-prefill a request
    /// whose KV another worker dropped.
    #[test]
    fn llama3_dense_pairs_with_barebone_or_chunked_prefill() {
        assert_iter_worker_contract::<BareboneWorker<Llama3DenseModel>>();
        assert_iter_worker_contract::<ChunkedPrefillWorker<Llama3DenseModel>>();
        ensure_barebone_or_chunked("llama3_dense", &barebone_worker())
            .expect("llama3_dense accepts barebone");
        ensure_barebone_or_chunked("llama3_dense", &chunked_prefill_worker())
            .expect("llama3_dense accepts chunked_prefill");
        let error = ensure_barebone_or_chunked("llama3_dense", &hp_worker())
            .unwrap_err()
            .to_string();
        assert!(error.contains("`barebone` or `chunked_prefill`"), "{error}");
    }

    /// Qwen3.6 local keeps the `barebone` preset tag — cadence, admission and
    /// execution really are barebone — but composes the hybrid KV store, so the
    /// contract is asserted against `Qwen36HybridWorker`, not `BareboneWorker`.
    #[test]
    fn qwen36_local_barebone_pair_satisfies_the_iter_worker_contract() {
        assert_iter_worker_contract::<crate::worker::Qwen36HybridWorker<Qwen36LocalModel>>();
        ensure_barebone(&barebone_worker()).expect("Qwen3.6 local accepts barebone");
        let error = ensure_barebone(&hp_worker()).unwrap_err().to_string();
        assert!(error.contains("requires worker `barebone`"));
    }

    #[test]
    fn the_snapshot_interval_override_is_barebone_only_and_defaults_to_the_arch() {
        assert_eq!(ssm_checkpoint_interval_tokens(&barebone_worker()), None);
        assert_eq!(ssm_checkpoint_interval_tokens(&hp_worker()), None);
        let IterWorkerSel::Barebone {
            attn_gpu_memory_gb,
            gpu_time_multiplier,
            max_batch_tokens,
            pending_order,
            prefix_cache_mode,
            prefix_cache_policy,
            prefix_cache_max_gpu_memory_gb,
            ..
        } = barebone_worker()
        else {
            unreachable!()
        };
        let overridden = IterWorkerSel::Barebone {
            attn_gpu_memory_gb,
            gpu_time_multiplier,
            max_batch_tokens,
            pending_order,
            prefix_cache_mode,
            prefix_cache_policy,
            prefix_cache_max_gpu_memory_gb,
            ssm_checkpoint_interval_tokens: Some(528),
        };
        assert_eq!(ssm_checkpoint_interval_tokens(&overridden), Some(528));
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
workload: { trace_files: ["trace/smoke.csv"], input_file_format: text-generation-independent, arrival_mode: trace_timed, session_dependency: independent, duration_ms: 1000.0, run_to_end: true, request_rate: 1.0 }
io: { log_dir: "logs/test", log_level: info, quiet: true, force_cache_build: false, log_output_token_times: false }
pools:
  main:
    placement: least-queued
    groups:
      - gpu: "NVIDIA H200"
        replicas: 1
        arch:
          type: glm52_vllm_dsa_moe
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
        let IterArchSel::Glm52VllmDsaMoe {
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

    fn speculative_worker(draft_tokens: u32) -> IterWorkerSel {
        IterWorkerSel::Speculative {
            attn_gpu_memory_gb: 180.0,
            max_batch_tokens: 8192,
            draft_tokens,
            acceptance_seed: Some(7),
            batch_policy: BatchPolicy::Mix,
            gpu_time_multiplier: 1.0,
        }
    }

    #[test]
    fn the_speculative_pair_satisfies_the_iter_worker_contract() {
        assert_iter_worker_contract::<
            crate::worker::SpeculativeWorker<Glm52VllmNvfp4DsaMoeSpeculativeModel>,
        >();
        ensure_speculative(&speculative_worker(3), 3).expect("matching widths pair");
    }

    #[test]
    fn a_verify_width_that_disagrees_with_the_arch_is_rejected_before_anything_is_built() {
        // The defect this catches: the arch compiles a tree for k=3 while the
        // worker submits k=5 rows. The model would reject the batch at runtime,
        // deep inside an iteration, instead of at configuration time.
        let error = ensure_speculative(&speculative_worker(5), 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("draft_tokens=5"));
        assert!(error.contains("draft_tokens=3"));

        let error = ensure_speculative(&chunked_prefill_worker(), 3)
            .unwrap_err()
            .to_string();
        assert!(error.contains("requires worker `speculative`"));
    }

    #[test]
    fn a_speculative_preset_parses_into_the_paired_arch_and_worker() {
        // Both selectors are new serde surfaces. This is the shape a preset
        // author actually writes, so a rename or a missing `serde(default)`
        // shows up here rather than at run time.
        let yaml = r#"
deployment: unified
workload: { trace_files: ["trace/smoke.csv"], input_file_format: text-generation-independent, arrival_mode: trace_timed, session_dependency: independent, duration_ms: 1000.0, run_to_end: true, request_rate: 1.0 }
io: { log_dir: "logs/test", log_level: info, quiet: true, force_cache_build: false, log_output_token_times: false }
pools:
  main:
    placement: least-queued
    groups:
      - gpu: "NVIDIA B200"
        replicas: 1
        arch:
          type: glm52_vllm_nvfp4_dsa_moe_speculative
          model_config: model/config/glm52.json
          fp8: false
          ep_size: 4
          nvl_num_gpu: 4
          max_model_len: 8192
          draft_tokens: 3
        worker:
          type: speculative
          attn_gpu_memory_gb: 180.0
          max_batch_tokens: 8192
          draft_tokens: 3
"#;
        let cfg: crate::deployment::RunConfig =
            serde_yaml::from_str(yaml).expect("speculative config parses");
        let crate::deployment::RunConfig::Unified(cfg) = cfg else {
            panic!("expected unified config")
        };
        let group = &cfg.pools.main.groups[0];
        let IterArchSel::Glm52VllmNvfp4DsaMoeSpeculative {
            draft_tokens,
            mtp_mode,
            max_model_len,
            ..
        } = &group.arch
        else {
            panic!("expected the speculative GLM arch")
        };
        assert_eq!((*draft_tokens, *max_model_len), (3, 8192));
        // A speculative arch has to run its drafter, so unlike the ordinary
        // selector it does not default `mtp_mode` to `off`.
        assert_eq!(*mtp_mode, Glm52MtpMode::IndexShare);
        ensure_speculative(&group.worker, *draft_tokens).expect("the pair is accepted");
        assert_eq!(speculative_draft_tokens(&group.worker), 3);
        assert_eq!(speculative_acceptance_seed(&group.worker), None);
    }

    #[test]
    fn only_the_speculative_selector_carries_a_verify_width_or_an_acceptance_seed() {
        // Every other worker must read as "not speculating"; a nonzero default
        // would silently turn an ordinary recipe's assertion into a live width.
        assert_eq!(speculative_draft_tokens(&speculative_worker(5)), 5);
        assert_eq!(speculative_acceptance_seed(&speculative_worker(5)), Some(7));
        for worker in [barebone_worker(), hp_worker(), chunked_prefill_worker()] {
            assert_eq!(speculative_draft_tokens(&worker), 0);
            assert_eq!(speculative_acceptance_seed(&worker), None);
        }
    }
}
