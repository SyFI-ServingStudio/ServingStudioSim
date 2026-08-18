//! Build an iter-wise arch model from its [`IterArchSel`] selector — the single
//! home for the per-arch `build_configs → resolve_configs → build` chain, the
//! `ModelSpec`→cfg layer-count override, and MoE routing resolution.
//!
//! Every caller that needs a built model goes through here: the `unified` and
//! `pd` deployments call the per-arch [`dense`] / [`dense_tp`] / … builders and
//! keep the *concrete* type for their dyn-free worker factories (the cost hot
//! path stays monomorphized, L4 §4.1); the offline `timing-predict` path
//! calls [`build_iter_model`], which boxes one as `dyn` (off the hot path). The
//! caller supplies `name` — the model's dotted-leaf prefix (`"unified"` / `"pd"`)
//! — so each deployment's cost manifests read naturally.

use std::fs::File;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

use crate::arch::config::{AttnArchSel, FfnArchSel, IterArchSel, ModelSpec, RoutingKind};
use crate::arch::model_cfg::ModelCfg;
use crate::arch::moe_model_cfg::MoeModelCfg;
use crate::arch::{
    glm52_dsa_moe, glm52_vllm_dsa_moe, llama3_dense, llama3_dense_tp, llama3_dp_attn_tp_ffn,
    qwen3_attn_layerwise, qwen3_ffn_moe_layerwise, qwen3_fp8_ffn_moe_layerwise,
    qwen3_moe_dp_attn_ep_ffn, qwen3_moe_fp8_dp_attn_ep_ffn, qwen3_vllm_moe_dp_attn_ep_ffn,
    AttnLayerwiseModel, DenseParallel, DenseTpParallel, DpAttnTpFfnParallel, FfnLayerwiseModel,
    Glm52DsaMoeModel, Glm52DsaMoeParallel, Glm52ModelCfg, Glm52MtpMode, Glm52VllmDsaMoeModel,
    Glm52VllmDsaMoeParallel, IterwiseUnifiedModel, Llama3DenseModel, Llama3DenseTpModel,
    Llama3DpAttnTpFfnModel, Qwen3AttnLayerwiseModel, Qwen3AttnParallel, Qwen3FfnMoeLayerwiseModel,
    Qwen3FfnMoeParallel, Qwen3Fp8FfnMoeLayerwiseModel, Qwen3Fp8FfnMoeParallel,
    Qwen3MoeDpAttnEpFfnModel, Qwen3MoeFp8DpAttnEpFfnModel, Qwen3MoeFp8Parallel, Qwen3MoeParallel,
    qwen36_local, Qwen36LocalModel, Qwen36LocalParallel, Qwen36ModelCfg,
    Qwen3VllmMoeDpAttnEpFfnModel, Qwen3VllmMoeParallel,
};
use crate::timing::routing::RoutingDistribution;
use crate::timing::PerfApiBridge;

/// `ModelSpec` → dense [`ModelCfg`], applying the `sim_num_layers` / `num_layers`
/// override that truncates layer COUNT before `build_configs` (per-layer shape
/// is unchanged, so it is not a cache key).
pub fn dense_model_cfg(model_spec: &ModelSpec) -> Result<ModelCfg> {
    let mut cfg = ModelCfg::from_json(Path::new(&model_spec.model_config))?;
    if let Some(n) = model_spec.sim_num_layers.or(model_spec.num_layers) {
        cfg.num_layers = n;
    }
    Ok(cfg)
}

/// `ModelSpec` → [`MoeModelCfg`] (separate from [`dense_model_cfg`]: MoE configs
/// add `num_experts` / `num_experts_per_tok` / `moe_intermediate_size`).
pub fn moe_model_cfg(model_spec: &ModelSpec) -> Result<MoeModelCfg> {
    let mut cfg = MoeModelCfg::from_json(Path::new(&model_spec.model_config))?;
    if let Some(n) = model_spec.sim_num_layers.or(model_spec.num_layers) {
        cfg.num_layers = n;
    }
    // Fold the `fp8` selector in here — the single choke point every MoE builder
    // (unified + both AFD sides) passes through — so FP8 backend/dtype selection
    // is uniform across archs.
    Ok(cfg.with_fp8(model_spec.fp8))
}

/// `ModelSpec` → exact heterogeneous GLM-5.2 model identity. Unlike the
/// homogeneous dense/MoE helpers, this architecture cannot truncate or scale a
/// representative subset of layers: its full-index and dense/sparse schedules
/// are tied to exact layer numbers.
/// Sparse (MoE) decoder layers, read off the checkpoint's own layer schedule
/// rather than assumed. An expert-popularity profile is keyed by this count.
fn num_sparse_layers(model_cfg: &Glm52ModelCfg) -> u32 {
    model_cfg
        .mlp_layer_types
        .iter()
        .filter(|kind| kind.as_str() == "sparse")
        .count() as u32
}

pub fn glm52_model_cfg(model_spec: &ModelSpec) -> Result<Glm52ModelCfg> {
    ensure_glm52_model_spec(model_spec)?;
    Glm52ModelCfg::from_json(Path::new(&model_spec.model_config))
}

fn ensure_glm52_model_spec(model_spec: &ModelSpec) -> Result<()> {
    if model_spec.num_layers.is_some() || model_spec.sim_num_layers.is_some() {
        bail!(
            "glm52_dsa_moe rejects num_layers/sim_num_layers overrides; the exact heterogeneous 78-layer schedule is required"
        );
    }
    Ok(())
}

/// Resolve the synthetic MoE routing distribution the L2 MoE op samples against from the
/// selector's [`RoutingKind`]: `uniform` spreads load evenly over `num_experts`;
/// `random` draws a `seed`-seeded deterministic skew (0 when unset). Profile-backed
/// Qwen builds use [`resolve_routing_source`] below.
pub fn resolve_routing(
    kind: RoutingKind,
    seed: Option<u64>,
    num_experts: u32,
) -> RoutingDistribution {
    match kind {
        RoutingKind::Uniform => RoutingDistribution::uniform(num_experts),
        RoutingKind::Random => RoutingDistribution::random(num_experts, seed.unwrap_or(0)),
    }
}

/// Legacy schema-v1 subset. Keep this permissive reader only so existing
/// alignment artifacts remain usable; every newly generated profile is v2.
#[derive(Debug, Deserialize)]
struct ExpertPopularityProfileV1 {
    #[serde(rename = "schema_version")]
    _schema_version: u32,
    num_logical_experts: u32,
    #[serde(default)]
    counts_by_layer: Vec<Vec<u64>>,
    #[serde(default)]
    probabilities_all_layers: Vec<f64>,
    #[serde(default)]
    counts_all_layers: Vec<u64>,
}

/// Schema v2 is deliberately closed: producer and consumer must change the
/// version when adding or reinterpreting fields. Cross-field dimensions are
/// validated after serde because JSON Schema cannot express them all.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpertPopularityProfileV2 {
    schema_version: u32,
    model: String,
    num_moe_layers: u32,
    num_logical_experts: u32,
    expert_parallel_size: u16,
    experts_per_rank: u32,
    experts_per_token: u32,
    count_semantics: String,
    aggregation: ExpertPopularityAggregationV2,
    expert_partitioning: ExpertPopularityPartitioningV2,
    counts_by_layer: Vec<Vec<u64>>,
    probabilities_by_layer: Vec<Vec<f64>>,
    counts_all_layers: Vec<u64>,
    probabilities_all_layers: Vec<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpertPopularityAggregationV2 {
    scope: String,
    observed_eplb_step_min: u64,
    observed_eplb_step_max: u64,
    record_count: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpertPopularityPartitioningV2 {
    kind: String,
    layout: String,
}

#[derive(Debug, Deserialize)]
struct ExpertPopularityVersion {
    schema_version: u32,
}

/// Canonicalize layerwise EP load before the L4 `Scale{num_layers}` fold.
///
/// Expert ids and EP rank ids are irrelevant to local expert-compute cost. For
/// every layer, sort experts within each physical rank by descending load, then
/// sort ranks by descending total load. Adding equal canonical slots across
/// layers makes canonical rank 0 carry `sum(layer-wise max rank)` instead of the
/// biased `max(rank-wise sum over layers)`. Ties use the sorted expert vector,
/// keeping the transformation deterministic without restoring physical ids.
fn canonicalize_layerwise_expert_counts(
    counts_by_layer: &[Vec<u64>],
    expected_num_experts: u32,
    ep_size: u16,
    path: &str,
) -> Result<Vec<f32>> {
    anyhow::ensure!(ep_size > 0, "ep_size must be non-zero");
    anyhow::ensure!(
        expected_num_experts % u32::from(ep_size) == 0,
        "expert popularity profile {} cannot partition {} experts across ep_size {}",
        path,
        expected_num_experts,
        ep_size
    );
    let experts_per_rank = expected_num_experts as usize / usize::from(ep_size);
    let mut canonical_counts = vec![0u64; expected_num_experts as usize];

    for (layer_index, layer_counts) in counts_by_layer.iter().enumerate() {
        anyhow::ensure!(
            layer_counts.len() == expected_num_experts as usize,
            "expert popularity profile {} layer {} has {} experts, expected {}",
            path,
            layer_index,
            layer_counts.len(),
            expected_num_experts
        );
        let mut rank_counts = layer_counts
            .chunks_exact(experts_per_rank)
            .map(|physical_rank_counts| {
                let mut sorted_expert_counts = physical_rank_counts.to_vec();
                sorted_expert_counts.sort_unstable_by(|left, right| right.cmp(left));
                let rank_total = sorted_expert_counts.iter().try_fold(0u64, |total, count| {
                    total.checked_add(*count).ok_or_else(|| {
                        anyhow::anyhow!(
                            "expert popularity profile {} layer {} rank count overflow",
                            path,
                            layer_index
                        )
                    })
                })?;
                Ok((rank_total, sorted_expert_counts))
            })
            .collect::<Result<Vec<_>>>()?;
        rank_counts.sort_unstable_by(|left, right| {
            right.0.cmp(&left.0).then_with(|| right.1.cmp(&left.1))
        });

        for (canonical_rank, (_, sorted_expert_counts)) in rank_counts.iter().enumerate() {
            let canonical_start = canonical_rank * experts_per_rank;
            for (expert_slot, count) in sorted_expert_counts.iter().enumerate() {
                let canonical_slot = canonical_start + expert_slot;
                canonical_counts[canonical_slot] = canonical_counts[canonical_slot]
                    .checked_add(*count)
                    .with_context(|| {
                        format!(
                            "expert popularity profile {path} canonical count overflow at slot {canonical_slot}"
                        )
                    })?;
            }
        }
    }

    Ok(canonical_counts
        .into_iter()
        .map(|count| count as f32)
        .collect())
}

fn load_expert_popularity(
    path: &str,
    expected_num_experts: u32,
    ep_size: u16,
    expected_num_moe_layers: u32,
    expected_experts_per_token: u32,
) -> Result<RoutingDistribution> {
    let profile_file =
        File::open(path).with_context(|| format!("opening expert popularity profile {path}"))?;
    let profile_value: serde_json::Value = serde_json::from_reader(profile_file)
        .with_context(|| format!("parsing expert popularity profile {path}"))?;
    let version: ExpertPopularityVersion = serde_json::from_value(profile_value.clone())
        .with_context(|| format!("reading expert popularity schema_version from {path}"))?;

    let (num_logical_experts, counts_by_layer, legacy_ratios) = match version.schema_version {
        1 => {
            let profile: ExpertPopularityProfileV1 = serde_json::from_value(profile_value)
                .with_context(|| format!("parsing legacy expert popularity v1 profile {path}"))?;
            let ratios = if profile.counts_by_layer.is_empty()
                && profile.probabilities_all_layers.len() == expected_num_experts as usize
            {
                Some(
                    profile
                        .probabilities_all_layers
                        .into_iter()
                        .map(|ratio| {
                            anyhow::ensure!(
                                ratio.is_finite() && ratio >= 0.0,
                                "expert popularity profile {} contains invalid probability {}",
                                path,
                                ratio
                            );
                            Ok(ratio as f32)
                        })
                        .collect::<Result<Vec<_>>>()?,
                )
            } else if profile.counts_by_layer.is_empty()
                && profile.counts_all_layers.len() == expected_num_experts as usize
            {
                Some(
                    profile
                        .counts_all_layers
                        .into_iter()
                        .map(|count| count as f32)
                        .collect(),
                )
            } else {
                None
            };
            (profile.num_logical_experts, profile.counts_by_layer, ratios)
        }
        2 => {
            let profile: ExpertPopularityProfileV2 = serde_json::from_value(profile_value)
                .with_context(|| format!("parsing strict expert popularity v2 profile {path}"))?;
            validate_expert_popularity_v2(
                &profile,
                expected_num_experts,
                ep_size,
                expected_num_moe_layers,
                expected_experts_per_token,
                path,
            )?;
            (profile.num_logical_experts, profile.counts_by_layer, None)
        }
        other => bail!(
            "unsupported expert popularity schema_version {} in {} (supported: 1 legacy, 2)",
            other,
            path
        ),
    };
    anyhow::ensure!(
        num_logical_experts == expected_num_experts,
        "expert popularity profile {} has {} experts, model requires {}",
        path,
        num_logical_experts,
        expected_num_experts
    );

    let ratios: Vec<f32> = if !counts_by_layer.is_empty() {
        canonicalize_layerwise_expert_counts(&counts_by_layer, expected_num_experts, ep_size, path)?
    } else if let Some(ratios) = legacy_ratios {
        ratios
    } else {
        bail!(
            "legacy expert popularity profile {} must contain counts_by_layer or {} probabilities_all_layers/counts_all_layers entries",
            path,
            expected_num_experts
        );
    };
    anyhow::ensure!(
        ratios.iter().any(|ratio| *ratio > 0.0),
        "expert popularity profile {} has zero total routing mass",
        path
    );
    Ok(RoutingDistribution::from_profile(&ratios))
}

fn validate_expert_popularity_v2(
    profile: &ExpertPopularityProfileV2,
    expected_num_experts: u32,
    expected_ep_size: u16,
    expected_num_moe_layers: u32,
    expected_experts_per_token: u32,
    path: &str,
) -> Result<()> {
    anyhow::ensure!(profile.schema_version == 2, "internal v2 version mismatch");
    anyhow::ensure!(
        !profile.model.is_empty(),
        "expert popularity profile {path} has empty model"
    );
    anyhow::ensure!(
        profile.num_logical_experts == expected_num_experts,
        "expert popularity profile {path} has {} experts, model requires {expected_num_experts}",
        profile.num_logical_experts
    );
    anyhow::ensure!(
        profile.num_moe_layers == expected_num_moe_layers,
        "expert popularity profile {path} has {} MoE layers, model requires {expected_num_moe_layers}",
        profile.num_moe_layers
    );
    anyhow::ensure!(
        profile.expert_parallel_size == expected_ep_size,
        "expert popularity profile {path} has ep_size {}, run requires {expected_ep_size}",
        profile.expert_parallel_size
    );
    anyhow::ensure!(
        profile.experts_per_rank * u32::from(profile.expert_parallel_size)
            == profile.num_logical_experts,
        "expert popularity profile {path} has inconsistent experts_per_rank"
    );
    anyhow::ensure!(
        profile.experts_per_token == expected_experts_per_token,
        "expert popularity profile {path} has top_k {}, model requires {expected_experts_per_token}",
        profile.experts_per_token
    );
    anyhow::ensure!(
        profile.count_semantics == "logical_routed_token_assignments",
        "expert popularity profile {path} has unsupported count_semantics {:?}",
        profile.count_semantics
    );
    anyhow::ensure!(
        profile.aggregation.scope == "all_captured_eplb_steps"
            && profile.aggregation.record_count > 0
            && profile.aggregation.observed_eplb_step_min
                <= profile.aggregation.observed_eplb_step_max,
        "expert popularity profile {path} has invalid aggregation metadata"
    );
    anyhow::ensure!(
        profile.expert_partitioning.kind == "contiguous_logical_expert_ids"
            && profile.expert_partitioning.layout == "rank_major",
        "expert popularity profile {path} has unsupported expert_partitioning"
    );
    anyhow::ensure!(
        profile.counts_by_layer.len() == expected_num_moe_layers as usize,
        "expert popularity profile {path} counts_by_layer has {} layers, expected {expected_num_moe_layers}",
        profile.counts_by_layer.len()
    );
    anyhow::ensure!(
        profile.probabilities_by_layer.len() == profile.counts_by_layer.len(),
        "expert popularity profile {path} probabilities_by_layer layer count mismatch"
    );
    anyhow::ensure!(
        profile.counts_all_layers.len() == expected_num_experts as usize
            && profile.probabilities_all_layers.len() == expected_num_experts as usize,
        "expert popularity profile {path} all-layer vector width mismatch"
    );
    let mut derived_all_layer_counts = vec![0u64; expected_num_experts as usize];
    for (layer_index, (counts, probabilities)) in profile
        .counts_by_layer
        .iter()
        .zip(&profile.probabilities_by_layer)
        .enumerate()
    {
        anyhow::ensure!(
            counts.len() == expected_num_experts as usize
                && probabilities.len() == expected_num_experts as usize,
            "expert popularity profile {path} layer {layer_index} width mismatch"
        );
        anyhow::ensure!(
            probabilities
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0),
            "expert popularity profile {path} layer {layer_index} has invalid probability"
        );
        ensure_normalized_probabilities(
            counts,
            probabilities,
            path,
            &format!("layer {layer_index}"),
        )?;
        for (expert_index, count) in counts.iter().enumerate() {
            derived_all_layer_counts[expert_index] = derived_all_layer_counts[expert_index]
                .checked_add(*count)
                .with_context(|| {
                    format!(
                        "expert popularity profile {path} all-layer count overflow at expert {expert_index}"
                    )
                })?;
        }
    }
    anyhow::ensure!(
        profile.counts_all_layers == derived_all_layer_counts,
        "expert popularity profile {path} counts_all_layers is not the sum of counts_by_layer"
    );
    ensure_normalized_probabilities(
        &profile.counts_all_layers,
        &profile.probabilities_all_layers,
        path,
        "all layers",
    )?;
    Ok(())
}

fn ensure_normalized_probabilities(
    counts: &[u64],
    probabilities: &[f64],
    path: &str,
    scope: &str,
) -> Result<()> {
    let total = counts.iter().try_fold(0u64, |sum, count| {
        sum.checked_add(*count).ok_or_else(|| {
            anyhow::anyhow!("expert popularity profile {path} {scope} count overflow")
        })
    })?;
    for (expert_index, (count, probability)) in counts.iter().zip(probabilities).enumerate() {
        let expected = if total == 0 {
            0.0
        } else {
            *count as f64 / total as f64
        };
        anyhow::ensure!(
            probability.is_finite() && (*probability - expected).abs() <= 1e-9,
            "expert popularity profile {path} {scope} probability {expert_index} is {probability}, expected {expected}"
        );
    }
    Ok(())
}

/// Resolve the Qwen routing source. An explicit popularity profile is a
/// complete routing snapshot, so combining it with the synthetic `random`
/// selector is rejected instead of silently choosing one policy.
pub fn resolve_routing_source(
    kind: RoutingKind,
    seed: Option<u64>,
    num_experts: u32,
    ep_size: u16,
    num_moe_layers: u32,
    experts_per_token: u32,
    expert_popularity_file: Option<&str>,
) -> Result<RoutingDistribution> {
    if let Some(path) = expert_popularity_file {
        anyhow::ensure!(
            kind == RoutingKind::Uniform,
            "expert_popularity_file cannot be combined with routing={:?}; omit the profile or use routing=uniform",
            kind
        );
        return load_expert_popularity(
            path,
            num_experts,
            ep_size,
            num_moe_layers,
            experts_per_token,
        );
    }
    Ok(resolve_routing(kind, seed, num_experts))
}

/// Build the dense (single-GPU) Llama3 model.
pub fn dense(
    model_spec: &ModelSpec,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Llama3DenseModel> {
    let model_cfg = dense_model_cfg(model_spec)?;
    let parallel = DenseParallel {
        gpu_name: gpu.to_string(),
    };
    let resolved =
        llama3_dense::resolve_configs(&llama3_dense::build_configs(&model_cfg, &parallel));
    llama3_dense::build(name.to_string(), resolved, bridge)
        .context("building Llama3-dense model (often a missing profile.db row)")
}

/// Build the tensor-parallel dense Llama3 model.
pub fn dense_tp(
    model_spec: &ModelSpec,
    tp_size: u16,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Llama3DenseTpModel> {
    let model_cfg = dense_model_cfg(model_spec)?;
    let parallel = DenseTpParallel {
        tp_size,
        gpu_name: gpu.to_string(),
    };
    let resolved =
        llama3_dense_tp::resolve_configs(&llama3_dense_tp::build_configs(&model_cfg, &parallel));
    llama3_dense_tp::build(name.to_string(), resolved, bridge)
        .context("building Llama3-dense-TP model (often a missing profile.db row)")
}

/// Build the exact local TP1/EP1 Qwen3.6 heterogeneous model. The nested
/// checkpoint parser validates the pinned identity before applying the
/// `ModelSpec` layer controls.
pub fn qwen36_local(
    model_spec: &ModelSpec,
    routing_kind: RoutingKind,
    routing_seed: Option<u64>,
    expert_popularity_file: Option<&str>,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Qwen36LocalModel> {
    let model_cfg = Qwen36ModelCfg::from_json(Path::new(&model_spec.model_config), model_spec)
        .context("loading exact Qwen3.6-35B-A3B FP8 model config")?;
    // EP1: the single rank owns every expert, so the "local" shard the grouped
    // GEMM sees is the whole global distribution.
    let routing = resolve_routing_source(
        routing_kind,
        routing_seed,
        model_cfg.num_experts.get(),
        1,
        model_cfg.num_layers,
        model_cfg.top_k,
        expert_popularity_file,
    )?;
    let parallel = Qwen36LocalParallel {
        gpu_name: gpu.to_string(),
    };
    let configs = qwen36_local::build_configs(&model_cfg, &parallel, &routing);
    let resolved = qwen36_local::resolve_configs(&configs);
    qwen36_local::build(name.to_string(), resolved, bridge)
        .context("building local Qwen3.6 TP1/EP1 model (often a missing profile.db row)")
}

/// Build the DP-attention + TP-FFN dense Llama3 model.
pub fn dp_attn_tp_ffn(
    model_spec: &ModelSpec,
    attn_tp_size: u16,
    ffn_tp_size: u16,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Llama3DpAttnTpFfnModel> {
    let model_cfg = dense_model_cfg(model_spec)?;
    let parallel = DpAttnTpFfnParallel {
        attn_tp_size,
        ffn_tp_size,
        gpu_name: gpu.to_string(),
    };
    let resolved = llama3_dp_attn_tp_ffn::resolve_configs(&llama3_dp_attn_tp_ffn::build_configs(
        &model_cfg, &parallel,
    ));
    llama3_dp_attn_tp_ffn::build(name.to_string(), resolved, bridge)
        .context("building Llama3 DP-attn TP-ffn model (often a missing profile.db row)")
}

/// Build the Qwen3-MoE DP-attention + EP-FFN model.
#[allow(clippy::too_many_arguments)]
pub fn qwen3_moe(
    model_spec: &ModelSpec,
    attn_tp_size: u16,
    ep_size: u16,
    hp_size: u16,
    nvl_num_gpu: u16,
    routing_kind: RoutingKind,
    routing_seed: Option<u64>,
    expert_popularity_file: Option<&str>,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Qwen3MoeDpAttnEpFfnModel> {
    let model_cfg = moe_model_cfg(model_spec)?;
    let routing = resolve_routing_source(
        routing_kind,
        routing_seed,
        model_cfg.num_experts.get(),
        ep_size,
        model_cfg.num_layers,
        model_cfg.top_k,
        expert_popularity_file,
    )?;
    let parallel = Qwen3MoeParallel {
        attn_tp_size,
        ep_size,
        hp_size,
        nvl_num_gpu,
        gpu_name: gpu.to_string(),
    };
    let resolved = qwen3_moe_dp_attn_ep_ffn::resolve_configs(
        &qwen3_moe_dp_attn_ep_ffn::build_configs(&model_cfg, &parallel, &routing),
    );
    qwen3_moe_dp_attn_ep_ffn::build(name.to_string(), resolved, bridge)
        .context("building native BF16 Qwen3-MoE model")
}

#[allow(clippy::too_many_arguments)]
pub fn qwen3_moe_fp8(
    model_spec: &ModelSpec,
    attn_tp_size: u16,
    ep_size: u16,
    hp_size: u16,
    nvl_num_gpu: u16,
    routing_kind: RoutingKind,
    routing_seed: Option<u64>,
    expert_popularity_file: Option<&str>,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Qwen3MoeFp8DpAttnEpFfnModel> {
    let model_cfg = moe_model_cfg(model_spec)?;
    let routing = resolve_routing_source(
        routing_kind,
        routing_seed,
        model_cfg.num_experts.get(),
        ep_size,
        model_cfg.num_layers,
        model_cfg.top_k,
        expert_popularity_file,
    )?;
    let parallel = Qwen3MoeFp8Parallel {
        attn_tp_size,
        ep_size,
        hp_size,
        nvl_num_gpu,
        gpu_name: gpu.to_string(),
    };
    let resolved = qwen3_moe_fp8_dp_attn_ep_ffn::resolve_configs(
        &qwen3_moe_fp8_dp_attn_ep_ffn::build_configs(&model_cfg, &parallel, &routing),
    );
    qwen3_moe_fp8_dp_attn_ep_ffn::build(name.to_string(), resolved, bridge)
        .context("building native FP8 Qwen3-MoE model")
}

/// Build the vLLM-alignment Qwen recipe while sharing model dimensions and
/// partitioning with the generic Qwen arch.
#[allow(clippy::too_many_arguments)]
pub fn qwen3_vllm_moe(
    model_spec: &ModelSpec,
    attn_tp_size: u16,
    ep_size: u16,
    hp_size: u16,
    nvl_num_gpu: u16,
    routing_kind: RoutingKind,
    routing_seed: Option<u64>,
    expert_popularity_file: Option<&str>,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Qwen3VllmMoeDpAttnEpFfnModel> {
    let model_cfg = moe_model_cfg(model_spec)?;
    let routing = resolve_routing_source(
        routing_kind,
        routing_seed,
        model_cfg.num_experts.get(),
        ep_size,
        model_cfg.num_layers,
        model_cfg.top_k,
        expert_popularity_file,
    )?;
    let parallel = Qwen3VllmMoeParallel {
        attn_tp_size,
        ep_size,
        hp_size,
        nvl_num_gpu,
        gpu_name: gpu.to_string(),
    };
    let resolved = qwen3_vllm_moe_dp_attn_ep_ffn::resolve_configs(
        &qwen3_vllm_moe_dp_attn_ep_ffn::build_configs(&model_cfg, &parallel, &routing),
    );
    qwen3_vllm_moe_dp_attn_ep_ffn::build(name.to_string(), resolved, bridge)
        .context("building vLLM-aligned FP8 Qwen3-MoE model")
}

/// Build the GLM-5.2 local-attention + EP-MoE model. Both offline timing
/// prediction and the unified `hp_unified` deployment consume this concrete
/// path.
#[allow(clippy::too_many_arguments)]
pub fn glm52_dsa_moe(
    model_spec: &ModelSpec,
    ep_size: u16,
    nvl_num_gpu: u16,
    routing_kind: RoutingKind,
    routing_seed: Option<u64>,
    mtp_mode: Glm52MtpMode,
    expert_popularity_file: Option<&str>,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Glm52DsaMoeModel> {
    let model_cfg = glm52_model_cfg(model_spec).context("loading exact GLM-5.2 model config")?;
    let routing = resolve_routing_source(
        routing_kind,
        routing_seed,
        model_cfg.num_experts.get(),
        ep_size,
        // GLM's first three decoder layers are dense, so the MoE-layer count a
        // popularity profile is keyed by is NOT `num_layers` (unlike Qwen,
        // where every layer is sparse).
        num_sparse_layers(&model_cfg),
        model_cfg.router_top_k,
        expert_popularity_file,
    )?;
    let parallel = Glm52DsaMoeParallel {
        ep_size,
        nvl_num_gpu,
        gpu_name: gpu.to_string(),
    };
    let configs =
        glm52_dsa_moe::build_configs(&model_cfg, &parallel, &routing, model_spec.fp8, mtp_mode)
            .context("expanding GLM-5.2 architecture configs")?;
    let resolved = glm52_dsa_moe::resolve_configs(&configs);
    glm52_dsa_moe::build(name.to_string(), resolved, bridge)
        .context("building GLM-5.2 DSA-MoE model (often a missing profile.db row)")
}

/// Build the GLM-5.2 model in vLLM kernel granularity. Same topology and
/// parameters as [`glm52_dsa_moe`]; only the leaf cuts differ, so the two share
/// the checkpoint identity and the layer-override refusal.
#[allow(clippy::too_many_arguments)]
pub fn glm52_vllm_dsa_moe(
    model_spec: &ModelSpec,
    ep_size: u16,
    nvl_num_gpu: u16,
    routing_kind: RoutingKind,
    routing_seed: Option<u64>,
    mtp_mode: Glm52MtpMode,
    expert_popularity_file: Option<&str>,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Glm52VllmDsaMoeModel> {
    let model_cfg = glm52_model_cfg(model_spec).context("loading exact GLM-5.2 model config")?;
    let routing = resolve_routing_source(
        routing_kind,
        routing_seed,
        model_cfg.num_experts.get(),
        ep_size,
        // GLM's first three decoder layers are dense, so the MoE-layer count a
        // popularity profile is keyed by is NOT `num_layers` (unlike Qwen,
        // where every layer is sparse).
        num_sparse_layers(&model_cfg),
        model_cfg.router_top_k,
        expert_popularity_file,
    )?;
    let parallel = Glm52VllmDsaMoeParallel {
        ep_size,
        nvl_num_gpu,
        gpu_name: gpu.to_string(),
    };
    let configs = glm52_vllm_dsa_moe::build_configs(
        &model_cfg,
        &parallel,
        &routing,
        model_spec.fp8,
        mtp_mode,
    )
    .context("expanding vLLM-granularity GLM-5.2 architecture configs")?;
    let resolved = glm52_vllm_dsa_moe::resolve_configs(&configs);
    glm52_vllm_dsa_moe::build(name.to_string(), resolved, bridge)
        .context("building vLLM-granularity GLM-5.2 model (often a missing profile.db row)")
}

/// Build the AFD attn-side (layer-wise) Qwen3-MoE model — attention only, for ONE
/// DP shard (`attn_tp_size` head-parallel ranks). The attn pool runs one of these
/// per DP shard (its `replicas`). Pairs with [`qwen3_ffn_moe`].
pub fn qwen3_attn(
    model_spec: &ModelSpec,
    attn_tp_size: u16,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Qwen3AttnLayerwiseModel> {
    let model_cfg = moe_model_cfg(model_spec)?;
    let parallel = Qwen3AttnParallel {
        attn_tp_size,
        gpu_name: gpu.to_string(),
    };
    let resolved = qwen3_attn_layerwise::resolve_configs(&qwen3_attn_layerwise::build_configs(
        &model_cfg, &parallel,
    ));
    qwen3_attn_layerwise::build(name.to_string(), resolved, bridge)
        .context("building Qwen3 AFD attn-side model (often a missing profile.db row)")
}

/// Build the AFD ffn-side (layer-wise) Qwen3-MoE model — qkv / o_proj / router /
/// EP MoE / embed / lm_head. Reuses the iter-wise arch's `build_configs` +
/// `resolve_configs` (so the split conserves every leaf). Pairs with [`qwen3_attn`].
#[allow(clippy::too_many_arguments)]
pub fn qwen3_ffn_moe(
    model_spec: &ModelSpec,
    attn_tp_size: u16,
    ep_size: u16,
    nvl_num_gpu: u16,
    routing_kind: RoutingKind,
    routing_seed: Option<u64>,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Qwen3FfnMoeLayerwiseModel> {
    let model_cfg = moe_model_cfg(model_spec)?;
    let routing = resolve_routing(routing_kind, routing_seed, model_cfg.num_experts.get());
    let parallel = Qwen3FfnMoeParallel {
        attn_tp_size,
        ep_size,
        nvl_num_gpu,
        gpu_name: gpu.to_string(),
    };
    let resolved = qwen3_ffn_moe_layerwise::resolve_configs(
        &qwen3_ffn_moe_layerwise::build_configs(&model_cfg, &parallel, &routing),
    );
    qwen3_ffn_moe_layerwise::build(name.to_string(), resolved, bridge)
        .context("building Qwen3 AFD ffn-side model (often a missing profile.db row)")
}

#[allow(clippy::too_many_arguments)]
pub fn qwen3_fp8_ffn_moe(
    model_spec: &ModelSpec,
    attn_tp_size: u16,
    ep_size: u16,
    nvl_num_gpu: u16,
    routing_kind: RoutingKind,
    routing_seed: Option<u64>,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Qwen3Fp8FfnMoeLayerwiseModel> {
    let model_cfg = moe_model_cfg(model_spec)?;
    let routing = resolve_routing(routing_kind, routing_seed, model_cfg.num_experts.get());
    let parallel = Qwen3Fp8FfnMoeParallel {
        attn_tp_size,
        ep_size,
        nvl_num_gpu,
        gpu_name: gpu.to_string(),
    };
    let resolved = qwen3_fp8_ffn_moe_layerwise::resolve_configs(
        &qwen3_fp8_ffn_moe_layerwise::build_configs(&model_cfg, &parallel, &routing),
    );
    qwen3_fp8_ffn_moe_layerwise::build(name.to_string(), resolved, bridge)
        .context("building native FP8 Qwen3 AFD ffn-side model")
}

/// Build ONE iter-wise arch model from its selector, boxed as `dyn`. The
/// model-only seam the offline `timing-predict` (iter arch) path uses (it evaluates
/// [`IterwiseUnifiedModel`] directly, no worker/flow). The deployments do NOT box
/// — they call the concrete `dense` / `dense_tp` / … builders above to keep their
/// worker factories monomorphized (L4 §4.1). This single match is the only place
/// the selector tag picks a builder.
pub fn build_iter_model(
    sel: &IterArchSel,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Box<dyn IterwiseUnifiedModel>> {
    Ok(match sel {
        IterArchSel::Qwen36Local {
            model,
            routing,
            routing_seed,
            expert_popularity_file,
        } => Box::new(qwen36_local(
            model,
            *routing,
            *routing_seed,
            expert_popularity_file.as_deref(),
            gpu,
            name,
            bridge,
        )?),
        IterArchSel::Llama3Dense { model } => Box::new(dense(model, gpu, name, bridge)?),
        IterArchSel::Llama3DenseTp { model, tp_size } => {
            Box::new(dense_tp(model, *tp_size, gpu, name, bridge)?)
        }
        IterArchSel::Llama3DpAttnTpFfn {
            model,
            attn_tp_size,
            ffn_tp_size,
        } => Box::new(dp_attn_tp_ffn(
            model,
            *attn_tp_size,
            *ffn_tp_size,
            gpu,
            name,
            bridge,
        )?),
        IterArchSel::Qwen3MoeDpAttnEpFfn {
            model,
            attn_tp_size,
            ep_size,
            hp_size,
            nvl_num_gpu,
            routing,
            routing_seed,
            expert_popularity_file,
        } => Box::new(qwen3_moe(
            model,
            *attn_tp_size,
            *ep_size,
            *hp_size,
            *nvl_num_gpu,
            *routing,
            *routing_seed,
            expert_popularity_file.as_deref(),
            gpu,
            name,
            bridge,
        )?),
        IterArchSel::Qwen3MoeFp8DpAttnEpFfn {
            model,
            attn_tp_size,
            ep_size,
            hp_size,
            nvl_num_gpu,
            routing,
            routing_seed,
            expert_popularity_file,
        } => Box::new(qwen3_moe_fp8(
            model,
            *attn_tp_size,
            *ep_size,
            *hp_size,
            *nvl_num_gpu,
            *routing,
            *routing_seed,
            expert_popularity_file.as_deref(),
            gpu,
            name,
            bridge,
        )?),
        IterArchSel::Qwen3VllmMoeDpAttnEpFfn {
            model,
            attn_tp_size,
            ep_size,
            hp_size,
            nvl_num_gpu,
            routing,
            routing_seed,
            expert_popularity_file,
        } => Box::new(qwen3_vllm_moe(
            model,
            *attn_tp_size,
            *ep_size,
            *hp_size,
            *nvl_num_gpu,
            *routing,
            *routing_seed,
            expert_popularity_file.as_deref(),
            gpu,
            name,
            bridge,
        )?),
        IterArchSel::Glm52DsaMoe {
            model,
            ep_size,
            nvl_num_gpu,
            routing,
            routing_seed,
            mtp_mode,
            expert_popularity_file,
        } => Box::new(glm52_dsa_moe(
            model,
            *ep_size,
            *nvl_num_gpu,
            *routing,
            *routing_seed,
            *mtp_mode,
            expert_popularity_file.as_deref(),
            gpu,
            name,
            bridge,
        )?),
        IterArchSel::Glm52VllmDsaMoe {
            model,
            ep_size,
            nvl_num_gpu,
            routing,
            routing_seed,
            mtp_mode,
            expert_popularity_file,
        } => Box::new(glm52_vllm_dsa_moe(
            model,
            *ep_size,
            *nvl_num_gpu,
            *routing,
            *routing_seed,
            *mtp_mode,
            expert_popularity_file.as_deref(),
            gpu,
            name,
            bridge,
        )?),
    })
}

/// Build ONE AFD attn-side model from its selector, boxed as `dyn` — the
/// [`build_iter_model`] counterpart for the attn arch. The model-only seam the
/// offline `timing-predict` (attn arch) path uses: it drives [`AttnLayerwiseModel`]
/// directly, no worker/flow. The `afd` deployment does NOT box — it calls the
/// concrete [`qwen3_attn`] builder to keep its worker factory monomorphized. Only
/// the qwen3 arch has a layer-wise predict path; the llama3 attn variant bails
/// (mirrors `AfdDeployment`). Returning `Box<dyn>` (not `impl`) is what lets a
/// second buildable attn arch land as one more match arm without a signature break.
pub fn build_attn_model(
    sel: &AttnArchSel,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Box<dyn AttnLayerwiseModel>> {
    Ok(match sel {
        AttnArchSel::Qwen3AttnTp {
            model,
            attn_tp_size,
        } => Box::new(qwen3_attn(model, *attn_tp_size, gpu, name, bridge)?),
        AttnArchSel::Llama3AttnTp { .. } => bail!(
            "timing-predict attn: only the qwen3_attn_tp arch has a layer-wise \
             predict path (got llama3_attn_tp)"
        ),
    })
}

/// Build ONE AFD ffn-side model from its selector, boxed as `dyn` — the ffn
/// counterpart to [`build_attn_model`]. Only the qwen3 arch has a layer-wise
/// predict path; the deepseek ffn variant bails (mirrors `AfdDeployment`).
pub fn build_ffn_model(
    sel: &FfnArchSel,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Box<dyn FfnLayerwiseModel>> {
    Ok(match sel {
        FfnArchSel::Qwen3FfnMoe {
            model,
            attn_tp_size,
            ep_size,
            nvl_num_gpu,
            routing,
            routing_seed,
        } => Box::new(qwen3_ffn_moe(
            model,
            *attn_tp_size,
            *ep_size,
            *nvl_num_gpu,
            *routing,
            *routing_seed,
            gpu,
            name,
            bridge,
        )?),
        FfnArchSel::Qwen3Fp8FfnMoe {
            model,
            attn_tp_size,
            ep_size,
            nvl_num_gpu,
            routing,
            routing_seed,
        } => Box::new(qwen3_fp8_ffn_moe(
            model,
            *attn_tp_size,
            *ep_size,
            *nvl_num_gpu,
            *routing,
            *routing_seed,
            gpu,
            name,
            bridge,
        )?),
        FfnArchSel::DeepseekFfnMoe { .. } => bail!(
            "timing-predict ffn: only qwen3_ffn_moe and qwen3_fp8_ffn_moe have \
             layer-wise predict paths (got deepseek_ffn_moe)"
        ),
    })
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    #[test]
    fn qwen36_local_uniform_builder_loads_the_pinned_nested_config() {
        let model = ModelSpec {
            model_config: Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("model/config/qwen3_6_35b_a3b_fp8.json")
                .to_string_lossy()
                .into_owned(),
            num_layers: None,
            sim_num_layers: None,
            fp8: true,
        };
        let selector = IterArchSel::Qwen36Local {
            model,
            routing: RoutingKind::Uniform,
            routing_seed: None,
            expert_popularity_file: None,
        };
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let built = build_iter_model(&selector, "NVIDIA H200", "test", &bridge)
            .expect("build exact local Qwen3.6 model");
        assert_eq!(built.gpus_per_replica(), 1);
        assert_eq!(built.total_kv_bytes_per_token(), 20_480);
        assert_eq!(built.cost_log_manifest().slots.len(), 75);
    }

    #[test]
    fn expert_popularity_profile_replaces_uniform_routing_and_preserves_ppm_sum() {
        let mut profile_file = tempfile::NamedTempFile::new().unwrap();
        write!(
            profile_file,
            r#"{{
                "schema_version": 1,
                "num_logical_experts": 4,
                "probabilities_all_layers": [0.7, 0.2, 0.09, 0.01],
                "counts_all_layers": [70, 20, 9, 1]
            }}"#
        )
        .unwrap();
        let profile_path = profile_file.path().to_str().unwrap();
        let routing =
            resolve_routing_source(RoutingKind::Uniform, None, 4, 2, 2, 2, Some(profile_path))
                .unwrap();
        assert_eq!(routing.num_experts(), 4);
        assert_eq!(
            routing.ppm().iter().sum::<u32>(),
            RoutingDistribution::TOTAL_PPM
        );
        assert!(routing.ppm()[0] > routing.ppm()[1]);
        assert!(resolve_routing_source(
            RoutingKind::Random,
            Some(7),
            4,
            2,
            2,
            2,
            Some(profile_path),
        )
        .is_err());
    }

    #[test]
    fn layerwise_expert_popularity_sorts_ep_ranks_and_experts_before_adding() {
        let mut profile_file = tempfile::NamedTempFile::new().unwrap();
        write!(
            profile_file,
            r#"{{
                "schema_version": 1,
                "num_logical_experts": 4,
                "counts_by_layer": [
                    [9, 2, 5, 5],
                    [2, 2, 0, 8]
                ],
                "counts_all_layers": [11, 4, 5, 13]
            }}"#
        )
        .unwrap();

        let routing = resolve_routing_source(
            RoutingKind::Uniform,
            None,
            4,
            2,
            2,
            2,
            profile_file.path().to_str(),
        )
        .unwrap();

        // Layer 0 canonicalizes to [[9, 2], [5, 5]], layer 1 to
        // [[8, 0], [2, 2]], and equal slots add to [17, 2, 7, 7]. The first
        // canonical rank therefore carries sum(layer-wise max rank) = 19.
        assert_eq!(routing.ppm(), &[515_152, 60_606, 212_121, 212_121]);
        assert_eq!(routing.ppm()[..2].iter().sum::<u32>(), 575_758);
    }

    #[test]
    fn expert_popularity_v2_validates_full_model_and_partition_contract() {
        let profile = serde_json::json!({
            "schema_version": 2,
            "model": "Qwen/test",
            "num_moe_layers": 2,
            "num_logical_experts": 4,
            "expert_parallel_size": 2,
            "experts_per_rank": 2,
            "experts_per_token": 2,
            "count_semantics": "logical_routed_token_assignments",
            "aggregation": {
                "scope": "all_captured_eplb_steps",
                "observed_eplb_step_min": 10,
                "observed_eplb_step_max": 11,
                "record_count": 2
            },
            "expert_partitioning": {
                "kind": "contiguous_logical_expert_ids",
                "layout": "rank_major"
            },
            "counts_by_layer": [[9, 2, 5, 5], [2, 2, 0, 8]],
            "probabilities_by_layer": [
                [9.0 / 21.0, 2.0 / 21.0, 5.0 / 21.0, 5.0 / 21.0],
                [2.0 / 12.0, 2.0 / 12.0, 0.0, 8.0 / 12.0]
            ],
            "counts_all_layers": [11, 4, 5, 13],
            "probabilities_all_layers": [11.0 / 33.0, 4.0 / 33.0, 5.0 / 33.0, 13.0 / 33.0]
        });
        let mut profile_file = tempfile::NamedTempFile::new().unwrap();
        write!(profile_file, "{profile}").unwrap();
        let profile_path = profile_file.path().to_str().unwrap();

        let routing =
            resolve_routing_source(RoutingKind::Uniform, None, 4, 2, 2, 2, Some(profile_path))
                .unwrap();
        assert_eq!(routing.ppm(), &[515_152, 60_606, 212_121, 212_121]);

        let mut unknown_field_profile = profile;
        unknown_field_profile["unregulated"] = serde_json::json!(true);
        let mut invalid_file = tempfile::NamedTempFile::new().unwrap();
        write!(invalid_file, "{unknown_field_profile}").unwrap();
        let error = resolve_routing_source(
            RoutingKind::Uniform,
            None,
            4,
            2,
            2,
            2,
            invalid_file.path().to_str(),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("unknown field"));
    }

    fn glm52_model_spec() -> ModelSpec {
        ModelSpec {
            model_config: Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("model/config/glm52.json")
                .to_string_lossy()
                .into_owned(),
            num_layers: None,
            sim_num_layers: None,
            fp8: false,
        }
    }

    #[test]
    fn glm52_sparse_layer_count_matches_the_checkpoint_schedule() {
        // A popularity profile is keyed by MoE layers, not decoder layers.
        // GLM's first three are dense, so these must differ by exactly three.
        let model = glm52_model_cfg(&glm52_model_spec()).unwrap();
        assert_eq!(model.num_layers, 78);
        assert_eq!(num_sparse_layers(&model), 75);
    }

    #[test]
    fn glm52_model_spec_loads_exact_identity_and_rejects_layer_overrides() {
        let model = glm52_model_cfg(&glm52_model_spec()).unwrap();
        assert_eq!(model.num_layers, 78);
        assert_eq!(model.num_experts.get(), 256);
        assert_eq!(model.full_index_layers.len(), 21);

        let mut fp8 = glm52_model_spec();
        fp8.fp8 = true;
        assert_eq!(glm52_model_cfg(&fp8).unwrap().dtype, model.dtype);
        for (num_layers, sim_num_layers) in [(Some(78), None), (None, Some(78))] {
            let mut overridden = glm52_model_spec();
            overridden.num_layers = num_layers;
            overridden.sim_num_layers = sim_num_layers;
            assert!(glm52_model_cfg(&overridden)
                .unwrap_err()
                .to_string()
                .contains("rejects num_layers/sim_num_layers"));
        }
    }

    #[test]
    fn glm52_selector_values_resolve_to_exact_parallel_routing_and_mtp_configs() {
        let model = glm52_model_cfg(&glm52_model_spec()).unwrap();
        for mode in [
            Glm52MtpMode::Off,
            Glm52MtpMode::FullIndex,
            Glm52MtpMode::IndexShare,
        ] {
            let parallel = Glm52DsaMoeParallel {
                ep_size: 16,
                nvl_num_gpu: 8,
                gpu_name: "NVIDIA H200".to_string(),
            };
            let routing = resolve_routing(RoutingKind::Random, Some(19), 256);
            let configs =
                glm52_dsa_moe::build_configs(&model, &parallel, &routing, false, mode).unwrap();
            assert_eq!(configs.parallel.ep_size, 16);
            assert_eq!(configs.parallel.nvl_num_gpu, 8);
            assert_eq!(configs.moe_dispatch.routing, routing);
            assert_eq!(configs.mtp_mode, mode);
            assert_eq!(configs.mtp_attention.is_some(), mode != Glm52MtpMode::Off);
            assert_eq!(
                configs
                    .mtp_attention
                    .as_ref()
                    .map(|attention| attention.include_indexer),
                match mode {
                    Glm52MtpMode::Off => None,
                    Glm52MtpMode::FullIndex => Some(true),
                    Glm52MtpMode::IndexShare => Some(false),
                }
            );
        }
    }

    /// The moesim-faithful comm sizes (`ref/moesim-rs/.../standard_moe.rs`): attn→ffn
    /// is the attention output `q_dim·bpe`; ffn→attn is the QKV projection
    /// `(q_dim + 2·kv_dim)·bpe`. Pure arithmetic on the model dims — no bridge.
    #[test]
    fn afd_comm_bytes_match_moesim_formulas() {
        let model = MoeModelCfg::qwen3_235b();
        let bpe = model.dtype.size_bytes() as u64;
        let q_dim = model.num_qo_heads.get() as u64 * model.head_dim.get() as u64;
        let kv_dim = model.num_kv_heads.get() as u64 * model.head_dim.get() as u64;

        let attn_cfgs = crate::arch::qwen3_attn_layerwise::build_configs(
            &model,
            &Qwen3AttnParallel {
                attn_tp_size: 4,
                gpu_name: "H200".to_string(),
            },
        );
        // attn→ffn outgoing bytes: the attention output, q_dim·bpe.
        assert_eq!(
            attn_cfgs.attn_to_ffn_bytes_per_token.get() as u64,
            q_dim * bpe
        );
        // total KV bytes: 2 (k+v) × kv_heads × head_dim × kv_dtype × layers.
        assert_eq!(
            attn_cfgs.total_kv_bytes_per_token.get() as u64,
            2 * model.num_kv_heads.get() as u64
                * model.head_dim.get() as u64
                * model.kv_dtype.size_bytes() as u64
                * model.num_layers as u64
        );
        // ffn→attn outgoing bytes (QKV projection) is the symmetric `(q+2kv)·bpe`,
        // computed in `qwen3_ffn_moe_layerwise::build` from the same model dims.
        let _ffn_to_attn = (q_dim + 2 * kv_dim) * bpe;
    }

    /// FP8 AFD end-to-end config wiring: every GEMM/handoff/KV role goes fp8
    /// (`bpe = 1`, `deepgemm` backend, `compute_dtype = Fp8E4m3`) while the RMSNorm
    /// ops and the model's base `dtype` stay bf16 (`bpe = 2`). Mirrors ref's
    /// `bytes_per_element(p.fp8)` — the attn↔ffn handoffs and KV cache are all
    /// 1 byte/elem in fp8.
    #[test]
    fn fp8_afd_configs_are_one_byte_per_element_and_deepgemm() {
        use crate::timing::bridge::DType;
        use crate::timing::routing::RoutingDistribution;

        let model = MoeModelCfg::qwen3_235b().with_fp8(true);
        // Base dtype is untouched (bf16); only the derived compute/kv dtypes flip.
        assert_eq!(model.dtype, DType::Bf16);
        assert_eq!(model.compute_dtype(), DType::Fp8E4m3);
        assert_eq!(model.kv_dtype, DType::Fp8E4m3);
        assert_eq!(model.single_gemm_backends(), vec!["deepgemm"]);
        assert_eq!(model.grouped_gemm_backends(), vec!["deepgemm"]);

        let q_dim = model.num_qo_heads.get() as u64 * model.head_dim.get() as u64;
        let kv_dim = model.num_kv_heads.get() as u64 * model.head_dim.get() as u64;

        // --- attn side ---
        let attn_cfgs = crate::arch::qwen3_attn_layerwise::build_configs(
            &model,
            &Qwen3AttnParallel {
                attn_tp_size: 4,
                gpu_name: "H200".to_string(),
            },
        );
        // Handoff + KV at fp8 = 1 byte/elem.
        assert_eq!(
            attn_cfgs.attn_to_ffn_bytes_per_token.get() as u64,
            q_dim * 1
        );
        assert_eq!(
            attn_cfgs.total_kv_bytes_per_token.get() as u64,
            2 * model.num_kv_heads.get() as u64
                * model.head_dim.get() as u64
                * 1
                * model.num_layers as u64
        );
        assert_eq!(attn_cfgs.attn.dtype, DType::Bf16);
        assert!(attn_cfgs.attn.fp8);
        assert_eq!(attn_cfgs.attn.kv_dtype(), DType::Fp8E4m3);

        // --- ffn side ---
        let routing = RoutingDistribution::uniform(model.num_experts.get());
        let ffn_cfgs = crate::arch::qwen3_fp8_ffn_moe_layerwise::build_configs(
            &model,
            &crate::arch::qwen3_fp8_ffn_moe_layerwise::Qwen3Fp8FfnMoeParallel {
                attn_tp_size: 4,
                ep_size: 8,
                nvl_num_gpu: 8,
                gpu_name: "H200".to_string(),
            },
            &routing,
        );
        let ffn_resolved = crate::arch::qwen3_fp8_ffn_moe_layerwise::resolve_configs(&ffn_cfgs);
        // Symmetric QKV-projection handoff at fp8.
        assert_eq!(
            ffn_cfgs.ffn_to_attn_bytes_per_token.get() as u64,
            (q_dim + 2 * kv_dim) * 1
        );
        // GEMM roles fp8+deepgemm; RMSNorm roles stay bf16.
        assert_eq!(ffn_cfgs.pre_attn.activation_dtype, DType::Bf16);
        assert_eq!(ffn_cfgs.pre_attn.gemm_backends, vec!["deepgemm"]);
        assert_eq!(ffn_resolved.pre_attn.qkv.gemm.dtype, DType::Fp8E4m3);
        assert_eq!(ffn_cfgs.post_attn.activation_dtype, DType::Bf16);
        assert_eq!(ffn_resolved.post_attn.o_proj.gemm.dtype, DType::Fp8E4m3);
        assert_eq!(ffn_cfgs.moe_expert_compute.dtype, DType::Fp8E4m3); // no norm inside
        assert_eq!(
            ffn_cfgs.moe_expert_compute.grouped_gemm_backends,
            vec!["deepgemm"]
        );
        assert_eq!(ffn_cfgs.lm_head.gemm.dtype, DType::Fp8E4m3);
        assert_eq!(ffn_cfgs.lm_head.gemm.backends, vec!["deepgemm"]);
        assert_eq!(ffn_cfgs.lm_head.quant.hidden_size, 4096);
        assert_eq!(ffn_cfgs.final_norm.dtype, DType::Bf16); // final_norm stays bf16
                                                            // dispatch/combine ship the hidden activation fp8.
        assert_eq!(ffn_cfgs.moe_dispatch.dtype, DType::Fp8E4m3);
        assert_eq!(ffn_cfgs.moe_combine.dtype, DType::Fp8E4m3);
    }

    /// The bf16 negative: no fp8 anywhere — dense GEMMs choose the faster Torch
    /// layout, grouped GEMMs stay `torch`, and handoffs/KV are 2 bytes/elem.
    #[test]
    fn bf16_afd_configs_keep_torch_and_two_bytes_per_element() {
        use crate::timing::bridge::DType;

        let model = MoeModelCfg::qwen3_235b(); // fp8: false
        assert_eq!(model.compute_dtype(), DType::Bf16);
        assert_eq!(model.single_gemm_backends(), vec!["torch", "torch_linear"]);
        assert_eq!(model.grouped_gemm_backends(), vec!["torch"]);

        let q_dim = model.num_qo_heads.get() as u64 * model.head_dim.get() as u64;
        let attn_cfgs = crate::arch::qwen3_attn_layerwise::build_configs(
            &model,
            &Qwen3AttnParallel {
                attn_tp_size: 4,
                gpu_name: "H200".to_string(),
            },
        );
        assert_eq!(
            attn_cfgs.attn_to_ffn_bytes_per_token.get() as u64,
            q_dim * 2
        );
        assert!(!attn_cfgs.attn.fp8);
        assert_eq!(attn_cfgs.attn.kv_dtype(), DType::Bf16);
    }
}
