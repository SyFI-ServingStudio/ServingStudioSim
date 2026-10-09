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

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;

use crate::arch::config::{AttnArchSel, FfnArchSel, IterArchSel, ModelSpec, RoutingKind};
use crate::arch::contract::SpeculativeUnifiedModel;
use crate::arch::model_cfg::ModelCfg;
use crate::arch::moe_model_cfg::MoeModelCfg;
use crate::arch::{
    AttnLayerwiseModel, DeepseekV4ModelCfg, DeepseekV4VllmModel, DeepseekV4VllmParallel,
    DeepseekV41ModelCfg, DeepseekV41VllmModel, DeepseekV41VllmParallel, DenseParallel,
    DenseTpParallel, Dflash2DraftResolved, DpAttnTpFfnParallel, FfnLayerwiseModel, Glm52ModelCfg,
    Glm52MtpMode, Glm52SglangNvfp4TpDsaMoeModel, Glm52SglangNvfp4TpDsaMoeParallel,
    Glm52VllmDsaMoeModel, Glm52VllmDsaMoeParallel, Glm52VllmNvfp4DsaMoeModel,
    Glm52VllmNvfp4DsaMoeParallel, Glm52VllmNvfp4DsaMoeSpeculativeModel, Glm53FlashModelCfg,
    Glm53FlashVllmModel, Glm53FlashVllmParallel, Glm53VllmNvfp4DsaMoeDflash2Model,
    IterwiseUnifiedModel, Llama3DenseModel, Llama3DenseTpModel, Llama3DpAttnTpFfnModel,
    Qwen3AttnLayerwiseModel, Qwen3AttnParallel, Qwen3FfnMoeLayerwiseModel, Qwen3FfnMoeParallel,
    Qwen3Fp8FfnMoeLayerwiseModel, Qwen3Fp8FfnMoeParallel, Qwen3MoeDpAttnEpFfnModel,
    Qwen3MoeFp8DpAttnEpFfnModel, Qwen3MoeFp8Parallel, Qwen3MoeParallel,
    Qwen3VllmMoeDpAttnEpFfnModel, Qwen3VllmMoeParallel, Qwen36LocalModel, Qwen36LocalParallel,
    Qwen36ModelCfg, deepseek_v4_vllm, deepseek_v41_vllm, glm52_sglang_nvfp4_tp_dsa_moe,
    glm52_vllm_dsa_moe, glm52_vllm_nvfp4_dsa_moe, glm53_flash_vllm_fp8_kda_dsa_moe,
    glm53_vllm_nvfp4_dsa_moe_dflash2, llama3_dense, llama3_dense_tp, llama3_dp_attn_tp_ffn,
    qwen3_attn_layerwise, qwen3_ffn_moe_layerwise, qwen3_fp8_ffn_moe_layerwise,
    qwen3_moe_dp_attn_ep_ffn, qwen3_moe_fp8_dp_attn_ep_ffn, qwen3_vllm_moe_dp_attn_ep_ffn,
    qwen36_local,
};
use crate::common::Fabric;
use crate::timing::ExpertDemand;
use crate::timing::bridge::DType;
use crate::timing::kernels::AllReduceKernelConfig;
use crate::timing::routing::RoutingDistribution;
use crate::timing::{Dim, PerfApiBridge};
use crate::worklet::{
    Dflash2ContextKvLocalWorklet, Dflash2ContextKvLocalWorkletConfig, Dflash2DraftAttnLocalWorklet,
    Dflash2DraftFfnLocalWorklet, Dflash2DraftLayerLocalWorkletConfig, Dflash2SelectorLocalWorklet,
    Dflash2SelectorLocalWorkletConfig,
};

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
            "GLM-5.2 architecture rejects num_layers/sim_num_layers overrides; the exact heterogeneous 78-layer schedule is required"
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
) -> Result<RoutingDistribution> {
    Ok(match kind {
        RoutingKind::Uniform => RoutingDistribution::uniform(num_experts),
        RoutingKind::Random => RoutingDistribution::random(num_experts, seed.unwrap_or(0)),
        RoutingKind::Popularity => anyhow::bail!(
            "routing=popularity requires expert_popularity_file on an arch that supports it"
        ),
        RoutingKind::Corpus => {
            anyhow::bail!("routing=corpus requires token_corpus_file on an arch that supports it")
        }
    })
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

/// Schemas v2 and v3 share the measured tensors but have distinct closed
/// aggregation metadata. Cross-field dimensions are validated after serde
/// because JSON Schema cannot express them all.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpertPopularityProfile {
    schema_version: u32,
    model: String,
    #[serde(default)]
    model_role: Option<String>,
    num_moe_layers: u32,
    num_logical_experts: u32,
    /// The EP the capture ran at, kept as provenance. The counts are of logical
    /// experts, so a run at any EP partitions them by its own `ep_size`.
    expert_parallel_size: u16,
    experts_per_rank: u32,
    experts_per_token: u32,
    count_semantics: String,
    aggregation: ExpertPopularityAggregation,
    expert_partitioning: ExpertPopularityPartitioning,
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
struct ExpertPopularityAggregationV3 {
    scope: String,
    observed_eplb_step_min: u64,
    observed_eplb_step_max: u64,
    /// Optional provenance added by newer recorders. It does not change the
    /// routed counts; older v3 artifacts remain valid without it.
    #[serde(default, rename = "observed_forward_pass_count")]
    _observed_forward_pass_count: Option<u64>,
    record_count: u64,
    raw_record_count: u64,
    discarded_oversized_record_count: u64,
    discarded_oversized_eplb_steps: Vec<u64>,
    max_tokens_per_step: u64,
    /// Number of leading dense layers removed by a recorder that reports every
    /// model layer. Optional because older v3 artifacts already stored only MoE
    /// layers and therefore had nothing to declare.
    #[serde(default, rename = "dense_prefix_layers")]
    _dense_prefix_layers: Option<u32>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ExpertPopularityAggregation {
    V2(ExpertPopularityAggregationV2),
    V3(ExpertPopularityAggregationV3),
    V4(ExpertPopularityAggregationV4),
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpertPopularityAggregationV4 {
    scope: String,
    observed_eplb_step_min: u64,
    observed_eplb_step_max: u64,
    record_count: u64,
    raw_record_count: u64,
    discarded_oversized_record_count: u64,
    discarded_oversized_eplb_steps: Vec<u64>,
    discarded_outside_replay_window_record_count: u64,
    max_tokens_per_step: u64,
    max_forwards_per_step: u64,
    replay_start_monotonic_ns: u64,
    replay_end_monotonic_ns: u64,
    observed_monotonic_ns_min: u64,
    observed_monotonic_ns_max: u64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpertPopularityPartitioning {
    kind: String,
    layout: String,
}

#[derive(Debug, Deserialize)]
struct ExpertPopularityVersion {
    schema_version: u32,
}

/// Canonicalize each layer's EP load while retaining the layer boundary.
///
/// Expert ids and EP rank ids are irrelevant to local expert-compute cost. For
/// every layer, sort experts within each physical rank by descending load, then
/// sort ranks by descending total load. A later routing sampler can then realize
/// every layer independently before folding equal canonical slots. Ties use the
/// sorted expert vector, keeping the transformation deterministic without
/// restoring physical ids.
fn canonicalize_layerwise_expert_counts(
    counts_by_layer: &[Vec<u64>],
    expected_num_experts: u32,
    ep_size: u16,
    path: &str,
) -> Result<Vec<Vec<f32>>> {
    anyhow::ensure!(ep_size > 0, "ep_size must be non-zero");
    anyhow::ensure!(
        expected_num_experts % u32::from(ep_size) == 0,
        "expert popularity profile {} cannot partition {} experts across ep_size {}",
        path,
        expected_num_experts,
        ep_size
    );
    let experts_per_rank = expected_num_experts as usize / usize::from(ep_size);
    let mut canonical_layers = Vec::with_capacity(counts_by_layer.len());

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

        let mut canonical_counts = vec![0u64; expected_num_experts as usize];
        for (canonical_rank, (_, sorted_expert_counts)) in rank_counts.iter().enumerate() {
            let canonical_start = canonical_rank * experts_per_rank;
            for (expert_slot, count) in sorted_expert_counts.iter().enumerate() {
                let canonical_slot = canonical_start + expert_slot;
                canonical_counts[canonical_slot] = *count;
            }
        }
        canonical_layers.push(
            canonical_counts
                .into_iter()
                .map(|count| count as f32)
                .collect(),
        );
    }
    Ok(canonical_layers)
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
        2 | 3 | 4 => {
            let profile: ExpertPopularityProfile = serde_json::from_value(profile_value)
                .with_context(|| {
                    format!(
                        "parsing strict expert popularity v{} profile {path}",
                        version.schema_version
                    )
                })?;
            validate_expert_popularity(
                &profile,
                expected_num_experts,
                expected_num_moe_layers,
                expected_experts_per_token,
                path,
            )?;
            (profile.num_logical_experts, profile.counts_by_layer, None)
        }
        other => bail!(
            "unsupported expert popularity schema_version {} in {} (supported: 1 legacy, 2, 3, 4)",
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

    let routing = if !counts_by_layer.is_empty() {
        let layers = canonicalize_layerwise_expert_counts(
            &counts_by_layer,
            expected_num_experts,
            ep_size,
            path,
        )?;
        anyhow::ensure!(
            layers.iter().flatten().any(|ratio| *ratio > 0.0),
            "expert popularity profile {} has zero total routing mass",
            path
        );
        RoutingDistribution::from_layer_profiles(&layers)
    } else if let Some(ratios) = legacy_ratios {
        anyhow::ensure!(
            ratios.iter().any(|ratio| *ratio > 0.0),
            "expert popularity profile {} has zero total routing mass",
            path
        );
        RoutingDistribution::from_profile(&ratios)
    } else {
        bail!(
            "legacy expert popularity profile {} must contain counts_by_layer or {} probabilities_all_layers/counts_all_layers entries",
            path,
            expected_num_experts
        );
    };
    Ok(routing)
}

fn validate_expert_popularity(
    profile: &ExpertPopularityProfile,
    expected_num_experts: u32,
    expected_num_moe_layers: u32,
    expected_experts_per_token: u32,
    path: &str,
) -> Result<()> {
    anyhow::ensure!(
        matches!(profile.schema_version, 2 | 3 | 4),
        "internal expert popularity version mismatch"
    );
    anyhow::ensure!(
        if profile.schema_version == 4 {
            matches!(profile.model_role.as_deref(), Some("target" | "draft"))
        } else {
            profile.model_role.is_none()
        },
        "expert popularity profile {path} has invalid model_role for its schema"
    );
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
    match (&profile.aggregation, profile.schema_version) {
        (ExpertPopularityAggregation::V2(aggregation), 2) => anyhow::ensure!(
            aggregation.scope == "all_captured_eplb_steps"
                && aggregation.record_count > 0
                && aggregation.observed_eplb_step_min <= aggregation.observed_eplb_step_max,
            "expert popularity profile {path} has invalid v2 aggregation metadata"
        ),
        (ExpertPopularityAggregation::V3(aggregation), 3) => anyhow::ensure!(
            aggregation.scope == "captured_eplb_steps_within_token_ceiling"
                && aggregation.record_count > 0
                && aggregation.observed_eplb_step_min <= aggregation.observed_eplb_step_max
                && aggregation.max_tokens_per_step > 0
                && aggregation.raw_record_count
                    == aggregation.record_count + aggregation.discarded_oversized_record_count
                && aggregation.discarded_oversized_record_count
                    == aggregation.discarded_oversized_eplb_steps.len() as u64,
            "expert popularity profile {path} has invalid v3 aggregation metadata"
        ),
        (ExpertPopularityAggregation::V4(aggregation), 4) => anyhow::ensure!(
            aggregation.scope == "replay_window_within_role_specific_token_ceiling"
                && aggregation.record_count > 0
                && aggregation.observed_eplb_step_min <= aggregation.observed_eplb_step_max
                && aggregation.max_tokens_per_step > 0
                && aggregation.max_forwards_per_step > 0
                && (profile.model_role.as_deref() != Some("target")
                    || aggregation.max_forwards_per_step == 1)
                && aggregation.replay_start_monotonic_ns > 0
                && aggregation.replay_start_monotonic_ns <= aggregation.observed_monotonic_ns_min
                && aggregation.observed_monotonic_ns_min <= aggregation.observed_monotonic_ns_max
                && aggregation.observed_monotonic_ns_max <= aggregation.replay_end_monotonic_ns
                && u128::from(aggregation.raw_record_count)
                    == u128::from(aggregation.record_count)
                        + u128::from(aggregation.discarded_oversized_record_count)
                        + u128::from(aggregation.discarded_outside_replay_window_record_count)
                && aggregation.discarded_oversized_record_count
                    == aggregation.discarded_oversized_eplb_steps.len() as u64,
            "expert popularity profile {path} has invalid v4 aggregation metadata"
        ),
        _ => bail!(
            "expert popularity profile {path} schema_version does not match its aggregation shape"
        ),
    }
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
        let layer_total = counts.iter().try_fold(0u64, |sum, count| {
            sum.checked_add(*count).ok_or_else(|| {
                anyhow::anyhow!(
                    "expert popularity profile {path} layer {layer_index} count overflow"
                )
            })
        })?;
        anyhow::ensure!(
            expected_experts_per_token > 0
                && layer_total % u64::from(expected_experts_per_token) == 0,
            "expert popularity profile {path} layer {layer_index} assignment count is not divisible by top_k"
        );
        anyhow::ensure!(
            counts.iter().all(|count| {
                u128::from(*count) * u128::from(expected_experts_per_token)
                    <= u128::from(layer_total)
            }),
            "expert popularity profile {path} layer {layer_index} has infeasible distinct top-k marginals"
        );
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

/// One selector's answer to "where does this model's routed demand come from",
/// resolved per MoE callable.
///
/// The callables differ by which layers they cover and how wide a verify block
/// their batch carries. A token corpus answers both from one artifact — the
/// layers are a slice and the width is a sampling parameter. A marginal has
/// already summed the layer axis away, so every callable of a model folds the
/// same one; the MTP layer's own routing is measured through a corpus.
struct ExpertDemandSource<'a> {
    kind: RoutingKind,
    seed: Option<u64>,
    num_experts: u32,
    experts_per_token: u32,
    ep_size: u16,
    expert_popularity_file: Option<&'a str>,
    token_corpus_file: Option<&'a str>,
    /// The body's routed layers — the axis an expert-popularity profile is
    /// validated against, and the boundary past which one has no evidence.
    num_routed_layers: u32,
}

impl ExpertDemandSource<'_> {
    /// `layers` is the callable's slice of the model's routed layer axis and
    /// `group_size` the verify width its batch presents.
    fn demand(&self, layers: std::ops::Range<usize>, group_size: u32) -> Result<ExpertDemand> {
        if self.kind == RoutingKind::Corpus {
            // A preset migrated from `popularity` keeps costing the same
            // whichever field it left behind, so a leftover is rejected rather
            // than ignored. The mirror case is checked below.
            anyhow::ensure!(
                self.expert_popularity_file.is_none(),
                "expert_popularity_file cannot be combined with routing=corpus; \
                 omit the profile or use routing=popularity"
            );
            let path = self
                .token_corpus_file
                .context("routing=corpus requires token_corpus_file on an arch that supports it")?;
            let demand = ExpertDemand::corpus(path, group_size, layers)?;
            let ExpertDemand::Corpus(config) = &demand else {
                unreachable!("ExpertDemand::corpus returns the corpus arm")
            };
            anyhow::ensure!(
                config.num_experts == self.num_experts as usize
                    && config.top_k == self.experts_per_token as usize,
                "token corpus {path} records top-{} of {} experts; this model routes top-{} of {}",
                config.top_k,
                config.num_experts,
                self.experts_per_token,
                self.num_experts
            );
            // Checking the layer axis is what makes the other dimensions
            // trustworthy: the payload length constrains only their product, so
            // a manifest that trades tokens for layers keeps its bytes and its
            // checksum while sampling every other token. The axis is the body,
            // plus one slot when the capture ran a drafter -- whether this build
            // prices that slot is its own business, so a speculative capture
            // still serves a counterfactual run without drafting.
            let body = self.num_routed_layers as usize;
            anyhow::ensure!(
                config.num_layers == body || config.num_layers == body + 1,
                "token corpus {path} records {} layers; this model has {body} routed \
                 body layers, and a capture records those plus at most one MTP layer",
                config.num_layers,
            );
            return Ok(demand);
        }
        anyhow::ensure!(
            self.token_corpus_file.is_none(),
            "token_corpus_file cannot be combined with routing={:?}; omit it or use routing=corpus",
            self.kind
        );
        let routing = resolve_routing_source(
            self.kind,
            self.seed,
            self.num_experts,
            self.ep_size,
            self.num_routed_layers,
            self.experts_per_token,
            self.expert_popularity_file,
        )?;
        Ok(if layers.end <= self.num_routed_layers as usize {
            ExpertDemand::popularity(&routing, layers.len() as u32)
        } else {
            ExpertDemand::popularity_summed(&routing)
        })
    }
}

/// Resolve measured routing from a required profile, or synthetic uniform/random
/// routing without a profile. Invalid combinations fail rather than falling back.
#[allow(clippy::too_many_arguments)]
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
            kind == RoutingKind::Popularity,
            "expert_popularity_file cannot be combined with routing={:?}; omit the profile or use routing=popularity",
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
    resolve_routing(kind, seed, num_experts)
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

/// Fixed Llama3.1 semantics required by the measured Neuron decoder backend.
fn ensure_neuron_math(raw: &serde_json::Value) -> Result<()> {
    let rope = &raw["rope_scaling"];
    ensure!(
        raw["model_type"].as_str() == Some("llama")
            && raw["hidden_act"].as_str() == Some("silu")
            && raw["rms_norm_eps"].as_f64() == Some(1e-5)
            && raw["rope_theta"].as_f64() == Some(500000.0)
            && rope["rope_type"].as_str() == Some("llama3")
            && rope["factor"].as_f64() == Some(8.0)
            && rope["low_freq_factor"].as_f64() == Some(1.0)
            && rope["high_freq_factor"].as_f64() == Some(4.0)
            && rope["original_max_position_embeddings"].as_u64() == Some(8192)
            && !raw["attention_bias"].as_bool().unwrap_or(false)
            && !raw["mlp_bias"].as_bool().unwrap_or(false),
        "Trainium2 decoder requires the measured Llama3.1 RoPE, RMSNorm and unbiased SiLU math"
    );
    Ok(())
}

pub fn llama3_neuron(
    model_spec: &ModelSpec,
    kv_capacity: u32,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<crate::arch::Llama3NeuronModel> {
    ensure!(
        !model_spec.fp8,
        "Trainium2 Llama initially supports BF16 only"
    );
    ensure!(
        kv_capacity == 512,
        "Trainium2 Llama initially supports KV capacity512 only"
    );
    ensure!(
        matches!(gpu, "AWS Trainium2 LNC2" | "Trainium2-LNC2"),
        "Trainium2 Llama requires an LNC2 target"
    );
    let raw: serde_json::Value = serde_json::from_reader(File::open(&model_spec.model_config)?)?;
    ensure_neuron_math(&raw)?;
    let model_cfg = dense_model_cfg(model_spec)?;
    ensure!(
        (
            model_cfg.hidden.get(),
            model_cfg.intermediate.get(),
            model_cfg.num_qo_heads.get(),
            model_cfg.num_kv_heads.get(),
            model_cfg.head_dim.get(),
            model_cfg.vocab.get()
        ) == (4096, 14336, 32, 8, 128, 128256)
            && model_cfg.dtype == DType::Bf16
            && model_cfg.kv_dtype == DType::Bf16
            && (1..=32).contains(&model_cfg.num_layers),
        "Trainium2 Llama requires the BF16 Llama3.1 8B dimensions and 1..32 layers"
    );
    let parallel = crate::arch::NeuronParallel {
        gpu_name: gpu.to_string(),
        kv_capacity,
    };
    let configs = crate::arch::llama3_neuron::build_configs(&model_cfg, &parallel);
    let resolved = crate::arch::llama3_neuron::resolve_configs(&configs);
    crate::arch::llama3_neuron::build(name.to_string(), resolved, bridge)
        .context("building Trainium2 Llama model (check Neuron setup and profile.db rows)")
}

/// Pinned stock vLLM Neuron full-model executable (all layers and sampling).
pub fn llama3_vllm_neuron(
    model_spec: &ModelSpec,
    max_model_len: u32,
    decode_buckets: &[u32],
    tp_size: u16,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<crate::arch::Llama3VllmNeuronModel> {
    ensure!(
        !model_spec.fp8 && model_spec.num_layers.is_none() && model_spec.sim_num_layers.is_none(),
        "stock vLLM Neuron requires BF16 and the original model without layer overrides"
    );
    let raw: serde_json::Value = serde_json::from_reader(File::open(&model_spec.model_config)?)?;
    ensure_neuron_math(&raw)?;
    let model_cfg = dense_model_cfg(model_spec)?;
    crate::arch::llama3_vllm_neuron::validate_model(&model_cfg)?;
    let parallel = crate::arch::VllmNeuronParallel {
        gpu_name: gpu.to_string(),
        max_model_len,
        decode_buckets: decode_buckets.to_vec(),
        tp_size,
    };
    let configs = crate::arch::llama3_vllm_neuron::build_configs(&model_cfg, &parallel);
    crate::arch::llama3_vllm_neuron::validate_config(&configs.forward)?;
    let resolved = crate::arch::llama3_vllm_neuron::resolve_configs(&configs);
    crate::arch::llama3_vllm_neuron::build(name.to_string(), resolved, bridge).context(
        "building stock vLLM Neuron full forward (check pinned runtime and profile.db rows)",
    )
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

/// The one expert-parallel width the DeepSeek V4 selectors model, over one
/// NVLink domain of as many GPUs: the selectors take no `ep_size`, and routing
/// and the parallel layout both read it from here.
const DEEPSEEK_V4_EP_SIZE: u16 = 4;

/// Build DeepSeek V4 through the same routing-profile loader used by Qwen and
/// GLM. `serialize_streams` changes only CostTree composition; kernel inputs and
/// profile identities remain identical. `step_tokens` is the most tokens one
/// step runs: a chunked-prefill worker's chunk, or None when steps are not
/// chunked, where a whole prompt runs at once and vLLM sizes for max_model_len.
#[allow(clippy::too_many_arguments)]
pub fn deepseek_v4_vllm(
    model_spec: &ModelSpec,
    routing_kind: RoutingKind,
    routing_seed: Option<u64>,
    expert_popularity_file: Option<&str>,
    serialize_streams: bool,
    step_tokens: Option<u32>,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<DeepseekV4VllmModel> {
    let model_cfg = DeepseekV4ModelCfg::from_json(Path::new(&model_spec.model_config), model_spec)
        .context("loading exact DeepSeek V4 model config")?;
    let routing = resolve_routing_source(
        routing_kind,
        routing_seed,
        model_cfg.num_experts.get(),
        DEEPSEEK_V4_EP_SIZE,
        model_cfg.num_layers,
        model_cfg.top_k,
        expert_popularity_file,
    )?;
    let parallel = DeepseekV4VllmParallel {
        ep_size: DEEPSEEK_V4_EP_SIZE,
        nvl_num_gpu: DEEPSEEK_V4_EP_SIZE,
        gpu_name: gpu.to_string(),
        serialize_streams,
        max_num_batched_tokens: step_tokens.unwrap_or(model_cfg.max_model_len),
    };
    let configs = deepseek_v4_vllm::build_configs(&model_cfg, &parallel, &routing)
        .context("expanding DeepSeek V4 architecture configs")?;
    let resolved = deepseek_v4_vllm::resolve_configs(&configs);
    deepseek_v4_vllm::build(name.to_string(), resolved, bridge)
        .context("building DeepSeek V4 vLLM model (often a missing profile.db row)")
}

/// Build DeepSeek-V4.1-Flash (B200, TP4 attention + EP4 experts). Routed demand
/// resolves through the shared [`ExpertDemandSource`]: one binding over all 40
/// routed layers, verify width 1 (no speculative decoding). The MoE rows were
/// profiled on the capture's token corpus, so `routing = corpus` is the
/// production choice. `serialize_streams` changes only CostTree composition;
/// `decoder_swa_bounded_replay` (a counterfactual, see
/// [`deepseek_v41_vllm::BoundedReplay`]) changes only the late layers' inputs.
#[allow(clippy::too_many_arguments)]
pub fn deepseek_v41_vllm(
    model_spec: &ModelSpec,
    routing_kind: RoutingKind,
    routing_seed: Option<u64>,
    expert_popularity_file: Option<&str>,
    token_corpus_file: Option<&str>,
    serialize_streams: bool,
    decoder_swa_bounded_replay: bool,
    max_model_len: Option<u32>,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<DeepseekV41VllmModel> {
    let model_cfg = DeepseekV41ModelCfg::from_json(Path::new(&model_spec.model_config), model_spec)
        .context("loading exact DeepSeek-V4.1-Flash model config")?;
    let parallel = DeepseekV41VllmParallel {
        tp_size: deepseek_v41_vllm::TP_SIZE,
        ep_size: deepseek_v41_vllm::EP_SIZE,
        gpu_name: gpu.to_string(),
        serialize_streams,
        decoder_swa_bounded_replay,
        max_model_len: deepseek_v41_vllm::resolve_max_model_len(
            max_model_len,
            model_cfg.max_position_embeddings,
        )?,
    };
    let source = ExpertDemandSource {
        kind: routing_kind,
        seed: routing_seed,
        num_experts: model_cfg.num_experts,
        experts_per_token: model_cfg.top_k,
        ep_size: parallel.ep_size as u16,
        expert_popularity_file,
        token_corpus_file,
        num_routed_layers: model_cfg.num_layers,
    };
    let demand = source.demand(0..model_cfg.num_layers as usize, 1)?;
    let configs = deepseek_v41_vllm::build_configs(&model_cfg, &parallel, &demand)
        .context("expanding DeepSeek-V4.1-Flash architecture configs")?;
    let resolved = deepseek_v41_vllm::resolve_configs(&configs);
    deepseek_v41_vllm::build(name.to_string(), resolved, bridge)
        .context("building DeepSeek-V4.1-Flash vLLM model (often a missing profile.db row)")
}

/// Build the GLM-5.2 model in aligned vLLM kernel granularity.
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

/// Build the B200 GLM-5.2 NVFP4 graph. `ep_size` is also the tensor-parallel
/// degree; the architecture derives each rank-local workload from one global
/// routing sample.
#[allow(clippy::too_many_arguments)]
pub fn glm52_vllm_nvfp4_dsa_moe(
    model_spec: &ModelSpec,
    ep_size: u16,
    nvl_num_gpu: u16,
    max_model_len: u32,
    routing_kind: RoutingKind,
    routing_seed: Option<u64>,
    mtp_mode: Glm52MtpMode,
    expert_popularity_file: Option<&str>,
    token_corpus_file: Option<&str>,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Glm52VllmNvfp4DsaMoeModel> {
    let model_cfg = glm52_model_cfg(model_spec).context("loading exact GLM-5.2 NVFP4 config")?;
    let source = ExpertDemandSource {
        kind: routing_kind,
        seed: routing_seed,
        num_experts: model_cfg.num_experts.get(),
        experts_per_token: model_cfg.router_top_k,
        ep_size,
        expert_popularity_file,
        token_corpus_file,
        num_routed_layers: num_sparse_layers(&model_cfg),
    };
    // Ordinary decode submits one row per request, so a verify block is one
    // token wide and a contiguous draw is a single draw.
    let body = num_sparse_layers(&model_cfg) as usize;
    let body_demand = source.demand(0..body, 1)?;
    // Only when the layer exists: a marginal has no layer axis left to slice,
    // so resolving one for an `mtp_mode: off` build could only fail on evidence
    // that build never reads.
    let mtp_demand = (mtp_mode != Glm52MtpMode::Off)
        .then(|| source.demand(body..body + 1, 1))
        .transpose()?;
    let parallel = Glm52VllmNvfp4DsaMoeParallel {
        ep_size,
        nvl_num_gpu,
        max_model_len,
        gpu_name: gpu.to_string(),
    };
    let configs = glm52_vllm_nvfp4_dsa_moe::build_configs(
        &model_cfg,
        &parallel,
        &body_demand,
        mtp_demand.as_ref(),
        model_spec.fp8,
        mtp_mode,
    )
    .context("expanding B200 GLM-5.2 NVFP4 architecture configs")?;
    let resolved = glm52_vllm_nvfp4_dsa_moe::resolve_configs(&configs);
    glm52_vllm_nvfp4_dsa_moe::build(name.to_string(), resolved, bridge)
        .context("building B200 GLM-5.2 NVFP4 model (often a missing profile.db row)")
}

/// Build the B200 GLM-5.2 NVFP4 target-verify graph with its MTP proposer.
///
/// The result is a different type from [`glm52_vllm_nvfp4_dsa_moe`]'s, not the
/// same one with speculation switched on, so a deployment that binds it has
/// committed to speculating.
#[allow(clippy::too_many_arguments)]
pub fn glm52_vllm_nvfp4_dsa_moe_speculative(
    model_spec: &ModelSpec,
    ep_size: u16,
    nvl_num_gpu: u16,
    max_model_len: u32,
    routing_kind: RoutingKind,
    routing_seed: Option<u64>,
    mtp_mode: Glm52MtpMode,
    expert_popularity_file: Option<&str>,
    token_corpus_file: Option<&str>,
    draft_tokens: u32,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Glm52VllmNvfp4DsaMoeSpeculativeModel> {
    let model_cfg = glm52_model_cfg(model_spec).context("loading exact GLM-5.2 NVFP4 config")?;
    let source = ExpertDemandSource {
        kind: routing_kind,
        seed: routing_seed,
        num_experts: model_cfg.num_experts.get(),
        experts_per_token: model_cfg.router_top_k,
        ep_size,
        expert_popularity_file,
        token_corpus_file,
        num_routed_layers: num_sparse_layers(&model_cfg),
    };
    // The target verifies the drafted positions plus the token they extend, so
    // one request contributes `draft_tokens + 1` consecutive rows to a body
    // step. The proposer's first call forwards those same rows through the MTP
    // layer; only its later calls submit one row per request.
    let verify_width = draft_tokens
        .checked_add(1)
        .context("speculative draft_tokens + 1 overflows u32")?;
    let body = num_sparse_layers(&model_cfg) as usize;
    let body_demand = source.demand(0..body, verify_width)?;
    let mtp_demand = source.demand(body..body + 1, verify_width)?;
    let mtp_recurrent_demand = source.demand(body..body + 1, 1)?;
    let parallel = Glm52VllmNvfp4DsaMoeParallel {
        ep_size,
        nvl_num_gpu,
        max_model_len,
        gpu_name: gpu.to_string(),
    };
    let configs = glm52_vllm_nvfp4_dsa_moe::build_speculative_configs(
        &model_cfg,
        &parallel,
        &body_demand,
        &mtp_demand,
        &mtp_recurrent_demand,
        model_spec.fp8,
        mtp_mode,
        draft_tokens,
    )
    .context("expanding speculative B200 GLM-5.2 NVFP4 architecture configs")?;
    let resolved = glm52_vllm_nvfp4_dsa_moe::resolve_configs(&configs);
    glm52_vllm_nvfp4_dsa_moe::build_speculative(name.to_string(), resolved, bridge)
        .context("building speculative B200 GLM-5.2 NVFP4 model (often a missing profile.db row)")
}

/// Build the B200 GLM-5.3-Flash FP8 KDA/DSA/MoE model (vLLM fork, TP = EP).
///
/// The checkpoint's layer schedule is exact, so layer-count overrides are
/// rejected. Routing is resolved over the 42 routed layers; a corpus captured
/// with MTP off records exactly those.
#[allow(clippy::too_many_arguments)]
pub fn glm53_flash_vllm_fp8_kda_dsa_moe(
    model_spec: &ModelSpec,
    tp_size: u16,
    max_model_len: u32,
    routing_kind: RoutingKind,
    routing_seed: Option<u64>,
    expert_popularity_file: Option<&str>,
    token_corpus_file: Option<&str>,
    cudagraph_capture_sizes: &[u32],
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Glm53FlashVllmModel> {
    if model_spec.num_layers.is_some() || model_spec.sim_num_layers.is_some() {
        bail!(
            "GLM-5.3-Flash architecture rejects num_layers/sim_num_layers overrides; the exact hybrid 45-layer schedule is required"
        );
    }
    anyhow::ensure!(
        model_spec.fp8,
        "GLM-5.3-Flash is modeled for its FP8 block checkpoint; set fp8: true"
    );
    let model_cfg = Glm53FlashModelCfg::from_json(Path::new(&model_spec.model_config))
        .context("loading GLM-5.3-Flash config")?;
    let routed_layers = model_cfg.num_moe_layers();
    let source = ExpertDemandSource {
        kind: routing_kind,
        seed: routing_seed,
        num_experts: model_cfg.n_routed_experts,
        experts_per_token: model_cfg.num_experts_per_tok,
        ep_size: tp_size,
        expert_popularity_file,
        token_corpus_file,
        num_routed_layers: routed_layers,
    };
    // Ordinary decode: one row per request, so a verify block is one token.
    let demand = source.demand(0..routed_layers as usize, 1)?;
    let parallel = Glm53FlashVllmParallel {
        tp_size,
        max_model_len,
        gpu_name: gpu.to_string(),
        cudagraph_capture_sizes: cudagraph_capture_sizes.to_vec(),
    };
    let configs = glm53_flash_vllm_fp8_kda_dsa_moe::build_configs(&model_cfg, &parallel, &demand)
        .context("expanding GLM-5.3-Flash architecture configs")?;
    let resolved = glm53_flash_vllm_fp8_kda_dsa_moe::resolve_configs(&configs);
    glm53_flash_vllm_fp8_kda_dsa_moe::build(name.to_string(), resolved, bridge)
        .context("building GLM-5.3-Flash model (often a missing profile.db row)")
}

/// Build the GLM target graph driven by a DFlash2 block-parallel proposer.
///
/// The draft is a separate dense checkpoint, so its dimensions come from that
/// checkpoint rather than from the GLM config expansion, and it routes nothing:
/// six GQA layers with a plain SwiGLU MLP.
#[allow(clippy::too_many_arguments)]
pub fn glm53_vllm_nvfp4_dsa_moe_dflash2(
    model_spec: &ModelSpec,
    ep_size: u16,
    nvl_num_gpu: u16,
    max_model_len: u32,
    routing_kind: RoutingKind,
    routing_seed: Option<u64>,
    expert_popularity_file: Option<&str>,
    token_corpus_file: Option<&str>,
    draft_tokens: u32,
    draft_sliding_window: u32,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Glm53VllmNvfp4DsaMoeDflash2Model> {
    let model_cfg = glm52_model_cfg(model_spec).context("loading exact GLM NVFP4 target config")?;
    let source = ExpertDemandSource {
        kind: routing_kind,
        seed: routing_seed,
        num_experts: model_cfg.num_experts.get(),
        experts_per_token: model_cfg.router_top_k,
        ep_size,
        expert_popularity_file,
        token_corpus_file,
        num_routed_layers: num_sparse_layers(&model_cfg),
    };
    let verify_width = draft_tokens
        .checked_add(1)
        .context("speculative draft_tokens + 1 overflows u32")?;
    let body = num_sparse_layers(&model_cfg) as usize;
    let body_demand = source.demand(0..body, verify_width)?;
    let parallel = Glm52VllmNvfp4DsaMoeParallel {
        ep_size,
        nvl_num_gpu,
        max_model_len,
        gpu_name: gpu.to_string(),
    };
    // The GLM-5.3 checkpoint does carry `num_nextn_predict_layers: 1`, but
    // `--speculative-config {"method": "dflash", ...}` never instantiates it --
    // the capture's server log mentions no MTP or nextn layer at all.
    let configs = glm52_vllm_nvfp4_dsa_moe::build_verify_configs(
        &model_cfg,
        &parallel,
        &body_demand,
        model_spec.fp8,
        draft_tokens,
    )
    .context("expanding GLM NVFP4 target configs for a DFlash2 deployment")?;
    let resolved = glm52_vllm_nvfp4_dsa_moe::resolve_configs(&configs);
    let draft = dflash2_draft_resolved(&parallel, draft_tokens, draft_sliding_window)
        .context("expanding the DFlash2 draft configs")?;
    glm53_vllm_nvfp4_dsa_moe_dflash2::build_dflash2(name.to_string(), resolved, draft, bridge)
        .context("building GLM-5.3 NVFP4 + DFlash2 (often a missing profile.db row)")
}

/// The DFlash2 draft checkpoint's dimensions, from its own `config.json`:
/// 6 layers, hidden 6144, intermediate 12288, 64 query / 8 KV heads at
/// head_dim 128, vocab 154880, `conv_kernel_size` 2, `conv_group_size` 16,
/// `selector_rank` 256, `selector_top_k` 16, and six `target_layer_ids`.
fn dflash2_draft_resolved(
    parallel: &Glm52VllmNvfp4DsaMoeParallel,
    draft_tokens: u32,
    sliding_window: u32,
) -> Result<Dflash2DraftResolved> {
    anyhow::ensure!(draft_tokens > 0, "DFlash2 draft_tokens must be positive");
    anyhow::ensure!(
        sliding_window > 0,
        "DFlash2 draft_sliding_window must be positive"
    );
    let tp_size = parallel.ep_size;
    let gpu_name = parallel.gpu_name.clone();
    let layer_cfg = Dflash2DraftLayerLocalWorkletConfig {
        residual_norm_backends: vec!["vllm_cuda"],
        norm_backends: vec!["flashinfer"],
        gemm_backends: vec!["torch_linear"],
        elementwise_backends: vec!["triton"],
        // `fa2` is the only rect backend with a working ragged path on B200,
        // and it is bf16-only -- see the worklet's note on why the leaf is
        // measured in bf16 while the engine runs fp8.
        attention_backends: vec!["fa2"],
        tp_size,
        gpu_name: gpu_name.clone(),
        hidden_dim: Dim::param("dflash2_hidden", 6144),
        intermediate_dim: Dim::param("dflash2_intermediate", 12288),
        num_qo_heads: Dim::param("dflash2_qo_heads", 64),
        num_kv_heads: Dim::param("dflash2_kv_heads", 8),
        head_dim: Dim::param("dflash2_head_dim", 128),
        conv_taps: 2,
        conv_group_size: 16,
        dtype: DType::Bf16,
        gemm_dtype: DType::Bf16,
        attn_q_dtype: DType::Bf16,
        attn_kv_dtype: DType::Bf16,
    };
    let context_kv_cfg = Dflash2ContextKvLocalWorkletConfig {
        norm_backends: vec!["flashinfer"],
        gemm_backends: vec!["torch_linear"],
        elementwise_backends: vec!["triton"],
        kv_cache_append_backends: vec!["vllm_cuda"],
        tp_size,
        gpu_name: gpu_name.clone(),
        hidden_dim: Dim::param("dflash2_hidden", 6144),
        num_draft_layers: 6,
        num_aux_layers: 6,
        num_kv_heads: Dim::param("dflash2_kv_heads", 8),
        head_dim: Dim::param("dflash2_head_dim", 128),
        kv_cache_block_size: 64,
        kv_cache_layout: "NHD".to_string(),
        kv_scale_granularity: "tensor".to_string(),
        dtype: DType::Bf16,
        gemm_dtype: DType::Bf16,
        kv_dtype: DType::Fp8E4m3,
    };
    let selector_cfg = Dflash2SelectorLocalWorkletConfig {
        gemm_backends: vec!["torch_linear"],
        elementwise_backends: vec!["triton"],
        // The capture runs flashinfer's radix select
        // (`RadixTopKKernel_Unified` and the filtered/finalize pair), so that
        // is the identity to price against; `torch` stays behind it as the
        // semantic fallback the kernel kind also registers.
        topk_backends: vec!["flashinfer", "torch"],
        tp_size,
        gpu_name: gpu_name.clone(),
        hidden_dim: Dim::param("dflash2_hidden", 6144),
        vocab_size: Dim::param("dflash2_vocab", 154880),
        selector_rank: 256,
        selector_top_k: 16,
        dtype: DType::Bf16,
        gemm_dtype: DType::Bf16,
    };
    Ok(Dflash2DraftResolved {
        sliding_window,
        context_kv: Dflash2ContextKvLocalWorklet::resolve_config(&context_kv_cfg),
        draft_attn: Dflash2DraftAttnLocalWorklet::resolve_config(&layer_cfg),
        draft_ffn: Dflash2DraftFfnLocalWorklet::resolve_config(&layer_cfg),
        selector: Dflash2SelectorLocalWorklet::resolve_config(&selector_cfg),
        tp_allreduce: AllReduceKernelConfig {
            backends: vec!["nccl", "nvshmem"],
            gpu_name,
            num_gpus: u32::from(tp_size),
            fabric: Fabric::Nvlink,
        },
    })
}

/// Build SGLang's B200 GLM-5.2 NVFP4 graph under pure tensor parallelism.
/// Routing is resolved at EP1 because every TP rank owns all experts.
#[allow(clippy::too_many_arguments)]
pub fn glm52_sglang_nvfp4_tp_dsa_moe(
    model_spec: &ModelSpec,
    tp_size: u16,
    max_model_len: u32,
    routing_kind: RoutingKind,
    routing_seed: Option<u64>,
    mtp_mode: Glm52MtpMode,
    expert_popularity_file: Option<&str>,
    token_corpus_file: Option<&str>,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Glm52SglangNvfp4TpDsaMoeModel> {
    let model_cfg = glm52_model_cfg(model_spec).context("loading exact GLM-5.2 NVFP4 config")?;
    let source = ExpertDemandSource {
        kind: routing_kind,
        seed: routing_seed,
        num_experts: model_cfg.num_experts.get(),
        experts_per_token: model_cfg.router_top_k,
        // EP1: every TP rank owns all experts, so the fold sees one whole rank.
        ep_size: 1,
        expert_popularity_file,
        token_corpus_file,
        num_routed_layers: num_sparse_layers(&model_cfg),
    };
    let demand = source.demand(0..num_sparse_layers(&model_cfg) as usize, 1)?;
    let parallel = Glm52SglangNvfp4TpDsaMoeParallel {
        tp_size,
        max_model_len,
        gpu_name: gpu.to_string(),
    };
    let configs = glm52_sglang_nvfp4_tp_dsa_moe::build_configs(
        &model_cfg,
        &parallel,
        &demand,
        model_spec.fp8,
        mtp_mode,
    )
    .context("expanding B200 GLM-5.2 NVFP4 pure-TP architecture configs")?;
    let resolved = glm52_sglang_nvfp4_tp_dsa_moe::resolve_configs(&configs);
    glm52_sglang_nvfp4_tp_dsa_moe::build(name.to_string(), resolved, bridge)
        .context("building B200 GLM-5.2 NVFP4 pure-TP model (often a missing profile.db row)")
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
    let routing = resolve_routing(routing_kind, routing_seed, model_cfg.num_experts.get())?;
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
    let routing = resolve_routing(routing_kind, routing_seed, model_cfg.num_experts.get())?;
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
        IterArchSel::Llama3Neuron { model, kv_capacity } => {
            Box::new(llama3_neuron(model, *kv_capacity, gpu, name, bridge)?)
        }
        IterArchSel::Llama3VllmNeuron {
            model,
            max_model_len,
            decode_buckets,
            tp_size,
        } => Box::new(llama3_vllm_neuron(
            model,
            *max_model_len,
            decode_buckets,
            *tp_size,
            gpu,
            name,
            bridge,
        )?),
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
        IterArchSel::DeepseekV4Vllm {
            model,
            routing,
            routing_seed,
            expert_popularity_file,
        } => Box::new(deepseek_v4_vllm(
            model,
            *routing,
            *routing_seed,
            expert_popularity_file.as_deref(),
            false,
            None,
            gpu,
            name,
            bridge,
        )?),
        IterArchSel::DeepseekV4VllmSerialStreams {
            model,
            routing,
            routing_seed,
            expert_popularity_file,
        } => Box::new(deepseek_v4_vllm(
            model,
            *routing,
            *routing_seed,
            expert_popularity_file.as_deref(),
            true,
            None,
            gpu,
            name,
            bridge,
        )?),
        IterArchSel::DeepseekV41Vllm {
            model,
            routing,
            routing_seed,
            expert_popularity_file,
            token_corpus_file,
            decoder_swa_bounded_replay,
            max_model_len,
        } => Box::new(deepseek_v41_vllm(
            model,
            *routing,
            *routing_seed,
            expert_popularity_file.as_deref(),
            token_corpus_file.as_deref(),
            false,
            *decoder_swa_bounded_replay,
            *max_model_len,
            gpu,
            name,
            bridge,
        )?),
        IterArchSel::DeepseekV41VllmSerialStreams {
            model,
            routing,
            routing_seed,
            expert_popularity_file,
            token_corpus_file,
            decoder_swa_bounded_replay,
            max_model_len,
        } => Box::new(deepseek_v41_vllm(
            model,
            *routing,
            *routing_seed,
            expert_popularity_file.as_deref(),
            token_corpus_file.as_deref(),
            true,
            *decoder_swa_bounded_replay,
            *max_model_len,
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
        IterArchSel::Glm52VllmNvfp4DsaMoe {
            model,
            ep_size,
            nvl_num_gpu,
            max_model_len,
            routing,
            routing_seed,
            mtp_mode,
            expert_popularity_file,
            token_corpus_file,
        } => Box::new(glm52_vllm_nvfp4_dsa_moe(
            model,
            *ep_size,
            *nvl_num_gpu,
            *max_model_len,
            *routing,
            *routing_seed,
            *mtp_mode,
            expert_popularity_file.as_deref(),
            token_corpus_file.as_deref(),
            gpu,
            name,
            bridge,
        )?),
        IterArchSel::Glm52SglangNvfp4TpDsaMoe {
            model,
            tp_size,
            max_model_len,
            routing,
            routing_seed,
            mtp_mode,
            expert_popularity_file,
            token_corpus_file,
        } => Box::new(glm52_sglang_nvfp4_tp_dsa_moe(
            model,
            *tp_size,
            *max_model_len,
            *routing,
            *routing_seed,
            *mtp_mode,
            expert_popularity_file.as_deref(),
            token_corpus_file.as_deref(),
            gpu,
            name,
            bridge,
        )?),
        IterArchSel::Glm53FlashVllmFp8KdaDsaMoe {
            model,
            tp_size,
            max_model_len,
            routing,
            routing_seed,
            expert_popularity_file,
            token_corpus_file,
            cudagraph_capture_sizes,
        } => Box::new(glm53_flash_vllm_fp8_kda_dsa_moe(
            model,
            *tp_size,
            *max_model_len,
            *routing,
            *routing_seed,
            expert_popularity_file.as_deref(),
            token_corpus_file.as_deref(),
            cudagraph_capture_sizes,
            gpu,
            name,
            bridge,
        )?),
        IterArchSel::Glm52VllmNvfp4DsaMoeSpeculative { .. }
        | IterArchSel::Glm53VllmNvfp4DsaMoeDflash2 { .. } => {
            bail!("timing-predict: use arch.speculative_iter for a speculative model")
        }
    })
}

/// Build the speculative query contract without an L5 worker.
pub fn build_speculative_iter_model(
    selector: &IterArchSel,
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<(Box<dyn SpeculativeUnifiedModel>, u32)> {
    match selector {
        IterArchSel::Glm52VllmNvfp4DsaMoeSpeculative {
            model,
            ep_size,
            nvl_num_gpu,
            max_model_len,
            routing,
            routing_seed,
            mtp_mode,
            draft_tokens,
            expert_popularity_file,
            token_corpus_file,
        } => Ok((
            Box::new(glm52_vllm_nvfp4_dsa_moe_speculative(
                model,
                *ep_size,
                *nvl_num_gpu,
                *max_model_len,
                *routing,
                *routing_seed,
                *mtp_mode,
                expert_popularity_file.as_deref(),
                token_corpus_file.as_deref(),
                *draft_tokens,
                gpu,
                name,
                bridge,
            )?),
            *draft_tokens,
        )),
        IterArchSel::Glm53VllmNvfp4DsaMoeDflash2 {
            model,
            ep_size,
            nvl_num_gpu,
            max_model_len,
            routing,
            routing_seed,
            draft_tokens,
            draft_sliding_window,
            expert_popularity_file,
            token_corpus_file,
        } => Ok((
            Box::new(glm53_vllm_nvfp4_dsa_moe_dflash2(
                model,
                *ep_size,
                *nvl_num_gpu,
                *max_model_len,
                *routing,
                *routing_seed,
                expert_popularity_file.as_deref(),
                token_corpus_file.as_deref(),
                *draft_tokens,
                *draft_sliding_window,
                gpu,
                name,
                bridge,
            )?),
            *draft_tokens,
        )),
        _ => bail!("timing-predict speculative_iter requires a speculative architecture"),
    }
}

/// Build ONE AFD attn-side model from its selector, boxed as `dyn` — the
/// [`build_iter_model`] counterpart for the attn arch. The model-only seam the
/// offline `timing-predict` (attn arch) path uses: it drives [`AttnLayerwiseModel`]
/// directly, no worker/flow. The `afd` deployment does NOT box — it calls the
/// concrete [`qwen3_attn`] builder to keep its worker factory monomorphized.
/// Returning `Box<dyn>` (not `impl`) lets another implemented attention arch be
/// added without a signature break.
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
    })
}

/// Build ONE AFD ffn-side model from its selector, boxed as `dyn` — the ffn
/// counterpart to [`build_attn_model`].
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
    })
}

/// One arch block on one GPU, built structure-only.
#[derive(Debug, serde::Serialize)]
pub struct ArchBuild {
    /// The arch contract, as `list-params` groups providers: `iter_wise`,
    /// `layer_wise_attn` or `layer_wise_ffn`.
    pub contract: &'static str,
    pub arch: String,
    /// The group's `gpu`, the profile.db key.
    pub gpu: String,
    /// The block's arch params, without `type`. Params the block leaves out
    /// took their schema defaults.
    pub params: serde_json::Map<String, serde_json::Value>,
    pub gpus_per_replica: Option<u16>,
    /// What a `timing-predict` case for this build looks like; `None` on error.
    pub predict: Option<PredictShape>,
    /// The cost tree, in the `cost_manifest/*.json` form; `None` on error.
    pub cost_manifest: Option<crate::timing::CostManifestDoc>,
    /// The kernel configs the build asks profile.db for, as
    /// [`config_records_document`](crate::timing::bridge::config_records_document)
    /// builds; only when asked for, and `None` on error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kernel_configs: Option<serde_json::Value>,
    /// Per `cost_manifest` section, per slot: the index in `kernel_configs`'
    /// `configs` of the config the slot's kernel reads, matched by kind, GPU
    /// and [`KernelConfig::identity`](crate::timing::KernelConfig::identity);
    /// null when none matches. Only with `kernel_configs`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub slot_configs: Option<Vec<Vec<Option<usize>>>>,
    pub error: Option<String>,
}

/// The `timing-predict` case shape of a built arch: the arch selector the
/// predict config names it under, and what each case must hold.
#[derive(Debug, serde::Serialize)]
pub struct PredictShape {
    /// `iter`, `speculative_iter`, `attn` or `ffn`.
    pub selector: &'static str,
    /// The groups each case lists (`groups`, or an ffn case's
    /// `tokens_per_group`): the model's attention DP shards.
    pub groups: u16,
    /// A speculative arch's verify width, `draft_tokens + 1`: every decode
    /// request's query length.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub query_width: Option<u32>,
    /// The arch's longest request ([`IterArchSel::max_model_len`]), which no
    /// case's request may exceed; an FFN side, which holds no context, has none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<u32>,
}

impl PredictShape {
    fn new(selector: &'static str, groups: u16) -> Self {
        Self {
            selector,
            groups,
            query_width: None,
            max_model_len: None,
        }
    }
}

/// What a structure-only build returns: the replica width, its prediction case
/// shape and its cost tree.
type Built = (u16, PredictShape, crate::timing::CostManifestDoc);

/// The dotted-leaf prefix a deployment gives its model, which starts every leaf
/// name and kernel-config role. Arch builds use the deployment's own, as
/// `timing-predict` does, so their leaves are named as a real run's are.
const UNIFIED_MODEL_NAME: &str = "unified";
const AFD_MODEL_NAME: &str = "afd";

/// One arch block on one GPU, as a run's pool group gives it.
/// `cost-trees FILE` reads a list of them.
#[derive(Debug, Deserialize)]
pub struct ArchBlock {
    pub gpu: String,
    /// `type` and the arch's params, file paths as this process opens them.
    pub arch: serde_json::Map<String, serde_json::Value>,
}

/// Build each block on its GPU structure only ([`PerfApiBridge::structure_only`]:
/// no Python, `profile.db` or GPU): a param the block leaves out takes its
/// schema default. Each result's `params` is its block without `type`. A block
/// whose tag no contract provides, or that fails to parse or build (including
/// a panicking shape assertion), is returned with its error, not raised.
pub fn build_arch_blocks(blocks: &[ArchBlock], kernel_configs: bool) -> Vec<ArchBuild> {
    let schema = crate::schema::list_params();
    blocks
        .iter()
        .map(|block| {
            let tag = block
                .arch
                .get("type")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            let mut params = block.arch.clone();
            params.remove("type");
            let contract = ["iter_wise", "layer_wise_attn", "layer_wise_ffn"]
                .into_iter()
                .find(|c| schema["providers"]["arch"][*c].get(tag).is_some());
            match contract {
                Some(contract) => build_arch(
                    &schema,
                    contract,
                    tag,
                    block.gpu.clone(),
                    params,
                    block.arch.clone(),
                    kernel_configs,
                ),
                None => ArchBuild {
                    contract: "",
                    arch: tag.to_string(),
                    gpu: block.gpu.clone(),
                    params,
                    gpus_per_replica: None,
                    predict: None,
                    cost_manifest: None,
                    kernel_configs: None,
                    slot_configs: None,
                    error: Some(format!("no arch contract provides {tag:?}")),
                },
            }
        })
        .collect()
}

/// Build one arch block of `contract` on `gpu`, every param the block leaves
/// out at its schema default. `params` is what the result reports.
fn build_arch(
    schema: &serde_json::Value,
    contract: &'static str,
    tag: &str,
    gpu: String,
    params: serde_json::Map<String, serde_json::Value>,
    block: serde_json::Map<String, serde_json::Value>,
    kernel_configs: bool,
) -> ArchBuild {
    let mut arch = serde_json::Map::new();
    let common = schema["arch_common"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let own = schema["providers"]["arch"][contract][tag]["params"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    for param in common.iter().chain(&own) {
        if let Some(default) = param.get("default") {
            arch.insert(
                param["name"].as_str().unwrap_or_default().into(),
                default.clone(),
            );
        }
    }
    arch.extend(block);
    let built = match contract {
        "iter_wise" => build_selector(
            arch,
            &gpu,
            kernel_configs,
            |selector: &IterArchSel, gpu, bridge| {
                // A speculative arch builds a different model type, with its
                // own tree: the verify pass plus its draft passes.
                let (gpus, shape, manifest) = match selector {
                    IterArchSel::Glm52VllmNvfp4DsaMoeSpeculative { .. }
                    | IterArchSel::Glm53VllmNvfp4DsaMoeDflash2 { .. } => {
                        let (model, draft_tokens) = build_speculative_iter_model(
                            selector,
                            gpu,
                            UNIFIED_MODEL_NAME,
                            bridge,
                        )?;
                        let shape = PredictShape {
                            query_width: Some(draft_tokens + 1),
                            max_model_len: Some(selector.max_model_len()?),
                            ..PredictShape::new("speculative_iter", model.num_attn_dp_groups())
                        };
                        (model.gpus_per_replica(), shape, model.cost_log_manifest())
                    }
                    _ => {
                        let model = build_iter_model(selector, gpu, UNIFIED_MODEL_NAME, bridge)?;
                        let shape = PredictShape {
                            max_model_len: Some(selector.max_model_len()?),
                            ..PredictShape::new("iter", model.num_attn_dp_groups())
                        };
                        (model.gpus_per_replica(), shape, model.cost_log_manifest())
                    }
                };
                Ok((
                    gpus,
                    shape,
                    crate::timing::CostManifestDoc::single("iter", manifest),
                ))
            },
        ),
        "layer_wise_attn" => build_selector(
            arch,
            &gpu,
            kernel_configs,
            |selector: &AttnArchSel, gpu, bridge| {
                let model = build_attn_model(selector, gpu, AFD_MODEL_NAME, bridge)?;
                let shape = PredictShape {
                    max_model_len: Some(selector.max_model_len()?),
                    ..PredictShape::new("attn", model.num_attn_dp_groups())
                };
                Ok((model.gpus_per_replica(), shape, model.cost_log_manifest()))
            },
        ),
        _ => build_selector(
            arch,
            &gpu,
            kernel_configs,
            |selector: &FfnArchSel, gpu, bridge| {
                let model = build_ffn_model(selector, gpu, AFD_MODEL_NAME, bridge)?;
                let shape = PredictShape::new("ffn", model.num_dp_groups());
                Ok((model.gpus_per_replica(), shape, model.cost_log_manifest()))
            },
        ),
    };
    let mut out = ArchBuild {
        contract,
        arch: tag.to_string(),
        gpu,
        params,
        gpus_per_replica: None,
        predict: None,
        cost_manifest: None,
        kernel_configs: None,
        slot_configs: None,
        error: None,
    };
    match built {
        Ok(((gpus_per_replica, shape, manifest), configs)) => {
            out.gpus_per_replica = Some(gpus_per_replica);
            out.predict = Some(shape);
            if let Some((document, slots)) = configs {
                out.kernel_configs = Some(document);
                out.slot_configs = Some(slots);
            }
            out.cost_manifest = Some(manifest);
        }
        Err(error) => out.error = Some(error),
    }
    out
}

/// The kernel-config records document and, per manifest section and slot, the
/// record its kernel reads ([`ArchBuild::slot_configs`]).
type Configs = (serde_json::Value, Vec<Vec<Option<usize>>>);

/// Parse `arch` as selector `S` and `build` it on `gpu` with a structure-only
/// bridge: the replica width, the prediction case shape, the cost tree and,
/// when asked, the kernel-config records with each slot's. A parse error, build error or panic is the error text.
fn build_selector<S: serde::de::DeserializeOwned>(
    arch: serde_json::Map<String, serde_json::Value>,
    gpu: &str,
    kernel_configs: bool,
    build: impl Fn(&S, &str, &PerfApiBridge) -> Result<Built>,
) -> std::result::Result<(Built, Option<Configs>), String> {
    let selector: S = serde_json::from_value(serde_json::Value::Object(arch))
        .map_err(|e| format!("config: {e}"))?;
    let bridge = PerfApiBridge::structure_only();
    if kernel_configs {
        bridge.enable_config_records();
    }
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        build(&selector, gpu, &bridge)
    }));
    match built {
        Ok(Ok(built)) => {
            let configs = if kernel_configs {
                let records = bridge.take_config_records();
                let slots = slot_configs(&built.2, &records).map_err(|e| format!("{e:#}"))?;
                Some((
                    crate::timing::bridge::config_records_document(&records),
                    slots,
                ))
            } else {
                None
            };
            Ok((built, configs))
        }
        Ok(Err(e)) => Err(format!("{e:#}")),
        Err(panic) => Err(panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
            .map_or_else(
                || "build panicked".into(),
                |message| format!("build panicked: {message}"),
            )),
    }
}

/// Per section and slot of `manifest`, the index in `records` of the config
/// the slot's kernel reads: the record with its kind, GPU and identity.
fn slot_configs(
    manifest: &crate::timing::CostManifestDoc,
    records: &[crate::timing::bridge::KernelConfigRecord],
) -> Result<Vec<Vec<Option<usize>>>> {
    let key = |kind: &str, gpu: &str, identity: &serde_json::Value| {
        serde_json::to_string(&(kind, gpu, identity)).expect("an identity serializes")
    };
    let index: std::collections::HashMap<String, usize> = records
        .iter()
        .enumerate()
        .map(|(i, r)| (key(r.kind, &r.gpu_name, &r.identity), i))
        .collect();
    manifest
        .sections
        .iter()
        .map(|section| {
            section
                .manifest
                .slots
                .iter()
                .map(|slot| {
                    let gpu = slot.kernel_config["gpu_name"].as_str().unwrap_or_default();
                    let identity = crate::timing::kernels::engine::config_identity(
                        &slot.kind,
                        slot.kernel_config.clone(),
                    )
                    .with_context(|| format!("slot {}", slot.name))?;
                    Ok(index.get(&key(&slot.kind, gpu, &identity)).copied())
                })
                .collect()
        })
        .collect()
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
        assert_eq!(built.cost_log_manifest().slots.len(), 72);
    }

    /// Two callables, one artifact. This is the whole reason the corpus is
    /// layer-indexed: a marginal has to be handed a second file to say that the
    /// MTP layer routes differently, and here it is a slice of the same bytes.
    #[test]
    fn a_corpus_serves_the_body_and_the_mtp_layer_from_one_artifact() {
        let dir = crate::timing::token_corpus::tests::temp_dir("arch-corpus");
        let corpus = crate::timing::token_corpus::tests::synthetic(&dir, 64, 4, 8, 512);
        let source = ExpertDemandSource {
            kind: RoutingKind::Corpus,
            seed: None,
            num_experts: 64,
            experts_per_token: 4,
            ep_size: 2,
            expert_popularity_file: None,
            token_corpus_file: Some(&corpus.data_file.replace("routes.u16", "manifest.json")),
            num_routed_layers: 7,
        };
        let body = source.demand(0..7, 6).unwrap();
        let mtp = source.demand(7..8, 1).unwrap();

        assert_eq!((body.num_experts(), mtp.num_experts()), (64, 64));
        assert_ne!(
            body.prepare().unwrap().sample_and_fold(4, 24, 32),
            mtp.prepare().unwrap().sample_and_fold(4, 24, 32),
            "the MTP slice must not be the body's fold"
        );
        // The verify width reaches the sampler: only the body's batch is
        // block-structured, so only its axis gains the block multiples.
        assert!(
            body.token_axis(vec![1.0]).contains(&48.0),
            "eight width-6 verify blocks must be a profiled shape"
        );
        assert!(
            !mtp.token_axis(vec![1.0]).contains(&48.0),
            "a width-1 draft pass has no block structure to widen the axis for"
        );
    }

    #[test]
    fn a_demand_source_refuses_the_artifact_its_routing_kind_did_not_name() {
        let dir = crate::timing::token_corpus::tests::temp_dir("arch-corpus-mismatch");
        let corpus = crate::timing::token_corpus::tests::synthetic(&dir, 64, 4, 8, 512);
        let manifest = corpus.data_file.replace("routes.u16", "manifest.json");
        let source = |kind, popularity, token_corpus| ExpertDemandSource {
            kind,
            seed: None,
            num_experts: 64,
            experts_per_token: 4,
            ep_size: 2,
            expert_popularity_file: popularity,
            token_corpus_file: token_corpus,
            num_routed_layers: 7,
        };
        let demand = |s: ExpertDemandSource<'_>| s.demand(0..7, 6);

        let missing = demand(source(RoutingKind::Corpus, None, None)).unwrap_err();
        assert!(missing.to_string().contains("requires token_corpus_file"));

        let unused = demand(source(RoutingKind::Uniform, None, Some(&manifest))).unwrap_err();
        assert!(unused.to_string().contains("use routing=corpus"));

        // A corpus recorded for a different router is a silent mis-pricing, so
        // it is rejected on dimensions rather than on the file name.
        let wrong = ExpertDemandSource {
            experts_per_token: 8,
            ..source(RoutingKind::Corpus, None, Some(&manifest))
        };
        let error = demand(wrong).unwrap_err();
        assert!(format!("{error:#}").contains("records top-4 of 64 experts"));

        // The payload length pins only the product of the dimensions, so a
        // manifest that trades tokens for layers keeps its byte count and its
        // checksum while sampling every other token.
        let restated = ExpertDemandSource {
            num_routed_layers: 3,
            ..source(RoutingKind::Corpus, None, Some(&manifest))
        };
        let error = ExpertDemandSource::demand(&restated, 0..3, 6).unwrap_err();
        assert!(format!("{error:#}").contains("records 8 layers; this model has 3"));

        // Both legitimate axes load: a capture without a drafter records the
        // body alone, and `restated` above failed only for being neither.
        // (A drafted capture pricing an undrafted build is the 0..7 call in
        // `a_corpus_serves_the_body_and_the_mtp_layer_from_one_artifact`.)
        let undrafted = ExpertDemandSource {
            num_routed_layers: 8,
            ..source(RoutingKind::Corpus, None, Some(&manifest))
        };
        ExpertDemandSource::demand(&undrafted, 0..8, 1).expect("the body of an 8-layer corpus");

        // Leaving the other kind's file behind is rejected in both directions:
        // the cost would not change, so nothing would show which source was read.
        let both = demand(source(
            RoutingKind::Corpus,
            Some("presets/alignment/x/expert_popularity.json"),
            Some(&manifest),
        ))
        .unwrap_err();
        assert!(format!("{both:#}").contains("use routing=popularity"));
    }

    /// The other half of the same point: a marginal was captured over the body's
    /// layers and stops there, so the MTP layer folds the layer-summed
    /// distribution rather than a layer of its own.
    #[test]
    fn a_marginal_gives_the_mtp_layer_the_only_resolution_it_has() {
        let mut profile = tempfile::NamedTempFile::new().unwrap();
        // Two layers that disagree completely, so a sum is distinguishable from
        // either of them.
        write!(
            profile,
            r#"{{"schema_version": 1, "num_logical_experts": 4,
                 "counts_by_layer": [[100, 0, 0, 0], [25, 25, 25, 25]]}}"#
        )
        .unwrap();
        let source = ExpertDemandSource {
            kind: RoutingKind::Popularity,
            seed: None,
            num_experts: 4,
            experts_per_token: 2,
            ep_size: 2,
            expert_popularity_file: profile.path().to_str(),
            token_corpus_file: None,
            num_routed_layers: 2,
        };
        let ppm = |demand| match demand {
            ExpertDemand::Popularity {
                layerwise_global_ppm,
            } => layerwise_global_ppm,
            _ => unreachable!("popularity routing resolves to the popularity arm"),
        };

        assert_eq!(
            ppm(source.demand(0..2, 1).unwrap()),
            vec![
                vec![1_000_000, 0, 0, 0],
                vec![250_000, 250_000, 250_000, 250_000]
            ],
            "the body reads the profile's own layers"
        );
        // One layer, and not either of the body's: the profile has no row for
        // the MTP layer, so the only honest answer is the mean of the rows it
        // does have.
        assert_eq!(
            ppm(source.demand(2..3, 1).unwrap()),
            vec![vec![625_000, 125_000, 125_000, 125_000]]
        );
    }

    #[test]
    fn routing_source_requires_explicit_custom_and_never_falls_back() {
        for kind in [RoutingKind::Uniform, RoutingKind::Random] {
            assert!(resolve_routing_source(kind, None, 4, 2, 2, 2, None).is_ok());
            let error =
                resolve_routing_source(kind, None, 4, 2, 2, 2, Some("unused.json")).unwrap_err();
            assert!(error.to_string().contains("use routing=popularity"));
        }
        let error =
            resolve_routing_source(RoutingKind::Popularity, None, 4, 2, 2, 2, None).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires expert_popularity_file")
        );
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("missing.json");
        assert!(
            resolve_routing_source(RoutingKind::Popularity, None, 4, 2, 2, 2, missing.to_str(),)
                .is_err()
        );
    }

    #[test]
    fn custom_expert_popularity_profile_loads_and_preserves_ppm_sum() {
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
        let routing = resolve_routing_source(
            RoutingKind::Popularity,
            None,
            4,
            2,
            2,
            2,
            Some(profile_path),
        )
        .unwrap();
        assert_eq!(routing.num_experts(), 4);
        assert_eq!(
            routing.ppm().iter().sum::<u32>(),
            RoutingDistribution::TOTAL_PPM
        );
        assert!(routing.ppm()[0] > routing.ppm()[1]);
        assert!(
            resolve_routing_source(RoutingKind::Random, Some(7), 4, 2, 2, 2, Some(profile_path),)
                .is_err()
        );
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
            RoutingKind::Popularity,
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
        assert_eq!(routing.layer_ppm().len(), 2);
        assert_eq!(routing.layer_ppm()[0], [428_572, 95_238, 238_095, 238_095]);
        assert_eq!(routing.layer_ppm()[1], [666_666, 0, 166_667, 166_667]);
    }

    #[test]
    fn expert_popularity_v2_and_v3_validate_full_model_and_partition_contract() {
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
            "counts_by_layer": [[6, 2, 5, 3], [2, 2, 6, 6]],
            "probabilities_by_layer": [
                [6.0 / 16.0, 2.0 / 16.0, 5.0 / 16.0, 3.0 / 16.0],
                [2.0 / 16.0, 2.0 / 16.0, 6.0 / 16.0, 6.0 / 16.0]
            ],
            "counts_all_layers": [8, 4, 11, 9],
            "probabilities_all_layers": [8.0 / 32.0, 4.0 / 32.0, 11.0 / 32.0, 9.0 / 32.0]
        });
        let mut profile_file = tempfile::NamedTempFile::new().unwrap();
        write!(profile_file, "{profile}").unwrap();
        let profile_path = profile_file.path().to_str().unwrap();

        let routing = resolve_routing_source(
            RoutingKind::Popularity,
            None,
            4,
            2,
            2,
            2,
            Some(profile_path),
        )
        .unwrap();
        assert_eq!(routing.ppm(), &[375_000, 250_000, 218_750, 156_250]);
        assert_eq!(routing.layer_ppm().len(), 2);

        // Captured at EP2, the counts are of logical experts, so a run at any EP
        // partitions them: at EP1 and EP4 each layer sorts all four experts.
        for ep_size in [1, 4] {
            let other_ep = resolve_routing_source(
                RoutingKind::Popularity,
                None,
                4,
                ep_size,
                2,
                2,
                Some(profile_path),
            )
            .unwrap();
            assert_eq!(other_ep.ppm(), &[375_000, 343_750, 156_250, 125_000]);
        }

        let mut infeasible_profile = profile.clone();
        infeasible_profile["counts_by_layer"][0] = serde_json::json!([9, 2, 3, 2]);
        infeasible_profile["probabilities_by_layer"][0] =
            serde_json::json!([9.0 / 16.0, 2.0 / 16.0, 3.0 / 16.0, 2.0 / 16.0]);
        infeasible_profile["counts_all_layers"] = serde_json::json!([11, 4, 9, 8]);
        infeasible_profile["probabilities_all_layers"] =
            serde_json::json!([11.0 / 32.0, 4.0 / 32.0, 9.0 / 32.0, 8.0 / 32.0]);
        let mut infeasible_file = tempfile::NamedTempFile::new().unwrap();
        write!(infeasible_file, "{infeasible_profile}").unwrap();
        let error = resolve_routing_source(
            RoutingKind::Popularity,
            None,
            4,
            2,
            2,
            2,
            infeasible_file.path().to_str(),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("infeasible distinct top-k marginals"));

        let mut v3_profile = profile.clone();
        v3_profile["schema_version"] = serde_json::json!(3);
        v3_profile["aggregation"] = serde_json::json!({
            "scope": "captured_eplb_steps_within_token_ceiling",
            "observed_eplb_step_min": 10,
            "observed_eplb_step_max": 11,
            "record_count": 2,
            "raw_record_count": 3,
            "discarded_oversized_record_count": 1,
            "discarded_oversized_eplb_steps": [9],
            "max_tokens_per_step": 32
        });
        let mut v3_file = tempfile::NamedTempFile::new().unwrap();
        write!(v3_file, "{v3_profile}").unwrap();
        assert!(
            resolve_routing_source(
                RoutingKind::Popularity,
                None,
                4,
                2,
                2,
                2,
                v3_file.path().to_str(),
            )
            .is_ok()
        );

        v3_profile["aggregation"]["raw_record_count"] = serde_json::json!(4);
        let mut invalid_v3_file = tempfile::NamedTempFile::new().unwrap();
        write!(invalid_v3_file, "{v3_profile}").unwrap();
        let error = resolve_routing_source(
            RoutingKind::Popularity,
            None,
            4,
            2,
            2,
            2,
            invalid_v3_file.path().to_str(),
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("invalid v3 aggregation metadata"));

        let mut v4_profile = profile.clone();
        v4_profile["schema_version"] = serde_json::json!(4);
        v4_profile["model_role"] = serde_json::json!("target");
        v4_profile["aggregation"] = serde_json::json!({
            "scope": "replay_window_within_role_specific_token_ceiling",
            "observed_eplb_step_min": 10,
            "observed_eplb_step_max": 11,
            "record_count": 2,
            "raw_record_count": 4,
            "discarded_oversized_record_count": 1,
            "discarded_oversized_eplb_steps": [9],
            "discarded_outside_replay_window_record_count": 1,
            "max_tokens_per_step": 32,
            "max_forwards_per_step": 1,
            "replay_start_monotonic_ns": 100,
            "replay_end_monotonic_ns": 200,
            "observed_monotonic_ns_min": 110,
            "observed_monotonic_ns_max": 190
        });
        let mut v4_file = tempfile::NamedTempFile::new().unwrap();
        write!(v4_file, "{v4_profile}").unwrap();
        let v4_path = v4_file.path().to_str().unwrap();
        let v4 = load_expert_popularity(v4_path, 4, 2, 2, 2).unwrap();
        assert_eq!(v4.ppm(), routing.ppm());
        v4_profile["aggregation"]["observed_monotonic_ns_max"] = serde_json::json!(201);
        let mut outside_file = tempfile::NamedTempFile::new().unwrap();
        write!(outside_file, "{v4_profile}").unwrap();
        assert!(load_expert_popularity(outside_file.path().to_str().unwrap(), 4, 2, 2, 2).is_err());

        let mut unknown_field_profile = profile;
        unknown_field_profile["unregulated"] = serde_json::json!(true);
        let mut invalid_file = tempfile::NamedTempFile::new().unwrap();
        write!(invalid_file, "{unknown_field_profile}").unwrap();
        let error = resolve_routing_source(
            RoutingKind::Popularity,
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
    fn glm52_nvfp4_selector_builds_through_the_shared_iter_dispatch() {
        let selector = IterArchSel::Glm52VllmNvfp4DsaMoe {
            model: ModelSpec {
                model_config: Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("model/config/glm52_nvfp4.json")
                    .to_string_lossy()
                    .into_owned(),
                num_layers: None,
                sim_num_layers: None,
                fp8: false,
            },
            ep_size: 4,
            nvl_num_gpu: 4,
            max_model_len: 8_192,
            routing: RoutingKind::Uniform,
            routing_seed: None,
            mtp_mode: Glm52MtpMode::Off,
            expert_popularity_file: None,
            token_corpus_file: None,
        };
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let built = build_iter_model(&selector, "NVIDIA B200", "test", &bridge)
            .expect("build NVFP4 GLM through shared iter dispatch");
        assert_eq!(built.gpus_per_replica(), 4);
        assert_eq!(built.num_attn_dp_groups(), 1);
        assert_eq!(built.num_attn_shards(), 4);
        // 3 sparse sections x 4 ranks x 3 leaves for the serial shared expert.
        assert_eq!(built.cost_log_manifest().slots.len(), 474 + 36);
    }

    #[test]
    fn speculative_glm52_builds_through_its_typed_predict_dispatch() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let selector = IterArchSel::Glm52VllmNvfp4DsaMoeSpeculative {
            model: glm52_model_spec(),
            ep_size: 4,
            nvl_num_gpu: 4,
            max_model_len: 8192,
            routing: RoutingKind::Uniform,
            routing_seed: None,
            mtp_mode: Glm52MtpMode::IndexShare,
            draft_tokens: 5,
            expert_popularity_file: None,
            token_corpus_file: None,
        };
        let (built, drafts) =
            build_speculative_iter_model(&selector, "NVIDIA B200", "test", &bridge)
                .expect("build the speculative NVFP4 GLM");
        assert_eq!(drafts, 5);
        assert!(build_iter_model(&selector, "NVIDIA B200", "test", &bridge).is_err());

        assert_eq!(built.gpus_per_replica(), 4);
        assert_eq!(built.max_model_len(), 8_192);
        // The target alone is 474 slots. A speculative iteration bills those
        // plus five draft passes, so its cost log is a different shape and no
        // consumer can read one as the other.
        assert!(built.cost_log_manifest().slots.len() > 474);
        assert_eq!(built.num_attn_shards(), 4);
    }

    #[test]
    fn dflash2_necessary_work_map_covers_the_compiled_locations() {
        use std::collections::BTreeSet;
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let mut spec = glm52_model_spec();
        spec.model_config = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("model/config/glm52_nvfp4.json")
            .to_string_lossy()
            .into_owned();
        let map: serde_json::Value = serde_json::from_str(include_str!(
            "../../../model/work/location_maps/glm53_vllm_nvfp4_dsa_moe_dflash2.json"
        ))
        .unwrap();
        let mapped: BTreeSet<_> = map["locations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["location"].as_str().unwrap())
            .collect();
        for ep in [4, 8] {
            let built = glm53_vllm_nvfp4_dsa_moe_dflash2(
                &spec,
                ep,
                ep,
                8192,
                RoutingKind::Uniform,
                None,
                None,
                None,
                7,
                2048,
                "NVIDIA B200",
                "unified",
                &bridge,
            )
            .unwrap();
            let manifest = built.cost_log_manifest();
            let actual: BTreeSet<_> = manifest
                .slots
                .iter()
                .filter(|slot| {
                    !matches!(
                        slot.kind.as_str(),
                        "all_reduce" | "all_reduce_fusion" | "all_reduce_residual_rms_norm"
                    )
                })
                .map(|slot| slot.name.as_str())
                .collect();
            assert_eq!(actual, mapped, "EP={ep}");
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
            assert!(
                glm52_model_cfg(&overridden)
                    .unwrap_err()
                    .to_string()
                    .contains("rejects num_layers/sim_num_layers")
            );
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
            let parallel = Glm52VllmDsaMoeParallel {
                ep_size: 16,
                nvl_num_gpu: 8,
                gpu_name: "NVIDIA H200".to_string(),
            };
            let routing = resolve_routing(RoutingKind::Random, Some(19), 256).unwrap();
            let configs =
                glm52_vllm_dsa_moe::build_configs(&model, &parallel, &routing, false, mode)
                    .unwrap();
            assert_eq!(configs.parallel.ep_size, 16);
            assert_eq!(configs.parallel.nvl_num_gpu, 8);
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

    #[test]
    fn the_dflash2_draft_kv_addend_ignores_the_sliding_window() {
        // The window bounds what the draft attends to, not what it stores: vLLM
        // cannot unify the draft's page size with the MLA latent's, so it
        // allocates the draft layers "as full attention for cache allocation".
        // Billing a window would under-count the pool by context / window,
        // which at 131072 tokens is 64x.
        let parallel = Glm52VllmNvfp4DsaMoeParallel {
            ep_size: 4,
            nvl_num_gpu: 4,
            max_model_len: 8192,
            gpu_name: "NVIDIA B200".to_string(),
        };
        let narrow = dflash2_draft_resolved(&parallel, 7, 512).unwrap();
        let wide = dflash2_draft_resolved(&parallel, 7, 131_072).unwrap();
        let narrow_bytes =
            glm53_vllm_nvfp4_dsa_moe_dflash2::draft_kv_bytes_per_token(&narrow).unwrap();
        assert_eq!(
            narrow_bytes,
            glm53_vllm_nvfp4_dsa_moe_dflash2::draft_kv_bytes_per_token(&wide).unwrap(),
        );
        // 6 layers * 8 KV heads * 128 * 2 (K and V) * 1 byte, summed over the
        // four attention ranks the heads are sharded across.
        assert_eq!(narrow_bytes, 6 * 8 * 128 * 2);
    }

    /// An arch block on `gpu` with `params`, its model config a `model/config/`
    /// file named by stem.
    fn arch_block(gpu: &str, tag: &str, model: &str, params: serde_json::Value) -> ArchBlock {
        let mut arch = params.as_object().cloned().unwrap_or_default();
        arch.insert("type".into(), tag.into());
        arch.insert(
            "model_config".into(),
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("model/config/{model}.json"))
                .to_string_lossy()
                .into_owned()
                .into(),
        );
        ArchBlock {
            gpu: gpu.into(),
            arch,
        }
    }

    /// An iter-wise block (GLM-5.2 NVFP4 at EP4) and an AFD attention block
    /// (Qwen3-235B attention at TP4).
    fn sample_blocks() -> Vec<ArchBlock> {
        vec![
            arch_block(
                "NVIDIA B200",
                "glm52_vllm_nvfp4_dsa_moe",
                "glm52_nvfp4",
                serde_json::json!({"ep_size": 4, "nvl_num_gpu": 4}),
            ),
            arch_block(
                "NVIDIA H200",
                "qwen3_attn_tp",
                "qwen3_235b",
                serde_json::json!({"attn_tp_size": 4}),
            ),
        ]
    }

    /// Each block builds a non-empty cost tree under its contract, its leaves
    /// named as a run of its deployment names them, so its kernel-config roles
    /// match a run's.
    #[test]
    fn arch_blocks_build_their_cost_trees_named_like_a_run() {
        let built = build_arch_blocks(&sample_blocks(), false);
        for (b, (contract, prefix)) in built
            .iter()
            .zip([("iter_wise", "unified."), ("layer_wise_attn", "afd.")])
        {
            assert_eq!(b.error, None, "{}", b.arch);
            assert_eq!(b.contract, contract);
            let doc = b.cost_manifest.as_ref().unwrap();
            assert!(
                !doc.sections.is_empty()
                    && doc.sections.iter().all(|s| !s.manifest.slots.is_empty()),
                "{}: empty cost tree",
                b.arch
            );
            for section in &doc.sections {
                for slot in &section.manifest.slots {
                    assert!(slot.name.starts_with(prefix), "{}: {}", b.arch, slot.name);
                }
            }
        }
    }

    /// Asked for, each build carries the kernel configs it builds; otherwise
    /// none.
    #[test]
    fn arch_blocks_carry_their_kernel_configs_when_asked() {
        for b in build_arch_blocks(&sample_blocks(), true) {
            let doc = b.kernel_configs.as_ref().expect("kernel_configs");
            assert_eq!(
                doc["schema_version"],
                crate::timing::bridge::CONFIG_RECORDS_SCHEMA_VERSION
            );
            let configs = doc["configs"].as_array().unwrap();
            assert!(!configs.is_empty(), "{} {:?}: no configs", b.arch, b.params);
            assert!(configs.iter().all(|c| c["gpu_name"] == b.gpu.as_str()));
            // Every slot names the record of the config its kernel reads.
            let manifest = b.cost_manifest.as_ref().unwrap();
            let slots = b.slot_configs.as_ref().expect("slot_configs");
            assert_eq!(slots.len(), manifest.sections.len());
            for (section, indices) in manifest.sections.iter().zip(slots) {
                assert_eq!(indices.len(), section.manifest.slots.len());
                for (slot, index) in section.manifest.slots.iter().zip(indices) {
                    let record = &configs[index.expect("a slot with no record")];
                    assert_eq!(record["kind"], slot.kind.as_str(), "{}", slot.name);
                }
            }
        }
        assert!(
            build_arch_blocks(&sample_blocks(), false)
                .iter()
                .all(|b| b.kernel_configs.is_none() && b.slot_configs.is_none())
        );
    }

    /// Each build names the `timing-predict` selector its cases go under and
    /// how many groups a case lists: one per attention DP shard, the context
    /// limit, and a speculative arch's verify width besides.
    #[test]
    fn arch_blocks_give_their_prediction_case_shape() {
        let mut blocks = sample_blocks();
        blocks.push(arch_block(
            "NVIDIA H200",
            "qwen3_moe_fp8_dp_attn_ep_ffn",
            "qwen3_235b_fp8",
            serde_json::json!({"ep_size": 8, "attn_tp_size": 2, "nvl_num_gpu": 8, "routing": "uniform", "fp8": true}),
        ));
        blocks.push(arch_block(
            "NVIDIA B200",
            "glm52_vllm_nvfp4_dsa_moe_speculative",
            "glm52_nvfp4",
            serde_json::json!({"ep_size": 4, "nvl_num_gpu": 4, "draft_tokens": 5, "max_model_len": 8192}),
        ));
        let shapes: Vec<_> = build_arch_blocks(&blocks, false)
            .into_iter()
            .map(|b| {
                let shape = b
                    .predict
                    .unwrap_or_else(|| panic!("{}: {:?}", b.arch, b.error));
                (
                    shape.selector,
                    shape.groups,
                    shape.query_width,
                    shape.max_model_len,
                )
            })
            .collect();
        assert_eq!(
            shapes,
            [
                // GLM-5.2 NVFP4's own max_model_len default; the others have
                // none, so their checkpoint's max_position_embeddings.
                ("iter", 1, None, Some(1_048_576)),
                ("attn", 1, None, Some(40_960)),
                ("iter", 4, None, Some(40_960)),
                ("speculative_iter", 1, Some(6), Some(8192)),
            ]
        );
    }

    /// A param a block sets changes the tree its defaults build, and a tag no
    /// contract provides is an error, not a panic.
    #[test]
    fn a_block_param_changes_the_tree_and_an_unknown_tag_is_an_error() {
        let [base, _] = <[ArchBlock; 2]>::try_from(sample_blocks()).unwrap();
        let mut short = base.arch.clone();
        short.insert("max_model_len".into(), 8192.into());
        let blocks = [
            ArchBlock {
                gpu: base.gpu.clone(),
                arch: short,
            },
            ArchBlock {
                gpu: base.gpu.clone(),
                arch: [("type".to_string(), "no_such_arch".into())]
                    .into_iter()
                    .collect(),
            },
            base,
        ];
        let built = build_arch_blocks(&blocks, false);
        let json = |b: &ArchBuild| serde_json::to_value(b.cost_manifest.as_ref()).unwrap();
        assert_eq!(built[0].error, None);
        assert_eq!(built[0].params["max_model_len"], 8192);
        assert_ne!(json(&built[0]), json(&built[2]));
        assert!(built[1].error.as_deref().unwrap().contains("no_such_arch"));
    }
}
#[test]
fn neuron_math_rejects_same_shape_with_different_rope_or_norm() {
    let mut config: serde_json::Value =
        serde_json::from_str(include_str!("../../../model/config/llama3_8b.json")).unwrap();
    assert!(ensure_neuron_math(&config).is_ok());
    config["rope_scaling"]["factor"] = serde_json::json!(4.0);
    assert!(ensure_neuron_math(&config).is_err());
    config["rope_scaling"]["factor"] = serde_json::json!(8.0);
    config["rms_norm_eps"] = serde_json::json!(1e-6);
    assert!(ensure_neuron_math(&config).is_err());
}
