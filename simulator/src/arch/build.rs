//! Build an iter-wise arch model from its [`IterArchSel`] selector — the single
//! home for the per-arch `build_configs → resolve_configs → build` chain, the
//! `ModelSpec`→cfg layer-count override, and MoE routing resolution.
//!
//! Every caller that needs a built model goes through here: the `unified` and
//! `pd` deployments call the per-arch [`dense`] / [`dense_tp`] / … builders and
//! keep the *concrete* type for their dyn-free worker factories (the cost hot
//! path stays monomorphized, L4 §4.1); the offline `iter-timing-predict` path
//! calls [`build_iter_model`], which boxes one as `dyn` (off the hot path). The
//! caller supplies `name` — the model's dotted-leaf prefix (`"unified"` / `"pd"`)
//! — so each deployment's cost manifests read naturally.

use std::path::Path;

use anyhow::{Context, Result};

use crate::arch::config::{IterArchSel, ModelSpec, RoutingKind};
use crate::arch::model_cfg::ModelCfg;
use crate::arch::moe_model_cfg::MoeModelCfg;
use crate::arch::{
    llama3_dense, llama3_dense_tp, llama3_dp_attn_tp_ffn, qwen3_moe_dp_attn_ep_ffn, DenseParallel,
    DenseTpParallel, DpAttnTpFfnParallel, IterwiseUnifiedModel, Llama3DenseModel,
    Llama3DenseTpModel, Llama3DpAttnTpFfnModel, Qwen3MoeDpAttnEpFfnModel, Qwen3MoeParallel,
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
    Ok(cfg)
}

/// Resolve the MoE routing distribution the L2 MoE op samples against from the
/// selector's [`RoutingKind`]: `uniform` spreads load evenly over `num_experts`;
/// `random` draws a `seed`-seeded deterministic skew (0 when unset). Infallible —
/// both kinds are valid for any `num_experts`. The model layer's `power_law` /
/// explicit `from_profile` are not yet wired to config.
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
    gpu: &str,
    name: &str,
    bridge: &PerfApiBridge,
) -> Result<Qwen3MoeDpAttnEpFfnModel> {
    let model_cfg = moe_model_cfg(model_spec)?;
    let routing = resolve_routing(routing_kind, routing_seed, model_cfg.num_experts);
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
        .context("building Qwen3-MoE DP-attn EP-ffn model (often a missing profile.db row)")
}

/// Build ONE iter-wise arch model from its selector, boxed as `dyn`. The
/// model-only seam the offline `iter-timing-predict` path uses (it evaluates
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
        } => Box::new(qwen3_moe(
            model,
            *attn_tp_size,
            *ep_size,
            *hp_size,
            *nvl_num_gpu,
            *routing,
            *routing_seed,
            gpu,
            name,
            bridge,
        )?),
    })
}
