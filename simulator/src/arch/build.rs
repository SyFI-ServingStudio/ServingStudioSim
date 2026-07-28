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

use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::arch::config::{AttnArchSel, FfnArchSel, IterArchSel, ModelSpec, RoutingKind};
use crate::arch::kimi_model_cfg::KimiModelCfg;
use crate::arch::model_cfg::ModelCfg;
use crate::arch::moe_model_cfg::MoeModelCfg;
use crate::arch::{
    kimi_k3_kda_mla, llama3_dense, llama3_dense_tp, llama3_dp_attn_tp_ffn, qwen3_attn_layerwise,
    qwen3_ffn_moe_layerwise, qwen3_moe_dp_attn_ep_ffn, AttnLayerwiseModel, DenseParallel,
    DenseTpParallel, DpAttnTpFfnParallel, FfnLayerwiseModel, IterwiseUnifiedModel,
    KimiK3KdaMlaModel, KimiK3Parallel, Llama3DenseModel, Llama3DenseTpModel,
    Llama3DpAttnTpFfnModel, Qwen3AttnLayerwiseModel, Qwen3AttnParallel, Qwen3FfnMoeLayerwiseModel,
    Qwen3FfnMoeParallel, Qwen3MoeDpAttnEpFfnModel, Qwen3MoeParallel,
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

/// `ModelSpec` → [`KimiModelCfg`] (Kimi-K3 hybrid KDA+MLA configs carry the MLA
/// LoRA dims, the KDA conv width, and the MoE `routed_expert_hidden_size`). The
/// layer-count override scales the MLA/KDA split proportionally.
pub fn kimi_model_cfg(model_spec: &ModelSpec) -> Result<KimiModelCfg> {
    let mut cfg = KimiModelCfg::from_json(Path::new(&model_spec.model_config))?;
    if let Some(n) = model_spec.sim_num_layers.or(model_spec.num_layers) {
        cfg = cfg.with_num_layers(n);
    }
    Ok(cfg.with_fp8(model_spec.fp8))
}

/// Build the Kimi-K3 hybrid KDA+MLA DP-attention + EP-FFN model.
#[allow(clippy::too_many_arguments)]
pub fn kimi_k3(
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
) -> Result<KimiK3KdaMlaModel> {
    if attn_tp_size != 1 {
        bail!(
            "kimi_k3_kda_mla: attn_tp_size must be 1 — MLA absorbed-weight decode has ONE \
             shared compressed KV head (MQA), which cannot be TP-split; use DP attention \
             (got attn_tp_size={attn_tp_size})"
        );
    }
    let model_cfg = kimi_model_cfg(model_spec)?;
    let routing = resolve_routing(routing_kind, routing_seed, model_cfg.num_experts);
    let parallel = KimiK3Parallel {
        attn_tp_size,
        ep_size,
        hp_size,
        nvl_num_gpu,
        gpu_name: gpu.to_string(),
    };
    let resolved = kimi_k3_kda_mla::resolve_configs(&kimi_k3_kda_mla::build_configs(
        &model_cfg, &parallel, &routing,
    ));
    kimi_k3_kda_mla::build(name.to_string(), resolved, bridge)
        .context("building Kimi-K3 KDA+MLA model (often a missing profile.db row)")
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
    let routing = resolve_routing(routing_kind, routing_seed, model_cfg.num_experts);
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
        IterArchSel::KimiK3KdaMla {
            model,
            attn_tp_size,
            ep_size,
            hp_size,
            nvl_num_gpu,
            routing,
            routing_seed,
        } => Box::new(kimi_k3(
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
        FfnArchSel::DeepseekFfnMoe { .. } => bail!(
            "timing-predict ffn: only the qwen3_ffn_moe arch has a layer-wise \
             predict path (got deepseek_ffn_moe)"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The moesim-faithful comm sizes (`ref/moesim-rs/.../standard_moe.rs`): attn→ffn
    /// is the attention output `q_dim·bpe`; ffn→attn is the QKV projection
    /// `(q_dim + 2·kv_dim)·bpe`. Pure arithmetic on the model dims — no bridge.
    #[test]
    fn afd_comm_bytes_match_moesim_formulas() {
        let model = MoeModelCfg::qwen3_235b();
        let bpe = model.dtype.size_bytes() as u64;
        let q_dim = model.num_qo_heads as u64 * model.head_dim as u64;
        let kv_dim = model.num_kv_heads as u64 * model.head_dim as u64;

        let attn_cfgs = crate::arch::qwen3_attn_layerwise::build_configs(
            &model,
            &Qwen3AttnParallel {
                attn_tp_size: 4,
                gpu_name: "H200".to_string(),
            },
        );
        // attn→ffn outgoing bytes: the attention output, q_dim·bpe.
        assert_eq!(attn_cfgs.attn_to_ffn_bytes_per_token, q_dim * bpe);
        // total KV bytes: 2 (k+v) × kv_heads × head_dim × kv_dtype × layers.
        assert_eq!(
            attn_cfgs.total_kv_bytes_per_token,
            2 * model.num_kv_heads as u64
                * model.head_dim as u64
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

        let q_dim = model.num_qo_heads as u64 * model.head_dim as u64;
        let kv_dim = model.num_kv_heads as u64 * model.head_dim as u64;

        // --- attn side ---
        let attn_cfgs = crate::arch::qwen3_attn_layerwise::build_configs(
            &model,
            &Qwen3AttnParallel {
                attn_tp_size: 4,
                gpu_name: "H200".to_string(),
            },
        );
        // Handoff + KV at fp8 = 1 byte/elem.
        assert_eq!(attn_cfgs.attn_to_ffn_bytes_per_token, q_dim * 1);
        assert_eq!(
            attn_cfgs.total_kv_bytes_per_token,
            2 * model.num_kv_heads as u64 * model.head_dim as u64 * 1 * model.num_layers as u64
        );
        // The attn block carries the base dtype + fp8 flag (op owns the preset);
        // its GEMMs use deepgemm, KV cache reads fp8.
        assert_eq!(attn_cfgs.attn_block.dtype, DType::Bf16);
        assert!(attn_cfgs.attn_block.fp8);
        assert_eq!(attn_cfgs.attn_block.gemm_backends, vec!["deepgemm"]);
        assert_eq!(attn_cfgs.attn_block.kv_dtype(), DType::Fp8E4m3);

        // --- ffn side ---
        let routing = RoutingDistribution::uniform(model.num_experts);
        let ffn_cfgs = crate::arch::qwen3_ffn_moe_layerwise::build_configs(
            &model,
            &crate::arch::qwen3_ffn_moe_layerwise::Qwen3FfnMoeParallel {
                attn_tp_size: 4,
                ep_size: 8,
                nvl_num_gpu: 8,
                gpu_name: "H200".to_string(),
            },
            &routing,
        );
        // Symmetric QKV-projection handoff at fp8.
        assert_eq!(
            ffn_cfgs.ffn_to_attn_bytes_per_token,
            (q_dim + 2 * kv_dim) * 1
        );
        // GEMM roles fp8+deepgemm; RMSNorm roles stay bf16.
        assert_eq!(ffn_cfgs.pre_attn.compute_dtype, DType::Fp8E4m3);
        assert_eq!(ffn_cfgs.pre_attn.dtype, DType::Bf16); // input_norm stays bf16
        assert_eq!(ffn_cfgs.pre_attn.gemm_backends, vec!["deepgemm"]);
        assert_eq!(ffn_cfgs.post_attn.compute_dtype, DType::Fp8E4m3);
        assert_eq!(ffn_cfgs.post_attn.dtype, DType::Bf16); // post_norm stays bf16
        assert_eq!(ffn_cfgs.moe_expert_compute.dtype, DType::Fp8E4m3); // no norm inside
        assert_eq!(
            ffn_cfgs.moe_expert_compute.grouped_gemm_backends,
            vec!["deepgemm"]
        );
        assert_eq!(ffn_cfgs.lm_head.dtype, DType::Fp8E4m3);
        assert_eq!(ffn_cfgs.lm_head.backends, vec!["deepgemm"]);
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

        let q_dim = model.num_qo_heads as u64 * model.head_dim as u64;
        let attn_cfgs = crate::arch::qwen3_attn_layerwise::build_configs(
            &model,
            &Qwen3AttnParallel {
                attn_tp_size: 4,
                gpu_name: "H200".to_string(),
            },
        );
        assert_eq!(attn_cfgs.attn_to_ffn_bytes_per_token, q_dim * 2);
        assert!(!attn_cfgs.attn_block.fp8);
        assert_eq!(
            attn_cfgs.attn_block.gemm_backends,
            vec!["torch", "torch_linear"]
        );
        assert_eq!(attn_cfgs.attn_block.kv_dtype(), DType::Bf16);
    }
}
