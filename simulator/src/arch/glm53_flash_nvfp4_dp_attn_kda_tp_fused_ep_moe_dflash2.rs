//! GLM-5.3-Flash NVFP4 + DFlash2 on one 8x B200 node: attention DP, the KDA
//! layers tensor-parallel in groups, MoE EP8 behind the fixed-pattern
//! exchange, and the KDA gate up-projections merged. Selector tag
//! `glm53_flash_nvfp4_dp_attn_kda_tp_fused_ep_moe_dflash2`.
//!
//! Not a vLLM layout. It is a fixed configuration of
//! `glm53_flash_vllm_nvfp4_dp_attn_ep_moe_dflash2`
//! ([`crate::arch::glm53_flash_vllm_fp8_dp_attn_ep_moe`] verifying a DFlash2
//! draft): [`target_selector`] expands this tag into that selector with the
//! design's knobs set, so both build the same cost tree from the same leaves.
//!
//! # Design
//!
//! * **Attention.** Every GPU is its own DP rank with its own scheduler and KV
//!   partition. With `kda_tp_gpus = b > 1` each of the 34 KDA layers splits
//!   its heads and projections over a group of `b` owner ranks: an all-gather
//!   brings every group GPU the group's KDA inputs before the layer, and a
//!   reduce-scatter returns each owner the sum of the output-projection
//!   partials after it (the fixed-pattern MNNVL exchange, bf16 rows of
//!   `hidden` values). The 11 DSA layers stay on the request's owner rank.
//! * **Group size.** Static per deployment; the presets set it per load
//!   point.
//! * **f_b / g_b merged.** Each KDA layer runs its two low-rank gate
//!   up-projections as one batched GEMM.
//! * **Expert exchange.** The fixed-pattern MNNVL exchange as measured:
//!   quantized all-gather dispatch and reduce-scatter combine, each its own
//!   launch, the dispatch quant separate.
//! * **mHC boundaries.** `mhc_boundary` ([`MhcBoundary`]): vLLM's kernels
//!   (default), or vLLM's rows less `mhc_fused_saving_us` per boundary, the
//!   one-launch kernel's saving measured against TileLang in one job at the
//!   deployment's rows per rank (`fused_less_saving`).
//! * **Router gate.** Once per MoE layer (`router_gate_once`, default on: the
//!   vLLM fork's GateLinear since #55736).
//!
//! # Where each component's time comes from
//!
//! | component | source |
//! |---|---|
//! | KDA kernels and projections at TP `b` | measured (TP-`b` rows) |
//! | f_b / g_b batched GEMM | measured (`batched_gemm` `torch_bmm_packed_lhs`) |
//! | DSA, dense FFN, shared expert, router gate, mHC pre/post | measured |
//! | KDA input all-gather / output reduce-scatter | measured at the group size `b` (`row_all_gather` / `moe_ep_reduce_scatter`) |
//! | routed experts, dispatch quant, router top-k select | measured |
//! | exchange dispatch + combine | measured (`moe_ep_quantized_all_gather` / `moe_ep_reduce_scatter` `mnnvl_lamport`, B200 EP8) |
//! | fused mHC boundary less the saving | extrapolated (labeled in the cost log) |
//! | DFlash2 draft (TP1 per rank) | measured |

use anyhow::{bail, ensure, Result};

use crate::arch::config::IterArchSel;

/// The selector tag.
pub const ARCH_TAG: &str = "glm53_flash_nvfp4_dp_attn_kda_tp_fused_ep_moe_dflash2";

/// How the mHC post/pre boundaries are priced (`mhc_boundary`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MhcBoundary {
    /// vLLM's kernels, unfused (the backends file picks TileLang).
    #[default]
    Unfused,
    /// Fused: vLLM's TileLang rows less `mhc_fused_saving_us` per boundary.
    FusedLessSaving,
}

/// The `mhc_boundary` choices the launcher schema advertises.
pub const MHC_BOUNDARIES: [&str; 2] = ["unfused", "fused_less_saving"];

/// How the KDA group's input all-gather + output reduce-scatter is priced
/// (`kda_tp_comm_pricing`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KdaTpCommPricing {
    /// The two collectives from the fixed-pattern MNNVL rows.
    #[default]
    GatherScatter,
}

/// The `kda_tp_comm_pricing` choices the launcher schema advertises.
pub const KDA_TP_COMM_PRICINGS: [&str; 1] = ["gather_scatter"];

/// How the expert exchange is priced (`expert_exchange`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertExchange {
    /// The fixed-pattern MNNVL exchange as measured.
    #[default]
    FixedMeasured,
}

/// The `expert_exchange` choices the launcher schema advertises.
pub const EXPERT_EXCHANGES: [&str; 1] = ["fixed_measured"];

/// `sel` (this tag) as the `glm53_flash_vllm_nvfp4_dp_attn_ep_moe_dflash2`
/// selector it fixes (module doc).
pub fn target_selector(sel: &IterArchSel) -> Result<IterArchSel> {
    let IterArchSel::Glm53FlashNvfp4DpAttnKdaTpFusedEpMoeDflash2 {
        model,
        ep_size,
        kda_tp_gpus,
        max_model_len,
        routing,
        routing_seed,
        expert_popularity_file,
        token_corpus_file,
        cudagraph_capture_sizes,
        mla_layout_copies,
        indexer_max_logits_mb,
        draft_tokens,
        kda_state_dtype,
        kda_spec_verify,
        mhc_boundary,
        mhc_fused_saving_us,
        kda_tp_comm_pricing: KdaTpCommPricing::GatherScatter,
        expert_exchange: ExpertExchange::FixedMeasured,
        dsa_glue,
        dsa_hidden_gemm,
        dsa_hidden_gemm_backend,
        dsa_q_lora_gemm,
        dsa_q_lora_gemm_backend,
        dsa_q_absorb_fp8_backend,
        dsa_topk_page_table_backend,
        router_gate_once,
        moe_input_copy,
        moe_combine_add_in_mhc_post,
        moe_shared_overlap,
        moe_shared_pair_file,
        moe_shared_pair_column,
        draft_attention,
        draft_topk,
        draft_context_kv_store,
        draft_qk_norm_rope_kv_write,
        draft_conv_residual_norm,
    } = sel
    else {
        bail!("{sel:?} is not {ARCH_TAG}");
    };
    ensure!(
        *kda_tp_gpus >= 1 && ep_size % kda_tp_gpus == 0,
        "{ARCH_TAG}: kda_tp_gpus {kda_tp_gpus} must divide ep_size {ep_size}"
    );
    let mhc_fused_saving_us = match (mhc_boundary, mhc_fused_saving_us) {
        (MhcBoundary::Unfused, None) => None,
        (MhcBoundary::FusedLessSaving, Some(us)) if *us >= 0.0 => Some(*us),
        (MhcBoundary::FusedLessSaving, _) => {
            bail!("{ARCH_TAG}: mhc_boundary fused_less_saving needs mhc_fused_saving_us >= 0")
        }
        (MhcBoundary::Unfused, Some(_)) => {
            bail!("{ARCH_TAG}: mhc_fused_saving_us is for mhc_boundary fused_less_saving")
        }
    };
    Ok(IterArchSel::Glm53FlashVllmNvfp4DpAttnEpMoeDflash2 {
        model: model.clone(),
        ep_size: *ep_size,
        kda_tp_gpus: *kda_tp_gpus,
        kda_gate_b_merged: true,
        max_model_len: *max_model_len,
        routing: *routing,
        routing_seed: *routing_seed,
        expert_popularity_file: expert_popularity_file.clone(),
        token_corpus_file: token_corpus_file.clone(),
        cudagraph_capture_sizes: cudagraph_capture_sizes.clone(),
        mla_layout_copies: *mla_layout_copies,
        indexer_max_logits_mb: *indexer_max_logits_mb,
        draft_tokens: *draft_tokens,
        kda_state_dtype: *kda_state_dtype,
        kda_spec_verify: *kda_spec_verify,
        expert_exchange:
            crate::arch::glm53_flash_vllm_fp8_dp_attn_ep_moe::ExpertExchange::FixedPattern,
        mhc_fused_saving_us,
        dsa_glue: *dsa_glue,
        dsa_hidden_gemm: *dsa_hidden_gemm,
        dsa_hidden_gemm_backend: dsa_hidden_gemm_backend.clone(),
        dsa_q_lora_gemm: *dsa_q_lora_gemm,
        dsa_q_lora_gemm_backend: dsa_q_lora_gemm_backend.clone(),
        dsa_q_absorb_fp8_backend: dsa_q_absorb_fp8_backend.clone(),
        dsa_topk_page_table_backend: dsa_topk_page_table_backend.clone(),
        router_gate_once: *router_gate_once,
        moe_input_copy: *moe_input_copy,
        moe_combine_add_in_mhc_post: *moe_combine_add_in_mhc_post,
        moe_shared_overlap: *moe_shared_overlap,
        moe_shared_pair_file: moe_shared_pair_file.clone(),
        moe_shared_pair_column: moe_shared_pair_column.clone(),
        draft_attention: *draft_attention,
        draft_topk: *draft_topk,
        draft_context_kv_store: *draft_context_kv_store,
        draft_qk_norm_rope_kv_write: *draft_qk_norm_rope_kv_write,
        draft_conv_residual_norm: *draft_conv_residual_norm,
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::arch::build::glm53_flash_vllm_nvfp4_dp_attn_ep_moe_dflash2;
    use crate::arch::contract::SpeculativeUnifiedModel;
    use crate::timing::PerfApiBridge;

    fn tag(kda_tp_gpus: u16) -> serde_json::Value {
        serde_json::json!({
            "type": ARCH_TAG,
            "model_config": Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("model/config/glm53_flash_nvfp4.json"),
            "fp8": false,
            "kda_tp_gpus": kda_tp_gpus,
            "routing": "uniform",
        })
    }

    /// Catches the tag drifting from the knob-by-knob selector: the two must
    /// compile the same slots, and each group size its own tree.
    #[test]
    fn the_tag_builds_the_knob_by_knob_selector() {
        let bridge = PerfApiBridge::new_uninit_for_test();
        bridge.enable_enumerate();
        let mut seen = std::collections::HashSet::new();
        for b in [1_u16, 2, 4, 8] {
            let knobs: IterArchSel = serde_json::from_value(serde_json::json!({
                "type": "glm53_flash_vllm_nvfp4_dp_attn_ep_moe_dflash2",
                "model_config": Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("model/config/glm53_flash_nvfp4.json"),
                "fp8": false,
                "ep_size": 8,
                "routing": "uniform",
                "mla_layout_copies": false,
                "kda_state_dtype": "bf16",
                "expert_exchange": "fixed_pattern",
                "kda_tp_gpus": b,
                "kda_gate_b_merged": true,
                "router_gate_once": true,
            }))
            .unwrap();
            let sel: IterArchSel = serde_json::from_value(tag(b)).unwrap();
            let build = |sel: &IterArchSel| {
                let model =
                    glm53_flash_vllm_nvfp4_dp_attn_ep_moe_dflash2(sel, "NVIDIA B200", "t", &bridge)
                        .unwrap();
                (
                    serde_json::to_string(&model.cost_log_manifest()).unwrap(),
                    model.num_attn_dp_groups(),
                )
            };
            let ours = build(&target_selector(&sel).unwrap());
            assert_eq!(ours, build(&knobs), "kda_tp_gpus {b}");
            // KDA-only TP keeps each request's DSA on its owner rank.
            assert_eq!(ours.1, 8, "kda_tp_gpus {b}");
            assert!(
                seen.insert(ours.0),
                "kda_tp_gpus {b} built another b's tree"
            );
        }
    }

    /// Catches a group size the EP group cannot split into, a tag with no
    /// routing (it has no synthetic default), and a saving without its
    /// boundary mode or the reverse.
    #[test]
    fn the_tag_rejects_what_it_cannot_build() {
        let sel: IterArchSel = serde_json::from_value(tag(3)).unwrap();
        assert!(target_selector(&sel)
            .unwrap_err()
            .to_string()
            .contains("must divide ep_size 8"));
        let mut no_routing = tag(8);
        no_routing.as_object_mut().unwrap().remove("routing");
        assert!(serde_json::from_value::<IterArchSel>(no_routing).is_err());
        for (boundary, saving) in [
            ("fused_less_saving", None),
            ("fused_less_saving", Some(-1.0)),
            ("unfused", Some(1.0)),
        ] {
            let mut bad = tag(8);
            bad["mhc_boundary"] = boundary.into();
            bad["mhc_fused_saving_us"] = saving.into();
            let sel: IterArchSel = serde_json::from_value(bad).unwrap();
            assert!(target_selector(&sel).is_err(), "{boundary} {saving:?}");
        }
    }

    /// Catches the mHC choice reaching the wrong knob of the target selector.
    #[test]
    fn the_tag_maps_its_mhc_choice() {
        let saving = |extra: serde_json::Value| {
            let mut sel = tag(8);
            sel.as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let sel: IterArchSel = serde_json::from_value(sel).unwrap();
            match target_selector(&sel).unwrap() {
                IterArchSel::Glm53FlashVllmNvfp4DpAttnEpMoeDflash2 {
                    mhc_fused_saving_us,
                    ..
                } => mhc_fused_saving_us,
                other => panic!("{other:?}"),
            }
        };
        assert_eq!(saving(serde_json::json!({})), None);
        assert_eq!(
            saving(serde_json::json!({
                "mhc_boundary": "fused_less_saving",
                "mhc_fused_saving_us": 1.796,
            })),
            Some(1.796)
        );
    }
}
