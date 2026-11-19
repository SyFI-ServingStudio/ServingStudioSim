//! GLM-5.3-Flash NVFP4 + DFlash2 on one 8x B200 node: attention DP, the KDA
//! layers tensor-parallel in groups, MoE EP8 behind the fixed-pattern
//! exchange, and the design's kernels. Selector tag
//! `glm53_flash_nvfp4_dp_attn_kda_tp_fused_ep_moe_dflash2`.
//!
//! Not a vLLM layout. It is a fixed configuration of
//! `glm53_flash_vllm_nvfp4_dp_attn_ep_moe_dflash2`
//! ([`crate::arch::glm53_flash_vllm_fp8_dp_attn_ep_moe`] verifying a DFlash2
//! draft): [`target_selector`] expands this tag into that selector with every
//! mechanism below set, and [`DESIGN_KERNELS`] replaces the fork's kernels, so
//! both build the same cost tree from the same leaves. Only the deployment's
//! shape is a parameter: the KDA group size, the context cap, the routing, the
//! CUDA-graph sizes and the draft depth.
//!
//! # Design
//!
//! Each line is one mechanism beyond the vLLM fork (`*_dflash2` with its
//! defaults):
//!
//! * **RecoverSSM verify.** KDA verifies the drafts with RecoverSSM on a bf16
//!   state pool: one state per request, a conv + verify launch and one commit
//!   per KDA cache group, not a state per position.
//! * **KDA tensor parallelism.** With `kda_tp_gpus = b > 1` each of the 34 KDA
//!   layers splits its heads and projections over a group of `b` owner ranks:
//!   an all-gather brings every group GPU the group's KDA inputs before the
//!   layer, and a reduce-scatter returns each owner the sum of the
//!   output-projection partials after it (the fixed-pattern MNNVL exchange,
//!   bf16 rows of `hidden` values). The 11 DSA layers stay on the request's
//!   owner rank. Static per deployment; the fastest `b` falls with load.
//! * **f_b / g_b merged.** Each KDA layer runs its two low-rank gate
//!   up-projections as one batched GEMM.
//! * **Expert exchange.** The fixed-pattern MNNVL exchange as measured:
//!   quantized all-gather dispatch and reduce-scatter combine, each its own
//!   launch, the dispatch quant separate.
//! * **Fused routed MoE.** The routed experts run FC1 and FC2 in one CuTe DSL
//!   kernel (`cutedsl_routed_fused_fc1_fc2_sm100`).
//! * **Shared expert capped, dispatch triggered.** The aux-stream shared
//!   expert's GEMMs run on 64 SMs and the dispatch releases its consumers
//!   early; the time the shared expert still adds is the measured
//!   `fused_cap64_trig` column of the shipped exposure table.
//! * **KDA in_proj / o_proj.** FlashInfer's CuTe DSL bf16 GEMM, or vLLM's
//!   linear where that is faster at the shape.
//! * **Fused conv + verify, commit v4.** The RecoverSSM conv + verify and the
//!   commit on the persistent CUDA kernels (`cuda_persistent_wy_mma`,
//!   `cuda_persistent_ws`).
//! * **DSA fused chain.** The hidden GEMM writes the fp32 split state in its
//!   epilogue, the q-LoRA GEMM concatenates and Hadamard-quantizes in its
//!   epilogue, `q_absorb` quantizes to fp8 in its epilogue, and the decode
//!   top-k writes the final page table in one launch.
//! * **DSA decode top-k.** DeepSelect's long-row kernel behind the fork's
//!   cooperative kernel.
//! * **Draft chain.** FA4 without splits from 16 requests per launch,
//!   threshold top-k at small row counts, one launch for the context K/V
//!   store, one for q/k norm + RoPE + the block's K/V write, and each conv
//!   finish fused with the residual RMSNorm after it.
//! * **Shared + routed add in the next mHC post.** The MoE block's add runs
//!   inside the next mHC post kernel.
//! * **One-launch mHC boundary.** Every mHC post + pre boundary, and the
//!   after-MoE boundary that also sums the MoE outputs, runs the one-launch
//!   CUDA kernel where its measured row beats TileLang's at the boundary's
//!   rows per rank (`cuda_one_launch`; on B200 that is a few dozen rows and
//!   below). The after-MoE boundary is one kernel either way: the add is
//!   inside it, priced once, by whichever kernel's row is faster.
//! * **Router gate.** Once per MoE layer, on the fork's GateLinear.
//! * **KDA prefill.** FlashInfer's CuTe persistent chunked prefill and Dao's
//!   channel-last short conv.
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
//! | shared-expert exposure | measured table (`model/glm53_flash_nvfp4_b200_ep8_moe_shared_exposure.json`) |
//! | DFlash2 draft (TP1 per rank) | measured |
//! | mHC boundaries | measured, faster of `vllm_tilelang(_post_addend)` and `cuda_one_launch` per row count |

use anyhow::{bail, ensure, Result};

use crate::arch::config::{
    DsaGluePricing, DsaHiddenGemmMode, DsaQLoraGemmMode, IterArchSel, KdaStateDtype,
};
use crate::arch::glm53_flash_vllm_fp8_dp_attn_ep_moe::{
    ExpertExchange, MoeSharedOverlap, MoeSharedPair,
};
use crate::arch::glm53_flash_vllm_nvfp4_dflash2::Glm53FlashDflash2Kernels;
use crate::arch::glm53_vllm_nvfp4_dsa_moe_dflash2::{DraftAttention, DraftTopk};
use crate::worklet::KdaSpecVerify;

/// The selector tag.
pub const ARCH_TAG: &str = "glm53_flash_nvfp4_dp_attn_kda_tp_fused_ep_moe_dflash2";

/// The design's kernels where the worklets take a choice (module doc).
pub const DESIGN_KERNELS: Glm53FlashDflash2Kernels = Glm53FlashDflash2Kernels {
    router_gate: &["vllm_gate_linear"],
    kda_chunk_prefill: &["flashinfer_cute_persistent"],
    kda_conv_prefill: &["dao_channellast"],
    kda_in_out_proj: &["flashinfer_cutedsl_bf16", "torch_linear_vllm"],
    kda_conv_verify: &["cuda_persistent_wy_mma"],
    kda_commit: &["cuda_persistent_ws"],
    routed_moe: &["cutedsl_routed_fused_fc1_fc2_sm100"],
    dsa_topk_decode: &["deep_select"],
    // Each boundary prices at the faster measured row of the TileLang kernel
    // and the one-launch CUDA kernel at its rows per rank.
    mhc_post_pre: &["vllm_tilelang", "cuda_one_launch"],
    mhc_sum_post_pre: &["vllm_tilelang_post_addend", "cuda_one_launch"],
};

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
        indexer_max_logits_mb,
        draft_tokens,
    } = sel
    else {
        bail!("{sel:?} is not {ARCH_TAG}");
    };
    ensure!(
        *kda_tp_gpus >= 1 && ep_size % kda_tp_gpus == 0,
        "{ARCH_TAG}: kda_tp_gpus {kda_tp_gpus} must divide ep_size {ep_size}"
    );
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
        mla_layout_copies: false,
        indexer_max_logits_mb: *indexer_max_logits_mb,
        draft_tokens: *draft_tokens,
        kda_state_dtype: KdaStateDtype::Bf16,
        kda_spec_verify: KdaSpecVerify::Recoverssm,
        expert_exchange: ExpertExchange::FixedPattern,
        mhc_fused_saving_us: None,
        dsa_glue: DsaGluePricing::ForkLaunchList,
        dsa_hidden_gemm: DsaHiddenGemmMode::Fp32SplitStateWriter,
        dsa_hidden_gemm_backend: Some("triton".into()),
        dsa_q_lora_gemm: DsaQLoraGemmMode::ConcatHadamardEpilogue,
        dsa_q_lora_gemm_backend: Some("triton".into()),
        dsa_q_absorb_fp8_backend: Some("cuda_tcgen05".into()),
        dsa_topk_page_table_backend: Some("cuda_row_scan_compact".into()),
        router_gate_once: true,
        moe_input_copy: false,
        moe_combine_add_in_mhc_post: true,
        moe_shared_overlap: MoeSharedOverlap::Exposed,
        moe_shared_pair_file: None,
        moe_shared_pair_column: Some(MoeSharedPair::FUSED_CAP64_TRIGGER.into()),
        draft_attention: DraftAttention::Fa4Fp8NosplitFrom16,
        draft_topk: DraftTopk::ThresholdSmallRows,
        draft_context_kv_store: true,
        draft_qk_norm_rope_kv_write: true,
        draft_conv_residual_norm: true,
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::arch::build::glm53_flash_nvfp4_dflash2_on;
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

    /// Catches the tag drifting from the knob-by-knob selector on the design's
    /// kernels: the two must compile the same slots, and each group size its
    /// own tree.
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
                "kda_state_dtype": "bf16",
                "kda_spec_verify": "recoverssm",
                "expert_exchange": "fixed_pattern",
                "kda_tp_gpus": b,
                "kda_gate_b_merged": true,
                "dsa_hidden_gemm": "fp32_split_state_writer",
                "dsa_hidden_gemm_backend": "triton",
                "dsa_q_lora_gemm": "concat_hadamard_epilogue",
                "dsa_q_lora_gemm_backend": "triton",
                "dsa_q_absorb_fp8_backend": "cuda_tcgen05",
                "dsa_topk_page_table_backend": "cuda_row_scan_compact",
                "moe_combine_add_in_mhc_post": true,
                "moe_shared_pair_column": "fused_cap64_trig",
                "draft_attention": "fa4_fp8_nosplit_from16",
                "draft_topk": "threshold_small_rows",
                "draft_context_kv_store": true,
                "draft_qk_norm_rope_kv_write": true,
                "draft_conv_residual_norm": true,
            }))
            .unwrap();
            let sel: IterArchSel = serde_json::from_value(tag(b)).unwrap();
            let build = |sel: &IterArchSel| {
                let model =
                    glm53_flash_nvfp4_dflash2_on(sel, &DESIGN_KERNELS, "NVIDIA B200", "t", &bridge)
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

    /// Catches a group size the EP group cannot split into and a tag with no
    /// routing (it has no synthetic default).
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
    }
}
