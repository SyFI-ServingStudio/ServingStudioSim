//! GLM-5.3-Flash NVFP4 + DFlash2 on one 8x B200 node: attention DP, the KDA
//! layers tensor-parallel in groups, MoE EP8 behind a fused direct-write
//! exchange, and the KDA gate up-projections merged. Selector tag
//! `glm53_flash_nvfp4_dp_attn_kda_tp_fused_ep_moe_dflash2`.
//!
//! Not a vLLM layout. It is a fixed configuration of
//! `glm53_flash_vllm_nvfp4_dp_attn_ep_moe_dflash2`
//! ([`crate::arch::glm53_flash_vllm_fp8_dp_attn_ep_moe`] verifying a DFlash2
//! draft): [`target_selector`] expands this tag into that selector with the
//! design's knobs set, so both build the same cost tree from the same leaves.
//! What-if knobs that did not pay off (the sparse exchange and expert
//! placement, the narrowed combine row, the all-to-all and its latency
//! scales, DSA tensor parallelism, the NCCL-only collective) and the mHC
//! fixed-cost what-if are not exposed.
//!
//! # Design
//!
//! * **Attention.** Every GPU is its own DP rank with its own scheduler and KV
//!   partition. With `kda_tp_gpus = b > 1` the ranks form `8 / b` groups that
//!   share a scheduler; each of the 34 KDA layers splits its heads and
//!   projections over the group's `b` GPUs: an all-gather brings every group
//!   GPU the group's KDA inputs before the layer, and a reduce-scatter returns
//!   each owner the sum of the output-projection partials after it. The two
//!   are separate launches with the KDA compute between them; one all-reduce
//!   cannot replace them, since each input exists only on its owner rank. The
//!   11 DSA (MLA + indexer) layers stay on the
//!   request's owner rank (`attn_tp_layers: kda_only`). Everything outside
//!   attention is sequence-parallel: each rank owns a slice of its group's
//!   rows.
//! * **Group size.** Static per deployment. The 2026-11-12 sweep
//!   (`logs/20261112_4_flash_all_stacked_refit`, k7 decode trace, corpus
//!   routing) and its rerun with the KDA pair priced as all-gather +
//!   reduce-scatter (`logs/20261112_6_kda_tp_agrs`) picked `b = 8` at 1-32
//!   concurrent requests, `b = 4` at 64, and `b = 1` (off) from 128 up,
//!   including the HiSparse c768 / c1280 points; the presets set it per load
//!   point.
//! * **KDA all-gather and reduce-scatter.** The fixed-pattern MNNVL exchange
//!   (`attn_tp_collective: fixed_gather_scatter`): bf16 rows of `hidden`
//!   values, each owner reading its peers' fabric-mapped buffers.
//! * **f_b / g_b merged.** Each KDA layer runs its two low-rank gate
//!   up-projections as one batched GEMM (`kda_gate_b_merged`).
//! * **Expert exchange.** `expert_exchange` ([`ExpertExchange`]): `fixed_measured`
//!   (default) is the fixed-pattern MNNVL exchange as measured (quantized
//!   all-gather dispatch + push reduce-scatter combine, each its own launch,
//!   dispatch quant separate); `fused_fit` is the design below, priced by the
//!   analytic fit. `fused_fit`: `ep_comm: fused_allgather` on MNNVL: each rank writes
//!   its NVFP4 rows (activations, block scales, top-k ids and weights) into
//!   every peer's memory in one fixed-pattern store, the experts run, and the
//!   bf16 partial sums go back by a reduce-scatter that each owner sums. The
//!   dispatch store is assumed fused into the activation-quant kernel, and
//!   the combine read and its sum into the next op.
//! * **mHC boundaries.** `mhc_boundary` ([`MhcBoundary`]): vLLM's kernels,
//!   unfused (default; the presets pick the TileLang ones), or fused into one
//!   launch per boundary, priced either from the faster DB row of TileLang
//!   and the measured one-launch kernel (`fused_rows`) or as TileLang less
//!   the one-launch kernel's saving measured against it in the same job
//!   (`fused_less_saving` + `mhc_fused_saving_us`, set per load point).
//! * **KDA group communication pricing.** `kda_tp_comm_pricing`
//!   ([`KdaTpCommPricing`]): the two collectives from fixed-pattern rows
//!   (default, conservative) or one borrowed all-reduce (optimistic); the
//!   real cost lies between them until dedicated all-gather rows exist.
//! * **Router gate.** Once per MoE layer (`router_gate_once`, default on: the
//!   vLLM fork's GateLinear since #55736).
//! * **Routed-input copy.** `moe_input_copy` (default on) prices a bf16
//!   [rows, hidden] copy ahead of the dispatch quant (`moe.input_glue`), a
//!   launch seen in the 2026-09 FP8 TP4/EP4 capture. The fork's NVFP4 naive
//!   DP/EP path quantizes the hidden directly and, with the TRT-LLM / CuTe DSL
//!   experts, skips the post-gather scale swizzle; `false` drops the leaf.
//!
//! # Where each component's time comes from
//!
//! 实测 = a `profile.db` row at this shape; 拟合 = a fit to measured rows;
//! 外推 = beyond any row (labeled `extrapolated` in the cost log).
//!
//! | component | source |
//! |---|---|
//! | KDA kernels and projections at TP `b` | 实测 (TP-`b` rows) |
//! | f_b / g_b batched GEMM | 实测 (`batched_gemm` `torch_bmm_packed_lhs`) |
//! | DSA, dense FFN, shared expert, router gate (GateLinear), mHC pre/post (TileLang) | 实测 |
//! | KDA output reduce-scatter | 实测 at the group size `b` (`moe_ep_reduce_scatter`, faster of `mnnvl_lamport` push and `mnnvl_pull`) |
//! | KDA input all-gather | 实测 at the group size `b` (`row_all_gather`, faster of push and pull; 8-GPU rows Slurm 21512) |
//! | routed experts (`nvfp4_fused_moe`, routing precomputed) | 实测 |
//! | dispatch activation quant, router top-k select | 实测 (elementwise rows) |
//! | exchange, `fixed_measured` (default): dispatch + combine | 实测 (`moe_ep_quantized_all_gather` / `moe_ep_reduce_scatter` `mnnvl_lamport`, B200 EP8, Slurm 21248) |
//! | exchange, `fused_fit`: dispatch and combine per-row and link cost | 拟合 (`arch::ep_comm` B200 EP8 `moe_alltoall` fit: c 3.86 GB/s, p 2.63 ns, L 605 GB/s, w 0.9) |
//! | exchange, `fused_fit`: fixed cost per leg | 外推: 0 us (the fusion; the fit's unfused `a` is 4.11 us per leg). Optimistic, not conservative |
//! | combine owner sum of the partials (`fused_fit`) | 外推: 0 us (assumed inside the next op; the unfused exchanges bill it as an elementwise row) |
//! | DFlash2 draft (TP1 per rank) | 实测 |
//! | HiSparse host -> device miss copy (HiSparse presets only) | 外推 (`arch::hisparse` analytic model) |
//!
//! # Hooks for work in flight (not merged here)
//!
//! * **One-launch mHC rows** (worktree `wt-flash-mhc-fuse`): the
//!   `cuda_one_launch` rows behind `mhc_boundary: fused_rows` are in this
//!   tree's DB, not yet in the shared DB.

use anyhow::{bail, ensure, Result};

use crate::arch::config::IterArchSel;
use crate::arch::ep_comm::{EpCommKind, EpCommModel, FastBackend, FastCombine, FixedCombineSide};
use crate::arch::glm53_flash_vllm_fp8_dp_attn_ep_moe::{AttnTpCollective, AttnTpLayers};

/// The selector tag.
pub const ARCH_TAG: &str = "glm53_flash_nvfp4_dp_attn_kda_tp_fused_ep_moe_dflash2";

/// How the mHC post/pre boundaries are priced (`mhc_boundary`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MhcBoundary {
    /// vLLM's kernels, unfused (the backends file picks TileLang).
    #[default]
    Unfused,
    /// Fused: each boundary at the faster `profile.db` row of vLLM's TileLang
    /// call and the measured one-launch kernel; the boundary after an MoE
    /// layer adds the shared expert.
    FusedRows,
    /// Fused: vLLM's TileLang rows less `mhc_fused_saving_us` per boundary,
    /// the one-launch kernel's saving measured against TileLang in one job at
    /// the deployment's rows per rank. Used where the TileLang rows and
    /// today's re-measure disagree.
    FusedLessSaving,
}

/// The `mhc_boundary` choices the launcher schema advertises.
pub const MHC_BOUNDARIES: [&str; 3] = ["unfused", "fused_rows", "fused_less_saving"];

/// How the KDA group's input all-gather + output reduce-scatter is priced
/// (`kda_tp_comm_pricing`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KdaTpCommPricing {
    /// The two collectives from the fixed-pattern MNNVL reduce-scatter rows
    /// (`attn_tp_collective: fixed_gather_scatter`; conservative).
    #[default]
    GatherScatter,
    /// One all-reduce of the group's rows at the group size
    /// (`attn_tp_collective: fast_unbudgeted`): the same bytes in one launch,
    /// which the layout cannot run (optimistic).
    AllReduce,
}

/// The `kda_tp_comm_pricing` choices the launcher schema advertises.
pub const KDA_TP_COMM_PRICINGS: [&str; 2] = ["gather_scatter", "all_reduce"];

/// How the expert exchange is priced (`expert_exchange`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExpertExchange {
    /// The fixed-pattern MNNVL exchange as measured: quantized all-gather
    /// dispatch and reduce-scatter combine, each its own launch
    /// (`fast_allgather`, `ep_comm_model: measured`; dispatch quant separate).
    #[default]
    FixedMeasured,
    /// The fused exchange (dispatch inside the quant kernel, combine inside
    /// the next op): the analytic link fit with 0 fixed cost per leg. No
    /// kernel; optimistic.
    FusedFit,
}

/// The `expert_exchange` choices the launcher schema advertises.
pub const EXPERT_EXCHANGES: [&str; 2] = ["fixed_measured", "fused_fit"];

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
        kda_tp_comm_pricing,
        expert_exchange,
        router_gate_once,
        moe_input_copy,
        routed_moe_whatif_file,
        moe_shared_pair_file,
        moe_shared_pair_column,
        moe_shared_overlap,
        hisparse_buffer_tokens,
        hisparse_top_k,
        hisparse_host_dram_gb_per_node,
        hisparse_gpus_per_node,
        hisparse_h2d_gb_per_s,
        hisparse_exposed_io_fraction,
        hisparse_intra_step_overlap,
    } = sel
    else {
        bail!("{sel:?} is not {ARCH_TAG}");
    };
    ensure!(
        *kda_tp_gpus >= 1 && ep_size % kda_tp_gpus == 0,
        "{ARCH_TAG}: kda_tp_gpus {kda_tp_gpus} must divide ep_size {ep_size}"
    );
    let (mhc_fuse_fixed_us, mhc_one_launch) = match (mhc_boundary, mhc_fused_saving_us) {
        (MhcBoundary::Unfused, None) => (None, false),
        (MhcBoundary::FusedRows, None) => (None, true),
        (MhcBoundary::FusedLessSaving, Some(us)) if *us >= 0.0 => (Some(*us), false),
        (MhcBoundary::FusedLessSaving, _) => {
            bail!("{ARCH_TAG}: mhc_boundary fused_less_saving needs mhc_fused_saving_us >= 0")
        }
        (_, Some(_)) => bail!("{ARCH_TAG}: mhc_fused_saving_us is for mhc_boundary fused_less_saving"),
    };
    let (ep_comm, ep_comm_model) = match expert_exchange {
        ExpertExchange::FixedMeasured => (EpCommKind::FastAllgather, EpCommModel::Measured),
        ExpertExchange::FusedFit => (EpCommKind::FusedAllgather, EpCommModel::Analytic),
    };
    let attn_tp_collective = match kda_tp_comm_pricing {
        KdaTpCommPricing::GatherScatter => AttnTpCollective::FixedGatherScatter,
        KdaTpCommPricing::AllReduce => AttnTpCollective::FastUnbudgeted,
    };
    Ok(IterArchSel::Glm53FlashVllmNvfp4DpAttnEpMoeDflash2 {
        model: model.clone(),
        ep_size: *ep_size,
        attn_tp: *kda_tp_gpus,
        attn_all_reduce_row_gpus: None,
        attn_tp_layers: AttnTpLayers::KdaOnly,
        attn_tp_collective,
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
        hisparse_buffer_tokens: *hisparse_buffer_tokens,
        hisparse_top_k: *hisparse_top_k,
        hisparse_host_dram_gb_per_node: *hisparse_host_dram_gb_per_node,
        hisparse_gpus_per_node: *hisparse_gpus_per_node,
        hisparse_h2d_gb_per_s: *hisparse_h2d_gb_per_s,
        hisparse_exposed_io_fraction: *hisparse_exposed_io_fraction,
        hisparse_intra_step_overlap: *hisparse_intra_step_overlap,
        ep_comm,
        ep_comm_model,
        ep_link_gb_per_s: None,
        ep_comm_latency_us: None,
        ep_comm_latency_scale: None,
        ep_prepare_latency_us: None,
        ep_prepare_scale: None,
        ep_fast_backend: FastBackend::Mnnvl,
        ep_fast_combine: FastCombine::ReduceScatter,
        ep_fast_latency_us: None,
        ep_fast_fused_residual_us: None,
        ep_fanout_ranks: None,
        ep_combine_row_bytes: None,
        // Read only by `ep_comm_model: measured`.
        ep_fixed_combine: FixedCombineSide::Auto,
        mhc_fuse_fixed_us,
        mhc_one_launch,
        router_gate_once: *router_gate_once,
        moe_input_copy: *moe_input_copy,
        routed_moe_whatif_file: routed_moe_whatif_file.clone(),
        moe_shared_pair_file: moe_shared_pair_file.clone(),
        moe_shared_pair_column: moe_shared_pair_column.clone(),
        moe_shared_overlap: *moe_shared_overlap,
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

    /// Catches the tag drifting from the knob-by-knob selector the design was
    /// measured with (logs/20261112_6_kda_tp_agrs strategies G5_ko*_gbm, and
    /// N5_gbm at b = 1, with the exchange now the measured fixed-pattern one;
    /// logs/20261113_5_k1_measured kfull): the two must compile the same slots.
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
                "ep_comm": "fast_allgather",
                "ep_comm_model": "measured",
                "ep_fast_backend": "mnnvl",
                "ep_fast_combine": "reduce_scatter",
                "attn_tp": b,
                "attn_tp_layers": "kda_only",
                "attn_tp_collective": "fixed_gather_scatter",
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

    /// Catches a group size the EP group cannot split into, and a tag with
    /// no routing (it has no synthetic default).
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
        for (boundary, saving) in [("fused_less_saving", None), ("unfused", Some(1.0)), ("fused_rows", Some(1.0))] {
            let mut bad = tag(8);
            bad["mhc_boundary"] = boundary.into();
            bad["mhc_fused_saving_us"] = saving.into();
            let sel: IterArchSel = serde_json::from_value(bad).unwrap();
            assert!(target_selector(&sel).is_err(), "{boundary} {saving:?}");
        }
    }

    /// Catches the mHC and KDA-communication choices reaching the wrong
    /// knob of the target selector.
    #[test]
    fn the_tag_maps_its_pricing_choices() {
        let target = |extra: serde_json::Value| {
            let mut sel = tag(8);
            sel.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            let sel: IterArchSel = serde_json::from_value(sel).unwrap();
            serde_json::to_value(format!("{:?}", target_selector(&sel).unwrap())).unwrap().to_string()
        };
        let default = target(serde_json::json!({}));
        assert!(default.contains("mhc_fuse_fixed_us: None") && default.contains("mhc_one_launch: false"));
        assert!(default.contains("attn_tp_collective: FixedGatherScatter"));
        let saving = target(serde_json::json!({"mhc_boundary": "fused_less_saving", "mhc_fused_saving_us": 1.796}));
        assert!(saving.contains("mhc_fuse_fixed_us: Some(1.796)") && saving.contains("mhc_one_launch: false"));
        let rows = target(serde_json::json!({"mhc_boundary": "fused_rows"}));
        assert!(rows.contains("mhc_fuse_fixed_us: None") && rows.contains("mhc_one_launch: true"));
        let ar = target(serde_json::json!({"kda_tp_comm_pricing": "all_reduce"}));
        assert!(ar.contains("attn_tp_collective: FastUnbudgeted"));
        assert!(default.contains("ep_comm: FastAllgather") && default.contains("ep_comm_model: Measured"));
        let fit = target(serde_json::json!({"expert_exchange": "fused_fit"}));
        assert!(fit.contains("ep_comm: FusedAllgather") && fit.contains("ep_comm_model: Analytic"));
    }
}
