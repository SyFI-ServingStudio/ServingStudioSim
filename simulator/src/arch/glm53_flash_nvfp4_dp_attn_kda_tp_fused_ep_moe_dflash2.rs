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
//! * **Expert exchange.** `ep_comm: fused_allgather` on MNNVL: each rank writes
//!   its NVFP4 rows (activations, block scales, top-k ids and weights) into
//!   every peer's memory in one fixed-pattern store, the experts run, and the
//!   bf16 partial sums go back by a reduce-scatter that each owner sums. The
//!   dispatch store is assumed fused into the activation-quant kernel, and
//!   the combine read and its sum into the next op.
//! * **mHC boundaries.** vLLM's kernels, unfused.
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
//! | DSA, dense FFN, shared expert, router, mHC pre/post (vLLM kernels) | 实测 |
//! | KDA output reduce-scatter | 实测 at 8 GPUs (`moe_ep_reduce_scatter` `mnnvl_pull` rows); `b < 8` reads them at the per-rank counts that receive the same bytes, keeping the 8-peer fixed cost (over-prices) |
//! | KDA input all-gather | stand-in: the reduce-scatter row at the same counts (same bytes received plus an fp32 sum; not measured on its own) |
//! | routed experts (`nvfp4_fused_moe`, routing precomputed) | 实测 |
//! | dispatch activation quant, router top-k select | 实测 (elementwise rows) |
//! | exchange dispatch and combine: per-row and link cost | 拟合 (`arch::ep_comm` B200 EP8 `moe_alltoall` fit: c 3.86 GB/s, p 2.63 ns, L 605 GB/s, w 0.9) |
//! | exchange dispatch and combine: fixed cost per leg | 外推: 0 us (the fusion; the fit's unfused `a` is 4.11 us per leg). Optimistic, not conservative |
//! | combine owner sum of the partials | 外推: 0 us (assumed inside the next op; the unfused exchanges bill it as an elementwise row) |
//! | DFlash2 draft (TP1 per rank) | 实测 |
//! | HiSparse host -> device miss copy (HiSparse presets only) | 外推 (`arch::hisparse` analytic model) |
//!
//! # Hooks for work in flight (not merged here)
//!
//! * **Measured fixed-pattern exchange** (branch `flash-exchange-kernel`):
//!   its `ep_comm_model: measured` prices `fused_allgather` from
//!   `moe_ep_quantize_and_all_gather` (`mnnvl_lamport`, quant folded in) and
//!   `moe_ep_reduce_scatter` (`mnnvl_pull`) rows at EP8. Once merged, set
//!   `ep_comm_model` in [`target_selector`] to `Measured` and move the two
//!   exchange rows above to 实测; no new param.
//! * **Measured KDA all-gather.** The all-gather leg reads the reduce-scatter
//!   row as a stand-in. A bf16 fixed-pattern all-gather row (and rows at 2
//!   and 4 GPUs) would replace it inside
//!   `glm53_flash_vllm_fp8_dp_attn_ep_moe::KdaGatherScatter`; no new param.
//! * **Fused mHC boundary / router gate once** (worktree `wt-flash-mhc-fuse`):
//!   it replaces the target builder's `mhc_fuse_fixed_us` with
//!   `Glm53FlashFusions { mhc_boundary, router_gate_once }`, read from new
//!   `mhc_one_launch` / `router_gate_once` selector keys. Once merged, add
//!   those two keys (same names) to this tag and copy them through
//!   [`target_selector`]; never the fixed-us what-if.

use anyhow::{bail, ensure, Result};

use crate::arch::config::IterArchSel;
use crate::arch::ep_comm::{EpCommKind, EpCommModel, FastBackend, FastCombine};
use crate::arch::glm53_flash_vllm_fp8_dp_attn_ep_moe::{AttnTpCollective, AttnTpLayers};

/// The selector tag.
pub const ARCH_TAG: &str = "glm53_flash_nvfp4_dp_attn_kda_tp_fused_ep_moe_dflash2";

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
    Ok(IterArchSel::Glm53FlashVllmNvfp4DpAttnEpMoeDflash2 {
        model: model.clone(),
        ep_size: *ep_size,
        attn_tp: *kda_tp_gpus,
        attn_all_reduce_row_gpus: None,
        attn_tp_layers: AttnTpLayers::KdaOnly,
        attn_tp_collective: AttnTpCollective::FixedGatherScatter,
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
        hisparse_buffer_tokens: *hisparse_buffer_tokens,
        hisparse_top_k: *hisparse_top_k,
        hisparse_host_dram_gb_per_node: *hisparse_host_dram_gb_per_node,
        hisparse_gpus_per_node: *hisparse_gpus_per_node,
        hisparse_h2d_gb_per_s: *hisparse_h2d_gb_per_s,
        hisparse_exposed_io_fraction: *hisparse_exposed_io_fraction,
        hisparse_intra_step_overlap: *hisparse_intra_step_overlap,
        ep_comm: EpCommKind::FusedAllgather,
        // Hook: `Measured` once the fixed-pattern exchange rows merge.
        ep_comm_model: EpCommModel::Analytic,
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
        mhc_fuse_fixed_us: None,
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
    /// N5_gbm at b = 1): the two must compile the same slots.
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
                "ep_comm": "fused_allgather",
                "ep_comm_model": "analytic",
                "ep_fast_backend": "mnnvl",
                "ep_fast_combine": "reduce_scatter",
                "attn_tp": b,
                "attn_tp_layers": "kda_only",
                "attn_tp_collective": "fixed_gather_scatter",
                "kda_gate_b_merged": true,
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
    }
}
