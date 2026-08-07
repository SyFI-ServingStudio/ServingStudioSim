//! FlashInfer MNNVL MoE all-to-all metadata-preparation timing leaf.
//!
//! One `mnnvl_moe_alltoallv_prepare_without_allgather` call decides, from the
//! router's expert ids alone, which rows go where: it counts them
//! (`computeCountAndIndiceDevice`), prefix-sums the counts (`computeCumsumDevice`),
//! compacts the indices (`moveIndiceDevice`), clears the expert-id scratch
//! (`memsetExpertIdsDevice`) and exchanges the result (`allToAllMetadataDevice`).
//! Five launches, one measurable call, and the whole reason this is a profiled
//! kind rather than a modelled one: none of it is bandwidth-bound. At 8k tokens
//! it moves a quarter of a megabyte and still costs ~0.33 ms per layer, being
//! dominated by atomics over `ep_size x slot_count` bins and by the inter-rank
//! metadata exchange. A byte-rate elementwise curve would predict roughly zero.
//!
//! The payload width is absent from the key for the same reason — nothing in
//! this call touches the activations. What it does depend on is the routing
//! table's shape (`top_k`, `slot_count`) and the number of destination bins
//! (`ep_size`), all of which are static recipe identity, leaving
//! `tokens_per_rank` as the single runtime axis.

use crate::common::Fabric;
use crate::timing::bridge::{de_backends, ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoeAlltoallPrepareKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Ranks in the expert-parallel group — also the number of destination bins
    /// the count kernel atomically increments into.
    pub ep_size: u32,
    pub top_k: u32,
    pub slot_count: u32,
    /// Row/cache key only: MNNVL is NVLink by construction.
    pub fabric: Fabric,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MoeAlltoallPrepareKernelInput {
    /// Same maximum-across-ranks token count the transfer itself is sized by.
    pub tokens_per_rank: u32,
}

pub struct MoeAlltoallPrepareSpec;

impl KernelSpec for MoeAlltoallPrepareSpec {
    type Config = MoeAlltoallPrepareKernelConfig;
    type Input = MoeAlltoallPrepareKernelInput;

    const KIND: KernelKind = "moe_alltoall_prepare";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // The full token ladder, unbounded by memory: the only token-sized
        // allocations here are the `tokens x top_k` index and scale tables,
        // which are megabytes at the top of the axis.
        SweepGrid::new(vec![Axis::chain([
            Axis::values([1, 4, 8, 16]),
            Axis::token_axis(),
        ])])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        assert!(
            config.top_k <= config.slot_count,
            "top_k {} cannot exceed slot_count {}",
            config.top_k,
            config.slot_count
        );
        grid.expand_1d(|tokens_per_rank| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("ep_size", config.ep_size)
                .with("tokens_per_rank", tokens_per_rank as u32)
                .with("top_k", config.top_k)
                .with("slot_count", config.slot_count)
                .with("fabric", config.fabric.as_str())
        })
    }
}

register_kernel!(MoeAlltoallPrepareKernel, MoeAlltoallPrepareSpec);

#[cfg(test)]
mod tests {
    use super::{
        MoeAlltoallPrepareKernelConfig, MoeAlltoallPrepareKernelInput, MoeAlltoallPrepareSpec,
    };
    use crate::common::Fabric;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords, SweepGrid};
    use serde_json::Value;

    fn config() -> MoeAlltoallPrepareKernelConfig {
        MoeAlltoallPrepareKernelConfig {
            backends: vec!["flashinfer_mnnvl"],
            gpu_name: "NVIDIA H200".to_string(),
            ep_size: 8,
            top_k: 8,
            slot_count: 256,
            fabric: Fabric::Nvlink,
        }
    }

    #[test]
    fn config_identity_and_description_include_all_static_axes() {
        let base = config();
        assert_eq!(base, base.clone());

        let mut changed_ep_size = config();
        changed_ep_size.ep_size = 4;
        assert_ne!(base, changed_ep_size);

        let mut changed_slot_count = config();
        changed_slot_count.slot_count = 512;
        assert_ne!(base, changed_slot_count);

        assert_eq!(
            base.describe_config(),
            serde_json::json!({
                "backends": ["flashinfer_mnnvl"],
                "gpu_name": "NVIDIA H200",
                "ep_size": 8,
                "top_k": 8,
                "slot_count": 256,
                "fabric": "nvlink",
            })
        );
    }

    #[test]
    fn input_projects_to_token_axis_and_converts_to_slot_input() {
        let input = MoeAlltoallPrepareKernelInput {
            tokens_per_rank: 8_190,
        };
        assert_eq!(&*input.coords(), &[8_190.0]);
        assert_eq!(
            MoeAlltoallPrepareKernelInput::coord_field_names(),
            &["tokens_per_rank"]
        );

        let slot_input: SlotInput = input.into();
        assert!(matches!(slot_input, SlotInput::MoeAlltoallPrepare(_)));
    }

    #[test]
    fn grid_is_the_full_token_ladder_from_decode_to_a_dp8_prefill_step() {
        let grid = MoeAlltoallPrepareSpec::sweep_grid(&config());
        assert_eq!(grid.axes().len(), 1);
        assert_eq!(&grid.axes()[0][..5], &[1.0, 4.0, 8.0, 16.0, 32.0]);
        assert_eq!(grid.axes()[0].last().copied(), Some(65_536.0));
        assert_eq!(
            MoeAlltoallPrepareSpec::cache_kind("flashinfer_mnnvl"),
            CacheKind::Cache1DLinear
        );
    }

    #[test]
    fn enumerate_emits_exactly_the_python_wire_schema() {
        let config = config();
        let grid = SweepGrid::new(vec![vec![8_190.0]]);
        let payloads = MoeAlltoallPrepareSpec::enumerate(&config, &grid, "flashinfer_mnnvl");
        let fields = payloads[0].fields();

        // Must stay aligned with `MoeAlltoallPrepareArgs` in
        // profiling/kernels/moe_alltoall_prepare.py — notably no payload width.
        assert_eq!(fields.len(), 6);
        assert_eq!(
            fields.get("backend"),
            Some(&Value::from("flashinfer_mnnvl"))
        );
        assert_eq!(fields.get("ep_size"), Some(&Value::from(8_u32)));
        assert_eq!(fields.get("tokens_per_rank"), Some(&Value::from(8_190_u32)));
        assert_eq!(fields.get("top_k"), Some(&Value::from(8_u32)));
        assert_eq!(fields.get("slot_count"), Some(&Value::from(256_u32)));
        assert_eq!(fields.get("fabric"), Some(&Value::from("nvlink")));
        assert!(fields.get("hidden_bytes").is_none());
    }
}
