//! FlashInfer/TensorRT-LLM `MoE` finalize-routing timing leaf.
//!
//! The physical kernel receives the global token count plus the number of
//! routed rows resident on this EP rank.  Keep only `token_count` as a runtime
//! cache axis: `top_k` and the rank-local expert popularity distribution are
//! static recipe identity, and deterministically derive the local row count for
//! every profiled grid point.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::routing::RoutingDistribution;
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MoeFinalizeRoutingKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden_size: Dim,
    pub top_k: u32,
    pub num_experts_per_rank: u32,
    /// Raw ppm values for this rank's local experts. The values need not sum to
    /// one million because this is one contiguous EP shard of the global
    /// popularity distribution.
    pub local_ppm: Vec<u32>,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MoeFinalizeRoutingKernelInput {
    pub token_count: u32,
}

/// Rows of this EP rank's expanded expert output for a given global token
/// count: the rank's popularity shard apportions the `token_count x top_k`
/// global selections. Shared by the sweep bound and the wire payload so the
/// grid can never claim a shape the payload then contradicts.
fn local_routed_token_count(token_count: u32, config: &MoeFinalizeRoutingKernelConfig) -> u32 {
    let global_expert_selections = token_count
        .checked_mul(config.top_k)
        .expect("token_count * top_k must fit u32");
    RoutingDistribution::to_per_expert_counts(global_expert_selections, &config.local_ppm)
        .into_iter()
        .sum()
}

pub struct MoeFinalizeRoutingSpec;

impl KernelSpec for MoeFinalizeRoutingSpec {
    type Config = MoeFinalizeRoutingKernelConfig;
    type Input = MoeFinalizeRoutingKernelInput;

    const KIND: KernelKind = "moe_finalize_routing";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        // Decode needs points below the shared token curve. The upper end is
        // bounded by the runner, which allocates the expanded routed rows plus
        // the unpermuted output — so the bound belongs in bytes, not in a token
        // count. It was a flat 8192 tokens while every caller was a
        // single-replica prefill; a DP8 replica schedules eight 8k prefills in
        // ONE step, which asks this leaf for 65,520 tokens and would otherwise
        // extrapolate the curve 8x past its last profiled point.
        const EXPANDED_ROW_BUDGET_BYTES: f64 = 2.0 * 1024.0 * 1024.0 * 1024.0;
        let row_bytes = f64::from(config.hidden_size.get())
            * f64::from(config.dtype.size_bytes())
            // The kernel reads the expanded rows and writes the reduced ones.
            * 2.0;
        let affordable = Axis::token_axis()
            .into_iter()
            .filter(|&token_count| {
                let local_rows = local_routed_token_count(token_count as u32, config);
                f64::from(local_rows) * row_bytes <= EXPANDED_ROW_BUDGET_BYTES
            })
            .collect::<Vec<_>>();
        SweepGrid::new(vec![Axis::chain([Axis::values([1, 4, 8, 16]), affordable])])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        assert_eq!(
            config.local_ppm.len(),
            config.num_experts_per_rank as usize,
            "local_ppm must contain one entry per local expert"
        );
        grid.expand_1d(|token_count| {
            let token_count = token_count as u32;
            let local_routed_token_count = local_routed_token_count(token_count, config);

            ArgsPayload::new()
                .with("backend", backend)
                .with("token_count", token_count)
                .with("hidden_size", config.hidden_size.get())
                .with("top_k", config.top_k)
                .with("num_experts_per_rank", config.num_experts_per_rank)
                .with("local_routed_token_count", local_routed_token_count)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(MoeFinalizeRoutingKernel, MoeFinalizeRoutingSpec);

#[cfg(test)]
mod tests {
    use super::{
        MoeFinalizeRoutingKernelConfig, MoeFinalizeRoutingKernelInput, MoeFinalizeRoutingSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::sweep::Axis;
    use crate::timing::{SlotInput, SweepCoords, SweepGrid};
    use serde_json::Value;

    fn uniform_quarter_ppm() -> Vec<u32> {
        let mut local_ppm = vec![7_812; 32];
        for expert_ppm in local_ppm.iter_mut().take(16) {
            *expert_ppm += 1;
        }
        local_ppm
    }

    fn config() -> MoeFinalizeRoutingKernelConfig {
        MoeFinalizeRoutingKernelConfig {
            backends: vec!["flashinfer_trtllm"],
            gpu_name: "NVIDIA H200".to_string(),
            hidden_size: 4096.into(),
            top_k: 8,
            num_experts_per_rank: 32,
            local_ppm: uniform_quarter_ppm(),
            dtype: DType::Bf16,
        }
    }

    #[test]
    fn config_identity_and_description_include_all_static_axes() {
        let base = config();
        assert_eq!(base, base.clone());

        let mut changed_backend = config();
        changed_backend.backends = vec!["another_backend"];
        assert_ne!(base, changed_backend);

        let mut changed_gpu = config();
        changed_gpu.gpu_name = "NVIDIA H100".to_string();
        assert_ne!(base, changed_gpu);

        let mut changed_hidden_size = config();
        changed_hidden_size.hidden_size = 8192.into();
        assert_ne!(base, changed_hidden_size);

        let mut changed_popularity = config();
        changed_popularity.local_ppm[0] += 1;
        assert_ne!(base, changed_popularity);

        let mut changed_top_k = config();
        changed_top_k.top_k = 4;
        assert_ne!(base, changed_top_k);

        let mut changed_expert_count = config();
        changed_expert_count.num_experts_per_rank = 16;
        changed_expert_count.local_ppm.truncate(16);
        assert_ne!(base, changed_expert_count);

        let mut changed_dtype = config();
        changed_dtype.dtype = DType::Fp16;
        assert_ne!(base, changed_dtype);

        assert_eq!(
            base.describe_config(),
            serde_json::json!({
                "backends": ["flashinfer_trtllm"],
                "gpu_name": "NVIDIA H200",
                "hidden_size": {"value": 4096, "expression": null, "bindings": {}},
                "top_k": 8,
                "num_experts_per_rank": 32,
                "local_ppm": uniform_quarter_ppm(),
                "dtype": "bf16",
            })
        );
    }

    #[test]
    fn input_projects_to_token_count_axis_and_converts_to_slot_input() {
        let input = MoeFinalizeRoutingKernelInput { token_count: 64 };
        assert_eq!(&*input.coords(), &[64.0]);
        assert_eq!(
            MoeFinalizeRoutingKernelInput::coord_field_names(),
            &["token_count"]
        );

        let slot_input: SlotInput = input.into();
        assert!(matches!(slot_input, SlotInput::MoeFinalizeRouting(_)));
    }

    #[test]
    fn grid_covers_decode_and_prefill_up_to_the_runner_allocation_budget() {
        // This config expands two rows per token (top-8 into a quarter shard)
        // of 4096 bf16 elements, so 65,536 tokens land exactly on the 2 GiB
        // budget and stay in.
        let grid = MoeFinalizeRoutingSpec::sweep_grid(&config());
        assert_eq!(grid.axes().len(), 1);
        assert_eq!(&grid.axes()[0][..5], &[1.0, 4.0, 8.0, 16.0, 32.0]);
        assert!(grid.axes()[0].contains(&64.0));
        assert!(grid.axes()[0].contains(&4096.0));
        assert_eq!(grid.axes()[0].last().copied(), Some(65536.0));
        assert_eq!(
            MoeFinalizeRoutingSpec::cache_kind("flashinfer_trtllm"),
            CacheKind::Cache1DLinear
        );
    }

    #[test]
    fn raising_the_bound_left_every_previously_profiled_point_in_place() {
        // The bound used to be a flat 8192 tokens. Widening it must only ADD
        // points past that mark: every existing caller (the Qwen vLLM arch, the
        // recorded goldens) looks up below it, and a changed point there would
        // move an interpolated cost that nothing asked to change.
        let grid = MoeFinalizeRoutingSpec::sweep_grid(&config());
        let previous = Axis::chain([
            Axis::values([1, 4, 8, 16]),
            Axis::token_axis()
                .into_iter()
                .filter(|&token_count| token_count <= 8192.0)
                .collect(),
        ]);
        assert_eq!(&grid.axes()[0][..previous.len()], &previous[..]);
        assert!(grid.axes()[0].len() > previous.len());
    }

    #[test]
    fn every_deployed_shape_keeps_its_whole_pre_widening_axis() {
        // The symmetric-regression point for widening the bound: only two archs
        // build this config — `qwen3_vllm_moe_dp_attn_ep_ffn` (Qwen3-235B, hidden
        // 4096, 128 experts, top-8) and `glm52_vllm_dsa_moe` (hidden 6144, 256
        // experts, top-8). Qwen is the one that must not move: it was already
        // recorded against the flat 8192 bound. A byte budget CAN cut a grid
        // shorter than a token bound would — that is the failure this pins shut,
        // across both models and every EP split they are run at.
        let previous_bound = Axis::token_axis()
            .into_iter()
            .filter(|&token_count| token_count <= 8192.0)
            .collect::<Vec<_>>();
        for (label, hidden_size, expert_count) in
            [("qwen3_235b", 4096, 128_u32), ("glm52", 6144, 256)]
        {
            for ep_size in [1_u32, 2, 4, 8, 16, 32] {
                let experts_per_rank = expert_count / ep_size;
                let mut deployed = config();
                deployed.hidden_size = hidden_size.into();
                deployed.num_experts_per_rank = experts_per_rank;
                // Uniform popularity over this rank's shard; the exact split does
                // not matter here, the row count it implies does.
                deployed.local_ppm = vec![1_000_000 / expert_count; experts_per_rank as usize];

                let grid = MoeFinalizeRoutingSpec::sweep_grid(&deployed);
                let tail = &grid.axes()[0][4..]; // past the fixed [1, 4, 8, 16] head
                assert!(
                    tail.len() >= previous_bound.len(),
                    "{label} ep{ep_size}: byte budget cut the axis shorter than the \
                     8192-token bound it replaced ({} < {} points)",
                    tail.len(),
                    previous_bound.len()
                );
                assert_eq!(
                    &tail[..previous_bound.len()],
                    &previous_bound[..],
                    "{label} ep{ep_size}: a point at or below 8192 tokens moved"
                );
            }
        }
    }

    #[test]
    fn a_wider_hidden_stops_the_grid_where_the_runner_would_run_out_of_memory() {
        // Four times the row bytes of `config()`, so the affordable token count
        // drops by the same factor rather than the axis being a fixed constant.
        let mut wide = config();
        wide.hidden_size = 16_384.into();
        let grid = MoeFinalizeRoutingSpec::sweep_grid(&wide);
        assert_eq!(grid.axes()[0].last().copied(), Some(16384.0));
        assert!(!grid.axes()[0].contains(&32768.0));
    }

    #[test]
    fn dtype_is_tagged_as_compute_dtype() {
        assert_eq!(config().compute_dtype(), Some(DType::Bf16));
    }

    #[test]
    fn enumerate_derives_local_count_and_matches_python_args_exactly() {
        let config = config();
        let grid = SweepGrid::new(vec![vec![64.0, 8192.0]]);
        let payloads = MoeFinalizeRoutingSpec::enumerate(&config, &grid, "flashinfer_trtllm");
        let fields = payloads[0].fields();

        assert_eq!(fields.len(), 7);
        assert_eq!(
            fields.get("backend"),
            Some(&Value::from("flashinfer_trtllm"))
        );
        assert_eq!(fields.get("token_count"), Some(&Value::from(64_u32)));
        assert_eq!(fields.get("hidden_size"), Some(&Value::from(4096_u32)));
        assert_eq!(fields.get("top_k"), Some(&Value::from(8_u32)));
        assert_eq!(
            fields.get("num_experts_per_rank"),
            Some(&Value::from(32_u32))
        );
        assert_eq!(
            fields.get("local_routed_token_count"),
            Some(&Value::from(128_u32))
        );
        assert_eq!(fields.get("dtype"), Some(&Value::from("bf16")));
        assert_eq!(payloads[0].backend(), Some("flashinfer_trtllm"));

        assert_eq!(
            payloads[1].fields().get("local_routed_token_count"),
            Some(&Value::from(16_384_u32))
        );
    }
}
