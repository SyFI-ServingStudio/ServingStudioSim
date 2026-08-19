//! vLLM Triton `fused_moe_kernel` timing leaf -- gather + blocked GEMM in one
//! launch.
//!
//! Distinct from `fp8_blockscale_grouped_gemm` on purpose. That leaf models the
//! permute-then-group realization; this one models the realization vLLM
//! actually runs on a non-EP local path, where work is quantized into
//! `BLOCK_SIZE_M`-row blocks *per expert* and no permutation buffer exists.
//! `local_ppm` therefore matters even more here than there: it decides how many
//! padded blocks the grid covers, and a uniform assumption systematically
//! under-counts them.
//!
//! `launch_role` is config identity, not a runtime axis, because the two roles
//! have different `n`/`k` anyway and are built as separate ops. It carries the
//! `top_k` / `mul_routed_weight` pair that vLLM never varies independently; see
//! `profiling/kernels/vllm_fused_moe.py` for the ABI it maps to.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::routing::RoutingDistribution;
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

/// Gate/up launch: one activation row per token, routed weights not applied.
pub const LAUNCH_ROLE_GATE_UP: &str = "w13";
/// Down launch: one activation row per (token, selected expert), routed weights
/// applied.
pub const LAUNCH_ROLE_DOWN: &str = "w2";

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct VllmFusedMoeKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub n: Dim,
    pub k: Dim,
    #[compute_dtype]
    pub dtype: DType,
    pub experts_per_token: u32,
    /// `"w13"` or `"w2"`; see the module header.
    pub launch_role: String,
    /// FP8 activation/weight scale granularity, not the Triton `BLOCK_SIZE_M`
    /// (which vLLM's own tuned-config lookup owns).
    pub block_size: u32,
    /// Raw ppm values for this rank's contiguous local-expert shard, same
    /// contract as the grouped-GEMM leaf.
    pub local_ppm: Vec<u32>,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct VllmFusedMoeKernelInput {
    pub num_tokens: u32,
}

pub struct VllmFusedMoeSpec;

impl KernelSpec for VllmFusedMoeSpec {
    type Config = VllmFusedMoeKernelConfig;
    type Input = VllmFusedMoeKernelInput;

    const KIND: KernelKind = "vllm_fused_moe";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // Same low-count prefix as the grouped-GEMM leaf: a routed decode batch
        // sits far below the shared token axis's first point, and that regime is
        // exactly where block padding dominates.
        SweepGrid::new(vec![Axis::chain([
            Axis::values([1, 4, 8, 16, 32, 48]),
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
        let num_local_experts = config.local_ppm.len() as u32;
        grid.expand_1d(|num_tokens| {
            let num_tokens = num_tokens as u32;
            let global_expert_selections = num_tokens
                .checked_mul(config.experts_per_token)
                .expect("num_tokens * experts_per_token must fit u32");
            let per_group_batches = RoutingDistribution::to_per_expert_counts(
                global_expert_selections,
                &config.local_ppm,
            );

            ArgsPayload::new()
                .with("backend", backend)
                .with("n", config.n.get())
                .with("k", config.k.get())
                .with("dtype", config.dtype.as_str())
                .with("num_local_experts", num_local_experts)
                .with("num_tokens", num_tokens)
                .with("experts_per_token", config.experts_per_token)
                .with("launch_role", config.launch_role.as_str())
                .with("block_size", config.block_size)
                .with("per_group_batches", per_group_batches)
        })
    }
}

register_kernel!(VllmFusedMoeKernel, VllmFusedMoeSpec);

#[cfg(test)]
mod tests {
    use super::{
        VllmFusedMoeKernelConfig, VllmFusedMoeKernelInput, VllmFusedMoeSpec, LAUNCH_ROLE_DOWN,
        LAUNCH_ROLE_GATE_UP,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn config() -> VllmFusedMoeKernelConfig {
        VllmFusedMoeKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA H200".to_string(),
            n: 1024.into(),
            k: 2048.into(),
            dtype: DType::Fp8E4m3,
            experts_per_token: 2,
            launch_role: LAUNCH_ROLE_GATE_UP.to_string(),
            block_size: 128,
            local_ppm: vec![300_000, 200_000],
        }
    }

    #[test]
    fn launch_role_is_part_of_the_config_identity() {
        let base = config();
        let mut down = config();
        down.launch_role = LAUNCH_ROLE_DOWN.to_string();
        assert_ne!(base, down);

        let mut changed_distribution = config();
        changed_distribution.local_ppm = vec![250_000, 250_000];
        assert_ne!(base, changed_distribution);
    }

    #[test]
    fn describe_config_preserves_the_static_identity() {
        assert_eq!(
            config().describe_config(),
            serde_json::json!({
                "backends": ["vllm_triton"],
                "gpu_name": "NVIDIA H200",
                "n": {"value": 1024, "expression": null, "bindings": {}},
                "k": {"value": 2048, "expression": null, "bindings": {}},
                "dtype": "fp8_e4m3",
                "experts_per_token": 2,
                "launch_role": "w13",
                "block_size": 128,
                "local_ppm": [300000, 200000],
            })
        );
    }

    #[test]
    fn input_projects_to_the_token_axis() {
        let input = VllmFusedMoeKernelInput { num_tokens: 48 };
        assert_eq!(&*input.coords(), &[48.0]);
        assert_eq!(
            VllmFusedMoeKernelInput::coord_field_names(),
            &["num_tokens"]
        );
        assert_eq!(
            VllmFusedMoeSpec::cache_kind("vllm_triton"),
            CacheKind::Cache1DLinear
        );
    }

    #[test]
    fn enumerate_matches_python_args_exactly() {
        let config = config();
        let grid = VllmFusedMoeSpec::sweep_grid(&config);
        let payload = &VllmFusedMoeSpec::enumerate(&config, &grid, "vllm_triton")[0];
        let fields = payload.fields();

        assert_eq!(fields.len(), 10);
        assert_eq!(fields.get("backend"), Some(&Value::from("vllm_triton")));
        assert_eq!(fields.get("n"), Some(&Value::from(1024_u32)));
        assert_eq!(fields.get("k"), Some(&Value::from(2048_u32)));
        assert_eq!(fields.get("dtype"), Some(&Value::from("fp8_e4m3")));
        assert_eq!(fields.get("num_local_experts"), Some(&Value::from(2_u32)));
        assert_eq!(fields.get("num_tokens"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("experts_per_token"), Some(&Value::from(2_u32)));
        assert_eq!(fields.get("launch_role"), Some(&Value::from("w13")));
        assert_eq!(fields.get("block_size"), Some(&Value::from(128_u32)));
        assert_eq!(
            fields.get("per_group_batches"),
            Some(&serde_json::json!([1, 0]))
        );
    }
}
