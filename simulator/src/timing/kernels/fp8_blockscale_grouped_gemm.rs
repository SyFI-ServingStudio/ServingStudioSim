//! Direct TensorRT-LLM FP8 block-scale `GroupedWithOffset` GEMM timing leaf.
//!
//! This production path keeps the global input-token count and top-k as recipe
//! axes: TensorRT-LLM sizes routed capacity from
//! `num_input_tokens * experts_per_token`, while the EP-local `local_ppm` shard
//! resolves that global selection count into the physical
//! `per_group_batches`. Consequently `num_input_tokens` is the one runtime
//! cache axis; `experts_per_token` and `local_ppm` remain config identity.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::routing::RoutingDistribution;
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Fp8BlockscaleGroupedGemmKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub n: Dim,
    pub k: Dim,
    #[compute_dtype]
    pub dtype: DType,
    pub experts_per_token: u32,
    /// Raw ppm values for this rank's contiguous local-expert shard. The sum
    /// can be below one million because routing apportions from the global
    /// selection count into only the experts resident on this rank.
    pub local_ppm: Vec<u32>,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct Fp8BlockscaleGroupedGemmKernelInput {
    pub num_input_tokens: u32,
}

pub struct Fp8BlockscaleGroupedGemmSpec;

impl KernelSpec for Fp8BlockscaleGroupedGemmSpec {
    type Config = Fp8BlockscaleGroupedGemmKernelConfig;
    type Input = Fp8BlockscaleGroupedGemmKernelInput;

    const KIND: KernelKind = "fp8_blockscale_grouped_gemm";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
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
        grid.expand_1d(|num_input_tokens| {
            let num_input_tokens = num_input_tokens as u32;
            let global_expert_selections = num_input_tokens
                .checked_mul(config.experts_per_token)
                .expect("num_input_tokens * experts_per_token must fit u32");
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
                .with("num_input_tokens", num_input_tokens)
                .with("experts_per_token", config.experts_per_token)
                .with("per_group_batches", per_group_batches)
        })
    }
}

register_kernel!(Fp8BlockscaleGroupedGemmKernel, Fp8BlockscaleGroupedGemmSpec);

#[cfg(test)]
mod tests {
    use super::{
        Fp8BlockscaleGroupedGemmKernelConfig, Fp8BlockscaleGroupedGemmKernelInput,
        Fp8BlockscaleGroupedGemmSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn config() -> Fp8BlockscaleGroupedGemmKernelConfig {
        Fp8BlockscaleGroupedGemmKernelConfig {
            backends: vec!["flashinfer_trtllm"],
            gpu_name: "NVIDIA H200".to_string(),
            n: 4096.into(),
            k: 8192.into(),
            dtype: DType::Fp8E4m3,
            experts_per_token: 2,
            local_ppm: vec![300_000, 200_000],
        }
    }

    #[test]
    fn config_identity_includes_recipe_and_distribution_axes() {
        let base = config();
        assert_eq!(base, base.clone());

        let mut changed_recipe = config();
        changed_recipe.experts_per_token = 4;
        assert_ne!(base, changed_recipe);

        let mut changed_distribution = config();
        changed_distribution.local_ppm = vec![250_000, 250_000];
        assert_ne!(base, changed_distribution);

        let mut changed_shape = config();
        changed_shape.n = 8192.into();
        assert_ne!(base, changed_shape);
    }

    #[test]
    fn describe_config_preserves_the_static_identity() {
        assert_eq!(
            config().describe_config(),
            serde_json::json!({
                "backends": ["flashinfer_trtllm"],
                "gpu_name": "NVIDIA H200",
                "n": {"value": 4096, "expression": null, "bindings": {}},
                "k": {"value": 8192, "expression": null, "bindings": {}},
                "dtype": "fp8_e4m3",
                "experts_per_token": 2,
                "local_ppm": [300000, 200000],
            })
        );
    }

    #[test]
    fn input_projects_to_num_input_tokens_axis() {
        let input = Fp8BlockscaleGroupedGemmKernelInput {
            num_input_tokens: 48,
        };
        assert_eq!(&*input.coords(), &[48.0]);
        assert_eq!(
            Fp8BlockscaleGroupedGemmKernelInput::coord_field_names(),
            &["num_input_tokens"]
        );
    }

    #[test]
    fn grid_covers_small_batches_and_shared_token_curve() {
        let grid = Fp8BlockscaleGroupedGemmSpec::sweep_grid(&config());
        assert_eq!(grid.axes().len(), 1);
        assert_eq!(
            &grid.axes()[0][..7],
            &[1.0, 4.0, 8.0, 16.0, 32.0, 48.0, 64.0]
        );
        assert!(grid.axes()[0].contains(&2048.0));
        assert_eq!(grid.axes()[0].last().copied(), Some(65536.0));
        assert_eq!(
            Fp8BlockscaleGroupedGemmSpec::cache_kind("flashinfer_trtllm"),
            CacheKind::Cache1DLinear
        );
    }

    #[test]
    fn dtype_is_the_compute_dtype_tag() {
        assert_eq!(config().compute_dtype(), Some(DType::Fp8E4m3));
    }

    #[test]
    fn enumerate_matches_python_args_exactly() {
        let config = config();
        let grid = Fp8BlockscaleGroupedGemmSpec::sweep_grid(&config);
        let payload =
            &Fp8BlockscaleGroupedGemmSpec::enumerate(&config, &grid, "flashinfer_trtllm")[0];
        let fields = payload.fields();

        assert_eq!(fields.len(), 8);
        assert_eq!(
            fields.get("backend"),
            Some(&Value::from("flashinfer_trtllm"))
        );
        assert_eq!(fields.get("n"), Some(&Value::from(4096_u32)));
        assert_eq!(fields.get("k"), Some(&Value::from(8192_u32)));
        assert_eq!(fields.get("dtype"), Some(&Value::from("fp8_e4m3")));
        assert_eq!(fields.get("num_local_experts"), Some(&Value::from(2_u32)));
        assert_eq!(fields.get("num_input_tokens"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("experts_per_token"), Some(&Value::from(2_u32)));
        assert_eq!(
            fields.get("per_group_batches"),
            Some(&serde_json::json!([1, 0]))
        );
        assert_eq!(payload.backend(), Some("flashinfer_trtllm"));
    }
}
