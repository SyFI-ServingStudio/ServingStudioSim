//! Grouped GEMM kernel: a per-GPU MoE expert-compute cost model, one cached
//! curve per `(n, k, dtype, local_ppm)` config. Distribution-sensitive op
//! (L1 design §2.8): the routing distribution shard `local_ppm` is baked into
//! the Config identity, while the runtime sweep is the scalar
//! `global_expert_selections` (= num_tokens × top_k, the global token-expert
//! assignment count). The cache stays 1D in `global_expert_selections`.
//!
//! `enumerate` is the distribution-sensitive twist: each swept
//! `global_expert_selections` is split across this GPU's local experts via
//! `RoutingDistribution::to_per_expert_counts(.., &config.local_ppm)`, and the
//! resulting per-expert `Vec<u32>` is the `per_group_batches` the Python
//! `GroupedGemmArgs` row carries. The scalar itself does NOT enter the wire
//! args — the `per_group_batches` vector already uniquely keys a measured point.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::routing::RoutingDistribution;
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Deserialize)]
pub struct GroupedGemmKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub n: u32,
    pub k: u32,
    #[compute_dtype]
    pub dtype: DType,
    /// This GPU's raw ppm shard for its local experts (a slice of the global
    /// `RoutingDistribution`, so `Σ < TOTAL_PPM`). Identity: distinct shards are
    /// distinct kernels/caches. `len` == this GPU's local expert count.
    pub local_ppm: Vec<u32>,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct GroupedGemmKernelInput {
    pub global_expert_selections: u32,
}

pub struct GroupedGemmSpec;

impl KernelSpec for GroupedGemmSpec {
    type Config = GroupedGemmKernelConfig;
    type Input = GroupedGemmKernelInput;

    const KIND: KernelKind = "grouped_gemm";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // grouped_gemm sweeps `global_expert_selections`, which on a per-GPU MoE
        // expert worklet is often tiny (decode / small batch / skewed routing),
        // so this kernel needs low-end points the shared `token_axis` (starts at
        // 32) lacks. Prepend [1,4,8,16,32,48]; 32 is included only so 48 lands
        // between 32 and 64 in the merged strictly-increasing axis (it dedupes
        // against token_axis's leading 32).
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
        grid.expand_1d(|global_expert_selections| {
            let global_expert_selections = global_expert_selections as u32;
            let per_group_batches = RoutingDistribution::to_per_expert_counts(
                global_expert_selections,
                &config.local_ppm,
            );
            ArgsPayload::new()
                .with("backend", backend)
                .with("n", config.n)
                .with("k", config.k)
                .with("dtype", config.dtype.as_str())
                .with("num_local_experts", num_local_experts)
                .with("per_group_batches", per_group_batches)
        })
    }
}

register_kernel!(GroupedGemmKernel, GroupedGemmSpec);

#[cfg(test)]
mod tests {
    use super::{GroupedGemmKernelConfig, GroupedGemmKernelInput, GroupedGemmSpec};
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn config() -> GroupedGemmKernelConfig {
        GroupedGemmKernelConfig {
            backends: vec!["torch"],
            gpu_name: "H100".to_string(),
            n: 4096,
            k: 8192,
            dtype: DType::Bf16,
            local_ppm: vec![300_000, 200_000],
        }
    }

    #[test]
    fn config_identity_includes_distribution_shard() {
        let cfg = config();
        assert_eq!(cfg.backends, vec!["torch"]);
        assert_eq!(cfg.local_ppm, vec![300_000, 200_000]);
        // A different shard is a different config (distinct cache).
        let mut other = config();
        other.local_ppm = vec![250_000, 250_000];
        assert_ne!(cfg, other);
    }

    #[test]
    fn describe_config_renders_every_field_in_order() {
        assert_eq!(
            config().describe_config(),
            r#"backends=["torch"] gpu_name="H100" n=4096 k=8192 dtype=Bf16 local_ppm=[300000, 200000]"#
        );
    }

    #[test]
    fn input_sweep_coords_flatten_global_expert_selections() {
        let input = GroupedGemmKernelInput {
            global_expert_selections: 2048,
        };
        assert_eq!(&*input.coords(), &[2048.0]);
    }

    #[test]
    fn sweep_grid_is_1d_and_cache_is_linear() {
        let grid = GroupedGemmSpec::sweep_grid(&config());
        assert_eq!(grid.axes().len(), 1);
        assert!(matches!(
            GroupedGemmSpec::cache_kind("torch"),
            CacheKind::Cache1DLinear
        ));
    }

    #[test]
    fn sweep_axis_prepends_low_end_points_and_stays_increasing() {
        let grid = GroupedGemmSpec::sweep_grid(&config());
        let axis = &grid.axes()[0];
        // MoE low-end points present (token_axis alone starts at 32).
        for v in [1.0, 4.0, 8.0, 16.0, 48.0] {
            assert!(axis.contains(&v), "axis missing low-end point {v}");
        }
        // 32 appears once (deduped against token_axis), 48 sits before 64.
        assert_eq!(axis.iter().filter(|&&v| v == 32.0).count(), 1);
        assert!(axis.windows(2).all(|w| w[0] < w[1]), "axis must be strictly increasing");
    }

    #[test]
    fn enumerate_emits_per_group_batches_from_distribution() {
        let cfg = config();
        let grid = GroupedGemmSpec::sweep_grid(&cfg);
        let payloads = GroupedGemmSpec::enumerate(&cfg, &grid, "torch");
        assert!(!payloads.is_empty(), "token sweep must yield points");

        let first = &payloads[0];
        let fields = first.fields();
        // Wire schema: { backend, n, k, dtype, num_local_experts, per_group_batches }
        // — must stay aligned with Python GroupedGemmArgs.
        assert_eq!(fields.len(), 6);
        assert_eq!(fields.get("backend"), Some(&Value::from("torch")));
        assert_eq!(fields.get("n"), Some(&Value::from(4096_u32)));
        assert_eq!(fields.get("k"), Some(&Value::from(8192_u32)));
        assert_eq!(fields.get("dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("num_local_experts"), Some(&Value::from(2_u32)));

        let batches = fields
            .get("per_group_batches")
            .and_then(Value::as_array)
            .expect("per_group_batches is a JSON array");
        // One entry per local expert; the 300k:200k shard splits 3:2.
        assert_eq!(batches.len(), 2);
        assert!(batches[0].as_u64().unwrap() >= batches[1].as_u64().unwrap());
        assert_eq!(first.backend(), Some("torch"));
    }
}
