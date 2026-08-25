//! DeepSeek V4 sparse FP8 MLA decode graph replay.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeepseekV4SparseMlaDecodeKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    pub value_dim: Dim,
    pub swa_window: u32,
    pub extra_index_capacity: u32,
    pub compress_ratio: u32,
    #[compute_dtype]
    pub q_dtype: DType,
    #[kv_dtype]
    pub cache_dtype: DType,
    pub output_dtype: DType,
    pub planner_mode: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeepseekV4SparseMlaDecodeKernelInput {
    pub swa_valid_counts: Vec<u32>,
    pub extra_valid_counts: Vec<u32>,
}

impl DeepseekV4SparseMlaDecodeKernelInput {
    fn work(&self) -> (u32, f64) {
        assert!(!self.swa_valid_counts.is_empty());
        assert_eq!(self.swa_valid_counts.len(), self.extra_valid_counts.len());
        assert!(self.swa_valid_counts.len() <= 256);
        let total = self
            .swa_valid_counts
            .iter()
            .chain(&self.extra_valid_counts)
            .try_fold(0_u64, |sum, &count| sum.checked_add(u64::from(count)))
            .expect("selected-row total must fit u64");
        let batch_size = self.swa_valid_counts.len() as u32;
        (batch_size, total as f64 / f64::from(batch_size))
    }
}

impl SweepCoords for DeepseekV4SparseMlaDecodeKernelInput {
    fn coords(&self) -> Coords {
        let (batch_size, mean_valid_rows) = self.work();
        Coords::new([f64::from(batch_size), mean_valid_rows])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["batch_size", "mean_valid_rows"]
    }
}

fn valid_axis(config: &DeepseekV4SparseMlaDecodeKernelConfig) -> Vec<f64> {
    match config.compress_ratio {
        1 => Axis::values([1, 32, 64, 96, 128]),
        4 => Axis::values([128, 160, 192, 256, 384, 512, 640]),
        128 => Axis::values([1, 64, 128, 160, 192, 256, 384, 512, 640])
            .into_iter()
            .filter(|&valid| valid <= f64::from(128 + config.extra_index_capacity))
            .collect(),
        ratio => panic!("unsupported compress_ratio {ratio}"),
    }
}

fn validate_config(config: &DeepseekV4SparseMlaDecodeKernelConfig) {
    match config.compress_ratio {
        1 => assert_eq!(config.extra_index_capacity, 0),
        4 => assert_eq!(config.extra_index_capacity, 512),
        128 => assert!(
            (128..=8192).contains(&config.extra_index_capacity)
                && config.extra_index_capacity % 128 == 0
        ),
        ratio => panic!("unsupported compress_ratio {ratio}"),
    }
    assert!(matches!(config.planner_mode.as_str(), "planned" | "reused"));
}

fn canonical_counts(
    batch_size: u32,
    total_valid_per_row: u32,
    compress_ratio: u32,
) -> (Vec<u32>, Vec<u32>) {
    let swa = total_valid_per_row.min(128);
    let extra = if compress_ratio == 1 {
        0
    } else {
        total_valid_per_row - swa
    };
    (
        vec![swa; batch_size as usize],
        vec![extra; batch_size as usize],
    )
}

pub struct DeepseekV4SparseMlaDecodeSpec;

impl KernelSpec for DeepseekV4SparseMlaDecodeSpec {
    type Config = DeepseekV4SparseMlaDecodeKernelConfig;
    type Input = DeepseekV4SparseMlaDecodeKernelInput;

    const KIND: KernelKind = "deepseek_v4_sparse_mla_decode";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        validate_config(config);
        SweepGrid::new(vec![
            Axis::values([1, 2, 4, 8, 16, 32, 64, 128, 256]),
            valid_axis(config),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|batch_size, total_valid_per_row| {
            let (swa_valid_counts, extra_valid_counts) = canonical_counts(
                batch_size as u32,
                total_valid_per_row as u32,
                config.compress_ratio,
            );
            ArgsPayload::new()
                .with("backend", backend)
                .with("swa_valid_counts", swa_valid_counts)
                .with("extra_valid_counts", extra_valid_counts)
                .with("num_heads", config.num_heads.get())
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("value_dim", config.value_dim.get())
                .with("swa_window", config.swa_window)
                .with("extra_index_capacity", config.extra_index_capacity)
                .with("compress_ratio", config.compress_ratio)
                .with("q_dtype", config.q_dtype.as_str())
                .with("cache_dtype", config.cache_dtype.as_str())
                .with("output_dtype", config.output_dtype.as_str())
                .with("planner_mode", config.planner_mode.clone())
        })
    }
}

register_kernel!(
    DeepseekV4SparseMlaDecodeKernel,
    DeepseekV4SparseMlaDecodeSpec
);

#[cfg(test)]
mod tests {
    use super::DeepseekV4SparseMlaDecodeKernelInput;
    use crate::timing::SweepCoords;

    #[test]
    fn ragged_counts_project_to_exact_mean_selected_work() {
        let input = DeepseekV4SparseMlaDecodeKernelInput {
            swa_valid_counts: vec![128, 64, 32, 16],
            extra_valid_counts: vec![256, 128, 64, 32],
        };
        assert_eq!(&*input.coords(), &[4.0, 180.0]);
    }
}
