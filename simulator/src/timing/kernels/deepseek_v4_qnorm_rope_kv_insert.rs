//! DeepSeek V4 fused Q normalization/RoPE and packed KV insert.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeepseekV4QnormRopeKvInsertKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: u32,
    pub padded_heads: u32,
    pub head_dim: Dim,
    pub rope_dim: Dim,
    pub block_size: u32,
    pub rms_eps_bits: u64,
    #[compute_dtype]
    pub input_dtype: DType,
    #[kv_dtype]
    pub kv_dtype: DType,
    pub cache_dtype: String,
    pub cache_layout: String,
    pub scale_format: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeepseekV4QnormRopeKvInsertKernelInput {
    pub num_tokens: u32,
    pub num_insert_tokens: u32,
}

impl SweepCoords for DeepseekV4QnormRopeKvInsertKernelInput {
    fn coords(&self) -> Coords {
        assert!(self.num_tokens > 0 && self.num_insert_tokens <= self.num_tokens);
        Coords::new([
            f64::from(self.num_tokens),
            f64::from(self.num_insert_tokens) / f64::from(self.num_tokens),
        ])
    }
    fn coord_field_names() -> &'static [&'static str] {
        &["num_tokens", "insert_fraction"]
    }
}

pub struct DeepseekV4QnormRopeKvInsertSpec;

impl KernelSpec for DeepseekV4QnormRopeKvInsertSpec {
    type Config = DeepseekV4QnormRopeKvInsertKernelConfig;
    type Input = DeepseekV4QnormRopeKvInsertKernelInput;
    const KIND: KernelKind = "deepseek_v4_qnorm_rope_kv_insert";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        assert_eq!(
            (
                config.num_heads,
                config.padded_heads,
                config.head_dim.get(),
                config.rope_dim.get(),
                config.block_size
            ),
            (64, 64, 512, 64, 256)
        );
        assert_eq!(f64::from_bits(config.rms_eps_bits), 1.0e-6);
        let mut tokens = Axis::chain([
            Axis::pow2(0, 9),
            Axis::values([1023, 1024, 1025]),
            Axis::token_axis(),
        ]);
        tokens.sort_by(f64::total_cmp);
        tokens.dedup();
        SweepGrid::new(vec![tokens, vec![0.0, 0.25, 0.5, 0.75, 1.0]])
    }

    fn cache_kind(_: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Weighted)
    }

    fn infeasible_mask(_: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_2d(|tokens, fraction| (tokens * fraction).fract() != 0.0)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|tokens, fraction| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", tokens as u32)
                .with("num_insert_tokens", (tokens * fraction).round() as u32)
                .with("num_heads", config.num_heads)
                .with("padded_heads", config.padded_heads)
                .with("head_dim", config.head_dim.get())
                .with("rope_dim", config.rope_dim.get())
                .with("block_size", config.block_size)
                .with("rms_eps", f64::from_bits(config.rms_eps_bits))
                .with("input_dtype", config.input_dtype.as_str())
                .with("cache_dtype", config.cache_dtype.clone())
                .with("cache_layout", config.cache_layout.clone())
                .with("scale_format", config.scale_format.clone())
        })
    }
}

register_kernel!(
    DeepseekV4QnormRopeKvInsertKernel,
    DeepseekV4QnormRopeKvInsertSpec
);

#[cfg(test)]
mod tests {
    use super::DeepseekV4QnormRopeKvInsertKernelInput;
    use crate::timing::SweepCoords;

    #[test]
    fn dp_padding_is_preserved_as_insert_fraction() {
        let input = DeepseekV4QnormRopeKvInsertKernelInput {
            num_tokens: 128,
            num_insert_tokens: 96,
        };
        assert_eq!(&*input.coords(), &[128.0, 0.75]);
    }
}
