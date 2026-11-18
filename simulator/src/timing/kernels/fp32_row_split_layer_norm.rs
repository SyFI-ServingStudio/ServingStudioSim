//! Epilogue of a concatenated GEMM with an fp32 output: each fp32 row is split
//! into a bf16-rounded lead span, a LayerNorm span (fp32 math on its
//! bf16-rounded values, written bf16), a bf16-rounded trail span, and an fp32
//! tail left in place.
//!
//! Backend `triton` is a repo-vendored kernel (one program per row) whose row
//! geometry is compiled in; its runner refuses any other geometry.

use crate::timing::bridge::{de_backends, ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Fp32RowSplitLayerNormKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Leading columns rounded to bf16 into their own buffer.
    pub bf16_lead_cols: Dim,
    /// Columns LayerNormed into bf16.
    pub norm_cols: Dim,
    /// Columns after the LayerNorm span, rounded to bf16.
    pub bf16_trail_cols: Dim,
    /// Last columns, read in place as fp32 (not touched by the launch).
    pub fp32_tail_cols: Dim,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct Fp32RowSplitLayerNormKernelInput {
    pub num_tokens: u32,
}

pub struct Fp32RowSplitLayerNormSpec;

impl KernelSpec for Fp32RowSplitLayerNormSpec {
    type Config = Fp32RowSplitLayerNormKernelConfig;
    type Input = Fp32RowSplitLayerNormKernelInput;
    const KIND: KernelKind = "fp32_row_split_layer_norm";

    fn sweep_grid(_: &Self::Config) -> SweepGrid {
        // One program per row: time grows with rows, no in-call switches.
        let mut tokens = Axis::chain([Axis::pow2(0, 4), Axis::token_axis()]);
        tokens.sort_by(f64::total_cmp);
        tokens.dedup();
        SweepGrid::new(vec![tokens])
    }

    fn cache_kind(_: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", tokens as u32)
                .with("bf16_lead_cols", config.bf16_lead_cols.get())
                .with("norm_cols", config.norm_cols.get())
                .with("bf16_trail_cols", config.bf16_trail_cols.get())
                .with("fp32_tail_cols", config.fp32_tail_cols.get())
        })
    }
}

register_kernel!(Fp32RowSplitLayerNormKernel, Fp32RowSplitLayerNormSpec);

#[cfg(test)]
mod tests {
    use super::{Fp32RowSplitLayerNormKernelConfig, Fp32RowSplitLayerNormSpec};
    use crate::timing::kernels::engine::KernelSpec;

    /// Catches a span width sent under another field's name: the runner
    /// would refuse the row (or time the wrong geometry).
    #[test]
    fn payload_carries_each_span_under_its_python_field() {
        let config = Fp32RowSplitLayerNormKernelConfig {
            backends: vec!["triton"],
            gpu_name: "NVIDIA B200".to_string(),
            bf16_lead_cols: 2048.into(),
            norm_cols: 128.into(),
            bf16_trail_cols: 128.into(),
            fp32_tail_cols: 32.into(),
        };
        let grid = Fp32RowSplitLayerNormSpec::sweep_grid(&config);
        let axis = &grid.axes()[0];
        assert!(axis.windows(2).all(|w| w[0] < w[1]));
        for rows in [1.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 384.0, 640.0] {
            assert!(axis.contains(&rows), "missing num_tokens={rows}");
        }
        assert!(axis.len() <= 500);
        let payload = Fp32RowSplitLayerNormSpec::enumerate(&config, &grid, "triton").remove(0);
        let fields = payload.fields();
        assert_eq!(fields["num_tokens"], 1);
        assert_eq!(fields["bf16_lead_cols"], 2048);
        assert_eq!(fields["norm_cols"], 128);
        assert_eq!(fields["bf16_trail_cols"], 128);
        assert_eq!(fields["fp32_tail_cols"], 32);
        assert_eq!(fields.len(), 6);
    }
}
