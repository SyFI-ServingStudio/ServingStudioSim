//! Fused gate|up GEMM + SwiGLU + NVFP4 quantization: one cached curve over `m`
//! per `(n, k, dtype)` config.
//!
//! One launch computes both projections of the same quantized rows, applies
//! SiLU(gate)·up to the accumulators and writes the product as NVFP4 rows for
//! the down projection. Its input quantization and the down projection are
//! separate kinds. The Python `GemmSwigluQuantArgs` dataclass
//! (`profiling/kernels/gemm_swiglu_quant.py`) owns the schema.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GemmSwigluQuantKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Intermediate size: the gate's and the up projection's width, each.
    pub n: Dim,
    /// Hidden size, the reduction dimension.
    pub k: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct GemmSwigluQuantKernelInput {
    pub m: u32,
}

pub struct GemmSwigluQuantSpec;

/// Rows of one block of TensorFold's fused launch (`gemm_gu_ck`: 128 x 256).
const ROW_TILE: u32 = 128;
/// Top of the densely measured range: TensorFold's largest prompt chunk.
const DENSE_M_MAX: u32 = 4096;
/// One point past the dense range so a longer step extrapolates from the
/// average per-tile slope instead of the last step's.
const M_MAX: u32 = 8192;

/// `m` axis: powers of two inside the first row tile, then both sides of every
/// row-tile boundary (`128 j` and `128 j + 1`) up to 4096, then 8192.
///
/// The launch has `ceil(m / 128)` row tiles of fixed cost, so time is a
/// staircase in `m`. Measuring both sides of each step keeps linear
/// interpolation from spanning one.
fn m_axis() -> Vec<f64> {
    let mut m: Vec<u32> = (0..=7)
        .map(|i| 1u32 << i)
        .chain((ROW_TILE..=DENSE_M_MAX).step_by(ROW_TILE as usize))
        .chain((ROW_TILE + 1..DENSE_M_MAX).step_by(ROW_TILE as usize))
        .chain([M_MAX])
        .collect();
    m.sort_unstable();
    m.dedup();
    Axis::values(m)
}

impl KernelSpec for GemmSwigluQuantSpec {
    type Config = GemmSwigluQuantKernelConfig;
    type Input = GemmSwigluQuantKernelInput;

    const KIND: KernelKind = "gemm_swiglu_quant";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![m_axis()])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|m| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("m", m as u32)
                .with("n", config.n.get())
                .with("k", config.k.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(GemmSwigluQuantKernel, GemmSwigluQuantSpec);

#[cfg(test)]
mod tests {
    use super::{GemmSwigluQuantKernelConfig, GemmSwigluQuantSpec};
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::KernelSpec;
    use serde_json::Value;

    fn cfg() -> GemmSwigluQuantKernelConfig {
        GemmSwigluQuantKernelConfig {
            backends: vec!["tensorfold_ck"],
            gpu_name: "NVIDIA GB10".to_string(),
            n: 17408.into(),
            k: 5120.into(),
            dtype: DType::Nvfp4E2m1,
        }
    }

    /// A missing side of a row-tile step makes the linear cache interpolate
    /// across it, mispricing every chunk just past a multiple of 128.
    #[test]
    fn grid_brackets_every_row_tile_step() {
        let grid = GemmSwigluQuantSpec::sweep_grid(&cfg());
        let axis = &grid.axes()[0];
        assert!(axis.windows(2).all(|w| w[0] < w[1]));
        for j in 1..32u32 {
            assert!(axis.contains(&f64::from(128 * j)), "m = {} missing", 128 * j);
            assert!(axis.contains(&f64::from(128 * j + 1)), "m = {} missing", 128 * j + 1);
        }
        assert_eq!(axis.first(), Some(&1.0));
        assert!(axis.contains(&4096.0));
        assert_eq!(axis.last(), Some(&8192.0));
        assert!(axis.len() <= 500);
    }

    /// The payload must carry exactly the Python `GemmSwigluQuantArgs` fields
    /// plus `backend`, or every query misses its table rows.
    #[test]
    fn enumerate_forwards_the_python_wire_schema() {
        let cfg = cfg();
        let grid = GemmSwigluQuantSpec::sweep_grid(&cfg);
        let payloads = GemmSwigluQuantSpec::enumerate(&cfg, &grid, "tensorfold_ck");
        assert_eq!(payloads.len(), grid.axes()[0].len());
        let fields = payloads[0].fields();
        assert_eq!(fields.len(), 5);
        assert_eq!(fields.get("backend"), Some(&Value::from("tensorfold_ck")));
        assert_eq!(fields.get("m"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("n"), Some(&Value::from(17408_u32)));
        assert_eq!(fields.get("k"), Some(&Value::from(5120_u32)));
        assert_eq!(fields.get("dtype"), Some(&Value::from("nvfp4_e2m1")));
    }
}
