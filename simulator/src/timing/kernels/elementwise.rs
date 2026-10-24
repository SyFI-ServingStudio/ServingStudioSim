//! Element-wise kernel: a generic byte-level fan-in elementwise/reduce, modeled
//! as one cached perf curve per `(input_bytes_per_token, output_bytes_per_token)`
//! config, swept over `num_tokens`.
//!
//! Everything generic (build / eval / the `Probe` impl /
//! for-backend loops) lives in `engine::Kernel<S>`. This file declares the
//! element-wise Config / Input, the `KIND` wire string, and the `enumerate`
//! body that lifts (config, sweep coord, backend) to the on-wire `ArgsPayload`.
//! The Python `ElementwiseArgs` dataclass owns the schema.
//!
//! Shape split: this mirrors `ref/profile/elementwise/elementwise_triton.py`,
//! whose DB is keyed by *total* `(input_size_bytes, output_size_bytes)`. The
//! byte footprint scales linearly with the token count, so the static identity
//! is the per-token byte rate (`input_bytes_per_token`, `output_bytes_per_token`
//! — the fan-in ratio is therefore config-stable), and the runtime sweep axis is
//! `num_tokens`. `enumerate` folds rate × num_tokens into the total-byte wire
//! payload, so the Python runner side stays keyed by total bytes (matching the
//! reference) while the Rust cache interpolates over tokens (`Cache1DLinear`).
//!
//! TensorFold's backends (`tensorfold_*`, each one fixed eager statement or the
//! plan's `_paths`) run from a single row or stream up, so a config whose
//! backends are all TensorFold's gets its own axis: powers of two and three
//! times powers of two from 1 to 65,536 units, stopping where one call would
//! move more than 1 GiB (in + out), past which every statement is a flat byte
//! rate and the cache extrapolates linearly.

use crate::timing::bridge::{de_backends, ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ElementwiseKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub input_bytes_per_token: Dim,
    pub output_bytes_per_token: Dim,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct ElementwiseKernelInput {
    pub num_tokens: u32,
}

pub struct ElementwiseSpec;

/// Largest bytes (input + output) one TensorFold grid point moves.
const TENSORFOLD_MAX_BYTES: u64 = 1 << 30;

fn is_tensorfold(config: &ElementwiseKernelConfig) -> bool {
    !config.backends.is_empty() && config.backends.iter().all(|b| b.starts_with("tensorfold_"))
}

/// `{1, 2, 3, 4, 6, 8, 12, ..., 49152, 65536}` units, cut at
/// [`TENSORFOLD_MAX_BYTES`] for this config's byte rates (at least two points).
fn tensorfold_units(config: &ElementwiseKernelConfig) -> Vec<f64> {
    let per_unit = (config.input_bytes_per_token.get() as u64
        + config.output_bytes_per_token.get() as u64)
        .max(1);
    let mut units: Vec<u64> = (0..=16)
        .map(|k| 1u64 << k)
        .chain((0..=14).map(|k| 3u64 << k))
        .collect();
    units.sort_unstable();
    let kept: Vec<f64> = units
        .iter()
        .enumerate()
        .filter(|&(i, &u)| i < 2 || u * per_unit <= TENSORFOLD_MAX_BYTES)
        .map(|(_, &u)| u as f64)
        .collect();
    kept
}

impl KernelSpec for ElementwiseSpec {
    type Config = ElementwiseKernelConfig;
    type Input = ElementwiseKernelInput;

    const KIND: KernelKind = "elementwise";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        if is_tensorfold(config) {
            return SweepGrid::new(vec![tensorfold_units(config)]);
        }
        SweepGrid::new(vec![Axis::token_axis()])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|num_tokens| {
            let tokens = num_tokens as u64;
            ArgsPayload::new()
                .with("backend", backend)
                .with(
                    "input_size_bytes",
                    config.input_bytes_per_token.get() as u64 * tokens,
                )
                .with(
                    "output_size_bytes",
                    config.output_bytes_per_token.get() as u64 * tokens,
                )
        })
    }
}

register_kernel!(ElementwiseKernel, ElementwiseSpec);

#[cfg(test)]
mod tests {
    use super::{ElementwiseKernelConfig, ElementwiseKernelInput, ElementwiseSpec};
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn config() -> ElementwiseKernelConfig {
        // MoE-activation-style fan-in 2 (2N -> N): 8192 in / 4096 out per token.
        ElementwiseKernelConfig {
            backends: vec!["triton"],
            gpu_name: "H100".to_string(),
            input_bytes_per_token: 8192.into(),
            output_bytes_per_token: 4096.into(),
        }
    }

    #[test]
    fn config_identity_includes_backend_and_byte_rates() {
        let cfg = config();
        assert_eq!(cfg.backends, vec!["triton"]);
        assert_eq!(cfg.input_bytes_per_token, 8192);
        assert_eq!(cfg.output_bytes_per_token, 4096);
    }

    #[test]
    fn describe_config_renders_tidy_field_list() {
        assert_eq!(
            config().describe_config(),
            serde_json::json!({
                "backends": ["triton"], "gpu_name": "H100",
                "input_bytes_per_token": {"value": 8192, "expression": null, "bindings": {}},
                "output_bytes_per_token": {"value": 4096, "expression": null, "bindings": {}},
            })
        );
    }

    #[test]
    fn input_sweep_coords_flatten_num_tokens() {
        let input = ElementwiseKernelInput { num_tokens: 1024 };
        assert_eq!(&*input.coords(), &[1024.0]);
    }

    #[test]
    fn cache_kind_is_one_dimensional_linear() {
        assert_eq!(
            ElementwiseSpec::cache_kind("triton"),
            CacheKind::Cache1DLinear
        );
    }

    #[test]
    fn enumerate_folds_byte_rate_times_tokens_into_total_bytes() {
        let cfg = config();
        let grid = ElementwiseSpec::sweep_grid(&cfg);
        let payloads = ElementwiseSpec::enumerate(&cfg, &grid, "triton");

        assert!(
            !payloads.is_empty(),
            "token sweep axis must yield at least one point"
        );
        let first = &payloads[0];
        let fields = first.fields();

        // Wire schema: { backend, input_size_bytes, output_size_bytes } — must
        // stay aligned with Python ElementwiseArgs in
        // profiling/kernels/elementwise.py. num_tokens / per-token rates are
        // Rust-side only; the wire carries the folded TOTAL bytes.
        assert_eq!(fields.len(), 3);
        assert_eq!(fields.get("backend"), Some(&Value::from("triton")));
        // First token-axis point is 32 (see Axis::token_axis): 8192*32, 4096*32.
        assert_eq!(
            fields.get("input_size_bytes"),
            Some(&Value::from(8192_u64 * 32))
        );
        assert_eq!(
            fields.get("output_size_bytes"),
            Some(&Value::from(4096_u64 * 32))
        );
        assert_eq!(first.backend(), Some("triton"));
    }

    #[test]
    fn tensorfold_axis_starts_at_one_and_stops_at_a_gib() {
        // Catches a TensorFold config landing on the 32-token vLLM axis (every
        // per-stream copy below 32 would extrapolate), or a grid point too big
        // to allocate (a sampling cast moves 1.5 MB a row).
        let cfg = ElementwiseKernelConfig {
            backends: vec!["tensorfold_torch_cast"],
            gpu_name: "NVIDIA GB10".to_string(),
            input_bytes_per_token: 496_640.into(),
            output_bytes_per_token: 993_280.into(),
        };
        let grid = ElementwiseSpec::sweep_grid(&cfg);
        let units = &grid.axes()[0];
        assert_eq!(&units[..4], &[1.0, 2.0, 3.0, 4.0]);
        let last = *units.last().unwrap() as u64;
        assert!(
            last * 1_489_920 <= 1 << 30 && last >= 512,
            "last unit {last}"
        );
        let mixed = ElementwiseKernelConfig {
            backends: vec!["tensorfold_torch_cast", "triton"],
            ..cfg
        };
        assert_eq!(ElementwiseSpec::sweep_grid(&mixed).axes()[0][0], 32.0);
    }
}
