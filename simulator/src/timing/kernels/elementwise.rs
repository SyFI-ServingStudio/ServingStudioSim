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

use crate::timing::bridge::{de_backends, ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Deserialize)]
pub struct ElementwiseKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub input_bytes_per_token: u32,
    pub output_bytes_per_token: u32,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct ElementwiseKernelInput {
    pub num_tokens: u32,
}

pub struct ElementwiseSpec;

impl KernelSpec for ElementwiseSpec {
    type Config = ElementwiseKernelConfig;
    type Input = ElementwiseKernelInput;

    const KIND: KernelKind = "elementwise";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
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
                .with("input_size_bytes", config.input_bytes_per_token as u64 * tokens)
                .with("output_size_bytes", config.output_bytes_per_token as u64 * tokens)
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
            input_bytes_per_token: 8192,
            output_bytes_per_token: 4096,
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
            r#"backends=["triton"] gpu_name="H100" input_bytes_per_token=8192 output_bytes_per_token=4096"#
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
        assert_eq!(fields.get("input_size_bytes"), Some(&Value::from(8192_u64 * 32)));
        assert_eq!(fields.get("output_size_bytes"), Some(&Value::from(4096_u64 * 32)));
        assert_eq!(first.backend(), Some("triton"));
    }
}
