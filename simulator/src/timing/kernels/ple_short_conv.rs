//! PLE dilated depthwise short convolution with state update: one call of the
//! fork's `ple_conv` (`_ple_conv_kernel`, plus `_ple_conv_writeback_kernel` in
//! `spec` and `prefill` modes). The mode is a constexpr of the kernel chosen by
//! the request type, so it is Config identity:
//!
//! - `decode`: one token per request, state shifted in the same launch;
//! - `spec`: `query_len` speculative verify rows per request, state rolled back
//!   to the last accepted token and rewritten by the second launch;
//! - `prefill`: `query_len` new tokens per request.
//!
//! The cache is over the physical `(batch_size, query_len)` caller shape of
//! equal-length requests; the grid's query axis depends on the mode.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_grid;
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

const MODES: [&str; 3] = ["decode", "spec", "prefill"];
/// Largest tokens (batch x query_len) in one profiled call. A grid budget that
/// bounds the prefill sweep, not a kernel limit.
const MAX_CALL_TOKENS: u64 = 65_536;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PleShortConvKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Depthwise conv channels (`hc_mult * hidden_size`).
    pub channels: u32,
    /// Filter taps per channel; the state holds `(kernel_size - 1) * dilation` inputs.
    pub kernel_size: u32,
    /// Spacing between taps, in tokens.
    pub dilation: u32,
    /// `decode`, `spec` or `prefill`.
    pub mode: String,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(SweepCoords, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PleShortConvKernelInput {
    /// Requests in the call.
    pub batch_size: u32,
    /// Tokens per request (1 in decode).
    pub query_len: u32,
}

pub struct PleShortConvSpec;

impl KernelSpec for PleShortConvSpec {
    type Config = PleShortConvKernelConfig;
    type Input = PleShortConvKernelInput;
    const KIND: KernelKind = "ple_short_conv";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        assert!(
            MODES.contains(&config.mode.as_str()),
            "mode must be one of {MODES:?}, got {}",
            config.mode
        );
        assert!(
            config.kernel_size >= 2,
            "ple_short_conv needs kernel_size >= 2"
        );
        if neuron_grid::is_neuron(&config.backends) {
            let decode = || neuron_grid::axis(neuron_grid::DECODE);
            return match config.mode.as_str() {
                "decode" => SweepGrid::new(vec![decode(), Axis::values([1])]),
                "spec" => SweepGrid::new(vec![decode(), Axis::values([2, 3, 4])]),
                _ => SweepGrid::new(vec![
                    Axis::values([1]),
                    neuron_grid::axis(neuron_grid::TOKENS),
                ]),
            };
        }
        let batch = || {
            Axis::values([
                1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024,
            ])
        };
        match config.mode.as_str() {
            "decode" => SweepGrid::new(vec![batch(), Axis::values([1])]),
            "spec" => SweepGrid::new(vec![batch(), Axis::values([1, 2, 3, 4, 6, 8])]),
            _ => SweepGrid::new(vec![
                Axis::values([1, 2, 4, 8, 16, 32, 64, 128, 256]),
                Axis::values([
                    1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
                ]),
            ]),
        }
    }

    fn cache_kind(_: &'static str) -> CacheKind {
        // Work is batch_size x query_len tokens.
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_2d(|batch, query| {
            (batch.round() as u64) * (query.round() as u64) > MAX_CALL_TOKENS
        })
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|batch, query| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("batch_size", batch.round() as u32)
                .with("query_len", query.round() as u32)
                .with("channels", config.channels)
                .with("kernel_size", config.kernel_size)
                .with("dilation", config.dilation)
                .with("mode", config.mode.clone())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(PleShortConvKernel, PleShortConvSpec);

#[cfg(test)]
mod tests {
    use super::*;

    fn config(mode: &str) -> PleShortConvKernelConfig {
        PleShortConvKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            channels: 10240,
            kernel_size: 4,
            dilation: 3,
            mode: mode.to_string(),
            dtype: DType::Bf16,
        }
    }

    /// Catches a decode grid that asks the runner for query_len > 1 (rejected
    /// in decode mode) and a payload field the Python schema would reject.
    #[test]
    fn decode_grid_has_single_token_requests_and_payload_matches_python_args() {
        let cfg = config("decode");
        let grid = PleShortConvSpec::sweep_grid(&cfg);
        let payloads = PleShortConvSpec::enumerate(&cfg, &grid, "vllm_triton");
        assert!(payloads.iter().all(|p| p.fields()["query_len"] == 1));
        let keys: Vec<_> = payloads[0].fields().keys().map(String::as_str).collect();
        assert_eq!(
            keys,
            [
                "backend",
                "batch_size",
                "channels",
                "dilation",
                "dtype",
                "kernel_size",
                "mode",
                "query_len"
            ]
        );
    }

    /// The prefill grid stays inside the profiled token budget and under the
    /// 500 feasible-coordinate ceiling.
    #[test]
    fn prefill_grid_masks_oversized_calls() {
        let cfg = config("prefill");
        let grid = PleShortConvSpec::sweep_grid(&cfg);
        let mask = PleShortConvSpec::infeasible_mask(&cfg, &grid);
        let feasible = mask.iter().filter(|m| !**m).count();
        assert_eq!(feasible, 90);
    }

    #[test]
    #[should_panic(expected = "mode")]
    fn rejects_unknown_mode() {
        PleShortConvSpec::sweep_grid(&config("mixed"));
    }

    /// Catches the Neuron branch leaking into a GPU or mixed config (which would
    /// resample every GPU row) or a Neuron config missing the contract grid.
    #[test]
    fn neuron_backends_take_the_contract_grids_and_gpu_grid_is_unchanged() {
        use crate::timing::kernels::neuron_grid::{axis, DECODE, TOKENS};
        let grid = |mode: &str, backends: Vec<&'static str>| {
            let mut cfg = config(mode);
            cfg.backends = backends;
            PleShortConvSpec::sweep_grid(&cfg).axes().to_vec()
        };
        let neuron = || vec!["neuron_torch"];
        assert_eq!(grid("decode", neuron()), [axis(DECODE), Axis::values([1])]);
        assert_eq!(
            grid("spec", neuron()),
            [axis(DECODE), Axis::values([2, 3, 4])]
        );
        assert_eq!(grid("prefill", neuron()), [Axis::values([1]), axis(TOKENS)]);
        let mut cfg = config("prefill");
        cfg.backends = neuron();
        let mask = PleShortConvSpec::infeasible_mask(&cfg, &PleShortConvSpec::sweep_grid(&cfg));
        assert!(mask.iter().all(|m| !*m));

        let batch = Axis::values([
            1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024,
        ]);
        for backends in [vec!["vllm_triton"], vec!["neuron_torch", "vllm_triton"]] {
            assert_eq!(
                grid("decode", backends.clone()),
                [batch.clone(), Axis::values([1])]
            );
            assert_eq!(
                grid("spec", backends.clone()),
                [batch.clone(), Axis::values([1, 2, 3, 4, 6, 8])]
            );
            assert_eq!(
                grid("prefill", backends),
                [
                    Axis::values([1, 2, 4, 8, 16, 32, 64, 128, 256]),
                    Axis::values([
                        1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
                    ]),
                ]
            );
        }
    }
}
