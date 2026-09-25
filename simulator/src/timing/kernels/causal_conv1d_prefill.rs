//! Kimi-K3 SGLang variable-length causal convolution during prefill.
//!
//! The runtime cache keys this fused launch by append length, request count,
//! and whether the request starts from a non-empty convolution state.  The
//! token count remains a Python-facing argument because it is the physical
//! launch size, while the canonical sweep uses the equivalent `(L, B)` axes.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

const MAX_TOKENS: u64 = 65_536;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CausalConv1dPrefillKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub channels: Dim,
    pub kernel_size: Dim,
    #[compute_dtype]
    pub dtype: DType,
    pub state_dtype: DType,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct CausalConv1dPrefillKernelInput {
    pub num_tokens: u32,
    pub max_sequence_length: u32,
    pub num_sequences: u32,
    pub prefix_len: u32,
}

impl SweepCoords for CausalConv1dPrefillKernelInput {
    fn coords(&self) -> Coords {
        Coords::new([
            self.max_sequence_length as f64,
            self.num_sequences as f64,
            self.prefix_len as f64,
        ])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["max_sequence_length", "num_sequences", "prefix_len"]
    }
}

pub struct CausalConv1dPrefillSpec;

impl KernelSpec for CausalConv1dPrefillSpec {
    type Config = CausalConv1dPrefillKernelConfig;
    type Input = CausalConv1dPrefillKernelInput;

    const KIND: KernelKind = "causal_conv1d_prefill";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![
            Axis::values([128, 256, 512, 1024, 2048, 4096, 8192, 16_384]),
            Axis::values([1, 2, 4, 8, 16]),
            Axis::values([0, 49_152]),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache3DLinear
    }

    fn infeasible_mask(_config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_3d(|max_sequence_length, num_sequences, _prefix_len| {
            max_sequence_length.round() as u64 * num_sequences.round() as u64 > MAX_TOKENS
        })
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_3d(|max_sequence_length, num_sequences, prefix_len| {
            let max_sequence_length = max_sequence_length.round() as u64;
            let num_sequences = num_sequences.round() as u64;
            ArgsPayload::new()
                .with("backend", backend)
                .with(
                    "num_tokens",
                    u32::try_from(max_sequence_length * num_sequences)
                        .expect("prefill token sweep must fit u32"),
                )
                .with(
                    "max_sequence_length",
                    u32::try_from(max_sequence_length).expect("sequence sweep must fit u32"),
                )
                .with(
                    "num_sequences",
                    u32::try_from(num_sequences).expect("batch sweep must fit u32"),
                )
                .with("prefix_len", prefix_len.round() as u32)
                .with("channels", config.channels.get())
                .with("kernel_size", config.kernel_size.get())
                .with("dtype", config.dtype.as_str())
                .with("state_dtype", config.state_dtype.as_str())
        })
    }
}

register_kernel!(CausalConv1dPrefillKernel, CausalConv1dPrefillSpec);
