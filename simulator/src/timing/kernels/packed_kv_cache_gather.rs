//! Packed KV cache dequantize-and-gather operation.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec, OffGrid};
use crate::timing::kernels::kv_compress_store::FP8_DS_MLA_ROW_BYTES;
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackedKvCacheGatherMode {
    Full,
    Suffix,
}

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PackedKvCacheGatherKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub mode: PackedKvCacheGatherMode,
    pub suffix_sequence_len: u32,
    pub max_gather_rows: u32,
    pub workspace_rows: u32,
    pub block_table_width: u32,
    pub block_size: u32,
    pub offset: u32,
    pub num_kv_heads: u32,
    pub head_dim: Dim,
    pub fp8_dim: u32,
    pub quant_group_size: u32,
    #[kv_dtype]
    pub kv_dtype: DType,
    pub cache_dtype: String,
    #[compute_dtype]
    pub output_dtype: DType,
    pub cache_layout: String,
    pub scale_format: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PackedKvCacheGatherKernelInput {
    pub seq_lens: Vec<u32>,
    pub gather_lens: Vec<u32>,
}

impl PackedKvCacheGatherKernelInput {
    fn work(&self) -> (u32, f64) {
        assert!(!self.seq_lens.is_empty());
        assert!(self.gather_lens.is_empty() || self.gather_lens.len() == self.seq_lens.len());
        let selected = if self.gather_lens.is_empty() {
            &self.seq_lens
        } else {
            assert!(self
                .gather_lens
                .iter()
                .zip(&self.seq_lens)
                .all(|(&gather, &sequence)| gather <= sequence));
            &self.gather_lens
        };
        let total = selected.iter().map(|&rows| u64::from(rows)).sum::<u64>();
        let requests = self.seq_lens.len() as u32;
        (requests, total as f64 / f64::from(requests))
    }
}

impl SweepCoords for PackedKvCacheGatherKernelInput {
    fn coords(&self) -> Coords {
        let (requests, mean_rows) = self.work();
        Coords::new([f64::from(requests), mean_rows])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["num_requests", "mean_selected_rows"]
    }
}

fn selected_axis(max_gather_rows: u32) -> Vec<f64> {
    Axis::values([
        1, 2, 4, 8, 16, 32, 64, 96, 128, 129, 160, 192, 256, 257, 384, 512, 768, 1024, 1536, 2048,
        3072, 4096, 8192, 16384,
    ])
    .into_iter()
    .filter(|&rows| rows <= f64::from(max_gather_rows))
    .collect()
}

fn canonical_input(
    config: &PackedKvCacheGatherKernelConfig,
    requests: u32,
    selected_rows: u32,
) -> PackedKvCacheGatherKernelInput {
    match config.mode {
        PackedKvCacheGatherMode::Full => PackedKvCacheGatherKernelInput {
            seq_lens: vec![selected_rows; requests as usize],
            gather_lens: Vec::new(),
        },
        PackedKvCacheGatherMode::Suffix => PackedKvCacheGatherKernelInput {
            seq_lens: vec![config.suffix_sequence_len; requests as usize],
            gather_lens: vec![selected_rows; requests as usize],
        },
    }
}

/// The profiler's logical bytes: each gathered row read with its slot and
/// written dequantized, plus the length arrays.
fn logical_bytes(
    config: &PackedKvCacheGatherKernelConfig,
    input: &PackedKvCacheGatherKernelInput,
) -> f64 {
    let lengths = if input.gather_lens.is_empty() {
        &input.seq_lens
    } else {
        &input.gather_lens
    };
    let gathered: f64 = lengths.iter().map(|&rows| f64::from(rows)).sum();
    let length_arrays = if input.gather_lens.is_empty() {
        1.0
    } else {
        2.0
    };
    gathered
        * (f64::from(FP8_DS_MLA_ROW_BYTES)
            + 4.0
            + f64::from(config.head_dim.get()) * f64::from(config.output_dtype.size_bytes()))
        + input.seq_lens.len() as f64 * 4.0 * length_arrays
}

pub struct PackedKvCacheGatherSpec;

impl KernelSpec for PackedKvCacheGatherSpec {
    type Config = PackedKvCacheGatherKernelConfig;
    type Input = PackedKvCacheGatherKernelInput;

    const KIND: KernelKind = "packed_kv_cache_gather";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        assert!(matches!(config.block_size, 2 | 64));
        assert!(config.max_gather_rows + config.offset <= config.workspace_rows);
        let covered_rows = config.block_table_width * config.block_size;
        match config.mode {
            PackedKvCacheGatherMode::Full => {
                assert!(config.max_gather_rows <= covered_rows);
            }
            PackedKvCacheGatherMode::Suffix => {
                assert!(config.max_gather_rows <= config.suffix_sequence_len);
                assert!(config.suffix_sequence_len <= covered_rows);
            }
        }
        SweepGrid::new(vec![
            Axis::values([1, 2, 3, 4]),
            selected_axis(config.max_gather_rows),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear(Extrapolation::Product)
    }

    /// Past the grid the copy holds its bandwidth. Unmeasured: the kernel only
    /// moves rows.
    fn off_grid(
        config: &Self::Config,
        input: &Self::Input,
        _backend: &'static str,
    ) -> OffGrid<Self::Input> {
        OffGrid::Bandwidth(logical_bytes(config, input))
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|requests, selected_rows| {
            let input = canonical_input(config, requests as u32, selected_rows as u32);
            ArgsPayload::new()
                .with("backend", backend)
                .with("seq_lens", serde_json::json!(input.seq_lens))
                .with("gather_lens", serde_json::json!(input.gather_lens))
                .with("workspace_rows", config.workspace_rows)
                .with("block_table_width", config.block_table_width)
                .with("block_size", config.block_size)
                .with("offset", config.offset)
                .with("num_kv_heads", config.num_kv_heads)
                .with("head_dim", config.head_dim.get())
                .with("fp8_dim", config.fp8_dim)
                .with("quant_group_size", config.quant_group_size)
                .with("cache_dtype", config.cache_dtype.clone())
                .with("output_dtype", config.output_dtype.as_str())
                .with("cache_layout", config.cache_layout.clone())
                .with("scale_format", config.scale_format.clone())
        })
    }
}

register_kernel!(PackedKvCacheGatherKernel, PackedKvCacheGatherSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::SweepCoords;

    #[test]
    fn ragged_suffix_gather_projects_selected_rows_not_full_sequences() {
        let input = PackedKvCacheGatherKernelInput {
            seq_lens: vec![65_536, 32_768],
            gather_lens: vec![128, 64],
        };
        assert_eq!(&*input.coords(), &[2.0, 96.0]);
    }

    #[test]
    fn logical_bytes_match_a_measured_row() {
        // A profiled H200 row's logged bandwidth x time.
        let config: PackedKvCacheGatherKernelConfig =
            serde_json::from_value(serde_json::json!({"gpu_name": "NVIDIA H200", "workspace_rows": 256, "block_table_width": 3, "block_size": 64, "offset": 128, "num_kv_heads": 1, "head_dim": 512, "fp8_dim": 448, "quant_group_size": 64, "cache_dtype": "fp8_ds_mla", "output_dtype": "bf16", "cache_layout": "block_segregated_data_then_scales", "scale_format": "ue8m0", "backends": ["vllm_cutedsl"], "mode": "full", "suffix_sequence_len": 0, "max_gather_rows": 1, "kv_dtype": "fp8_e4m3"}))
            .unwrap();
        let input = PackedKvCacheGatherKernelInput {
            seq_lens: vec![5, 3],
            gather_lens: Vec::new(),
        };
        assert_eq!(logical_bytes(&config, &input), 12_904.0);
    }
}
