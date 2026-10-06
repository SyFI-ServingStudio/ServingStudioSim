//! Request-local to global sparse-index remapping.
//!
//! `selected_k` is the index-table width, e.g. 2048, or 2176 for a kpool table
//! (`round_up(index_topk + index_kpool - 1, 128)`, at most 2051 valid). vLLM's
//! `triton_convert_req_index_to_global_index` asserts
//! `NUM_TOPK_TOKENS % BLOCK_N == 0` with `BLOCK_N = 128`
//! (`v1/attention/backends/mla/sparse_utils.py`), so any positive multiple of
//! 128 is a legal width; the block size and block-table width are free as long
//! as the int32 slot ids fit.
//!
//! The native Triton launch scans a fixed `selected_k` row for every query.
//! Its variable physical work is captured by query rows, mean valid slots, and
//! the number of rows routed into the prefill workspace. Request boundaries and
//! all per-row vectors remain typed in `Input` and in the cost log even though
//! they are intentionally not independent cache dimensions; the log carries them
//! run-encoded (`timing::run_encoded`), since they are one value per query row.

use crate::timing::bridge::{de_backends, ArgsPayload, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

/// The Triton wrapper's `BLOCK_N`: `selected_k` must be a multiple of it.
const SELECTED_K_TILE: u32 = 128;
/// Remapped slot ids are int32, so one request's block table must fit in it.
const MAX_SLOT_ID: u64 = i32::MAX as u64;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaSparseIndexRemapWorkspacePartition {
    pub decode_requests: u32,
    pub chunk_sizes: Vec<u32>,
}

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaSparseIndexRemapKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub selected_k: u32,
    pub block_size: u32,
    pub max_blocks_per_request: u32,
    pub index_distribution: String,
    pub page_table_mapping: String,
    pub return_valid_counts: bool,
    pub index_dtype: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaSparseIndexRemapKernelInput {
    #[serde(with = "crate::timing::run_encoded")]
    pub request_row_counts: Vec<u32>,
    #[serde(with = "crate::timing::run_encoded")]
    pub local_span_lengths: Vec<u32>,
    #[serde(with = "crate::timing::run_encoded")]
    pub valid_counts: Vec<u32>,
    pub workspace_partition: Option<DsaSparseIndexRemapWorkspacePartition>,
}

impl DsaSparseIndexRemapKernelInput {
    fn work(&self) -> (u32, f64, u32) {
        assert!(!self.request_row_counts.is_empty());
        assert!(self.request_row_counts.iter().all(|&rows| rows > 0));

        let num_queries = self.request_row_counts.iter().copied().sum::<u32>();
        assert!(num_queries > 0);
        assert_eq!(self.local_span_lengths.len(), num_queries as usize);
        assert_eq!(self.valid_counts.len(), num_queries as usize);
        assert!(self
            .local_span_lengths
            .iter()
            .all(|&span| u64::from(span) <= MAX_SLOT_ID));
        assert!(self
            .valid_counts
            .iter()
            .zip(&self.local_span_lengths)
            .all(|(&count, &span)| count <= span));

        let workspace_rows = match &self.workspace_partition {
            None => 0,
            Some(partition) => {
                let decode_requests = partition.decode_requests as usize;
                assert!(decode_requests < self.request_row_counts.len());
                assert!(!partition.chunk_sizes.is_empty());
                assert!(partition.chunk_sizes.iter().all(|&size| size > 0));
                assert_eq!(
                    partition.chunk_sizes.iter().copied().sum::<u32>() as usize,
                    self.request_row_counts.len() - decode_requests
                );
                self.request_row_counts[decode_requests..]
                    .iter()
                    .copied()
                    .sum()
            }
        };
        let mean_valid = self.valid_counts.iter().map(|&v| u64::from(v)).sum::<u64>() as f64
            / f64::from(num_queries);
        (num_queries, mean_valid, workspace_rows)
    }
}

impl SweepCoords for DsaSparseIndexRemapKernelInput {
    fn coords(&self) -> Coords {
        let (num_queries, mean_valid, workspace_rows) = self.work();
        Coords::new([
            f64::from(num_queries),
            mean_valid,
            f64::from(workspace_rows),
        ])
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["num_queries", "mean_valid_count", "workspace_query_rows"]
    }
}

fn canonical_input(
    selected_k: u32,
    num_queries: u32,
    valid_count: u32,
    workspace_rows: u32,
) -> Option<DsaSparseIndexRemapKernelInput> {
    if num_queries == 0 || valid_count > selected_k || workspace_rows > num_queries {
        return None;
    }
    let (request_row_counts, workspace_partition) = match workspace_rows {
        0 => (vec![num_queries], None),
        rows if rows == num_queries => (
            vec![num_queries],
            Some(DsaSparseIndexRemapWorkspacePartition {
                decode_requests: 0,
                chunk_sizes: vec![1],
            }),
        ),
        rows => (
            vec![num_queries - rows, rows],
            Some(DsaSparseIndexRemapWorkspacePartition {
                decode_requests: 1,
                chunk_sizes: vec![1],
            }),
        ),
    };
    Some(DsaSparseIndexRemapKernelInput {
        request_row_counts,
        local_span_lengths: vec![valid_count; num_queries as usize],
        valid_counts: vec![valid_count; num_queries as usize],
        workspace_partition,
    })
}

fn encode_vector(values: &[u32], allow_clipped: bool, selected_k: u32) -> String {
    assert!(!values.is_empty());
    if values.iter().all(|&value| value == values[0]) {
        return format!("u:{}x{}", values[0], values.len());
    }
    if values
        .windows(2)
        .all(|pair| pair[1] == pair[0].saturating_add(1))
    {
        return format!("r:{}..{}", values[0], values[values.len() - 1]);
    }
    if allow_clipped {
        let unclipped_last = u64::from(values[0]) + values.len() as u64 - 1;
        let clipped = (0..values.len())
            .map(|index| (u64::from(values[0]) + index as u64).min(u64::from(selected_k)) as u32)
            .collect::<Vec<_>>();
        if values[0] < selected_k
            && u64::from(selected_k) < unclipped_last
            && values == clipped.as_slice()
        {
            return format!("c:{}..{}@{selected_k}", values[0], unclipped_last);
        }
    }

    let period = (1..=values.len())
        .find(|&candidate| {
            values.len() % candidate == 0
                && values
                    .iter()
                    .enumerate()
                    .all(|(index, &value)| value == values[index % candidate])
        })
        .expect("the full vector is always a valid repetition period");
    let group = values[..period]
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("g:({group})x{}", values.len() / period)
}

fn encode_workspace(input: &DsaSparseIndexRemapKernelInput) -> String {
    match &input.workspace_partition {
        None => "none".to_string(),
        Some(partition) => format!(
            "suffix:{}@{}",
            partition.decode_requests,
            partition
                .chunk_sizes
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

pub struct DsaSparseIndexRemapSpec;

impl KernelSpec for DsaSparseIndexRemapSpec {
    type Config = DsaSparseIndexRemapKernelConfig;
    type Input = DsaSparseIndexRemapKernelInput;

    const KIND: KernelKind = "dsa_sparse_index_remap";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        assert!(
            config.selected_k > 0 && config.selected_k % SELECTED_K_TILE == 0,
            "selected_k must be a positive multiple of {SELECTED_K_TILE}, got {}",
            config.selected_k
        );
        assert!(config.block_size > 0, "block_size must be positive");
        assert!(
            config.max_blocks_per_request > 0,
            "max_blocks_per_request must be positive"
        );
        assert!(
            u64::from(config.block_size) * u64::from(config.max_blocks_per_request) <= MAX_SLOT_ID,
            "one request's block table must fit int32 slot ids"
        );
        SweepGrid::new(vec![
            Axis::values([1, 8, 24, 64, 128, 512, 2048, 4096, 8192]),
            Axis::values([0, 1, 64, 512, 2048]),
            Axis::values([0, 1, 8, 24, 64, 128, 512, 2048, 4096, 8192]),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache3DLinear
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        grid.expand_3d(|num_queries, valid_count, workspace_rows| {
            canonical_input(
                config.selected_k,
                num_queries as u32,
                valid_count as u32,
                workspace_rows as u32,
            )
            .is_none()
        })
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_3d(|num_queries, valid_count, workspace_rows| {
            let input = canonical_input(
                config.selected_k,
                num_queries as u32,
                valid_count as u32,
                workspace_rows as u32,
            )
            .unwrap_or_else(|| canonical_input(config.selected_k, 1, 0, 0).unwrap());
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_queries", input.valid_counts.len() as u32)
                .with("num_requests", input.request_row_counts.len() as u32)
                .with("selected_k", config.selected_k)
                .with("block_size", config.block_size)
                .with("max_blocks_per_request", config.max_blocks_per_request)
                .with(
                    "request_row_counts",
                    encode_vector(&input.request_row_counts, false, config.selected_k),
                )
                .with(
                    "local_span_lengths",
                    encode_vector(&input.local_span_lengths, false, config.selected_k),
                )
                .with(
                    "valid_counts",
                    encode_vector(&input.valid_counts, true, config.selected_k),
                )
                .with("index_distribution", config.index_distribution.clone())
                .with("page_table_mapping", config.page_table_mapping.clone())
                .with("workspace_partition", encode_workspace(&input))
                .with("return_valid_counts", config.return_valid_counts)
                .with("index_dtype", config.index_dtype.clone())
        })
    }
}

register_kernel!(DsaSparseIndexRemapKernel, DsaSparseIndexRemapSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn config() -> DsaSparseIndexRemapKernelConfig {
        DsaSparseIndexRemapKernelConfig {
            backends: vec!["vllm_triton"],
            gpu_name: "NVIDIA B200".to_string(),
            selected_k: 2048,
            block_size: 64,
            max_blocks_per_request: 16_384,
            index_distribution: "recent_contiguous".to_string(),
            page_table_mapping: "request_contiguous".to_string(),
            return_valid_counts: true,
            index_dtype: "int32".to_string(),
        }
    }

    #[test]
    fn ragged_input_preserves_workspace_topology() {
        let input = DsaSparseIndexRemapKernelInput {
            request_row_counts: vec![8, 16],
            local_span_lengths: vec![128; 8].into_iter().chain(vec![256; 16]).collect(),
            valid_counts: vec![128; 8].into_iter().chain(vec![256; 16]).collect(),
            workspace_partition: Some(DsaSparseIndexRemapWorkspacePartition {
                decode_requests: 1,
                chunk_sizes: vec![1],
            }),
        };
        assert_eq!(input.coords()[0], 24.0);
        assert!((input.coords()[1] - 213.333_333_333_333_34).abs() < 1e-12);
        assert_eq!(input.coords()[2], 16.0);
    }

    #[test]
    fn long_context_rows_serialize_as_runs_and_round_trip() {
        // One decode row, then two causal prefill chunks deep into a 1M context.
        let pairs = [(4096_u32, 323_318_u32), (4096, 1_000_000)];
        let mut input = DsaSparseIndexRemapKernelInput {
            request_row_counts: vec![1],
            local_span_lengths: vec![70_001],
            valid_counts: vec![2048],
            workspace_partition: None,
        };
        for (rows, context) in pairs {
            input.request_row_counts.push(rows);
            for span in context - rows + 1..=context {
                input.local_span_lengths.push(span);
                input.valid_counts.push(span.min(2048));
            }
        }
        let text = serde_json::to_string(&input).unwrap();
        assert!(text.len() < 200, "{text}");
        let back: DsaSparseIndexRemapKernelInput = serde_json::from_str(&text).unwrap();
        assert_eq!(back.request_row_counts, input.request_row_counts);
        assert_eq!(back.local_span_lengths, input.local_span_lengths);
        assert_eq!(back.valid_counts, input.valid_counts);
        assert_eq!(back.coords()[..], input.coords()[..]);
    }

    #[test]
    fn coords_admit_any_request_count_and_rows_past_the_grid() {
        let rows = 16_384 + 64;
        let input = DsaSparseIndexRemapKernelInput {
            request_row_counts: vec![1; rows as usize],
            local_span_lengths: vec![4096; rows as usize],
            valid_counts: vec![2048; rows as usize],
            workspace_partition: None,
        };
        assert_eq!(input.coords()[0], f64::from(rows));
        assert_eq!(input.coords()[1], 2048.0);
    }

    #[test]
    fn canonical_payload_matches_python_schema() {
        let config = config();
        let grid = SweepGrid::new(vec![
            Axis::values([24]),
            Axis::values([512]),
            Axis::values([8]),
        ]);
        let payload = &DsaSparseIndexRemapSpec::enumerate(&config, &grid, "vllm_triton")[0];
        let fields = payload.fields();

        assert_eq!(fields.len(), 14);
        assert_eq!(fields.get("num_queries"), Some(&Value::from(24_u32)));
        assert_eq!(fields.get("num_requests"), Some(&Value::from(2_u32)));
        assert_eq!(
            fields.get("max_blocks_per_request"),
            Some(&Value::from(16_384_u32))
        );
        assert_eq!(
            fields.get("request_row_counts"),
            Some(&Value::from("g:(16,8)x1"))
        );
        assert_eq!(
            fields.get("local_span_lengths"),
            Some(&Value::from("u:512x24"))
        );
        assert_eq!(
            fields.get("workspace_partition"),
            Some(&Value::from("suffix:1@1"))
        );
        assert_eq!(fields.get("return_valid_counts"), Some(&Value::from(true)));
    }

    #[test]
    fn impossible_workspace_shape_is_masked() {
        let config = config();
        let grid = SweepGrid::new(vec![
            Axis::values([8]),
            Axis::values([64]),
            Axis::values([24]),
        ]);
        assert_eq!(
            DsaSparseIndexRemapSpec::infeasible_mask(&config, &grid),
            vec![true]
        );
    }

    #[test]
    fn compact_vector_encoding_matches_python_canonical_forms() {
        assert_eq!(encode_vector(&[7, 7, 7], false, 2048), "u:7x3");
        assert_eq!(encode_vector(&[7, 8, 9], false, 2048), "r:7..9");
        assert_eq!(
            encode_vector(&[2047, 2048, 2048], true, 2048),
            "c:2047..2049@2048"
        );
        assert_eq!(encode_vector(&[2, 4, 2, 4], false, 2048), "g:(2,4)x2");
    }

    #[test]
    fn any_128_multiple_width_and_block_size_build_a_grid() {
        for (selected_k, block_size) in [(128, 16), (512, 64), (4096, 128)] {
            let mut config = config();
            config.selected_k = selected_k;
            config.block_size = block_size;
            config.max_blocks_per_request = 1024;
            let grid = DsaSparseIndexRemapSpec::sweep_grid(&config);
            let payloads = DsaSparseIndexRemapSpec::enumerate(&config, &grid, "vllm_triton");
            assert!(payloads.iter().all(|payload| {
                payload.fields().get("selected_k") == Some(&Value::from(selected_k))
                    && payload.fields().get("block_size") == Some(&Value::from(block_size))
            }));
            // Valid counts above the table width are masked, not enumerated.
            let mask = DsaSparseIndexRemapSpec::infeasible_mask(&config, &grid);
            assert_eq!(mask.len(), payloads.len());
            assert!(mask.iter().any(|&masked| !masked));
        }
    }

    #[test]
    #[should_panic(expected = "positive multiple of 128")]
    fn width_off_the_128_tile_is_rejected() {
        let mut config = config();
        config.selected_k = 2000;
        DsaSparseIndexRemapSpec::sweep_grid(&config);
    }

    #[test]
    fn kpool_table_width_enumerates_2176_and_admits_tail_counts() {
        let mut config = config();
        config.backends = vec!["vllm_triton"];
        config.selected_k = 2176;
        let grid = DsaSparseIndexRemapSpec::sweep_grid(&config);
        let payloads = DsaSparseIndexRemapSpec::enumerate(&config, &grid, "vllm_triton");
        assert!(payloads
            .iter()
            .all(|payload| payload.fields().get("selected_k") == Some(&Value::from(2176_u32))));
        // kpool rows hold up to index_topk + index_kpool - 1 = 2051 valid slots.
        let input = DsaSparseIndexRemapKernelInput {
            request_row_counts: vec![4],
            local_span_lengths: vec![4096; 4],
            valid_counts: vec![2048, 2049, 2050, 2051],
            workspace_partition: None,
        };
        assert_eq!(input.coords()[1], 2049.5);
    }
}
