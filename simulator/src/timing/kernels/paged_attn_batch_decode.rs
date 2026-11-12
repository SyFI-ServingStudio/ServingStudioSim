//! Paged batch decode attention: FlashInfer's `trtllm_batch_decode_with_kv_cache`
//! (trtllm-gen) as vLLM's `FlashInferImpl.forward` calls it for the decode
//! requests of a step (vllm/v1/attention/backends/flashinfer.py). The step's
//! prefill requests are a separate call, `paged_attn_batch_prefill`.
//!
//! The kernel gives each (request, KV-head tile) one CTA column and splits the
//! request's KV range over up to `ceil(max_seq_len / 512)` CTAs, bounded by
//! `SMs / (ctas_y * batch)` (FlashInfer `fmhaKernels.cuh`, with `ctas_y =
//! q_heads / min(q_heads_per_kv, 8)`). The splits of the longest request run
//! in parallel only while the batch leaves SMs free, so its time is set by the
//! longest context while the batch is small and by the batch's total context
//! once it is large; batches that share `(batch_size, max_kv_len,
//! total_kv_len)` measured within about 10%.
//!
//! The cache axes are `(batch_size, max_kv_len, fill)`, with `fill` the share
//! of the other requests' room that their context fills past the one token a
//! decode reads at least: `(total - max - (batch - 1)) / ((batch - 1) * (max -
//! 1))`. The rows are canonical batches (one request at `max_kv_len`, the rest
//! sharing the remainder evenly), keyed by `(batch_size, max_kv_len,
//! total_kv_len)`. The batch axis holds every batch
//! where the split count steps and the edge of one CTA wave, so a step between
//! knots is never smeared.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::fa4_varlen_projection::B200_SMS;
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

/// Query heads one CTA covers at most (trtllm-gen `mStepQ`).
const MAX_HEADS_PER_CTA: u32 = 8;
/// Batch knots past the split edges: the largest decode batches a step sees.
const BATCH_TAIL: [u32; 9] = [16, 24, 32, 48, 64, 96, 128, 192, 256];
/// Longest-context knots; the split is linear in the context between them.
const CONTEXT_KNOTS: [u32; 8] = [1, 256, 1024, 4096, 16384, 32768, 65536, 131072];
const FILL_KNOTS: [f64; 4] = [0.0, 0.25, 0.5, 1.0];
/// Profiling cap on one batch's KV bytes (the runner's `_MAX_KV_BYTES`).
const MAX_KV_BYTES: f64 = (32u64 << 30) as f64;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PagedAttnBatchDecodeKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    #[compute_dtype]
    pub q_dtype: DType,
    #[kv_dtype]
    pub kv_dtype: DType,
    pub o_dtype: DType,
}

impl PagedAttnBatchDecodeKernelConfig {
    /// CTA columns per request: query heads over the heads one CTA covers.
    fn ctas_per_request(&self) -> u32 {
        let group = self.num_qo_heads.get() / self.num_kv_heads.get();
        self.num_qo_heads.get() / group.min(MAX_HEADS_PER_CTA)
    }

    fn kv_bytes_per_token(&self) -> f64 {
        2.0 * f64::from(self.num_kv_heads.get() * self.head_dim.get())
            * f64::from(self.kv_dtype.size_bytes())
    }

    /// Batch knots: both sides of every step in the split count (each batch
    /// while it drops at every request) and of the one-wave edge, then the tail.
    fn batch_axis(&self) -> Vec<f64> {
        let wave = (B200_SMS as u32 / self.ctas_per_request()).max(1);
        let splits = |batch: u32| (wave / batch).max(1);
        let mut knots: Vec<u32> = (1..=wave + 1)
            .filter(|&batch| {
                batch == 1
                    || splits(batch) != splits(batch - 1)
                    || (batch < wave && splits(batch) != splits(batch + 1))
            })
            .collect();
        knots.extend([wave, wave + 1]);
        knots.extend(BATCH_TAIL);
        knots.sort_unstable();
        knots.dedup();
        Axis::values(knots)
    }
}

/// The decode requests of one step.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PagedAttnBatchDecodeKernelInput {
    /// Rows in the call; a FULL CUDA-graph replay counts its zero-length
    /// padding rows.
    pub batch_size: u32,
    /// Context of the longest request, its query token included.
    pub max_kv_len: u32,
    /// Context tokens summed over the batch.
    pub total_kv_len: u64,
}

impl PagedAttnBatchDecodeKernelInput {
    /// Share of the other requests' room their context fills past the one
    /// token each reads at least (up to `max_kv_len` each). Padding rows of a
    /// graph replay hold nothing and clamp to 0.
    fn fill(&self) -> f64 {
        if self.batch_size <= 1 || self.max_kv_len <= 1 {
            return 0.0;
        }
        let others = u64::from(self.batch_size - 1);
        let rest = self
            .total_kv_len
            .saturating_sub(u64::from(self.max_kv_len) + others) as f64;
        let room = others as f64 * f64::from(self.max_kv_len - 1);
        (rest / room).clamp(0.0, 1.0)
    }
}

impl SweepCoords for PagedAttnBatchDecodeKernelInput {
    fn coords(&self) -> Coords {
        Coords::new([
            f64::from(self.batch_size),
            f64::from(self.max_kv_len),
            self.fill(),
        ])
    }
    fn coord_field_names() -> &'static [&'static str] {
        &["batch_size", "max_kv_len", "fill"]
    }
}

/// Total context of the canonical batch at a grid cell.
fn total_kv_len(batch: f64, max_kv_len: f64, fill: f64) -> u64 {
    let others = batch - 1.0;
    (max_kv_len + others + (fill * others * (max_kv_len - 1.0)).round()) as u64
}

pub struct PagedAttnBatchDecodeSpec;

impl KernelSpec for PagedAttnBatchDecodeSpec {
    type Config = PagedAttnBatchDecodeKernelConfig;
    type Input = PagedAttnBatchDecodeKernelInput;

    const KIND: KernelKind = "paged_attn_batch_decode";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        assert!(
            config.num_qo_heads.get() % config.num_kv_heads.get() == 0,
            "num_qo_heads must be a multiple of num_kv_heads"
        );
        SweepGrid::new(vec![
            config.batch_axis(),
            Axis::values(CONTEXT_KNOTS),
            FILL_KNOTS.to_vec(),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache3DLinear
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        let per_token = config.kv_bytes_per_token();
        grid.expand_3d(|batch, max_kv_len, fill| {
            total_kv_len(batch, max_kv_len, fill) as f64 * per_token > MAX_KV_BYTES
        })
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        // A single request has no room to fill: its cells share one row.
        grid.expand_3d(|batch, max_kv_len, fill| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_qo_heads", config.num_qo_heads.get())
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("q_dtype", config.q_dtype.as_str())
                .with("kv_dtype", config.kv_dtype.as_str())
                .with("o_dtype", config.o_dtype.as_str())
                .with("batch_size", batch as u32)
                .with("max_kv_len", max_kv_len as u32)
                .with("total_kv_len", total_kv_len(batch, max_kv_len, fill))
        })
    }
}

register_kernel!(PagedAttnBatchDecodeKernel, PagedAttnBatchDecodeSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::kernels::engine::KernelConfig;
    use serde_json::Value;

    /// 24 query / 4 KV heads of dim 256, FP8 query over an FP8 cache.
    fn config() -> PagedAttnBatchDecodeKernelConfig {
        PagedAttnBatchDecodeKernelConfig {
            backends: vec!["flashinfer_trtllm_gen"],
            gpu_name: "NVIDIA B200".to_string(),
            num_qo_heads: 24.into(),
            num_kv_heads: 4.into(),
            head_dim: 256.into(),
            q_dtype: DType::Fp8E4m3,
            kv_dtype: DType::Fp8E4m3,
            o_dtype: DType::Bf16,
        }
    }

    #[test]
    fn describe_config_lists_geometry_and_dtypes() {
        assert_eq!(
            config().describe_config(),
            serde_json::json!({
                "backends": ["flashinfer_trtllm_gen"], "gpu_name": "NVIDIA B200",
                "num_qo_heads": {"value": 24, "expression": null, "bindings": {}},
                "num_kv_heads": {"value": 4, "expression": null, "bindings": {}},
                "head_dim": {"value": 256, "expression": null, "bindings": {}},
                "q_dtype": "fp8_e4m3", "kv_dtype": "fp8_e4m3", "o_dtype": "bf16",
            })
        );
    }

    #[test]
    fn batch_axis_brackets_every_measured_split_step() {
        // Defect caught: a knot set that misses a split step. Measured on B200
        // with 24/4 heads, a one-long batch's time steps at 10, 13 and 19
        // requests and a uniform batch's at the 37 -> 38 wave edge; each step
        // needs knots on both sides.
        let axis = config().batch_axis();
        for batch in [
            1.0, 2.0, 8.0, 9.0, 10.0, 12.0, 13.0, 18.0, 19.0, 37.0, 38.0, 256.0,
        ] {
            assert!(axis.contains(&batch), "batch axis {axis:?} misses {batch}");
        }
        assert!(axis.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn batch_axis_follows_the_head_geometry() {
        // Defect caught: knots fixed to one model. With 8 query heads per KV
        // head and 8 KV heads a request takes 8 CTA columns, so one wave holds
        // 148 / 8 = 18 requests.
        let mut wide = config();
        wide.num_qo_heads = 64.into();
        wide.num_kv_heads = 8.into();
        let axis = wide.batch_axis();
        assert!(axis.contains(&18.0) && axis.contains(&19.0));
        assert!(!axis.contains(&37.0));
    }

    #[test]
    fn coords_project_to_batch_longest_and_fill() {
        // Defect caught: fill measured against the whole batch instead of the
        // other requests' room, or a lone request reading as anything but 0.
        // Four other requests, one token each plus half of 999 more.
        let input = PagedAttnBatchDecodeKernelInput {
            batch_size: 5,
            max_kv_len: 1000,
            total_kv_len: 1000 + 4 + 1998,
        };
        assert_eq!(&*input.coords(), &[5.0, 1000.0, 0.5]);
        let lone = PagedAttnBatchDecodeKernelInput {
            batch_size: 1,
            max_kv_len: 1000,
            total_kv_len: 1000,
        };
        assert_eq!(&*lone.coords(), &[1.0, 1000.0, 0.0]);
        // Zero-length graph padding clamps to the one-token batch.
        let padded = PagedAttnBatchDecodeKernelInput {
            batch_size: 8,
            max_kv_len: 1000,
            total_kv_len: 1003,
        };
        assert_eq!(padded.coords()[2], 0.0);
    }

    #[test]
    fn enumerate_round_trips_the_cell_through_the_input() {
        // Defect caught: a row whose (batch, max, total) projects back to a
        // different cell than the one it fills.
        let cfg = config();
        let grid = PagedAttnBatchDecodeSpec::sweep_grid(&cfg);
        let payloads = PagedAttnBatchDecodeSpec::enumerate(&cfg, &grid, "flashinfer_trtllm_gen");
        let cells = grid.expand_3d(|b, l, f| (b, l, f));
        assert_eq!(payloads.len(), cells.len());
        for (payload, (batch, max_kv_len, fill)) in payloads.iter().zip(cells) {
            let fields = payload.fields();
            let input = PagedAttnBatchDecodeKernelInput {
                batch_size: fields["batch_size"].as_u64().unwrap() as u32,
                max_kv_len: fields["max_kv_len"].as_u64().unwrap() as u32,
                total_kv_len: fields["total_kv_len"].as_u64().unwrap(),
            };
            let coords = input.coords();
            assert_eq!((coords[0], coords[1]), (batch, max_kv_len));
            if batch > 1.0 && max_kv_len > 1.0 {
                // Off by at most the half token the total rounds.
                let token = 1.0 / ((batch - 1.0) * (max_kv_len - 1.0));
                assert!(
                    (coords[2] - fill).abs() <= token / 2.0 + 1e-12,
                    "{batch} {max_kv_len} {fill}"
                );
            }
        }
        let fields = payloads[0].fields();
        assert_eq!(fields.len(), 10);
        assert_eq!(fields.get("q_dtype"), Some(&Value::from("fp8_e4m3")));
        assert_eq!(fields.get("o_dtype"), Some(&Value::from("bf16")));
    }

    #[test]
    fn mask_drops_only_batches_past_the_kv_cap() {
        let cfg = config();
        let grid = PagedAttnBatchDecodeSpec::sweep_grid(&cfg);
        let mask = PagedAttnBatchDecodeSpec::infeasible_mask(&cfg, &grid);
        let cells = grid.expand_3d(|b, l, f| (b, l, f));
        for (&masked, (batch, max_kv_len, fill)) in mask.iter().zip(cells) {
            let bytes = total_kv_len(batch, max_kv_len, fill) as f64 * 2048.0;
            assert_eq!(masked, bytes > MAX_KV_BYTES);
        }
        assert!(mask.iter().any(|&m| m) && mask.iter().any(|&m| !m));
    }
}
