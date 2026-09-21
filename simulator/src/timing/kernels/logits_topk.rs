//! Row-wise top-k selection over a dense score matrix.
//!
//! One call of vLLM's `logits_processor._topk`, whose production backend is
//! `flashinfer.top_k(scores, k, sorted=True, deterministic=True)`. The measured
//! cost is the whole fixed launch sequence that call issues — the selection
//! kernel plus the on-device stable value sort — because `sorted=True` is part
//! of the call, not an option the caller varies.
//!
//! Matrix width, top-k width, and score dtype identify the call site and so
//! live in the Config; only the row count moves with the batch, so the cache is
//! a single interpolated curve over `num_rows`. That split matters here: the
//! two call sites in one vocab-parallel reduction differ only by width (a
//! vocabulary shard, then the gathered candidate block), and pricing them off
//! one shared curve would charge the narrow one a wide row's sweep.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct LogitsTopkKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Columns of one score row: the width the selection sweeps.
    pub num_columns: Dim,
    pub top_k: u32,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct LogitsTopkKernelInput {
    /// Rows selected over — one per scored position.
    pub num_rows: u32,
}

pub struct LogitsTopkSpec;

impl KernelSpec for LogitsTopkSpec {
    type Config = LogitsTopkKernelConfig;
    type Input = LogitsTopkKernelInput;

    const KIND: KernelKind = "logits_topk";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // Powers through 4096 with the halfway points that matter in between.
        // A speculative head scores `requests * draft_tokens` rows, which a
        // 2048-token batch bounds well below 4096; the landmarks stay dense
        // where the curve is still leaving its launch-bound floor.
        SweepGrid::new(vec![Axis::values([
            1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096,
        ])])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|num_rows| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_rows", num_rows as u32)
                .with("num_columns", config.num_columns.get())
                .with("top_k", config.top_k)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(LogitsTopkKernel, LogitsTopkSpec);

#[cfg(test)]
mod tests {
    use super::{LogitsTopkKernelConfig, LogitsTopkKernelInput, LogitsTopkSpec};
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config() -> LogitsTopkKernelConfig {
        LogitsTopkKernelConfig {
            backends: vec!["flashinfer"],
            gpu_name: "NVIDIA B200".to_string(),
            num_columns: 38720.into(),
            top_k: 16,
            dtype: DType::Bf16,
        }
    }

    /// Two call sites of one vocab-parallel reduction differ only by width. If
    /// `num_columns` ever stopped taking part in config identity they would
    /// share a cache, and the gathered-candidate select would be priced as a
    /// vocabulary sweep — the exact defect this kernel replaced.
    #[test]
    fn matrix_width_top_k_and_dtype_are_part_of_the_cache_identity() {
        let cfg = config();
        assert_eq!(LogitsTopkSpec::KIND, "logits_topk");
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), None);

        let mut narrow = config();
        narrow.num_columns = 64.into();
        assert_ne!(narrow, cfg);
        let mut wider_k = config();
        wider_k.top_k = 32;
        assert_ne!(wider_k, cfg);
        let mut fp32 = config();
        fp32.dtype = DType::Fp32;
        assert_ne!(fp32, cfg);

        let encoded = serde_json::to_value(&cfg).unwrap();
        let decoded: LogitsTopkKernelConfig = serde_json::from_value(encoded.clone()).unwrap();
        assert_eq!(decoded, cfg);
        assert_eq!(serde_json::to_value(decoded).unwrap(), encoded);
    }

    /// The row count is the only runtime axis, and it must reach the cache and
    /// the cost log under that name. A rename here silently reads the curve at
    /// coordinate zero.
    #[test]
    fn the_row_count_is_the_only_sweep_coordinate() {
        let input = LogitsTopkKernelInput { num_rows: 448 };
        assert_eq!(&*input.coords(), &[448.0]);
        assert_eq!(LogitsTopkKernelInput::coord_field_names(), &["num_rows"]);

        let slot: SlotInput = input.into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_rows": 448})
        );
    }

    /// The payload must carry exactly the Python `LogitsTopkArgs` fields. An
    /// extra or missing name makes every profiled row unreachable at query
    /// time, which shows up as a silent cache miss rather than an error.
    #[test]
    fn payload_carries_exactly_the_python_args_and_the_swept_row_count() {
        let cfg = config();
        let grid = LogitsTopkSpec::sweep_grid(&cfg);
        assert!(LogitsTopkSpec::infeasible_mask(&cfg, &grid).is_empty());

        for backend in ["torch", "flashinfer"] {
            assert_eq!(
                LogitsTopkSpec::cache_kind(backend),
                CacheKind::Cache1DLinear
            );
            let payloads = LogitsTopkSpec::enumerate(&cfg, &grid, backend);
            assert_eq!(payloads.len(), 17);
            for payload in &payloads {
                let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
                assert_eq!(
                    names,
                    ["backend", "dtype", "num_columns", "num_rows", "top_k"]
                );
                assert_eq!(payload.backend(), Some(backend));
                assert_eq!(payload.fields()["num_columns"], Value::from(38720_u32));
                assert_eq!(payload.fields()["top_k"], Value::from(16_u32));
                assert_eq!(payload.fields()["dtype"], Value::from("bf16"));
            }
        }

        let rows: Vec<u64> = LogitsTopkSpec::enumerate(&cfg, &grid, "flashinfer")
            .iter()
            .map(|payload| payload.fields()["num_rows"].as_u64().unwrap())
            .collect();
        assert_eq!(rows.first(), Some(&1));
        assert_eq!(rows.last(), Some(&4096));
    }

    /// Every profiled point costs GPU time on a shared cluster, and the grid is
    /// profiled once per distinct config. Keeping it well under the 500-cell
    /// ceiling is what lets a second call-site width be added for free.
    #[test]
    fn the_grid_stays_a_single_short_row_axis() {
        let grid = LogitsTopkSpec::sweep_grid(&config());
        assert_eq!(grid.axes().len(), 1);
        assert_eq!(grid.axes()[0].len(), 17);
        assert!(grid.axes()[0].windows(2).all(|pair| pair[0] < pair[1]));
    }
}
