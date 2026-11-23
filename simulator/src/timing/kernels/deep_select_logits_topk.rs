//! DeepSelect's top-k with values over fp32 sampler logits
//! (`deep_select.interface.topk(logits, top_k, sorted, indices_type=int32)`,
//! one `topk_select_fp32::topk_kernel` launch, one CTA per row).
//!
//! DeepInfer calls it in two places: the sampler's force_topk restriction over
//! full-vocabulary logits (verify rows and every DFlash Markov step,
//! `sorted = false`), and the requested top logprobs over the force_topk-wide
//! logprobs (`sorted = true`). Row width, `top_k` and `sorted` identify the
//! call site and live in the Config; only the row count moves with the batch,
//! so the cache is one interpolated curve over `num_rows`.
//!
//! The row axis brackets DeepSelect's waves: one CTA per row and one CTA per
//! SM at this kernel's shared memory (217 KB), so the time steps at every
//! multiple of the SM count (148 on SM100). The first waves are bracketed on
//! both sides; past them a step is a small fraction of the total.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

const ROW_POINTS: [u32; 30] = [
    1, 2, 4, 8, 16, 32, 64, 96, 128, 148, 149, 192, 256, 296, 297, 384, 444, 445, 512, 592, 593,
    768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 8192,
];

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DeepSelectLogitsTopkKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Scores in one row: the vocabulary, or the force_topk width.
    pub num_columns: Dim,
    pub top_k: u32,
    /// Whether the call asks for the kept values in descending order.
    pub sorted: bool,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct DeepSelectLogitsTopkKernelInput {
    /// Logit rows selected from, one per sampled position.
    pub num_rows: u32,
}

pub struct DeepSelectLogitsTopkSpec;

impl KernelSpec for DeepSelectLogitsTopkSpec {
    type Config = DeepSelectLogitsTopkKernelConfig;
    type Input = DeepSelectLogitsTopkKernelInput;

    const KIND: KernelKind = "deep_select_logits_topk";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::values(ROW_POINTS)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        anyhow::ensure!(
            config.top_k >= 1 && config.top_k <= config.num_columns.get(),
            "top_k must be in 1..=num_columns"
        );
        Ok(())
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
                .with("sorted", config.sorted)
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(DeepSelectLogitsTopkKernel, DeepSelectLogitsTopkSpec);

#[cfg(test)]
mod tests {
    use super::{
        DeepSelectLogitsTopkKernelConfig, DeepSelectLogitsTopkKernelInput, DeepSelectLogitsTopkSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn config() -> DeepSelectLogitsTopkKernelConfig {
        DeepSelectLogitsTopkKernelConfig {
            backends: vec!["deep_select"],
            gpu_name: "NVIDIA B200".to_string(),
            num_columns: 129280.into(),
            top_k: 1024,
            sorted: false,
            dtype: DType::Fp32,
        }
    }

    /// The force_topk call (vocabulary wide, unsorted) and the top-logprobs
    /// call (force_topk wide, sorted) must never share a curve.
    #[test]
    fn width_top_k_and_sortedness_are_part_of_the_cache_identity() {
        let cfg = config();
        assert_eq!(cfg.compute_dtype(), Some(DType::Fp32));
        let mut sorted = config();
        sorted.sorted = true;
        assert_ne!(sorted, cfg);
        let mut narrow = config();
        narrow.num_columns = 1024.into();
        assert_ne!(narrow, cfg);
        let mut k = config();
        k.top_k = 5;
        assert_ne!(k, cfg);
        let decoded: DeepSelectLogitsTopkKernelConfig =
            serde_json::from_value(serde_json::to_value(&cfg).unwrap()).unwrap();
        assert_eq!(decoded, cfg);
    }

    /// The payload must carry exactly the Python `DeepSelectLogitsTopkArgs`
    /// fields, or every profiled row is a silent cache miss.
    #[test]
    fn payload_carries_exactly_the_python_args() {
        let cfg = config();
        let grid = DeepSelectLogitsTopkSpec::sweep_grid(&cfg);
        assert_eq!(
            DeepSelectLogitsTopkSpec::cache_kind("deep_select"),
            CacheKind::Cache1DLinear
        );
        let payloads = DeepSelectLogitsTopkSpec::enumerate(&cfg, &grid, "deep_select");
        assert_eq!(payloads.len(), 30);
        for payload in &payloads {
            let names: Vec<&str> = payload.fields().keys().map(String::as_str).collect();
            assert_eq!(
                names,
                [
                    "backend",
                    "dtype",
                    "num_columns",
                    "num_rows",
                    "sorted",
                    "top_k"
                ]
            );
            assert_eq!(payload.fields()["sorted"], Value::from(false));
            assert_eq!(payload.fields()["dtype"], Value::from("fp32"));
        }
        let input = DeepSelectLogitsTopkKernelInput { num_rows: 384 };
        assert_eq!(&*input.coords(), &[384.0]);
        assert_eq!(
            DeepSelectLogitsTopkKernelInput::coord_field_names(),
            &["num_rows"]
        );
    }

    /// Linear interpolation must not span a wave step: both sides of the first
    /// SM-count multiples are measured.
    #[test]
    fn row_axis_brackets_the_first_waves() {
        let axis = DeepSelectLogitsTopkSpec::sweep_grid(&config()).axes()[0].clone();
        for r in [
            148.0, 149.0, 296.0, 297.0, 444.0, 445.0, 592.0, 593.0, 8192.0,
        ] {
            assert!(axis.contains(&r), "missing {r}");
        }
    }
}
