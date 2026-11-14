//! Paged FP8 sparse MLA over a sliding-window cache and a compressed cache: one
//! DeepSeek-V4-family attention layer as SGLang runs it on SM12x
//! (`flash_mla_with_kvcache_sm120`), for decode, speculative verify and chunked
//! prefill alike.
//!
//! The call's work is set by three runtime numbers -- requests, query rows per
//! request, and the cached context before the first row -- so they are the
//! cache axes. FlashInfer picks the split-K decode kernel for up to 64 query
//! rows in the call and the paged prefill kernel above that; that switch sits
//! inside the (requests x rows) plane and both sides are measured.
//!
//! Context saturates. Row `i` attends to `min(swa_window, pos + 1)` window keys
//! and `min(extra_index_capacity, (pos + 1) / compress_ratio)` compressed keys
//! (`pos = context_before + i`), so once `context_before` reaches
//! `max(swa_window, compress_ratio * extra_index_capacity)` every row's key set
//! is full and the call no longer depends on it. The context axis stops there
//! and longer contexts are looked up at that point (`cache_coords`): on-grid,
//! not extrapolated.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, Coords, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

/// Granularity of the padded index arrays (FlashInfer's page re-split works in
/// 64-token pages).
const INDEX_ALIGN: u32 = 64;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PagedSparseMlaKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Query heads on this GPU before any kernel padding.
    pub num_heads: Dim,
    pub head_dim: Dim,
    pub value_dim: Dim,
    pub swa_window: u32,
    pub swa_index_capacity: u32,
    /// Source tokens per compressed key; 0 is a window-only layer.
    pub compress_ratio: u32,
    pub extra_index_capacity: u32,
    pub page_size: u32,
    pub kv_layout: String,
    #[compute_dtype]
    pub q_dtype: DType,
    #[kv_dtype]
    pub cache_dtype: DType,
    pub output_dtype: DType,
}

/// One call: `num_requests` requests of identical geometry, each with
/// `rows_per_request` query rows after `context_before` cached tokens.
#[derive(Clone, Debug, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct PagedSparseMlaKernelInput {
    pub num_requests: u32,
    pub rows_per_request: u32,
    pub context_before: u32,
}

fn validate_config(config: &PagedSparseMlaKernelConfig) {
    assert!(
        matches!(config.compress_ratio, 0 | 1 | 2 | 4 | 128),
        "paged_sparse_mla: compress_ratio {} is not a DeepSeek-V4 ratio",
        config.compress_ratio
    );
    assert!(
        config.swa_window >= 1
            && config.swa_index_capacity >= config.swa_window
            && config.swa_index_capacity % INDEX_ALIGN == 0,
        "paged_sparse_mla: swa_index_capacity must cover swa_window in multiples of {INDEX_ALIGN}"
    );
    assert!(
        config.extra_index_capacity % INDEX_ALIGN == 0
            && (config.compress_ratio == 0 || config.extra_index_capacity > 0),
        "paged_sparse_mla: extra_index_capacity must be a positive multiple of {INDEX_ALIGN} \
         on compressed layers"
    );
    assert!(
        config.page_size > 0 && config.page_size % INDEX_ALIGN == 0,
        "paged_sparse_mla: page_size must be a multiple of {INDEX_ALIGN}"
    );
    assert!(
        !config.kv_layout.is_empty(),
        "paged_sparse_mla: kv_layout is required"
    );
}

/// The context from which every row's key set is full.
fn saturated_context(config: &PagedSparseMlaKernelConfig) -> u32 {
    let compressed = config
        .compress_ratio
        .checked_mul(config.extra_index_capacity)
        .expect("compressed saturation point must fit u32");
    config.swa_window.max(compressed)
}

/// `0`, the window, `64 * 4^k` below saturation, and the saturation point.
fn context_axis(config: &PagedSparseMlaKernelConfig) -> Vec<f64> {
    let saturated = saturated_context(config);
    let mut points = vec![0, config.swa_window, saturated];
    let mut point = 64_u32;
    while point < saturated {
        points.push(point);
        point = point.saturating_mul(4);
    }
    points.sort_unstable();
    points.dedup();
    points.into_iter().map(f64::from).collect()
}

pub struct PagedSparseMlaSpec;

impl KernelSpec for PagedSparseMlaSpec {
    type Config = PagedSparseMlaKernelConfig;
    type Input = PagedSparseMlaKernelInput;

    const KIND: KernelKind = "paged_sparse_mla";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        validate_config(config);
        SweepGrid::new(vec![
            Axis::values([1, 2, 4, 8, 16, 32]),
            // Decode (1), DSpark/MTP verify widths, and prefill chunks to 2048.
            Axis::values([1, 4, 7, 16, 32, 64, 128, 512, 1024, 2048]),
            context_axis(config),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache3DLinear
    }

    fn cache_coords(config: &Self::Config, input: &Self::Input) -> Coords {
        Coords::new([
            f64::from(input.num_requests),
            f64::from(input.rows_per_request),
            f64::from(input.context_before.min(saturated_context(config))),
        ])
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_3d(|num_requests, rows_per_request, context_before| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_requests", num_requests as u32)
                .with("rows_per_request", rows_per_request as u32)
                .with("context_before", context_before as u32)
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("value_dim", config.value_dim.get())
                .with("swa_window", config.swa_window)
                .with("swa_index_capacity", config.swa_index_capacity)
                .with("compress_ratio", config.compress_ratio)
                .with("extra_index_capacity", config.extra_index_capacity)
                .with("page_size", config.page_size)
                .with("kv_layout", config.kv_layout.clone())
                .with("q_dtype", config.q_dtype.as_str())
                .with("cache_dtype", config.cache_dtype.as_str())
                .with("output_dtype", config.output_dtype.as_str())
        })
    }
}

register_kernel!(PagedSparseMlaKernel, PagedSparseMlaSpec);

#[cfg(test)]
mod tests {
    use super::*;

    /// DeepSeek-V4.1-Flash on SGLang SM12x: 16 local heads, a 128-token window,
    /// 512 compressed keys, 256-token pages.
    fn config(compress_ratio: u32) -> PagedSparseMlaKernelConfig {
        PagedSparseMlaKernelConfig {
            backends: vec!["sglang_flashinfer_sm120"],
            gpu_name: "NVIDIA RTX PRO 6000 Blackwell Server Edition".to_string(),
            num_heads: 16.into(),
            head_dim: 512.into(),
            value_dim: 512.into(),
            swa_window: 128,
            swa_index_capacity: 128,
            compress_ratio,
            extra_index_capacity: 512,
            page_size: 256,
            kv_layout: "v4".to_string(),
            q_dtype: DType::Bf16,
            cache_dtype: DType::Fp8E4m3,
            output_dtype: DType::Bf16,
        }
    }

    /// Past the saturation point a longer context costs nothing more, so it
    /// must read the last measured context, not extrapolate along it.
    #[test]
    fn long_contexts_read_the_saturated_point() {
        let cfg = config(2);
        let input = |context_before| PagedSparseMlaKernelInput {
            num_requests: 8,
            rows_per_request: 7,
            context_before,
        };
        assert_eq!(
            &*PagedSparseMlaSpec::cache_coords(&cfg, &input(300_000)),
            &[8.0, 7.0, 1024.0]
        );
        assert_eq!(
            &*PagedSparseMlaSpec::cache_coords(&cfg, &input(600)),
            &[8.0, 7.0, 600.0]
        );
        // Window-only layers saturate at the window.
        assert_eq!(
            &*PagedSparseMlaSpec::cache_coords(&config(0), &input(5_000)),
            &[8.0, 7.0, 128.0]
        );
    }

    /// The context axis ends exactly at saturation, and the grid stays inside
    /// the profiling budget for every V4 ratio.
    #[test]
    fn context_axis_ends_at_saturation_and_grid_fits_the_budget() {
        assert_eq!(context_axis(&config(2)), [0.0, 64.0, 128.0, 256.0, 1024.0]);
        assert_eq!(context_axis(&config(0)), [0.0, 64.0, 128.0]);
        for ratio in [0, 1, 2, 4, 128] {
            let cfg = config(ratio);
            let points: usize = PagedSparseMlaSpec::sweep_grid(&cfg)
                .axes()
                .iter()
                .map(Vec::len)
                .product();
            assert!(points <= 500, "ratio {ratio}: {points} grid points");
        }
    }

    /// The payload must carry exactly the Python `PagedSparseMlaArgs` columns.
    #[test]
    fn payload_matches_the_python_args_columns() {
        let cfg = config(1);
        let payloads = PagedSparseMlaSpec::enumerate(
            &cfg,
            &PagedSparseMlaSpec::sweep_grid(&cfg),
            "sglang_flashinfer_sm120",
        );
        let fields: Vec<_> = payloads[0].fields().keys().cloned().collect();
        assert_eq!(
            fields,
            [
                "backend",
                "cache_dtype",
                "compress_ratio",
                "context_before",
                "extra_index_capacity",
                "head_dim",
                "kv_layout",
                "num_heads",
                "num_requests",
                "output_dtype",
                "page_size",
                "q_dtype",
                "rows_per_request",
                "swa_index_capacity",
                "swa_window",
                "value_dim"
            ]
        );
    }

    #[test]
    #[should_panic(expected = "not a DeepSeek-V4 ratio")]
    fn an_unknown_compress_ratio_is_rejected() {
        PagedSparseMlaSpec::sweep_grid(&config(3));
    }
}
