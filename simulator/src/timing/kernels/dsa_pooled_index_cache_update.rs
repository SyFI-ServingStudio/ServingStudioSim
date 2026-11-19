//! DSA pooled index-key decode/verify cache update.
//!
//! One launch per decode step: one program per request walks its `next_n` new
//! tokens, stashes each in the request's tail ring and compresses every pool a
//! token completes into the page-planar FP8 index cache. Static identity is the
//! verify width (`next_n`), pool size, tail ring, index width, cache page size
//! and dtypes; runtime is the number of requests. This kind owns its Python
//! profile table.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaPooledIndexCacheUpdateKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// New tokens per request in one step (1 for plain decode).
    pub next_n: u32,
    /// Tokens compressed into one cached index key.
    pub pool_size: u32,
    /// Slots in each request's raw-token tail ring.
    pub tail_ring: u32,
    pub index_dim: Dim,
    /// Pooled keys per index-cache page.
    pub block_size: u32,
    #[compute_dtype]
    pub input_dtype: DType,
    #[kv_dtype]
    pub cache_dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct DsaPooledIndexCacheUpdateKernelInput {
    /// Requests in the decode step (one program each).
    pub num_requests: u32,
}

pub struct DsaPooledIndexCacheUpdateSpec;

impl KernelSpec for DsaPooledIndexCacheUpdateSpec {
    type Config = DsaPooledIndexCacheUpdateKernelConfig;
    type Input = DsaPooledIndexCacheUpdateKernelInput;

    const KIND: KernelKind = "dsa_pooled_index_cache_update";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // One program per request, so time steps at CTA waves: the 148/296/592/
        // 1184 points sit on B200 SM multiples, the 1.5x points between doublings
        // keep the linear cache honest across them. 2048 requests covers decode
        // and verify batches past 1024 rows at every next_n.
        SweepGrid::new(vec![Axis::values([
            1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 148, 192, 256, 296, 384, 512, 592,
            768, 1024, 1184, 1536, 2048,
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
        grid.expand_1d(|num_requests| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_requests", num_requests as u32)
                .with("next_n", config.next_n)
                .with("pool_size", config.pool_size)
                .with("tail_ring", config.tail_ring)
                .with("index_dim", config.index_dim.get())
                .with("block_size", config.block_size)
                .with("input_dtype", config.input_dtype.as_str())
                .with("cache_dtype", config.cache_dtype.as_str())
        })
    }
}

register_kernel!(
    DsaPooledIndexCacheUpdateKernel,
    DsaPooledIndexCacheUpdateSpec
);

#[cfg(test)]
mod tests {
    use super::{
        DsaPooledIndexCacheUpdateKernelConfig, DsaPooledIndexCacheUpdateKernelInput,
        DsaPooledIndexCacheUpdateSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{Dim, SlotInput, SweepCoords};
    use serde_json::Value;

    const BACKEND: &str = "vllm_fork_triton";

    fn config() -> DsaPooledIndexCacheUpdateKernelConfig {
        DsaPooledIndexCacheUpdateKernelConfig {
            backends: vec![BACKEND],
            gpu_name: "NVIDIA B200".to_string(),
            next_n: 8,
            pool_size: 4,
            tail_ring: 16,
            index_dim: Dim::param("index_head_dim", 128),
            block_size: 64,
            input_dtype: DType::Bf16,
            cache_dtype: DType::Fp8E4m3,
        }
    }

    #[test]
    fn input_is_exactly_the_request_count() {
        // Catches a re-axis onto rows (requests x next_n): the kernel launches
        // one program per request, so rows would mis-price verify steps.
        let input = DsaPooledIndexCacheUpdateKernelInput { num_requests: 16 };
        assert_eq!(&*input.coords(), &[16.0]);
        let slot: SlotInput = input.into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_requests": 16})
        );
    }

    #[test]
    fn sweep_is_sorted_and_reaches_past_1024_rows_at_every_next_n() {
        let grid = DsaPooledIndexCacheUpdateSpec::sweep_grid(&config());
        let axis = &grid.axes()[0];
        assert_eq!(axis.first(), Some(&1.0));
        assert_eq!(axis.last(), Some(&2048.0));
        assert!(axis.windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn enumerate_matches_the_python_wire_schema() {
        let cfg = config();
        let grid = DsaPooledIndexCacheUpdateSpec::sweep_grid(&cfg);
        let payloads = DsaPooledIndexCacheUpdateSpec::enumerate(&cfg, &grid, BACKEND);
        assert_eq!(payloads.len(), grid.axes()[0].len());
        let fields = payloads[0].fields();
        let names: Vec<&str> = fields.keys().map(String::as_str).collect();
        assert_eq!(
            names,
            [
                "backend",
                "block_size",
                "cache_dtype",
                "index_dim",
                "input_dtype",
                "next_n",
                "num_requests",
                "pool_size",
                "tail_ring",
            ]
        );
        assert_eq!(fields.get("num_requests"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("next_n"), Some(&Value::from(8_u32)));
        assert_eq!(fields.get("index_dim"), Some(&Value::from(128_u32)));
        assert_eq!(fields.get("input_dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("cache_dtype"), Some(&Value::from("fp8_e4m3")));
        assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
        assert_eq!(cfg.kv_dtype(), Some(DType::Fp8E4m3));
    }
}
