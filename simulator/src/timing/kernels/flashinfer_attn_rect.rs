//! Non-causal rect attention kernel: one cached perf model per attention-dims
//! config, swept over `(num_requests, q_len, kv_len)`.
//!
//! One call covers `num_requests` requests, each a block of `q_len` queries
//! attending (no causal mask) its own `kv_len` keys and values -- the DFlash2
//! draft layer's varlen call. Rows written before the batch axis existed are
//! one request (`num_requests` absent on the Python side). Everything generic
//! lives in `engine::Kernel<S>`; the Python `FlashinferAttnRectArgs` dataclass
//! owns the matching schema.
//!
//! Three monotone axes -> `Cache3DLinear` + `grid.expand_3d`. Work and K/V bytes
//! both scale with `num_requests`, so a batch is priced at its own row instead
//! of one shared context.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::SweepGrid;
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FlashinferAttnRectKernelConfig {
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

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct FlashinferAttnRectKernelInput {
    /// Requests in the one call.
    pub num_requests: u32,
    /// Query rows of each request.
    pub q_len: u32,
    /// Keys and values each request attends.
    pub kv_len: u32,
}

/// Requests per call: dense to 64 (a DP rank's draft batch), then sparser.
pub const RECT_REQUEST_AXIS: [u32; 12] = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256];
/// Query block per request: a speculative block is `1 + draft_tokens` >= 2.
pub const RECT_Q_AXIS: [u32; 4] = [2, 4, 8, 16];
/// Keys per request; work is linear in it.
pub const RECT_KV_AXIS: [u32; 8] = [128, 256, 512, 1024, 2048, 4096, 8192, 16384];

pub struct FlashinferAttnRectSpec;

impl KernelSpec for FlashinferAttnRectSpec {
    type Config = FlashinferAttnRectKernelConfig;
    type Input = FlashinferAttnRectKernelInput;

    const KIND: KernelKind = "flashinfer_attn_rect";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // 12 x 4 x 8 = 384 points per config.
        let axis = |v: &[u32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
        SweepGrid::new(vec![
            axis(&RECT_REQUEST_AXIS),
            axis(&RECT_Q_AXIS),
            axis(&RECT_KV_AXIS),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache3DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_3d(|num_requests, q_len, kv_len| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_qo_heads", config.num_qo_heads.get())
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("q_dtype", config.q_dtype.as_str())
                .with("kv_dtype", config.kv_dtype.as_str())
                .with("o_dtype", config.o_dtype.as_str())
                .with("q_len", q_len as u32)
                .with("kv_len", kv_len as u32)
                .with("num_requests", num_requests as u32)
        })
    }
}

register_kernel!(FlashinferAttnRectKernel, FlashinferAttnRectSpec);

#[cfg(test)]
mod tests {
    use super::{
        FlashinferAttnRectKernelConfig, FlashinferAttnRectKernelInput, FlashinferAttnRectSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::KernelSpec;
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn config() -> FlashinferAttnRectKernelConfig {
        FlashinferAttnRectKernelConfig {
            backends: vec!["fa2", "vllm_fa4_nosplit"],
            gpu_name: "NVIDIA B200".to_string(),
            num_qo_heads: 32.into(),
            num_kv_heads: 8.into(),
            head_dim: 128.into(),
            q_dtype: DType::Bf16,
            kv_dtype: DType::Bf16,
            o_dtype: DType::Bf16,
        }
    }

    /// The batch is its own axis: a call of B requests must not collapse to one
    /// request of B * q rows over one shared context.
    #[test]
    fn input_coords_keep_the_request_count_separate() {
        let input = FlashinferAttnRectKernelInput {
            num_requests: 32,
            q_len: 8,
            kv_len: 2048,
        };
        assert_eq!(&*input.coords(), &[32.0, 8.0, 2048.0]);
        assert_eq!(
            FlashinferAttnRectSpec::cache_kind("fa2"),
            CacheKind::Cache3DLinear
        );
    }

    #[test]
    fn grid_stays_under_the_profiling_ceiling_and_forwards_every_field() {
        let cfg = config();
        let grid = FlashinferAttnRectSpec::sweep_grid(&cfg);
        let payloads = FlashinferAttnRectSpec::enumerate(&cfg, &grid, "fa2");
        assert_eq!(payloads.len(), 384);
        let fields = payloads[0].fields();
        // Wire schema must stay aligned with Python FlashinferAttnRectArgs.
        assert_eq!(fields.len(), 10);
        assert_eq!(fields.get("num_requests"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("q_len"), Some(&Value::from(2_u32)));
        assert_eq!(fields.get("kv_len"), Some(&Value::from(128_u32)));
        assert_eq!(payloads[0].backend(), Some("fa2"));
    }
}
