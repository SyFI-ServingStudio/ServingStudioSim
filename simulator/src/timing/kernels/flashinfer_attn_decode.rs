//! `FlashInfer` decode attention kernel: one cached perf model per attention-dims
//! config, swept over a 2D `(batch_size, avg_len)` grid.
//!
//! Decode is batched single-token-query attention against a paged KV cache. The
//! cache exception (vs prefill/rect): batch is a real sweep axis. Identity is
//! `(batch_size, total_tokens)` — the TOTAL kv tokens across the batch, not the
//! per-request length. The Python runner derives the per-request (mean) length
//! `avg_len = max(1, total_tokens / batch_size)`, sets `q_len = 1`,
//! `kv_len = avg_len`, non-causal.
//!
//! Why `total_tokens` (not `avg_len)`: feasibility is bounded by the PRODUCT
//! `batch_size` * `avg_len` (= total kv tokens / memory / profiling cost), so making
//! the product an axis lets a plain rectangular `total_tokens` cap keep every
//! grid corner affordable (e.g. 1x4M and 256x16384 both = 4M total), instead
//! of a rectangular `(batch, avg_len)` grid whose `256 x 4M` corner is
//! unaffordable. Everything generic lives in `engine::Kernel<S>`; the Python
//! `FlashinferAttnDecodeArgs` dataclass owns the matching schema.
//!
//! Axes: `batch_size` is pow2 (1..=256); `total_tokens` is pow2 (32..=4194304, i.e.
//! up to 4M). Two monotonic axes -> `Cache2DLinear` + `grid.expand_2d`.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::{CacheKind, Extrapolation};
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct FlashinferAttnDecodeKernelConfig {
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
pub struct FlashinferAttnDecodeKernelInput {
    pub batch_size: u32,
    pub total_tokens: u32,
}

pub struct FlashinferAttnDecodeSpec;

impl KernelSpec for FlashinferAttnDecodeSpec {
    type Config = FlashinferAttnDecodeKernelConfig;
    type Input = FlashinferAttnDecodeKernelInput;

    const KIND: KernelKind = "flashinfer_attn_decode";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // batch_size: 1..=256 (pow2); total_tokens: 32..=4194304 (pow2, up to
        // 4M). A flat total_tokens cap keeps every corner feasible. Row-major.
        SweepGrid::new(vec![Axis::pow2(0, 8), Axis::pow2(5, 22)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        // `total_tokens` is ALREADY the product (total KV across the batch), so
        // the two axes must not be multiplied again: doubling `batch_size` at a
        // fixed `total_tokens` re-slices the same KV read, it does not double it.
        // Time follows the tokens; batch only moves parallelism and launch cost.
        CacheKind::Cache2DLinear(Extrapolation::Weighted)
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "batch_size/total_tokens are non-negative sweep-grid coordinates, far below u32::MAX"
        )]
        grid.expand_2d(|batch_size, total_tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_qo_heads", config.num_qo_heads.get())
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("q_dtype", config.q_dtype.as_str())
                .with("kv_dtype", config.kv_dtype.as_str())
                .with("o_dtype", config.o_dtype.as_str())
                .with("batch_size", batch_size as u32)
                .with("total_tokens", total_tokens as u32)
        })
    }
}

register_kernel!(FlashinferAttnDecodeKernel, FlashinferAttnDecodeSpec);

#[cfg(test)]
mod tests {
    use super::{
        FlashinferAttnDecodeKernelConfig, FlashinferAttnDecodeKernelInput, FlashinferAttnDecodeSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::{CacheKind, Extrapolation};
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn config() -> FlashinferAttnDecodeKernelConfig {
        FlashinferAttnDecodeKernelConfig {
            backends: vec!["fa2", "fa3", "trt", "cudnn"],
            gpu_name: "H100".to_string(),
            num_qo_heads: 32.into(),
            num_kv_heads: 8.into(),
            head_dim: 128.into(),
            q_dtype: DType::Bf16,
            kv_dtype: DType::Bf16,
            o_dtype: DType::Bf16,
        }
    }

    #[test]
    fn config_identity_includes_backends_and_attention_dims() {
        let cfg = config();
        assert_eq!(cfg.backends, vec!["fa2", "fa3", "trt", "cudnn"]);
        assert_eq!(cfg.num_qo_heads, 32);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.q_dtype, DType::Bf16);
        assert_eq!(cfg.kv_dtype, DType::Bf16);
        assert_eq!(cfg.o_dtype, DType::Bf16);
    }

    #[test]
    fn describe_config_renders_tidy_field_list() {
        assert_eq!(
            config().describe_config(),
            serde_json::json!({
                "backends": ["fa2", "fa3", "trt", "cudnn"], "gpu_name": "H100",
                "num_qo_heads": {"value": 32, "expression": null, "bindings": {}},
                "num_kv_heads": {"value": 8, "expression": null, "bindings": {}},
                "head_dim": {"value": 128, "expression": null, "bindings": {}},
                "q_dtype": "bf16", "kv_dtype": "bf16", "o_dtype": "bf16",
            })
        );
    }

    #[test]
    fn input_sweep_coords_flatten_both_axes_in_order() {
        let input = FlashinferAttnDecodeKernelInput {
            batch_size: 64,
            total_tokens: 262144,
        };
        assert_eq!(&*input.coords(), &[64.0, 262144.0]);
    }

    #[test]
    fn cache_kind_is_two_dimensional_linear() {
        assert_eq!(
            FlashinferAttnDecodeSpec::cache_kind("fa2"),
            CacheKind::Cache2DLinear(Extrapolation::Weighted)
        );
    }

    #[test]
    fn sweep_grid_batch_pow2_and_total_tokens_to_4m() {
        let grid = FlashinferAttnDecodeSpec::sweep_grid(&config());
        assert_eq!(grid.axes().len(), 2);
        // batch_size pow2 1..=256; total_tokens pow2 32..=4194304 (4M).
        assert_eq!(grid.axes()[0][0], 1.0);
        assert_eq!(*grid.axes()[0].last().unwrap(), 256.0);
        assert_eq!(grid.axes()[1][0], 32.0);
        assert_eq!(*grid.axes()[1].last().unwrap(), 4194304.0);
    }

    #[test]
    fn enumerate_emits_payload_with_all_wire_fields() {
        let cfg = config();
        let grid = FlashinferAttnDecodeSpec::sweep_grid(&cfg);
        let payloads = FlashinferAttnDecodeSpec::enumerate(&cfg, &grid, "fa2");

        assert_eq!(payloads.len(), grid.axes()[0].len() * grid.axes()[1].len());
        let first = &payloads[0];
        let fields = first.fields();

        // Wire schema must stay aligned with Python FlashinferAttnDecodeArgs in
        // profiling/kernels/flashinfer_attn_decode.py.
        assert_eq!(fields.len(), 9);
        assert_eq!(fields.get("backend"), Some(&Value::from("fa2")));
        assert_eq!(fields.get("num_qo_heads"), Some(&Value::from(32_u32)));
        assert_eq!(fields.get("num_kv_heads"), Some(&Value::from(8_u32)));
        assert_eq!(fields.get("head_dim"), Some(&Value::from(128_u32)));
        assert_eq!(fields.get("q_dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("kv_dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("o_dtype"), Some(&Value::from("bf16")));
        // First row: batch_size == 1 (pow2 start), total_tokens == 32.
        assert_eq!(fields.get("batch_size"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("total_tokens"), Some(&Value::from(32_u32)));
        assert_eq!(first.backend(), Some("fa2"));
    }
}
