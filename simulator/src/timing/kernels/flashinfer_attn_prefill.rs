//! FlashInfer prefill (causal) attention kernel: one cached perf model per
//! attention-dims config, swept over a 2D `(prefix_len, append_len)` grid.
//!
//! Everything generic (build / eval / the `Probe` impl /
//! for-backend loops) lives in `engine::Kernel<S>`. This file declares the
//! prefill-specific Config / Input, the `KIND` wire string, and the `enumerate`
//! body that lifts (config, sweep coord, backend) to the on-wire `ArgsPayload`.
//! The Python `FlashinferAttnPrefillArgs` dataclass owns the matching schema.
//!
//! Novel bit — **cache dims are not the kernel params**. The cache identity is
//! `(prefix_len, append_len)`; the Python runner derives `q_len = append_len`,
//! `kv_len = prefix_len + append_len`. Merging the ref's pure-prefill + chunked
//! ops, pure prefill is just `prefix_len == 0` — so the `prefix_len` axis chains
//! `[0]` ahead of a log2 token curve (128..32k). `rect` is the non-causal sibling and shares
//! this exact shape (see `flashinfer_attn_rect.rs`), differing only by `causal`
//! on the Python side.
//!
//! 2D note: like `attn_prefill` before it, two monotonic axes -> `Cache2DLinear`
//! + `grid.expand_2d`. See `timing/sweep.rs`.

use crate::timing::bridge::{ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{Kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug)]
pub struct FlashinferAttnPrefillKernelConfig {
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_qo_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub q_dtype: DType,
    pub kv_dtype: DType,
    pub o_dtype: DType,
}

#[derive(SweepCoords)]
pub struct FlashinferAttnPrefillKernelInput {
    pub prefix_len: u32,
    pub append_len: u32,
}

pub struct FlashinferAttnPrefillSpec;

impl KernelSpec for FlashinferAttnPrefillSpec {
    type Config = FlashinferAttnPrefillKernelConfig;
    type Input = FlashinferAttnPrefillKernelInput;

    const KIND: KernelKind = "flashinfer_attn_prefill";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // prefix_len (existing context): 0 for pure prefill, chained ahead of a
        // log2 curve 128..32k; append_len (new tokens): the same log2 curve.
        // Row-major over (prefix_len, append_len). pow2(7, 15) = 128..32768.
        SweepGrid::new(vec![
            Axis::chain([Axis::values([0]), Axis::pow2(7, 15)]),
            Axis::pow2(7, 15),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|prefix_len, append_len| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_qo_heads", config.num_qo_heads)
                .with("num_kv_heads", config.num_kv_heads)
                .with("head_dim", config.head_dim)
                .with("q_dtype", config.q_dtype.as_str())
                .with("kv_dtype", config.kv_dtype.as_str())
                .with("o_dtype", config.o_dtype.as_str())
                .with("prefix_len", prefix_len as u32)
                .with("append_len", append_len as u32)
        })
    }
}

pub type FlashinferAttnPrefillKernel = Kernel<FlashinferAttnPrefillSpec>;

#[cfg(test)]
mod tests {
    use super::{
        FlashinferAttnPrefillKernelConfig, FlashinferAttnPrefillKernelInput,
        FlashinferAttnPrefillSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn config() -> FlashinferAttnPrefillKernelConfig {
        FlashinferAttnPrefillKernelConfig {
            backends: vec!["fa2", "fa3", "trt", "cudnn"],
            gpu_name: "H100".to_string(),
            num_qo_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
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
            r#"backends=["fa2", "fa3", "trt", "cudnn"] gpu_name="H100" num_qo_heads=32 num_kv_heads=8 head_dim=128 q_dtype=Bf16 kv_dtype=Bf16 o_dtype=Bf16"#
        );
    }

    #[test]
    fn input_sweep_coords_flatten_both_axes_in_order() {
        let input = FlashinferAttnPrefillKernelInput {
            prefix_len: 2048,
            append_len: 512,
        };
        assert_eq!(&*input.coords(), &[2048.0, 512.0]);
    }

    #[test]
    fn cache_kind_is_two_dimensional_linear() {
        assert_eq!(
            FlashinferAttnPrefillSpec::cache_kind("fa2"),
            CacheKind::Cache2DLinear
        );
    }

    #[test]
    fn sweep_grid_prefix_axis_includes_zero_for_pure_prefill() {
        let grid = FlashinferAttnPrefillSpec::sweep_grid(&config());
        assert_eq!(grid.axes().len(), 2);
        // prefix_len axis chains [0] ahead of the log2 curve (128 = 2^7).
        assert_eq!(grid.axes()[0][0], 0.0);
        assert_eq!(grid.axes()[0][1], 128.0);
        assert_eq!(grid.axes()[0].len(), 10); // [0] + pow2(7..=15) = 1 + 9
        // append_len axis is the log2 curve 128..32k (2^7..2^15).
        assert_eq!(grid.axes()[1][0], 128.0);
        assert_eq!(*grid.axes()[1].last().unwrap(), 32768.0);
        assert_eq!(grid.axes()[1].len(), 9);
    }

    #[test]
    fn enumerate_emits_payload_with_all_wire_fields() {
        let cfg = config();
        let grid = FlashinferAttnPrefillSpec::sweep_grid(&cfg);
        let payloads = FlashinferAttnPrefillSpec::enumerate(&cfg, &grid, "fa2");

        // 2D cartesian product of the two axes.
        assert_eq!(payloads.len(), grid.axes()[0].len() * grid.axes()[1].len());
        let first = &payloads[0];
        let fields = first.fields();

        // Wire schema: { backend, num_qo_heads, num_kv_heads, head_dim, q_dtype,
        // kv_dtype, o_dtype, prefix_len, append_len } — must stay aligned with
        // Python FlashinferAttnPrefillArgs in
        // profiling/kernels/flashinfer_attn_prefill.py.
        assert_eq!(fields.len(), 9);
        assert_eq!(fields.get("backend"), Some(&Value::from("fa2")));
        assert_eq!(fields.get("num_qo_heads"), Some(&Value::from(32_u32)));
        assert_eq!(fields.get("num_kv_heads"), Some(&Value::from(8_u32)));
        assert_eq!(fields.get("head_dim"), Some(&Value::from(128_u32)));
        assert_eq!(fields.get("q_dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("kv_dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("o_dtype"), Some(&Value::from("bf16")));
        // First row is pure prefill: prefix_len == 0.
        assert_eq!(fields.get("prefix_len"), Some(&Value::from(0_u32)));
        assert!(fields.get("append_len").and_then(Value::as_u64).is_some());
        assert_eq!(first.backend(), Some("fa2"));
    }
}
