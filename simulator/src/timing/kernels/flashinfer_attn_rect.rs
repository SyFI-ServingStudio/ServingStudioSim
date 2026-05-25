//! FlashInfer rect (non-causal) attention kernel: one cached perf model per
//! attention-dims config, swept over a 2D `(prefix_len, append_len)` grid.
//!
//! Rect is the non-causal sibling of `flashinfer_attn_prefill`. Unlike prefill
//! (causal, parametrized by `(prefix_len, append_len)`), rect is parametrized by
//! `(q_len, kv_len)` *directly* — for non-causal attention there is no causal
//! prefix/append split, and the direct form lets rect sweep `q_len > kv_len`
//! (wide rectangles) which the prefill encoding cannot. `causal=False` is a
//! Python-runner concern; the Rust spec carries only identity, sweep shape, and
//! cache kind. Everything generic lives in `engine::Kernel<S>`. The Python
//! `FlashinferAttnRectArgs` dataclass owns the matching schema.
//!
//! Both axes are independent token curves -> `Cache2DLinear` + `grid.expand_2d`.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Deserialize)]
pub struct FlashinferAttnRectKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_qo_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub q_dtype: DType,
    pub kv_dtype: DType,
    pub o_dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct FlashinferAttnRectKernelInput {
    pub q_len: u32,
    pub kv_len: u32,
}

pub struct FlashinferAttnRectSpec;

impl KernelSpec for FlashinferAttnRectSpec {
    type Config = FlashinferAttnRectKernelConfig;
    type Input = FlashinferAttnRectKernelInput;

    const KIND: KernelKind = "flashinfer_attn_rect";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        // q_len and kv_len are independent token curves; their cartesian product
        // covers both q_len <= kv_len and q_len > kv_len (wide rectangles).
        SweepGrid::new(vec![Axis::token_axis(), Axis::token_axis()])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache2DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_2d(|q_len, kv_len| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_qo_heads", config.num_qo_heads)
                .with("num_kv_heads", config.num_kv_heads)
                .with("head_dim", config.head_dim)
                .with("q_dtype", config.q_dtype.as_str())
                .with("kv_dtype", config.kv_dtype.as_str())
                .with("o_dtype", config.o_dtype.as_str())
                .with("q_len", q_len as u32)
                .with("kv_len", kv_len as u32)
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
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::SweepCoords;
    use serde_json::Value;

    fn config() -> FlashinferAttnRectKernelConfig {
        FlashinferAttnRectKernelConfig {
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
        let input = FlashinferAttnRectKernelInput {
            q_len: 1024,
            kv_len: 256,
        };
        // q_len > kv_len is representable (wide rectangle).
        assert_eq!(&*input.coords(), &[1024.0, 256.0]);
    }

    #[test]
    fn cache_kind_is_two_dimensional_linear() {
        assert_eq!(
            FlashinferAttnRectSpec::cache_kind("fa2"),
            CacheKind::Cache2DLinear
        );
    }

    #[test]
    fn sweep_grid_both_axes_are_token_curves() {
        let grid = FlashinferAttnRectSpec::sweep_grid(&config());
        assert_eq!(grid.axes().len(), 2);
        // Both q_len and kv_len start at the token-curve minimum (no chained 0).
        assert_eq!(grid.axes()[0][0], 32.0);
        assert_eq!(grid.axes()[1][0], 32.0);
    }

    #[test]
    fn enumerate_emits_payload_with_all_wire_fields() {
        let cfg = config();
        let grid = FlashinferAttnRectSpec::sweep_grid(&cfg);
        let payloads = FlashinferAttnRectSpec::enumerate(&cfg, &grid, "fa2");

        assert_eq!(payloads.len(), grid.axes()[0].len() * grid.axes()[1].len());
        let first = &payloads[0];
        let fields = first.fields();

        // Wire schema must stay aligned with Python FlashinferAttnRectArgs in
        // profiling/kernels/flashinfer_attn_rect.py.
        assert_eq!(fields.len(), 9);
        assert_eq!(fields.get("backend"), Some(&Value::from("fa2")));
        assert_eq!(fields.get("num_qo_heads"), Some(&Value::from(32_u32)));
        assert_eq!(fields.get("num_kv_heads"), Some(&Value::from(8_u32)));
        assert_eq!(fields.get("head_dim"), Some(&Value::from(128_u32)));
        assert_eq!(fields.get("q_dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("kv_dtype"), Some(&Value::from("bf16")));
        assert_eq!(fields.get("o_dtype"), Some(&Value::from("bf16")));
        assert!(fields.get("q_len").and_then(Value::as_u64).is_some());
        assert!(fields.get("kv_len").and_then(Value::as_u64).is_some());
        assert_eq!(first.backend(), Some("fa2"));
    }
}
