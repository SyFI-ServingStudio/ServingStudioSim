//! GLM-5.2 production-layout batched GEMMs: one cached perf model per
//! `(num_batches, n, k, dtype)` config.
//!
//! Q absorption and V-up use separate kernel instances with singleton backend
//! lists and different static `(n, k)` dimensions. This kind owns its Python
//! profile table and does not reuse `single_gemm` through `profile_kind()`.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BatchedGemmKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_batches: Dim,
    pub n: Dim,
    pub k: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct BatchedGemmKernelInput {
    pub m: u32,
}

pub struct BatchedGemmSpec;

impl KernelSpec for BatchedGemmSpec {
    type Config = BatchedGemmKernelConfig;
    type Input = BatchedGemmKernelInput;

    const KIND: KernelKind = "batched_gemm";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::chain([Axis::pow2(0, 4), Axis::token_axis()])])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|m| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_batches", config.num_batches.get())
                .with("m", m as u32)
                .with("n", config.n.get())
                .with("k", config.k.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(BatchedGemmKernel, BatchedGemmSpec);

#[cfg(test)]
mod tests {
    use super::{BatchedGemmKernelConfig, BatchedGemmKernelInput, BatchedGemmSpec};
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{Dim, SlotInput, SweepCoords};
    use serde_json::Value;

    const Q_BACKEND: &str = "torch_mla_q_absorb_glm52";
    const V_BACKEND: &str = "torch_mla_v_up_glm52";

    fn local_heads() -> Dim {
        Dim::param("num_attention_heads", 64) / Dim::param("attn_tp", 1)
    }

    fn q_config() -> BatchedGemmKernelConfig {
        BatchedGemmKernelConfig {
            backends: vec![Q_BACKEND],
            gpu_name: "NVIDIA H200".to_string(),
            num_batches: local_heads(),
            n: Dim::param("kv_lora_rank", 512),
            k: Dim::param("qk_nope_head_dim", 192),
            dtype: DType::Bf16,
        }
    }

    fn v_config() -> BatchedGemmKernelConfig {
        BatchedGemmKernelConfig {
            backends: vec![V_BACKEND],
            gpu_name: "NVIDIA H200".to_string(),
            num_batches: local_heads(),
            n: Dim::param("v_head_dim", 256),
            k: Dim::param("kv_lora_rank", 512),
            dtype: DType::Bf16,
        }
    }

    #[test]
    fn q_and_v_configs_keep_layout_backends_and_shapes_separate() {
        let q = q_config();
        let v = v_config();

        assert_eq!(BatchedGemmSpec::KIND, "batched_gemm");
        assert_eq!(BatchedGemmSpec::profile_kind(), "batched_gemm");

        assert_eq!(q.backends(), &[Q_BACKEND]);
        assert_eq!(q.gpu_name(), "NVIDIA H200");
        assert_eq!(q.num_batches, 64);
        assert_eq!(q.n, 512);
        assert_eq!(q.k, 192);
        assert_eq!(q.dtype, DType::Bf16);

        assert_eq!(v.backends(), &[V_BACKEND]);
        assert_eq!(v.gpu_name(), "NVIDIA H200");
        assert_eq!(v.num_batches, 64);
        assert_eq!(v.n, 256);
        assert_eq!(v.k, 512);
        assert_eq!(v.dtype, DType::Bf16);
    }

    #[test]
    fn describe_config_preserves_rich_q_and_v_dims() {
        assert_eq!(
            q_config().describe_config(),
            serde_json::json!({
                "backends": [Q_BACKEND],
                "gpu_name": "NVIDIA H200",
                "num_batches": {
                    "value": 64,
                    "expression": "num_attention_heads/attn_tp",
                    "bindings": {"attn_tp": 1, "num_attention_heads": 64},
                },
                "n": {
                    "value": 512,
                    "expression": "kv_lora_rank",
                    "bindings": {"kv_lora_rank": 512},
                },
                "k": {
                    "value": 192,
                    "expression": "qk_nope_head_dim",
                    "bindings": {"qk_nope_head_dim": 192},
                },
                "dtype": "bf16",
            })
        );
        assert_eq!(
            v_config().describe_config(),
            serde_json::json!({
                "backends": [V_BACKEND],
                "gpu_name": "NVIDIA H200",
                "num_batches": {
                    "value": 64,
                    "expression": "num_attention_heads/attn_tp",
                    "bindings": {"attn_tp": 1, "num_attention_heads": 64},
                },
                "n": {
                    "value": 256,
                    "expression": "v_head_dim",
                    "bindings": {"v_head_dim": 256},
                },
                "k": {
                    "value": 512,
                    "expression": "kv_lora_rank",
                    "bindings": {"kv_lora_rank": 512},
                },
                "dtype": "bf16",
            })
        );
    }

    #[test]
    fn input_coords_and_slot_input_are_exactly_m() {
        let input = BatchedGemmKernelInput { m: 128 };

        assert_eq!(&*input.coords(), &[128.0]);
        assert_eq!(BatchedGemmKernelInput::coord_field_names(), &["m"]);

        let slot: SlotInput = input.into();
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"m": 128})
        );
    }

    #[test]
    fn sweep_has_frozen_decode_and_prefill_coverage_and_empty_mask() {
        for cfg in [q_config(), v_config()] {
            let grid = BatchedGemmSpec::sweep_grid(&cfg);

            assert_eq!(grid.axes().len(), 1);
            assert_eq!(&grid.axes()[0][..6], &[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]);
            assert_eq!(grid.axes()[0].last(), Some(&65536.0));
            assert_eq!(grid.axes()[0].len(), 68);
            assert!(BatchedGemmSpec::infeasible_mask(&cfg, &grid).is_empty());
        }
    }

    #[test]
    fn both_layout_backends_use_linear_1d_cache() {
        assert_eq!(
            BatchedGemmSpec::cache_kind(Q_BACKEND),
            CacheKind::Cache1DLinear
        );
        assert_eq!(
            BatchedGemmSpec::cache_kind(V_BACKEND),
            CacheKind::Cache1DLinear
        );
    }

    fn assert_first_payload(
        config: &BatchedGemmKernelConfig,
        backend: &'static str,
        expected_n: u32,
        expected_k: u32,
    ) {
        let grid = BatchedGemmSpec::sweep_grid(config);
        let payloads = BatchedGemmSpec::enumerate(config, &grid, backend);
        assert_eq!(payloads.len(), 68);

        let first = &payloads[0];
        let fields = first.fields();
        let field_names: Vec<&str> = fields.keys().map(String::as_str).collect();

        assert_eq!(
            field_names,
            ["backend", "dtype", "k", "m", "n", "num_batches"]
        );
        assert_eq!(fields.len(), 6);
        assert_eq!(fields.get("backend"), Some(&Value::from(backend)));
        assert_eq!(fields.get("num_batches"), Some(&Value::from(64_u32)));
        assert_eq!(fields.get("m"), Some(&Value::from(1_u32)));
        assert_eq!(fields.get("n"), Some(&Value::from(expected_n)));
        assert_eq!(fields.get("k"), Some(&Value::from(expected_k)));
        assert_eq!(fields.get("dtype"), Some(&Value::from("bf16")));
        assert_eq!(first.backend(), Some(backend));
    }

    #[test]
    fn q_and_v_enumeration_match_the_python_wire_schema() {
        assert_first_payload(&q_config(), Q_BACKEND, 512, 192);
        assert_first_payload(&v_config(), V_BACKEND, 256, 512);
    }

    #[test]
    fn dtype_tags_expose_compute_only_for_both_configs() {
        for cfg in [q_config(), v_config()] {
            assert_eq!(cfg.compute_dtype(), Some(DType::Bf16));
            assert_eq!(cfg.kv_dtype(), None);
        }
    }
}
