//! SGLang fused DSA indexer-query RoPE and FP8 quantization.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DsaIndexerQRopeQuantKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: Dim,
    pub head_dim: Dim,
    pub rope_dim: Dim,
    pub rope_layout: String,
    pub hadamard: bool,
    #[compute_dtype]
    pub input_dtype: DType,
    pub q_output_dtype: DType,
    pub weight_output_dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct DsaIndexerQRopeQuantKernelInput {
    pub num_tokens: u32,
}

pub struct DsaIndexerQRopeQuantSpec;

impl KernelSpec for DsaIndexerQRopeQuantSpec {
    type Config = DsaIndexerQRopeQuantKernelConfig;
    type Input = DsaIndexerQRopeQuantKernelInput;

    const KIND: KernelKind = "dsa_indexer_q_rope_quant";

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::chain([
            Axis::values([1, 4, 8, 16, 32, 48]),
            Axis::token_axis(),
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
        grid.expand_1d(|num_tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", num_tokens as u32)
                .with("num_heads", config.num_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("rope_dim", config.rope_dim.get())
                .with("rope_layout", config.rope_layout.clone())
                .with("hadamard", config.hadamard)
                .with("input_dtype", config.input_dtype.as_str())
                .with("q_output_dtype", config.q_output_dtype.as_str())
                .with("weight_output_dtype", config.weight_output_dtype.as_str())
        })
    }
}

register_kernel!(DsaIndexerQRopeQuantKernel, DsaIndexerQRopeQuantSpec);

#[cfg(test)]
mod tests {
    use super::{
        DsaIndexerQRopeQuantKernelConfig, DsaIndexerQRopeQuantKernelInput, DsaIndexerQRopeQuantSpec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config() -> DsaIndexerQRopeQuantKernelConfig {
        DsaIndexerQRopeQuantKernelConfig {
            backends: vec!["sglang_cuda"],
            gpu_name: "NVIDIA B200".to_string(),
            num_heads: 32.into(),
            head_dim: 128.into(),
            rope_dim: 64.into(),
            rope_layout: "rope_first".to_string(),
            hadamard: false,
            input_dtype: DType::Bf16,
            q_output_dtype: DType::Fp8E4m3,
            weight_output_dtype: DType::Fp32,
        }
    }

    #[test]
    fn payload_matches_python_schema_and_token_coordinates() {
        let config = config();
        let grid = DsaIndexerQRopeQuantSpec::sweep_grid(&config);
        let axis = &grid.axes()[0];
        assert_eq!(axis.len(), 68);
        assert_eq!(axis.first(), Some(&1.0));
        assert_eq!(axis.last(), Some(&65_536.0));
        for anchor in [32.0, 48.0, 2_048.0, 8_192.0] {
            assert!(axis.contains(&anchor), "token grid is missing {anchor}");
        }
        assert!(matches!(
            DsaIndexerQRopeQuantSpec::cache_kind("sglang_cuda"),
            CacheKind::Cache1DLinear
        ));

        let payloads = DsaIndexerQRopeQuantSpec::enumerate(&config, &grid, "sglang_cuda");
        assert_eq!(payloads.last().unwrap().fields()["num_tokens"], 65_536);
        let payload = &payloads[0];
        let fields = payload.fields();
        assert_eq!(fields.len(), 10);
        assert_eq!(fields["backend"], Value::from("sglang_cuda"));
        assert_eq!(fields["num_tokens"], Value::from(1_u32));
        assert_eq!(fields["num_heads"], Value::from(32_u32));
        assert_eq!(fields["head_dim"], Value::from(128_u32));
        assert_eq!(fields["rope_dim"], Value::from(64_u32));
        assert_eq!(fields["rope_layout"], Value::from("rope_first"));
        assert_eq!(fields["hadamard"], Value::from(false));
        assert_eq!(fields["input_dtype"], Value::from("bf16"));
        assert_eq!(fields["q_output_dtype"], Value::from("fp8_e4m3"));
        assert_eq!(fields["weight_output_dtype"], Value::from("fp32"));
        assert_eq!(config.compute_dtype(), Some(DType::Bf16));
        assert_eq!(config.kv_dtype(), None);

        let input = DsaIndexerQRopeQuantKernelInput { num_tokens: 48 };
        assert_eq!(&*input.coords(), &[48.0]);
        assert_eq!(
            DsaIndexerQRopeQuantKernelInput::coord_field_names(),
            &["num_tokens"]
        );
        let slot: SlotInput = input.into();
        assert!(matches!(&slot, SlotInput::DsaIndexerQRopeQuant(_)));
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_tokens": 48})
        );
    }

    #[test]
    fn compile_time_instantiations_are_config_identity() {
        let base = config();
        let mut different_layout = base.clone();
        different_layout.rope_layout = "rope_last".to_string();
        assert_ne!(base, different_layout);

        let mut with_hadamard = config();
        with_hadamard.hadamard = true;
        assert_ne!(config(), with_hadamard);
    }
}
