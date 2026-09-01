//! FlashInfer fused MLA RoPE, FP8 quantization, and query concatenation.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct MlaRopeQuantizeFp8KernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_heads: Dim,
    pub kv_lora_rank: Dim,
    pub rope_dim: Dim,
    pub max_position: Dim,
    pub is_neox_style: bool,
    #[compute_dtype]
    pub input_dtype: DType,
    #[kv_dtype]
    pub quant_dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct MlaRopeQuantizeFp8KernelInput {
    pub num_tokens: u32,
}

pub struct MlaRopeQuantizeFp8Spec;

impl KernelSpec for MlaRopeQuantizeFp8Spec {
    type Config = MlaRopeQuantizeFp8KernelConfig;
    type Input = MlaRopeQuantizeFp8KernelInput;

    const KIND: KernelKind = "mla_rope_quantize_fp8";

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
                .with("kv_lora_rank", config.kv_lora_rank.get())
                .with("rope_dim", config.rope_dim.get())
                .with("max_position", config.max_position.get())
                .with("is_neox_style", config.is_neox_style)
                .with("input_dtype", config.input_dtype.as_str())
                .with("quant_dtype", config.quant_dtype.as_str())
        })
    }
}

register_kernel!(MlaRopeQuantizeFp8Kernel, MlaRopeQuantizeFp8Spec);

#[cfg(test)]
mod tests {
    use super::{
        MlaRopeQuantizeFp8KernelConfig, MlaRopeQuantizeFp8KernelInput, MlaRopeQuantizeFp8Spec,
    };
    use crate::timing::bridge::DType;
    use crate::timing::cache::CacheKind;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};
    use crate::timing::{SlotInput, SweepCoords};
    use serde_json::Value;

    fn config() -> MlaRopeQuantizeFp8KernelConfig {
        MlaRopeQuantizeFp8KernelConfig {
            backends: vec!["flashinfer"],
            gpu_name: "NVIDIA B200".to_string(),
            num_heads: 16.into(),
            kv_lora_rank: 512.into(),
            rope_dim: 64.into(),
            max_position: 1_048_576.into(),
            is_neox_style: false,
            input_dtype: DType::Bf16,
            quant_dtype: DType::Fp8E4m3,
        }
    }

    #[test]
    fn payload_matches_python_schema_dtype_tags_and_token_coordinates() {
        let config = config();
        let grid = MlaRopeQuantizeFp8Spec::sweep_grid(&config);
        let axis = &grid.axes()[0];
        assert_eq!(axis.len(), 68);
        assert_eq!(axis.first(), Some(&1.0));
        assert_eq!(axis.last(), Some(&65_536.0));
        for anchor in [32.0, 48.0, 2_048.0, 8_192.0] {
            assert!(axis.contains(&anchor), "token grid is missing {anchor}");
        }
        assert!(matches!(
            MlaRopeQuantizeFp8Spec::cache_kind("flashinfer"),
            CacheKind::Cache1DLinear
        ));

        let payloads = MlaRopeQuantizeFp8Spec::enumerate(&config, &grid, "flashinfer");
        assert_eq!(payloads.last().unwrap().fields()["num_tokens"], 65_536);
        let payload = &payloads[0];
        let fields = payload.fields();
        assert_eq!(fields.len(), 9);
        assert_eq!(fields["backend"], Value::from("flashinfer"));
        assert_eq!(fields["num_tokens"], Value::from(1_u32));
        assert_eq!(fields["num_heads"], Value::from(16_u32));
        assert_eq!(fields["kv_lora_rank"], Value::from(512_u32));
        assert_eq!(fields["rope_dim"], Value::from(64_u32));
        assert_eq!(fields["max_position"], Value::from(1_048_576_u32));
        assert_eq!(fields["is_neox_style"], Value::from(false));
        assert_eq!(fields["input_dtype"], Value::from("bf16"));
        assert_eq!(fields["quant_dtype"], Value::from("fp8_e4m3"));
        assert_eq!(config.compute_dtype(), Some(DType::Bf16));
        assert_eq!(config.kv_dtype(), Some(DType::Fp8E4m3));

        let input = MlaRopeQuantizeFp8KernelInput { num_tokens: 2_048 };
        assert_eq!(&*input.coords(), &[2_048.0]);
        assert_eq!(
            MlaRopeQuantizeFp8KernelInput::coord_field_names(),
            &["num_tokens"]
        );
        let slot: SlotInput = input.into();
        assert!(matches!(&slot, SlotInput::MlaRopeQuantizeFp8(_)));
        assert_eq!(
            serde_json::to_value(slot).unwrap(),
            serde_json::json!({"num_tokens": 2_048})
        );
    }
}
