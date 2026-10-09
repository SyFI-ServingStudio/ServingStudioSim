//! One compiled NxDI Llama3.1 8B decoder and its aliased KV update on LNC2.
//!
//! Static phase and cache capacity select different production executables.
//! Decode measures one token; prefix-free prefill measures eight token counts.
//! KV capacity is an allocation identity, independent of runtime query rows.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuronLlamaDecoderPhase {
    Prefill,
    Decode,
}

impl NeuronLlamaDecoderPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NeuronLlamaDecoderKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub phase: NeuronLlamaDecoderPhase,
    pub batch: Dim,
    pub kv_capacity: Dim,
    pub hidden: Dim,
    pub intermediate: Dim,
    pub q_heads: Dim,
    pub kv_heads: Dim,
    pub head_dim: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct NeuronLlamaDecoderKernelInput {
    pub q_tokens: u32,
}

pub struct NeuronLlamaDecoderSpec;

impl KernelSpec for NeuronLlamaDecoderSpec {
    type Config = NeuronLlamaDecoderKernelConfig;
    type Input = NeuronLlamaDecoderKernelInput;

    const KIND: KernelKind = "neuron_llama_decoder";

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        anyhow::ensure!(
            config.dtype == DType::Bf16,
            "Neuron Llama decoder supports BF16 only"
        );
        anyhow::ensure!(
            matches!(
                config.gpu_name.as_str(),
                "AWS Trainium2 LNC2" | "Trainium2-LNC2"
            ),
            "Neuron timing requires a Trainium2 LNC2 target"
        );
        anyhow::ensure!(
            config.batch == 1 && config.kv_capacity == 512,
            "Neuron decoder supports batch1 and KV capacity512 only"
        );
        anyhow::ensure!(
            config.hidden == 4096
                && config.intermediate == 14336
                && config.q_heads == 32
                && config.kv_heads == 8
                && config.head_dim == 128,
            "Neuron decoder requires Llama3.1 8B dimensions H4096/I14336/Q32/KV8/D128"
        );
        Ok(())
    }

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        let axis = match config.phase {
            NeuronLlamaDecoderPhase::Decode => Axis::values([1u32]),
            NeuronLlamaDecoderPhase::Prefill => Axis::pow2(0, 7),
        };
        SweepGrid::new(vec![axis])
    }

    fn cache_coords(config: &Self::Config, input: &Self::Input) -> Coords {
        match config.phase {
            NeuronLlamaDecoderPhase::Decode => assert_eq!(
                input.q_tokens, 1,
                "Neuron decoder decode requires one query token"
            ),
            NeuronLlamaDecoderPhase::Prefill => assert!(
                (1..=128).contains(&input.q_tokens),
                "Neuron decoder prefill supports1..128 query tokens"
            ),
        }
        input.coords()
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|q_tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("phase", config.phase.as_str())
                .with("batch", config.batch.get())
                .with("q_tokens", q_tokens as u32)
                .with("kv_capacity", config.kv_capacity.get())
                .with("hidden", config.hidden.get())
                .with("intermediate", config.intermediate.get())
                .with("q_heads", config.q_heads.get())
                .with("kv_heads", config.kv_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(NeuronLlamaDecoderKernel, NeuronLlamaDecoderSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};

    fn config() -> NeuronLlamaDecoderKernelConfig {
        NeuronLlamaDecoderKernelConfig {
            backends: vec!["nxdi_compiler"],
            gpu_name: "AWS Trainium2 LNC2".into(),
            phase: NeuronLlamaDecoderPhase::Decode,
            batch: 1.into(),
            kv_capacity: 512.into(),
            hidden: 4096.into(),
            intermediate: 14336.into(),
            q_heads: 32.into(),
            kv_heads: 8.into(),
            head_dim: 128.into(),
            dtype: DType::Bf16,
        }
    }

    #[test]
    fn phases_keep_distinct_compile_identity_and_bounded_query_grids() {
        let mut config = config();
        assert_eq!(config.identity()["phase"], "decode");
        assert_eq!(NeuronLlamaDecoderSpec::sweep_grid(&config).axes()[0], [1.]);
        config.phase = NeuronLlamaDecoderPhase::Prefill;
        assert_eq!(config.identity()["phase"], "prefill");
        assert_eq!(
            NeuronLlamaDecoderSpec::sweep_grid(&config).axes()[0],
            [1., 2., 4., 8., 16., 32., 64., 128.]
        );
        assert!(
            serde_json::from_value::<NeuronLlamaDecoderPhase>(serde_json::json!("invalid"))
                .is_err()
        );
    }

    #[test]
    fn decoder_payload_preserves_physical_allocation_and_every_python_field() {
        let config = config();
        let payload = NeuronLlamaDecoderSpec::enumerate(
            &config,
            &NeuronLlamaDecoderSpec::sweep_grid(&config),
            "nxdi_compiler",
        )
        .remove(0);
        assert_eq!(
            serde_json::to_value(payload.fields()).unwrap(),
            serde_json::json!({
                "backend": "nxdi_compiler", "phase":"decode", "batch":1,
                "q_tokens":1, "kv_capacity":512, "hidden":4096, "intermediate":14336,
                "q_heads":32, "kv_heads":8, "head_dim":128, "dtype":"bf16",
            })
        );
        assert_eq!(
            serde_json::to_value(NeuronLlamaDecoderKernelInput { q_tokens: 16 }).unwrap(),
            serde_json::json!({"q_tokens":16})
        );
    }

    #[test]
    fn decoder_rejects_unvalidated_batch_capacity_model_or_dtype() {
        let config = config();
        NeuronLlamaDecoderSpec::validate_config(&config).unwrap();
        let mut invalid = config.clone();
        invalid.batch = 2.into();
        assert!(NeuronLlamaDecoderSpec::validate_config(&invalid).is_err());
        invalid = config.clone();
        invalid.kv_capacity = 1024.into();
        assert!(NeuronLlamaDecoderSpec::validate_config(&invalid).is_err());
        invalid = config.clone();
        invalid.intermediate = 8192.into();
        assert!(NeuronLlamaDecoderSpec::validate_config(&invalid).is_err());
        invalid = config;
        invalid.dtype = DType::Fp16;
        assert!(NeuronLlamaDecoderSpec::validate_config(&invalid).is_err());
    }

    #[test]
    fn decoder_cannot_extrapolate_across_invalid_phase_specific_inputs() {
        let mut config = config();
        for q_tokens in [0, 2] {
            assert!(
                std::panic::catch_unwind(|| NeuronLlamaDecoderSpec::cache_coords(
                    &config,
                    &NeuronLlamaDecoderKernelInput { q_tokens },
                ))
                .is_err()
            );
        }
        config.phase = NeuronLlamaDecoderPhase::Prefill;
        for q_tokens in [0, 129] {
            assert!(
                std::panic::catch_unwind(|| NeuronLlamaDecoderSpec::cache_coords(
                    &config,
                    &NeuronLlamaDecoderKernelInput { q_tokens },
                ))
                .is_err()
            );
        }
        assert_eq!(
            &*NeuronLlamaDecoderSpec::cache_coords(
                &config,
                &NeuronLlamaDecoderKernelInput { q_tokens: 16 }
            ),
            &[16.]
        );
    }
}
