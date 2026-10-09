//! Compiled token embedding on one Trainium2 LNC2 unit.
//!
//! The vocabulary allocation affects the executable and belongs in Config;
//! runtime token rows are the single cache axis. Eight points bound compiler
//! work for the initial one-request, 1..128-token model path.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NeuronEmbeddingKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden: Dim,
    pub vocab: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, SweepCoords, serde::Serialize, serde::Deserialize)]
pub struct NeuronEmbeddingKernelInput {
    pub num_tokens: u32,
}

pub struct NeuronEmbeddingSpec;

impl KernelSpec for NeuronEmbeddingSpec {
    type Config = NeuronEmbeddingKernelConfig;
    type Input = NeuronEmbeddingKernelInput;

    const KIND: KernelKind = "neuron_embedding";

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        anyhow::ensure!(
            config.dtype == DType::Bf16,
            "Neuron embedding supports BF16 only"
        );
        anyhow::ensure!(
            config.hidden.get() > 0 && config.vocab.get() > 0,
            "embedding dimensions must be positive"
        );
        anyhow::ensure!(
            matches!(
                config.gpu_name.as_str(),
                "AWS Trainium2 LNC2" | "Trainium2-LNC2"
            ),
            "Neuron timing requires a Trainium2 LNC2 target"
        );
        Ok(())
    }

    fn sweep_grid(_config: &Self::Config) -> SweepGrid {
        SweepGrid::new(vec![Axis::pow2(0, 7)])
    }

    fn cache_coords(_config: &Self::Config, input: &Self::Input) -> Coords {
        assert!(
            (1..=128).contains(&input.num_tokens),
            "Neuron embedding supports1..128 token rows"
        );
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
        grid.expand_1d(|num_tokens| {
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_tokens", num_tokens as u32)
                .with("hidden", config.hidden.get())
                .with("vocab", config.vocab.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(NeuronEmbeddingKernel, NeuronEmbeddingSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::kernels::engine::{KernelConfig, KernelSpec};

    fn config() -> NeuronEmbeddingKernelConfig {
        NeuronEmbeddingKernelConfig {
            backends: vec!["neuron_torch"],
            gpu_name: "AWS Trainium2 LNC2".into(),
            hidden: 4096.into(),
            vocab: 128256.into(),
            dtype: DType::Bf16,
        }
    }

    #[test]
    fn embedding_payload_keeps_vocabulary_allocation_and_token_axis() {
        let config = config();
        let grid = NeuronEmbeddingSpec::sweep_grid(&config);
        assert_eq!(grid.axes()[0], [1., 2., 4., 8., 16., 32., 64., 128.]);
        let payloads = NeuronEmbeddingSpec::enumerate(&config, &grid, "neuron_torch");
        assert_eq!(payloads.len(), 8);
        assert_eq!(
            serde_json::to_value(payloads[7].fields()).unwrap(),
            serde_json::json!({
                "backend": "neuron_torch", "num_tokens": 128,
                "hidden": 4096, "vocab": 128256, "dtype": "bf16",
            })
        );
        assert_eq!(
            &*NeuronEmbeddingKernelInput { num_tokens: 3 }.coords(),
            &[3.]
        );
        assert_eq!(config.identity()["vocab"], 128256);
    }

    #[test]
    fn embedding_rejects_unvalidated_dtype_and_empty_table() {
        let mut config = config();
        NeuronEmbeddingSpec::validate_config(&config).unwrap();
        config.dtype = DType::Fp16;
        assert!(NeuronEmbeddingSpec::validate_config(&config).is_err());
        config.dtype = DType::Bf16;
        config.vocab = 0.into();
        assert!(NeuronEmbeddingSpec::validate_config(&config).is_err());
    }

    #[test]
    fn embedding_rejects_empty_and_out_of_domain_runtime_inputs() {
        for num_tokens in [0, 129] {
            assert!(
                std::panic::catch_unwind(|| NeuronEmbeddingSpec::cache_coords(
                    &config(),
                    &NeuronEmbeddingKernelInput { num_tokens },
                ))
                .is_err()
            );
        }
    }
}
