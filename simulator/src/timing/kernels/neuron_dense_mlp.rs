//! Verified public fused MLP specializations on Trainium2 LNC2.
//!
//! The full-width NKI backend is restricted to one token. The stock vLLM
//! TP4 shard has three verified compiled shapes, represented by categorical
//! coordinates so unverified token counts cannot interpolate or round to them.
//! These rank-local timings exclude collectives and do not prove additivity
//! inside a compiled full-model forward.

use crate::timing::bridge::{ArgsPayload, DType, KernelKind, de_backends};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{KernelSpec, register_kernel};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NeuronDenseMlpKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub hidden: Dim,
    pub intermediate: Dim,
    #[compute_dtype]
    pub dtype: DType,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct NeuronDenseMlpKernelInput {
    pub m: u32,
}

impl SweepCoords for NeuronDenseMlpKernelInput {
    fn coords(&self) -> Coords {
        panic!("Neuron MLP coordinates require the backend Config")
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["mlp_variant"]
    }
}

const VLLM_TOKEN_ROWS: [u32; 3] = [1, 16, 512];

pub struct NeuronDenseMlpSpec;

impl KernelSpec for NeuronDenseMlpSpec {
    type Config = NeuronDenseMlpKernelConfig;
    type Input = NeuronDenseMlpKernelInput;
    const KIND: KernelKind = "neuron_dense_mlp";

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        anyhow::ensure!(
            config.hidden == 4096 && config.dtype == DType::Bf16,
            "Neuron MLP supports only H4096 BF16"
        );
        anyhow::ensure!(
            matches!(
                config.gpu_name.as_str(),
                "AWS Trainium2 LNC2" | "Trainium2-LNC2"
            ),
            "Neuron timing requires a Trainium2 LNC2 target"
        );
        let intermediate = match config.backends.as_slice() {
            ["nki_library"] => 14336,
            ["vllm_neuron"] => 3584,
            _ => {
                anyhow::bail!("Neuron MLP requires exactly one backend: nki_library or vllm_neuron")
            }
        };
        anyhow::ensure!(
            config.intermediate == intermediate,
            "Neuron MLP backend {} requires intermediate={intermediate}",
            config.backends[0]
        );
        Ok(())
    }

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        Self::validate_config(config).expect("invalid Neuron MLP config");
        if config.backends == ["vllm_neuron"] {
            // Three feasible categorical cells, not a token interpolation grid.
            SweepGrid::new(vec![Axis::values(0..VLLM_TOKEN_ROWS.len() as u32)])
        } else {
            SweepGrid::new(vec![Axis::values([1u32])])
        }
    }

    fn cache_coords(config: &Self::Config, input: &Self::Input) -> Coords {
        if config.backends == ["vllm_neuron"] {
            let index = VLLM_TOKEN_ROWS
                .binary_search(&input.m)
                .expect("stock vLLM Neuron MLP supports only verified m=1,16,512");
            Coords::new([index as f64])
        } else {
            assert_eq!(input.m, 1, "NKI fused MLP supports only m=1");
            Coords::new([1.])
        }
    }

    fn cache_kind(backend: &'static str) -> CacheKind {
        match backend {
            "vllm_neuron" => CacheKind::Cache1DDirect,
            "nki_library" => CacheKind::Cache1DLinear,
            _ => panic!("unsupported Neuron MLP backend: {backend}"),
        }
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        Self::validate_config(config).expect("invalid Neuron MLP config");
        assert_eq!(
            config.backends,
            [backend],
            "MLP enumeration backend must match Config"
        );
        grid.expand_1d(|coordinate| {
            let m = if backend == "vllm_neuron" {
                assert!(coordinate.is_finite() && coordinate >= 0. && coordinate.fract() == 0.);
                *VLLM_TOKEN_ROWS
                    .get(coordinate as usize)
                    .expect("invalid MLP variant index")
            } else {
                assert_eq!(coordinate, 1., "NKI MLP grid supports only m=1");
                1
            };
            ArgsPayload::new()
                .with("backend", backend)
                .with("m", m)
                .with("hidden", config.hidden.get())
                .with("intermediate", config.intermediate.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(NeuronDenseMlpKernel, NeuronDenseMlpSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::kernels::engine::KernelSpec;

    fn config() -> NeuronDenseMlpKernelConfig {
        NeuronDenseMlpKernelConfig {
            backends: vec!["nki_library"],
            gpu_name: "AWS Trainium2 LNC2".into(),
            hidden: 4096.into(),
            intermediate: 14336.into(),
            dtype: DType::Bf16,
        }
    }

    #[test]
    fn mlp_singleton_payload_matches_the_only_validated_public_shape() {
        let config = config();
        NeuronDenseMlpSpec::validate_config(&config).unwrap();
        let grid = NeuronDenseMlpSpec::sweep_grid(&config);
        assert_eq!(grid.axes()[0], [1.]);
        let payloads = NeuronDenseMlpSpec::enumerate(&config, &grid, "nki_library");
        assert_eq!(payloads.len(), 1);
        assert_eq!(
            serde_json::to_value(payloads[0].fields()).unwrap(),
            serde_json::json!({
                "backend":"nki_library", "m":1, "hidden":4096, "intermediate":14336, "dtype":"bf16",
            })
        );
    }

    #[test]
    fn mlp_rejects_shapes_without_native_correctness_and_memory_proof() {
        let mut config = config();
        config.intermediate = 8192.into();
        assert!(NeuronDenseMlpSpec::validate_config(&config).is_err());
        config.intermediate = 14336.into();
        config.dtype = DType::Fp16;
        assert!(NeuronDenseMlpSpec::validate_config(&config).is_err());
    }

    #[test]
    #[should_panic(expected = "supports only m=1")]
    fn mlp_cannot_extrapolate_to_a_vendor_memory_failure_shape() {
        NeuronDenseMlpSpec::cache_coords(&config(), &NeuronDenseMlpKernelInput { m: 128 });
    }
    #[test]
    fn stock_mlp_exact_variants_forward_physical_rows_and_reject_unverified_inputs() {
        let mut config = config();
        config.backends = vec!["vllm_neuron"];
        config.intermediate = 3584.into();
        let grid = NeuronDenseMlpSpec::sweep_grid(&config);
        assert_eq!(grid.axes()[0], [0., 1., 2.]);
        let payloads = NeuronDenseMlpSpec::enumerate(&config, &grid, "vllm_neuron");
        assert_eq!(payloads.len(), 3);
        for (index, (m, payload)) in VLLM_TOKEN_ROWS.into_iter().zip(payloads).enumerate() {
            assert_eq!(
                &*NeuronDenseMlpSpec::cache_coords(&config, &NeuronDenseMlpKernelInput { m }),
                &[index as f64]
            );
            assert_eq!(
                serde_json::to_value(payload.fields()).unwrap(),
                serde_json::json!({
                    "backend":"vllm_neuron", "m":m, "hidden":4096, "intermediate":3584, "dtype":"bf16"
                })
            );
        }
        for m in [0, 2, 15, 17, 128, 511, 513] {
            assert!(
                std::panic::catch_unwind(|| NeuronDenseMlpSpec::cache_coords(
                    &config,
                    &NeuronDenseMlpKernelInput { m }
                ))
                .is_err()
            );
        }
    }

    #[test]
    fn mlp_rejects_backend_dimension_mismatch_and_mixed_dispatch() {
        let mut config = config();
        config.backends = vec!["vllm_neuron"];
        assert!(NeuronDenseMlpSpec::validate_config(&config).is_err());
        config.intermediate = 3584.into();
        NeuronDenseMlpSpec::validate_config(&config).unwrap();
        for backends in [
            vec!["nki_library"],
            vec![],
            vec!["unknown"],
            vec!["vllm_neuron", "nki_library"],
        ] {
            config.backends = backends;
            assert!(NeuronDenseMlpSpec::validate_config(&config).is_err());
        }
    }
}
