//! Stock TP4/LNC2 full-model forward, including all layers and sampling.
//!
//! Coordinates are exact entries in a compiled-variant inventory, not token
//! lengths. There is no interpolation across phases or compiler buckets. Input
//! lowering into a configured bucket belongs to the consumer, not this leaf.
//! The direct cache may warn about nonmonotonic time across categorical entries
//! (prefill then decode); it retains those measured cells unchanged.

use crate::timing::bridge::{ArgsPayload, DType, KernelKind, de_backends};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{KernelSpec, register_kernel};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuronLlamaForwardPhase {
    Prefill,
    Decode,
}

impl NeuronLlamaForwardPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Prefill => "prefill",
            Self::Decode => "decode",
        }
    }
}

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NeuronLlamaForwardKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub max_model_len: Dim,
    pub kv_blocks: Dim,
    pub block_size: Dim,
    pub tp_size: Dim,
    #[compute_dtype]
    #[kv_dtype]
    pub dtype: DType,
    /// Exact compiled variants to profile/cache; not an interpolation grid.
    pub decode_buckets: Vec<u32>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NeuronLlamaForwardKernelInput {
    pub phase: NeuronLlamaForwardPhase,
    pub token_bucket: u32,
}

impl SweepCoords for NeuronLlamaForwardKernelInput {
    fn coords(&self) -> Coords {
        panic!("Neuron full-forward coordinates require the Config variant inventory")
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["compiled_variant"]
    }
}

pub struct NeuronLlamaForwardSpec;

impl KernelSpec for NeuronLlamaForwardSpec {
    type Config = NeuronLlamaForwardKernelConfig;
    type Input = NeuronLlamaForwardKernelInput;

    const KIND: KernelKind = "neuron_llama_forward";

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        anyhow::ensure!(
            config.backends == ["vllm_neuron"],
            "stock Neuron forward requires the pinned vllm_neuron backend"
        );
        anyhow::ensure!(
            matches!(
                config.gpu_name.as_str(),
                "AWS Trainium2 LNC2" | "Trainium2-LNC2"
            ),
            "stock Neuron forward requires a Trainium2 LNC2 target"
        );
        anyhow::ensure!(
            config.dtype == DType::Bf16,
            "stock Neuron forward requires BF16"
        );
        anyhow::ensure!(
            config.kv_blocks == 6782 && config.block_size == 32 && config.tp_size == 4,
            "stock Neuron forward requires 6782 KV blocks, page size32 and TP4"
        );
        let largest_bucket = match config.max_model_len.get() {
            128 | 2048 => 128,
            512 => 512,
            _ => anyhow::bail!("stock Neuron context must be128,512 or2048"),
        };
        anyhow::ensure!(
            !config.decode_buckets.is_empty()
                && config
                    .decode_buckets
                    .iter()
                    .all(|&b| b.is_power_of_two() && b <= largest_bucket)
                && config
                    .decode_buckets
                    .windows(2)
                    .all(|pair| pair[0] < pair[1]),
            "decode buckets must be sorted unique powers of two within the verified context limit"
        );
        Ok(())
    }

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        // At most11 feasible entries, including the one prefill executable.
        SweepGrid::new(vec![Axis::values(0..=config.decode_buckets.len() as u32)])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DDirect
    }

    fn cache_coords(config: &Self::Config, input: &Self::Input) -> Coords {
        let index = match input.phase {
            NeuronLlamaForwardPhase::Prefill => {
                assert_eq!(
                    input.token_bucket,
                    config.max_model_len.get(),
                    "prefill input must name the configured compiled context bucket"
                );
                0
            }
            NeuronLlamaForwardPhase::Decode => {
                config
                    .decode_buckets
                    .binary_search(&input.token_bucket)
                    .expect("decode input must name a configured compiled bucket")
                    + 1
            }
        };
        Coords::new([index as f64])
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_1d(|coordinate| {
            assert!(
                coordinate.is_finite() && coordinate >= 0.0 && coordinate.fract() == 0.0,
                "compiled variant index must be a nonnegative integer"
            );
            let index = coordinate as usize;
            let (phase, token_bucket) = if index == 0 {
                (NeuronLlamaForwardPhase::Prefill, config.max_model_len.get())
            } else {
                (
                    NeuronLlamaForwardPhase::Decode,
                    *config
                        .decode_buckets
                        .get(index - 1)
                        .expect("compiled variant index exceeds inventory"),
                )
            };
            ArgsPayload::new()
                .with("backend", backend)
                .with("phase", phase.as_str())
                .with("token_bucket", token_bucket)
                .with("max_model_len", config.max_model_len.get())
                .with("kv_blocks", config.kv_blocks.get())
                .with("block_size", config.block_size.get())
                .with("tp_size", config.tp_size.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(NeuronLlamaForwardKernel, NeuronLlamaForwardSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::kernels::engine::KernelConfig;

    fn config() -> NeuronLlamaForwardKernelConfig {
        NeuronLlamaForwardKernelConfig {
            backends: vec!["vllm_neuron"],
            gpu_name: "AWS Trainium2 LNC2".into(),
            max_model_len: 512.into(),
            kv_blocks: 6782.into(),
            block_size: 32.into(),
            tp_size: 4.into(),
            dtype: DType::Bf16,
            decode_buckets: vec![1, 16],
        }
    }

    #[test]
    fn exact_variants_do_not_interpolate_across_phase_or_missing_bucket() {
        let config = config();
        for (phase, token_bucket, expected) in [
            (NeuronLlamaForwardPhase::Prefill, 512, 0.),
            (NeuronLlamaForwardPhase::Decode, 1, 1.),
            (NeuronLlamaForwardPhase::Decode, 16, 2.),
        ] {
            assert_eq!(
                &*NeuronLlamaForwardSpec::cache_coords(
                    &config,
                    &NeuronLlamaForwardKernelInput {
                        phase,
                        token_bucket
                    }
                ),
                &[expected]
            );
        }
        for input in [
            NeuronLlamaForwardKernelInput {
                phase: NeuronLlamaForwardPhase::Decode,
                token_bucket: 2,
            },
            NeuronLlamaForwardKernelInput {
                phase: NeuronLlamaForwardPhase::Decode,
                token_bucket: 17,
            },
            NeuronLlamaForwardKernelInput {
                phase: NeuronLlamaForwardPhase::Prefill,
                token_bucket: 504,
            },
        ] {
            assert!(
                std::panic::catch_unwind(|| NeuronLlamaForwardSpec::cache_coords(&config, &input))
                    .is_err()
            );
        }
    }

    #[test]
    fn enumeration_preserves_only_the_seven_python_fields_and_backend() {
        let config = config();
        let grid = NeuronLlamaForwardSpec::sweep_grid(&config);
        assert_eq!(grid.axes()[0], [0., 1., 2.]);
        let rows = NeuronLlamaForwardSpec::enumerate(&config, &grid, "vllm_neuron");
        for (row, (phase, bucket)) in
            rows.iter()
                .zip([("prefill", 512), ("decode", 1), ("decode", 16)])
        {
            assert_eq!(
                serde_json::to_value(row.fields()).unwrap(),
                serde_json::json!({
                    "backend":"vllm_neuron", "phase":phase, "token_bucket":bucket,
                    "max_model_len":512, "kv_blocks":6782, "block_size":32, "tp_size":4, "dtype":"bf16"
                })
            );
        }
        assert_eq!(config.compute_dtype(), Some(DType::Bf16));
        assert_eq!(config.kv_dtype(), Some(DType::Bf16));
    }

    #[test]
    fn config_rejects_wrong_runtime_layout_and_ambiguous_inventory() {
        let config = config();
        NeuronLlamaForwardSpec::validate_config(&config).unwrap();
        for buckets in [
            vec![],
            vec![0],
            vec![3],
            vec![16, 1],
            vec![1, 1],
            vec![1024],
        ] {
            let mut bad = config.clone();
            bad.decode_buckets = buckets;
            assert!(NeuronLlamaForwardSpec::validate_config(&bad).is_err());
        }
        let mut bad = config.clone();
        bad.tp_size = 1.into();
        assert!(NeuronLlamaForwardSpec::validate_config(&bad).is_err());
        bad = config.clone();
        bad.max_model_len = 128.into();
        bad.decode_buckets = vec![256];
        assert!(NeuronLlamaForwardSpec::validate_config(&bad).is_err());
        bad = config.clone();
        bad.max_model_len = 4096.into();
        assert!(NeuronLlamaForwardSpec::validate_config(&bad).is_err());
        bad = config.clone();
        bad.backends = vec!["nxdi_compiler"];
        assert!(NeuronLlamaForwardSpec::validate_config(&bad).is_err());
        bad = config;
        bad.dtype = DType::Fp16;
        assert!(NeuronLlamaForwardSpec::validate_config(&bad).is_err());
    }
}
