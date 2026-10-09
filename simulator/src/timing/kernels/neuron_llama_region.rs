//! One region of the stock TP4/LNC2 Llama forward compiled as two executables.
//!
//! The stock fullgraph is split at the public `LlamaModel.forward` return:
//! `model` holds embedding, all 32 layers, the final norm and every KV update;
//! `head` holds row selection, the vocabulary projection, logit collectives and
//! greedy sampling. Each region runs once per forward, model before head.
//! Coordinates are the same exact compiled-variant inventory as
//! `neuron_llama_forward`; there is no interpolation across phases or buckets.

use crate::timing::bridge::{ArgsPayload, DType, KernelKind, de_backends};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{KernelSpec, register_kernel};
use crate::timing::kernels::neuron_llama_forward::{
    NeuronLlamaForwardKernelConfig, NeuronLlamaForwardKernelInput, NeuronLlamaForwardPhase,
    NeuronLlamaForwardSpec,
};
use crate::timing::sweep::SweepGrid;
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

#[derive(Hash, PartialEq, Eq, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuronLlamaRegion {
    Model,
    Head,
}

impl NeuronLlamaRegion {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Model => "model",
            Self::Head => "head",
        }
    }
}

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NeuronLlamaRegionKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub region: NeuronLlamaRegion,
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

impl NeuronLlamaRegionKernelConfig {
    /// The unsplit forward with the same compiled inventory; it owns the
    /// shared variant indexing so both kinds address identical executables.
    fn whole_forward(&self) -> NeuronLlamaForwardKernelConfig {
        NeuronLlamaForwardKernelConfig {
            backends: vec!["vllm_neuron"],
            gpu_name: self.gpu_name.clone(),
            max_model_len: self.max_model_len.clone(),
            kv_blocks: self.kv_blocks.clone(),
            block_size: self.block_size.clone(),
            tp_size: self.tp_size.clone(),
            dtype: self.dtype,
            decode_buckets: self.decode_buckets.clone(),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NeuronLlamaRegionKernelInput {
    pub phase: NeuronLlamaForwardPhase,
    pub token_bucket: u32,
}

impl NeuronLlamaRegionKernelInput {
    fn as_forward(&self) -> NeuronLlamaForwardKernelInput {
        NeuronLlamaForwardKernelInput {
            phase: self.phase,
            token_bucket: self.token_bucket,
        }
    }
}

impl SweepCoords for NeuronLlamaRegionKernelInput {
    fn coords(&self) -> Coords {
        panic!("Neuron region coordinates require the Config variant inventory")
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["compiled_variant"]
    }
}

pub struct NeuronLlamaRegionSpec;

impl KernelSpec for NeuronLlamaRegionSpec {
    type Config = NeuronLlamaRegionKernelConfig;
    type Input = NeuronLlamaRegionKernelInput;

    const KIND: KernelKind = "neuron_llama_region";

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        anyhow::ensure!(
            config.backends == ["vllm_neuron_fx_regions"],
            "stock Neuron regions require the vllm_neuron_fx_regions backend"
        );
        NeuronLlamaForwardSpec::validate_config(&config.whole_forward())?;
        // Python verifies only the C512 [1,16] split executables.
        anyhow::ensure!(
            config.max_model_len == 512
                && config.decode_buckets.iter().all(|b| [1, 16].contains(b)),
            "stock Neuron regions are verified only at context512 with decode buckets 1 and16"
        );
        Ok(())
    }

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        NeuronLlamaForwardSpec::sweep_grid(&config.whole_forward())
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DDirect
    }

    fn cache_coords(config: &Self::Config, input: &Self::Input) -> Coords {
        NeuronLlamaForwardSpec::cache_coords(&config.whole_forward(), &input.as_forward())
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        NeuronLlamaForwardSpec::enumerate(&config.whole_forward(), grid, backend)
            .into_iter()
            .map(|row| row.with("region", config.region.as_str()))
            .collect()
    }
}

register_kernel!(NeuronLlamaRegionKernel, NeuronLlamaRegionSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::kernels::engine::KernelConfig;

    fn config(region: NeuronLlamaRegion) -> NeuronLlamaRegionKernelConfig {
        NeuronLlamaRegionKernelConfig {
            backends: vec!["vllm_neuron_fx_regions"],
            gpu_name: "AWS Trainium2 LNC2".into(),
            region,
            max_model_len: 512.into(),
            kv_blocks: 6782.into(),
            block_size: 32.into(),
            tp_size: 4.into(),
            dtype: DType::Bf16,
            decode_buckets: vec![1, 16],
        }
    }

    #[test]
    fn enumeration_adds_only_region_to_the_forward_fields() {
        for region in [NeuronLlamaRegion::Model, NeuronLlamaRegion::Head] {
            let config = config(region);
            NeuronLlamaRegionSpec::validate_config(&config).unwrap();
            let grid = NeuronLlamaRegionSpec::sweep_grid(&config);
            assert_eq!(grid.axes()[0], [0., 1., 2.]);
            let rows = NeuronLlamaRegionSpec::enumerate(&config, &grid, "vllm_neuron_fx_regions");
            for (row, (phase, bucket)) in
                rows.iter()
                    .zip([("prefill", 512), ("decode", 1), ("decode", 16)])
            {
                assert_eq!(
                    serde_json::to_value(row.fields()).unwrap(),
                    serde_json::json!({
                        "backend":"vllm_neuron_fx_regions", "phase":phase, "token_bucket":bucket,
                        "max_model_len":512, "kv_blocks":6782, "block_size":32, "tp_size":4,
                        "dtype":"bf16", "region":region.as_str()
                    })
                );
            }
            assert_eq!(config.compute_dtype(), Some(DType::Bf16));
        }
    }

    #[test]
    fn coordinates_match_the_whole_forward_inventory() {
        let config = config(NeuronLlamaRegion::Head);
        for (phase, token_bucket, expected) in [
            (NeuronLlamaForwardPhase::Prefill, 512, 0.),
            (NeuronLlamaForwardPhase::Decode, 1, 1.),
            (NeuronLlamaForwardPhase::Decode, 16, 2.),
        ] {
            let input = NeuronLlamaRegionKernelInput {
                phase,
                token_bucket,
            };
            assert_eq!(
                &*NeuronLlamaRegionSpec::cache_coords(&config, &input),
                &[expected]
            );
        }
    }

    #[test]
    fn config_rejects_unverified_backend_context_and_buckets() {
        let mut bad = config(NeuronLlamaRegion::Model);
        bad.backends = vec!["vllm_neuron"];
        assert!(NeuronLlamaRegionSpec::validate_config(&bad).is_err());
        bad = config(NeuronLlamaRegion::Model);
        bad.decode_buckets = vec![1, 16, 32];
        assert!(NeuronLlamaRegionSpec::validate_config(&bad).is_err());
        bad = config(NeuronLlamaRegion::Model);
        bad.max_model_len = 128.into();
        bad.decode_buckets = vec![1];
        assert!(NeuronLlamaRegionSpec::validate_config(&bad).is_err());
    }
}
