//! Stock TP4/LNC2 full-model forward compiled with segmented (chunked) prefill.
//!
//! The engine runs `max_num_batched_tokens = S` with one compiled prefill
//! executable per segment size: a prompt executes as `ceil(prompt / S)` calls
//! of the same S-token graph, the k-th attending `k * S` cached tokens. Decode
//! graphs are the ordinary compiled `num_seqs_buckets`. All layers, collectives
//! and sampling stay inside each measured call.
//!
//! One cache axis carries both phases. A prefill cell sits at its own
//! `prior_tokens`, so lookups between measured priors interpolate linearly in
//! cached-token count; decode cells sit past every feasible prior, at
//! `max_model_len + bucket index`, so they are exact and never blend with
//! prefill. Lookups beyond the largest measured prior, off the block grid, or
//! for an unconfigured bucket are rejected rather than extrapolated. The
//! linear cache may warn about nonmonotonic time where the last prefill cell
//! meets the first decode cell; it keeps the measured cells unchanged.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::neuron_llama_forward::{
    largest_decode_bucket, NeuronLlamaForwardKernelConfig, NeuronLlamaForwardPhase,
    NeuronLlamaForwardSpec,
};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

/// Compiled contexts the segmented profiler accepts. 8192 cross-checks the
/// non-segmented whole forward at the top of the stock ladder.
pub const SEGMENTED_CONTEXTS: [u32; 3] = [8192, 16384, 32768];

/// Segment sizes (`max_num_batched_tokens`) the segmented profiler accepts.
pub const SEGMENT_SIZES: [u32; 4] = [512, 1024, 2048, 4096];

/// Output tokens every profiled prompt leaves room for: the grid covers
/// prompts of up to `max_model_len - PROMPT_TAIL` tokens.
/// `stock_contract.VALIDATION_OUTPUT_TOKENS` mirrors this value.
pub const PROMPT_TAIL: u32 = 8;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NeuronLlamaSegmentedForwardKernelConfig {
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
    /// Compiled prefill segment S, equal to the engine's token budget.
    pub segment_size: Dim,
    /// Exact compiled decode variants to profile/cache; not an interpolation grid.
    pub decode_buckets: Vec<u32>,
}

impl NeuronLlamaSegmentedForwardKernelConfig {
    /// Prefill executions of the longest profiled prompt (`C - 8` tokens).
    pub fn prefill_segments(&self) -> u32 {
        (self.max_model_len.get() - PROMPT_TAIL).div_ceil(self.segment_size.get())
    }

    /// Largest measured `prior_tokens`; lookups past it are rejected.
    pub fn largest_prior(&self) -> u32 {
        (self.prefill_segments() - 1) * self.segment_size.get()
    }

    /// The unsegmented stock config with the same runtime identity, for the
    /// shared layout checks.
    fn stock_identity(&self) -> NeuronLlamaForwardKernelConfig {
        NeuronLlamaForwardKernelConfig {
            backends: vec!["vllm_neuron"],
            gpu_name: self.gpu_name.clone(),
            max_model_len: self.max_model_len.clone(),
            kv_blocks: self.kv_blocks.clone(),
            block_size: self.block_size.clone(),
            tp_size: self.tp_size.clone(),
            dtype: self.dtype,
            decode_buckets: self.decode_buckets.clone(),
            // Empty: the shared layout checks need only the runtime identity.
            // This engine compiles its own S-token prefill graph.
            prefill_buckets: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NeuronLlamaSegmentedForwardKernelInput {
    pub phase: NeuronLlamaForwardPhase,
    /// Prefill: the segment size S. Decode: the compiled decode bucket.
    pub token_bucket: u32,
    /// Prefill: cached tokens attended before this segment. Decode: 0.
    pub prior_tokens: u32,
}

impl SweepCoords for NeuronLlamaSegmentedForwardKernelInput {
    fn coords(&self) -> Coords {
        panic!("Neuron segmented-forward coordinates require the Config variant inventory")
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["prior_tokens_or_decode_variant"]
    }
}

pub struct NeuronLlamaSegmentedForwardSpec;

impl KernelSpec for NeuronLlamaSegmentedForwardSpec {
    type Config = NeuronLlamaSegmentedForwardKernelConfig;
    type Input = NeuronLlamaSegmentedForwardKernelInput;

    const KIND: KernelKind = "neuron_llama_segmented_forward";

    fn validate_config(config: &Self::Config) -> anyhow::Result<()> {
        anyhow::ensure!(
            config.backends == ["vllm_neuron_segmented"],
            "stock Neuron segmented forward requires the pinned vllm_neuron_segmented backend"
        );
        let context = config.max_model_len.get();
        anyhow::ensure!(
            SEGMENTED_CONTEXTS.contains(&context),
            "stock Neuron segmented context must be one of {SEGMENTED_CONTEXTS:?}"
        );
        let segment = config.segment_size.get();
        anyhow::ensure!(
            SEGMENT_SIZES.contains(&segment) && segment < context,
            "segment size must be one of {SEGMENT_SIZES:?} and below the context"
        );
        // Target, dtype, page layout, TP and the decode inventory follow the
        // unsegmented stock rules. Its context list is the C512-C8192 ladder, so
        // check with a ladder context and the real context's bucket ceiling.
        let mut stock = config.stock_identity();
        stock.max_model_len = Dim::from(8192);
        let ceiling = largest_decode_bucket(context);
        anyhow::ensure!(
            config
                .decode_buckets
                .iter()
                .all(|&b| b <= ceiling && b <= segment),
            "decode buckets must fit the KV-resident capacity ({ceiling}) and the segment token budget"
        );
        NeuronLlamaForwardSpec::validate_config(&stock)
    }

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        // Prefill cells at k*S, then decode cells past every feasible prior.
        // At most 64 prefill cells (C32768/S512) plus the decode inventory.
        let segment = config.segment_size.get();
        let context = config.max_model_len.get();
        SweepGrid::new(vec![Axis::values(
            (0..config.prefill_segments())
                .map(|k| k * segment)
                .chain((0..config.decode_buckets.len() as u32).map(|i| context + i)),
        )])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache1DLinear
    }

    fn cache_coords(config: &Self::Config, input: &Self::Input) -> Coords {
        let coordinate = match input.phase {
            NeuronLlamaForwardPhase::Prefill => {
                assert_eq!(
                    input.token_bucket,
                    config.segment_size.get(),
                    "prefill input must name the configured segment size"
                );
                assert!(
                    input.prior_tokens % config.block_size.get() == 0,
                    "prefill prior tokens must be block-aligned"
                );
                assert!(
                    input.prior_tokens <= config.largest_prior(),
                    "prefill prior tokens exceed the largest measured prior (no extrapolation)"
                );
                input.prior_tokens
            }
            NeuronLlamaForwardPhase::Decode => {
                assert_eq!(input.prior_tokens, 0, "decode carries no prior tokens");
                let index = config
                    .decode_buckets
                    .binary_search(&input.token_bucket)
                    .expect("decode input must name a configured compiled bucket");
                config.max_model_len.get() + index as u32
            }
        };
        Coords::new([f64::from(coordinate)])
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        let context = config.max_model_len.get();
        grid.expand_1d(|coordinate| {
            assert!(
                coordinate.is_finite() && coordinate >= 0.0 && coordinate.fract() == 0.0,
                "segmented-forward coordinate must be a nonnegative integer"
            );
            let coordinate = coordinate as u32;
            let (phase, token_bucket, prior_tokens) = if coordinate < context {
                (
                    NeuronLlamaForwardPhase::Prefill,
                    config.segment_size.get(),
                    coordinate,
                )
            } else {
                (
                    NeuronLlamaForwardPhase::Decode,
                    *config
                        .decode_buckets
                        .get((coordinate - context) as usize)
                        .expect("decode variant index exceeds inventory"),
                    0,
                )
            };
            ArgsPayload::new()
                .with("backend", backend)
                .with("phase", phase.as_str())
                .with("token_bucket", token_bucket)
                .with("prior_tokens", prior_tokens)
                .with("segment_size", config.segment_size.get())
                .with("max_model_len", context)
                .with("kv_blocks", config.kv_blocks.get())
                .with("block_size", config.block_size.get())
                .with("tp_size", config.tp_size.get())
                .with("dtype", config.dtype.as_str())
        })
    }
}

register_kernel!(
    NeuronLlamaSegmentedForwardKernel,
    NeuronLlamaSegmentedForwardSpec
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::bridge::KernelMetrics;
    use crate::timing::cache::build_cache;
    use crate::timing::kernels::engine::KernelConfig;

    fn config(
        context: u32,
        segment: u32,
        buckets: &[u32],
    ) -> NeuronLlamaSegmentedForwardKernelConfig {
        NeuronLlamaSegmentedForwardKernelConfig {
            backends: vec!["vllm_neuron_segmented"],
            gpu_name: "AWS Trainium2 LNC2".into(),
            max_model_len: context.into(),
            kv_blocks: 6782.into(),
            block_size: 32.into(),
            tp_size: 4.into(),
            dtype: DType::Bf16,
            segment_size: segment.into(),
            decode_buckets: buckets.to_vec(),
        }
    }

    fn prefill(prior_tokens: u32) -> NeuronLlamaSegmentedForwardKernelInput {
        NeuronLlamaSegmentedForwardKernelInput {
            phase: NeuronLlamaForwardPhase::Prefill,
            token_bucket: 2048,
            prior_tokens,
        }
    }

    fn decode(token_bucket: u32) -> NeuronLlamaSegmentedForwardKernelInput {
        NeuronLlamaSegmentedForwardKernelInput {
            phase: NeuronLlamaForwardPhase::Decode,
            token_bucket,
            prior_tokens: 0,
        }
    }

    #[test]
    fn enumeration_emits_one_row_per_segment_prior_and_decode_bucket() {
        let config = config(16384, 2048, &[1, 2, 4, 8, 16]);
        // ceil((16384 - 8) / 2048) = 8 prefill executions: priors 0..=14336.
        assert_eq!(config.prefill_segments(), 8);
        assert_eq!(config.largest_prior(), 14336);
        let grid = NeuronLlamaSegmentedForwardSpec::sweep_grid(&config);
        let rows =
            NeuronLlamaSegmentedForwardSpec::enumerate(&config, &grid, "vllm_neuron_segmented");
        assert_eq!(rows.len(), 13);
        let expected = (0..8)
            .map(|k| ("prefill", 2048, k * 2048))
            .chain([1, 2, 4, 8, 16].map(|b| ("decode", b, 0)));
        for (row, (phase, bucket, prior)) in rows.iter().zip(expected) {
            assert_eq!(
                serde_json::to_value(row.fields()).unwrap(),
                serde_json::json!({
                    "backend":"vllm_neuron_segmented", "phase":phase, "token_bucket":bucket,
                    "prior_tokens":prior, "segment_size":2048, "max_model_len":16384,
                    "kv_blocks":6782, "block_size":32, "tp_size":4, "dtype":"bf16"
                })
            );
        }
        assert_eq!(config.compute_dtype(), Some(DType::Bf16));
        assert_eq!(config.kv_dtype(), Some(DType::Bf16));
        // The densest supported grid stays well under the 500-cell ceiling.
        let dense = super::tests::config(32768, 512, &[1, 2, 4, 8]);
        assert_eq!(dense.prefill_segments(), 64);
        assert_eq!(
            NeuronLlamaSegmentedForwardSpec::sweep_grid(&dense).axes()[0].len(),
            68
        );
    }

    #[test]
    fn prefill_interpolates_over_prior_and_decode_is_exact() {
        let config = config(16384, 2048, &[1, 16]);
        let grid = NeuronLlamaSegmentedForwardSpec::sweep_grid(&config);
        // Prefill k costs 100 + 10k ms; decode buckets cost 7 and 9 ms.
        let samples: Vec<KernelMetrics> = (0..8)
            .map(|k| 100.0 + 10.0 * k as f64)
            .chain([7.0, 9.0])
            .map(|time_ms| KernelMetrics {
                time_ms,
                tflops: None,
                memory_bandwidth_gbps: None,
                algbw_gbps: None,
                busbw_gbps: None,
                energy_j: 0.0,
            })
            .collect();
        let (cache, _) = build_cache(
            NeuronLlamaSegmentedForwardSpec::KIND,
            NeuronLlamaSegmentedForwardSpec::cache_kind("vllm_neuron_segmented"),
            &grid,
            &samples,
        )
        .unwrap();
        let time = |input: NeuronLlamaSegmentedForwardKernelInput| {
            let coords = NeuronLlamaSegmentedForwardSpec::cache_coords(&config, &input);
            let leaf = cache.eval(&coords);
            assert!(leaf.coverage.is_empty(), "lookup left the measured grid");
            leaf.m.time_ms
        };
        assert_eq!(time(prefill(0)), 100.0);
        assert_eq!(time(prefill(4096)), 120.0);
        assert_eq!(time(prefill(14336)), 170.0);
        // Halfway between priors 2048 and 4096; a quarter past 12288.
        assert_eq!(time(prefill(3072)), 115.0);
        assert_eq!(time(prefill(12288 + 512)), 162.5);
        assert_eq!(time(decode(1)), 7.0);
        assert_eq!(time(decode(16)), 9.0);
    }

    #[test]
    fn lookups_off_the_measured_domain_are_rejected() {
        let config = config(16384, 2048, &[1, 16]);
        for input in [
            prefill(14336 + 32),
            prefill(2048 + 16),
            NeuronLlamaSegmentedForwardKernelInput {
                phase: NeuronLlamaForwardPhase::Prefill,
                token_bucket: 1024,
                prior_tokens: 0,
            },
            decode(2),
            decode(32),
            NeuronLlamaSegmentedForwardKernelInput {
                phase: NeuronLlamaForwardPhase::Decode,
                token_bucket: 1,
                prior_tokens: 32,
            },
        ] {
            assert!(
                std::panic::catch_unwind(|| {
                    NeuronLlamaSegmentedForwardSpec::cache_coords(&config, &input)
                })
                .is_err(),
                "{input:?} should be rejected"
            );
        }
    }

    #[test]
    fn config_accepts_the_segmented_ladder_and_rejects_bad_identity() {
        for (context, segment, buckets) in [
            (16384, 2048, vec![1, 2, 4, 8, 16]),
            (32768, 2048, vec![1, 2, 4, 8]),
            (32768, 512, vec![1, 8]),
            (8192, 4096, vec![1, 16, 32]),
        ] {
            NeuronLlamaSegmentedForwardSpec::validate_config(&config(context, segment, &buckets))
                .unwrap();
        }
        let good = config(16384, 2048, &[1, 16]);
        let mut cases = Vec::new();
        for (context, segment, buckets) in [
            (4096, 2048, vec![1, 16]),   // ladder context without a segmented profile
            (16384, 16384, vec![1, 16]), // segment must be below the context
            (16384, 3072, vec![1, 16]),  // unsupported segment
            (16384, 2048, vec![1, 32]),  // bucket above next_pow2(217024 / 16384)
            (32768, 2048, vec![1, 16]),  // bucket above next_pow2(217024 / 32768)
            (16384, 2048, vec![16, 1]),  // unsorted inventory
            (16384, 2048, vec![]),       // empty inventory
        ] {
            cases.push(config(context, segment, &buckets));
        }
        let mut bad = good.clone();
        bad.backends = vec!["vllm_neuron"];
        cases.push(bad);
        bad = good.clone();
        bad.tp_size = 1.into();
        cases.push(bad);
        bad = good.clone();
        bad.block_size = 16.into();
        cases.push(bad);
        bad = good;
        bad.dtype = DType::Fp16;
        cases.push(bad);
        for bad in cases {
            assert!(
                NeuronLlamaSegmentedForwardSpec::validate_config(&bad).is_err(),
                "{bad:?} should be rejected"
            );
        }
    }
}
