//! Shared identity of the standalone NKI per-op leaves of the Llama 3.1 8B
//! TP4/LNC2 forward (`neuron_nki_*` and `neuron_glue`).
//!
//! Terms: a *launch unit* is what the Python runner builds and times (one NKI
//! custom call, or the torch glue of a layer); a *kind* is its registry entry;
//! a *leaf* is the Rust cost leaf that reads the kind's rows.
//!
//! Each leaf is one launch unit of the vLLM Neuron forward. It is compiled and
//! timed alone on one logical core at the per-rank shapes of TP4. A coordinate
//! is one entry of the compiled-variant inventory: the prefill token buckets,
//! then the decode batch buckets. It is never a token length. There is no
//! interpolation. A kind that runs in one phase only leaves the other
//! inventory empty. The shape-only decode kinds (mlp and glue) read no block
//! table, so no context of their own caps their buckets. The direct cache can
//! warn about nonmonotonic time across the entries; it keeps the measured
//! cells unchanged.

use crate::timing::bridge::{de_backends, ArgsPayload, DType};
use crate::timing::kernels::neuron_llama_forward::{
    is_stock_gpu, largest_decode_bucket, NeuronLlamaForwardPhase, STOCK_BLOCK_SIZE, STOCK_CONTEXTS,
    STOCK_KV_BLOCKS, STOCK_TP_SIZE,
};
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

/// The one backend every standalone NKI leaf is measured with.
pub const NKI_BACKEND: &str = "vllm_neuron_nki";

/// The largest decode bucket of the whole context ladder (C512).
pub fn largest_ladder_decode_bucket() -> u32 {
    STOCK_CONTEXTS
        .iter()
        .map(|&c| largest_decode_bucket(c))
        .max()
        .expect("the stock ladder is not empty")
}

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NeuronNkiKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    /// Context of the paged KV cache the decode attention leaf reads; the
    /// other leaves carry it for identity only.
    pub max_model_len: Dim,
    pub kv_blocks: Dim,
    pub block_size: Dim,
    pub tp_size: Dim,
    #[compute_dtype]
    #[kv_dtype]
    pub dtype: DType,
    /// Exact prefill token buckets to profile/cache; empty for decode-only kinds.
    pub prefill_buckets: Vec<u32>,
    /// Exact decode batch buckets to profile/cache; empty for prefill-only kinds.
    pub decode_buckets: Vec<u32>,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct NeuronNkiKernelInput {
    pub phase: NeuronLlamaForwardPhase,
    pub token_bucket: u32,
}

impl SweepCoords for NeuronNkiKernelInput {
    fn coords(&self) -> Coords {
        panic!("Neuron NKI leaf coordinates require the Config variant inventory")
    }

    fn coord_field_names() -> &'static [&'static str] {
        &["compiled_variant"]
    }
}

/// Which phases a kind is measured in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NkiPhases {
    PrefillOnly,
    DecodeOnly,
    Both,
}

fn sorted_unique_pow2(buckets: &[u32]) -> bool {
    buckets.iter().all(|b| b.is_power_of_two()) && buckets.windows(2).all(|p| p[0] < p[1])
}

/// The identity checks every NKI leaf shares; `phases` adds the kind's own
/// inventory rule.
pub fn validate_config(config: &NeuronNkiKernelConfig, phases: NkiPhases) -> anyhow::Result<()> {
    anyhow::ensure!(
        config.backends == [NKI_BACKEND],
        "standalone NKI leaves require the {NKI_BACKEND} backend"
    );
    anyhow::ensure!(
        is_stock_gpu(&config.gpu_name),
        "standalone NKI leaves require a Trainium2 LNC2 target"
    );
    anyhow::ensure!(
        config.dtype == DType::Bf16,
        "standalone NKI leaves require BF16"
    );
    anyhow::ensure!(
        config.kv_blocks == STOCK_KV_BLOCKS
            && config.block_size == STOCK_BLOCK_SIZE
            && config.tp_size == STOCK_TP_SIZE,
        "standalone NKI leaves require {STOCK_KV_BLOCKS} KV blocks, page size \
         {STOCK_BLOCK_SIZE} and TP{STOCK_TP_SIZE}"
    );
    let context = config.max_model_len.get();
    anyhow::ensure!(
        STOCK_CONTEXTS.contains(&context),
        "stock Neuron context must be one of {STOCK_CONTEXTS:?}"
    );
    anyhow::ensure!(
        sorted_unique_pow2(&config.prefill_buckets)
            && config
                .prefill_buckets
                .iter()
                .all(|b| STOCK_CONTEXTS.contains(b)),
        "prefill buckets must be sorted unique entries of the stock context ladder"
    );
    // Decode attention reads a block table that spans the context, so the
    // context caps its buckets. The shape-only decode kinds use the ladder's cap.
    let largest = match phases {
        NkiPhases::DecodeOnly => largest_decode_bucket(context),
        _ => largest_ladder_decode_bucket(),
    };
    anyhow::ensure!(
        sorted_unique_pow2(&config.decode_buckets)
            && config.decode_buckets.iter().all(|&b| b <= largest),
        "decode buckets must be sorted unique powers of two within the context limit"
    );
    match phases {
        NkiPhases::PrefillOnly => anyhow::ensure!(
            config.decode_buckets.is_empty(),
            "a prefill-only NKI leaf has no decode inventory"
        ),
        NkiPhases::DecodeOnly => anyhow::ensure!(
            config.prefill_buckets.is_empty(),
            "a decode-only NKI leaf has no prefill inventory"
        ),
        NkiPhases::Both => {}
    }
    // The direct cache needs two grid points; one compiled variant cannot be cached.
    anyhow::ensure!(
        config.prefill_buckets.len() + config.decode_buckets.len() >= 2,
        "an NKI leaf inventory needs at least two compiled variants"
    );
    Ok(())
}

/// One axis over the inventory: prefill variants first, then decode variants.
/// `Kernel::build` never calls `validate_config`, so the two-variant rule the
/// direct cache needs is enforced here too, where every path goes through.
pub fn sweep_grid(config: &NeuronNkiKernelConfig) -> SweepGrid {
    let n = (config.prefill_buckets.len() + config.decode_buckets.len()) as u32;
    assert!(
        n >= 2,
        "an NKI leaf inventory needs at least two compiled variants (validate_config)"
    );
    SweepGrid::new(vec![Axis::values(0..n)])
}

/// Position of the input's variant on the inventory axis; panics on an
/// unlisted bucket (an unmeasured shape never borrows a neighbour's time).
pub fn cache_coords(config: &NeuronNkiKernelConfig, input: &NeuronNkiKernelInput) -> Coords {
    let index = match input.phase {
        NeuronLlamaForwardPhase::Prefill => config
            .prefill_buckets
            .binary_search(&input.token_bucket)
            .expect("prefill input must name a configured compiled token bucket"),
        NeuronLlamaForwardPhase::Decode => {
            config
                .decode_buckets
                .binary_search(&input.token_bucket)
                .expect("decode input must name a configured compiled batch bucket")
                + config.prefill_buckets.len()
        }
    };
    Coords::new([index as f64])
}

/// The variant each grid coordinate names.
pub fn variant(config: &NeuronNkiKernelConfig, coordinate: f64) -> (NeuronLlamaForwardPhase, u32) {
    assert!(
        coordinate.is_finite() && coordinate >= 0.0 && coordinate.fract() == 0.0,
        "compiled variant index must be a nonnegative integer"
    );
    let index = coordinate as usize;
    if index < config.prefill_buckets.len() {
        (
            NeuronLlamaForwardPhase::Prefill,
            config.prefill_buckets[index],
        )
    } else {
        (
            NeuronLlamaForwardPhase::Decode,
            *config
                .decode_buckets
                .get(index - config.prefill_buckets.len())
                .expect("compiled variant index exceeds inventory"),
        )
    }
}

/// Rows for every variant; `fields` adds the kind's own args after `backend`.
pub fn enumerate(
    config: &NeuronNkiKernelConfig,
    grid: &SweepGrid,
    backend: &'static str,
    fields: impl Fn(ArgsPayload, NeuronLlamaForwardPhase, u32) -> ArgsPayload,
) -> Vec<ArgsPayload> {
    grid.expand_1d(|coordinate| {
        let (phase, bucket) = variant(config, coordinate);
        fields(ArgsPayload::new().with("backend", backend), phase, bucket)
    })
}

/// The identity fields every kind emits after its shape fields.
pub fn identity_fields(payload: ArgsPayload, config: &NeuronNkiKernelConfig) -> ArgsPayload {
    payload
        .with("tp_size", config.tp_size.get())
        .with("dtype", config.dtype.as_str())
}

/// Rows of a prefill-only kind: `backend, token_bucket, tp_size, dtype`.
pub fn enumerate_prefill_only(
    config: &NeuronNkiKernelConfig,
    grid: &SweepGrid,
    backend: &'static str,
) -> Vec<ArgsPayload> {
    enumerate(config, grid, backend, |row, _phase, bucket| {
        identity_fields(row.with("token_bucket", bucket), config)
    })
}

/// Test helper for the per-kind files: the kind's rows for `config` must be
/// exactly `expected`, in inventory order, and its cache the direct lookup.
#[cfg(test)]
pub(crate) fn assert_rows<S>(config: &NeuronNkiKernelConfig, expected: &[serde_json::Value])
where
    S: crate::timing::kernels::engine::KernelSpec<
        Config = NeuronNkiKernelConfig,
        Input = NeuronNkiKernelInput,
    >,
{
    S::validate_config(config).unwrap();
    let rows = S::enumerate(config, &S::sweep_grid(config), NKI_BACKEND);
    assert_eq!(rows.len(), expected.len());
    for (row, want) in rows.iter().zip(expected) {
        assert_eq!(&serde_json::to_value(row.fields()).unwrap(), want);
    }
    assert_eq!(
        S::cache_kind(NKI_BACKEND),
        crate::timing::cache::CacheKind::Cache1DDirect
    );
}

#[cfg(test)]
pub(crate) fn test_config(prefill: Vec<u32>, decode: Vec<u32>) -> NeuronNkiKernelConfig {
    NeuronNkiKernelConfig {
        backends: vec![NKI_BACKEND],
        gpu_name: "AWS Trainium2 LNC2".into(),
        max_model_len: 512.into(),
        kv_blocks: 6782.into(),
        block_size: 32.into(),
        tp_size: 4.into(),
        dtype: DType::Bf16,
        prefill_buckets: prefill,
        decode_buckets: decode,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants_index_prefill_then_decode_and_reject_unlisted_buckets() {
        let config = test_config(vec![128, 512], vec![1, 16]);
        for (phase, bucket, expected) in [
            (NeuronLlamaForwardPhase::Prefill, 128, 0.),
            (NeuronLlamaForwardPhase::Prefill, 512, 1.),
            (NeuronLlamaForwardPhase::Decode, 1, 2.),
            (NeuronLlamaForwardPhase::Decode, 16, 3.),
        ] {
            let input = NeuronNkiKernelInput {
                phase,
                token_bucket: bucket,
            };
            assert_eq!(&*cache_coords(&config, &input), &[expected]);
            assert_eq!(variant(&config, expected), (phase, bucket));
        }
        assert_eq!(sweep_grid(&config).axes()[0], [0., 1., 2., 3.]);
        for (phase, bucket) in [
            (NeuronLlamaForwardPhase::Prefill, 256),
            (NeuronLlamaForwardPhase::Decode, 2),
            (NeuronLlamaForwardPhase::Decode, 128),
        ] {
            let input = NeuronNkiKernelInput {
                phase,
                token_bucket: bucket,
            };
            assert!(std::panic::catch_unwind(|| cache_coords(&config, &input)).is_err());
        }
    }

    #[test]
    fn inventory_rules_follow_the_kind_phases() {
        validate_config(&test_config(vec![128, 512], vec![]), NkiPhases::PrefillOnly).unwrap();
        validate_config(&test_config(vec![], vec![1, 16]), NkiPhases::DecodeOnly).unwrap();
        validate_config(&test_config(vec![512], vec![1]), NkiPhases::Both).unwrap();
        // a single variant cannot fill the direct cache
        assert!(validate_config(&test_config(vec![512], vec![]), NkiPhases::PrefillOnly).is_err());
        // the wrong phase inventory
        assert!(
            validate_config(&test_config(vec![512], vec![1, 16]), NkiPhases::PrefillOnly).is_err()
        );
        assert!(
            validate_config(&test_config(vec![512], vec![1, 16]), NkiPhases::DecodeOnly).is_err()
        );
        // decode attention is capped by its context; shape-only kinds by the ladder
        let mut cfg = test_config(vec![], vec![1, 64]);
        cfg.max_model_len = 8192.into();
        assert!(validate_config(&cfg, NkiPhases::DecodeOnly).is_err());
        validate_config(&cfg, NkiPhases::Both).unwrap();
        assert_eq!(largest_ladder_decode_bucket(), 512);
        // the grid itself refuses a one-variant inventory, for callers that skip validate_config
        assert!(std::panic::catch_unwind(|| sweep_grid(&test_config(vec![512], vec![]))).is_err());
        // identity
        for bad in [
            {
                let mut c = test_config(vec![128, 512], vec![]);
                c.backends = vec!["vllm_neuron"];
                c
            },
            {
                let mut c = test_config(vec![128, 512], vec![]);
                c.dtype = DType::Fp16;
                c
            },
            {
                let mut c = test_config(vec![128, 512], vec![]);
                c.tp_size = 8.into();
                c
            },
            test_config(vec![512, 128], vec![]),
            test_config(vec![256, 512], vec![]),
            test_config(vec![], vec![1, 3]),
        ] {
            assert!(validate_config(&bad, NkiPhases::Both).is_err());
        }
    }
}
