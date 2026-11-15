//! How DeepSeek-V4.1-Flash's Engram rows reach the GPU (a deployment choice).
//!
//! - `uva` (vLLM's default, `EngramConfig.cpu_offload`): one Triton lookup
//!   per Engram layer reads the pinned host table over UVA on the GPU's SMs
//!   (the profiled `engram_lookup` at `residency = host_uva`).
//! - `off`: the fetch is priced out (the Engram blocks' compute stays).
//! - `none`: Engram fully off, DeepInfer's `DEEPINFER_ENGRAM_SANITY_OFF=1`
//!   (no fetch, no `wkv` GEMM, no gate): the baseline its CPU-fetch
//!   captures are measured against.
//! - `cpu_threads`: `engram_cpu_threads` host threads per rank, started at
//!   the step's start, gather the step's rows into a pinned staging buffer,
//!   one copy-engine H2D per layer moves it, and a device kernel unpacks
//!   (dequantizes) it. The gather and the copy use no SM time; the GPU waits
//!   only for what they have not finished by the time the consumer layer
//!   needs its rows:
//!   `exposed_k = max(0, ready_k - (available_k + exposed_<k))`, with layer
//!   `k`'s rows ready at `(k + 1) * gather + h2d` (one gather after another
//!   on the rank's threads, each copy overlapping the next gather) and
//!   `available_k` the step's GPU time before the consumer layer, taken as the
//!   step's mean per-layer time times the layers ahead of it on this GPU.
//!   `cpu_gather` (the earlier name) is `cpu_threads` at its default 8.
//! - `prefetch` (earlier `cpu_gather_prefetch`): `cpu_threads` started one
//!   step early. The Engram hashes depend only on token ids, and the
//!   scheduler knows step `k`'s tokens while step `k-1` runs, so the host
//!   gather and the H2D copies of step `k`'s rows start at step `k-1`'s
//!   start. They share nothing with step `k-1` but the host threads and the
//!   copy engine, so in the same formula the rows are ready `T_prev` earlier:
//!   `ready_k = (k + 1) * gather + h2d - T_prev`. `T_prev`, the previous
//!   step's GPU time, is taken as this step's own GPU time before the wait
//!   (the arch evaluates one step without its history; a chunked-prefill
//!   engine's consecutive steps carry the same token budget). The unpack
//!   stays on the GPU, as under `cpu_threads`.
//! - `gpu`: the table resident in HBM, read by the device `engram_lookup`
//!   (`residency = device`) over this GPU's hash columns (24 per rank). A
//!   theoretical bound only: the two layers' tables (101.4 GB each) do not
//!   fit beside the weights, and DeepInfer with
//!   `DEEPINFER_ENGRAM_FORCE_GPU_STORAGE=1` OOMs at load on B200 (Slurm job
//!   22879, wt-dsv41-deepinfer logs/20261114_2_deepinfer_prefill).
//!
//! Measured inputs (experiments/20261114_engram_cpu_gather, B200 host):
//! - gather: 393,216 rows per rank (8192 tokens x 2 layers x 24 hash columns)
//!   with 4 ranks gathering at once (the 4 GPUs of every deployment here), THP
//!   on, split layout, threads on the table's NUMA node, no software prefetch;
//!   median step time over the repeats ([`GATHER_STEP_MS`]). Thread counts
//!   between measured points take the nearest measured point (ties to the
//!   fewer threads); none is extrapolated past 14 per rank, the most measured
//!   with 4 ranks (16 and 32 were measured with one rank only).
//! - H2D: one layer's 51,904,512 B in 1.150 ms (45.1 GB/s) with 4 GPUs
//!   copying at once while a GEMM and the CPU gather run
//!   (`results/h2d_table.md`, `all4gpu_with_cpu_gather_52thr`); the GEMM ran
//!   at most 0.9% slower beside it, so the copy is charged no SM time.
//!
//! The unpack (`cpu_threads`, `prefetch`) and the `gpu` lookup are both the
//! profiled device-resident `engram_lookup`. Its profiled geometry is 6 hash
//! columns per token over a quarter table (a full 24-column, 384M-row device
//! table does not fit one B200), so a GPU's `rows_per_token` columns are priced
//! as `rows_per_token / 6` times the tokens at 6 columns: the same row count,
//! a random gather from HBM either way (for the unpack, an upper bound on a
//! sequential read of the packed buffer).

use crate::arch::deepseek_v41_vllm::{PlanNode, ENGRAM_TABLE_ROWS_RANK0};
use crate::timing::{FlatCostNode, LeafMetrics};
use crate::worklet::DeepseekV41EngramPrefetchLocalWorkletConfig;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EngramFetch {
    #[default]
    Uva,
    /// Host threads gather at the step's start (`engram_cpu_threads`).
    CpuThreads,
    /// `cpu_threads` with the next step's rows gathered during this step.
    Prefetch,
    /// The table in HBM, read by the device lookup (a bound; OOMs on B200).
    Gpu,
    /// The fetch priced out (the Engram blocks' compute stays): the engine's
    /// own fetch path cannot be measured here.
    Off,
    /// No fetch and no Engram consumer (`wkv` + gate): DeepInfer's
    /// `DEEPINFER_ENGRAM_SANITY_OFF=1`.
    None,
}

impl EngramFetch {
    /// Whether any Engram rows are fetched (`off` and `none` price none).
    pub fn fetches(self) -> bool {
        !matches!(self, Self::Off | Self::None)
    }
}

/// The `engram_fetch` spellings: `uva`, `cpu_threads` (or `cpu_gather`),
/// `prefetch` (or `cpu_gather_prefetch`), `gpu`, `off`, `none`, or a bool (`true` =
/// uva, `false` = off) as the DeepInfer engine option spells it.
impl<'de> serde::Deserialize<'de> for EngramFetch {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Bool(bool),
            Name(String),
        }
        match Raw::deserialize(d)? {
            Raw::Bool(true) => Ok(Self::Uva),
            Raw::Bool(false) => Ok(Self::Off),
            Raw::Name(name) => match name.as_str() {
                "uva" | "true" => Ok(Self::Uva),
                "cpu_threads" | "cpu_gather" => Ok(Self::CpuThreads),
                "prefetch" | "cpu_gather_prefetch" => Ok(Self::Prefetch),
                "gpu" => Ok(Self::Gpu),
                "off" | "false" => Ok(Self::Off),
                "none" => Ok(Self::None),
                other => Err(serde::de::Error::custom(format!(
                    "engram_fetch: expected uva, cpu_threads, prefetch, gpu, off or none, got {other}"
                ))),
            },
        }
    }
}

/// The `engram_fetch` choices the launcher offers.
pub const ENGRAM_FETCH_CHOICES: [&str; 8] = [
    "uva",
    "cpu_threads",
    "prefetch",
    "gpu",
    "off",
    "none",
    "cpu_gather",
    "cpu_gather_prefetch",
];

/// Rows per rank in one measured gather step (8192 tokens x 2 layers x 24).
const GATHER_STEP_ROWS: f64 = 393_216.0;
/// Measured gather step (ms) per threads-per-rank, 4 ranks at once
/// (median over the repeats; experiments/20261114_engram_cpu_gather/logs):
/// 1 and 2 from `cpu_gather3_threads.log` (26.837/26.119/42.967,
/// 13.897/15.975/15.911; a 76M-row table, still far past the LLC), 4, 8 and
/// 14 from `cpu_gather2.log` (8.212/6.932/7.801, 6.645/5.437/6.875,
/// 5.741/5.060/5.774; the full 384M-row table).
const GATHER_STEP_MS: [(u32, f64); 5] = [
    (1, 26.837),
    (2, 15.911),
    (4, 7.801),
    (8, 6.645),
    (14, 5.741),
];

/// The measured gather step (ms) nearest `threads` per rank (ties to the
/// fewer threads; past the last point, the last point).
pub fn gather_step_ms(threads: u32) -> f64 {
    GATHER_STEP_MS
        .iter()
        .min_by_key(|(t, _)| (t.abs_diff(threads), *t))
        .map(|&(_, ms)| ms)
        .expect("non-empty table")
}

/// H2D bytes per ms per GPU with four GPUs copying at once.
const H2D_BYTES_PER_MS: f64 = 51_904_512.0 / 1.150;
/// Hash columns of the profiled device-resident lookup geometry.
const UNPACK_PROFILED_HEADS: u32 = 6;

/// The host-gathered fetch of one GPU (`cpu_threads` / `prefetch`): its row
/// geometry, gather rate and start.
#[derive(Clone, Copy, Debug)]
pub struct EngramHostGather {
    /// Rows per token on this GPU (its hash columns).
    pub rows_per_token: u32,
    /// Bytes per row: FP8 values plus UE8M0 scales.
    pub row_bytes: u32,
    /// Tokens of the profiled unpack per scheduled token (same row count).
    pub unpack_token_scale: u32,
    /// `prefetch`: the rows are fetched during the previous step.
    pub prefetch: bool,
    /// Host gather rate of one rank (rows per ms) at its thread count.
    pub gather_rows_per_ms: f64,
}

impl EngramHostGather {
    /// Turn the UVA prefetch config into the device-resident lookup over the
    /// same rows (the `gpu` fetch, and the unpack of a host gather) and
    /// describe the host side at the measured 8 threads per rank.
    pub fn from_prefetch(prefetch: &mut DeepseekV41EngramPrefetchLocalWorkletConfig) -> Self {
        let local_heads = prefetch.engram_num_heads / prefetch.tp_size;
        assert_eq!(
            local_heads % UNPACK_PROFILED_HEADS,
            0,
            "device lookup: {local_heads} hash columns per GPU is not a multiple of the profiled {UNPACK_PROFILED_HEADS}"
        );
        let gather = Self {
            rows_per_token: local_heads,
            row_bytes: prefetch.engram_head_dim
                + prefetch.engram_head_dim / prefetch.quant_block_size,
            unpack_token_scale: local_heads / UNPACK_PROFILED_HEADS,
            prefetch: false,
            gather_rows_per_ms: GATHER_STEP_ROWS / gather_step_ms(8),
        };
        prefetch.tp_size = prefetch.engram_num_heads / UNPACK_PROFILED_HEADS;
        prefetch.table_rows = ENGRAM_TABLE_ROWS_RANK0;
        prefetch.residency = "device".into();
        prefetch.token_scale = gather.unpack_token_scale;
        gather
    }

    /// Apply `fetch` to the prefetch config: `cpu_threads` / `prefetch` (at
    /// `threads` per rank) return the host side, `gpu` turns the lookup
    /// device-resident and returns `None`, `uva` / `off` leave it.
    pub fn for_fetch(
        fetch: EngramFetch,
        threads: u32,
        prefetch: &mut DeepseekV41EngramPrefetchLocalWorkletConfig,
    ) -> Option<Self> {
        let early = match fetch {
            EngramFetch::CpuThreads => false,
            EngramFetch::Prefetch => true,
            EngramFetch::Gpu => {
                Self::from_prefetch(prefetch);
                return None;
            }
            EngramFetch::Uva | EngramFetch::Off | EngramFetch::None => return None,
        };
        Some(Self {
            prefetch: early,
            gather_rows_per_ms: GATHER_STEP_ROWS / gather_step_ms(threads),
            ..Self::from_prefetch(prefetch)
        })
    }

    /// GPU time (ms) the step waits for host-gathered rows: `tokens` scheduled
    /// tokens on this GPU, `available_ms[k]` the GPU time ahead of the k-th
    /// local Engram consumer layer, `step_ms` the step's GPU time before the
    /// wait (the previous step's under `prefetch`, see the module doc).
    pub fn exposed_ms(&self, tokens: u32, available_ms: &[f64], step_ms: f64) -> f64 {
        if tokens == 0 {
            return 0.0;
        }
        let rows = f64::from(tokens) * f64::from(self.rows_per_token);
        let gather = rows / self.gather_rows_per_ms;
        let h2d = rows * f64::from(self.row_bytes) / H2D_BYTES_PER_MS;
        let head_start = if self.prefetch { step_ms } else { 0.0 };
        let mut exposed = 0.0;
        for (k, &available) in available_ms.iter().enumerate() {
            let ready = (k + 1) as f64 * gather + h2d - head_start;
            exposed += (ready - (available + exposed)).max(0.0);
        }
        exposed
    }
}

/// Decoder layers one plan node runs (a body is one layer).
pub(crate) fn plan_layers(plan: &[PlanNode]) -> u32 {
    plan.iter()
        .map(|node| match node {
            PlanNode::Body(_) => 1,
            PlanNode::Repeat { n, body } => n * plan_layers(body),
        })
        .sum()
}

/// GPU time (ms) ahead of each local Engram consumer in one aggregated step:
/// the time of the root `Sum`'s sections that run before the consumer's
/// layer (its Engram block opens the layer). `sections[i]` is the decoder
/// layers the root's `i`-th section runs (0 for a non-layer section: it counts
/// when it runs between two layers ahead of the consumer, or right before
/// it); `ahead[k]` is this GPU's layers before the `k`-th consumer. A section
/// holding the consumer's layer inside it (a folded repeat) counts by its
/// share of layers ahead. `scratch` is the node buffer
/// [`crate::timing::CostTree::aggregate`] just filled.
pub(crate) fn time_ahead_ms(
    flat: &[FlatCostNode],
    scratch: &[LeafMetrics],
    sections: &[u32],
    ahead: &[u32],
) -> Vec<f64> {
    let Some(FlatCostNode::Sum { children }) = flat.first() else {
        panic!("the step's cost tree root must be a Sum of sections");
    };
    assert_eq!(
        children.len(),
        sections.len(),
        "one layer count per root section"
    );
    let layers: u32 = sections.iter().sum();
    assert!(
        ahead.iter().all(|&a| a < layers),
        "a consumer opens one of the {layers} layers"
    );
    ahead
        .iter()
        .map(|&target| {
            let (mut layers, mut ms) = (0u32, 0.0f64);
            for (node, &n) in children.clone().zip(sections) {
                let t = f64::from(scratch[node].m.time_ms);
                if layers + n <= target {
                    ms += t;
                    layers += n;
                } else {
                    ms += t * f64::from(target - layers) / f64::from(n);
                    break;
                }
            }
            ms
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gather() -> EngramHostGather {
        EngramHostGather {
            rows_per_token: 24,
            row_bytes: 264,
            unpack_token_scale: 4,
            prefetch: false,
            gather_rows_per_ms: GATHER_STEP_ROWS / 6.645,
        }
    }

    #[test]
    fn the_measured_step_reproduces_the_bench() {
        // 8192 tokens x 24 columns = one layer of the bench: 3.32 ms gather,
        // 1.15 ms copy. Nothing ahead of the layer: both exposed.
        let g = gather();
        let exposed = g.exposed_ms(8192, &[0.0], 0.0);
        assert!((exposed - (6.645 / 2.0 + 1.150)).abs() < 1e-9, "{exposed}");
    }

    #[test]
    fn enough_gpu_time_ahead_hides_the_fetch_and_waits_count_once() {
        let g = gather();
        assert_eq!(g.exposed_ms(8192, &[10.0, 50.0], 60.0), 0.0);
        assert_eq!(g.exposed_ms(0, &[0.0], 0.0), 0.0);
        // Layer 1 waits 0.47 ms; layer 14 (ready at 2g + h = 7.795) has
        // 7.0 + 0.47 ms ahead of it, so it waits the remaining 0.325 ms.
        let (gt, h) = (6.645 / 2.0, 1.150);
        let exposed = g.exposed_ms(8192, &[4.0, 7.0], 11.0);
        let first = gt + h - 4.0;
        let second = 2.0 * gt + h - (7.0 + first);
        assert!((exposed - (first + second)).abs() < 1e-9, "{exposed}");
    }

    #[test]
    fn prefetch_starts_the_fetch_one_step_early() {
        let g = EngramHostGather {
            prefetch: true,
            ..gather()
        };
        // A 4.473 ms fetch fully behind a previous step of 4 ms: 0.473 left.
        let exposed = g.exposed_ms(8192, &[0.0], 4.0);
        assert!(
            (exposed - (6.645 / 2.0 + 1.150 - 4.0)).abs() < 1e-9,
            "{exposed}"
        );
        assert_eq!(g.exposed_ms(8192, &[0.0, 3.0], 100.0), 0.0);
    }

    #[test]
    fn the_unpack_reuses_the_profiled_device_lookup_geometry() {
        let mut cfg = crate::worklet::deepseek_v41_engram_prefetch_local::tests::config();
        cfg.tp_size = 1;
        cfg.table_rows = 384_006_168;
        let g = EngramHostGather::from_prefetch(&mut cfg);
        assert_eq!(
            (g.rows_per_token, g.row_bytes, g.unpack_token_scale),
            (24, 264, 4)
        );
        assert_eq!(
            (cfg.tp_size, cfg.table_rows, cfg.residency.as_str(), cfg.token_scale),
            (4, 96_000_564, "device", 4)
        );
    }

    #[test]
    fn time_ahead_counts_the_priced_sections_before_each_consumer() {
        // Root sections: embedding (no layer) 0.5, layer 0 2.0, layer 1 1.0,
        // a 3-layer repeat of layers 2..5 3.0, the head 0.25, the host step 9.
        let leaf = |ms: f32| {
            let mut m = LeafMetrics::ZERO;
            m.m.time_ms = ms;
            m
        };
        let flat: Vec<FlatCostNode> = std::iter::once(FlatCostNode::Sum { children: 1..7 })
            .chain((0..6).map(FlatCostNode::Leaf))
            .collect();
        let scratch: Vec<LeafMetrics> = [16.25, 0.5, 2.0, 1.0, 3.0, 0.25, 9.0]
            .into_iter()
            .map(leaf)
            .collect();
        let ahead = time_ahead_ms(&flat, &scratch, &[0, 1, 1, 3, 0, 0], &[0, 1, 3, 4]);
        // Layer 0: what opens the step; layer 1: + layer 0; layers 3 and 4:
        // + layer 1 and one or two of the repeat's three layers; never the
        // head or the host step.
        assert_eq!(ahead, vec![0.5, 2.5, 4.5, 5.5]);
    }

    #[test]
    fn none_is_its_own_spelling_and_fetches_nothing() {
        let none: EngramFetch = serde_json::from_str("\"none\"").unwrap();
        assert_eq!(none, EngramFetch::None);
        assert!(!none.fetches() && !EngramFetch::Off.fetches());
        assert!(EngramFetch::CpuThreads.fetches() && EngramFetch::Uva.fetches());
    }

    #[test]
    fn threads_take_the_nearest_measured_point_without_extrapolating() {
        assert_eq!(gather_step_ms(1), 26.837);
        assert_eq!(gather_step_ms(3), 15.911); // tie 2/4 -> fewer threads
        assert_eq!(gather_step_ms(8), 6.645); // the former cpu_gather
        assert_eq!(gather_step_ms(16), 5.741);
        assert_eq!(gather_step_ms(32), 5.741);
        let mut cfg = crate::worklet::deepseek_v41_engram_prefetch_local::tests::config();
        cfg.tp_size = 1;
        cfg.table_rows = 384_006_168;
        let one = EngramHostGather::for_fetch(EngramFetch::CpuThreads, 1, &mut cfg.clone()).unwrap();
        let eight = EngramHostGather::for_fetch(EngramFetch::CpuThreads, 8, &mut cfg.clone()).unwrap();
        let ratio = one.exposed_ms(8192, &[0.0], 0.0) - 1.150;
        assert!((ratio - 26.837 / 2.0).abs() < 1e-9, "{ratio}");
        assert!(eight.exposed_ms(8192, &[0.0], 0.0) < one.exposed_ms(8192, &[0.0], 0.0));
        let mut gpu = cfg.clone();
        assert!(EngramHostGather::for_fetch(EngramFetch::Gpu, 8, &mut gpu).is_none());
        assert_eq!((gpu.residency.as_str(), gpu.token_scale), ("device", 4));
    }
}
