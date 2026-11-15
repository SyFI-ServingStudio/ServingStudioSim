//! How DeepSeek-V4.1-Flash's Engram rows reach the GPU (a deployment choice).
//!
//! - `uva` (vLLM's default, `EngramConfig.cpu_offload`): one Triton lookup
//!   per Engram layer reads the pinned host table over UVA on the GPU's SMs
//!   (the profiled `engram_lookup` at `residency = host_uva`).
//! - `off`: the fetch is priced out (the Engram blocks' compute stays).
//! - `cpu_gather`: host threads gather the step's rows into a pinned staging
//!   buffer, one copy-engine H2D per layer moves it, and a device kernel
//!   unpacks (dequantizes) it. The gather and the copy use no SM time; the
//!   GPU waits only for what they have not finished by the time the consumer
//!   layer needs its rows:
//!   `exposed_k = max(0, ready_k - (available_k + exposed_<k))`, with layer
//!   `k`'s rows ready at `(k + 1) * gather + h2d` (one gather after another
//!   on the rank's threads, each copy overlapping the next gather) and
//!   `available_k` the step's GPU time before the consumer layer, taken as the
//!   step's mean per-layer time times the layers ahead of it on this GPU.
//! - `cpu_gather_prefetch`: `cpu_gather` started one step early. The Engram
//!   hashes depend only on token ids, and the scheduler knows step `k`'s
//!   tokens while step `k-1` runs, so the host gather and the H2D copies of
//!   step `k`'s rows start at step `k-1`'s start. They share nothing with step
//!   `k-1` but the host threads and the copy engine (each step's threads
//!   gather one step's rows, the next step's), so in the same formula the
//!   rows are ready `T_prev` earlier: `ready_k = (k + 1) * gather + h2d -
//!   T_prev`. `T_prev`, the previous step's GPU time, is taken as this step's
//!   own GPU time before the wait (the arch evaluates one step without its
//!   history; a chunked-prefill engine's consecutive steps carry the same
//!   token budget). The unpack stays on the GPU, as under `cpu_gather`.
//!
//! Measured inputs (experiments/20261114_engram_cpu_gather, B200 host):
//! - gather: 393,216 rows per rank (8192 tokens x 2 layers x 24 hash columns)
//!   in 6.645 ms median, 4 ranks x 8 threads at once, THP on, threads on the
//!   table's NUMA node (`logs/cpu_gather2.log`);
//! - H2D: one layer's 51,904,512 B in 1.150 ms (45.1 GB/s) with 4 GPUs
//!   copying at once while a GEMM and the CPU gather run
//!   (`results/h2d_table.md`, `all4gpu_with_cpu_gather_52thr`); the GEMM ran
//!   at most 0.9% slower beside it, so the copy is charged no SM time.
//!
//! The unpack has no measured kernel of its own: it is priced as the profiled
//! device-resident `engram_lookup` over the same row count (its 6-column
//! geometry), a random gather from HBM, which bounds a sequential read of the
//! packed buffer from above.

use crate::arch::deepseek_v41_vllm::ENGRAM_TABLE_ROWS_RANK0;
use crate::worklet::DeepseekV41EngramPrefetchLocalWorkletConfig;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EngramFetch {
    #[default]
    Uva,
    CpuGather,
    /// `cpu_gather` with the next step's rows gathered during this step.
    CpuGatherPrefetch,
    /// The fetch priced out (the Engram blocks' compute stays): the engine's
    /// own fetch path cannot be measured here.
    Off,
}

/// `uva` / `cpu_gather` / `off`, or a bool (`true` = uva, `false` = off) as
/// the DeepInfer engine option spells it.
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
                "cpu_gather" => Ok(Self::CpuGather),
                "cpu_gather_prefetch" => Ok(Self::CpuGatherPrefetch),
                "off" | "false" => Ok(Self::Off),
                other => Err(serde::de::Error::custom(format!(
                    "engram_fetch: expected uva, cpu_gather, cpu_gather_prefetch or off, got {other}"
                ))),
            },
        }
    }
}

/// Rows one rank gathers per ms with four ranks gathering at once.
const GATHER_ROWS_PER_MS: f64 = 393_216.0 / 6.645;
/// H2D bytes per ms per GPU with four GPUs copying at once.
const H2D_BYTES_PER_MS: f64 = 51_904_512.0 / 1.150;
/// Hash columns of the profiled device-resident lookup geometry.
const UNPACK_PROFILED_HEADS: u32 = 6;

/// The `cpu_gather` fetch of one GPU: its row geometry and the per-row unpack.
#[derive(Clone, Copy, Debug)]
pub struct EngramHostGather {
    /// Rows per token on this GPU (its hash columns).
    pub rows_per_token: u32,
    /// Bytes per row: FP8 values plus UE8M0 scales.
    pub row_bytes: u32,
    /// Tokens of the profiled unpack per scheduled token (same row count).
    pub unpack_token_scale: u32,
    /// `cpu_gather_prefetch`: the rows are fetched during the previous step.
    pub prefetch: bool,
}

impl EngramHostGather {
    /// Turn the UVA prefetch config into the unpack's (the device-resident
    /// lookup over the same rows) and describe the host side.
    pub fn from_prefetch(prefetch: &mut DeepseekV41EngramPrefetchLocalWorkletConfig) -> Self {
        let local_heads = prefetch.engram_num_heads / prefetch.tp_size;
        assert_eq!(
            local_heads % UNPACK_PROFILED_HEADS,
            0,
            "cpu_gather unpack: {local_heads} hash columns per GPU is not a multiple of the profiled {UNPACK_PROFILED_HEADS}"
        );
        let gather = Self {
            rows_per_token: local_heads,
            row_bytes: prefetch.engram_head_dim
                + prefetch.engram_head_dim / prefetch.quant_block_size,
            unpack_token_scale: local_heads / UNPACK_PROFILED_HEADS,
            prefetch: false,
        };
        prefetch.tp_size = prefetch.engram_num_heads / UNPACK_PROFILED_HEADS;
        prefetch.table_rows = ENGRAM_TABLE_ROWS_RANK0;
        prefetch.residency = "device".into();
        gather
    }

    /// [`Self::from_prefetch`] for `fetch` (`cpu_gather` or
    /// `cpu_gather_prefetch`; `None` for the others).
    pub fn for_fetch(
        fetch: EngramFetch,
        prefetch: &mut DeepseekV41EngramPrefetchLocalWorkletConfig,
    ) -> Option<Self> {
        let early = match fetch {
            EngramFetch::CpuGather => false,
            EngramFetch::CpuGatherPrefetch => true,
            EngramFetch::Uva | EngramFetch::Off => return None,
        };
        Some(Self {
            prefetch: early,
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
        let gather = rows / GATHER_ROWS_PER_MS;
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

#[cfg(test)]
mod tests {
    use super::*;

    fn gather() -> EngramHostGather {
        EngramHostGather {
            rows_per_token: 24,
            row_bytes: 264,
            unpack_token_scale: 4,
            prefetch: false,
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
            (cfg.tp_size, cfg.table_rows, cfg.residency.as_str()),
            (4, 96_000_564, "device")
        );
    }
}
