//! Paged batch prefill attention: FlashInfer's `trtllm_batch_context_with_kv_cache`
//! as vLLM's `FlashInferImpl.forward` calls it for the prefill requests of a
//! step (vllm/v1/attention/backends/flashinfer.py): causal attention of each
//! request's new tokens over its cached prefix and themselves. The step's
//! decode requests are a separate call, `paged_attn_batch_decode`.
//!
//! The kernel cuts each request's new tokens into 128-token query tiles and
//! runs one CTA per (tile, query head); a tile reads every key up to its last
//! token, and a partial tile costs a full one. Measured on B200, one request of
//! 16 or 512 new tokens after 131072 cached ones takes the same 1.05 ms (four
//! tiles of 24 heads, one wave of CTAs), 1024 tokens take twice that, and the
//! time steps at every wave edge (`tiles * q_heads / SMs`).
//!
//! A batch projects to `(num_seqs, query_tiles, equivalent_prefix)`. The rows
//! are canonical batches: `num_seqs` requests sharing `query_tiles` whole tiles
//! evenly (the remainder one tile each to the first) after one common prefix.
//! A real batch keeps its request and tile counts, and its equivalent prefix is
//! the common prefix that gives the canonical batch the keys its own tiles read
//! (`Σ over tiles of the prefix plus the tile's last position`). The tile axis
//! holds both sides of every wave edge up to `WAVE_KNOTS` waves.
//!
//! A step whose keys come mostly from one long request beside many short ones
//! runs longer than the even batch: the long request's tiles all read its whole
//! prefix while the short ones finish early.

use crate::timing::bridge::{de_backends, ArgsPayload, DType, KernelKind};
use crate::timing::cache::CacheKind;
use crate::timing::kernels::engine::{register_kernel, KernelSpec};
use crate::timing::kernels::fa4_varlen_projection::B200_SMS;
use crate::timing::sweep::{Axis, SweepGrid};
use crate::timing::{Coords, Dim, KernelConfig, SweepCoords};

/// New tokens per query tile (trtllm-gen context kernel, FP8 query, measured).
const QUERY_TILE: u64 = 128;
/// Waves whose both edges are tile knots; past them a wave adds at most a
/// ninth of the time and the tail is enough.
const WAVE_KNOTS: u32 = 8;
/// Tile knots past the wave edges, up to 8192 new tokens beside 64 short
/// prompts.
const TILE_TAIL: [u32; 3] = [64, 96, 128];
const SEQ_KNOTS: [u32; 10] = [1, 2, 3, 4, 6, 8, 12, 16, 32, 64];
const PREFIX_KNOTS: [u32; 6] = [0, 512, 2048, 8192, 32768, 131072];
/// Profiling cap on one batch's KV bytes (the runner's `_MAX_KV_BYTES`).
const MAX_KV_BYTES: f64 = (32u64 << 30) as f64;

#[derive(KernelConfig, Hash, PartialEq, Eq, Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PagedAttnBatchPrefillKernelConfig {
    #[serde(deserialize_with = "de_backends")]
    pub backends: Vec<&'static str>,
    pub gpu_name: String,
    pub num_qo_heads: Dim,
    pub num_kv_heads: Dim,
    pub head_dim: Dim,
    #[compute_dtype]
    pub q_dtype: DType,
    #[kv_dtype]
    pub kv_dtype: DType,
    pub o_dtype: DType,
}

impl PagedAttnBatchPrefillKernelConfig {
    /// Tile knots: both sides of each wave edge for the first waves, then the tail.
    fn tile_axis(&self) -> Vec<f64> {
        let heads = self.num_qo_heads.get();
        let last_in_wave = |wave: u32| (f64::from(wave) * B200_SMS) as u32 / heads;
        let mut knots: Vec<u32> = (1..=last_in_wave(1).clamp(1, 4)).collect();
        for wave in 1..=WAVE_KNOTS {
            let last = last_in_wave(wave);
            if last >= 1 {
                knots.extend([last, last + 1]);
            }
        }
        knots.extend(TILE_TAIL);
        knots.retain(|&tiles| tiles <= *TILE_TAIL.last().unwrap());
        knots.sort_unstable();
        knots.dedup();
        Axis::values(knots)
    }

    fn kv_bytes_per_token(&self) -> f64 {
        2.0 * f64::from(self.num_kv_heads.get() * self.head_dim.get())
            * f64::from(self.kv_dtype.size_bytes())
    }
}

/// The prefill requests of one step.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct PagedAttnBatchPrefillKernelInput {
    /// `(cached prefix, new tokens)` per request.
    pub requests: Vec<(u32, u32)>,
}

fn tiles_of(new: u32) -> u64 {
    u64::from(new).div_ceil(QUERY_TILE)
}

/// Σ over a request's tiles of the keys each reads (prefix + last position).
fn tile_keys(prefix: u32, new: u32) -> u64 {
    let tiles = tiles_of(new);
    let full = tiles.saturating_sub(1);
    tiles * u64::from(prefix) + QUERY_TILE * full * (full + 1) / 2 + u64::from(new)
}

/// Tiles per request of the canonical batch: `tiles` split evenly over
/// `seqs`, the remainder one each to the first.
fn canonical_tiles(seqs: u64, tiles: u64) -> impl Iterator<Item = u64> {
    let (base, extra) = (tiles / seqs, tiles % seqs);
    (0..seqs).map(move |i| base + u64::from(i < extra))
}

impl SweepCoords for PagedAttnBatchPrefillKernelInput {
    fn coords(&self) -> Coords {
        let seqs = self.requests.len() as u64;
        let tiles: u64 = self.requests.iter().map(|&(_, new)| tiles_of(new)).sum();
        if seqs == 0 || tiles == 0 {
            return Coords::new([seqs as f64, tiles as f64, 0.0]);
        }
        let keys: u64 = self
            .requests
            .iter()
            .map(|&(p, new)| tile_keys(p, new))
            .sum();
        // The canonical batch's tile keys with no prefix.
        let own: u64 = canonical_tiles(seqs, tiles)
            .map(|t| QUERY_TILE * t * (t + 1) / 2)
            .sum();
        let prefix = keys.saturating_sub(own) as f64 / tiles as f64;
        Coords::new([seqs as f64, tiles as f64, prefix])
    }
    fn coord_field_names() -> &'static [&'static str] {
        &["num_seqs", "query_tiles", "equivalent_prefix"]
    }
}

/// The canonical batch at a cell, as `(prefix, new tokens)` per request.
fn canonical(seqs: f64, tiles: f64, prefix: f64) -> Vec<(u32, u32)> {
    canonical_tiles(seqs as u64, tiles as u64)
        .map(|t| (prefix as u32, (t * QUERY_TILE) as u32))
        .collect()
}

pub struct PagedAttnBatchPrefillSpec;

impl KernelSpec for PagedAttnBatchPrefillSpec {
    type Config = PagedAttnBatchPrefillKernelConfig;
    type Input = PagedAttnBatchPrefillKernelInput;

    const KIND: KernelKind = "paged_attn_batch_prefill";

    fn sweep_grid(config: &Self::Config) -> SweepGrid {
        assert!(
            config.num_qo_heads.get() % config.num_kv_heads.get() == 0,
            "num_qo_heads must be a multiple of num_kv_heads"
        );
        SweepGrid::new(vec![
            Axis::values(SEQ_KNOTS),
            config.tile_axis(),
            Axis::values(PREFIX_KNOTS),
        ])
    }

    fn cache_kind(_backend: &'static str) -> CacheKind {
        CacheKind::Cache3DLinear
    }

    fn infeasible_mask(config: &Self::Config, grid: &SweepGrid) -> Vec<bool> {
        // The cap bounds the KV pool; a request count past the tiles is not
        // masked but holds the one-tile batch (see `enumerate`).
        let per_token = config.kv_bytes_per_token();
        grid.expand_3d(|seqs, tiles, prefix| {
            let tokens = seqs.min(tiles) * prefix + tiles * QUERY_TILE as f64;
            tokens * per_token > MAX_KV_BYTES
        })
    }

    fn enumerate(
        config: &Self::Config,
        grid: &SweepGrid,
        backend: &'static str,
    ) -> Vec<ArgsPayload> {
        grid.expand_3d(|seqs, tiles, prefix| {
            // A request holds at least one tile, so past `tiles` requests the
            // count saturates: the cell holds the one-tile-per-request batch.
            // Masking it instead would leave a batch of short prompts, which
            // sits on the `seqs == tiles` edge, between a real and a dropped
            // corner.
            let requests = canonical(seqs.min(tiles), tiles, prefix);
            ArgsPayload::new()
                .with("backend", backend)
                .with("num_qo_heads", config.num_qo_heads.get())
                .with("num_kv_heads", config.num_kv_heads.get())
                .with("head_dim", config.head_dim.get())
                .with("q_dtype", config.q_dtype.as_str())
                .with("kv_dtype", config.kv_dtype.as_str())
                .with("o_dtype", config.o_dtype.as_str())
                .with("requests", serde_json::json!(requests))
        })
    }
}

register_kernel!(PagedAttnBatchPrefillKernel, PagedAttnBatchPrefillSpec);

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn config() -> PagedAttnBatchPrefillKernelConfig {
        PagedAttnBatchPrefillKernelConfig {
            backends: vec!["flashinfer_trtllm_gen"],
            gpu_name: "NVIDIA B200".to_string(),
            num_qo_heads: 24.into(),
            num_kv_heads: 4.into(),
            head_dim: 256.into(),
            q_dtype: DType::Fp8E4m3,
            kv_dtype: DType::Fp8E4m3,
            o_dtype: DType::Bf16,
        }
    }

    fn coords(requests: &[(u32, u32)]) -> Vec<f64> {
        PagedAttnBatchPrefillKernelInput {
            requests: requests.to_vec(),
        }
        .coords()
        .to_vec()
    }

    #[test]
    fn tiles_read_the_keys_up_to_their_last_token() {
        // Defect caught: a partial tile dropped, or a tile priced by its filled
        // rows instead of the keys up to its last token. 300 new tokens after
        // 1000 cached: tiles end at 128, 256 and 300.
        assert_eq!(tile_keys(1000, 300), 3 * 1000 + 128 + 256 + 300);
        assert_eq!(tile_keys(0, 16), 16);
    }

    #[test]
    fn canonical_rows_project_onto_their_own_cell() {
        // Defect caught: rows that land off the cell they fill.
        let cfg = config();
        let grid = PagedAttnBatchPrefillSpec::sweep_grid(&cfg);
        let mask = PagedAttnBatchPrefillSpec::infeasible_mask(&cfg, &grid);
        let cells = grid.expand_3d(|n, t, p| (n, t, p));
        for ((seqs, tiles, prefix), masked) in cells.into_iter().zip(mask) {
            if !masked && seqs <= tiles {
                let got = coords(&canonical(seqs, tiles, prefix));
                assert_eq!(
                    got,
                    vec![seqs, tiles, prefix],
                    "cell ({seqs}, {tiles}, {prefix})"
                );
            }
        }
    }

    #[test]
    fn a_partial_tile_counts_as_a_whole_one() {
        // Defect caught: 130 new tokens priced as 1.02 tiles. They run two
        // tiles, the canonical 256-token request with a slightly shorter prefix.
        let got = coords(&[(1000, 130)]);
        assert_eq!(&got[..2], &[1.0, 2.0]);
        assert!((got[2] - (2.0 * 1000.0 + 128.0 + 130.0 - 384.0) / 2.0).abs() < 1e-9);
    }

    #[test]
    fn a_long_chain_reads_more_than_its_share() {
        // Defect caught: a chunk after a long prefix diluted over the batch's
        // short prompts as if every request had the same prefix and tiles.
        let ragged = coords(&[(40000, 448), (0, 66), (0, 377)]);
        let even = coords(&[(0, 448), (0, 66), (0, 377)]);
        assert_eq!(&ragged[..2], &even[..2]);
        assert!((ragged[2] - even[2] - 40000.0 * 4.0 / 8.0).abs() < 1e-9);
    }

    #[test]
    fn tile_axis_brackets_each_wave_edge() {
        // Defect caught: knots that miss a wave edge. With 24 query heads a
        // wave of 148 SMs holds 6 tiles: 768 new tokens run in one wave, 896 in
        // two.
        let axis = config().tile_axis();
        for tiles in [1.0, 6.0, 7.0, 12.0, 13.0, 49.0, 50.0, 128.0] {
            assert!(axis.contains(&tiles), "tile axis {axis:?} misses {tiles}");
        }
        assert!(axis.windows(2).all(|pair| pair[0] < pair[1]));
        let mut wide = config();
        wide.num_qo_heads = 64.into();
        wide.num_kv_heads = 8.into();
        assert!(wide.tile_axis().contains(&3.0) && wide.tile_axis().contains(&4.0));
    }

    #[test]
    fn mask_drops_only_cells_past_the_kv_cap() {
        // 32 KV heads hold 16 KiB per token, so the longest prefixes of the
        // largest batches pass the cap.
        let mut cfg = config();
        cfg.num_qo_heads = 64.into();
        cfg.num_kv_heads = 32.into();
        let grid = PagedAttnBatchPrefillSpec::sweep_grid(&cfg);
        let mask = PagedAttnBatchPrefillSpec::infeasible_mask(&cfg, &grid);
        let cells = grid.expand_3d(|n, t, p| (n, t, p));
        for (&masked, (seqs, tiles, prefix)) in mask.iter().zip(cells) {
            let bytes = (seqs.min(tiles) * prefix + tiles * 128.0) * 16384.0;
            assert_eq!(masked, bytes > MAX_KV_BYTES);
        }
        assert!(mask.iter().any(|&m| m) && mask.iter().any(|&m| !m));
    }

    #[test]
    fn requests_past_the_tiles_hold_the_one_tile_batch() {
        // Defect caught: a cell with more requests than tiles left without a
        // row, so a batch of short prompts falls between a row and a hole.
        let cfg = config();
        let grid = PagedAttnBatchPrefillSpec::sweep_grid(&cfg);
        let payloads = PagedAttnBatchPrefillSpec::enumerate(&cfg, &grid, "flashinfer_trtllm_gen");
        let cells = grid.expand_3d(|n, t, p| (n, t, p));
        let (index, _) = cells
            .iter()
            .enumerate()
            .find(|(_, &(n, t, p))| n == 64.0 && t == 7.0 && p == 512.0)
            .unwrap();
        assert_eq!(
            payloads[index].fields().get("requests"),
            Some(&serde_json::json!(vec![[512, 128]; 7]))
        );
    }

    #[test]
    fn enumerate_emits_the_canonical_batch_as_runner_args() {
        let cfg = config();
        let grid = PagedAttnBatchPrefillSpec::sweep_grid(&cfg);
        let payloads = PagedAttnBatchPrefillSpec::enumerate(&cfg, &grid, "flashinfer_trtllm_gen");
        assert_eq!(payloads.len(), grid.expand_3d(|_, _, _| ()).len());
        let fields = payloads[0].fields();
        // Wire schema of PagedAttnBatchPrefillArgs plus the backend.
        assert_eq!(fields.len(), 8);
        assert_eq!(fields.get("requests"), Some(&serde_json::json!([[0, 128]])));
        assert_eq!(fields.get("kv_dtype"), Some(&Value::from("fp8_e4m3")));
    }
}
