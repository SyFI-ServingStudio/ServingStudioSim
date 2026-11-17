//! Shared sweep grids for the AWS Trainium2 (Neuron) L1 kinds.
//!
//! The binding source is `doc/pr/trn2-glm53-flash/L1_CONTRACT.md`: its named
//! grids live here once, and every Neuron kind reads them from this module so
//! the imported rows and the Rust caches cannot drift per kind. (Ported from
//! the `trn2-flashnext` branch and cut to what the GLM-5.3-Flash kinds use.)
//!
//! A kind takes a Neuron grid only when [`is_neuron`] holds — every candidate
//! backend is a `neuron_<impl>` one.

use crate::timing::sweep::{Axis, SweepGrid};

/// Prefix every Neuron backend carries (`neuron_<impl>`).
pub const NEURON_BACKEND_PREFIX: &str = "neuron_";

/// Token rows of one region on one rank (decode through the largest chunk).
pub const TOKENS: [u32; 14] = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
];
/// Decode requests in one step (per DP group).
pub const DECODE: [u32; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];
/// Tokens of one prefill request chunk.
pub const CHUNK: [u32; 7] = [128, 256, 512, 1024, 2048, 4096, 8192];
/// DSA prefill chunk rows.
pub const DSA_PREFILL_ROWS: [u32; 7] = [128, 256, 512, 1024, 2048, 4096, 8192];
/// DSA context buckets (the region reads the latent cache `[:C]`).
pub const DSA_CONTEXT: [u32; 5] = [2048, 4096, 8192, 16384, 32768];
/// Rows gathered over the world for the FFN (every DP group's rows).
pub const FFN_ROWS: [u32; 14] = [
    8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
];
/// lm_head rows (decode rows or one per prefill chunk, per DP group).
pub const HEAD_ROWS: [u32; 11] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];
/// Per-rank collective message sizes, in bytes.
pub const COMM_BYTES: [u64; 5] = [16384, 262144, 1048576, 4194304, 16777216];

/// True when every candidate backend is a Neuron one (and there is at least one).
pub fn is_neuron(backends: &[&str]) -> bool {
    !backends.is_empty()
        && backends
            .iter()
            .all(|backend| backend.starts_with(NEURON_BACKEND_PREFIX))
}

/// Reject a Neuron-only kind configured with a GPU (or no) backend.
pub fn assert_neuron(kind: &str, backends: &[&str]) {
    assert!(
        is_neuron(backends),
        "{kind} is a Neuron kind: every backend must start with \
         {NEURON_BACKEND_PREFIX:?}, got {backends:?}"
    );
}

/// The axis of one of the `u32` grids above.
pub fn axis(values: &[u32]) -> Vec<f64> {
    Axis::values(values.iter().copied())
}

/// The 1D grid over one of the `u32` grids above.
pub fn grid_1d<const N: usize>(values: [u32; N]) -> SweepGrid {
    SweepGrid::new(vec![axis(&values)])
}

/// The 1D [`COMM_BYTES`] grid.
pub fn comm_bytes_grid() -> SweepGrid {
    SweepGrid::new(vec![COMM_BYTES.iter().map(|&b| b as f64).collect()])
}

/// DSA grid: `rows x DSA_CONTEXT`.
pub fn dsa_grid(rows: &[u32]) -> SweepGrid {
    SweepGrid::new(vec![axis(rows), axis(&DSA_CONTEXT)])
}

/// Row-major mask of the 2D cells whose rows exceed the context (a chunk
/// always fits in the context it reads).
pub fn rows_over_context_mask(grid: &SweepGrid) -> Vec<bool> {
    grid.expand_2d(|rows, context| rows > context)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Catches a GPU or mixed backend list switching a kind to the Neuron grid.
    #[test]
    fn neuron_predicate_needs_every_backend() {
        assert!(is_neuron(&["neuron_torch"]));
        assert!(is_neuron(&["neuron_torch", "neuron_nki"]));
        assert!(!is_neuron(&["neuron_torch", "vllm_triton"]));
        assert!(!is_neuron(&[]));
    }

    #[test]
    fn prefill_mask_drops_rows_past_the_context() {
        let grid = dsa_grid(&DSA_PREFILL_ROWS);
        let mask = rows_over_context_mask(&grid);
        assert_eq!(mask.len(), 35);
        // 4096 rows at context 2048 and 8192 rows at 2048 / 4096.
        assert_eq!(mask.iter().filter(|&&m| m).count(), 3);
    }
}
