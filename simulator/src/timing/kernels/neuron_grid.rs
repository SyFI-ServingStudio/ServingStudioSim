//! Shared sweep grids for the AWS Trainium2 (Neuron) L1 kinds.
//!
//! The binding source is `doc/pr/trn2-flashnext/L1_CONTRACT.md`: its named
//! grids live here once, and every Neuron kind (the Neuron-only kinds and the
//! Neuron branch of the shared GPU kinds) reads them from this module so the
//! profiled rows and the Rust caches cannot drift per kind.
//!
//! A config takes the Neuron grid only when [`is_neuron`] holds — every
//! candidate backend is a `neuron_<impl>` one. A mixed or GPU config keeps the
//! kind's GPU grid unchanged.

use crate::timing::sweep::{Axis, SweepGrid};

/// Prefix every Neuron backend carries (`neuron_<impl>`).
pub const NEURON_BACKEND_PREFIX: &str = "neuron_";

/// Token rows of one step (decode through the largest prefill step).
pub const TOKENS: [u32; 14] = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
];
/// Decode / MTP-verify requests in one step.
pub const DECODE: [u32; 9] = [1, 2, 4, 8, 16, 32, 64, 128, 256];
/// Tokens of one prefill request chunk.
pub const CHUNK: [u32; 8] = [16, 128, 256, 512, 1024, 2048, 4096, 8192];
/// QSA prefill rows (tokens of the one request chunk).
pub const QSA_PREFILL_ROWS: [u32; 4] = [128, 512, 2048, 8192];
/// QSA prefill prior context (visible tokens before the chunk).
pub const QSA_PRIOR: [u32; 2] = [0, 32768];
/// QSA decode rows (requests, one row each).
pub const QSA_DECODE_ROWS: [u32; 4] = [1, 8, 64, 256];
/// QSA decode KV length, including the new token.
pub const QSA_CONTEXT: [u32; 3] = [4096, 32768, 131072];
/// Per-rank collective message sizes, in bytes.
pub const COMM_BYTES: [u64; 5] = [16384, 262144, 1048576, 4194304, 16777216];
/// Back-to-back compiled regions launched in one step.
pub const REGIONS: [u32; 7] = [1, 2, 4, 8, 16, 32, 64];

/// True when every candidate backend is a Neuron one (and there is at least
/// one). The single predicate that switches a shared kind to its Neuron grid.
pub fn is_neuron(backends: &[&str]) -> bool {
    !backends.is_empty()
        && backends
            .iter()
            .all(|backend| backend.starts_with(NEURON_BACKEND_PREFIX))
}

/// Reject a Neuron-only kind configured with a GPU (or no) backend: its grid
/// and profiler rows exist only for Neuron.
pub fn assert_neuron(kind: &str, backends: &[&str]) {
    assert!(
        is_neuron(backends),
        "{kind} is a Neuron kind: every backend must start with \
         {NEURON_BACKEND_PREFIX:?}, got {backends:?}"
    );
}

/// The axis of one of the `u32` grids above.
pub fn axis<const N: usize>(values: [u32; N]) -> Vec<f64> {
    Axis::values(values)
}

/// The 1D grid over one of the `u32` grids above.
pub fn grid_1d<const N: usize>(values: [u32; N]) -> SweepGrid {
    SweepGrid::new(vec![axis(values)])
}

/// The 1D [`COMM_BYTES`] grid (byte sizes exceed nothing in `u32`, but they
/// are `u64` on the wire like every collective's `message_size_bytes`).
pub fn comm_bytes_grid() -> SweepGrid {
    SweepGrid::new(vec![COMM_BYTES.iter().map(|&b| b as f64).collect()])
}

/// QSA `mode` values: one request chunk (`prefill`) or one row per request
/// (`decode`).
pub const QSA_MODES: [&str; 2] = ["prefill", "decode"];

/// Whether a QSA `mode` is prefill; panics on an unknown mode.
pub fn qsa_is_prefill(kind: &str, mode: &str) -> bool {
    assert!(
        QSA_MODES.contains(&mode),
        "{kind} mode must be one of {QSA_MODES:?}, got {mode:?}"
    );
    mode == "prefill"
}

/// Row-only QSA grid: `QSA_PREFILL_ROWS` in prefill, `QSA_DECODE_ROWS` in decode.
pub fn qsa_rows_grid(kind: &str, mode: &str) -> SweepGrid {
    if qsa_is_prefill(kind, mode) {
        grid_1d(QSA_PREFILL_ROWS)
    } else {
        grid_1d(QSA_DECODE_ROWS)
    }
}

/// Context-dependent QSA grid: prefill `QSA_PREFILL_ROWS x QSA_PRIOR` (cache
/// axis 1 = prior), decode `QSA_DECODE_ROWS x QSA_CONTEXT` (axis 1 = KV length).
pub fn qsa_context_grid(kind: &str, mode: &str) -> SweepGrid {
    if qsa_is_prefill(kind, mode) {
        SweepGrid::new(vec![axis(QSA_PREFILL_ROWS), axis(QSA_PRIOR)])
    } else {
        SweepGrid::new(vec![axis(QSA_DECODE_ROWS), axis(QSA_CONTEXT)])
    }
}

/// The QSA payload `context_len` of a grid cell `(rows, axis1)`: prefill
/// `prior + rows`, decode `axis1` itself.
pub fn qsa_payload_context_len(prefill: bool, rows: f64, axis1: f64) -> u32 {
    let (rows, axis1) = (rows.round() as u32, axis1.round() as u32);
    if prefill {
        axis1 + rows
    } else {
        axis1
    }
}

/// The QSA cache axis 1 of a physical `(num_rows, context_len)`: prefill the
/// prior `context_len - num_rows`, decode `context_len`.
pub fn qsa_cache_axis1(prefill: bool, num_rows: u32, context_len: u32) -> f64 {
    if prefill {
        f64::from(context_len.saturating_sub(num_rows))
    } else {
        f64::from(context_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Catches a GPU or mixed backend list switching a shared kind to the
    /// Neuron grid (which would resample every GPU row).
    #[test]
    fn neuron_predicate_needs_every_backend() {
        assert!(is_neuron(&["neuron_torch"]));
        assert!(is_neuron(&["neuron_torch", "neuron_nki"]));
        assert!(!is_neuron(&["neuron_torch", "vllm_triton"]));
        assert!(!is_neuron(&["torch"]));
        assert!(!is_neuron(&[]));
    }

    /// Catches a prefill cell whose payload context is not prior + rows, or a
    /// cache projection that is not its inverse.
    #[test]
    fn qsa_prefill_context_round_trips() {
        let context = qsa_payload_context_len(true, 2048.0, 32768.0);
        assert_eq!(context, 34816);
        assert_eq!(qsa_cache_axis1(true, 2048, context), 32768.0);
        assert_eq!(qsa_payload_context_len(false, 64.0, 4096.0), 4096);
        assert_eq!(qsa_cache_axis1(false, 64, 4096), 4096.0);
    }
}

#[cfg(test)]
mod qsa_cache_tests {
    use super::*;
    use crate::timing::bridge::KernelMetrics;
    use crate::timing::cache::{Cache, Cache2DLinear, Extrapolation};

    fn fit(grid: &SweepGrid, policy: Extrapolation) -> Cache2DLinear {
        let samples: Vec<KernelMetrics> = grid
            .expand_2d(|rows, ctx| 1.0 + rows * 1e-3 + ctx * 1e-4)
            .into_iter()
            .map(|time_ms| KernelMetrics {
                time_ms,
                tflops: None,
                memory_bandwidth_gbps: Some(100.0),
                algbw_gbps: None,
                busbw_gbps: None,
                energy_j: 0.0,
            })
            .collect();
        Cache2DLinear::from_samples_with(grid, &samples, policy).0
    }

    /// `cache_kind` cannot see `mode`, so the context-dependent QSA kinds
    /// declare `Weighted` for both modes. Catches a prefill grid that gains
    /// enough top-band cells for `Weighted` to fit slopes, which would silently
    /// turn the contract's prefill `Clamp` into linear growth past the prior.
    #[test]
    fn weighted_is_clamp_on_the_prefill_grid_and_fits_slopes_on_decode() {
        let prefill = qsa_context_grid("qsa_attention", "prefill");
        let (weighted, clamp) = (
            fit(&prefill, Extrapolation::Weighted),
            fit(&prefill, Extrapolation::Clamp),
        );
        for point in [[2048.0, 131072.0], [16384.0, 65536.0], [64.0, 100.0]] {
            let (w, c) = (weighted.eval(&point).m, clamp.eval(&point).m);
            assert_eq!((w.time_ms, w.bytes), (c.time_ms, c.bytes), "{point:?}");
        }

        let decode = qsa_context_grid("qsa_attention", "decode");
        let far = [256.0, 262144.0];
        let weighted = fit(&decode, Extrapolation::Weighted).eval(&far).m.time_ms;
        let clamp = fit(&decode, Extrapolation::Clamp).eval(&far).m.time_ms;
        assert!(weighted > clamp, "decode extends past QSA_CONTEXT");
    }
}
