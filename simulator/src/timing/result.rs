//! `Probe` — the per-leaf cost-query trait for the CostTree eval walk.
//!
//! Agent note: keep this independent of any concrete kernel/cache. Upper layers
//! import it through `crate::timing::Probe`. The old `LookupResult` tree +
//! `lookup`/`lookup_time` and the `Describe` trait were retired once the CostTree
//! became the single cost path: leaves are evaluated via `eval` and the
//! shape print is rendered by [`CostTree::describe`](crate::timing::CostTree) from
//! the kernel `kind`/`config` captured at compile.

use crate::timing::LeafMetrics;

pub trait Probe {
    type Input;

    /// Metrics + coverage for one leaf — the per-leaf `buf[slot]` value the
    /// CostTree eval walk streams in (then [`CostTree::aggregate`] rolls up). For
    /// `Kernel` this is the alloc-free best-of-N over the backend caches.
    fn eval(&self, input: &Self::Input) -> LeafMetrics;

    /// Kernel KIND tag for the compiled leaf's manifest entry (the `(<KIND>)` in
    /// the shape render). Folds in the old `Describe` leaf line.
    fn kind(&self) -> &'static str;

    /// Structured config for the compiled manifest entry. `Dim` fields contain
    /// their folded value plus expression provenance.
    fn describe_config(&self) -> serde_json::Value;
}

/// Object-safe sibling of [`Probe`] for cost-model *introspection* (the
/// `kernel-query` subcommand): query a built kernel's cache by either a
/// **physical / natural** shape or its cache coordinates, and read back the
/// interpolated metrics per backend plus the fitted grid.
///
/// Unlike `Probe` (whose `eval` takes the typed `Self::Input`), this is `dyn`-able
/// — `eval_json` takes the kernel's Input as JSON, **deserializes it straight into
/// the kernel's existing `Self::Input` struct**, and calls the kernel's existing
/// [`eval`](crate::timing::Probe::eval). So the real `coords()` projection and
/// best-of-N run unchanged — the harness reuses the kernel, it doesn't reimplement
/// anything. One blanket impl over `Kernel<S>` (where `S::Input: Deserialize`)
/// covers every kernel; adding a kernel just needs `#[derive(Deserialize)]` on its
/// Input.
pub trait CacheProbe {
    /// Kernel KIND tag (the profile.db table / Python facade stem).
    fn kind(&self) -> &'static str;

    /// Structured config (same contract as [`Probe::describe_config`]).
    fn describe_config(&self) -> serde_json::Value;

    /// The sweep grid this cache was fitted on, in coords space: one ascending
    /// `Vec<f64>` per axis (so a caller can place off-grid probes).
    fn grid_axes(&self) -> Vec<Vec<f64>>;

    /// Best-of-N interpolated metrics for an Input given as JSON (the kernel's
    /// own fields, e.g. `{"prefix_len":0,"append_len":192}`). Errors if the JSON
    /// doesn't match the kernel's Input schema.
    fn eval_json(&self, input: &serde_json::Value) -> anyhow::Result<LeafMetrics>;

    /// Best-of-N interpolated metrics at an explicit point in cache-coordinate
    /// space. This is the authoritative path for inspecting a declared grid:
    /// cache axes need not be physical scalar Input fields (ragged and re-axis
    /// kernels are the common counterexamples).
    fn eval_coords(&self, coords: &[f64]) -> anyhow::Result<LeafMetrics>;

    /// Peak achieved compute / BW rates over the kernel's fitted grid — the
    /// per-config "best batching" ceiling. Reads the built cache cells directly
    /// (no re-eval, no coords remap), so it is correct for re-axis kernels too.
    fn peak_rates(&self) -> crate::timing::cache::PeakRates;
}
