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

    /// One-line shape/dtype config summary for the compiled leaf's manifest entry.
    fn describe_config(&self) -> String;
}
