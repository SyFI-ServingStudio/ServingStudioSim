//! Shared lookup result tree returned by L1-L4 runtime cost APIs.
//!
//! Agent note: keep these types independent of any concrete kernel/cache.
//! Upper layers import them through `crate::timing::{LookupResult, Probe}`.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::common::time::Time;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LookupResult {
    /// `Arc<str>` (not `String`) so the hot `Kernel::lookup` path stamps the
    /// caller's dotted-path name onto a result with a refcount bump instead of a
    /// per-tick heap allocation.
    pub name: Arc<str>,
    pub time: Time,
    pub flops: u64,
    pub bytes: u64,
    pub energy_j: f64,
    /// Backend whose cache produced this result. `Some(_)` only on leaf
    /// results emitted from `BackendCache::lookup` (post best-of-N selection
    /// the winner's label propagates up through `Kernel::lookup`). Compound
    /// results built via `sum` are `None`: there is no single winning backend
    /// across heterogeneous parts — recurse into `breakdown` to attribute.
    pub selected_backend: Option<&'static str>,
    pub breakdown: Vec<LookupResult>,
    pub warnings: Vec<CoverageWarning>,
}

impl LookupResult {
    pub fn leaf(
        name: impl Into<Arc<str>>,
        time: Time,
        flops: u64,
        bytes: u64,
        energy_j: f64,
        warnings: Vec<CoverageWarning>,
    ) -> Self {
        Self {
            name: name.into(),
            time,
            flops,
            bytes,
            energy_j,
            selected_backend: None,
            breakdown: Vec::new(),
            warnings,
        }
    }

    pub fn sum(name: impl Into<Arc<str>>, parts: Vec<LookupResult>) -> Self {
        let time = parts.iter().fold(Time::ZERO, |acc, part| acc + part.time);
        let flops = parts.iter().map(|part| part.flops).sum();
        let bytes = parts.iter().map(|part| part.bytes).sum();
        let energy_j = parts.iter().map(|part| part.energy_j).sum();
        let warnings = parts
            .iter()
            .flat_map(|part| part.warnings.iter().cloned())
            .collect();
        Self {
            name: name.into(),
            time,
            flops,
            bytes,
            energy_j,
            selected_backend: None,
            breakdown: parts,
            warnings,
        }
    }

    /// Parallel composition: wallclock = max(parts.time) / overlap_factor,
    /// while flops / bytes / energy_j still sum, warnings flat-concat,
    /// breakdown is preserved, and selected_backend clears to None.
    ///
    /// `overlap_factor` ∈ (0, 1] models residual serialization in nominally
    /// parallel work: 1.0 = perfect overlap (L4 INV-11 pins this to 1.0
    /// everywhere — L4 only composes already-fully-fused stages); L3 worklets
    /// pass < 1.0 (typical 0.85–0.95) to widen the wallclock above the naive
    /// max.
    ///
    /// Boundary cases (L3 §5.3):
    /// - `parts.len() == 0` → degenerates to a zero node (matches `sum`).
    /// - `parts.len() == 1` → fast path: returns the only part with `name`
    ///   overridden; `overlap_factor` is NOT applied since a single-part
    ///   "parallel" composition has no parallelism to model.
    pub fn max(name: impl Into<Arc<str>>, parts: Vec<LookupResult>, overlap_factor: f32) -> Self {
        assert!(
            overlap_factor > 0.0 && overlap_factor <= 1.0,
            "overlap_factor must be in (0, 1], got {overlap_factor}"
        );
        if parts.len() == 1 {
            let mut only = parts.into_iter().next().unwrap();
            only.name = name.into();
            return only;
        }
        let max_time = parts
            .iter()
            .map(|part| part.time)
            .max()
            .unwrap_or(Time::ZERO);
        let time = if overlap_factor < 1.0 {
            Time::from_ns((max_time.as_ns() as f64 / overlap_factor as f64) as u64)
        } else {
            max_time
        };
        let flops = parts.iter().map(|part| part.flops).sum();
        let bytes = parts.iter().map(|part| part.bytes).sum();
        let energy_j = parts.iter().map(|part| part.energy_j).sum();
        let warnings = parts
            .iter()
            .flat_map(|part| part.warnings.iter().cloned())
            .collect();
        Self {
            name: name.into(),
            time,
            flops,
            bytes,
            energy_j,
            selected_backend: None,
            breakdown: parts,
            warnings,
        }
    }

    /// Empty placeholder leaf for inclusion-conditional components. L4 §4.5
    /// uses this when a layer skips a stage (e.g. pre-attention norm absent)
    /// so the breakdown tree retains a slot at the expected name with all
    /// metrics zeroed. Equivalent to `leaf(name, Time::ZERO, 0, 0, 0.0, vec![])`.
    pub fn zero(name: impl Into<Arc<str>>) -> Self {
        Self::leaf(name, Time::ZERO, 0, 0, 0.0, Vec::new())
    }

    /// Builder-style rename: overwrites `name` with the role label the caller
    /// wants exposed downstream (Perfetto span, breakdown report row,
    /// validator key). Pure setter — no other state is touched. Used by
    /// upper layers to stamp role names like `"embedding"` or
    /// `format!("layer{:02}", idx)` onto a sub-result returned by an opaque
    /// helper that doesn't know its caller-side role.
    pub fn with_label(mut self, label: impl Into<Arc<str>>) -> Self {
        self.name = label.into();
        self
    }

    pub fn tflops(&self) -> f64 {
        let seconds = self.time.as_s();
        if seconds == 0.0 {
            return 0.0;
        }
        self.flops as f64 / seconds / 1e12
    }

    pub fn gbps(&self) -> f64 {
        let seconds = self.time.as_s();
        if seconds == 0.0 {
            return 0.0;
        }
        self.bytes as f64 / seconds / 1e9
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoverageWarning {
    pub kind: CoverageKind,
    pub detail: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoverageKind {
    Extrapolated,
    Jit,
    /// Lookup had no usable measurement to return: either every fit-time sample
    /// was dropped as non-finite, the landed direct bucket was dropped, or the
    /// lookup coordinate itself was NaN. The returned metrics are all zero (a
    /// placeholder, not a real measurement); this kind exists so a 0-time result
    /// can't pass silently as valid downstream.
    NoCoverage,
}

pub trait Probe {
    type Input;

    fn lookup(&self, input: &Self::Input) -> LookupResult;

    /// Wallclock-only fast path for per-tick callers that just advance the sim
    /// clock. The default builds a full `LookupResult` and discards everything
    /// but `time`; impls backed by an interpolation cache (e.g. `Kernel`)
    /// override this to skip the name clone and metric/warning machinery.
    fn lookup_time(&self, input: &Self::Input) -> Time {
        self.lookup(input).time
    }
}

#[cfg(test)]
mod tests {
    use super::{CoverageKind, CoverageWarning, LookupResult};
    use crate::common::time::Time;

    fn leaf(name: &str, time_ms: f64) -> LookupResult {
        LookupResult::leaf(
            name.to_string(),
            Time::from_ms(time_ms),
            0,
            0,
            0.0,
            Vec::new(),
        )
    }

    fn leaf_full(name: &str, time_ms: f64, flops: u64, bytes: u64, energy_j: f64) -> LookupResult {
        LookupResult::leaf(
            name.to_string(),
            Time::from_ms(time_ms),
            flops,
            bytes,
            energy_j,
            Vec::new(),
        )
    }

    #[test]
    fn leaf_starts_with_no_selected_backend() {
        assert!(leaf("k", 1.0).selected_backend.is_none());
    }

    #[test]
    fn sum_loses_single_backend_attribution() {
        let mut a = leaf("a", 1.0);
        a.selected_backend = Some("torch");
        let mut b = leaf("b", 2.0);
        b.selected_backend = Some("torch");
        let compound = LookupResult::sum("op".to_string(), vec![a, b]);

        assert!(compound.selected_backend.is_none());
        // The leaves are still attributable via breakdown.
        assert_eq!(compound.breakdown[0].selected_backend, Some("torch"));
        assert_eq!(compound.breakdown[1].selected_backend, Some("torch"));
    }

    #[test]
    fn max_with_perfect_overlap_takes_longest_time() {
        let result = LookupResult::max(
            "parallel".to_string(),
            vec![leaf("a", 1.0), leaf("b", 3.0), leaf("c", 2.0)],
            1.0,
        );
        assert_eq!(result.time, Time::from_ms(3.0));
        assert_eq!(result.breakdown.len(), 3);
        assert!(result.selected_backend.is_none());
    }

    #[test]
    fn max_widens_wallclock_when_overlap_factor_below_one() {
        let result = LookupResult::max(
            "imperfect".to_string(),
            vec![leaf("a", 4.0), leaf("b", 6.0)],
            0.5,
        );
        // max(4, 6) = 6 ms, divided by 0.5 = 12 ms.
        assert_eq!(result.time, Time::from_ms(12.0));
    }

    #[test]
    fn max_sums_metrics_and_concats_warnings() {
        let mut a = leaf_full("a", 2.0, 100, 1_000, 0.5);
        a.warnings.push(CoverageWarning {
            kind: CoverageKind::Extrapolated,
            detail: "a-warn".to_string(),
        });
        let mut b = leaf_full("b", 4.0, 200, 2_000, 1.5);
        b.warnings.push(CoverageWarning {
            kind: CoverageKind::Jit,
            detail: "b-warn".to_string(),
        });

        let result = LookupResult::max("p".to_string(), vec![a, b], 1.0);
        assert_eq!(result.flops, 300);
        assert_eq!(result.bytes, 3_000);
        assert_eq!(result.energy_j, 2.0);
        assert_eq!(result.warnings.len(), 2);
    }

    #[test]
    fn max_single_part_fast_path_skips_overlap_division() {
        let only = leaf_full("inner", 5.0, 42, 84, 0.25);
        let result = LookupResult::max("outer".to_string(), vec![only], 0.5);

        // Single-part path returns parts[0] verbatim with name overridden;
        // overlap_factor is NOT applied, so wallclock stays at 5 ms.
        assert_eq!(&*result.name, "outer");
        assert_eq!(result.time, Time::from_ms(5.0));
        assert_eq!(result.flops, 42);
        assert_eq!(result.bytes, 84);
        assert!(
            result.breakdown.is_empty(),
            "fast path inlines the part — no nested breakdown"
        );
    }

    #[test]
    fn max_empty_parts_degenerates_to_zero() {
        let result = LookupResult::max("nothing".to_string(), Vec::new(), 1.0);
        assert_eq!(result.time, Time::ZERO);
        assert_eq!(result.flops, 0);
        assert_eq!(result.bytes, 0);
        assert_eq!(result.energy_j, 0.0);
        assert!(result.breakdown.is_empty());
    }

    #[test]
    #[should_panic(expected = "overlap_factor must be in (0, 1]")]
    fn max_panics_on_overlap_factor_above_one() {
        let _ = LookupResult::max("x".to_string(), vec![leaf("a", 1.0), leaf("b", 1.0)], 1.5);
    }

    #[test]
    #[should_panic(expected = "overlap_factor must be in (0, 1]")]
    fn max_panics_on_zero_overlap_factor() {
        let _ = LookupResult::max("x".to_string(), vec![leaf("a", 1.0), leaf("b", 1.0)], 0.0);
    }

    #[test]
    fn zero_returns_empty_leaf_with_given_name() {
        let z = LookupResult::zero("placeholder".to_string());
        assert_eq!(&*z.name, "placeholder");
        assert_eq!(z.time, Time::ZERO);
        assert_eq!(z.flops, 0);
        assert_eq!(z.bytes, 0);
        assert_eq!(z.energy_j, 0.0);
        assert!(z.breakdown.is_empty());
        assert!(z.warnings.is_empty());
    }

    #[test]
    fn with_label_overwrites_name_only() {
        let original = leaf_full("orig", 1.5, 10, 20, 0.1);
        let renamed = original.clone().with_label("renamed");

        assert_eq!(&*renamed.name, "renamed");
        assert_eq!(renamed.time, original.time);
        assert_eq!(renamed.flops, original.flops);
        assert_eq!(renamed.bytes, original.bytes);
        assert_eq!(renamed.energy_j, original.energy_j);
    }
}
