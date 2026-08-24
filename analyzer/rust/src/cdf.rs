//! Shared numeric kernels reused by every metric: percentile + CDF downsample +
//! the `(MetricStats, CdfSeries)` builders, plus the serde output shapes those
//! builders return. Pure functions over `&[f64]` so they unit-test without any
//! parquet. Ported from ref `analyze-rust` latency math.

use serde::Serialize;

/// Distribution summary of one metric over its sample population. `None` fields
/// mean "no samples" (e.g. a metric that needs ≥2 tokens on a 1-token run).
#[derive(Debug, Clone, Serialize)]
pub struct MetricStats {
    pub n: usize,
    pub mean: Option<f64>,
    pub p50: Option<f64>,
    pub p90: Option<f64>,
    pub p99: Option<f64>,
    pub max: Option<f64>,
}

/// One metric's CDF for the plotter: sorted `x` (metric value) vs cumulative
/// `y_pct` (0→100), downsampled to ≤`MAX_CDF_POINTS`, plus the percentile marks
/// the renderer draws as labeled vlines.
#[derive(Debug, Clone, Serialize)]
pub struct CdfSeries {
    pub key: String,
    pub label: String,
    pub unit: String,
    pub n: usize,
    pub x: Vec<f64>,
    pub y_pct: Vec<f64>,
    pub markers: CdfMarkers,
}

/// Percentile x-positions the plotter annotates on the curve.
#[derive(Debug, Clone, Serialize)]
pub struct CdfMarkers {
    pub p50: Option<f64>,
    pub p90: Option<f64>,
    pub p99: Option<f64>,
}

/// Max points kept per CDF curve — a sweep of millions of requests still plots
/// as ≤1000 evenly-spaced samples (the curve is visually identical).
pub const MAX_CDF_POINTS: usize = 1000;

/// Linear-interpolated percentile of an already-sorted ascending slice.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "sorted.len() is this metric's sample count, realistically far below 2^53; rank is derived from pct in [0, 100] times that bounded length, so its floor/ceil indices always land within [0, len-1] (nonnegative) and the length-to-f64 conversions used to compute/interpolate rank lose no precision"
)]
pub fn percentile_sorted(sorted: &[f64], pct: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = (pct / 100.0) * (sorted.len().saturating_sub(1)) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    Some(if lo == hi {
        sorted[lo]
    } else {
        let w = rank - lo as f64;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    })
}

/// Sort + drop non-finite **and negative** samples. The negative filter is a
/// deliberate latency assumption (all SLO metrics are ≥0) — a future metric that
/// can legitimately go negative (e.g. a signed imbalance delta) must NOT route
/// through this; give it its own cleaner rather than relaxing this one.
pub fn clean_nonnegative_sorted(samples: &[f64]) -> Vec<f64> {
    let mut v: Vec<f64> = samples
        .iter()
        .copied()
        .filter(|x| x.is_finite() && *x >= 0.0)
        .collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v
}

#[allow(
    clippy::cast_precision_loss,
    reason = "sorted.len() is this metric's sample count, realistically far below 2^53, so the mean's divisor conversion to f64 is exact"
)]
pub fn stats(sorted: &[f64]) -> MetricStats {
    if sorted.is_empty() {
        return MetricStats {
            n: 0,
            mean: None,
            p50: None,
            p90: None,
            p99: None,
            max: None,
        };
    }
    let sum: f64 = sorted.iter().sum();
    MetricStats {
        n: sorted.len(),
        mean: Some(sum / sorted.len() as f64),
        p50: percentile_sorted(sorted, 50.0),
        p90: percentile_sorted(sorted, 90.0),
        p99: percentile_sorted(sorted, 99.0),
        max: sorted.last().copied(),
    }
}

/// Evenly-spaced downsample of a sorted slice into `(x, y_pct)` CDF points,
/// `y_pct` rising to 100 at the last sample (ref's even-index method).
#[allow(
    clippy::cast_precision_loss,
    reason = "idx+1 and n are CDF sample indices/counts bounded by the population size, realistically far below 2^53"
)]
fn downsample(sorted: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = sorted.len();
    let count = n.min(MAX_CDF_POINTS);
    let mut x = Vec::with_capacity(count);
    let mut y = Vec::with_capacity(count);
    if count == 1 {
        x.push(sorted[0]);
        y.push(100.0);
    } else if count > 1 {
        for i in 0..count {
            let idx = (i * (n - 1)) / (count - 1);
            x.push(sorted[idx]);
            y.push(((idx + 1) as f64 / n as f64) * 100.0);
        }
    }
    (x, y)
}

/// Build the plot-ready series for one metric from raw samples (cleans + sorts
/// internally). `key`/`label`/`unit` describe the metric to the renderer.
pub fn cdf_series(key: &str, label: &str, unit: &str, samples: &[f64]) -> CdfSeries {
    let sorted = clean_nonnegative_sorted(samples);
    let (x, y_pct) = downsample(&sorted);
    CdfSeries {
        key: key.to_string(),
        label: label.to_string(),
        unit: unit.to_string(),
        n: sorted.len(),
        x,
        y_pct,
        markers: CdfMarkers {
            p50: percentile_sorted(&sorted, 50.0),
            p90: percentile_sorted(&sorted, 90.0),
            p99: percentile_sorted(&sorted, 99.0),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_interpolates() {
        let s = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        assert_eq!(percentile_sorted(&s, 50.0), Some(30.0));
        assert_eq!(percentile_sorted(&s, 90.0), Some(46.0));
        assert_eq!(percentile_sorted(&[], 50.0), None);
    }

    #[test]
    fn downsample_caps_and_ends_at_100() {
        let samples: Vec<f64> = (0..2000).map(|v| v as f64).collect();
        let series = cdf_series("x", "X", "ms", &samples);
        assert_eq!(series.n, 2000);
        assert_eq!(series.x.len(), MAX_CDF_POINTS);
        assert_eq!(series.y_pct.last().copied(), Some(100.0));
        // monotonic non-decreasing in both axes
        assert!(series.x.windows(2).all(|w| w[0] <= w[1]));
        assert!(series.y_pct.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn clean_drops_negative_and_nan() {
        let sorted = clean_nonnegative_sorted(&[3.0, -1.0, f64::NAN, 1.0]);
        assert_eq!(sorted, vec![1.0, 3.0]);
    }

    #[test]
    fn stats_empty_is_none() {
        let s = stats(&[]);
        assert_eq!(s.n, 0);
        assert!(s.p50.is_none());
    }
}
