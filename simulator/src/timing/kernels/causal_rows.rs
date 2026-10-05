//! Shared by the DSA prefill kinds: the query axis a step's batch is measured
//! on, and closed-form sums of the rows each query row of a causal prefill
//! request reads, for kernels whose logical work counts them. Row `x` of a
//! `(queries, context)` request runs over `context - queries + 1 ..= context`.

use crate::timing::sweep::Axis;

/// Queries in one step: a kernel's measured points `base`, cut at the most
/// a step runs (`max_num_batched_tokens`), which ends the axis when it is
/// below the last point. A step budget past `base` (an unchunked worker's
/// max_model_len) keeps `base`: a longer step is answered past the grid
/// (`off_grid`), not measured.
pub(crate) fn query_axis(base: &[u32], max_num_batched_tokens: u32) -> Vec<f64> {
    let mut values: Vec<u32> = base
        .iter()
        .copied()
        .filter(|&queries| queries < max_num_batched_tokens)
        .collect();
    if values.len() < base.len() {
        values.push(max_num_batched_tokens);
    }
    Axis::values(values)
}

/// The query points the DSA prefill kinds over a whole request batch are
/// measured at.
pub(crate) const PREFILL_QUERIES: [u32; 11] = [1, 4, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192];

/// `Σ min(x, cap)`: rows of a window (or top-k) that grows with the position
/// until it is full.
pub(crate) fn capped(queries: u32, context: u32, cap: u32) -> f64 {
    let (first, last, cap) = (
        f64::from(context - queries + 1),
        f64::from(context),
        f64::from(cap),
    );
    let ramp_last = last.min(cap);
    let ramp = if ramp_last >= first {
        (first + ramp_last) * (ramp_last - first + 1.0) / 2.0
    } else {
        0.0
    };
    ramp + cap * (last - ramp_last.max(first - 1.0))
}

/// `Σ min(⌊x / ratio⌋, cap)`: compressed rows, one per `ratio` positions,
/// with at most `cap` selected (`None` for no cap).
pub(crate) fn compressed(queries: u32, context: u32, ratio: u32, cap: Option<u32>) -> f64 {
    let ratio = u64::from(ratio);
    let prefix = |n: u64| -> u64 {
        // Σ_{x=1..=n} min(⌊x/ratio⌋, cap)
        let floor_sum = |n: u64| {
            let m = n / ratio;
            ratio * m * m.saturating_sub(1) / 2 + m * (n + 1 - m * ratio)
        };
        match cap.map(u64::from) {
            Some(0) => 0,
            Some(cap) if n >= cap * ratio => {
                floor_sum(cap * ratio - 1) + cap * (n + 1 - cap * ratio)
            }
            _ => floor_sum(n),
        }
    };
    (prefix(u64::from(context)) - prefix(u64::from(context - queries))) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_query_axis_ends_at_the_most_a_step_runs() {
        // An 8192-token chunk keeps the grid its rows were measured on.
        assert_eq!(
            query_axis(&PREFILL_QUERIES, 8192),
            Axis::values(PREFILL_QUERIES)
        );
        // A smaller chunk stops there, an odd one included.
        assert_eq!(
            query_axis(&PREFILL_QUERIES, 2052),
            Axis::values([1, 4, 16, 64, 128, 256, 512, 1024, 2048, 2052])
        );
        // An unchunked step keeps the measured points; a longer one is
        // answered past the grid.
        assert_eq!(
            query_axis(&PREFILL_QUERIES, 1_048_576),
            Axis::values(PREFILL_QUERIES)
        );
    }

    #[test]
    fn closed_forms_match_the_row_by_row_sums() {
        for (queries, context) in [
            (1, 1),
            (4, 10),
            (3000, 3000),
            (100, 2100),
            (1024, 8192),
            (7, 9000),
        ] {
            let rows = || context - queries + 1..=context;
            let capped_sum: u32 = rows().map(|x| x.min(2048)).sum();
            assert_eq!(capped(queries, context, 2048), f64::from(capped_sum));
            for ratio in [1, 4, 128] {
                let uncapped: u64 = rows().map(|x| u64::from(x / ratio)).sum();
                assert_eq!(compressed(queries, context, ratio, None), uncapped as f64);
                let top: u64 = rows().map(|x| u64::from((x / ratio).min(512))).sum();
                assert_eq!(compressed(queries, context, ratio, Some(512)), top as f64);
            }
        }
    }
}
