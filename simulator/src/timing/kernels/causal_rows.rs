//! Closed-form sums of the rows each query row of a causal prefill request
//! reads, for kernels whose logical work counts them. Row `x` of a
//! `(queries, context)` request runs over `context - queries + 1 ..= context`.

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
