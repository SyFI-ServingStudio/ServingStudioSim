//! In-flight requests over simulated wall-clock time.
//!
//! SQL collapses the terminal-per-request table into timestamped deltas. Rust
//! then performs one ordered event sweep: exact peak is retained while the UI
//! series is bounded to equal-width, time-weighted means.

use std::path::Path;

use anyhow::{bail, Context, Result};
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use crate::io::{resolve_artifact_path, SCHEMA_VERSION};
use crate::session::{col, collect, register_if_exists, require_columns, value_f64};

const SLO_COLS: &[&str] = &["arrival_time_ms", "finish_decode_time_ms", "logging_time"];
pub(crate) const MAX_POINTS: usize = 512;

#[derive(Clone, Copy, Debug, PartialEq)]
struct Event {
    time_ms: f64,
    delta: i64,
}

#[derive(Debug, PartialEq)]
struct ConcurrencySeries {
    t_ms: Vec<f64>,
    active: Vec<f64>,
    peak: u64,
    request_count: u64,
    mean: f64,
    span_ms: f64,
}

pub async fn run_concurrency(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    let slo_path = resolve_artifact_path(log_dir, "request_slo.parquet");
    if !register_if_exists(ctx, "slo", slo_path).await? {
        let reason = "request_slo.parquet not found";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }
    require_columns(ctx, "slo", SLO_COLS).await?;

    let events = collect_events(ctx).await?;
    let Some(series) = build_series(&events, MAX_POINTS)? else {
        let reason = "request_slo has no positive wall-clock concurrency span";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    };
    let definitions = definitions();
    let meta = json!({
        "log_dir": log_dir.display().to_string(),
        "request_count": series.request_count,
        "span_ms": series.span_ms,
        "bins": series.t_ms.len(),
        "max_points": MAX_POINTS,
        "aggregation": "equal-width time-weighted mean",
    });
    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "available": true,
        "totals": {
            "requests": series.request_count,
            "span_ms": series.span_ms,
            "peak_active": series.peak,
            "mean_active": series.mean,
        },
        "definitions": definitions,
    });
    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "t_ms": series.t_ms,
        "active": series.active,
        "peak": series.peak,
        "definitions": definitions,
    });
    Ok((report, payload))
}

/// Collapse simultaneous starts/terminals in SQL. This is at most two scalar
/// rows per distinct request timestamp, never one Arrow row per event kind plus
/// the request's unused SLO columns.
async fn collect_events(ctx: &SessionContext) -> Result<Vec<Event>> {
    let batches = collect(
        ctx,
        "SELECT time_ms, SUM(delta) AS delta FROM (\
             SELECT arrival_time_ms AS time_ms, CAST(1 AS BIGINT) AS delta FROM slo \
             UNION ALL \
             SELECT COALESCE(CAST(finish_decode_time_ms AS DOUBLE), logging_time) AS time_ms, \
                    CAST(-1 AS BIGINT) AS delta FROM slo\
         ) events WHERE time_ms IS NOT NULL GROUP BY time_ms ORDER BY time_ms",
    )
    .await?;
    let mut events = Vec::new();
    for batch in &batches {
        let times = col(batch, "time_ms")?;
        let deltas = col(batch, "delta")?;
        for row in 0..batch.num_rows() {
            let time_ms = value_f64(times, row)?;
            let delta_value = value_f64(deltas, row)?;
            if !time_ms.is_finite() || time_ms < 0.0 {
                bail!("request_slo produced invalid concurrency event time {time_ms}");
            }
            if !delta_value.is_finite() || delta_value.fract() != 0.0 {
                bail!("request_slo produced non-integral concurrency delta {delta_value}");
            }
            events.push(Event {
                time_ms,
                delta: delta_value as i64,
            });
        }
    }
    Ok(events)
}

fn build_series(events: &[Event], max_points: usize) -> Result<Option<ConcurrencySeries>> {
    let Some(last) = events.last() else {
        return Ok(None);
    };
    if !last.time_ms.is_finite() || last.time_ms <= 0.0 || max_points == 0 {
        return Ok(None);
    }
    if events
        .windows(2)
        .any(|pair| pair[1].time_ms < pair[0].time_ms)
    {
        bail!("concurrency events are not ordered by time");
    }

    let n_bins = max_points.min(events.len().max(1));
    let span_ms = last.time_ms;
    let bin_width = span_ms / n_bins as f64;
    let mut active_area = vec![0.0; n_bins];
    let mut active = 0i64;
    let mut peak = 0u64;
    let mut request_count = 0u64;
    let mut previous_ms = 0.0;

    for event in events {
        if event.time_ms > previous_ms {
            add_interval_area(
                &mut active_area,
                bin_width,
                previous_ms,
                event.time_ms,
                active as f64,
            );
        }
        if event.delta > 0 {
            request_count = request_count
                .checked_add(event.delta as u64)
                .context("concurrency request count overflow")?;
        }
        active = active
            .checked_add(event.delta)
            .context("concurrency active count overflow")?;
        if active < 0 {
            bail!(
                "concurrency event sweep became negative at {} ms",
                event.time_ms
            );
        }
        peak = peak.max(active as u64);
        previous_ms = event.time_ms;
    }
    if active != 0 {
        bail!("concurrency event sweep ended with {active} active requests");
    }

    let t_ms = (1..=n_bins)
        .map(|index| {
            // Preserve the source span exactly: the UI uses the final point as
            // its wall-clock range and validates it against meta.span_ms.
            if index == n_bins {
                span_ms
            } else {
                index as f64 * bin_width
            }
        })
        .collect::<Vec<_>>();
    let active = active_area
        .iter()
        .map(|area| area / bin_width)
        .collect::<Vec<_>>();
    let mean = active_area.iter().sum::<f64>() / span_ms;
    Ok(Some(ConcurrencySeries {
        t_ms,
        active,
        peak,
        request_count,
        mean,
        span_ms,
    }))
}

fn add_interval_area(bins: &mut [f64], bin_width: f64, start_ms: f64, end_ms: f64, active: f64) {
    let first_bin = ((start_ms / bin_width).floor() as usize).min(bins.len() - 1);
    let last_bin = (((end_ms / bin_width).ceil() as usize).saturating_sub(1)).min(bins.len() - 1);
    for (bin, area) in bins
        .iter_mut()
        .enumerate()
        .take(last_bin + 1)
        .skip(first_bin)
    {
        let overlap_start = start_ms.max(bin as f64 * bin_width);
        let overlap_end = end_ms.min((bin + 1) as f64 * bin_width);
        if overlap_end > overlap_start {
            *area += active * (overlap_end - overlap_start);
        }
    }
}

fn definitions() -> Value {
    json!({
        "scope": "all requests admitted by the simulator and written to request_slo",
        "active": "time-weighted mean number of requests in flight during the equal-width bin",
        "t_ms": "right edge of the equal-width bin in simulated wall-clock milliseconds",
        "peak": "exact maximum in-flight request count from the ordered event sweep before binning",
        "binning": "at most 512 equal-width bins; event timestamps are grouped in SQL before the sweep",
    })
}

fn unavailable(log_dir: &Path, reason: &str) -> Value {
    json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {"log_dir": log_dir.display().to_string()},
        "available": false,
        "reason": reason,
        "definitions": definitions(),
    })
}

fn unavailable_payload(log_dir: &Path, reason: &str) -> Value {
    json!({
        "schema_version": SCHEMA_VERSION,
        "meta": {"log_dir": log_dir.display().to_string(), "available": false, "reason": reason},
        "t_ms": [],
        "active": [],
        "peak": 0,
        "definitions": definitions(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_sweep_keeps_exact_peak_and_time_weighted_bins() {
        let series = build_series(
            &[
                Event {
                    time_ms: 0.0,
                    delta: 2,
                },
                Event {
                    time_ms: 5.0,
                    delta: -1,
                },
                Event {
                    time_ms: 10.0,
                    delta: -1,
                },
            ],
            2,
        )
        .expect("build series")
        .expect("positive span");

        assert_eq!(series.t_ms, vec![5.0, 10.0]);
        assert_eq!(series.active, vec![2.0, 1.0]);
        assert_eq!(series.peak, 2);
        assert_eq!(series.request_count, 2);
        assert_eq!(series.mean, 1.5);
    }

    #[test]
    fn bins_include_idle_time_before_first_arrival() {
        let series = build_series(
            &[
                Event {
                    time_ms: 2.0,
                    delta: 1,
                },
                Event {
                    time_ms: 6.0,
                    delta: -1,
                },
            ],
            3,
        )
        .expect("build series")
        .expect("positive span");

        assert_eq!(series.t_ms, vec![3.0, 6.0]);
        assert_eq!(series.active, vec![1.0 / 3.0, 1.0]);
        assert_eq!(series.peak, 1);
        assert_eq!(series.mean, 2.0 / 3.0);
    }

    #[test]
    fn rejects_unbalanced_terminal_events() {
        let error = build_series(
            &[Event {
                time_ms: 1.0,
                delta: 1,
            }],
            1,
        )
        .expect_err("unbalanced events must fail");

        assert!(error.to_string().contains("ended with 1 active"));
    }
}
