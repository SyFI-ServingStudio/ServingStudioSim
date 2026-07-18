//! Request-stage populations over time, reconstructed from each request's
//! transition timeline in `request_slo`.
//!
//! The cluster view preserves every open `category:*` term from `stage_vocab`,
//! including `done`, so the stacked categories conserve every request that has
//! entered stage tracking. Pool and worker views intentionally focus on
//! `pending:*`: they expose backpressure as
//! per-worker queue depth, pool total, and pool worker-average. Equal-width,
//! time-weighted bins bound the payload while exact peaks remain in the report.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use arrow_array::{Array, ArrayRef, ListArray, UInt16Array};
use datafusion::prelude::SessionContext;
use serde_json::{json, Value};

use super::series::add_interval_area;
use crate::io::{read_stage_vocab, read_worker_pools, resolve_artifact_path, SCHEMA_VERSION};
use crate::session::{
    col, collect, column_f64, register_if_exists, require_columns, value_f32_list,
};

const SLO_COLS: &[&str] = &[
    "logging_time",
    "stage_times_ms",
    "stage_codes",
    "stage_pool_ids",
    "stage_worker_ids",
];
const FINE_BINS: usize = 200;
const PENDING_CATEGORY: &str = "pending";

type WorkerKey = (u64, u64); // (numeric pool id, worker id)

#[derive(Clone, Debug, PartialEq)]
struct Transition {
    time_ms: f64,
    code: usize,
    pool: u64,
    worker: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct TimedDelta {
    time_ms: f64,
    delta: i64,
}

#[derive(Debug, Default, PartialEq)]
struct EventSet {
    span_ms: f64,
    requests_with_history: usize,
    transitions: usize,
    cluster: BTreeMap<String, Vec<TimedDelta>>,
    pending_by_worker: BTreeMap<WorkerKey, Vec<TimedDelta>>,
}

#[derive(Debug, PartialEq)]
struct BinnedSeries {
    values: Vec<f64>,
    peak: u64,
    mean: f64,
}

pub async fn run_request_state(ctx: &SessionContext, log_dir: &Path) -> Result<(Value, Value)> {
    let slo_path = resolve_artifact_path(log_dir, "request_slo.parquet");
    if !register_if_exists(ctx, "slo", slo_path).await? {
        let reason = "request_slo.parquet not found";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }
    require_columns(ctx, "slo", SLO_COLS).await?;

    let Some(vocab) = read_stage_vocab(log_dir) else {
        let reason = "run_meta.json has no stage_vocab (request stage logging predates schema v5)";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    };
    let categories = ordered_categories(&vocab.names)?;
    let timelines = collect_timelines(ctx).await?;
    let events = build_event_set(&timelines, &vocab.names)?;
    if events.transitions == 0 {
        let reason = "request_slo stage timelines are empty (io.log_stage_transitions was off)";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }
    if events.span_ms <= 0.0 {
        let reason = "request stage timeline has no positive simulated-time span";
        return Ok((
            unavailable(log_dir, reason),
            unavailable_payload(log_dir, reason),
        ));
    }

    let n_bins = FINE_BINS;
    let bin_width_ms = events.span_ms / n_bins as f64;
    let t_start_ms = (0..n_bins)
        .map(|bin| bin as f64 * bin_width_ms)
        .collect::<Vec<_>>();
    let t_end_ms = (1..=n_bins)
        .map(|bin| {
            if bin == n_bins {
                events.span_ms
            } else {
                bin as f64 * bin_width_ms
            }
        })
        .collect::<Vec<_>>();

    let mut cluster_series = Vec::new();
    let mut cluster_totals = Vec::new();
    for category in &categories {
        let series = bin_deltas(
            events
                .cluster
                .get(category)
                .map(Vec::as_slice)
                .unwrap_or(&[]),
            events.span_ms,
            n_bins,
        )?;
        cluster_series.push(json!({
            "category": category,
            "values": series.values,
        }));
        cluster_totals.push(json!({
            "category": category,
            "peak": series.peak,
            "mean": series.mean,
        }));
    }

    let mut pool_roster: BTreeMap<u64, (String, BTreeSet<u64>)> = BTreeMap::new();
    for (pool_tag, worker_id, pool) in read_worker_pools(log_dir).unwrap_or_default() {
        let entry = pool_roster
            .entry(pool)
            .or_insert_with(|| (pool_tag.clone(), BTreeSet::new()));
        if entry.0 != pool_tag {
            bail!(
                "run_meta pool {pool} maps to both {:?} and {pool_tag:?}",
                entry.0
            );
        }
        entry.1.insert(worker_id);
    }
    for &(pool, worker) in events.pending_by_worker.keys() {
        pool_roster
            .entry(pool)
            .or_insert_with(|| (format!("pool_{pool}"), BTreeSet::new()))
            .1
            .insert(worker);
    }

    let mut pools = Vec::new();
    let mut pool_totals = Vec::new();
    let mut worker_totals = Vec::new();
    for (pool, (pool_tag, worker_ids)) in pool_roster {
        let mut worker_payloads = Vec::new();
        let mut pool_events = Vec::new();
        for worker_id in &worker_ids {
            let worker_events = events
                .pending_by_worker
                .get(&(pool, *worker_id))
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            pool_events.extend_from_slice(worker_events);
            let series = bin_deltas(worker_events, events.span_ms, n_bins)?;
            worker_payloads.push(json!({
                "worker_id": worker_id,
                "pending": series.values,
            }));
            worker_totals.push(json!({
                "pool": pool,
                "pool_tag": pool_tag,
                "worker_id": worker_id,
                "peak_pending": series.peak,
                "mean_pending": series.mean,
            }));
        }
        let total = bin_deltas(&pool_events, events.span_ms, n_bins)?;
        let n_workers = worker_ids.len().max(1);
        let average = total
            .values
            .iter()
            .map(|value| value / n_workers as f64)
            .collect::<Vec<_>>();
        pools.push(json!({
            "pool": pool,
            "pool_tag": pool_tag,
            "n_workers": n_workers,
            "total_pending": total.values,
            "average_pending": average,
            "workers": worker_payloads,
        }));
        pool_totals.push(json!({
            "pool": pool,
            "pool_tag": pool_tag,
            "n_workers": n_workers,
            "peak_total_pending": total.peak,
            "mean_total_pending": total.mean,
            "mean_pending_per_worker": total.mean / n_workers as f64,
        }));
    }

    let definitions = definitions();
    let meta = json!({
        "log_dir": log_dir.display().to_string(),
        "deployment": vocab.deployment,
        "requests_with_stage_history": events.requests_with_history,
        "transitions": events.transitions,
        "span_ms": events.span_ms,
        "num_bins": n_bins,
        "bin_width_ms": bin_width_ms,
        "aggregation": "equal-width time-weighted mean",
    });
    let report = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "available": true,
        "totals": {
            "cluster_categories": cluster_totals,
            "pools": pool_totals,
            "workers": worker_totals,
        },
        "definitions": definitions,
    });
    let payload = json!({
        "schema_version": SCHEMA_VERSION,
        "meta": meta,
        "t_start_ms": t_start_ms,
        "t_end_ms": t_end_ms,
        "cluster_series": cluster_series,
        "pools": pools,
        "definitions": definitions,
    });
    Ok((report, payload))
}

fn ordered_categories(names: &[String]) -> Result<Vec<String>> {
    let mut seen = BTreeSet::new();
    let mut categories = Vec::new();
    for name in names {
        let (category, detail) = name
            .split_once(':')
            .ok_or_else(|| anyhow!("stage vocab entry {name:?} is not category:detail"))?;
        if category.is_empty() || detail.is_empty() {
            bail!("stage vocab entry {name:?} has an empty category or detail");
        }
        if seen.insert(category.to_owned()) {
            categories.push(category.to_owned());
        }
    }
    Ok(categories)
}

fn build_event_set(timelines: &[(f64, Vec<Transition>)], names: &[String]) -> Result<EventSet> {
    let categories = names
        .iter()
        .map(|name| {
            name.split_once(':')
                .map(|(category, _)| category)
                .ok_or_else(|| anyhow!("stage vocab entry {name:?} is not category:detail"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut out = EventSet::default();
    for (logging_time_ms, transitions) in timelines {
        if !logging_time_ms.is_finite() || *logging_time_ms < 0.0 {
            bail!("invalid request_slo logging_time {logging_time_ms}");
        }
        out.span_ms = out.span_ms.max(*logging_time_ms);
        if transitions.is_empty() {
            continue;
        }
        out.requests_with_history += 1;
        let mut previous: Option<(&str, u64, u64)> = None;
        let mut previous_time = 0.0;
        for transition in transitions {
            if !transition.time_ms.is_finite() || transition.time_ms < previous_time {
                bail!("request stage times are invalid or non-monotonic");
            }
            let category = *categories.get(transition.code).ok_or_else(|| {
                anyhow!(
                    "request stage code {} is outside vocab length {}",
                    transition.code,
                    categories.len()
                )
            })?;
            if let Some((old_category, old_pool, old_worker)) = previous {
                out.cluster
                    .entry(old_category.to_owned())
                    .or_default()
                    .push(TimedDelta {
                        time_ms: transition.time_ms,
                        delta: -1,
                    });
                if old_category == PENDING_CATEGORY {
                    out.pending_by_worker
                        .entry((old_pool, old_worker))
                        .or_default()
                        .push(TimedDelta {
                            time_ms: transition.time_ms,
                            delta: -1,
                        });
                }
            }
            out.cluster
                .entry(category.to_owned())
                .or_default()
                .push(TimedDelta {
                    time_ms: transition.time_ms,
                    delta: 1,
                });
            if category == PENDING_CATEGORY {
                out.pending_by_worker
                    .entry((transition.pool, transition.worker))
                    .or_default()
                    .push(TimedDelta {
                        time_ms: transition.time_ms,
                        delta: 1,
                    });
            }
            previous = Some((category, transition.pool, transition.worker));
            previous_time = transition.time_ms;
            out.span_ms = out.span_ms.max(transition.time_ms);
            out.transitions += 1;
        }
    }
    Ok(out)
}

fn bin_deltas(events: &[TimedDelta], span_ms: f64, n_bins: usize) -> Result<BinnedSeries> {
    if span_ms <= 0.0 || n_bins == 0 {
        bail!("request-state binning requires a positive span and bin count");
    }
    let mut events = events.to_vec();
    events.sort_by(|left, right| {
        left.time_ms
            .partial_cmp(&right.time_ms)
            .unwrap_or(Ordering::Equal)
    });
    let bin_width_ms = span_ms / n_bins as f64;
    let mut area = vec![0.0; n_bins];
    let mut current = 0i64;
    let mut peak = 0u64;
    let mut previous_time = 0.0;
    let mut index = 0;
    while index < events.len() {
        let time_ms = events[index].time_ms;
        if !time_ms.is_finite() || time_ms < previous_time || time_ms > span_ms {
            bail!("request-state event time {time_ms} is outside [0, {span_ms}]");
        }
        if time_ms > previous_time {
            add_interval_area(
                &mut area,
                bin_width_ms,
                previous_time,
                time_ms,
                current as f64,
            );
        }
        let mut delta = 0i64;
        while index < events.len() && events[index].time_ms == time_ms {
            delta = delta
                .checked_add(events[index].delta)
                .ok_or_else(|| anyhow!("request-state delta overflow"))?;
            index += 1;
        }
        current = current
            .checked_add(delta)
            .ok_or_else(|| anyhow!("request-state population overflow"))?;
        if current < 0 {
            bail!("request-state population became negative at {time_ms} ms");
        }
        peak = peak.max(current as u64);
        previous_time = time_ms;
    }
    if previous_time < span_ms {
        add_interval_area(
            &mut area,
            bin_width_ms,
            previous_time,
            span_ms,
            current as f64,
        );
    }
    let mean = area.iter().sum::<f64>() / span_ms;
    let values = area.into_iter().map(|value| value / bin_width_ms).collect();
    Ok(BinnedSeries { values, peak, mean })
}

async fn collect_timelines(ctx: &SessionContext) -> Result<Vec<(f64, Vec<Transition>)>> {
    let batches = collect(
        ctx,
        "SELECT logging_time, stage_times_ms, stage_codes, stage_pool_ids, stage_worker_ids FROM slo",
    )
    .await?;
    let mut out = Vec::new();
    for batch in &batches {
        let logging_times = column_f64(col(batch, "logging_time")?)?;
        let time_col = col(batch, "stage_times_ms")?;
        let code_col = col(batch, "stage_codes")?;
        let pool_col = col(batch, "stage_pool_ids")?;
        let worker_col = col(batch, "stage_worker_ids")?;
        for row in 0..batch.num_rows() {
            let times = value_f32_list(time_col, row)?;
            let codes = value_u16_list(code_col, row, "stage_codes")?;
            let pools = value_u16_list(pool_col, row, "stage_pool_ids")?;
            let workers = value_u16_list(worker_col, row, "stage_worker_ids")?;
            if !(times.len() == codes.len()
                && codes.len() == pools.len()
                && pools.len() == workers.len())
            {
                bail!("request_slo stage list columns have unequal lengths at row {row}");
            }
            let transitions = times
                .into_iter()
                .zip(codes)
                .zip(pools)
                .zip(workers)
                .map(|(((time_ms, code), pool), worker)| Transition {
                    time_ms,
                    code: code as usize,
                    pool: pool as u64,
                    worker: worker as u64,
                })
                .collect();
            out.push((logging_times[row], transitions));
        }
    }
    Ok(out)
}

fn value_u16_list(array: &ArrayRef, row: usize, label: &str) -> Result<Vec<u16>> {
    let list = array
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("{label} is not a List array"))?;
    if list.is_null(row) {
        return Ok(Vec::new());
    }
    let values = list.value(row);
    let values = values
        .as_any()
        .downcast_ref::<UInt16Array>()
        .ok_or_else(|| anyhow!("{label} items are not UInt16"))?;
    Ok((0..values.len()).map(|index| values.value(index)).collect())
}

fn definitions() -> Value {
    json!({
        "scope": "all request_slo stage transitions when io.log_stage_transitions is enabled",
        "category": "the open category term before ':' in run_meta.stage_vocab.names; every category, including done, is preserved at cluster level",
        "done": "terminal category; completed requests remain resident here so the stacked cluster populations conserve all stage-tracked requests",
        "pending": "all pending:* stages, grouped by the event's numeric pool and worker location",
        "pool_total_pending": "sum of worker pending queue depths in the pool",
        "pool_average_pending": "pool total pending divided by the run_meta worker roster size",
        "bin": "one of 200 equal-width simulated-time bins; plotted values are exact time-weighted means within each bin",
        "peak": "exact event-sweep peak before binning",
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
        "t_start_ms": [],
        "t_end_ms": [],
        "cluster_series": [],
        "pools": [],
        "definitions": definitions(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tr(time_ms: f64, code: usize, pool: u64, worker: u64) -> Transition {
        Transition {
            time_ms,
            code,
            pool,
            worker,
        }
    }

    #[test]
    fn event_reconstruction_tracks_categories_and_worker_pending() {
        let names = vec![
            "pending:prefill".to_owned(),
            "active:prefill".to_owned(),
            "transfer:kv_pull".to_owned(),
            "done:request".to_owned(),
        ];
        let timelines = vec![
            (
                10.0,
                vec![tr(0.0, 0, 0, 0), tr(2.0, 1, 0, 0), tr(8.0, 3, 0, 0)],
            ),
            (10.0, vec![tr(1.0, 0, 0, 1), tr(5.0, 2, 1, 0)]),
        ];

        let events = build_event_set(&timelines, &names).unwrap();
        let pending = bin_deltas(&events.cluster["pending"], 10.0, 10).unwrap();
        let active = bin_deltas(&events.cluster["active"], 10.0, 10).unwrap();
        let transfer = bin_deltas(&events.cluster["transfer"], 10.0, 10).unwrap();
        let done = bin_deltas(&events.cluster["done"], 10.0, 10).unwrap();

        assert_eq!(
            pending.values,
            vec![1.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            active.values,
            vec![0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(
            transfer.values,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );
        assert_eq!(
            done.values,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
        );
        assert_eq!(events.requests_with_history, 2);
        assert_eq!(events.transitions, 5);
        assert_eq!(events.pending_by_worker.len(), 2);
    }

    #[test]
    fn vocabulary_preserves_open_categories_and_done() {
        let categories = ordered_categories(&[
            "pending:prefill".to_owned(),
            "active:prefill".to_owned(),
            "pending:decode".to_owned(),
            "suspended:preempted".to_owned(),
            "done:request".to_owned(),
        ])
        .unwrap();

        assert_eq!(categories, vec!["pending", "active", "suspended", "done"]);
    }

    #[test]
    fn simultaneous_category_move_does_not_create_a_false_dip() {
        let events = vec![
            TimedDelta {
                time_ms: 0.0,
                delta: 1,
            },
            TimedDelta {
                time_ms: 5.0,
                delta: -1,
            },
            TimedDelta {
                time_ms: 5.0,
                delta: 1,
            },
        ];
        let series = bin_deltas(&events, 10.0, 2).unwrap();

        assert_eq!(series.values, vec![1.0, 1.0]);
        assert_eq!(series.peak, 1);
        assert_eq!(series.mean, 1.0);
    }
}
