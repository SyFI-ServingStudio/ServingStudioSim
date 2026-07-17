//! Bounded overview of the repository trace files named by run parameters.

use std::path::{Component, Path};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};

use super::artifact::read_run_json;
use super::discovery::{regular_file, DiscoveredRun};
use super::ArtifactNotFound;

const MAX_POINTS: usize = 72;

#[derive(Clone, Debug, Deserialize)]
struct TraceEntry {
    id: u32,
    input_len: u32,
    output_len: u32,
    arrival_time: f64,
}

pub(super) fn read_workload(run: &DiscoveredRun, repo_root: &Path) -> Result<Value> {
    let params = read_run_json(&run.path, "raw/params.json")?;
    let source_paths = trace_file_paths(&params)?;
    if source_paths.is_empty() {
        return Err(ArtifactNotFound.into());
    }
    for source_path in &source_paths {
        validate_trace_path(source_path)?;
    }

    let trace_root = repo_root.join("trace");
    let canonical_root = trace_root
        .canonicalize()
        .context("canonicalize trace root")?;
    let mut entries = Vec::new();
    for source_path in &source_paths {
        let trace_path = resolve_trace_path(repo_root, &canonical_root, source_path)?;
        read_trace_file(&trace_path, &mut entries)?;
    }
    validate_trace(&entries)?;

    let request_rate = params
        .pointer("/workload/request_rate")
        .and_then(Value::as_f64)
        .filter(|rate| rate.is_finite() && *rate > 0.0)
        .context("raw/params.json workload.request_rate must be finite and positive")?;
    let closed_loop = params.pointer("/workload/max_concurrency").is_some();
    let arrival_scale = if closed_loop { 1.0 } else { request_rate };
    let request_count = entries.len() as f64;
    let average_input_tokens = entries
        .iter()
        .map(|entry| u64::from(entry.input_len))
        .sum::<u64>() as f64
        / request_count;
    let average_output_tokens = entries
        .iter()
        .map(|entry| u64::from(entry.output_len))
        .sum::<u64>() as f64
        / request_count;
    let (token_lengths, input_density, output_density) = length_distribution(&entries);
    let (arrival_seconds, arrivals, arrival_trend, peak_to_mean) =
        arrival_distribution(&entries, arrival_scale);

    Ok(json!({
        "schema_version": 1,
        "scope": "configured_trace",
        "source_paths": source_paths,
        "request_count": entries.len(),
        "average_input_tokens": average_input_tokens,
        "average_output_tokens": average_output_tokens,
        "arrival_basis": if closed_loop { "source_trace" } else { "effective_open_loop" },
        "request_rate": request_rate,
        "token_lengths": token_lengths,
        "input_density": input_density,
        "output_density": output_density,
        "arrival_seconds": arrival_seconds,
        "arrivals": arrivals,
        "arrival_trend": arrival_trend,
        "peak_to_mean": peak_to_mean,
    }))
}

pub(super) fn trace_file_paths(params: &Value) -> Result<Vec<String>> {
    let Some(paths) = params
        .pointer("/workload/trace_files")
        .and_then(Value::as_array)
    else {
        return Ok(Vec::new());
    };
    paths
        .iter()
        .map(|path| {
            path.as_str()
                .filter(|path| !path.is_empty())
                .map(str::to_owned)
                .context("workload.trace_files entries must be non-empty strings")
        })
        .collect()
}

fn resolve_trace_path(
    repo_root: &Path,
    canonical_root: &Path,
    source_path: &str,
) -> Result<std::path::PathBuf> {
    validate_trace_path(source_path)?;
    let relative = Path::new(source_path);
    let trace_path = repo_root.join(relative);
    if !regular_file(&trace_path) {
        return Err(ArtifactNotFound.into());
    }
    let canonical_trace = trace_path
        .canonicalize()
        .with_context(|| format!("canonicalize trace file {}", trace_path.display()))?;
    if !canonical_trace.starts_with(canonical_root) {
        bail!("trace file resolves outside trace");
    }
    Ok(canonical_trace)
}

fn validate_trace_path(source_path: &str) -> Result<()> {
    let relative = Path::new(source_path);
    if relative.is_absolute()
        || !relative.starts_with("trace")
        || relative
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("trace file must be a relative path below trace");
    }
    Ok(())
}

fn read_trace_file(path: &Path, entries: &mut Vec<TraceEntry>) -> Result<()> {
    let mut reader = csv::Reader::from_path(path)
        .with_context(|| format!("open trace file {}", path.display()))?;
    let headers = reader
        .headers()
        .with_context(|| format!("read trace header {}", path.display()))?;
    if headers.iter().any(|header| header == "round_idx") {
        bail!("multi-round traces are not supported");
    }
    for (row, result) in reader.deserialize().enumerate() {
        let entry: TraceEntry =
            result.with_context(|| format!("parse {} row {row}", path.display()))?;
        if entry.input_len == 0 || entry.output_len == 0 {
            bail!("{} row {row} has a zero token length", path.display());
        }
        if !entry.arrival_time.is_finite() || entry.arrival_time < 0.0 {
            bail!("{} row {row} has an invalid arrival_time", path.display());
        }
        entries.push(entry);
    }
    Ok(())
}

fn validate_trace(entries: &[TraceEntry]) -> Result<()> {
    if entries.is_empty() {
        bail!("trace files contained no rows");
    }
    for (index, entry) in entries.iter().enumerate() {
        if entry.id != index as u32 {
            bail!("trace row {index} has non-sequential id {}", entry.id);
        }
        if index > 0 && entry.arrival_time < entries[index - 1].arrival_time {
            bail!("trace row {index} has decreasing arrival_time");
        }
    }
    Ok(())
}

fn length_distribution(entries: &[TraceEntry]) -> (Vec<u32>, Vec<f64>, Vec<f64>) {
    let min_length = entries
        .iter()
        .map(|entry| entry.input_len.min(entry.output_len))
        .min()
        .unwrap_or(1);
    let max_length = entries
        .iter()
        .map(|entry| entry.input_len.max(entry.output_len))
        .max()
        .unwrap_or(min_length);
    if min_length == max_length {
        return (vec![min_length], vec![1.0], vec![1.0]);
    }

    let log_min = f64::from(min_length).ln();
    let log_max = f64::from(max_length).ln();
    let width = (log_max - log_min) / MAX_POINTS as f64;
    let mut input_counts = vec![0u64; MAX_POINTS];
    let mut output_counts = vec![0u64; MAX_POINTS];
    for entry in entries {
        input_counts[log_bin(entry.input_len, log_min, width)] += 1;
        output_counts[log_bin(entry.output_len, log_min, width)] += 1;
    }
    let token_lengths = (0..MAX_POINTS)
        .map(|index| (log_min + (index as f64 + 0.5) * width).exp().round() as u32)
        .collect();
    (
        token_lengths,
        normalized_counts(&input_counts),
        normalized_counts(&output_counts),
    )
}

fn log_bin(value: u32, log_min: f64, width: f64) -> usize {
    (((f64::from(value).ln() - log_min) / width).floor() as usize).min(MAX_POINTS - 1)
}

fn normalized_counts(counts: &[u64]) -> Vec<f64> {
    let peak = counts.iter().copied().max().unwrap_or(1).max(1) as f64;
    counts
        .iter()
        .map(|count| ((10000.0 * *count as f64 / peak).round()) / 10000.0)
        .collect()
}

fn arrival_distribution(
    entries: &[TraceEntry],
    arrival_scale: f64,
) -> (Vec<f64>, Vec<u64>, Vec<f64>, f64) {
    let start_ms = entries
        .first()
        .map(|entry| entry.arrival_time)
        .unwrap_or(0.0);
    let end_ms = entries
        .last()
        .map(|entry| entry.arrival_time)
        .unwrap_or(start_ms);
    let bucket_count = MAX_POINTS.min(entries.len()).max(1);
    let span_ms = end_ms - start_ms;
    let mut arrivals = vec![0u64; bucket_count];
    for entry in entries {
        let index = if span_ms > 0.0 {
            ((((entry.arrival_time - start_ms) / span_ms) * bucket_count as f64).floor() as usize)
                .min(bucket_count - 1)
        } else {
            0
        };
        arrivals[index] += 1;
    }
    let bucket_width = if span_ms > 0.0 {
        span_ms / bucket_count as f64
    } else {
        0.0
    };
    let arrival_seconds = (0..bucket_count)
        .map(|index| (start_ms + (index as f64 + 0.5) * bucket_width) / arrival_scale / 1000.0)
        .collect();
    let arrival_trend = moving_average(&arrivals, 2);
    let mean = entries.len() as f64 / bucket_count as f64;
    let peak = arrivals.iter().copied().max().unwrap_or(0) as f64;
    let peak_to_mean = ((peak / mean) * 10.0).round() / 10.0;
    (arrival_seconds, arrivals, arrival_trend, peak_to_mean)
}

fn moving_average(values: &[u64], radius: usize) -> Vec<f64> {
    (0..values.len())
        .map(|index| {
            let start = index.saturating_sub(radius);
            let end = (index + radius + 1).min(values.len());
            let sum: u64 = values[start..end].iter().sum();
            ((sum as f64 / (end - start) as f64) * 100.0).round() / 100.0
        })
        .collect()
}
