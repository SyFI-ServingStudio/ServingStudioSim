//! Bounded overview of the requests a run released: what each asked for and
//! when it arrived, from the run's own request record
//! (`raw/request_slo.parquet`: `declared_prefix_tokens + fresh_prompt_tokens`
//! in, `target_output_tokens` out, `arrival_time_ms`). The simulator read the
//! trace files and recorded every request it released, in whatever trace
//! format and replay mode the run used; the trace paths here only label them.

use std::fs::File;
use std::path::{Component, Path};

use anyhow::{bail, Context, Result};
use datafusion::arrow::array::{Array, Float64Array, UInt32Array};
use datafusion::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use datafusion::parquet::arrow::ProjectionMask;
use serde_json::{json, Value};

use super::artifact::read_run_json;
use super::discovery::{regular_file, DiscoveredRun};
use super::ArtifactNotFound;

const MAX_POINTS: usize = 72;
const REQUESTS: &str = "raw/request_slo.parquet";

#[derive(Clone, Debug)]
struct TraceEntry {
    input_len: u32,
    output_len: u32,
    arrival_time: f64,
}

pub(super) fn read_workload(run: &DiscoveredRun, repo_root: &Path) -> Result<Value> {
    let params = read_run_json(&run.path, "raw/params.json")?;
    let source_paths = trace_file_paths(&params)?
        .iter()
        .map(|path| trace_label(path, repo_root))
        .collect::<Result<Vec<_>>>()?;
    if source_paths.is_empty() {
        return Err(ArtifactNotFound.into());
    }
    let mut entries = read_requests(&run.path.join(REQUESTS))?;
    if entries.is_empty() {
        bail!("{REQUESTS} records no request");
    }
    entries.sort_by(|a, b| a.arrival_time.total_cmp(&b.arrival_time));

    let request_rate = params
        .pointer("/workload/request_rate")
        .and_then(Value::as_f64)
        .filter(|rate| rate.is_finite() && *rate > 0.0)
        .context("raw/params.json workload.request_rate must be finite and positive")?;
    // Arrival times are the run's own releases: a trace-timed run's are
    // already rescaled by `request_rate`, a saturated run's are when it
    // released each request.
    let saturated = params
        .pointer("/workload/arrival_mode")
        .and_then(Value::as_str)
        == Some("saturated");
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
    let (arrival_seconds, arrivals, arrival_trend, peak_to_mean) = arrival_distribution(&entries);

    Ok(json!({
        "schema_version": 1,
        "scope": "configured_trace",
        "source_paths": source_paths,
        "request_count": entries.len(),
        "average_input_tokens": average_input_tokens,
        "average_output_tokens": average_output_tokens,
        "arrival_basis": if saturated { "effective_open_loop" } else { "effective_trace_timed" },
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

/// A trace file as the overview names it: repository-relative. A run the
/// launcher wrote outside its repository's `logs/` (a service's run
/// directory) names its trace by an absolute path under the configured root;
/// it is named from that root's `logs` or `trace` directory down.
fn trace_label(source_path: &str, repo_root: &Path) -> Result<String> {
    let path = Path::new(source_path);
    if !path.is_absolute() {
        validate_trace_path(source_path)?;
        return Ok(source_path.to_owned());
    }
    let components = path.components().collect::<Vec<_>>();
    let from = components.iter().rposition(|component| {
        matches!(component, Component::Normal(name) if *name == "logs" || *name == "trace")
    });
    let label = from.map(|from| {
        components[from..]
            .iter()
            .map(|component| component.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/")
    });
    let under_root = repo_root
        .canonicalize()
        .ok()
        .zip(path.canonicalize().ok())
        .is_some_and(|(root, path)| path.starts_with(root));
    match label {
        Some(label) if under_root && regular_file(path) => {
            validate_trace_path(&label)?;
            Ok(label)
        }
        _ => {
            bail!("trace file must be a relative path inside a trace directory or a CSV under logs")
        }
    }
}

/// Each released request's prompt, target output and arrival, as the run
/// recorded it.
fn read_requests(path: &Path) -> Result<Vec<TraceEntry>> {
    if !regular_file(path) {
        return Err(ArtifactNotFound.into());
    }
    let open = || File::open(path).with_context(|| format!("open {}", path.display()));
    let builder = ParquetRecordBatchReaderBuilder::try_new(open()?)
        .with_context(|| format!("read parquet metadata of {}", path.display()))?;
    let schema = builder.schema().clone();
    let names = [
        "arrival_time_ms",
        "declared_prefix_tokens",
        "fresh_prompt_tokens",
        "target_output_tokens",
    ];
    let roots = names
        .iter()
        .map(|name| {
            schema
                .index_of(name)
                .with_context(|| format!("{} has no `{name}` column", path.display()))
        })
        .collect::<Result<Vec<_>>>()?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), roots);
    let mut entries = Vec::new();
    for batch in builder.with_projection(mask).build()? {
        let batch = batch?;
        let column = |name: &str| {
            batch
                .column_by_name(name)
                .with_context(|| format!("{} lost `{name}`", path.display()))
        };
        let arrival = column("arrival_time_ms")?
            .as_any()
            .downcast_ref::<Float64Array>()
            .context("arrival_time_ms is not Float64")?
            .clone();
        let u32s = |name: &str| -> Result<UInt32Array> {
            Ok(column(name)?
                .as_any()
                .downcast_ref::<UInt32Array>()
                .with_context(|| format!("{name} is not UInt32"))?
                .clone())
        };
        let (prefix, fresh, output) = (
            u32s("declared_prefix_tokens")?,
            u32s("fresh_prompt_tokens")?,
            u32s("target_output_tokens")?,
        );
        for row in 0..batch.num_rows() {
            entries.push(TraceEntry {
                input_len: prefix.value(row) + fresh.value(row),
                output_len: output.value(row),
                arrival_time: arrival.value(row),
            });
        }
    }
    Ok(entries)
}

fn validate_trace_path(source_path: &str) -> Result<()> {
    let relative = Path::new(source_path);
    let components = relative.components().collect::<Vec<_>>();
    let is_normal_relative = !relative.is_absolute()
        && !components.is_empty()
        && components
            .iter()
            .all(|component| matches!(component, Component::Normal(_)));
    let passes_through_trace_directory = components
        .iter()
        .take(components.len().saturating_sub(1))
        .any(|component| matches!(component, Component::Normal(name) if *name == "trace"));
    // Launchers also store generated CSVs directly in an experiment directory.
    // Canonical containment is checked separately before the file is opened.
    let is_experiment_csv = components
        .iter()
        .take(components.len().saturating_sub(1))
        .any(|component| matches!(component, Component::Normal(name) if *name == "logs"))
        && relative
            .extension()
            .is_some_and(|extension| extension == "csv");
    if !is_normal_relative || !(passes_through_trace_directory || is_experiment_csv) {
        bail!("trace file must be a relative path inside a trace directory or a CSV under logs");
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

fn arrival_distribution(entries: &[TraceEntry]) -> (Vec<f64>, Vec<u64>, Vec<f64>, f64) {
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
        .map(|index| (start_ms + (index as f64 + 0.5) * bucket_width) / 1000.0)
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
