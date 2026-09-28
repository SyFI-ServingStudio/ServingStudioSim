//! Replay samples: measured kernel rows held in memory, so a
//! [`PerfApiBridge::replay`](super::PerfApiBridge::replay) bridge answers
//! `get_times` without Python or `profile.db` (the wasm32 build's only bridge).
//!
//! The wire format ([`SAMPLE_FORMAT`] 1) is a JSON array of [`SampleRow`]s; the
//! public API's predict bundle serves it. A row is one profile.db row as a
//! strict perf_api returns it: `kind` is the `profile_kind` (the table), args
//! are the table's arg columns without `backend`, and outlier rows are left out
//! because perf_api never returns them. The lookup key is the canonical JSON of
//! `[kind, gpu_name, backend, args]` with integral numbers written as integers,
//! so args produced by Rust `enumerate`, by Python and by the public API meet on
//! one spelling.

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};

use crate::timing::bridge::{ArgsPayload, KernelKind, KernelMetrics, PerfApiError};

/// Version of the sample wire format; the predict bundle and the wasm module
/// must agree on it.
pub const SAMPLE_FORMAT: u32 = 1;

/// One measured row of a replay samples array.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampleRow {
    /// The `profile_kind`: the profile.db table the kernel reads.
    pub kind: String,
    pub gpu_name: String,
    pub backend: String,
    /// profile.db args columns, without `backend`.
    pub args: BTreeMap<String, Value>,
    pub metrics: SampleMetrics,
}

/// A row's metrics as profile.db stores them: only `time_ms` is required, and
/// a missing `energy_j` reads as zero, as it does through the Python bridge.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampleMetrics {
    pub time_ms: f64,
    #[serde(default)]
    pub tflops: Option<f64>,
    #[serde(default)]
    pub memory_bandwidth_gbps: Option<f64>,
    #[serde(default)]
    pub algbw_gbps: Option<f64>,
    #[serde(default)]
    pub busbw_gbps: Option<f64>,
    #[serde(default)]
    pub energy_j: Option<f64>,
}

impl SampleRow {
    /// The row a perf_api `get_times` answered `payload` with.
    pub fn answer(
        kind: KernelKind,
        backend: &str,
        gpu_name: &str,
        payload: &ArgsPayload,
        metrics: &KernelMetrics,
    ) -> Self {
        let mut args = payload.fields().clone();
        args.remove("backend");
        SampleRow {
            kind: kind.to_string(),
            gpu_name: gpu_name.to_string(),
            backend: backend.to_string(),
            args,
            metrics: SampleMetrics {
                time_ms: metrics.time_ms,
                tflops: metrics.tflops,
                memory_bandwidth_gbps: metrics.memory_bandwidth_gbps,
                algbw_gbps: metrics.algbw_gbps,
                busbw_gbps: metrics.busbw_gbps,
                energy_j: Some(metrics.energy_j),
            },
        }
    }

    /// The replay lookup key: two rows with one key answer the same spec.
    pub fn key(&self) -> String {
        key(&self.kind, &self.gpu_name, &self.backend, &self.args)
    }
}

/// Write `rows` as a samples array to `path`.
pub fn write_samples(path: &std::path::Path, rows: &[SampleRow]) -> anyhow::Result<()> {
    use anyhow::Context;
    let text = serde_json::to_string(rows).expect("sample rows serialize");
    std::fs::write(path, text).with_context(|| format!("writing samples {}", path.display()))
}

impl From<SampleMetrics> for KernelMetrics {
    fn from(m: SampleMetrics) -> Self {
        KernelMetrics {
            time_ms: m.time_ms,
            tflops: m.tflops,
            memory_bandwidth_gbps: m.memory_bandwidth_gbps,
            algbw_gbps: m.algbw_gbps,
            busbw_gbps: m.busbw_gbps,
            energy_j: m.energy_j.unwrap_or(0.0),
        }
    }
}

/// Measured rows keyed for `get_times`.
#[derive(Clone, Debug, Default)]
pub struct ReplaySamples {
    rows: HashMap<String, KernelMetrics>,
}

impl ReplaySamples {
    /// Parse a samples array. A later duplicate replaces an earlier one.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        let rows: Vec<SampleRow> = serde_json::from_str(text)?;
        Ok(Self::from_rows(rows))
    }

    pub fn from_rows(rows: impl IntoIterator<Item = SampleRow>) -> Self {
        let rows = rows
            .into_iter()
            .map(|row| (row.key(), row.metrics.into()))
            .collect();
        Self { rows }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    fn lookup(
        &self,
        kind: KernelKind,
        backend: &str,
        gpu_name: &str,
        payload: &ArgsPayload,
    ) -> Option<&KernelMetrics> {
        let mut args = payload.fields().clone();
        args.remove("backend");
        self.rows.get(&key(kind, gpu_name, backend, &args))
    }

    /// `get_times` over the held rows; the first spec without one is a
    /// [`PerfApiError::MissingEntry`], as a strict (no-JIT) perf_api reports it.
    pub fn get_times(
        &self,
        payloads: &[ArgsPayload],
        kind: KernelKind,
        backend: &str,
        gpu_name: &str,
    ) -> Result<Vec<KernelMetrics>, PerfApiError> {
        payloads
            .iter()
            .map(|payload| {
                self.lookup(kind, backend, gpu_name, payload)
                    .cloned()
                    .ok_or_else(|| PerfApiError::MissingEntry {
                        kind,
                        backend: backend.to_string(),
                        spec: payload.clone(),
                    })
            })
            .collect()
    }

    pub fn count_missing(
        &self,
        payloads: &[ArgsPayload],
        kind: KernelKind,
        backend: &str,
        gpu_name: &str,
    ) -> usize {
        payloads
            .iter()
            .filter(|payload| self.lookup(kind, backend, gpu_name, payload).is_none())
            .count()
    }
}

fn key(kind: &str, gpu_name: &str, backend: &str, args: &BTreeMap<String, Value>) -> String {
    let args: BTreeMap<&str, Value> = args
        .iter()
        .map(|(k, v)| (k.as_str(), canonical(v)))
        .collect();
    serde_json::to_string(&(kind, gpu_name, backend, args)).expect("replay key serializes")
}

/// Integral floats as integers (`2048.0` -> `2048`), recursively.
fn canonical(value: &Value) -> Value {
    match value {
        Value::Number(n) if n.is_f64() => {
            let f = n.as_f64().expect("f64 number");
            if f.fract() == 0.0 && f.abs() < 9.0e15 {
                Value::Number(Number::from(f as i64))
            } else {
                value.clone()
            }
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        Value::Object(fields) => Value::Object(
            fields
                .iter()
                .map(|(k, v)| (k.clone(), canonical(v)))
                .collect(),
        ),
        _ => value.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(time_ms: f64) -> KernelMetrics {
        KernelMetrics {
            time_ms,
            tflops: Some(1.5),
            memory_bandwidth_gbps: None,
            algbw_gbps: None,
            busbw_gbps: None,
            energy_j: 0.0,
        }
    }

    #[test]
    fn replay_matches_integral_floats_and_reports_the_first_miss() {
        let mut args = BTreeMap::new();
        args.insert("m".to_string(), serde_json::json!(2048.0));
        args.insert("dtype".to_string(), serde_json::json!("bf16"));
        let samples = ReplaySamples::from_json(
            &serde_json::json!([{
                "kind": "gemm",
                "gpu_name": "NVIDIA B200",
                "backend": "cublas",
                "args": args,
                "metrics": {"time_ms": 0.25, "tflops": 1.5, "energy_j": null},
            }])
            .to_string(),
        )
        .unwrap();
        let hit = ArgsPayload::new()
            .with("m", 2048)
            .with("dtype", "bf16")
            .with("backend", "cublas");
        let miss = ArgsPayload::new()
            .with("m", 4096)
            .with("dtype", "bf16")
            .with("backend", "cublas");
        let got = samples
            .get_times(std::slice::from_ref(&hit), "gemm", "cublas", "NVIDIA B200")
            .unwrap();
        assert_eq!(got, vec![metrics(0.25)]);
        assert_eq!(
            samples.count_missing(
                &[hit.clone(), miss.clone()],
                "gemm",
                "cublas",
                "NVIDIA B200"
            ),
            1
        );
        assert!(matches!(
            samples.get_times(&[hit, miss], "gemm", "cublas", "NVIDIA B200"),
            Err(PerfApiError::MissingEntry { .. })
        ));
    }
}
