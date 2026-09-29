//! Kernel data: the public kernel API's config documents held in memory, so a
//! [`PerfApiBridge::kernel_data`](super::PerfApiBridge::kernel_data) bridge
//! fits every kernel cache without Python or `profile.db` (the wasm32 build's
//! only bridge).
//!
//! A document is what `GET /kernels/{kind}/configs/{config_hash}?gpu=` returns:
//! the config's identity, its grid axes, and one point per grid cell in
//! row-major order, each with its feasibility and its measured row per
//! backend. A kernel finds its document by `(kind, gpu, identity)` -- the
//! identity it computes itself, compared as JSON -- and reads its samples
//! straight off the points; the args a cell was measured at are not consulted,
//! since the cache needs only the grid and the samples.
//!
//! A [`KernelData::registry`] holds no documents up front: it fetches each
//! one from profile.db through perf_api the first time a kernel asks, which is
//! how `kernel-query` builds a kernel from the registry rather than from args.
//!
//! The document is the source of truth. A kernel whose grid disagrees with its
//! document (other axes, another infeasible set), or whose document lacks a
//! measured, non-outlier row at a feasible cell, fails to build: the fix is in
//! the data, not here. Outlier rows count as missing because the Python bridge
//! never returns them.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use serde::Deserialize;
use serde_json::Value;

use crate::timing::bridge::{KernelKind, KernelMetrics};

/// Version of what a kernel-data bridge reads: an array of config documents,
/// or an object whose `configs` is one. The wasm module reports it so a page
/// can check it against the API it fetches from.
pub const KERNEL_DATA_FORMAT: u32 = 1;

/// One config document, as the kernel API serves it. Fields the bridge does
/// not read (labels, uses, deployments) are ignored.
#[derive(Clone, Debug, Deserialize)]
pub struct ConfigDocument {
    pub kind: String,
    pub gpu: String,
    pub config_hash: String,
    pub identity: Value,
    pub axes: Vec<Vec<f64>>,
    pub points: Vec<ConfigPoint>,
}

/// One grid cell of a config document.
#[derive(Clone, Debug, Deserialize)]
pub struct ConfigPoint {
    pub feasible: bool,
    /// The cell's profile.db row under each backend that has one.
    #[serde(default)]
    pub measured: BTreeMap<String, MeasuredRow>,
}

/// A measured row's metrics as profile.db stores them. `time_ms` is null only
/// on a row that failed to measure; a missing `energy_j` reads as zero, as it
/// does through the Python bridge.
#[derive(Clone, Debug, Deserialize)]
pub struct MeasuredRow {
    #[serde(default)]
    pub time_ms: Option<f64>,
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
    #[serde(default)]
    pub outlier: bool,
}

impl MeasuredRow {
    fn metrics(&self) -> Option<KernelMetrics> {
        if self.outlier {
            return None;
        }
        Some(KernelMetrics {
            time_ms: self.time_ms?,
            tflops: self.tflops,
            memory_bandwidth_gbps: self.memory_bandwidth_gbps,
            algbw_gbps: self.algbw_gbps,
            busbw_gbps: self.busbw_gbps,
            energy_j: self.energy_j.unwrap_or(0.0),
        })
    }
}

type Key = (String, String, String);

/// Config documents keyed by `(kind, gpu, identity)`.
#[derive(Debug, Default)]
pub struct KernelData {
    configs: Mutex<HashMap<Key, Arc<ConfigDocument>>>,
    /// Fetch a document it does not hold from profile.db's registry.
    registry: bool,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Documents {
    List(Vec<ConfigDocument>),
    Wrapped { configs: Vec<ConfigDocument> },
}

impl KernelData {
    /// Parse config documents: a JSON array, or an object with `configs`.
    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        let (Documents::List(configs) | Documents::Wrapped { configs }) =
            serde_json::from_str(text)?;
        Ok(Self::from_documents(configs))
    }

    /// Key `documents`; a later duplicate replaces an earlier one.
    pub fn from_documents(documents: impl IntoIterator<Item = ConfigDocument>) -> Self {
        let configs = documents
            .into_iter()
            .map(|doc| (key(&doc.kind, &doc.gpu, &doc.identity), Arc::new(doc)))
            .collect();
        Self {
            configs: Mutex::new(configs),
            registry: false,
        }
    }

    /// Documents read from profile.db's kernel-config registry (through
    /// perf_api, so only with the `python` feature) as kernels ask for them.
    /// A config the registry does not hold fails to build, as a missing
    /// document does: register it by running what builds it.
    pub fn registry() -> Self {
        Self {
            registry: true,
            ..Self::default()
        }
    }

    /// How many documents it holds.
    pub fn len(&self) -> usize {
        self.configs.lock().expect("kernel data").len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn document(
        &self,
        kind: KernelKind,
        gpu_name: &str,
        identity: &Value,
    ) -> Result<Arc<ConfigDocument>, String> {
        let key = key(kind, gpu_name, identity);
        if let Some(doc) = self.configs.lock().expect("kernel data").get(&key) {
            return Ok(Arc::clone(doc));
        }
        let missing = || {
            format!("no config document for this {kind} config on {gpu_name} (identity {identity})")
        };
        if !self.registry {
            return Err(missing());
        }
        let text = super::python::get_config_document(kind, gpu_name, identity)
            .map_err(|e| format!("reading the {kind} config document from profile.db: {e}"))?
            .ok_or_else(|| format!("{}: profile.db registers no such config", missing()))?;
        let doc: ConfigDocument = serde_json::from_str(&text)
            .map_err(|e| format!("parsing the {kind} config document: {e}"))?;
        let doc = Arc::new(doc);
        self.configs
            .lock()
            .expect("kernel data")
            .insert(key, Arc::clone(&doc));
        Ok(doc)
    }

    /// The document of the config `(kind, gpu_name, identity)`, checked against
    /// the grid the kernel builds: the same axes, and the same infeasible cells
    /// (`infeasible` row-major, empty when every cell is feasible).
    pub fn config(
        &self,
        kind: KernelKind,
        gpu_name: &str,
        identity: &Value,
        axes: &[Vec<f64>],
        infeasible: &[bool],
    ) -> Result<Arc<ConfigDocument>, String> {
        let doc = self.document(kind, gpu_name, identity)?;
        if doc.axes != axes {
            return Err(format!(
                "config {} has grid axes {:?}, but the kernel builds {:?}",
                doc.config_hash, doc.axes, axes
            ));
        }
        let cells: usize = axes.iter().map(Vec::len).product();
        if doc.points.len() != cells {
            return Err(format!(
                "config {} has {} points for a grid of {cells} cells",
                doc.config_hash,
                doc.points.len()
            ));
        }
        let differs = doc
            .points
            .iter()
            .enumerate()
            .find(|(i, p)| p.feasible == infeasible.get(*i).copied().unwrap_or(false));
        if let Some((cell, point)) = differs {
            return Err(format!(
                "config {} marks cell {cell} {}, but the kernel finds it {}",
                doc.config_hash,
                feasibility(point.feasible),
                feasibility(!point.feasible),
            ));
        }
        Ok(doc)
    }
}

impl ConfigDocument {
    /// The row-major samples of `backend`: its measured row at each feasible
    /// cell and a non-finite placeholder at each infeasible one, as the cache
    /// fit expects. Errors at the first feasible cell without a usable row.
    pub fn samples(&self, backend: &str) -> Result<Vec<KernelMetrics>, String> {
        self.points
            .iter()
            .enumerate()
            .map(|(cell, point)| {
                if !point.feasible {
                    return Ok(KernelMetrics::non_finite());
                }
                point
                    .measured
                    .get(backend)
                    .and_then(MeasuredRow::metrics)
                    .ok_or_else(|| {
                        format!(
                            "config {} has no measured {backend} row at cell {cell}",
                            self.config_hash
                        )
                    })
            })
            .collect()
    }

    /// How many feasible cells have no usable `backend` row (the dry-run count).
    pub fn missing(&self, backend: &str) -> (usize, usize) {
        let feasible = self.points.iter().filter(|p| p.feasible);
        let total = feasible.clone().count();
        let measured = feasible
            .filter(|p| {
                p.measured
                    .get(backend)
                    .and_then(MeasuredRow::metrics)
                    .is_some()
            })
            .count();
        (total - measured, total)
    }
}

fn feasibility(feasible: bool) -> &'static str {
    if feasible {
        "feasible"
    } else {
        "infeasible"
    }
}

/// The lookup key: kind, GPU, and the identity as sorted-key JSON, so the
/// identity a kernel computes and the one a document carries meet on one
/// spelling whatever order their fields were written in.
fn key(kind: &str, gpu_name: &str, identity: &Value) -> (String, String, String) {
    (
        kind.to_string(),
        gpu_name.to_string(),
        serde_json::to_string(&sorted(identity)).expect("identity serializes"),
    )
}

fn sorted(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let sorted: BTreeMap<&String, Value> =
                map.iter().map(|(k, v)| (k, sorted(v))).collect();
            Value::Object(sorted.into_iter().map(|(k, v)| (k.clone(), v)).collect())
        }
        Value::Array(items) => Value::Array(items.iter().map(sorted).collect()),
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn document() -> Value {
        json!({
            "kind": "single_gemm",
            "gpu": "NVIDIA H200",
            "config_hash": "abc",
            "identity": {"n": 6144, "k": 4096, "dtype": "bf16"},
            "axes": [[1.0, 2.0]],
            "points": [
                {"feasible": true, "measured": {"torch": {"time_ms": 0.5, "tflops": 1.0}}},
                {"feasible": false, "measured": {}},
            ],
        })
    }

    fn data(doc: Value) -> KernelData {
        KernelData::from_json(&json!({ "configs": [doc] }).to_string()).unwrap()
    }

    #[test]
    fn a_kernel_finds_its_document_by_identity_in_any_field_order() {
        let data = data(document());
        let identity = json!({"dtype": "bf16", "k": 4096, "n": 6144});
        let doc = data
            .config(
                "single_gemm",
                "NVIDIA H200",
                &identity,
                &[vec![1.0, 2.0]],
                &[false, true],
            )
            .unwrap();
        let samples = doc.samples("torch").unwrap();
        assert_eq!(samples[0].time_ms, 0.5);
        assert_eq!(samples[0].energy_j, 0.0);
        assert!(!samples[1].time_ms.is_finite());
        assert_eq!(doc.missing("torch"), (0, 1));
    }

    #[test]
    fn a_grid_or_feasibility_the_document_disagrees_with_fails() {
        let data = data(document());
        let identity = json!({"n": 6144, "k": 4096, "dtype": "bf16"});
        let axes = data
            .config(
                "single_gemm",
                "NVIDIA H200",
                &identity,
                &[vec![1.0, 4.0]],
                &[false, true],
            )
            .unwrap_err();
        assert!(axes.contains("grid axes"), "{axes}");
        let cells = data
            .config(
                "single_gemm",
                "NVIDIA H200",
                &identity,
                &[vec![1.0, 2.0]],
                &[],
            )
            .unwrap_err();
        assert!(cells.contains("cell 1 infeasible"), "{cells}");
        let other = data
            .config(
                "single_gemm",
                "NVIDIA B200",
                &identity,
                &[vec![1.0, 2.0]],
                &[false, true],
            )
            .unwrap_err();
        assert!(other.contains("no config document"), "{other}");
    }

    #[test]
    fn an_outlier_or_unmeasured_backend_is_missing() {
        let mut doc = document();
        doc["points"][0]["measured"]["torch"]["outlier"] = json!(true);
        let data = data(doc);
        let identity = json!({"n": 6144, "k": 4096, "dtype": "bf16"});
        let doc = data
            .config(
                "single_gemm",
                "NVIDIA H200",
                &identity,
                &[vec![1.0, 2.0]],
                &[false, true],
            )
            .unwrap();
        assert!(doc.samples("torch").unwrap_err().contains("cell 0"));
        assert!(doc.samples("cublas").is_err());
        assert_eq!(doc.missing("torch"), (1, 1));
    }
}
