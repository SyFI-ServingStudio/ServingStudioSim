//! Hardware roofline peaks from `gpu/spec.json` — the R5 "hardware limit" rung.
//!
//! R5 divides each compute leaf's work by the GPU's *spec-sheet* dense peak (not a
//! profiled kernel's peak), so the R4→R5 gap is the profiled↔hardware maturity gap.
//! `gpu/spec.json` is a flat list of GPU dicts (`fp8_tflops`, `bf16_tflops`,
//! `mem_bandwidth_gbps`, …), each carrying an explicit `aliases` list — the
//! authoritative set of names a run's `gpu_name` might use (`"NVIDIA H200"`,
//! `"H200"`, …) for the canonical SKU (`"H200-SXM-141GB"`). Matching is a
//! case-insensitive exact lookup against `name` ∪ `aliases`; no fuzzy guessing.
//!
//! That alias table is meant to be the single GPU-name canonicalization source for
//! the whole system (this roofline today; preset parsing / profile.db key matching
//! later), so extending coverage is one JSON edit, not a code change here.

use std::path::Path;

use serde_json::Value;

/// The peak rates the R5 roofline needs for one GPU model.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct GpuSpec {
    fp8_tflops: f64,
    bf16_tflops: f64,
    fp16_tflops: f64,
    fp32_tflops: f64,
    int8_tops: f64,
    pub mem_bandwidth_gbps: f64,
}

impl GpuSpec {
    /// Dense compute peak (TFLOP/s) for a leaf's compute `dtype`. Falls back to
    /// bf16 (the common training/inference default) for an unrecognized dtype, and
    /// to `0.0` — meaning "no hardware roofline, R5 leaf degrades to R4" — when the
    /// matched spec entry lacks that field (e.g. an fp8 kernel on a pre-fp8 GPU).
    pub fn peak_tflops(&self, dtype: &str) -> f64 {
        let d = dtype.to_ascii_lowercase();
        if d.contains("fp8") || d.contains("e4m3") || d.contains("e5m2") {
            self.fp8_tflops
        } else if d.contains("int8") {
            self.int8_tops
        } else if d.contains("fp16") || d.contains("half") {
            self.fp16_tflops
        } else if d.contains("fp32") || d.contains("float32") || d.contains("tf32") {
            self.fp32_tflops
        } else {
            // bf16 / bfloat16 / anything else → bf16 peak.
            self.bf16_tflops
        }
    }
}

/// Resolve the run's `gpu_name` to a `gpu/spec.json` entry via its explicit
/// `name`/`aliases`. Returns the matched canonical spec name (for the report, so
/// the resolution is auditable) + its peaks. `None` when the file is
/// absent/unparseable or no alias matches — R5 then collapses onto R4 (no
/// hardware-gap bucket) with a logged caveat pointing at the missing alias.
pub(crate) fn load_gpu_spec(repo_root: &Path, gpu_name: &str) -> Option<(String, GpuSpec)> {
    let text = std::fs::read_to_string(repo_root.join("gpu/spec.json")).ok()?;
    let doc: Value = serde_json::from_str(&text).ok()?;
    let gpus = doc.get("gpus")?.as_array()?;

    let target = gpu_name.trim().to_ascii_lowercase();
    if target.is_empty() {
        return None;
    }
    for gpu in gpus {
        let name = gpu.get("name").and_then(Value::as_str).unwrap_or("");
        let matches = std::iter::once(name)
            .chain(
                gpu.get("aliases")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str),
            )
            .any(|candidate| candidate.trim().to_ascii_lowercase() == target);
        if matches {
            let spec = GpuSpec {
                fp8_tflops: field(gpu, "fp8_tflops"),
                bf16_tflops: field(gpu, "bf16_tflops"),
                fp16_tflops: field(gpu, "fp16_tflops"),
                fp32_tflops: field(gpu, "fp32_tflops"),
                int8_tops: field(gpu, "int8_tops"),
                mem_bandwidth_gbps: field(gpu, "mem_bandwidth_gbps"),
            };
            return Some((name.to_string(), spec));
        }
    }
    None
}

/// A spec field, treating JSON `null` (unsupported dtype for that GPU) as `0.0`.
fn field(gpu: &Value, key: &str) -> f64 {
    gpu.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_the_real_spec_by_alias() {
        // The shipped gpu/spec.json is two dirs up from analyzer/rust at build time;
        // resolve against the repo root the way the subject does (current dir).
        let root = std::env::current_dir().unwrap();
        // analyzer/rust is the crate dir under `cargo test`; hop to the repo root.
        let repo = root
            .ancestors()
            .find(|p| p.join("gpu/spec.json").is_file())
            .expect("find gpu/spec.json above the crate dir");
        let (name, spec) = load_gpu_spec(repo, "NVIDIA H200").expect("H200 alias resolves");
        assert_eq!(name, "H200-SXM-141GB");
        assert_eq!(spec.peak_tflops("fp8_e4m3"), 1979.0);
        assert_eq!(spec.peak_tflops("bf16"), 990.0);
        assert_eq!(spec.mem_bandwidth_gbps, 4800.0);
        // A bare "H200" is an alias too; an unknown name does not resolve.
        assert!(load_gpu_spec(repo, "H200").is_some());
        assert!(load_gpu_spec(repo, "Totally Made Up GPU").is_none());
    }

    #[test]
    fn dtype_maps_to_the_right_peak() {
        let s = GpuSpec {
            fp8_tflops: 1979.0,
            bf16_tflops: 990.0,
            int8_tops: 1979.0,
            ..Default::default()
        };
        assert_eq!(s.peak_tflops("fp8_e4m3"), 1979.0);
        assert_eq!(s.peak_tflops("bf16"), 990.0);
        assert_eq!(s.peak_tflops("unknown"), 990.0);
        assert_eq!(s.peak_tflops("int8"), 1979.0);
    }
}
