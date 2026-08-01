//! Hardware peaks from `gpu/spec.json` — the R5 "hardware limit" rung.
//!
//! R5 divides each leaf's active throughput unit by the matching spec-sheet
//! peak. Unlocked analysis reuses R3's grid regime; locked-batch analysis uses
//! the leaf's current operating-point regime. The R4→R5 gap is therefore the
//! profiled↔hardware maturity gap under the chosen batching assumption.
//!
//! The catalog JSON is parsed in exactly one place: [`crate::hardware::resolve_gpu`].
//! This module is a thin adapter that maps the shared `ResolvedGpu` (Option fields)
//! onto the `GpuSpec` shape the fold needs. Adapter semantics are explicit here:
//!
//! - `gpu/spec.json` JSON `null` (unsupported dtype for that GPU) maps to `0.0`,
//!   meaning "no hardware ceiling, R5 leaf degrades to R4".
//! - `peak_tflops` falls back to bf16 for an unrecognized dtype (the common
//!   training/inference default) — the UI resolver, by contrast, keeps `None` for
//!   unknown dtypes and never defaulted GPUs.

use std::path::Path;

use crate::hardware::resolve_gpu;

/// The peak rates the R5 ceiling needs for one GPU model.
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
    /// to `0.0` — meaning "no hardware ceiling, R5 leaf degrades to R4" — when the
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

/// Resolve the run's `gpu_name` to a `gpu/spec.json` entry via the shared
/// exact `name`/`aliases` resolver. Returns the matched canonical spec name (for
/// the report, so the resolution is auditable) + its peaks. `None` when the
/// file is absent/unparseable or no alias matches — R5 then collapses onto R4 (no
/// hardware-gap bucket) with a logged caveat pointing at the missing alias.
pub(crate) fn load_gpu_spec(repo_root: &Path, gpu_name: &str) -> Option<(String, GpuSpec)> {
    let resolved = resolve_gpu(repo_root, gpu_name)?;
    let spec = GpuSpec {
        fp8_tflops: resolved.fp8_tflops.unwrap_or(0.0),
        bf16_tflops: resolved.bf16_tflops.unwrap_or(0.0),
        fp16_tflops: resolved.fp16_tflops.unwrap_or(0.0),
        fp32_tflops: resolved.fp32_tflops.unwrap_or(0.0),
        int8_tops: resolved.int8_tops.unwrap_or(0.0),
        mem_bandwidth_gbps: resolved.mem_bandwidth_gbps.unwrap_or(0.0),
    };
    Some((resolved.canonical_name, spec))
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
