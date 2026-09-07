//! Read-only GPU hardware catalog resolution - the single parser of `gpu/spec.json`.
//!
//! This is the crate-level home of the exact, case-insensitive `name` / `aliases`
//! lookup that both consumers share:
//!
//! - `ui_service/hardware.rs` - the read-only `/api/v1/hardware/gpus` endpoint
//!   and the curve/series hardware ceilings. Unknown dtype or GPU yields `None` /
//!   `unavailable`, never a fabricated default (e.g. H200).
//! - `optimality/spec.rs` - the R5 hardware-limit rung. It adapts the same
//!   resolver to its own `GpuSpec` with the documented bf16 fallback; adapter
//!   semantics live there, not here.
//!
//! All bandwidths are bytes/s; `interconnect_bandwidth_gbps` is BIDIRECTIONAL
//! and the derived one-way rate is exactly half. TFLOPS are DENSE (no 2:4
//! sparsity). `gpu/spec.json` is NOT on the timing path: profile.db rows are
//! keyed by the requested `gpu_name` string and modeled timings never read these
//! peaks.

use std::path::Path;

use serde_json::Value;

/// The peak table one canonical catalog entry contributes to curve enrichment /
/// the hardware API. Numeric fields are `None` when the catalog declares `null` for
/// that GPU (never fabricated zeros).
#[derive(Clone, Debug)]
pub struct ResolvedGpu {
    pub canonical_name: String,
    pub matched_alias: String,
    pub mem_bandwidth_gbps: Option<f64>,
    pub fp16_tflops: Option<f64>,
    pub bf16_tflops: Option<f64>,
    pub fp8_tflops: Option<f64>,
    pub fp4_tflops: Option<f64>,
    pub fp32_tflops: Option<f64>,
    pub int8_tops: Option<f64>,
    pub interconnect: Option<String>,
    pub interconnect_bandwidth_gbps: Option<f64>,
}

impl ResolvedGpu {
    /// Dense tensor-core peak for a dtype string; `None` = unrecognized dtype or
    /// a GPU lacking that dtype (callers must not draw a line).
    pub fn dense_tflops(&self, dtype: &str) -> Option<f64> {
        let normalized = dtype.trim().to_ascii_lowercase();
        if normalized.contains("fp4") || normalized.contains("e2m1") {
            self.fp4_tflops
        } else if normalized.contains("fp8")
            || normalized.contains("e4m3")
            || normalized.contains("e5m2")
        {
            self.fp8_tflops
        } else if normalized == "int8" {
            self.int8_tops
        } else if normalized.contains("fp16")
            || normalized.contains("half")
            || normalized.contains("float16")
        {
            self.fp16_tflops
        } else if normalized.contains("fp32")
            || normalized.contains("float32")
            || normalized.contains("tf32")
        {
            self.fp32_tflops
        } else if normalized.contains("bf16") || normalized.contains("bfloat16") {
            self.bf16_tflops
        } else {
            None
        }
    }

    /// The one-way interconnect rate: catalog bandwidth is BIDIRECTIONAL, so a
    /// transfer is exactly half.
    pub fn one_way_gbps(&self) -> Option<f64> {
        self.interconnect_bandwidth_gbps
            .map(|bidirectional| bidirectional / 2.0)
    }
}

/// Resolve one GPU name against `gpu/spec.json`. `None` = unmatched /
/// unavailable (file missing, unparseable, or no exact name/alias match). This is
/// the only place the catalog file is read or parsed.
pub fn resolve_gpu(repo_root: &Path, name: &str) -> Option<ResolvedGpu> {
    let target = name.trim().to_ascii_lowercase();
    if target.is_empty() {
        return None;
    }
    let spec_path = repo_root.join("gpu/spec.json");
    let text = std::fs::read_to_string(&spec_path).ok()?;
    let document: Value = serde_json::from_str(&text).ok()?;
    let gpus = document.get("gpus").and_then(Value::as_array)?;
    gpus.iter().find_map(|gpu| match_gpu(gpu, &target))
}

fn match_gpu(gpu: &Value, target: &str) -> Option<ResolvedGpu> {
    let canonical = gpu.get("name").and_then(Value::as_str)?.to_string();
    let aliases = gpu
        .get("aliases")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let matched = std::iter::once(canonical.clone())
        .chain(aliases.iter().cloned())
        .find(|candidate| candidate.trim().to_ascii_lowercase() == target)?;
    let num = |key: &str| gpu.get(key).and_then(Value::as_f64);
    Some(ResolvedGpu {
        canonical_name: canonical,
        matched_alias: matched,
        mem_bandwidth_gbps: num("mem_bandwidth_gbps"),
        fp16_tflops: num("fp16_tflops"),
        bf16_tflops: num("bf16_tflops"),
        fp8_tflops: num("fp8_tflops"),
        fp4_tflops: num("fp4_tflops"),
        fp32_tflops: num("fp32_tflops"),
        int8_tops: num("int8_tops"),
        interconnect: gpu
            .get("interconnect")
            .and_then(Value::as_str)
            .map(str::to_owned),
        interconnect_bandwidth_gbps: num("interconnect_bandwidth_gbps"),
    })
}
