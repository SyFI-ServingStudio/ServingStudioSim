//! Read-only hardware endpoint projection for ``gpu/spec.json``.
//!
//! Thin adapter over the crate-level catalog resolver ([`crate::hardware`]): the
//! one parser, the UI resolver keeps ``Option`` semantics (unknown dtype/GPU yields
//! ``None``/``unavailable`` and is never defaulted, e.g. to H200). This module
//! owns only the HTTP projection shape.

use anyhow::Result;
use serde_json::{json, Value};

pub(super) use crate::hardware::ResolvedGpu;

/// Resolve one GPU name against ``gpu/spec.json`` via the shared catalog.
/// ``Ok(None)`` = unmatched / unavailable (file missing, unparseable, or no exact
/// name/alias match); the resolver never fabricates a SKU.
pub(super) fn resolve_gpu(repo_root: &std::path::Path, name: &str) -> Result<Option<ResolvedGpu>> {
    Ok(crate::hardware::resolve_gpu(repo_root, name))
}

/// The public ``GET /api/analyzer/v1/hardware/gpus?name=...`` projection.
pub(super) fn hardware_gpu_response(requested: &str, resolved: Option<&ResolvedGpu>) -> Value {
    match resolved {
        Some(gpu) => json!({
            "schema_version": 1,
            "requested": requested,
            "matched": true,
            "available": true,
            "canonical_name": gpu.canonical_name,
            "matched_alias": gpu.matched_alias,
            "provenance": "catalog",
            "peaks": {
                "fp16_tflops": gpu.fp16_tflops,
                "bf16_tflops": gpu.bf16_tflops,
                "fp8_tflops": gpu.fp8_tflops,
                "fp32_tflops": gpu.fp32_tflops,
                "int8_tops": gpu.int8_tops,
            },
            "hbm_bandwidth_gbps": gpu.mem_bandwidth_gbps,
            "interconnect": {
                "name": gpu.interconnect,
                "bidirectional_gbps": gpu.interconnect_bandwidth_gbps,
                "one_way_gbps": gpu.one_way_gbps(),
            },
        }),
        None => json!({
            "schema_version": 1,
            "requested": requested,
            "matched": false,
            "available": false,
            "reason": "unmatched: no gpu/spec.json exact name or alias match",
        }),
    }
}
