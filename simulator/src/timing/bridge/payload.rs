use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Kernel-kind identifier shared with Python (`profiling.db.kind.KernelKind`
/// values) and used as the facade stem: every kind `K` maps to Python facade
/// functions `get_{K}_times` and `count_missing_{K}`, derived via `format!` in
/// `bridge::core`. Adding a new kernel only declares one `KernelSpec::KIND`
/// constant in `kernels/<name>.rs` — there is no central enum to extend.
pub type KernelKind = &'static str;

/// Floating-/integer-point precision tag carried on every kernel spec.
///
/// **Wire form vs. serde form differ.** The on-wire string Python expects in
/// `ArgsPayload` is the snake_case literal returned by [`DType::as_str`]
/// (`"fp16"`, `"fp8_e4m3"`, ...). The derived `Serialize` / `Deserialize`
/// impls emit the variant *identifier* (`"Fp16"`, `"Fp8E4m3"`, ...) and are
/// intended only for internal Rust-to-Rust round-trips (e.g. cache identity
/// hashing). When constructing an `ArgsPayload` field that will cross the
/// PyO3 boundary, *always* go through `dtype.as_str()` — see
/// `kernels/single_gemm.rs::enumerate` for the canonical pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    Fp16,
    Bf16,
    Fp32,
    Fp8E4m3,
    Fp8E5m2,
    Int8,
    Int4,
}

impl DType {
    pub fn as_str(self) -> &'static str {
        match self {
            DType::Fp16 => "fp16",
            DType::Bf16 => "bf16",
            DType::Fp32 => "fp32",
            DType::Fp8E4m3 => "fp8_e4m3",
            DType::Fp8E5m2 => "fp8_e5m2",
            DType::Int8 => "int8",
            DType::Int4 => "int4",
        }
    }

    /// Bytes per element. Used by L3 worklets to size byte-keyed kernels (e.g.
    /// the elementwise activation's per-token I/O footprint). Int4 is sub-byte;
    /// it rounds up to 1 (no current model uses Int4 on a byte-keyed path).
    pub fn size_bytes(self) -> u32 {
        match self {
            DType::Fp32 => 4,
            DType::Fp16 | DType::Bf16 => 2,
            DType::Fp8E4m3 | DType::Fp8E5m2 | DType::Int8 | DType::Int4 => 1,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ArgsPayload {
    fields: BTreeMap<String, Value>,
}

impl ArgsPayload {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<Value>) {
        self.fields.insert(key.into(), value.into());
    }

    /// Builder-style insert: takes and returns `self` so a kernel's
    /// `enumerate` body can construct a payload in a single chained
    /// expression. The mutating `insert` is kept for incremental builds and
    /// for tests that pre-build a payload and assert on it.
    pub fn with(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.fields.insert(key.into(), value.into());
        self
    }

    pub fn fields(&self) -> &BTreeMap<String, Value> {
        &self.fields
    }

    pub fn backend(&self) -> Option<&str> {
        self.fields.get("backend").and_then(Value::as_str)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KernelMetrics {
    pub time_ms: f64,
    pub tflops: Option<f64>,
    pub memory_bandwidth_gbps: Option<f64>,
    pub algbw_gbps: Option<f64>,
    pub busbw_gbps: Option<f64>,
    pub message_size_bytes: Option<u64>,
    pub energy_j: f64,
}

impl KernelMetrics {
    /// True iff `time_ms` and every populated optional rate field is a
    /// non-negative finite f64. Cache `from_samples` scans this to emit
    /// `OutlierKind::NonFinite` instead of swallowing NaN/Inf into 0 via the
    /// `f64 as u64` cast in `flops()` / `bytes()`.
    pub fn is_finite(&self) -> bool {
        if !finite_non_negative(self.time_ms) || !finite_non_negative(self.energy_j) {
            return false;
        }
        for value in [
            self.tflops,
            self.memory_bandwidth_gbps,
            self.algbw_gbps,
            self.busbw_gbps,
        ]
        .into_iter()
        .flatten()
        {
            if !finite_non_negative(value) {
                return false;
            }
        }
        true
    }

    /// Absolute FLOPs from `tflops × time`. Returns 0 if `tflops` is absent
    /// (this is the **comm path** — collective ops legitimately don't carry
    /// a `tflops` rate; callers reading `flops == 0` on a comm row must not
    /// interpret it as a failure) or if the computed value is NaN/Inf via
    /// the [`finite_non_negative`] guard.
    pub fn flops(&self) -> u64 {
        let Some(tflops) = self.tflops else {
            return 0;
        };
        let value = tflops * (self.time_ms / 1000.0) * 1e12;
        if !finite_non_negative(value) {
            return 0;
        }
        value as u64
    }

    pub fn bytes(&self) -> u64 {
        if let Some(memory_bandwidth_gbps) = self.memory_bandwidth_gbps {
            let value = memory_bandwidth_gbps * (self.time_ms / 1000.0) * 1e9;
            if !finite_non_negative(value) {
                return 0;
            }
            return value as u64;
        }
        self.message_size_bytes.unwrap_or(0)
    }
}

fn finite_non_negative(value: f64) -> bool {
    value.is_finite() && value >= 0.0
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DbMetadata {
    pub schema_version: u32,
    pub schema_hash: String,
    pub created_at: Option<String>,
    pub last_migrated_at: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfilerVersion {
    pub op_family: String,
    pub profiler_git_hash: String,
}

#[cfg(test)]
mod tests {
    use super::{ArgsPayload, DType, KernelMetrics};
    use serde_json::Value;

    #[test]
    fn args_payload_with_builds_in_one_expression() {
        let payload = ArgsPayload::new()
            .with("backend", "torch")
            .with("m", 1024_u32)
            .with("dtype", DType::Fp16.as_str());
        let fields = payload.fields();
        assert_eq!(fields.get("backend"), Some(&Value::from("torch")));
        assert_eq!(fields.get("m"), Some(&Value::from(1024_u32)));
        assert_eq!(fields.get("dtype"), Some(&Value::from("fp16")));
        assert_eq!(payload.backend(), Some("torch"));
    }

    fn metrics(time_ms: f64, tflops: Option<f64>) -> KernelMetrics {
        KernelMetrics {
            time_ms,
            tflops,
            memory_bandwidth_gbps: None,
            algbw_gbps: None,
            busbw_gbps: None,
            message_size_bytes: None,
            energy_j: 0.0,
        }
    }

    #[test]
    fn is_finite_rejects_nan_inf_and_negative() {
        assert!(metrics(1.0, Some(1.0)).is_finite());
        assert!(!metrics(f64::NAN, Some(1.0)).is_finite());
        assert!(!metrics(f64::INFINITY, Some(1.0)).is_finite());
        assert!(!metrics(-1.0, Some(1.0)).is_finite());
        assert!(!metrics(1.0, Some(f64::NAN)).is_finite());
        assert!(!metrics(1.0, Some(-1.0)).is_finite());
    }

    #[test]
    fn flops_and_bytes_return_zero_on_nan_inf() {
        assert_eq!(metrics(f64::NAN, Some(1.0)).flops(), 0);
        assert_eq!(metrics(1.0, Some(f64::INFINITY)).flops(), 0);

        let mut m = metrics(f64::NAN, None);
        m.memory_bandwidth_gbps = Some(1.0);
        assert_eq!(m.bytes(), 0);
    }

    #[test]
    fn bytes_falls_back_to_message_size_when_no_mem_bw() {
        let mut m = metrics(1.0, None);
        m.message_size_bytes = Some(4096);
        assert_eq!(m.bytes(), 4096);
    }

    #[test]
    fn bytes_prefers_memory_bandwidth_when_both_fields_present() {
        // Compute rows always carry mem_bw; comm rows always carry
        // message_size_bytes. When both happen to be present, the rate-based
        // path must win to stay consistent with the compute interpretation.
        let mut m = metrics(2.0, None);
        m.memory_bandwidth_gbps = Some(100.0);
        m.message_size_bytes = Some(4096);
        // 100 GB/s * 2 ms = 100e9 * 2e-3 = 2e8 bytes
        assert_eq!(m.bytes(), 200_000_000);
    }
}
