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
/// `Serialize`/`Deserialize` are hand-written so the wire form is single-sourced
/// on [`DType::as_str`] / [`DType::from_wire`] (`"fp16"`, `"bf16"`, `"fp8_e4m3"`,
/// ...) — NOT serde's `rename_all`, which would re-derive the mapping from the
/// variant identifiers independently and could silently drift from `as_str`. A
/// `KernelConfig` thus deserializes its dtype fields directly from the wire
/// literal with no per-field helper. `as_str`/`from_wire` are kept in lockstep by
/// `dtype_wire_roundtrips_for_every_variant` below.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

    /// Inverse of [`DType::as_str`]: parse the snake_case wire literal back to a
    /// variant. The single source of the wire→variant mapping (paired with
    /// `as_str` for variant→wire); used by the hand-written `Deserialize`.
    pub fn from_wire(s: &str) -> Option<DType> {
        Some(match s {
            "fp16" => DType::Fp16,
            "bf16" => DType::Bf16,
            "fp32" => DType::Fp32,
            "fp8_e4m3" => DType::Fp8E4m3,
            "fp8_e5m2" => DType::Fp8E5m2,
            "int8" => DType::Int8,
            "int4" => DType::Int4,
            _ => return None,
        })
    }
}

impl Serialize for DType {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for DType {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        DType::from_wire(&s).ok_or_else(|| {
            serde::de::Error::custom(format!(
                "unknown dtype {s:?} (expected fp16/bf16/fp32/fp8_e4m3/fp8_e5m2/int8/int4)"
            ))
        })
    }
}

/// serde `deserialize_with` for a `KernelConfig`'s `backends: Vec<&'static str>`
/// field, which is otherwise not `Deserialize` (a borrowed `'static` can't own
/// JSON-provided strings). Reads `Vec<String>` and leaks each into a
/// `&'static str`. Only invoked on the `kernel-query` path, which builds one
/// kernel in a short-lived subprocess and exits — leaking a few backend strings
/// is negligible and avoids maintaining a central known-backend registry.
pub(crate) fn de_backends<'de, D>(d: D) -> Result<Vec<&'static str>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v = Vec::<String>::deserialize(d)?;
    Ok(v.into_iter()
        .map(|s| &*Box::leak(s.into_boxed_str()))
        .collect())
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
    /// A non-finite sentinel for a grid cell that is physically infeasible (no
    /// shape exists to profile). `Cache2DLinear::from_samples` treats this as a
    /// dropped cell (`!is_finite()`), so lookups renormalize over the feasible
    /// corners instead of trusting fabricated data. Used to slot placeholders
    /// back into a sample vector after the infeasible cells were excluded from
    /// profiling — see `KernelSpec::infeasible_mask`.
    pub fn non_finite() -> Self {
        Self {
            time_ms: f64::NAN,
            tflops: None,
            memory_bandwidth_gbps: None,
            algbw_gbps: None,
            busbw_gbps: None,
            message_size_bytes: None,
            energy_j: f64::NAN,
        }
    }

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
    fn dtype_wire_roundtrips_for_every_variant() {
        // `as_str` and `from_wire` are the single source of the DType wire
        // mapping (the hand-written serde impls defer to them). Assert they're
        // mutual inverses for every variant, so neither can drift unnoticed.
        for dt in [
            DType::Fp16,
            DType::Bf16,
            DType::Fp32,
            DType::Fp8E4m3,
            DType::Fp8E5m2,
            DType::Int8,
            DType::Int4,
        ] {
            assert_eq!(DType::from_wire(dt.as_str()), Some(dt));
            // serde goes through the same path: "bf16" not "Bf16".
            assert_eq!(
                serde_json::from_value::<DType>(Value::from(dt.as_str())).unwrap(),
                dt
            );
        }
        assert_eq!(DType::from_wire("Bf16"), None);
    }

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
