//! The compute dtype that picks a leaf's R5 throughput peak.
//!
//! Each Rust kernel config marks its compute-dtype field with `#[compute_dtype]`
//! (`simulator kernel-list`, the same field the Kernel Library shows as a row's
//! precision). Config keys alone are not enough: an NVFP4 fused MoE takes BF16
//! activations (`input_dtype`) but runs its tensor cores on the FP4 expert
//! weights (`weight_format`), so a key-order guess gives it the BF16 peak and
//! the kernel looks faster than the hardware.
//!
//! The field table is read once per checkout and simulator build through
//! `kernel-query`'s `kernel_list` op. When it cannot be read (no built
//! simulator), lookups fall back to the old key-order guess and `source` says so.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::kernel_query::{run_kernel_query, simulator_binary};

#[derive(Debug, Default)]
pub(crate) struct ComputeDtypeFields {
    /// kind → its `#[compute_dtype]` field, or `None` for a dtype-agnostic kind.
    field_by_kind: HashMap<String, Option<String>>,
    /// `"kernel-list"`, or `"unavailable: <why>"` when lookups guess from keys.
    pub source: String,
}

type CacheKey = (PathBuf, Option<SystemTime>);

static CACHE: OnceLock<Mutex<HashMap<CacheKey, Arc<ComputeDtypeFields>>>> = OnceLock::new();

impl ComputeDtypeFields {
    /// The field table of the checkout at `repo_root`, cached per simulator build.
    pub(crate) fn for_repo(repo_root: Option<&Path>) -> Arc<Self> {
        let Some(repo_root) = repo_root else {
            return Arc::new(Self::unavailable("no owning checkout"));
        };
        let simulator = match simulator_binary(repo_root) {
            Ok(simulator) => simulator,
            Err(error) => return Arc::new(Self::unavailable(&format!("{error:#}"))),
        };
        let stamp = std::fs::metadata(&simulator)
            .and_then(|meta| meta.modified())
            .ok();
        let key = (repo_root.to_path_buf(), stamp);
        let cache = CACHE.get_or_init(Default::default);
        if let Some(fields) = cache.lock().unwrap().get(&key) {
            return fields.clone();
        }
        let fields = match read_kernel_list(repo_root, &simulator) {
            Ok(fields) => Arc::new(fields),
            // Not cached: the next analysis retries.
            Err(error) => return Arc::new(Self::unavailable(&format!("{error:#}"))),
        };
        cache.lock().unwrap().insert(key, fields.clone());
        fields
    }

    fn unavailable(reason: &str) -> Self {
        Self {
            field_by_kind: HashMap::new(),
            source: format!("unavailable: {reason}"),
        }
    }

    pub(crate) fn is_available(&self) -> bool {
        !self.field_by_kind.is_empty()
    }

    /// The dtype literal whose peak bounds this leaf; `"bf16"` for a
    /// dtype-agnostic kind (it is bounded by bandwidth, not this peak).
    pub(crate) fn dtype<'a>(&self, kind: &str, config: &'a Value) -> &'a str {
        match self.field_by_kind.get(kind) {
            Some(Some(field)) => config.get(field).and_then(Value::as_str).unwrap_or("bf16"),
            Some(None) => "bf16",
            None => guessed_dtype(config),
        }
    }
}

/// The pre-`kernel-list` guess: the first present of `dtype`, `q_dtype`,
/// `input_dtype`, `kv_dtype`, else bf16.
fn guessed_dtype(config: &Value) -> &str {
    ["dtype", "q_dtype", "input_dtype", "kv_dtype"]
        .into_iter()
        .find_map(|key| config.get(key).and_then(Value::as_str))
        .unwrap_or("bf16")
}

fn read_kernel_list(repo_root: &Path, simulator: &Path) -> Result<ComputeDtypeFields> {
    let response = run_kernel_query(repo_root, simulator, json!({"op": "kernel_list"}))
        .context("kernel-query kernel_list")?;
    let kernels = response
        .get("kernels")
        .and_then(Value::as_array)
        .context("kernel_list response has no kernels array")?;
    let mut field_by_kind = HashMap::with_capacity(kernels.len());
    for entry in kernels {
        let kind = entry
            .get("kind")
            .and_then(Value::as_str)
            .context("kernel_list entry has no kind")?;
        let field = entry
            .get("compute_dtype")
            .and_then(Value::as_str)
            .map(str::to_owned);
        field_by_kind.insert(kind.to_owned(), field);
    }
    if field_by_kind.is_empty() {
        anyhow::bail!("kernel_list returned no kinds");
    }
    Ok(ComputeDtypeFields {
        field_by_kind,
        source: "kernel-list".to_owned(),
    })
}

#[cfg(test)]
impl ComputeDtypeFields {
    pub(crate) fn from_fields(fields: &[(&str, Option<&str>)]) -> Self {
        Self {
            field_by_kind: fields
                .iter()
                .map(|(kind, field)| ((*kind).to_owned(), field.map(str::to_owned)))
                .collect(),
            source: "test".to_owned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_marked_field_wins_over_an_activation_dtype_key() {
        // Catches an NVFP4 MoE getting the BF16 peak from its `input_dtype`.
        let fields = ComputeDtypeFields::from_fields(&[("nvfp4_fused_moe", Some("weight_format"))]);
        let config = json!({"input_dtype": "bf16", "weight_format": "nvfp4_e2m1"});
        assert_eq!(fields.dtype("nvfp4_fused_moe", &config), "nvfp4_e2m1");
    }

    #[test]
    fn a_dtype_agnostic_kind_does_not_borrow_a_config_key() {
        let fields = ComputeDtypeFields::from_fields(&[("elementwise", None)]);
        assert_eq!(
            fields.dtype("elementwise", &json!({"dtype": "fp8_e4m3"})),
            "bf16"
        );
    }

    #[test]
    fn an_unlisted_kind_keeps_the_key_order_guess() {
        let fields = ComputeDtypeFields::unavailable("test");
        assert_eq!(
            fields.dtype("x", &json!({"q_dtype": "fp8_e4m3"})),
            "fp8_e4m3"
        );
        assert_eq!(fields.dtype("x", &json!({})), "bf16");
    }
}
