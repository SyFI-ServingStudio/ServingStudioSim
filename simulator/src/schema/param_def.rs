//! Compile-time `ParamDef` data backing the `simulator list-params` schema dump.
//!
//! Per L7 design.md §1.8.1 / §1.2.7: each deployment carries a
//! `pub const PARAMS: &[ParamDef]` that the launcher reads as JSON via the
//! `simulator list-params` subcommand. Everything below must therefore be
//! `const`-constructible and hold only `'static` borrows — no `String`,
//! no heap.
//!
//! Wire shape (one JSON object per `ParamDef`, matching the §1.2.7 example):
//!
//! ```jsonc
//! { "name": "model_config", "type": "string", "required": true, "description": "..." }
//! { "name": "tp_size",      "type": "int",    "default": 4,     "description": "..." }
//! { "name": "duration_ms",  "type": "float",  "default": 5000.0, "description": "..." }
//! ```
//!
//! - `default` is a **bare** JSON scalar (int / float / bool / string),
//!   *never* an enum envelope.
//! - `choices` is an optional string array emitted for params whose legal values
//!   are a closed set. The launcher validates presets against it generically;
//!   do not mirror these choices in Python.
//! - `required` is emitted as `true` when the param has no default and is not
//!   `.optional()`; as `false` when it has no default but IS optional (an
//!   omittable `Option<T>` clap flag like `max_batch_tokens`, where `None` is a
//!   meaningful "unlimited" value); and omitted entirely when a `default` is
//!   present (default ⇒ not required).
//! - `.optional()` exists because a clap `Option<T>` field carries no default
//!   yet must not force the launcher to demand a value. The two-state model
//!   "has default / required" cannot represent an omittable-but-defaultless
//!   flag, so design.md §1.8.3 adds the optional marker.
//! - `.cache_key()` marks a param that changes **which kernel/comm configs
//!   `profile.db` must contain** (model identity/dims, sharding, dtype, fabric).
//!   The launcher groups sweep runs by the tuple of cache-key params and runs
//!   one `build-cache-only` per unique group (design §1.2.2). This criterion is
//!   genuinely L1/Rust knowledge — only Rust knows which params flow into
//!   `*KernelInput` / profile.db lookup keys — so it is Rust-authoritative here
//!   rather than a hardcoded list in Python (emitted as `"affects_cache": true`
//!   and read by `launcher.cache_build`). **Safe direction: when unsure, tag
//!   it.** Over-tagging only costs extra prebuild passes; under-tagging makes
//!   the launcher treat two runs needing different kernels as one group, so the
//!   second JIT-profiles on the GPU concurrently — the exact SQLite/contention
//!   bug §1.2.2 prevents.
//! - Round-trip is one-way (Rust → JSON → Python launcher); the Rust binary
//!   never reads the JSON back, so no `Deserialize` impls live here.

use serde::ser::{SerializeMap, Serializer};
use serde::Serialize;

/// Logical type of a `ParamDef`. Serde-serialized as snake_case
/// (`int` / `int_list` / `path_list` / ...).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamType {
    Int,
    Float,
    Bool,
    String,
    Path,
    IntList,
    FloatList,
    StringList,
    PathList,
}

/// Default value carried inline on a `ParamDef`. Serializes as a **bare**
/// JSON scalar (no `{"kind": ..., "value": ...}` envelope) so the launcher
/// JSON example in L7 §1.2.7 matches as-is. The Rust variant tag is only used
/// internally to record which scalar type the default holds; the static
/// `ParamType` field on `ParamDef` carries the authoritative wire type.
#[derive(Clone, Copy, Debug)]
pub enum DefaultValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(&'static str),
    Path(&'static str),
}

impl Serialize for DefaultValue {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        match *self {
            DefaultValue::Int(v) => ser.serialize_i64(v),
            DefaultValue::Float(v) => ser.serialize_f64(v),
            DefaultValue::Bool(v) => ser.serialize_bool(v),
            DefaultValue::String(v) | DefaultValue::Path(v) => ser.serialize_str(v),
        }
    }
}

/// Pool fragment that a `ParamDef` originated from. Code-organization tag
/// only — not part of the launcher wire format, so it is `#[serde(skip)]`-able
/// at the `ParamDef` level (no field added yet; introduced for downstream
/// `list-params --human` grouping and validators).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamSection {
    ModelCommon,
    ParallelismCommon,
    WorkloadCommon,
    IoCommon,
    DeploymentOwn,
}

/// Single CLI parameter declaration, immutable and `const`-constructible.
///
/// Builder methods return `Self` so deployment files can write
/// `ParamDef::int("foo").default_int(1).desc("...")` inside a `const`. A param
/// is **required** iff it has neither a default nor `.optional()`; an
/// `.optional()` param with no default serializes `"required": false`.
#[derive(Clone, Copy, Debug)]
pub struct ParamDef {
    pub name: &'static str,
    pub ty: ParamType,
    pub default: Option<DefaultValue>,
    /// Marks an omittable clap `Option<T>` flag that carries no default.
    /// Mutually meaningful only when `default` is `None`.
    pub optional: bool,
    /// Marks a param that determines which kernel/comm configs `profile.db`
    /// needs — i.e. part of the launcher's cache-grouping key (design §1.2.2).
    /// See module docs for the safe-direction rule.
    pub affects_cache: bool,
    /// Closed set of allowed string values. Empty means unrestricted.
    pub choices: &'static [&'static str],
    pub description: &'static str,
}

impl Serialize for ParamDef {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        let mut m = ser.serialize_map(None)?;
        m.serialize_entry("name", &self.name)?;
        m.serialize_entry("type", &self.ty)?;
        match self.default {
            Some(ref d) => m.serialize_entry("default", d)?,
            None => m.serialize_entry("required", &!self.optional)?,
        }
        if self.affects_cache {
            m.serialize_entry("affects_cache", &true)?;
        }
        if !self.choices.is_empty() {
            m.serialize_entry("choices", &self.choices)?;
        }
        m.serialize_entry("description", &self.description)?;
        m.end()
    }
}

impl ParamDef {
    const fn bare(name: &'static str, ty: ParamType) -> Self {
        Self {
            name,
            ty,
            default: None,
            optional: false,
            affects_cache: false,
            choices: &[],
            description: "",
        }
    }

    pub const fn int(name: &'static str) -> Self {
        Self::bare(name, ParamType::Int)
    }

    pub const fn float(name: &'static str) -> Self {
        Self::bare(name, ParamType::Float)
    }

    pub const fn bool(name: &'static str) -> Self {
        Self::bare(name, ParamType::Bool)
    }

    pub const fn string(name: &'static str) -> Self {
        Self::bare(name, ParamType::String)
    }

    pub const fn path(name: &'static str) -> Self {
        Self::bare(name, ParamType::Path)
    }

    pub const fn int_list(name: &'static str) -> Self {
        Self::bare(name, ParamType::IntList)
    }

    pub const fn float_list(name: &'static str) -> Self {
        Self::bare(name, ParamType::FloatList)
    }

    pub const fn string_list(name: &'static str) -> Self {
        Self::bare(name, ParamType::StringList)
    }

    pub const fn path_list(name: &'static str) -> Self {
        Self::bare(name, ParamType::PathList)
    }

    pub const fn default_int(mut self, v: i64) -> Self {
        self.default = Some(DefaultValue::Int(v));
        self
    }

    pub const fn default_float(mut self, v: f64) -> Self {
        self.default = Some(DefaultValue::Float(v));
        self
    }

    pub const fn default_bool(mut self, v: bool) -> Self {
        self.default = Some(DefaultValue::Bool(v));
        self
    }

    pub const fn default_string(mut self, v: &'static str) -> Self {
        self.default = Some(DefaultValue::String(v));
        self
    }

    pub const fn default_path(mut self, v: &'static str) -> Self {
        self.default = Some(DefaultValue::Path(v));
        self
    }

    pub const fn desc(mut self, description: &'static str) -> Self {
        self.description = description;
        self
    }

    /// Mark a defaultless clap `Option<T>` flag as omittable. Has no effect when
    /// a default is set (default already implies not-required).
    pub const fn optional(mut self) -> Self {
        self.optional = true;
        self
    }

    /// Mark this param as part of the launcher's `profile.db` cache-grouping key
    /// (design §1.2.2). Tag a param when changing it changes which kernel/comm
    /// configs must be profiled. See module docs for the safe-direction rule
    /// (when unsure, tag it).
    pub const fn cache_key(mut self) -> Self {
        self.affects_cache = true;
        self
    }

    /// Attach Rust-authoritative allowed values for a string-like param. The
    /// launcher consumes this as generic schema metadata rather than carrying
    /// deployment-specific enum tables in Python.
    pub const fn choices(mut self, choices: &'static [&'static str]) -> Self {
        self.choices = choices;
        self
    }

    /// Whether this param is required: no default AND not `.optional()`.
    /// Mirrors the `"required": true` field on the JSON wire form.
    pub const fn is_required(&self) -> bool {
        self.default.is_none() && !self.optional
    }
}

/// Compile-time schema composition for a clap `Args` struct: the ordered
/// `ParamDef` groups — each `#[command(flatten)]` fragment's `OWN_PARAMS`
/// followed by the struct's own params — that `#[derive(DeploymentParams)]`
/// assembles. A `Deployment`'s `PARAM_GROUPS` defaults to its `Args`' value
/// here, so deployments never re-list their fragments by hand.
pub trait ParamSchema {
    const PARAM_GROUPS: &'static [&'static [ParamDef]];
}
