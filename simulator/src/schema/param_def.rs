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
//! - `required` is emitted as `true` exactly when the param has no default;
//!   otherwise it is omitted (the presence of `default` implies not-required).
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
/// `ParamDef::int("foo").default_int(1).desc("...")` inside a `const`. There
/// is no `.optional()` builder: a param is **required** iff it has no default.
#[derive(Clone, Copy, Debug)]
pub struct ParamDef {
    pub name: &'static str,
    pub ty: ParamType,
    pub default: Option<DefaultValue>,
    pub description: &'static str,
}

impl Serialize for ParamDef {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        let mut m = ser.serialize_map(None)?;
        m.serialize_entry("name", &self.name)?;
        m.serialize_entry("type", &self.ty)?;
        match self.default {
            Some(ref d) => m.serialize_entry("default", d)?,
            None => m.serialize_entry("required", &true)?,
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

    /// Whether this param is required (i.e. has no default). Mirrors the
    /// `"required": true` field on the JSON wire form.
    pub const fn is_required(&self) -> bool {
        self.default.is_none()
    }
}
