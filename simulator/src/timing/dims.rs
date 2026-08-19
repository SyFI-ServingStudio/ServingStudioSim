//! `Dim` — a symbolic model dimension: its folded `u32` value **and** the
//! expression that derived it from model-config inputs.
//!
//! Every per-op fixed shape (a GEMM's `n`/`k`, a norm's `hidden`, an
//! elementwise's `*_bytes_per_token`) is computed from `ModelCfg` inputs by
//! straight-line arithmetic in `arch::build_configs` / `worklet::resolve_config`
//! (`(num_qo_heads + 2*num_kv_heads)*head_dim`, `2*intermediate`, `hidden/tp`,
//! …). Storing those shapes as `Dim` instead of `u32` means the SAME arithmetic
//! that folds the value ALSO records the formula (operator overloading builds
//! the tree), so `describe()` / the cost manifest can render
//! `n=(num_qo_heads+2*num_kv_heads)*head_dim=6144` — value and provenance, for
//! free. See `arch/README.md`.
//!
//! ## Two invariants that keep this cheap and correct
//!
//! - **Value-based identity.** `PartialEq`/`Eq`/`Hash`/`Ord` compare only the
//!   folded [`Dim::get`] value, ignoring provenance. A `*KernelConfig`'s derived
//!   `Hash`/`Eq` is its `profile.db` cache key, so two configs that fold to the
//!   same shape stay ONE cache row regardless of how each dim was derived —
//!   provenance rides along invisibly and changes no cache behavior.
//! - **Collapse only at external boundaries.** [`Dim::get`] (fold to `u32`) is
//!   allowed only where a concrete integer must leave the cost-model world: the
//!   profiler bridge (`enumerate` → `ArgsPayload` → `profile.db`) and resource
//!   sizing (a worker dividing its memory budget by kv-bytes-per-token). A
//!   `.get()` inside `arch`/`worklet`/`op` arithmetic is a smell — keep it `Dim`.
//!
//! Backed by `Arc` (not `Rc`): a built model is shared as `Arc<M>` across worker
//! threads, so `Dim` must be `Send + Sync`. Clone is an `Arc` bump.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Sub};
use std::sync::Arc;

use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A model dimension carrying its folded value + derivation. Cheap to clone.
#[derive(Clone)]
pub struct Dim(Arc<DimNode>);

#[derive(Debug)]
enum DimNode {
    /// A model-config input (the provenance leaf), e.g. `hidden` = 4096.
    Param { name: &'static str, value: u32 },
    /// A bare literal (a config knob, or a value deserialized without a name).
    Const(u32),
    /// A derived dim: the memoized folded `value` plus the `op` and operands
    /// that produced it.
    Expr {
        value: u32,
        op: DimOp,
        lhs: Dim,
        rhs: Dim,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DimOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl Dim {
    /// A named model-config input — the root of a provenance chain.
    #[must_use]
    pub fn param(name: &'static str, value: u32) -> Dim {
        Dim(Arc::new(DimNode::Param { name, value }))
    }

    /// An anonymous literal (config knob or deserialized value).
    #[must_use]
    pub fn lit(value: u32) -> Dim {
        Dim(Arc::new(DimNode::Const(value)))
    }
}

impl From<u32> for Dim {
    fn from(value: u32) -> Dim {
        Dim::lit(value)
    }
}

impl Dim {
    /// The folded value. O(1): every node memoizes it. THIS is the collapse to a
    /// raw integer — only call it at an external boundary (see module docs).
    #[must_use]
    pub fn get(&self) -> u32 {
        match &*self.0 {
            DimNode::Param { value, .. } | DimNode::Const(value) => *value,
            DimNode::Expr { value, .. } => *value,
        }
    }

    /// The set of model-config input names this dim was derived from — the
    /// "which model dims feed this shape" query.
    #[must_use]
    pub fn params(&self) -> BTreeSet<&'static str> {
        let mut set = BTreeSet::new();
        self.collect_params(&mut set);
        set
    }

    fn collect_params(&self, set: &mut BTreeSet<&'static str>) {
        match &*self.0 {
            DimNode::Param { name, .. } => {
                set.insert(*name);
            }
            DimNode::Const(_) => {}
            DimNode::Expr { lhs, rhs, .. } => {
                lhs.collect_params(set);
                rhs.collect_params(set);
            }
        }
    }

    /// The `symbol -> value` bindings of every named input in this dim's formula
    /// — the legend that resolves a rendered expression (`num_qo_heads/attn_tp`)
    /// to its concrete parts (`{num_qo_heads: 64, attn_tp: 4}`). Powers a UI that
    /// toggles a leaf between its expression and its value. `Const` numbers carry
    /// no name and are not listed (they already read literally in the formula).
    #[must_use]
    pub fn bindings(&self) -> BTreeMap<&'static str, u32> {
        let mut map = BTreeMap::new();
        self.collect_bindings(&mut map);
        map
    }

    fn collect_bindings(&self, map: &mut BTreeMap<&'static str, u32>) {
        match &*self.0 {
            DimNode::Param { name, value } => {
                map.insert(*name, *value);
            }
            DimNode::Const(_) => {}
            DimNode::Expr { lhs, rhs, .. } => {
                lhs.collect_bindings(map);
                rhs.collect_bindings(map);
            }
        }
    }
}

/// Build a derived node, folding the value eagerly so `get()` stays O(1).
fn expr(op: DimOp, lhs: Dim, rhs: Dim) -> Dim {
    let (a, b) = (lhs.get(), rhs.get());
    let value = match op {
        DimOp::Add => a + b,
        DimOp::Sub => a - b,
        DimOp::Mul => a * b,
        DimOp::Div => a / b,
    };
    Dim(Arc::new(DimNode::Expr {
        value,
        op,
        lhs,
        rhs,
    }))
}

// --- Arithmetic: Dim⊕Dim, Dim⊕u32, u32⊕Dim for +, -, *, / ------------------
// Operators consume `self` by value (Dim is not Copy). A dim field read out of a
// borrowed `&cfg` therefore needs `.clone()` before it enters an expression —
// the Arc clone is cheap. The `u32⊕Dim` direction (e.g. the left literal in
// `2 * intermediate`) is orphan-rule-legal because `Dim` is a local type.
macro_rules! impl_binop {
    ($trait:ident, $method:ident, $variant:ident) => {
        impl $trait for Dim {
            type Output = Dim;
            fn $method(self, rhs: Dim) -> Dim {
                expr(DimOp::$variant, self, rhs)
            }
        }
        impl $trait<u32> for Dim {
            type Output = Dim;
            fn $method(self, rhs: u32) -> Dim {
                expr(DimOp::$variant, self, Dim::lit(rhs))
            }
        }
        impl $trait<Dim> for u32 {
            type Output = Dim;
            fn $method(self, rhs: Dim) -> Dim {
                expr(DimOp::$variant, Dim::lit(self), rhs)
            }
        }
    };
}
impl_binop!(Add, add, Add);
impl_binop!(Sub, sub, Sub);
impl_binop!(Mul, mul, Mul);
impl_binop!(Div, div, Div);

// --- Value-based identity (see module docs: cache-key invariant) -------------
impl PartialEq for Dim {
    fn eq(&self, other: &Dim) -> bool {
        self.get() == other.get()
    }
}
impl Eq for Dim {}
impl Hash for Dim {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.get().hash(state);
    }
}
impl PartialOrd for Dim {
    fn partial_cmp(&self, other: &Dim) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Dim {
    fn cmp(&self, other: &Dim) -> std::cmp::Ordering {
        self.get().cmp(&other.get())
    }
}

/// Compare against a raw `u32` so tests / stray checks read `dim == 6144`
/// without an explicit `.get()`.
impl PartialEq<u32> for Dim {
    fn eq(&self, other: &u32) -> bool {
        self.get() == *other
    }
}
impl PartialEq<Dim> for u32 {
    fn eq(&self, other: &Dim) -> bool {
        *self == other.get()
    }
}

// --- Rendering: Display = formula, Debug = formula=value ---------------------
fn op_str(op: DimOp) -> &'static str {
    match op {
        DimOp::Add => "+",
        DimOp::Sub => "-",
        DimOp::Mul => "*",
        DimOp::Div => "/",
    }
}

fn op_prec(op: DimOp) -> u8 {
    match op {
        DimOp::Add | DimOp::Sub => 1,
        DimOp::Mul | DimOp::Div => 2,
    }
}

/// Render `child` inside a parent of precedence `parent_prec`, parenthesizing
/// only a lower-precedence sub-expression (`(a+b)*c` but `a+b*c`). Best-effort:
/// the resolve formulas never nest division on the right, where associativity
/// would otherwise need extra parens.
fn write_child(f: &mut fmt::Formatter<'_>, child: &Dim, parent_prec: u8) -> fmt::Result {
    match &*child.0 {
        DimNode::Expr { op, .. } if op_prec(*op) < parent_prec => write!(f, "({child})"),
        _ => write!(f, "{child}"),
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &*self.0 {
            DimNode::Param { name, .. } => write!(f, "{name}"),
            DimNode::Const(v) => write!(f, "{v}"),
            DimNode::Expr { op, lhs, rhs, .. } => {
                let prec = op_prec(*op);
                write_child(f, lhs, prec)?;
                write!(f, "{}", op_str(*op))?;
                write_child(f, rhs, prec)
            }
        }
    }
}

impl fmt::Debug for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // A bare literal is just its number; anything with a formula shows
        // `formula=value` (this is what `describe_config`'s `{:?}` renders).
        match &*self.0 {
            DimNode::Const(v) => write!(f, "{v}"),
            _ => write!(f, "{}={}", self, self.get()),
        }
    }
}

// --- serde: numeric value plus provenance ------------------------------------
// Cost manifests are the machine-readable kernel-config boundary. Preserve the
// expression and its bindings there while keeping `value` authoritative for a
// kernel-query round-trip; deserialization intentionally rebuilds an anonymous
// `Const` because cache identity depends only on the folded value.
impl Serialize for Dim {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut state = s.serialize_struct("Dim", 3)?;
        state.serialize_field("value", &self.get())?;
        let expression = match &*self.0 {
            DimNode::Const(_) => None,
            _ => Some(self.to_string()),
        };
        state.serialize_field("expression", &expression)?;
        state.serialize_field("bindings", &self.bindings())?;
        state.end()
    }
}
impl<'de> Deserialize<'de> for Dim {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum DimWire {
            Value(u32),
            Rich {
                value: u32,
                #[serde(default, rename = "expression")]
                _expression: Option<String>,
                #[serde(default, rename = "bindings")]
                _bindings: BTreeMap<String, u32>,
            },
        }
        let value = match DimWire::deserialize(d)? {
            DimWire::Value(value) => value,
            DimWire::Rich {
                value,
                _expression: _,
                _bindings: _,
            } => value,
        };
        Ok(Dim::lit(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::hash_map::DefaultHasher;

    fn hash_of(d: &Dim) -> u64 {
        let mut h = DefaultHasher::new();
        d.hash(&mut h);
        h.finish()
    }

    #[test]
    fn folds_value_through_arithmetic() {
        let qo = Dim::param("num_qo_heads", 32);
        let kv = Dim::param("num_kv_heads", 8);
        let head = Dim::param("head_dim", 128);
        let qkv_n = (qo + 2 * kv) * head;
        assert_eq!(qkv_n.get(), (32 + 2 * 8) * 128); // 6144
    }

    #[test]
    fn bindings_maps_each_named_input_to_its_value() {
        let qo = Dim::param("num_qo_heads", 64);
        let kv = Dim::param("num_kv_heads", 4);
        let head = Dim::param("head_dim", 128);
        let tp = Dim::param("attn_tp", 4);
        // (num_qo_heads/attn_tp + 2*num_kv_heads/attn_tp) * head_dim
        let n = (qo / tp.clone() + 2 * kv / tp) * head;
        let b = n.bindings();
        assert_eq!(b.get("num_qo_heads"), Some(&64));
        assert_eq!(b.get("num_kv_heads"), Some(&4));
        assert_eq!(b.get("attn_tp"), Some(&4));
        assert_eq!(b.get("head_dim"), Some(&128));
        assert_eq!(b.len(), 4); // the literal `2` (Const) carries no name
    }

    #[test]
    fn display_renders_precedence_aware_formula() {
        let qo = Dim::param("num_qo_heads", 32);
        let kv = Dim::param("num_kv_heads", 8);
        let head = Dim::param("head_dim", 128);
        let qkv_n = (qo + 2 * kv) * head;
        // Mul inside Add needs no parens; Add inside Mul does.
        assert_eq!(format!("{qkv_n}"), "(num_qo_heads+2*num_kv_heads)*head_dim");
        // Debug appends the folded value.
        assert_eq!(
            format!("{qkv_n:?}"),
            "(num_qo_heads+2*num_kv_heads)*head_dim=6144"
        );
    }

    #[test]
    fn const_debug_is_bare_number() {
        assert_eq!(format!("{:?}", Dim::lit(16)), "16");
        assert_eq!(format!("{:?}", Dim::param("hidden", 4096)), "hidden=4096");
    }

    #[test]
    fn identity_is_value_based_ignoring_provenance() {
        // Same folded value, different derivations → equal and same hash, so a
        // profile.db cache keyed on the config stays one row.
        let via_expr = Dim::param("hidden", 4096) + 0; // Expr folding to 4096
        let via_param = Dim::param("something_else", 4096);
        let via_lit = Dim::lit(4096);
        assert_eq!(via_expr, via_param);
        assert_eq!(via_param, via_lit);
        assert_eq!(hash_of(&via_expr), hash_of(&via_lit));
        assert_eq!(hash_of(&via_param), hash_of(&via_lit));
    }

    #[test]
    fn compares_against_raw_u32() {
        let n = (Dim::param("num_qo_heads", 32) + 2 * Dim::param("num_kv_heads", 8))
            * Dim::param("head_dim", 128);
        assert_eq!(n, 6144);
        assert_eq!(6144, n);
    }

    #[test]
    fn division_folds_and_records() {
        let heads = Dim::param("num_qo_heads", 32);
        let per_rank = heads / 8u32; // TP shard
        assert_eq!(per_rank.get(), 4);
        assert_eq!(format!("{per_rank}"), "num_qo_heads/8");
    }

    #[test]
    fn params_collects_only_named_leaves() {
        let n = (Dim::param("num_qo_heads", 32) + 2 * Dim::param("num_kv_heads", 8))
            * Dim::param("head_dim", 128);
        let ps = n.params();
        assert!(ps.contains("num_qo_heads"));
        assert!(ps.contains("num_kv_heads"));
        assert!(ps.contains("head_dim"));
        assert_eq!(ps.len(), 3); // the literal 2 is not a named param
    }

    #[test]
    fn serde_round_trips_value_and_emits_provenance() {
        let d = Dim::param("hidden", 4096);
        let json = serde_json::to_value(&d).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "value": 4096,
                "expression": "hidden",
                "bindings": {"hidden": 4096},
            })
        );
        let back: Dim = serde_json::from_value(json).unwrap();
        assert_eq!(back.get(), 4096); // value survives; provenance does not
        let legacy_number: Dim = serde_json::from_value(serde_json::json!(2048)).unwrap();
        assert_eq!(legacy_number.get(), 2048);
    }
}
