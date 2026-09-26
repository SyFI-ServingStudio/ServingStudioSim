//! Which deployments an arch supports: the `#[supported(...)]` rows on an arch
//! selector variant.
//!
//! A param's type says what a value may look like; it cannot say that
//! `llama3_dense_tp` runs Llama 3 8B on H200 at TP 1, 2, 4 or 8 and nothing
//! else. Each arch variant states that itself, one row per attribute:
//!
//! ```ignore
//! #[supported(gpu = ["NVIDIA H200"], model_config = ["llama3_8b"], tp_size = [1, 2, 4, 8])]
//! Llama3DenseTp { .. }
//! ```
//!
//! Every row names a `gpu` (the group's `gpu`, the profile.db key) and a
//! `model_config`: a file in `model/config/` without the extension, so
//! `"llama3_8b"` is `model/config/llama3_8b.json`. The other names are the
//! arch's own params.
//!
//! A row supports every combination of its listed values; a variant supports
//! the union of its rows. Params a row does not name are free. Params that must
//! move together (FFN TP a multiple of attention TP) or a model with a smaller
//! set than another get their own rows.
//!
//! `#[derive(ProviderSchema)]` emits the rows as a `SUPPORTED` const beside
//! `SCHEMA`; `list-params` publishes them under each arch tag as `"supported"`,
//! the launcher rejects a config no row covers, and `arch::build` tests build
//! every combination structurally, so a row cannot claim a deployment the arch
//! cannot build. An arch with no rows is not checked.

use serde::ser::{SerializeMap, Serializer};
use serde::Serialize;

/// The values one param may take inside a [`SupportedRow`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SupportedValues {
    Int(&'static [i64]),
    Str(&'static [&'static str]),
}

impl Serialize for SupportedValues {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        match *self {
            SupportedValues::Int(v) => v.serialize(ser),
            SupportedValues::Str(v) => v.serialize(ser),
        }
    }
}

/// One `#[supported(...)]` attribute: `(param, allowed values)` pairs, in the
/// order written. Serializes as `{param: [values...]}`.
#[derive(Clone, Copy, Debug)]
pub struct SupportedRow(pub &'static [(&'static str, SupportedValues)]);

impl Serialize for SupportedRow {
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        let mut m = ser.serialize_map(Some(self.0.len()))?;
        for (name, values) in self.0 {
            m.serialize_entry(name, values)?;
        }
        m.end()
    }
}

impl SupportedRow {
    /// Every combination of the row's values as `(param, JSON value)` pairs.
    pub fn combinations(&self) -> Vec<Vec<(&'static str, serde_json::Value)>> {
        let mut out = vec![Vec::new()];
        for (name, values) in self.0 {
            let options: Vec<serde_json::Value> = match values {
                SupportedValues::Int(v) => v.iter().map(|x| (*x).into()).collect(),
                SupportedValues::Str(v) => v.iter().map(|x| (*x).into()).collect(),
            };
            out = out
                .into_iter()
                .flat_map(|prefix| {
                    options.iter().map(move |option| {
                        let mut next = prefix.clone();
                        next.push((*name, option.clone()));
                        next
                    })
                })
                .collect();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROW: SupportedRow = SupportedRow(&[
        ("model_config", SupportedValues::Str(&["a.json", "b.json"])),
        ("tp_size", SupportedValues::Int(&[1, 2])),
    ]);

    #[test]
    fn serializes_as_a_param_to_values_map() {
        assert_eq!(
            serde_json::to_value(ROW).unwrap(),
            serde_json::json!({"model_config": ["a.json", "b.json"], "tp_size": [1, 2]})
        );
    }

    #[test]
    fn combinations_are_the_cartesian_product_in_row_order() {
        let combos = ROW.combinations();
        assert_eq!(combos.len(), 4);
        assert_eq!(
            combos[1],
            vec![("model_config", "a.json".into()), ("tp_size", 2.into())]
        );
    }
}
