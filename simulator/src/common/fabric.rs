//! Network fabric kinds used by L1 comm runners + L5 transport layer.
//!
//! Variant names follow Rust UpperCamelCase; the serde wire form is fixed via
//! per-variant `rename` so acronym-ish variants land on the conventional
//! lowercase tokens (`nvlink`, not the snake_case `nv_link`).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Fabric {
    #[serde(rename = "nvlink")]
    Nvlink,
    #[serde(rename = "infinity_fabric")]
    InfinityFabric,
    #[serde(rename = "pcie")]
    Pcie,
    #[serde(rename = "infiniband")]
    Infiniband,
    #[serde(rename = "roce")]
    Roce,
    #[serde(rename = "ethernet")]
    Ethernet,
}
