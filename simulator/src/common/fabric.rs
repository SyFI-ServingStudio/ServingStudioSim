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
    /// AWS NeuronLink: the on-chip and chip-to-chip links of a Trainium
    /// instance (the four LNC2 cores of one Trainium2 chip at TP4).
    #[serde(rename = "neuron_link")]
    NeuronLink,
}

impl Fabric {
    /// The serde wire token, for emitting into an L1 `ArgsPayload` field (the
    /// comm-kernel cache key) or any other string sink. Kept in lockstep with
    /// the per-variant `#[serde(rename = ...)]` above.
    pub fn as_str(self) -> &'static str {
        match self {
            Fabric::Nvlink => "nvlink",
            Fabric::InfinityFabric => "infinity_fabric",
            Fabric::Pcie => "pcie",
            Fabric::Infiniband => "infiniband",
            Fabric::Roce => "roce",
            Fabric::Ethernet => "ethernet",
            Fabric::NeuronLink => "neuron_link",
        }
    }
}
