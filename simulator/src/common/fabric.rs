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
    /// Neuron collective group with one rank per Trainium2 chip (chips strided).
    #[serde(rename = "neuron_c1")]
    NeuronC1,
    /// Neuron collective group with two ranks (logical cores) per chip.
    #[serde(rename = "neuron_c2")]
    NeuronC2,
    /// Neuron collective group spanning all four logical cores of each chip.
    #[serde(rename = "neuron_c4")]
    NeuronC4,
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
            Fabric::NeuronC1 => "neuron_c1",
            Fabric::NeuronC2 => "neuron_c2",
            Fabric::NeuronC4 => "neuron_c4",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Fabric;

    /// Catches `as_str` drifting from the serde wire name (the comm cache key).
    #[test]
    fn as_str_matches_serde_name() {
        for fabric in [
            Fabric::Nvlink,
            Fabric::InfinityFabric,
            Fabric::Pcie,
            Fabric::Infiniband,
            Fabric::Roce,
            Fabric::Ethernet,
            Fabric::NeuronC1,
            Fabric::NeuronC2,
            Fabric::NeuronC4,
        ] {
            assert_eq!(
                serde_json::to_value(fabric).unwrap(),
                serde_json::Value::from(fabric.as_str())
            );
        }
    }
}
