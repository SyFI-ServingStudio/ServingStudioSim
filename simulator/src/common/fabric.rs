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
    /// PCIe without GPU peer-to-peer between the ranks (NCCL falls back to
    /// host shared memory), e.g. the Massed Compute RTX PRO 6000 VMs.
    #[serde(rename = "pcie")]
    Pcie,
    /// PCIe with GPU peer-to-peer (`nvidia-smi topo -p2p r` OK on every pair),
    /// e.g. RunPod's 4x RTX PRO 6000 hosts. Kept apart from `pcie` because
    /// NCCL's transport, and so every collective row, differs.
    #[serde(rename = "pcie_p2p")]
    PcieP2p,
    #[serde(rename = "infiniband")]
    Infiniband,
    #[serde(rename = "roce")]
    Roce,
    #[serde(rename = "ethernet")]
    Ethernet,
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
            Fabric::PcieP2p => "pcie_p2p",
            Fabric::Infiniband => "infiniband",
            Fabric::Roce => "roce",
            Fabric::Ethernet => "ethernet",
        }
    }
}
