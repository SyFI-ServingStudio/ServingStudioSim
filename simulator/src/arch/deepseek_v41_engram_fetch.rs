//! How DeepSeek-V4.1-Flash's Engram rows reach the GPU (a deployment choice).
//!
//! - `uva` (vLLM's default, `EngramConfig.cpu_offload`): one Triton lookup
//!   per Engram layer reads the pinned host table over UVA on the GPU's SMs
//!   (the profiled `engram_lookup` at `residency = host_uva`). Under
//!   `engine: deepinfer` this is an approximation: DeepInfer reads the tables
//!   from host memory over GPU-initiated RDMA (DeepEP `EngramBuffer`, NCCL
//!   GIN), which is not modeled, and vLLM's host-UVA lookup stands in.
//! - `off`: the fetch is priced out (the Engram blocks' compute stays).
//! - `none`: Engram fully off, DeepInfer's `DEEPINFER_ENGRAM_SANITY_OFF=1`
//!   (no fetch, no `wkv` GEMM, no gate).

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EngramFetch {
    #[default]
    Uva,
    /// The fetch priced out (the Engram blocks' compute stays).
    Off,
    /// No fetch and no Engram consumer (`wkv` + gate): DeepInfer's
    /// `DEEPINFER_ENGRAM_SANITY_OFF=1`.
    None,
}

impl EngramFetch {
    /// Whether any Engram rows are fetched (`off` and `none` price none).
    pub fn fetches(self) -> bool {
        !matches!(self, Self::Off | Self::None)
    }
}

/// The `engram_fetch` spellings: `uva`, `off`, `none`, or a bool (`true` =
/// uva, `false` = off) as the DeepInfer engine option spells it.
impl<'de> serde::Deserialize<'de> for EngramFetch {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum Raw {
            Bool(bool),
            Name(String),
        }
        match Raw::deserialize(d)? {
            Raw::Bool(true) => Ok(Self::Uva),
            Raw::Bool(false) => Ok(Self::Off),
            Raw::Name(name) => match name.as_str() {
                "uva" | "true" => Ok(Self::Uva),
                "off" | "false" => Ok(Self::Off),
                "none" => Ok(Self::None),
                other => Err(serde::de::Error::custom(format!(
                    "engram_fetch: expected uva, off or none, got {other}"
                ))),
            },
        }
    }
}

/// The `engram_fetch` choices the launcher offers.
pub const ENGRAM_FETCH_CHOICES: [&str; 3] = ["uva", "off", "none"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_is_its_own_spelling_and_fetches_nothing() {
        let none: EngramFetch = serde_json::from_str("\"none\"").unwrap();
        assert_eq!(none, EngramFetch::None);
        assert!(!none.fetches() && !EngramFetch::Off.fetches());
        assert!(EngramFetch::Uva.fetches());
        assert_eq!(
            serde_json::from_str::<EngramFetch>("false").unwrap(),
            EngramFetch::Off
        );
        assert!(serde_json::from_str::<EngramFetch>("\"cpu_threads\"").is_err());
    }
}
