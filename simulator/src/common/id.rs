//! Newtyped identifiers used across simulator layers.
//!
//! Each id wraps a fixed-width integer chosen to match the matching parquet
//! column width in `simulator/src/log/README.md` (e.g. `request_id : u32`, `worker_id : u16`).
//! Newtyping prevents accidentally passing a worker id where a request id is
//! expected.

use std::hash::{BuildHasherDefault, Hasher};

use serde::{Deserialize, Serialize};

macro_rules! define_id {
    ($name:ident, $repr:ty) => {
        #[derive(
            Clone,
            Copy,
            Debug,
            Default,
            PartialEq,
            Eq,
            PartialOrd,
            Ord,
            Hash,
            Serialize,
            Deserialize,
        )]
        #[serde(transparent)]
        pub struct $name(pub $repr);

        impl $name {
            pub const fn new(v: $repr) -> Self {
                Self(v)
            }

            pub const fn raw(self) -> $repr {
                self.0
            }
        }

        impl From<$repr> for $name {
            fn from(v: $repr) -> Self {
                Self(v)
            }
        }
    };
}

/// A fast, deterministic hasher for the small-integer newtype ids. std `HashMap`
/// defaults to SipHash (DoS-resistant, ~ns per hash) — overkill for a dense
/// `u32`/`u16` id looked up on the per-token sim hot path (e.g. `Batch`'s decode
/// index, `promised` / `request_to_slot`). This is a single Fibonacci multiply,
/// std-only (no external crate). It is also *more* deterministic than the default
/// (whose `RandomState` seed is per-map random), which only helps the sim's
/// run-to-run reproducibility. Only ever used for by-id `get`/`insert`/`remove` —
/// none of these maps are iterated for output — so hash order is irrelevant.
#[derive(Default)]
pub struct IdHasher(u64);

impl Hasher for IdHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.write_u64(i as u64);
    }
    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.write_u64(i as u64);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = i.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        // Fallback keeping the impl total; the id keys use the integer paths above.
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
}

/// A `HashMap` keyed by a small-integer id, hashed with [`IdHasher`].
pub type IdMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<IdHasher>>;

define_id!(RequestId, u32);
define_id!(WorkerId, u16);
define_id!(PoolId, u16);
define_id!(BatchId, u64);
define_id!(GroupId, u8);
define_id!(ExpertId, u16);
