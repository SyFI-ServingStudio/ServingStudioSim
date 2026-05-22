//! Newtyped identifiers used across simulator layers.
//!
//! Each id wraps a fixed-width integer chosen to match the matching parquet
//! column width in `docs/logging.md` (e.g. `request_id : u32`, `worker_id : u16`).
//! Newtyping prevents accidentally passing a worker id where a request id is
//! expected.

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

define_id!(RequestId, u32);
define_id!(WorkerId, u16);
define_id!(PoolId, u16);
define_id!(BatchId, u64);
define_id!(GroupId, u8);
define_id!(ExpertId, u16);
