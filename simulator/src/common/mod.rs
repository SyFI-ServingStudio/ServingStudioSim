//! Cross-cutting types referenced by every simulator layer.
//!
//! See `docs/file_structure.md` (top-level `common/` block).

pub mod fabric;
pub mod id;
pub mod request;
pub mod time;

pub use fabric::Fabric;
pub use id::{BatchId, ExpertId, GroupId, IdMap, PoolId, RequestId, WorkerId};
pub use request::{Request, RequestRecord, RequestStore, SharedRequests};
pub use time::Time;
