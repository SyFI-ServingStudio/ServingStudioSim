//! Reusable pieces the KV store is built from — algorithms, not orchestration.
//!
//! What lives here is only what has real algorithmic content and no
//! store-specific behavior:
//!
//! - [`kv_accounting`] — the arch's `KvCacheLayout` resolved into the units
//!   every charge is counted in: what a request holds, needs to be admitted
//!   and leaves behind for prefix reuse.
//! - [`kv_cache_report`] — one resolved pool restated as numbers for
//!   `raw/kv_cache.json` and `simulator kv-cache`.
//! - [`resident_partition`] — one partition's resident set and its
//!   peak-occupancy projection.
//! - [`request_ledger`] — the request-keyed side tables (placement, promised,
//!   held, resolved prefill context) and their per-partition sums.
//! - [`prefix_cache`] — the evictable retained-session tier and its four
//!   replacement policies.
//! - [`prefix_cache_journal`] — turning cache mutations into logged events.

pub(crate) mod kv_accounting;
pub(crate) mod kv_cache_report;
pub(crate) mod prefix_cache;
pub(crate) mod prefix_cache_journal;
pub(crate) mod request_ledger;
pub(crate) mod resident_partition;
