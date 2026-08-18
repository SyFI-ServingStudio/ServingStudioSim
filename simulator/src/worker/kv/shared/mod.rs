//! Reusable pieces a KV store is built from — algorithms, not orchestration.
//!
//! Each concrete store (`full_attn`, `hybrid_gdn`) writes its own `KvStore` /
//! `PrefixKv` implementation and wires these together itself. That is
//! deliberate: the stores differ in ways a shared implementation would have to
//! branch on (a hybrid model charges a fixed recurrent state per request, and
//! its prefix hits quantize to a checkpoint interval), and pushing those
//! branches into one god-object is how a "shared core" turns back into the
//! tangle this split exists to prevent. What lives here is only what has real
//! algorithmic content and no store-specific behavior:
//!
//! - [`resident_partition`] — one partition's resident set and its
//!   peak-occupancy projection.
//! - [`request_ledger`] — the request-keyed side tables (placement, promised,
//!   held, resolved prefill context) and their per-partition sums.
//! - [`prefix_cache`] — the evictable retained-session tier and its four
//!   replacement policies.
//! - [`prefix_cache_journal`] — turning cache mutations into logged events.
//!
//! The store-specific knobs each piece takes are constructor parameters, so a
//! full-attention store passes the identity values (no fixed charge, quantum of
//! one token) and gets exactly its pre-split behavior back.

pub(crate) mod prefix_cache;
pub(crate) mod prefix_cache_journal;
pub(crate) mod request_ledger;
pub(crate) mod resident_partition;
