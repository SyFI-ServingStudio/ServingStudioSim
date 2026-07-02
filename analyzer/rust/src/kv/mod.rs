//! `kv` category — KV-cache pool occupancy over time. Grain: per-worker-pool
//! (aggregated across a pool's DP shards) time-series; source: the `kv_snapshot`
//! stream (a distinct source family from `cost_log`, hence its own category).
//! Tier-1, deployment-agnostic — any deployment that logs KV occupancy applies;
//! one that doesn't degrades to `unavailable`.
pub mod occupancy;
