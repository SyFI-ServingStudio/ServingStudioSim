//! `batch` category — per-batch (per-iteration) composition and per-slot achieved
//! kernel throughput. Tier-1, deployment-agnostic; both sourced from `cost_log`
//! (`composition` from the per-iteration `groups`, `kernel_throughput` from the
//! per-slot `slot_flops` / `slot_bytes` / `slot_time_ms`).
pub mod composition;
pub mod kernel_throughput;
