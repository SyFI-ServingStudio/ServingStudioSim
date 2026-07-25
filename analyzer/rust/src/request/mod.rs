//! `request` category — per-request analyses (session = rollup), read from the
//! fixed-schema `request_slo` / `request_state` parquet. Tier-1, deployment-
//! agnostic. `slo` and `slo_goodput` share the request distribution shape;
//! length/queueing metrics on the same records would be siblings here.
pub mod slo;
pub mod slo_goodput;
