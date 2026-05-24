//! `request` category — per-request analyses (session = rollup), read from the
//! fixed-schema `request_slo` / `request_state` parquet. Tier-1, deployment-
//! agnostic. `slo` is the first subject; length/queueing metrics on the same
//! records would be siblings here.
pub mod slo;
