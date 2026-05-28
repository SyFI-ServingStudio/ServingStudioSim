//! `throughput` category — system serving rate over time/iter windows. Grain is
//! "the system over a time segment" (not per-request like `request`), and the
//! source is the periodic `request_state` snapshot: the sim writes one aggregate
//! row per tick (`prefill_tokens_cum` / `decode_tokens_cum` summed across the
//! admitted set, monotonic non-decreasing), so per-segment workload is a plain
//! diff of consecutive snapshots' columns. Per-GPU normalization reads
//! `num_gpus` from `run_meta.json`.

pub mod segment;
