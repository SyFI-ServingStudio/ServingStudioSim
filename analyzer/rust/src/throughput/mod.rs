//! `throughput` category — system serving rate over time/iter windows. Grain is
//! "the system over a time segment" (not per-request like `request`), and the
//! source is the periodic `request_state` snapshot: the sim logs each admitted
//! request's cumulative `completed_input_len` / `completed_output_len` every tick,
//! so per-segment workload is a plain diff of consecutive snapshots' column sums
//! (sim `run.rs` designed the snapshot for exactly this). Per-GPU normalization
//! reads `num_gpus` from `run_meta.json`.

pub mod segment;
