//! `backend` category — how best-of-N backend selection distributes over each
//! kernel position's input feature space. Grain = one cost-tree leaf position's
//! selected backend across its inputs; source = the `cost_log` `slot_input` +
//! `slot_backend` list columns joined to the `cost_manifest` candidate lists.
//!
//! One subject today: [`kernel_input_distribution`], which projects each
//! position's sampled inputs to 2-D (raw axes or PCA) and colors every point by
//! the backend that won there — so a multi-backend position's selection boundary
//! is visible.

pub mod kernel_input_distribution;
