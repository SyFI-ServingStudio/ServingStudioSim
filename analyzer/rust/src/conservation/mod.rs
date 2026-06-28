//! `conservation` category — run-wide work-accounting invariants. Cross-checks
//! the work the cost model was actually asked to do (summed from `cost_log`'s
//! per-iteration `groups`) against the work each request *should* incur (closed
//! forms over the per-request `(p, d)` terminal state in `request_slo`). The
//! grain is the whole run (one report of pass/fail checks), and the source spans
//! two tables — distinct from the per-batch `batch` category (same `cost_log`
//! source, but per-iteration scatter) and the per-request `request` category.
//! Catches redundant / dropped prefill·decode·FFN work and KV miscounting.
pub mod workload;
