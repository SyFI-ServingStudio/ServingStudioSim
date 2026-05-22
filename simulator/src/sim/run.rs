//! L7-β tick driver — the single deployment-independent sim loop. Drains
//! arrivals, drives the L6 `Flow`, emits per-request parquet rows, and applies
//! termination policy. One process, one thread, one clock.
//!
//! Deviation from L7 design §2.4: the `Flow` trait here takes `on_arrival(Request)`
//! + `tick(now) -> Vec<OrchAction>` against a `SharedRequests` injected at
//! construction (signed off in the L5/L6 batch), rather than threading
//! `&mut RequestStore` per tick. The driver holds an `Rc::clone` of that store to
//! read lifecycle state back for logging.

use std::time::Instant;

use crate::common::{RequestId, RequestRecord, SharedRequests, Time};
use crate::log::{LoggerSession, RequestSloEntry, RequestStateEntry};
use crate::orchestrator::{Flow, OrchAction};

/// Sim-time between heartbeat log lines (matches ref/moesim-rs's 1 s).
const HEARTBEAT_INTERVAL_MS: f64 = 1000.0;

/// Sim-time between stuck-watchdog samples. The watchdog's `progress_signature`
/// scan is O(requests) and Stuck only fires during the post-exhaustion drain
/// tail, so sampling at this cadence keeps the hot loop scan-free without
/// meaningfully delaying stall detection. Ref/moesim-rs samples every 1 s.
const WATCHDOG_SAMPLE_MS: f64 = 1000.0;

/// Iteration-count gate shared by every periodic step in the tick loop
/// (`request_state` snapshot, heartbeat, stuck watchdog). `fire()` returns true
/// on the first call and then on every `period`-th call thereafter, so a hot
/// loop runs an expensive step once every few iterations instead of every pass.
/// Firing on the first call gives the dense snapshot a t=0 baseline row.
struct EveryN {
    period: u64,
    count: u64,
}

impl EveryN {
    fn new(period: u64) -> Self {
        debug_assert!(period > 0, "EveryN period must be > 0");
        Self { period, count: 0 }
    }

    fn fire(&mut self) -> bool {
        let fires = self.count % self.period == 0;
        self.count += 1;
        fires
    }
}

/// Why the run loop stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminationCause {
    /// Every request completed (and, under `run_to_end`, the trace was drained).
    DrainComplete,
    /// `--duration-ms` reached with work still outstanding (default, no run_to_end).
    DurationReached,
    /// Trace exhausted but in-flight work made no progress for `stuck_threshold`.
    Stuck,
}

/// Tick-loop knobs. Defaults mirror ref/moesim-rs's unified loop: `tick_dt` 1 ms,
/// `snapshot_dt` (the `request_state` cadence) 100 s, `stuck_threshold` 60 s of
/// no progress. (Ref's tick_dt is 1 µs — see note in `new`.)
#[derive(Clone, Copy, Debug)]
pub struct TickCfg {
    pub tick_dt: Time,
    pub duration: Time,
    pub run_to_end: bool,
    pub snapshot_dt: Time,
    pub stuck_threshold: Time,
}

impl TickCfg {
    pub fn new(duration_ms: f64, run_to_end: bool) -> Self {
        // Ref/moesim-rs's unified loop ticks at 1 µs; we tick at 10 µs — 10x
        // fewer iterations than ref, with 10 µs quantization on TTFT/TPOT
        // (negligible against typical 10-50 ms TPOT).
        let tick_dt = Time::from_us(10);
        Self {
            tick_dt,
            duration: Time::from_ms(duration_ms),
            run_to_end,
            snapshot_dt: Time::from_ms(100_000.0),
            stuck_threshold: Time::from_ms(60_000.0),
        }
    }
}

/// Run the simulation to termination. `store` must be the same `SharedRequests`
/// injected into `flow` at construction (the flow inserts arrivals there; the
/// driver reads them back to log lifecycle state).
pub fn run_sim(
    flow: &mut dyn Flow,
    store: &SharedRequests,
    frontend: &mut super::frontend::TraceFrontend,
    logger: &mut LoggerSession,
    cfg: &TickCfg,
) -> anyhow::Result<TerminationCause> {
    let wall_start = Instant::now();
    let mut clock = Time::ZERO;
    let mut prev_progress = 0u64;
    let mut idle_for = Time::ZERO;

    // Every periodic step shares one iteration-count gate. With a fixed
    // `tick_dt`, "every K ticks" and "every K·tick_dt of sim-time" are the same,
    // so the sim-time intervals (snapshot/heartbeat) become tick periods here.
    let ticks = |dt: Time| (dt.as_ns() / cfg.tick_dt.as_ns()).max(1);
    let sample_dt = Time::from_ms(WATCHDOG_SAMPLE_MS);
    let mut snapshot = EveryN::new(ticks(cfg.snapshot_dt));
    let mut heartbeat = EveryN::new(ticks(Time::from_ms(HEARTBEAT_INTERVAL_MS)));
    let mut watchdog = EveryN::new(ticks(sample_dt));

    let cause = loop {
        // 1. Drain arrivals due by now → Flow inserts into the shared store.
        //    The frontend owns the drain loop, so the whole tick is emptied here.
        frontend.drain_due(clock, |req| flow.on_arrival(req));

        // 2. Advance one tick; each completion writes its terminal `request_slo`
        //    row. `request_state` is a periodic table (2b), not written here.
        for action in flow.tick(clock) {
            let OrchAction::Complete { req } = action;
            let s = store.borrow();
            logger.record_request_slo(slo_entry(req, clock, &s[req]))?;
        }

        // 2b. Periodic `request_state` snapshot — DENSE: every touched request is
        //     logged each interval. In-flight reqs advance; completed reqs keep
        //     re-emitting their frozen terminal values. This makes per-segment
        //     workload a plain diff of consecutive snapshots' column sums (no
        //     fill-forward over dropped-out requests needed).
        if snapshot.fire() {
            let s = store.borrow();
            for (id, rec) in s.iter() {
                logger.record_request_state(state_entry(id, clock, rec))?;
            }
        }

        // 2c. Periodic heartbeat (skip the stat scan unless INFO is enabled).
        //     `fire()` runs first so the gate advances every tick regardless.
        if heartbeat.fire() && tracing::enabled!(tracing::Level::INFO) {
            let s = store.borrow();
            let completed = s.iter().filter(|(_, r)| r.completed).count();
            tracing::info!(
                "t={:.0}ms: completed={} in_flight={}",
                clock.as_ms(),
                completed,
                s.in_flight(),
            );
        }

        // 3. Termination — drain-complete and duration are cheap (an in-flight
        //    counter + a trace cursor), so check them on every tick.
        let (in_flight, exhausted) = {
            let s = store.borrow();
            (s.in_flight(), frontend.exhausted())
        };
        if exhausted && in_flight == 0 {
            break TerminationCause::DrainComplete;
        }
        if !cfg.run_to_end && clock >= cfg.duration {
            break TerminationCause::DurationReached;
        }

        // 3b. Stuck watchdog. `progress_signature` is the one O(requests) scan in
        //     the loop and Stuck only matters once the trace is drained, so we
        //     sample it every `WATCHDOG_SAMPLE_TICKS` ticks during the drain tail
        //     rather than on every clock advance.
        if exhausted && in_flight > 0 && watchdog.fire() {
            let progress = progress_signature(&store.borrow());
            if progress > prev_progress {
                prev_progress = progress;
                idle_for = Time::ZERO;
            } else {
                idle_for = idle_for + sample_dt;
                if idle_for >= cfg.stuck_threshold {
                    break TerminationCause::Stuck;
                }
            }
        }

        // 4. Advance clock.
        clock = clock + cfg.tick_dt;
    };

    finalize(store, logger, clock)?;
    logger.flush_all()?;

    // End-of-run summary (multi-line): finished count, then prefill / decode /
    // total token counts each with their own throughput, then sim + wall time.
    let wall_s = wall_start.elapsed().as_secs_f64();
    let safe_wall = wall_s.max(1e-9);
    {
        let s = store.borrow();
        let total = s.iter().count();
        let completed = s.iter().filter(|(_, r)| r.completed).count();
        let prefill_tok: u64 = s.iter().map(|(_, r)| r.prefill_processed as u64).sum();
        let decode_tok: u64 = s.iter().map(|(_, r)| r.tokens_emitted as u64).sum();
        let all_tok = prefill_tok + decode_tok;
        tracing::info!("sim complete: cause={:?}", cause);
        tracing::info!("  requests: finished {}/{}", completed, total);
        tracing::info!(
            "  prefill:  {} tokens ({:.0} tok/s)",
            prefill_tok,
            prefill_tok as f64 / safe_wall
        );
        tracing::info!(
            "  decode:   {} tokens ({:.0} tok/s)",
            decode_tok,
            decode_tok as f64 / safe_wall
        );
        tracing::info!(
            "  total:    {} tokens ({:.0} tok/s)",
            all_tok,
            all_tok as f64 / safe_wall
        );
        // Speedup = simulated wall-clock collapsed into real wall-clock: how many
        // seconds of modeled time we cover per second of compute.
        tracing::info!(
            "  time:     sim {:.1}ms, wall {:.2}s ({:.0} req/s, {:.1}x real-time)",
            clock.as_ms(),
            wall_s,
            completed as f64 / safe_wall,
            (clock.as_ms() / 1000.0) / safe_wall,
        );
    }
    Ok(cause)
}

/// Sim-end flush: any request still in-flight gets a final (incomplete) state
/// row + a partial slo row. Completed requests were already flushed at their
/// completion tick.
fn finalize(store: &SharedRequests, logger: &mut LoggerSession, clock: Time) -> anyhow::Result<()> {
    let s = store.borrow();
    for (id, rec) in s.iter() {
        if !rec.completed {
            logger.record_request_state(state_entry(id, clock, rec))?;
            logger.record_request_slo(slo_entry(id, clock, rec))?;
        }
    }
    Ok(())
}

/// Watchdog progress metric: Σ over all requests of (prefill tokens processed +
/// decode tokens emitted). Mirrors the ref's progress signature in counting
/// *both* phases — a request grinding through a long prefill (no decode yet)
/// still advances this, so it isn't falsely flagged `Stuck`. Summed over all
/// requests (not just in-flight) so it stays monotonic for the strict
/// `progress > prev_progress` check: a completing request keeps its token
/// counts in the total instead of dropping out and lowering the sum.
fn progress_signature(store: &crate::common::RequestStore) -> u64 {
    store
        .iter()
        .map(|(_, r)| r.prefill_processed as u64 + r.tokens_emitted as u64)
        .sum()
}

fn final_phase(rec: &RequestRecord) -> &'static str {
    if rec.completed {
        "complete"
    } else if rec.first_token_time.is_some() {
        "decode"
    } else if rec.prefill_processed > 0 {
        "prefill"
    } else {
        "queued"
    }
}

fn state_entry(id: RequestId, now: Time, rec: &RequestRecord) -> RequestStateEntry {
    let arrival_ms = rec.arrival_time.as_ms();
    RequestStateEntry {
        request_id: id.0,
        logging_time_ms: now.as_ms(),
        arrival_time_ms: arrival_ms,
        first_token_time_ms: rec.first_token_time.map(|t| t.as_ms()),
        completion_time_ms: if rec.completed {
            rec.last_token_time.map(|t| t.as_ms())
        } else {
            None
        },
        completed: rec.completed,
        input_len: rec.prompt_len,
        output_len: rec.decode_len,
        completed_input_len: rec.prefill_processed,
        completed_output_len: rec.tokens_emitted,
        final_phase: final_phase(rec).to_string(),
        // single-round defaults
        session_id: id.0,
        round_idx: 0,
        total_rounds: 1,
        tool_wait_after_ms: 0.0,
        session_arrival_time_ms: arrival_ms,
        preserved_prefix_kv: 0,
    }
}

fn slo_entry(id: RequestId, now: Time, rec: &RequestRecord) -> RequestSloEntry {
    let times_ms: Vec<f32> = rec
        .output_token_times
        .iter()
        .map(|t| t.as_ms() as f32)
        .collect();
    let ttft_ms = rec
        .first_token_time
        .map(|t| (t.as_ms() - rec.arrival_time.as_ms()) as f32);
    let (tpot_mean, tpot_p50, tpot_p99, tpot_max) = tpot_stats(&rec.output_token_times);
    RequestSloEntry {
        request_id: id.0,
        logging_time_ms: now.as_ms(),
        completed: rec.completed,
        arrival_time_ms: rec.arrival_time.as_ms(),
        output_token_times_ms: times_ms,
        ttft_ms,
        tpot_mean_ms: tpot_mean,
        tpot_p50_ms: tpot_p50,
        tpot_p99_ms: tpot_p99,
        tpot_max_ms: tpot_max,
    }
}

/// Inter-token gaps (ms) → (mean, p50, p99, max). `None` for all when there are
/// fewer than two output tokens (no gap defined).
fn tpot_stats(times: &[Time]) -> (Option<f32>, Option<f32>, Option<f32>, Option<f32>) {
    if times.len() < 2 {
        return (None, None, None, None);
    }
    let mut gaps: Vec<f32> = times
        .windows(2)
        .map(|w| (w[1].as_ms() - w[0].as_ms()) as f32)
        .collect();
    let mean = gaps.iter().sum::<f32>() / gaps.len() as f32;
    gaps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| -> f32 {
        let idx = ((p * (gaps.len() - 1) as f64).round() as usize).min(gaps.len() - 1);
        gaps[idx]
    };
    (Some(mean), Some(pct(0.5)), Some(pct(0.99)), Some(*gaps.last().unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arch::contract::{IterwiseUnifiedModel, UnifiedArchInput};
    use crate::common::{PoolId, RequestStore};
    use crate::orchestrator::{
        DpPlacementPolicy, SimpleDpConfig, SimpleDpFlow, SimpleDpPoolConfig, UnifiedWorkerFactory,
    };
    use crate::sim::frontend::TraceFrontend;
    use crate::timing::LookupResult;
    use crate::worker::WorkerConfig;
    use std::cell::RefCell;
    use std::io::Write;
    use std::path::PathBuf;
    use std::rc::Rc;
    use std::sync::Arc;

    struct FakeModel {
        ms: f64,
    }
    impl IterwiseUnifiedModel for FakeModel {
        fn cost_whole_iter(&self, _b: &UnifiedArchInput) -> LookupResult {
            LookupResult::leaf("fake", Time::from_ms(self.ms), 0, 0, 0.0, Vec::new())
        }
        fn kv_bytes_per_token(&self) -> u64 {
            1
        }
    }

    fn write_trace(dir: &std::path::Path, n: u32) -> PathBuf {
        let path = dir.join("trace.csv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "id,input_len,output_len,arrival_time").unwrap();
        for id in 0..n {
            writeln!(f, "{id},8,3,0.0").unwrap();
        }
        path
    }

    #[test]
    fn every_n_fires_on_first_then_each_nth_call() {
        let mut e = EveryN::new(3);
        let fires: Vec<bool> = (0..7).map(|_| e.fire()).collect();
        // Fires on call 0 (t=0 baseline), then every 3rd call.
        assert_eq!(fires, vec![true, false, false, true, false, false, true]);
    }

    #[test]
    fn end_to_end_completes_and_writes_parquet() {
        let dir = tempfile::tempdir().unwrap();
        let trace = write_trace(dir.path(), 4);

        let store: SharedRequests = Rc::new(RefCell::new(RequestStore::new()));
        let factory = UnifiedWorkerFactory::new(
            Arc::new(FakeModel { ms: 1.0 }),
            Rc::clone(&store),
            WorkerConfig::default(),
        );
        let cfg = SimpleDpConfig {
            dp_pool: SimpleDpPoolConfig {
                pool: PoolId(0),
                num_workers: 2,
                placement: DpPlacementPolicy::RoundRobin,
            },
        };
        let mut flow = SimpleDpFlow::new(cfg, factory);
        let mut frontend = TraceFrontend::load(&[trace], 1.0).unwrap();
        let mut logger = LoggerSession::open(dir.path()).unwrap();

        let cause = run_sim(
            &mut flow,
            &store,
            &mut frontend,
            &mut logger,
            &TickCfg::new(5000.0, true),
        )
        .unwrap();
        assert_eq!(cause, TerminationCause::DrainComplete);

        // Every request finished: 3 output tokens each.
        let s = store.borrow();
        for id in 0..4u32 {
            let r = &s[RequestId(id)];
            assert!(r.completed, "req {id} should complete");
            assert_eq!(r.tokens_emitted, 3);
            assert_eq!(r.output_token_times.len(), 3);
        }
        drop(s);

        // Both per-request parquet files exist with rows.
        let raw = dir.path().join("raw");
        let slo = raw.join("request_slo.parquet");
        let state = raw.join("request_state.parquet");
        assert!(slo.exists(), "request_slo.parquet missing");
        assert!(state.exists(), "request_state.parquet missing");
        assert_eq!(parquet_rows(&slo), 4, "one slo row per completed request");
    }

    fn parquet_rows(path: &std::path::Path) -> usize {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(path).unwrap())
            .unwrap()
            .build()
            .unwrap()
            .map(|b| b.unwrap().num_rows())
            .sum()
    }
}
