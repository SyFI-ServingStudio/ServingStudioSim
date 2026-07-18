//! L7-β tick driver — the single deployment-independent sim loop. Drains
//! arrivals, drives the L6 `Flow`, emits request SLO rows plus aggregate
//! request-state snapshots, and applies termination policy. One sim thread owns
//! modeled state and clock; logger/cost writer threads sit outside that modeled
//! loop.
//!
//! Deviation from L7 design §2.4: the `Flow` trait here takes `on_arrival(Request)`
//! + `tick(now) -> Vec<OrchAction>` against a `SharedRequests` injected at
//! construction (signed off in the L5/L6 batch), rather than threading
//! `&mut RequestStore` per tick. The driver holds an `Rc::clone` of that store to
//! read lifecycle state back for logging.

use std::time::Instant;

use serde::Serialize;

use crate::common::{RequestId, RequestRecord, RequestStore, SharedRequests, Time};
use crate::log::{LoggerSession, RequestSloEntry, RequestStateEntry};
use crate::orchestrator::{Flow, OrchAction};

/// Sim-time between heartbeat log lines (matches ref/moesim-rs's 1 s).
const HEARTBEAT_INTERVAL_MS: f64 = 1000.0;

/// Sim-time between stuck-watchdog samples. The watchdog is an O(1) liveness
/// check (did `completed` or the admitted-id watermark advance since the last
/// sample), so the cadence only bounds how fast a true stall is detected, not
/// hot-loop cost. With `stuck_threshold` (60 s) below this interval, one sample
/// window with no progress flags the deadlock.
const WATCHDOG_SAMPLE_MS: f64 = 100_000.0;

/// Iteration-count gate shared by every periodic step in the tick loop
/// (`request_state` snapshot, heartbeat, stuck watchdog). `fire()` returns true
/// on the first call and then on every `period`-th call thereafter, so a hot
/// loop runs an expensive step once every few iterations instead of every pass.
/// Firing on the first call gives the dense snapshot a t=0 baseline row.
struct EveryN {
    period: u64,
    /// Ticks remaining until the next fire. Starts at 0 so the first call fires
    /// (the t=0 baseline). A countdown decrement avoids the per-tick integer
    /// `div` that `count % period` compiled to — at 100 µs ticks that modulo ran
    /// hundreds of millions of times and dominated the loop's self-time.
    countdown: u64,
}

impl EveryN {
    fn new(period: u64) -> Self {
        debug_assert!(period > 0, "EveryN period must be > 0");
        Self {
            period,
            countdown: 0,
        }
    }

    fn fire(&mut self) -> bool {
        if self.countdown == 0 {
            self.countdown = self.period - 1;
            true
        } else {
            self.countdown -= 1;
            false
        }
    }
}

/// Why the run loop stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum TerminationCause {
    /// Every request completed (and, under `run_to_end`, the trace was drained).
    DrainComplete,
    /// `--duration-ms` reached with work still outstanding (default, no run_to_end).
    DurationReached,
    /// Trace exhausted but in-flight work made no progress for `stuck_threshold`.
    Stuck,
}

/// Machine-readable end-of-run summary — the structured counterpart of the
/// end-of-run `tracing::info!` lines, serialized to `<log_dir>/summary.json` so
/// regression tests (and the future L7-β aggregator) read structured metrics
/// instead of scraping logs.
///
/// Every field except `wall_s` / `realtime_x` is **deterministic** given a fixed
/// trace + config + cost model (the modeled cluster's serving behavior in
/// *simulated* time). `wall_s` / `realtime_x` are the simulator's own speed
/// (host-load dependent) — useful for trend/bench, never for a tight assertion.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct RunSummary {
    pub cause: TerminationCause,
    pub requests_total: u64,
    pub requests_finished: u64,
    pub prefill_tokens: u64,
    pub decode_tokens: u64,
    pub total_tokens: u64,
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    pub total_tok_s: f64,
    /// Total GPUs the run modeled — the run inventory size: the sum over every
    /// worker of its `gpus_per_worker` (i.e. replicas × the per-replica
    /// parallel-dim product).
    pub num_gpus: u64,
    /// `total_tok_s / num_gpus` — the headline per-accelerator serving rate.
    pub total_tok_s_per_gpu: f64,
    pub completed_req_s: f64,
    pub sim_ms: f64,
    pub wall_s: f64,
    pub realtime_x: f64,
}

impl RunSummary {
    /// Write `<log_dir>/summary.json` (pretty-printed). Caller owns the path.
    pub fn write_json(&self, log_dir: &std::path::Path) -> anyhow::Result<()> {
        let path = log_dir.join("summary.json");
        std::fs::write(&path, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
}

/// Tick-loop knobs. `snapshot_dt` (the `request_state` cadence) is 100 s and
/// `stuck_threshold` (no-progress budget) is 60 s; `tick_dt` is the caller-chosen
/// tick step (`WorkloadSpec::tick_dt_us`, default 100 µs — see `new`).
#[derive(Clone, Copy, Debug)]
pub struct TickCfg {
    pub tick_dt: Time,
    pub duration: Time,
    pub run_to_end: bool,
    pub snapshot_dt: Time,
    pub stuck_threshold: Time,
}

impl TickCfg {
    /// `tick_dt_us` is the fixed tick step in microseconds (from the workload
    /// config, default 100). Ref/moesim-rs's unified loop ticks at 1 µs; we tick
    /// coarser to run fewer iterations, trading a little TTFT/TPOT quantization
    /// for ~no throughput cost (per-tick work is O(events), not O(ticks)). Finer
    /// ticks also shrink the inter-slice gaps the analyzer trace shows (each
    /// slice end snaps to the tick grid). Clamp 0 → 1 µs: a zero step would make
    /// the derived periodic gates (`ticks()` below) divide by zero.
    pub fn new(duration_ms: f64, run_to_end: bool, tick_dt_us: u64) -> Self {
        let tick_dt = Time::from_us(tick_dt_us.max(1));
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
) -> anyhow::Result<RunSummary> {
    let wall_start = Instant::now();
    let mut clock = Time::ZERO;
    let mut prev_progress = 0u64;
    let mut idle_for = Time::ZERO;
    // In-flight is the frontend's ledger (its emitted cursor − completions fed
    // back below), not a per-tick store scan (that scan was ~90% of runtime on
    // large backlogs). The run loop is a pure consumer: it forwards each
    // completion to the frontend and reads `submitted`/`completed`/`in_flight`
    // back — it keeps no arrival/completion tally of its own.
    // Clock of the most recent periodic `request_state` census. `finalize` reads
    // it to avoid re-writing the same census when the run ends on a snapshot tick
    // (which would duplicate every `(request_id, logging_time)` row).
    let mut last_state_clock: Option<Time> = None;

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
            frontend.record_completion();
            let s = store.borrow();
            logger.record_request_slo(slo_entry(req, clock, &s[req]))?;
        }

        // 2b. Periodic `request_state` snapshot — one AGGREGATE row per tick over
        //     the *admitted* set (`iter_admitted`, the `records[0..=admitted_hi]`
        //     prefix). The analyzer only ever needs `Σ completed_input_len` /
        //     `Σ completed_output_len` per tick (it diffs consecutive ticks for
        //     per-segment throughput), so the sum is computed here instead of
        //     emitting one row per request. The never-admitted pending tail has
        //     zero processed tokens and is skipped — but even on a saturated run,
        //     where prefill admits ~the whole trace, this is one row, not ~150k.
        if snapshot.fire() {
            let s = store.borrow();
            logger.record_request_state(state_agg(&s, clock))?;
            last_state_clock = Some(clock);
        }

        // 2c. Periodic heartbeat. `fire()` runs first so the gate advances every
        //     tick regardless. `submitted`/`completed` are the frontend's ledger
        //     counts; `admitted` (requests that have started prefill = "touched")
        //     is the O(1) store watermark, read only on a heartbeat tick (no
        //     scan). `processing` = admitted - completed (touched but not done).
        if heartbeat.fire() && tracing::enabled!(tracing::Level::INFO) {
            let admitted = store.borrow().admitted_watermark();
            let completed = frontend.num_completed();
            tracing::info!(
                "t={:.0}ms: completed={} submitted={} admitted={} processing={}",
                clock.as_ms(),
                completed,
                frontend.submitted(),
                admitted,
                admitted - completed,
            );
        }

        // 3. Termination — both checks are O(1): in-flight is the maintained
        //    arrival/completion delta, exhaustion is a trace cursor compare.
        let in_flight = frontend.in_flight();
        let exhausted = frontend.exhausted();
        if exhausted && in_flight == 0 {
            break TerminationCause::DrainComplete;
        }
        if !cfg.run_to_end && clock >= cfg.duration {
            break TerminationCause::DurationReached;
        }

        // 3b. Stuck watchdog — O(1), no store scan. Progress means a request
        //     completed OR a new request was admitted (the highest-id admitted
        //     watermark advanced) since the last sample. Both are monotonic, so
        //     their sum advances iff one did. A full sample window with neither
        //     advancing (trace already drained, work still in flight) is a
        //     deadlock. Only checked post-exhaustion, where Stuck can occur.
        if exhausted && in_flight > 0 && watchdog.fire() {
            let progress = frontend.num_completed() + store.borrow().admitted_watermark();
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

    finalize(store, logger, clock, last_state_clock)?;
    logger.flush_all()?;

    // End-of-run summary: finished count, then prefill / decode / total token
    // counts each with their own throughput, then sim + wall time. Built into a
    // `RunSummary` (returned + serialized by the caller); the tracing lines below
    // render from it so the log and `summary.json` can't diverge.
    let wall_s = wall_start.elapsed().as_secs_f64();
    let safe_wall = wall_s.max(1e-9);
    let num_gpus = flow.cluster().borrow().num_gpus();
    let summary = {
        let s = store.borrow();
        let total = s.iter().count() as u64;
        let completed = s.iter().filter(|(_, r)| r.completed).count() as u64;
        let prefill_tok: u64 = s.iter().map(|(_, r)| r.prefill_processed as u64).sum();
        let decode_tok: u64 = s.iter().map(|(_, r)| r.tokens_emitted as u64).sum();
        let all_tok = prefill_tok + decode_tok;
        // Throughput is the *modeled* serving rate: tokens / requests per second
        // of SIMULATED time (what the modeled cluster achieves), not per wall
        // second (that would just be the simulator's speed, captured separately
        // by the "x real-time" ratio below).
        let sim_s = (clock.as_ms() / 1000.0).max(1e-9);
        let total_tok_s = all_tok as f64 / sim_s;
        RunSummary {
            cause,
            requests_total: total,
            requests_finished: completed,
            prefill_tokens: prefill_tok,
            decode_tokens: decode_tok,
            total_tokens: all_tok,
            prefill_tok_s: prefill_tok as f64 / sim_s,
            decode_tok_s: decode_tok as f64 / sim_s,
            total_tok_s,
            num_gpus: num_gpus as u64,
            total_tok_s_per_gpu: total_tok_s / num_gpus.max(1) as f64,
            completed_req_s: completed as f64 / sim_s,
            sim_ms: clock.as_ms(),
            wall_s,
            realtime_x: sim_s / safe_wall,
        }
    };
    tracing::info!("sim complete: cause={:?}", summary.cause);
    tracing::info!(
        "  requests: finished {}/{}",
        summary.requests_finished,
        summary.requests_total
    );
    tracing::info!(
        "  prefill:  {} tokens ({:.0} tok/s)",
        summary.prefill_tokens,
        summary.prefill_tok_s
    );
    tracing::info!(
        "  decode:   {} tokens ({:.0} tok/s)",
        summary.decode_tokens,
        summary.decode_tok_s
    );
    tracing::info!(
        "  total:    {} tokens ({:.0} tok/s, {:.0} tok/s/gpu over {} gpus)",
        summary.total_tokens,
        summary.total_tok_s,
        summary.total_tok_s_per_gpu,
        summary.num_gpus,
    );
    // Modeled completion rate (req per sim-second); then sim vs wall time and
    // the speedup = how many seconds of modeled time we cover per real second.
    tracing::info!(
        "  time:     sim {:.1}ms, wall {:.2}s ({:.1} req/s, {:.1}x real-time)",
        summary.sim_ms,
        summary.wall_s,
        summary.completed_req_s,
        summary.realtime_x,
    );
    Ok(summary)
}

/// The two tables flush asymmetrically because they have different shapes:
/// - `request_state` is a *snapshot* table (one aggregate row per tick), so a
///   final census over the admitted set captures the cumulative token totals at
///   sim-end. Never-admitted requests have no processed tokens and stay outside
///   this throughput/accounting stream.
/// - `request_slo` is *terminal-per-request*: completed requests already wrote
///   their row at their completion tick. Every still-incomplete arrived request
///   gets a partial row here, including the never-admitted pending tail whose
///   queue-stage history is required for backpressure analysis.
///
/// The `request_state` census is skipped when the periodic snapshot (2b) already
/// wrote one at this exact `clock` (`last_state_clock == Some(clock)`, i.e. the
/// run ended on a snapshot tick) — re-writing it would duplicate that tick's
/// aggregate row. `request_slo` is unaffected (the periodic snapshot never writes it).
fn finalize(
    store: &SharedRequests,
    logger: &mut LoggerSession,
    clock: Time,
    last_state_clock: Option<Time>,
) -> anyhow::Result<()> {
    let s = store.borrow();
    let census_already_written = last_state_clock == Some(clock);
    if !census_already_written {
        logger.record_request_state(state_agg(&s, clock))?;
    }
    for (id, rec) in s.iter() {
        if !rec.completed {
            logger.record_request_slo(slo_entry(id, clock, rec))?;
        }
    }
    Ok(())
}

/// Aggregate the admitted set into one `request_state` row: cumulative prefill /
/// decode tokens (the analyzer diffs these between ticks for per-segment
/// throughput) plus admitted/completed counts (diagnostics). O(admitted) — the
/// only per-tick scan the snapshot needs now that rows are not per-request.
fn state_agg(store: &RequestStore, now: Time) -> RequestStateEntry {
    let mut prefill_tokens_cum = 0u64;
    let mut decode_tokens_cum = 0u64;
    let mut n_admitted = 0u64;
    let mut n_completed = 0u64;
    for (_id, rec) in store.iter_admitted() {
        prefill_tokens_cum += rec.prefill_processed as u64;
        decode_tokens_cum += rec.tokens_emitted as u64;
        n_admitted += 1;
        if rec.completed {
            n_completed += 1;
        }
    }
    RequestStateEntry {
        logging_time_ms: now.as_ms(),
        prefill_tokens_cum,
        decode_tokens_cum,
        n_admitted,
        n_completed,
    }
}

fn slo_entry(id: RequestId, now: Time, rec: &RequestRecord) -> RequestSloEntry {
    // Per-token array only when it was logged (empty otherwise) — feeds the
    // `slo-detailed` ITL. The `slo-general` scalars below are computed from the
    // always-tracked first/last token times + count, so they survive the array
    // being off.
    let times_ms: Vec<f32> = rec
        .output_token_times
        .iter()
        .map(|t| t.as_ms() as f32)
        .collect();
    let first_ms = rec.first_token_time.map(|t| t.as_ms());
    let last_ms = rec.last_token_time.map(|t| t.as_ms());
    let ttft_ms = first_ms.map(|f| (f - rec.arrival_time.as_ms()) as f32);
    let finish_decode_time_ms = last_ms.map(|l| l as f32);
    // Mean inter-token gap = total decode span / number of gaps (tokens − 1).
    let tpot_mean_ms = match (first_ms, last_ms) {
        (Some(f), Some(l)) if rec.tokens_emitted > 1 => {
            Some(((l - f) / (rec.tokens_emitted - 1) as f64) as f32)
        }
        _ => None,
    };
    // Stage-transition timeline → four parallel arrays. `rec.stage_log` is empty
    // when `io.log_stage_transitions` is off (`record_stage` never appended), so
    // this is a no-op unpack in that case.
    let n_stages = rec.stage_log.len();
    let mut stage_times_ms = Vec::with_capacity(n_stages);
    let mut stage_codes = Vec::with_capacity(n_stages);
    let mut stage_pool_ids = Vec::with_capacity(n_stages);
    let mut stage_worker_ids = Vec::with_capacity(n_stages);
    for ev in &rec.stage_log {
        stage_times_ms.push(ev.time.as_ms() as f32);
        stage_codes.push(ev.code);
        stage_pool_ids.push(ev.pool.0);
        stage_worker_ids.push(ev.worker.0);
    }
    RequestSloEntry {
        request_id: id.0,
        logging_time_ms: now.as_ms(),
        completed: rec.completed,
        arrival_time_ms: rec.arrival_time.as_ms(),
        output_token_times_ms: times_ms,
        ttft_ms,
        num_output_tokens: rec.tokens_emitted,
        tpot_mean_ms,
        finish_decode_time_ms,
        prefill_processed: rec.prefill_processed,
        stage_times_ms,
        stage_codes,
        stage_pool_ids,
        stage_worker_ids,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::{PoolId, Request, RequestStore, UnifiedStage, WorkerId};
    use crate::orchestrator::{
        DpPlacementPolicy, SimpleDpConfig, SimpleDpFlow, SimpleDpPoolConfig, UnifiedWorkerFactory,
    };
    use crate::sim::frontend::TraceFrontend;
    use crate::test_helpers::FakeModel;
    use crate::worker::{BareboneWorker, WorkerConfig};
    use std::cell::RefCell;
    use std::io::Write;
    use std::path::PathBuf;
    use std::rc::Rc;
    use std::sync::Arc;

    fn write_trace_with_output_len(dir: &std::path::Path, n: u32, output_len: u32) -> PathBuf {
        let path = dir.join("trace.csv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "id,input_len,output_len,arrival_time").unwrap();
        for id in 0..n {
            writeln!(f, "{id},8,{output_len},0.0").unwrap();
        }
        path
    }

    fn write_trace(dir: &std::path::Path, n: u32) -> PathBuf {
        write_trace_with_output_len(dir, n, 3)
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
            Arc::new(FakeModel::for_ms(1.0)),
            Rc::clone(&store),
            // This test asserts the per-token array length, so opt into it.
            WorkerConfig {
                log_output_token_times: true,
                ..WorkerConfig::default()
            },
            None,
            "test-gpu".to_string(),
            "main",
            BareboneWorker::<FakeModel>::new,
        );
        let cfg = SimpleDpConfig {
            dp_pool: SimpleDpPoolConfig {
                pool: PoolId(0),
                num_workers: 2,
                placement: DpPlacementPolicy::RoundRobin,
            },
        };
        let mut flow = SimpleDpFlow::new(cfg, factory);
        let mut frontend = TraceFrontend::load(&[trace], 1.0, None).unwrap();
        let mut logger = LoggerSession::open(dir.path(), true, false).unwrap();

        let summary = run_sim(
            &mut flow,
            &store,
            &mut frontend,
            &mut logger,
            &TickCfg::new(5000.0, true, 100),
        )
        .unwrap();
        assert_eq!(summary.cause, TerminationCause::DrainComplete);
        assert_eq!(summary.requests_finished, 4);
        assert_eq!(summary.decode_tokens, 12); // 4 requests × 3 output tokens
        assert!(summary.total_tok_s > 0.0);
        // 2 workers × 1 GPU each (UnifiedWorkerFactory above), so per-gpu = half.
        assert_eq!(summary.num_gpus, 2);
        assert_eq!(summary.total_tok_s_per_gpu, summary.total_tok_s / 2.0);

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

    #[test]
    fn single_token_request_writes_one_completed_slo_row() {
        let dir = tempfile::tempdir().unwrap();
        let trace = write_trace_with_output_len(dir.path(), 1, 1);
        let store: SharedRequests = Rc::new(RefCell::new(RequestStore::new()));
        let factory = UnifiedWorkerFactory::new(
            Arc::new(FakeModel::for_ms(1.0)),
            Rc::clone(&store),
            WorkerConfig::default(),
            None,
            "test-gpu".to_string(),
            "main",
            BareboneWorker::<FakeModel>::new,
        );
        let cfg = SimpleDpConfig {
            dp_pool: SimpleDpPoolConfig {
                pool: PoolId(0),
                num_workers: 1,
                placement: DpPlacementPolicy::RoundRobin,
            },
        };
        let mut flow = SimpleDpFlow::new(cfg, factory);
        let mut frontend = TraceFrontend::load(&[trace], 1.0, None).unwrap();
        let mut logger = LoggerSession::open(dir.path(), false, false).unwrap();

        let summary = run_sim(
            &mut flow,
            &store,
            &mut frontend,
            &mut logger,
            &TickCfg::new(5000.0, true, 100),
        )
        .unwrap();

        assert_eq!(summary.requests_finished, 1);
        assert!(store.borrow()[RequestId(0)].completed);
        assert_eq!(
            parquet_rows(&dir.path().join("raw/request_slo.parquet")),
            1,
            "completion and sim-end flush must not duplicate the SLO row",
        );
    }

    #[test]
    fn finalize_preserves_never_admitted_request_stage_log() {
        use arrow_array::{Array, ListArray, UInt16Array, UInt32Array};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let dir = tempfile::tempdir().unwrap();
        let store: SharedRequests = Rc::new(RefCell::new(RequestStore::new()));
        {
            let mut s = store.borrow_mut();
            s.insert(&Request::new(RequestId(0), 8, 2, Time::ZERO));
            s[RequestId(0)].record_stage(
                Time::ZERO,
                UnifiedStage::Pending as u16,
                PoolId(0),
                WorkerId(0),
                true,
            );
        }
        assert_eq!(store.borrow().admitted_watermark(), 0);

        let mut logger = LoggerSession::open(dir.path(), false, true).unwrap();
        finalize(&store, &mut logger, Time::from_ms(10.0), None).unwrap();
        logger.flush_all().unwrap();

        let slo_path = dir.path().join("raw/request_slo.parquet");
        let batch =
            ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(slo_path).unwrap())
                .unwrap()
                .build()
                .unwrap()
                .next()
                .unwrap()
                .unwrap();
        assert_eq!(batch.num_rows(), 1);
        let request_ids = batch
            .column_by_name("request_id")
            .unwrap()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        assert_eq!(request_ids.value(0), 0);
        let stage_codes = batch
            .column_by_name("stage_codes")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let codes = stage_codes.value(0);
        let codes = codes.as_any().downcast_ref::<UInt16Array>().unwrap();
        assert_eq!(codes.values(), &[UnifiedStage::Pending as u16]);
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
