# L7 Sim core — the tick driver & trace frontend

The deployment-independent heart of the simulator: **one process, one sim thread,
one clock**. It loads a workload trace, drives whatever L6 `Flow` it is handed,
emits per-request parquet rows, and stops on a termination policy. Logging may
spawn writer threads, but the modeled simulation loop itself is single-threaded.
It knows nothing about *which* deployment it runs — it only ticks the `Flow` trait
and reads request lifecycle back from the shared store for logging.

This is the practical, code-matching reference; the code is the ground truth.
For the layer overview see `doc/detailed_design/L7.md`.

## Two parts

- **`frontend/` (L7-γ) — `TraceFrontend<Definition>`.** Loads a declared exact
  CSV schema into one concrete request family. `LoadedTrace` performs startup
  dispatch only; the tick loop receives the statically narrowed text frontend.
  Family definitions live one-per-file under `common/request_family/`; the
  directional text/media families remain distinct, while `omni_generation`
  alone accepts ordered mixed text/image/audio/video segment vectors.
  `ReplayScheduler` reads a separate `ReleaseMetadata` projection and composes
  three independent axes:
  - **Arrival mode** — `arrival_mode: trace_timed | saturated`. Trace-timed
    `drain_due(now, emit)` emits every `Request` whose **effective** arrival
    time `≤ now`, where effective time = `arrival_time / request_rate`.
    Saturated ignores the CSV timeline and `request_rate`: every unit is
    eligible from the start.
  - **Capacity** — `max_concurrency: N`, optional and independent of arrival
    mode. It caps active *units*, and a unit is a session when chained and a
    request otherwise. A unit that the cap held back is stamped with the instant
    its slot opened, not its trace arrival, so it is not charged for a wait the
    measured runner does not report either.
  - **Session dependency** —
    `session_dependency: independent | chained`. Independent rows have no causal
    gate. Chained session heads follow the arrival mode and the cap, while each
    successor waits for predecessor completion plus `tool_wait_after_ms`.

  All combinations are valid; the axes were deliberately split apart because a
  capped replay of a recorded timeline is a real workload and a single fused
  enum could not express it.

  Under `chained`, a session takes its slot when its head is released and holds
  it until its final round completes — **including across tool waits**, when it
  has no request in flight at all. This is why the cap cannot be served by the
  in-flight count: that would hand the slot to another conversation and let
  both run. A successor therefore bypasses the capacity check entirely; its
  session already owns a slot, and re-gating it would deadlock, since the only
  thing that frees a slot is the session completion that successor leads to.

  The frontend owns the in-flight ledger (`emitted − completed` fed back via
  `record_completion`); the tick loop is a pure consumer of
  `submitted`/`in_flight`/`num_completed`.

  The drain loop lives here so a caller can't under-drain by polling once per
  tick. The text schema remains the four legacy columns and is declared as
  `trace_kind: text_generation`. `RequestStore::reserve_slots` creates empty
  `Option` slots; requests are inserted only when the scheduler releases them.
  Compact arrived/admitted id indexes keep lifecycle scans proportional to the
  relevant live set rather than the full reserved trace.
- **`run.rs` (L7-β) — `run_sim`.** The single tick loop. Returns
  `anyhow::Result<RunSummary>`.

## The tick loop

```rust
run_sim(flow: &mut dyn Flow, store, frontend, logger, cfg) -> anyhow::Result<RunSummary>
```

Each tick, in order:

1. **Drain arrivals** due by `clock` → `flow.on_arrival(req)` (the flow inserts
   facts into the shared `RequestStore`).
2. **`flow.tick(clock)`** → `Vec<OrchAction>`; each `Complete { req }` writes that
   request's terminal `request_slo` parquet row.
3. **Periodic `request_state` snapshot** (every `snapshot_dt`) — dense over the
   *admitted* set, so per-segment workload is a plain diff of consecutive
   snapshots. Plus a heartbeat log line.
4. **Termination check** (all O(1)): `DrainComplete` (trace exhausted + zero
   in-flight), `DurationReached` (`clock ≥ duration`, unless `run_to_end`), or
   `Stuck` (a watchdog: trace drained but neither completions nor the
   admitted-id watermark advanced for `stuck_threshold`).
5. **Advance** `clock += tick_dt`.

In-flight is tracked from arrival/completion **counters**, never by scanning the
store each tick (that scan was ~90% of runtime on large backlogs). Periodic steps
share one `EveryN` iteration gate.

At sim-end, `finalize` flushes the two streams over their respective scopes. The
final aggregate `request_state` census covers the admitted set, so work between
the last snapshot and sim-end is not lost. Partial `request_slo` rows cover every
still-incomplete arrived request, including a never-admitted pending tail whose
stage timeline is needed for queue/backpressure analysis. The
`census_already_written` guard skips the `request_state` census when the run
ended on a snapshot tick (`last_state_clock == Some(clock)`) to avoid duplicating
that aggregate row.

## `TickCfg` — the loop knobs

`tick_dt` 100 µs, `snapshot_dt` 10 s, `stuck_threshold` 60 s. The 10 s snapshot
cadence is the raw temporal resolution of segmented throughput. The 100 µs tick is
a deliberate quantization choice: per-iteration/token work is `O(events)`, not
`O(ticks)`, so a finer tick buys ~no accuracy for more iterations (the doc-comment
records the bias measurements).

## `RunSummary` — the end-of-run artifact

Written to `<log_dir>/summary.json` (and rendered to the end-of-run log lines from
the same struct, so they can't diverge). Every field **except** `wall_s` /
`realtime_x` is **deterministic** given a fixed trace + config + cost model:
finished count, prefill/decode/total tokens and their throughput — all in
**simulated** time (the modeled cluster's serving rate), not wall time. `wall_s` /
`realtime_x` are the *simulator's own* speed (host-load dependent) — for
trend/bench, never a tight assertion. `TerminationCause` records why the loop
stopped.

`repro/` is a stub for the reproducibility sidecar.

## How it's driven (the binary)

`main.rs` is the L7 CLI that wires this core. `Run` (and `build-cache-only` /
`dry-run` / `list-params` / `kernel-query`) is the entry: parse the structured
`RunConfig` → `LoadedTrace::load` → narrow to the current text family → reserve
empty store slots → `deployment::build_flow` to get the `Box<dyn Flow>` →
`LoggerSession::open` → write `raw/run_meta.json` (the
registry/KV/comm facts borrowed from `flow.cluster()`) → `run_sim` → serialize
`summary.json`. The launcher
spawns this binary (see `launcher/README.md`).

## Up / down

- **Above (driver):** the `simulator` binary's `run` subcommand; one level up,
  the Python launcher spawns it.
- **Below (used):** the L6 `Flow` it ticks (built by the `deployment` layer), the
  `LoggerSession` it writes parquet through, and the `SharedRequests` store it
  reads lifecycle state from.
