---
name: profile-sim-speed
description: Use when asked to profile, perf, or speed up the MLSim *simulator wallclock* (how fast a run executes), not the modeled cluster throughput and not L1 kernel profiling. Covers building a symbol-rich release binary, capturing a `perf record` of a representative run, and reading the per-thread / flat breakdown to find the bottleneck.
---

# Profile Sim Speed

This is for the question "why is the sim *slow to run*" — wallclock seconds, the
`x real-time` ratio, where CPU goes across the sim thread and the background
`mlsim-logger` thread. It is NOT about modeled tok/s (that's the run's output
metric) and NOT about L1 kernel profile.db (that's `profile-run-existing-kernel`).

## Why the env wiring matters

The simulator embeds Python via PyO3 to query profile.db at build time. A bare
`./target/release/simulator` or `cargo run` links/loads the wrong interpreter
(system 3.9) and crashes with `init_fs_encoding` / `libpython3.12.so.1.0: cannot
open shared object file`. The launcher's `launcher.exec._build_subprocess_env`
sets `PYO3_PYTHON`, `LD_LIBRARY_PATH`, `PYTHONHOME`, and `PYTHONPATH` correctly.
Always run the binary under that env. The `--profile` launcher flag (Step 2)
does this for you: it reuses the launcher's preset expansion + env and wraps the
run with `perf record`. The inline `python -c` recipe below it is the manual
fallback for ad-hoc argv that no preset expresses.

## Step 1 — build a symbol-rich release binary

Profile `release`, not `dev` (dev is ~100× slower and unrepresentative). The
workspace `[profile.release]` already sets `debug = true`, so release carries
DWARF for inline attribution — no separate `profiling` profile needed.

```bash
cd /m-coriander/coriander/kanzhu/MLSim_workspace/main
uv run cargo build --release -p simulator
```

`uv run` is mandatory (pins PyO3 to the 3.12 venv). See memory
`mlsim_pyo3_build_python_pin`.

## Step 2 — capture a representative run under perf

Two rules for representativeness:

- **Run long enough that the one-time model build amortizes.** The L4 build
  (profile.db queries + JIT) shows as `python_build` / `_PyEval_EvalFrameDefault`
  in the profile. At 5000s sim it's still ~18%; at **20000s** it's ~4% and the
  steady-state loop dominates. Use `duration_ms = 20000000` for a clean picture.
- **Pin BLAS/OMP to 1 thread** so the build's numpy/torch calls don't smear
  samples across worker threads. `--profile` does this for you.

### Preferred: the `--profile` launcher flag

`python -m launcher <preset>.json --profile` reuses the launcher's preset
expansion, validation, and PyO3 env, and wraps the `run` subprocess with `perf
record -F 499 --call-graph dwarf` under a single-threaded BLAS/OMP env. The
`perf.data` lands in that run's own `log_dir/perf.data`.

```bash
uv run python -m launcher preset/unified_aime_perf.json --profile
# custom sampling Hz (499/999 to avoid lock-step with timers):
uv run python -m launcher preset/unified_aime_perf.json --profile --profile-freq 999
```

`--profile` **requires the preset to expand to exactly one run** (perf on a
parallel sweep is meaningless) — narrow a sweep with `--override` if needed. The
flag enforces this and exits early. `<log_dir>/perf.data` is **binary** — read it
with `perf report`, never `cat`/Read.

### Fallback: inline `perf` for ad-hoc argv

When no preset expresses the argv you want, wrap the binary directly. `-F 499`
(sampling Hz), `--call-graph dwarf` (release omits frame pointers); the grep
keeps the end-of-run summary (wall seconds + `x real-time`) and perf's
sample/byte count.

```bash
uv run python -c "
import os
from launcher.exec import _build_subprocess_env
env=_build_subprocess_env(); env['OPENBLAS_NUM_THREADS']='1'; env['OMP_NUM_THREADS']='1'
b='target/release/simulator'
args=['run','unified','--cp-plan','no-cp','--duration-ms','20000000','--request-rate','150.0','--log-dir','logs/unified_aime_perf','--attn-gpu-memory-gb','80.0','--model-config','model/config/llama3_8b.json','--trace-files','trace/aime_long.csv']
argv=['perf','record','-F','499','--call-graph','dwarf','-o','logs/perf_aime.data','--',b]+args
os.execvpe('perf',argv,env)
" 2>&1 | grep -E 'time:|finished|prefill:|decode:|total:|samples'
```

Swap the workload to whatever you're profiling. `trace/aime_long.csv` (150k
single-round rows, rate 150) is the standard saturating workload — it fills the
KV pool, so most arrivals pile up in the pending queue (a stress case for
logging and the tick loop).

## Step 3 — read the breakdown (native perf, not `perf script`)

`perf script` piping through Python folding is slow on 100MB+ data. Use
`perf report`'s native filters. The `-i` paths below use the inline-recipe
`logs/perf_aime.data`; for a `--profile` run substitute `<log_dir>/perf.data`.

**Per-thread CPU split** (is the cost in the sim thread or the logger thread?):

```bash
perf report -i logs/perf_aime.data --stdio -F overhead --sort comm 2>/dev/null | grep -v '^#' | head -8
```

**Flat top symbols across all threads:**

```bash
perf report -i logs/perf_aime.data --stdio -g none -F overhead,symbol 2>/dev/null | grep -v '^#' | head -20
```

**One thread at a time** (thread names: `simulator` = sim thread,
`mlsim-logger` = background parquet writer):

```bash
perf report -i logs/perf_aime.data --stdio -g none -F overhead,symbol --comm=mlsim-logger 2>/dev/null | grep -v '^#' | head -12
perf report -i logs/perf_aime.data --stdio -g none -F overhead,symbol --comm=simulator   2>/dev/null | grep -v '^#' | head -14
```

## Interpreting — known cost centers and what they mean

- `sim::run::run_sim`, `SimpleDpFlow::tick`, `BareboneWorker::tick` — the real
  per-tick simulation loop. If these dominate, the lever is the loop itself
  (e.g. event-driven next-event advance instead of fixed 100µs ticks), not I/O.
- `mlsim-logger` thread / `parquet::*` / `Interner::intern` /
  `compare_greater` — parquet encode + ZSTD on the background writer. If this is
  large, the lever is **logged row volume** (the dense `request_state` snapshot)
  or per-row allocations.
- `_int_malloc` / `_int_free` / `realloc` / `Vec::from_iter` — allocation. Past
  win: `final_phase` was a per-row `String`; making it a `#[repr(u8)]` enum with
  a `&'static str` mapping killed ~30M allocs.
- `_PyEval_EvalFrameDefault` / `python_build` — the one-time L4 build. Should be
  small at 20000s; if large, you ran too short.
- `slice::sort` — usually `tpot_stats` sorting per-token gaps in `slo_entry`, or
  a batch-formation sort. Check `--comm` to see which thread.

## Reference: optimizations already landed (don't re-discover)

History (50s sim 37s→0.19s ≈ 195×; 20000s wall ~18s ≈ 1090× real-time):

1. `RequestStore` `HashMap`→dense `Vec` + O(1) in-flight counters (was an O(n)
   per-tick scan = ~87%).
2. Cost model `cost_whole_iter_time` fast path (no `LookupResult` tree alloc);
   `cost_verbose` config keeps the full tree opt-in.
3. Background `mlsim-logger` thread (bounded `sync_channel`, `CHANNEL_CAP=64` so
   a dense-snapshot burst queues without stalling the sim).
4. 100µs tick (was 1µs).
5. Dense snapshot logs only the **admitted** prefix (`RequestStore::iter_admitted`),
   not the never-served pending tail; `final_phase` String→enum.

## Cleanup

The `perf.data` is large (~100–230MB) — `logs/perf_aime.data` for the inline
recipe, `<log_dir>/perf.data` for a `--profile` run. Leave it for follow-up
`perf report` passes; `rm` it only when done profiling.

## The `--profile` launcher flag

Implemented. `python -m launcher <preset>.json --profile [--profile-freq HZ]`
wraps the single expanded run with `perf record` using the launcher's own preset
expansion + PyO3 env, writing `<log_dir>/perf.data`. Wiring:
`launcher.exec.wrap_with_perf` / `_profile_env` / `perf_available`, threaded
through `launcher.sweep._launch_one` → `run_single`, gated in
`launcher.__main__` (requires exactly one run; checks `perf` is on PATH). Use it
in preference to the inline `python -c` recipe whenever a preset expresses the run.
