---
name: operate-profile-sim-speed
description: Use when asked to profile, perf, or speed up the VibeSim *simulator wallclock* (how fast a run executes), not the modeled cluster throughput and not L1 kernel profiling. Covers building a symbol-rich release binary, capturing a `perf record` of a representative run, and reading the per-thread / flat breakdown to find the bottleneck.
---

# Profile Sim Speed

This is for the question "why is the sim *slow to run*" — wallclock seconds, the
`x real-time` ratio, where CPU goes across the sim thread and the background
`vibesim-logger` thread. It is NOT about modeled tok/s (that's the run's output
metric) and NOT about L1 kernel profile.db (that's
`operate-profile-existing-kernel`).

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
cd /m-coriander/coriander/kanzhu/VibeSim_workspace/main
uv run cargo build --release -p simulator
```

`uv run` is mandatory (pins PyO3 to the 3.12 venv). See memory
`vibesim_pyo3_build_python_pin`.

## Step 2 — capture a representative run under perf

Two rules for representativeness:

- **Isolate the steady state from the one-time build.** The L4 build (profile.db
  queries + JIT + MoE cost-curve build) shows as `python_build` /
  `_PyEval_EvalFrameDefault` / `__memcmp` / `for_each_routed_token` /
  `simulate_once` in the profile, all front-loaded at startup. Two ways to keep it
  out of the numbers:
  - **PREFERRED — modest run + window a ~10s steady slice (Step 3a).** Decouple the
    *run length* from the *sample window*: run long enough that the steady regime is
    well-developed and representative — aim for **~1min of wall** (e.g. `duration_ms`
    ≈ 600000-1000000; the KV pool has filled, admission has reached its saturating
    rhythm) — but still restrict the perf analysis to a **~10s window taken from the
    middle-to-late part of the run** with `perf report --time <start>,<stop>`. That
    region is fully warmed up (build long done, caches hot, steady state settled) yet
    the slice is small, so `perf report` on a 500MB+ `perf.data` returns in seconds
    instead of minutes. This is the fast iteration loop — a modest capture plus a
    windowed read beats a 20000s capture for both build-exclusion *and* analysis
    speed. **10s of steady sim is plenty of samples** (at -F 499 that's ~5000
    sim-thread samples). Don't sample the last second or two — window a slice that is
    safely mid-steady, not at the ramp-down tail.
  - Alternative — run so long the build amortizes globally: at 5000s it's still
    ~18%, at **20000s** ~4%. Only worth it when you specifically want a whole-run
    average rather than a clean steady slice; it costs far more wall to capture.
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

**Fast path — one command for the whole breakdown.**
`tools/sim-speed-perf/perf_breakdown.sh <perf.data>` runs all five views below
concurrently over a single auto-derived steady window (Step 3a), auto-annotates the
top-N hot `simulator::` symbols to source lines (Step 3b), and saves the full raw
output (every symbol / annotated line) next to the `perf.data` under
`perf_breakdown/` (`*.full.txt` + a head-limited `breakdown.txt` also tee'd to
stdout). Reach for it first; the manual `perf report` invocations below are the
fallback when you want a view it doesn't cover or need to tweak a filter.

```bash
tools/sim-speed-perf/perf_breakdown.sh <log_dir>/perf.data                 # auto window, NSYM=4
NSYM=6 tools/sim-speed-perf/perf_breakdown.sh <log_dir>/perf.data 1234 1244 # explicit abs-second window
```

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
`vibesim-logger` = background parquet writer):

```bash
perf report -i logs/perf_aime.data --stdio -g none -F overhead,symbol --comm=vibesim-logger 2>/dev/null | grep -v '^#' | head -12
perf report -i logs/perf_aime.data --stdio -g none -F overhead,symbol --comm=simulator   2>/dev/null | grep -v '^#' | head -14
```

## Step 3a — window to a warmed-up ~10s steady slice (do this first)

perf's absolute sample clock lets you cut the one-time build out of the read
without re-running. Get the sim-thread sample time bounds, then take a ~10s window
from the **middle-to-late** range (warmed up, build excluded):

```bash
# 1. sim-thread sample time bounds (first .. last, in seconds). One awk pass;
#    filtered to --comm=simulator so only the sim thread counts. ~30-60s on 500MB.
perf script -i <log_dir>/perf.data --comm=simulator -F time 2>/dev/null \
  | awk 'NR==1{f=$1} {l=$1} END{gsub(/:/,"",f); gsub(/:/,"",l); print "first="f" last="l" span="l-f}'
# e.g. first=1558739.57 last=1558793.39 span=53.82  (≈11s build + ≈43s steady sim)

# 2. pick a ~10s window well past the build front (here the build is ~first 11s,
#    so start ~20s in) and read the flat self-times inside it:
perf report -i <log_dir>/perf.data -g none -F overhead,symbol --comm=simulator \
  --percentage relative --time 1558770,1558780 --stdio 2>/dev/null | grep -vE '^#|^$' | head -20
```

- `--time <start>,<stop>` takes **absolute seconds** from the same clock `perf
  script -F time` prints (NOT offsets, NOT percentages — the `45%,100%` form errors
  with `Invalid time string` on this perf build).
- `--percentage relative --comm=simulator` rescales so the sim thread = 100%.
- **Confirm the window is steady, not build:** the build symbols (`__memcmp`,
  `for_each_routed_token`, `simulate_once`, `_PyEval_EvalFrameDefault`) should be
  **gone** inside the window. If they're still there, move `start` later.
- Windowing a 10s slice also makes `perf report` itself fast (fewer samples to
  fold) — the whole point of the short-run + window loop.

## Step 3b — the release-inlining caveat (read before trusting the call tree)

In `thin`-LTO release the DWARF **call graph is unreliable** — `perf report -g
graph` shows broken frames (`0x3`, `0xb`, raw addresses) and OOMs on 100MB+ data.
Do NOT present a literal nested call tree from it. Two consequences:

- **Trust flat self-times, not the call graph.** `perf report -g none -F
  overhead,symbol --comm=simulator` is clean and reliable. Build the cost tree by
  *logically grouping* those self-time symbols (see Presenting), not by reading
  perf's `-g graph` output.
- **Self-time absorbs inlined callees.** A fat self-time on an outer function is
  often inlined work, not the function's own body. Example: `run_sim` showed 35%
  self — most of it was the **inlined** stuck-watchdog `progress_signature` O(n)
  store scan, not loop overhead. **Confirm inlining by absence:** if a function
  you expect (`progress_signature`, `state_entry`, `drain_due`) does NOT appear as
  its own symbol in the flat list, it was inlined into its caller's self-time.
  Then read the *source* of the hot outer function to find what it inlines.

Method that works: flat self-times (where) → read the hot symbol's source (what
it inlines / how often it runs) → form a hypothesis → confirm with dhat (allocs)
or a targeted code-level count (tick count × per-iter cost).

## Step 4 — attribute allocations with dhat (when alloc symbols dominate)

When `_int_malloc`/`_int_free`/`realloc`/`memmove`/`from_iter` are a big slice
(they were ~14% of the sim thread once the watchdog was fixed), perf tells you
*allocation is costly* but not *which call sites*. dhat gives exact Rust
allocation call stacks. It is wired behind an **opt-in cargo feature** so normal
builds are byte-identical (zero cost): `dhat` is an `optional` dep and both the
`#[global_allocator]` and the `dhat::Profiler` guard in `main.rs` are
`#[cfg(feature = "dhat-heap")]`.

dhat's allocator only intercepts Rust's `GlobalAlloc`, so Python/torch C-side
allocations bypass it — the dump is dominated by the sim's own Rust allocations
(exactly what we want). It IS ~10× slower, so profile a **short** run (the
per-iteration allocation *pattern* is the same at 300s as at 20000s).

```bash
uv run cargo build --release --features dhat-heap -p simulator
# run the SHORT workload directly under the launcher's PyO3 env (dhat writes
# dhat-heap.json to CWD on exit):
uv run python -c "
import subprocess, os
from launcher.exec import _build_subprocess_env
env=_build_subprocess_env(); env['OPENBLAS_NUM_THREADS']='1'; env['OMP_NUM_THREADS']='1'
b='target/release/simulator'
args=['run','unified','--cp-plan','no-cp','--duration-ms','300000','--request-rate','150.0','--log-dir','logs/dhat_run','--attn-gpu-memory-gb','80.0','--model-config','model/config/llama3_8b.json','--trace-files','trace/aime_long.csv']
r=subprocess.run([b]+args, env=env); print('exit', r.returncode)
"
```

Rank the call sites by **block count** (allocation churn → malloc/free overhead)
and by **bytes**, mapping each program point to its first `simulator::` frame:

```bash
uv run python -c "
import json
d=json.load(open('dhat-heap.json')); ftbl=d['ftbl']
def sim_frame(fs):
    for fi in fs:
        f=ftbl[fi]
        if 'simulator::' in f and 'dhat' not in f: return f.split(' (')[0]
    return ftbl[fs[0]] if fs else '?'
rows=[(p.get('tbk',0), p.get('tb',0), sim_frame(p.get('fs',[]))) for p in d['pps']]
print('blocks=%d bytes=%d'%(sum(r[0] for r in rows), sum(r[1] for r in rows)))
for tbk,tb,fr in sorted(rows, reverse=True)[:18]:
    print(f'{tbk:>9} blk  {tb:>12,} B   {fr[:92]}')
"
```

`tbk` (blocks) ≈ malloc/free calls = the churn that shows as `_int_*` in perf;
`tb` (bytes) flags large transient buffers. The recurring VibeSim finding is
**per-iteration scratch `Vec`s** (`build_arch_input`, `complete_iter`,
`CostTree::aggregate`, `projected_peak`) that should be reused buffers held on the
worker (the `cost_slots` pattern), plus stray `.clone()`s (`eval_buf`).

## Presenting results

Three artifacts, in this order:

1. **Per-thread split** (`--sort comm`): is cost in `simulator` (sim thread) or
   `vibesim-logger` / `vibesim-cost-logg` (background writers)? Quote each thread's %.
2. **Sim-thread cost tree, normalized to "sim thread = 100%"** — built from flat
   self-times grouped into logical buckets (driver loop / allocation / cost-model
   eval / admission+sort / one-time build / tail), NOT from perf's `-g graph`.
   Always note the inlining caveat for any fat self-time bucket.
3. **Before/after table** for any change: `wall`, `× real-time`, the moved
   symbol's self-%, and **termination cause + finished count** (prove behavior is
   unchanged — a speedup that changes the sim result is a bug, not a win).

Optional: export a flamegraph SVG from an existing `perf.data` with the
`flamegraph` binary (`~/.cargo/bin/flamegraph`, bundles inferno) — it streams via
`perf script` so it tolerates the 100MB+ file the in-memory `-g graph` chokes on:
`~/.cargo/bin/flamegraph --perfdata <log_dir>/perf.data -o <log_dir>/flamegraph.svg`.

## Interpreting — known cost centers and what they mean

- `sim::run::run_sim`, `SimpleDpFlow::tick`, `BareboneWorker::tick` — the real
  per-tick simulation loop. If these dominate, the lever is the loop itself
  (e.g. event-driven next-event advance instead of fixed 100µs ticks), not I/O.
- `vibesim-logger` thread / `parquet::*` / `Interner::intern` /
  `compare_greater` — parquet encode + ZSTD on the background writer. If this is
  large, the lever is **logged row volume** (the dense `request_state` snapshot)
  or per-row allocations.
- `_int_malloc` / `_int_free` / `realloc` / `memmove` / `Vec::from_iter` —
  allocation. perf can't name the call sites (broken DWARF); use **dhat** (Step 4)
  to attribute. The recurring driver is per-iteration scratch `Vec`s in
  `build_arch_input` / `complete_iter` / `CostTree::aggregate` / `projected_peak`.
  Past win: `final_phase` per-row `String` → `#[repr(u8)]` enum killed ~30M allocs.
- `_PyEval_EvalFrameDefault` / `python_build` — the one-time L4 build. Should be
  small at 20000s; if large, you ran too short.
- `slice::sort` — the per-batch-formation `projected_peak_kv` decode sort
  (`admission_helpers.rs`). The TPOT per-token-gap sort was moved OFF the sim
  thread to the writer thread (`log::rows::tpot_stats_ms`); if you see TPOT
  sorting on `simulator`, that regressed. Check `--comm` to see which thread.

## Reference: optimizations already landed (don't re-discover)

History (50s sim 37s→0.19s ≈ 195×; aime_long 20000s wall **~12.3s ≈ 1620×
real-time** after the watchdog fix below):

1. `RequestStore` `HashMap`→dense `Vec` + O(1) in-flight counters (was an O(n)
   per-tick scan = ~87%).
2. Cost model `cost_whole_iter_time` fast path (no `LookupResult` tree alloc);
   `cost_verbose` config keeps the full tree opt-in.
3. Background `vibesim-logger` thread (bounded `sync_channel`, `CHANNEL_CAP=64` so
   a dense-snapshot burst queues without stalling the sim).
4. 100µs tick (was 1µs).
5. Dense snapshot logs only the **admitted** prefix (`RequestStore::iter_admitted`),
   not the never-served pending tail; `final_phase` String→enum.
6. **Stuck-watchdog made O(1)** (biggest recent win: 18.6s→12.3s, −34%). The old
   `progress_signature` summed token counts over *all* requests every 1s of sim
   once the trace drained (~19k scans × 150k reqs) — inlined into `run_sim`, it
   was ~30pp of `run_sim`'s self-time. Now it compares `completed +
   RequestStore::admitted_watermark()` (two monotonic O(1) counters) every 100s.
7. **TPOT percentile sort moved sim-thread → writer thread** (`slo_entry` no
   longer calls `tpot_stats`; `log::rows::tpot_stats_ms` derives it in
   `slo_to_record_batch` from the already-logged `output_token_times`). Removed
   ~6pp of sim-thread sort.
8. **`cost_log` size −31%**: dropped the redundant `wall_end_ms` column (=
   `wall_start_ms + total_time_ms`, derivable; it was a high-entropy f64 that
   barely compressed, 6 MiB of a 19 MiB file).

After 6–8 the top sim-thread cost is `SimpleDpFlow::tick` (~29%, the 200M-tick
orchestration) and per-iteration allocation (~14%, see dhat Step 4). The next
structural lever is the fixed-tick count itself (coarser `tick_dt` / event-driven
next-event advance).

## Cleanup

The `perf.data` is large (~80–230MB) — `logs/perf_aime.data` for the inline
recipe, `<log_dir>/perf.data` for a `--profile` run. Leave it for follow-up
`perf report` passes; `rm` it only when done profiling. `dhat-heap.json` is
written to CWD — move it into the run's `<log_dir>/dhat/` (or `rm`). The
`dhat-heap` feature stays in the tree (off by default, zero cost); don't rip it
out — it's the canonical alloc-attribution path.

## The `--profile` launcher flag

Implemented. `python -m launcher <preset>.json --profile [--profile-freq HZ]`
wraps the single expanded run with `perf record` using the launcher's own preset
expansion + PyO3 env, writing `<log_dir>/perf.data`. Wiring:
`launcher.exec.wrap_with_perf` / `_profile_env` / `perf_available`, threaded
through `launcher.sweep._launch_one` → `run_single`, gated in
`launcher.__main__` (requires exactly one run; checks `perf` is on PATH). Use it
in preference to the inline `python -c` recipe whenever a preset expresses the run.
