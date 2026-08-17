---
name: operate-profile-serving-run
description: >-
  Capture and attribute a bounded nsys profile of a real LLM-serving process.
  Not for simulator wallclock, L1 profile DB rows, or the full alignment pipeline.
---

# Operate Profile Serving Run

This is the operating manual for **profiling a real serving process** — the
framework being built or optimized under
`top-compose-real-framework-from-sim`. It answers "where did the wall time
actually go, per engine phase" for a server you control.

Three neighbours it is deliberately NOT:

- `operate-profile-sim-speed` — how fast the *simulator binary* executes.
- `operate-profile-existing-kernel` — filling L1 `profile.db` rows.
- `operate-run-alignment` — the phased VibeSim↔vLLM alignment pipeline, which
  measures vLLM in order to correct VibeSim. That is the opposite direction.

For tool-altitude selection (torch profiler vs nsys vs ncu) read
[`dev-llm-serving/references/tooling/profiler.md`](../dev-llm-serving/references/tooling/profiler.md)
once. This skill assumes nsys and covers what that generic note leaves out: how
to make two captures **comparable**, and what to do when the trace is opaque
because the engine is your own code.

## Step 1 — Preflight

A profile taken on a contended GPU measures the neighbour, not the engine.

```bash
# Every process on every GPU. The target GPU must show no foreign PID.
nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used --format=csv
```

Then:

- **Pin the GPU** — `CUDA_VISIBLE_DEVICES=<idx>` on both the server and any
  client, and record which physical index that was.
- **Resolve the profiler executable** — do not assume `nsys` is on `PATH`, and
  do not treat one machine's installation path as universal. Require an exact
  `NSYS_BIN`, resolve it, verify that it is executable, and record its version:

  ```bash
  NSYS_BIN="${NSYS_BIN:?set NSYS_BIN to the exact Nsight Systems executable}"
  NSYS_BIN="$(readlink -f -- "$NSYS_BIN")"
  test -x "$NSYS_BIN"
  "$NSYS_BIN" --version | tee <artifact_dir>/nsys-version.txt
  ```

- **Clean up before starting** — a previous run's server holding the port or KV
  pool changes the numbers. Check the port is free and no orphan worker
  survives (`ss -ltnp | grep <port>`; check for stale shared-memory segments if
  the engine uses them). Kill only processes you started; never blanket-`pkill`
  a pattern that could match another user's job.
- **Record provenance** *before* the run, not after:

  | Field | Why it is needed |
  |:------|:-----------------|
  | engine revision + dirty-tree state | the diff under test |
  | full server command + env | reproduction |
  | full client/benchmark command | workload identity |
  | capture tool + exact flags | comparability (Step 2) |
  | instrumentation switch state | on/off changes the timeline |
  | GPU model + index + driver/CUDA version | hardware identity |
  | output artifact paths | later re-reads |

Provenance that is reconstructed after the fact is not provenance. Write it
into the run's artifact directory as the run starts.

## Step 2 — Comparable real-capture parity

Two **real baseline/trial Probe captures** are comparable only if everything
except the code under test is identical. This rule does not imply that a
simulation has, or must reproduce, a profiler command. Fix these once, then
never vary them within a real-capture comparison:

- the same resolved `NSYS_BIN` version, capture mode, and **exact flag string**;
- the same workload, request rate/concurrency, and seed policy;
- the same warmup and the same measurement window;
- the same instrumentation switch state (both on, or both off);
- the same GPU, at the same clock policy.

Any change to the above **invalidates the pair** — re-measure the baseline
under the new flags rather than comparing across them. Record the flag string
verbatim in provenance so a later reader can tell whether two artifacts belong
to the same comparison.

A profiled run is also not a benchmark run: capture overhead perturbs
throughput. Take the score from the unprofiled benchmark run and the
attribution from the profiled run; do not quote a profiled run's throughput as
the result.

For small differences, alternate the baseline/trial run order to reduce clock
and thermal drift. Before comparing two captures, check that both contain the
expected producer ranges and iteration records, cover a comparable share of
kernel work, use the same analysis window, and actually ran the intended
backend and graph mode. An artifact directory only proves that a command ran.

## Step 3 — Warmup, then a bounded capture

Everything one-time must be outside the window: process start, CUDA context
init, weight load, JIT/autotune, CUDA-graph capture, and the cold KV pool.

Prefer a **window inside a longer run** over a short whole-run trace — the
former reaches steady state, the latter measures the ramp.

```bash
# Time-bounded: skip the first 60 s, capture 10 s of steady state.
"$NSYS_BIN" profile \
  --trace=cuda,nvtx,osrt \
  --cuda-graph-trace=node \
  --delay=60 --duration=10 \
  --force-overwrite=true \
  -o <artifact_dir>/steady \
  <server command>
```

```bash
# Range-bounded: the engine itself declares the window. More precise than a
# timer when the steady point is workload-dependent rather than clock-dependent.
"$NSYS_BIN" profile \
  --trace=cuda,nvtx \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o <artifact_dir>/steady \
  <server command>
```

### External-load lifecycle

Profile only the server process tree. The load generator is an external sibling,
not a child of `nsys`; otherwise client CPU/network activity pollutes the
profiled process tree.

Run the lifecycle in this order:

1. create a run-local PID directory and install an exit trap;
2. launch `"$NSYS_BIN" profile ... <server command>` in the background and
   record the profiler PID (and a server PID from a small `exec` wrapper when
   the launcher does not preserve it);
3. poll both `/health` and `/v1/models` until the server is genuinely ready, or
   fail if the profiled process exits;
4. run the fixed warmup client outside the capture window;
5. launch the identical bounded client workload that spans the delayed/range
   capture, recording its PID separately;
6. wait for the client and capture to finish, then send TERM and finally KILL
   only to still-live PIDs recorded by this workflow, in reverse launch order.

Never use a blanket `pkill`/`killall`, never attach `nsys` to the external
client, and never reuse an unverified server left from another run. Preserve
the server command, client command, readiness result, warmup boundary, capture
window, and owned PID list in provenance. The client result from this profiled
run is diagnostic only; capture overhead means its throughput is not the
canonical score.

Keep the trace small enough to actually read. A multi-minute full-server trace
is unreadable and slow to export; seconds of steady state answer the question.
A targeted window is also more *correct*, not merely smaller: a long full-run
capture can silently stop recording CUDA activity partway through, leaving late
iterations with NVTX ranges and no kernels.

### Record iteration time in the framework

Have the framework record each iteration directly with monotonic start/end
timestamps and elapsed time. Include the phase, prefill chunks, decode KV
lengths, scheduled tokens and requests, batch size, rank, and actual graph and
backend mode.

Shape-only records can build timing-predict inputs, but do not say how long a
real iteration took. Do not estimate exact iteration time from client latency
or low-resolution log timestamps. Report framework elapsed time, kernel busy
time, collective wait, device gaps, and uncovered host time separately.

### Trace CUDA graphs at node level, not graph level

**`--cuda-graph-trace=node` is mandatory for any engine that replays CUDA
graphs.** This is not a resolution preference. At the default graph level, a
replay is recorded as one opaque unit, and the decode iterations that run under
graphs come back with NVTX ranges but **no overlapping CUDA kernel rows at all**
— the per-op timing you are trying to attribute simply is not in the export. The
symptom reads like "the profiler lost the decode phase".

At node level each kernel node inside the graph is recorded individually, with
its own `correlationId`, so replayed kernels join to enclosing ranges exactly
like normal launches do (Step 5).

Two adjacent flags decide whether the rows appear at all:

- **`--trace-fork-before-exec=true`** when the CUDA-owning process is a *child*
  interpreter (the common shape: an API/parent process spawns the engine core or
  worker). Without it the trace can contain the parent's NVTX and even
  graph-creation metadata while omitting every replay kernel from the spawned
  worker.
- **`--capture-range=cudaProfilerApi`** (or an NVTX trigger) armed **from the
  CUDA-owning process**, not the parent. A capture armed in the wrong process
  produces the same empty-kernel symptom.

**Do not "solve" this by disabling CUDA graphs.** Running the engine eagerly
does make every kernel a plain launch, but it **changes the kernel set** — graph
mode typically runs compiler-fused kernels that the eager path never emits — so
an eager profile attributes time for a code path the benchmark does not run.
Eager capture is a cross-check, never the measurement of record.

**Verify before analyzing.** After export, confirm node tracing actually took:
the kernel table exists, is non-empty, and has rows whose timestamps fall
*inside* a decode-phase range. An empty or range-disjoint kernel table means the
capture is unusable — re-capture, do not analyze around the hole.

## Step 4 — NVTX readiness, and the instrumentation contract

**The readiness test.** Open the capture and ask one question: *can this trace
attribute time to named engine phases?* For a third-party engine the answer is
usually yes (it ships NVTX ranges). For an engine you wrote yourself the answer
is usually **no** — you see kernels, and between them opaque host gaps with no
name attached.

An opaque host gap is not evidence of anything. It is compatible with
scheduling, input preparation, attention planning, a hidden device sync, and
plain Python overhead — which are different bugs with different fixes.
**When the trace cannot name the gap, add instrumentation before choosing an
optimization.** Guessing at this point is how a plausible-but-wrong hypothesis
gets implemented and measured.

**The instrumentation contract.** Diagnostic ranges are allowed in the engine
under these rules:

- **Opt-in, default off.** One switch (env var or config flag). With it off the
  code path must be free — no range objects constructed, no string formatting.
- **Exception-safe.** Use a context manager / `try-finally` so an exception
  inside the range cannot leak an unclosed range or change control flow.
- **Behavior-neutral.** No numerics change, no added synchronization, no
  reordering. An NVTX range that forces a sync to "get accurate timing" has
  changed the thing it measures.
- **A separate diagnostic commit**, never mixed into the commit under test.
- **Reverted when the Probe ends**, unless deliberately kept as dormant
  observability support — and if kept, it stays default-off.

The Orchestrator defines the diagnostic question and range set, then gives the
Implementer a bounded instrumentation brief. Only the Implementer writes the
patch. To compare baseline and trial, apply the **identical diagnostic
instrumentation patch** to both revisions and enable it in both captures.
Instrumentation present in only one revision, or enabled in only one capture,
invalidates the pair.

**CUDA-graph caveat.** A host-side range around a graph *replay* measures the
launch, not the replayed GPU work; ranges captured *inside* a graph are baked in
at capture time and do not re-emit per replay. Read graph work from the CUDA
kernel rows — which requires the node-level tracing in Step 3 — never from the
enclosing host range. (`operate-run-alignment` carries the same warning for
phase envelopes.)

**The serving range taxonomy.** Cover the phases below; each exists because it
fails in a distinct way and a merged range cannot separate them.

| Range | Covers | The question it answers |
|:------|:-------|:------------------------|
| `<prefix>.engine.iteration` | one scheduler step end-to-end | what is the real per-iteration wall, and how much of it is not kernels |
| `<prefix>.engine.scheduling` | admission, batch composition, preemption | is batch formation on the critical path |
| `<prefix>.engine.input_preparation` | token/position/slot-mapping tensor build, H2D | is per-step tensor construction host-bound |
| `<prefix>.attention.planning` | backend plan/metadata build (e.g. FlashInfer plan) | does planning cost scale with batch and dominate at low batch |
| `<prefix>.graph.preparation` | shape bucketing, buffer copy-in before replay | is graph dispatch overhead eating the graph's win |
| `<prefix>.graph.replay` | the replay launch itself | is the graph actually being used on this path |
| `<prefix>.sampling.d2h` | logits post-processing + the device→host token transfer | is there a blocking sync per step |
| `<prefix>.token.acceptance` | accept/bookkeeping after sampling (incl. speculative) | is acceptance bookkeeping serialized against the next step |
| `<prefix>.request.cleanup` | finished-request teardown, KV block release | does teardown spike with completion bursts |
| `<prefix>.api.detokenization` | token ids → text | is detokenization on the critical path or offloaded |
| `<prefix>.api.streaming` | response emit / network write | is client I/O back-pressuring the engine |

Range names must be stable, low-cardinality, and prefixed for machine filtering
(for example `vibeserve.decode.model_submission`). Never embed a step, request,
sequence, or token ID in the name. If one iteration must be followed
individually, carry its identifier in an NVTX payload/category or emit a
separate marker; keep the enclosing range label fixed so occurrence aggregation
remains bounded and comparable.

## Step 5 — Export and aggregate

Read exported numbers, not screenshots.

```bash
"$NSYS_BIN" stats <artifact_dir>/steady.nsys-rep
"$NSYS_BIN" export --type sqlite \
  --output <artifact_dir>/steady.sqlite \
  <artifact_dir>/steady.nsys-rep
uv run python skills/operate-profile-serving-run/scripts/aggregate_nsys.py \
  <artifact_dir>/steady.sqlite \
  --range-prefix '<prefix>.' \
  > <artifact_dir>/steady-ranges.json
```

`aggregate_nsys.py` uses only the Python standard library and opens the input
SQLite with `mode=ro`. It creates no tables or indexes and never modifies the
evidence artifact. It introspects the required tables and columns before
reading data; a schema mismatch is a hard error, not a partially populated
report.

The script deliberately does not assume
`CUPTI_ACTIVITY_KIND_RUNTIME.globalPid` exists. It:

1. maps each range's `globalTid` into a process namespace from `PROCESSES`
   (including the workspace export's Python-main-thread identity
   `globalPid + pid`);
2. finds runtime calls whose launch start is inside the range on that same
   `globalTid`;
3. matches kernels by the process-qualified key
   `(resolved globalPid, correlationId)`, so equal correlation IDs from
   different workers cannot cross-join.

CUDA-owning ranges on mapped non-main threads are listed explicitly in
`thread_diagnostics`; unmapped or ambiguous threads are also listed, with
runtime calls counted but kernels intentionally left unattributed. Never
replace an unmapped thread with a guessed process.

The JSON aggregates each fixed range label by occurrence and reports:

- occurrence, runtime-call, and kernel counts;
- summed host wall time;
- summed kernel work (individual kernel durations may overlap);
- GPU-busy time from the **union of kernel intervals clipped to each occurrence
  before aggregation**;
- nonnegative uncovered host time and GPU-busy fraction.

Because busy intervals are clipped and unioned per occurrence,
`gpu_busy_fraction` cannot exceed 1.0. `kernel_work_ns` can exceed host wall
time when kernels overlap or extend past the host range. NVTX labels may be
nested, so never sum metrics across parent/child levels; compare one stable
level at a time.

For blocking synchronization specifically, aggregate the sync APIs by enclosing
range (`cudaStreamSynchronize` / `cudaMemcpyAsync`-then-sync / `cudaEventSynchronize`
in `CUPTI_ACTIVITY_KIND_RUNTIME` joined to `StringIds`) in a separate read-only
analysis, rather than inferring a sync from gap shape.

## Step 6 — Report

Reuse the bottleneck vocabulary already defined in
[`profiler.md`](../dev-llm-serving/references/tooling/profiler.md) —
`cpu_launch_bound`, `python_overhead_bound`, `sync_bound`, `memcpy_bound`,
`comm_bound`, `kernel_bound`, `mixed_or_unclear`. Do not invent a parallel
vocabulary.

A report is one primary class plus:

1. the evidence rows that establish it (range label, `gpu_busy_time_ns`,
   `host_wall_time_ns`, `uncovered_host_time_ns`, and counts) — numbers, not
   impressions;
2. the artifact paths and the provenance block from Step 1;
3. what remains unattributed, stated explicitly. Unattributed host time is a
   real finding; silently dropping it makes a partial picture look complete.

## Verify shutdown and generated modules

Record every process started by the run. On success, timeout, or failure, stop
only that process tree and verify that its ports and GPU resources are released.
A finished tmux foreground command does not prove that compiler or worker
children exited.

For custom or JIT-compiled modules:

- record build flags, paths, module identity, and the actual compile/load event;
- finish every required process-local build before reporting server readiness;
- in strict-prebuilt mode, fail before model loading when an artifact is absent;
- record source, dependency, and ABI identity so stale binaries are rejected;
- reject the sample if a worker fails but its parent survives, or if the run
  artifact never reaches a terminal state;
- destroy graph objects that retain collective communicators before destroying
  distributed process groups.

A cached binary is not proof that this run loaded it. A run is complete only
after its processes exit, its artifact reaches a terminal state, and its ports
and GPU resources are released.

## What invalidates a comparison

Any of these means the two captures are not comparable — fix and re-measure,
do not reason across the difference:

- different capture flags, capture mode, or trace set;
- different resolved profiler executable/version;
- one capture at `--cuda-graph-trace=node` and the other at graph level — the
  two do not even contain the same rows;
- one run with CUDA graphs enabled and the other eager — different kernel sets,
  not a faster or slower version of the same set;
- different measurement window or warmup;
- a foreign process on the target GPU in either run;
- different diagnostic instrumentation patches, or instrumentation on in one
  run and off in the other;
- a profiled run's throughput quoted against an unprofiled run's throughput;
- a different GPU, different clock policy, or a thermally throttled capture.

## Neighboring workflows

- Tool-altitude selection and the bottleneck taxonomy:
  [`dev-llm-serving/references/tooling/profiler.md`](../dev-llm-serving/references/tooling/profiler.md).
- Benchmark hygiene (the input to any profile):
  [`dev-llm-serving/references/tooling/serving-benchmark.md`](../dev-llm-serving/references/tooling/serving-benchmark.md).
- The workflow that calls this skill at its Probe step:
  `top-compose-real-framework-from-sim`.
- Simulator wallclock: `operate-profile-sim-speed`. L1 kernel rows:
  `operate-profile-existing-kernel`. VibeSim↔vLLM alignment:
  `operate-run-alignment`.
