---
name: operate-profile-serving-run
description: >-
  Use when capturing or re-capturing a profile of a REAL LLM-serving process
  (the framework under optimization) so a measured result can be attributed to
  engine phases — GPU-idleness preflight, warmup exclusion, bounded nsys
  capture, capture-flag parity between baseline and trial, node-level CUDA-graph
  tracing (`--cuda-graph-trace=node`) so graph-replay kernels appear at all, the
  NVTX readiness check and its opt-in default-off instrumentation contract,
  `nsys export` SQLite aggregation correlating NVTX ranges with CUDA API calls
  and GPU kernels, process cleanup, and artifact provenance. NOT the VibeSim simulator's
  own wallclock (that is operate-profile-sim-speed), NOT L1 kernel profile.db
  rows (that is operate-profile-existing-kernel), and NOT the phased
  VibeSim-to-vLLM alignment pipeline (that is operate-run-alignment).
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

## Step 2 — Capture parity

Two captures are comparable only if **everything except the code under test is
identical**. Fix these once, then never vary them within a comparison:

- the same capture tool, capture mode, and **exact flag string**;
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

## Step 3 — Warmup, then a bounded capture

Everything one-time must be outside the window: process start, CUDA context
init, weight load, JIT/autotune, CUDA-graph capture, and the cold KV pool.

Prefer a **window inside a longer run** over a short whole-run trace — the
former reaches steady state, the latter measures the ramp.

```bash
# Time-bounded: skip the first 60 s, capture 10 s of steady state.
nsys profile \
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
nsys profile \
  --trace=cuda,nvtx \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o <artifact_dir>/steady \
  <server command>
```

Keep the trace small enough to actually read. A multi-minute full-server trace
is unreadable and slow to export; seconds of steady state answer the question.
A targeted window is also more *correct*, not merely smaller: a long full-run
capture can silently stop recording CUDA activity partway through, leaving late
iterations with NVTX ranges and no kernels.

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
| `engine_iteration` | one scheduler step end-to-end | what is the real per-iteration wall, and how much of it is not kernels |
| `scheduling` | admission, batch composition, preemption | is batch formation on the critical path |
| `input_preparation` | token/position/slot-mapping tensor build, H2D | is per-step tensor construction host-bound |
| `attention_planning` | backend plan/metadata build (e.g. FlashInfer plan) | does planning cost scale with batch and dominate at low batch |
| `graph_preparation` | shape bucketing, buffer copy-in before replay | is graph dispatch overhead eating the graph's win |
| `graph_replay` | the replay launch itself | is the graph actually being used on this path |
| `sampling_and_d2h` | logits post-processing + the device→host token transfer | is there a blocking sync per step |
| `token_acceptance` | accept/bookkeeping after sampling (incl. speculative) | is acceptance bookkeeping serialized against the next step |
| `cleanup` | finished-request teardown, KV block release | does teardown spike with completion bursts |
| `detokenization` | token ids → text | is detokenization on the critical path or offloaded |
| `streaming` | response emit / network write | is client I/O back-pressuring the engine |

Name ranges with a stable identifier (e.g. the step index) so a range can be
followed across iterations rather than only aggregated.

## Step 5 — Export and aggregate

Read exported numbers, not screenshots.

```bash
nsys stats <artifact_dir>/steady.nsys-rep            # quick per-kernel / per-API summary
nsys export --type sqlite \
  --output <artifact_dir>/steady.sqlite \
  <artifact_dir>/steady.nsys-rep
```

**Check the schema first — table and column names vary across nsys versions:**

```bash
sqlite3 <artifact_dir>/steady.sqlite ".tables"
sqlite3 <artifact_dir>/steady.sqlite ".schema NVTX_EVENTS"
sqlite3 <artifact_dir>/steady.sqlite ".schema CUPTI_ACTIVITY_KIND_KERNEL"
```

The correlation has two joins, and they are different in kind:

1. **NVTX range → CUDA API call**: by *time containment* on the same thread
   (`globalTid`), because an NVTX range is a host-side interval.
2. **CUDA API call → GPU kernel**: by `correlationId` **scoped to `globalPid`**,
   because that is the launch↔execution identity CUPTI records and it is only
   unique within a process. Omitting `globalPid` cross-joins workers in any
   multi-process (TP/EP) capture.

Under node-level graph tracing, replayed graph kernels carry their own
`correlationId` and join through this same path — which is exactly why Step 3
insists on it.

Chaining both gives GPU time attributed to a named engine phase:

```sql
-- GPU kernel time and launch count attributed to each NVTX range name.
WITH ranges AS (
  SELECT COALESCE(e.text, s.value) AS range_name, e.start, e.end, e.globalTid
  FROM NVTX_EVENTS e
  LEFT JOIN StringIds s ON s.id = e.textId
  WHERE e.end IS NOT NULL
)
SELECT r.range_name,
       COUNT(*)                                   AS kernel_launches,
       SUM(k.end - k.start) / 1e6                 AS gpu_ms,
       SUM(r.end - r.start) / 1e6                 AS host_range_ms
FROM ranges r
JOIN CUPTI_ACTIVITY_KIND_RUNTIME api
  ON api.start >= r.start AND api.end <= r.end
 AND api.globalTid = r.globalTid
JOIN CUPTI_ACTIVITY_KIND_KERNEL k
  ON k.correlationId = api.correlationId
 AND k.globalPid    = api.globalPid      -- correlationId is per-process
GROUP BY r.range_name
ORDER BY gpu_ms DESC;
```

The CUPTI tables are exported **without indexes**, so this join degrades into
repeated full scans on a real trace. Adding B-tree indexes to the SQLite file is
safe — it is a rebuildable derivative of the immutable `.nsys-rep`, and indexes
change neither the rows nor the attribution:

```sql
CREATE INDEX IF NOT EXISTS ix_rt  ON CUPTI_ACTIVITY_KIND_RUNTIME(globalTid, start);
CREATE INDEX IF NOT EXISTS ix_krn ON CUPTI_ACTIVITY_KIND_KERNEL(globalPid, correlationId, start);
```

Two derived numbers carry most of the diagnosis:

- **`host_range_ms − gpu_ms` per range** — the host-side cost the GPU did not
  absorb. This is the launch-overhead / sync / Python cost, localized to a named
  phase instead of an anonymous gap.
- **GPU busy fraction inside `engine_iteration`** — `gpu_ms / host_range_ms`.
  Low means the iteration is not kernel-bound, and the ranking of the other
  ranges says which phase to attack.

For blocking synchronization specifically, aggregate the sync APIs by enclosing
range (`cudaStreamSynchronize` / `cudaMemcpyAsync`-then-sync / `cudaEventSynchronize`
in `CUPTI_ACTIVITY_KIND_RUNTIME` joined to `StringIds`), rather than inferring a
sync from gap shape.

Nested ranges double-count if summed naively — either aggregate one nesting
level at a time, or subtract child ranges from their parent before comparing.

## Step 6 — Report

Reuse the bottleneck vocabulary already defined in
[`profiler.md`](../dev-llm-serving/references/tooling/profiler.md) —
`cpu_launch_bound`, `python_overhead_bound`, `sync_bound`, `memcpy_bound`,
`comm_bound`, `kernel_bound`, `mixed_or_unclear`. Do not invent a parallel
vocabulary.

A report is one primary class plus:

1. the evidence rows that establish it (range name, `gpu_ms`, `host_range_ms`,
   counts) — numbers, not impressions;
2. the artifact paths and the provenance block from Step 1;
3. what remains unattributed, stated explicitly. Unattributed host time is a
   real finding; silently dropping it makes a partial picture look complete.

## What invalidates a comparison

Any of these means the two captures are not comparable — fix and re-measure,
do not reason across the difference:

- different capture flags, capture mode, or trace set;
- one capture at `--cuda-graph-trace=node` and the other at graph level — the
  two do not even contain the same rows;
- one run with CUDA graphs enabled and the other eager — different kernel sets,
  not a faster or slower version of the same set;
- different measurement window or warmup;
- a foreign process on the target GPU in either run;
- instrumentation on in one run and off in the other;
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
