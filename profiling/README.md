# VibeSim Profiling (L1)

The **L1 cost layer**: the source of truth for "how long does one kernel take on
one GPU". The Rust simulator never measures a kernel itself — it asks this Python
package for a cached time, and this package either returns a stored measurement
or (when allowed) runs the real kernel on a real GPU to produce one.

This is the practical, code-matching reference. For the layer overview see
`doc/detailed_design/L1.md`. If this file disagrees with the code, the
code wins — open an issue.

## Two halves: measure (L1a) vs. cache+serve (L1b)

The package is split by responsibility, and the split is strict:

- **L1a — measurement.** `runners/` + `profilers/`. Given concrete kernel
  arguments, allocate tensors, launch the kernel on a GPU, time it, and return a
  `Metrics` dataclass. Knows nothing about SQLite, caching, or who asked.
- **L1b — cache + serving.** `db/`, `facade.py`, `perf_api.py`, `exec/`. Owns the
  SQLite cache, the `(kernel, backend)` registry, the public query API, the
  JIT-profiling policy, and the subprocess execution that drives L1a runners.

A runner returns `Metrics` and nothing DB-shaped. L1b owns persistence and
query policy. Keep that boundary: it is why a runner can be developed and tested
without a database, and why the cache can be reorganized without touching kernels.

## What the upper layer gets / what the lower layer must provide

**Exposed upward** (the entire public surface is `profiling.perf_api`):

| Symbol | Purpose |
|---|---|
| `get_<kind>_times(specs, *, backend, gpu_name=None, force=False)` | Batch query: cached `Metrics` per spec, or `MissingEntry` on a miss. |
| `count_missing_<kind>(specs, *, backend, gpu_name=None)` | Read-only dry run: how many specs are not cached. Never profiles. |
| `enable_jit_profiling()` / `disable_jit_profiling()` | Process-wide toggle: may a cache miss trigger a real GPU run? |
| `get_current_gpu_name()` | Resolve the CUDA device-0 name used as the DB key. |
| `get_db_metadata()` / `get_profiler_versions(...)` | Schema + provenance for run manifests. |

The `get_<kind>_times` / `count_missing_<kind>` names are **generated** — one
pair per registered kernel kind (see "Adding a kernel"). Callers do not hand-write
wrappers.

**The Rust simulator** is the primary caller. `simulator/src/timing/bridge`
imports `profiling.perf_api` over PyO3 and calls `get_{kind}_times(specs,
backend=, gpu_name=)`, deriving the function name with `format!("get_{kind}_times")`.
This is why `KernelKind` strings must match exactly across the boundary (below).

**Required from below** (only paid when an actual measurement happens):

- One or more **idle GPUs** (`exec/local.py:find_idle_gpus` reads `nvidia-smi`).
- The **backend's runtime libraries** — torch, FlashInfer, Triton, NCCL/NVSHMEM —
  available in the selected subprocess interpreter. These are imported **lazily
  inside the worker subprocess**, never in the simulator/main process.
- The `profilers/` timing primitives (`Timer.cupti`, `Energy.perf`, the CUPTI
  C++ extension).

`Timer.cupti`'s duration path is a two-pass GPU-active-time measurement. It
first records 10 real callable launches, computes
`ceil(min_duration_ms / estimate_mean_ms)`, then records exactly that many
launches in one uninterrupted CUPTI activity window. The default budget is
2000 ms; `min_rep` floors the formal launch count and `max_rep` caps it when the
active-time estimate requests more launches. The default cap is 50,000 launches.
By default, both passes run a
read-only reduction over a 64 MiB (or
`2 × reported L2`, whichever is larger) FP32 tensor before every logical
callable launch. The reduction's CUPTI records validate ordering but are
excluded from the callable time. This clean-line displacement avoids the dirty
writeback artifact of memset/`zero_()`; it remains a cold-ish preconditioner,
not a hardware invalidate. `Timer.cupti(clear_l2=False)` is the explicit
warm-cache diagnostic escape hatch. Keeping the formal workload in one window
is intentional: periodic CUPTI restart gaps materially change the power/clock
state of sustained GEMMs. Fixed `rep` remains the explicit median-of-three
escape hatch. This is distinct from `Energy.perf`'s 500 ms minimum
wall-clock/NVML window.

## First-class Analyzer resources

`python -m profiling run ... --output-dir X` and `... measure ... --output-dir X`
generate first-class Analyzer resources without requiring any conversation backend:
direct development runs write the same `kernel-profile.meta.json` (`kp_<uuid>`) /
`kernel-measurement.meta.json` (`km_<uuid>`) resource identity. A managed kernel
job registers with `analyzerResourceId` before any official artifact is written; the
Analyzer later reads the metadata as its discovery source of truth.

GPU provenance is owned by the execution layer (`local_worker` → `ChunkResult`):
the worker reports the physical GPU it ran on; the requested cache key keys the DB
row. When a worker actually ran, the job verifies the requested cache key and the
observed physical GPU resolve to the same catalog SKU (gpu/spec.json exact
name/aliases) and fails otherwise. A cached-only job keeps its cache key and records
`provenance.source = cache_key` without fabricating an observed GPU. The metadata
records schema version, resource id, kernel kind/table/backend, metric family, GPU
cache key + observed physical name + count, mode, created time, and artifact
declarations. `get_<kind>_times` return types and the Metrics schema are
unchanged. GPU provenance is a typed per-invocation result: the internal funnel
`profiling.db.batch.execute_profile_batch` returns a `ProfileBatchOutcome`, and
the CLI reads results + provenance from the shared `facade.run_kind_times`
entry — never a process-global side channel. The legacy `run_profile_batch`
remains a thin wrapper for existing callers.

## Directory map

```
perf_api.py        Public facade. The ONLY entry point. Holds process state
                   (DB_PATH, JIT toggle); generated query fns are attached here.
facade.py          Factory that builds the generated get_/count_ fns from the
                   registry, and the read path (query → JIT-on-miss → re-query).
cli.py             `python -m profiling` — thin human/agent wrapper over perf_api.

db/                L1b core: cache, registry, schema, scheduling.
  kind.py            `KernelKind = str`. The cross-language wire string.
  args.py            `KernelArgs` base + shared `DType`. Per-kind subclasses live
                     in kernels/, not here.
  registry.py        `(KernelKind, backend) -> KernelProfilerSpec`. The authority
                     for what kernels/backends exist and how to run each.
  table.py           One SQLite table per args schema. Owns all SQL + schema.
  batch.py           `run_profile_batch`: the single funnel for all runner
                     execution (group by backend/GPU-count → pool → insert rows).
  outlier.py         `BatchOutlierPolicy` — the per-spec `batch_outlier_policy`
                     field carried by every `KernelProfilerSpec` (placeholder).
  migrate.py         Schema version/hash + `_db_metadata`.
  metadata.py        Read-only DB metadata + per-op profiler git hashes.

kernels/           One file per kernel kind. Each declares KIND + <Kind>Args and
                   calls register(...) at import. __init__ is the barrel that
                   imports them all. Mirrors simulator/src/timing/kernels/ for
                   every kind the simulator consumes; a kind may exist here
                   alone while it is still only being measured.

runners/           L1a measurement. Subpackage per op family (gemm, attention,
  metrics.py         comm, norm, elementwise, idle). Return ComputeMetrics or
  exceptions.py      CommMetrics. Typed failures (OOMError, KernelLaunchFailed,
                     ProfilerNotImplemented) are understood by L1b.

profilers/         Low-level timing/energy primitives used by runners
                   (Timer.cupti, Energy.perf, CUPTI kernel profiler + C++ ext).
                   Also the `measure`-verb diagnostic: measure_context (the
                   Timer.cupti seam), trend (sustained capture + summary),
                   telemetry (NVML sampler + alignment), trend_plot (figures).

measure.py         `python -m profiling measure` driver: spawns the worker with a
                   measure block, collects artifacts. Cache-free; never writes DB.
                   The worker's observed physical GPU is kept as provenance
                   (never dropped) and stamped into the measurement metadata.

artifacts.py       Immutable per-job snapshots: request/results/curve/job.meta.json
                   plus the Analyzer discovery metadata files
                   kernel-profile.meta.json (kp_<uuid>) and
                   kernel-measurement.meta.json (km_<uuid>).
gpu_catalog.py     Read-only gpu/spec.json resolution (exact case-insensitive
                   name/aliases → canonical SKU); used to fail a measured job
                   whose requested cache key vs observed physical GPU mismatch, and
                   to stamp resolved canonical names into metadata. NOT on the
                   timing path.

exec/              How a runner actually runs. GpuPool/GpuChunk contracts,
  pool.py            LocalGpuPool (spawns a worker subprocess per chunk),
  local.py           the ProfileEnv registry (which interpreter), the JSON
  local_worker.py    process-boundary payload schema, and a RemoteGpuPool stub.
  env.py / payload.py / remote.py
```

## The two paths

Everything flows through `perf_api`. There are exactly two internal paths.

**Read path — `get_<kind>_times`** (`facade._get_times`):

1. Resolve `gpu_name` (explicit arg, else auto-detect via torch).
2. `find_kernel_profiler_spec(kind, backend)` → `Table(spec, DB_PATH).query(...)`.
3. Hits return `Metrics`; misses return `MissingEntry`.
4. If there are misses **and** JIT is enabled, call `run_profile_batch` to fill
   them, then re-query. With `force=True`, skip the first query and re-profile
   every spec. `count_missing` uses `Table.exists` and **never** profiles.

**Profile path — `run_profile_batch`** (`db/batch.py`), the single runner funnel:

1. Validate + coerce each spec into typed `KernelArgs` (`coerce_args`).
2. Group specs by `(backend, gpu_count)`. `gpu_count` comes from the spec's
   `gpu_count_fn` (e.g. all-reduce needs `num_gpus` real GPUs); default is 1.
3. Ask the `GpuPool` (default `LocalGpuPool`) for chunks; round-robin specs onto
   them; run chunks (threads coordinate the blocking subprocesses).
4. `LocalGpuChunk.run` writes the chunk payload to a temp JSON, sets
   `CUDA_VISIBLE_DEVICES`, applies the selected `ProfileEnv`'s ordered Python
   and shared-library paths, and spawns `python -m profiling.exec.local_worker`
   in that environment. The worker lazy-loads the registered runner via
   `RunnerRef`, executes each spec, and writes JSON results back.
5. Require every successful worker result to report its observed physical GPU,
   validate all observations against the requested cache key, then persist the
   rows through `Table.insert`. Cached-only provenance is produced only by a DB
   hit for which no worker ran.

The main/simulator process therefore **never imports torch or a runner**: the
registry holds lazy `RunnerRef`s, and the heavy import only happens in the worker
subprocess after a GPU has been reserved.

## The registry contract (the spine)

A `KernelProfilerSpec` is one row binding a `(kernel_kind, backend)` pair to
everything L1b needs: the `args_schema`, the lazy `runner_ref`, the SQLite
`table_name`, the `metric_family` (compute vs. comm), the required
`batch_outlier_policy` (a `BatchOutlierPolicy`), the `subprocess_env`, and an
optional `gpu_count_fn`. Per-kernel modules call `register(...)` at import;
`registry._ensure_loaded()` imports the `kernels` barrel once on first access,
then validates the whole set.

Three identifiers must stay equal for the cross-language facade to resolve, and
the registry validator enforces it:

```
KernelKind  ==  table_name  ==  Rust KernelSpec::KIND
   e.g.  "single_gemm"
→ Python exposes get_single_gemm_times / count_missing_single_gemm
→ Rust calls   format!("get_{kind}_times")  → "get_single_gemm_times"
```

`KernelArgs` field names are a triple contract: they are the **runner kwargs**,
the **DB key columns** (in declaration order), and the **public spec dict** keys.
`backend` and `gpu_name` are routing/identity, not args — they stay out of the
schema and are passed as separate kwargs / DB columns.

## DB shape

One table per args schema, keyed `UNIQUE(gpu_name, backend, <args...>)`. Re-profiling
the same key overwrites the measurement + provenance columns (and clears the
outlier flag); key columns are never updated. Metric columns are a table-level
contract chosen by `metric_family`:

- **compute** → `time_ms, tflops, memory_bandwidth_gbps, energy_j`
- **comm** → `time_ms, algbw_gbps, busbw_gbps, energy_j`
  (`message_size_bytes` is an **args/cache-key** column, not a measured result —
  the simulator derives moved bytes from `busbw × time`.)

## Adding a kernel

Normally one new file: `kernels/<kind>.py` declaring `KIND`, a frozen
`<Kind>Args(KernelArgs)`, and a `register(KernelProfilerSpec(...))` call (which
must pass `batch_outlier_policy=...`); then add
`from . import <kind>` to `kernels/__init__.py`. The generated `get_<kind>_times`
/ `count_missing_<kind>` appear automatically — do **not** hand-write wrappers in
`perf_api.py`. The matching runner goes under `runners/<family>/` and is wired
lazily via `RunnerRef`. Full end-to-end procedure (Python + the Rust `KernelSpec`
side): skill `top-add-kernel`; the Python registration alone is skill
`impl-register-kernel`. To fill/refresh rows for an existing kernel: skill
`operate-profile-existing-kernel`.

## CLI

`python -m profiling` is a thin wrapper over the `perf_api` facades (it never
calls runners or `run_profile_batch` directly):

```
python -m profiling list [--json]
python -m profiling query         <table> --backend <b> [--spec '{...}'] [--specs file] [--gpu-name N] [--db path]
python -m profiling count-missing <table> --backend <b> ...
python -m profiling run           <table> --backend <b> [--force] ...    # JIT-fills (or force-refreshes) then reports
python -m profiling measure       <table> --backend <b> --spec '{...}' [--output-dir DIR] [--duration-s 10] [--telemetry-hz 20] [--no-clear-l2]
```

`run` enables JIT for the call (or uses `force=True`), so it is the one CLI verb
that can launch real GPU work; `query`/`count-missing` are read-only.

Artifact-producing `run --output-dir` and `measure --output-dir` publish
`artifact.meta.json` as `kernel_profile` and `kernel_measurement`, respectively,
before GPU execution. Analyzer discovery requires that explicit kind in addition
to the resource metadata/legacy payload; it never guesses a profiling resource
from `curve.json`, `summary.json`, or managed-job metadata.

`single_gemm` uses a GEMM-local extension of the shared token axis: measured
points `m=1,2,4,8,16` precede the common `m=32..65536` grid. Dense decode must
therefore use direct/interior small-batch samples rather than extrapolating the
first `[32,64]` segment. Other token-shaped kernels retain the shared axis.

`measure` is a **cache-free diagnostic** (it never touches `profile.db`). For one
CUPTI-timed compute spec it runs a sustained ~`duration_s` capture in a single
CUPTI window, records **every** per-launch kernel duration, samples NVML telemetry
(power / SM-clock / mem-clock / util / temp / throttle) on a background thread, and
writes `runtimes.csv`, `telemetry.csv`, `summary.json`, `runtime_trend.png`, and
`runtime_telemetry.png` into `--output-dir`. It reaches the kernel's callable
through the shared `Timer.cupti` seam via a process-wide `MeasureContext` set only
for this verb (see `profilers/measure_context.py`, `profilers/trend.py`) — **no
runner changes**, and the guard is inert for every other call. `--no-clear-l2`
switches from the default cold per-launch L2-displacement (matching the `profile.db`
measurement) to a warm continuous window that surfaces sustained power/clock drift.
Kernels whose runner does not time through `Timer.cupti` (e.g. comm) are rejected.
