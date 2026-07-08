---
name: impl-register-kernel
description: >-
  Use when implementing the Python profiling registration for an VibeSim L1 kernel
  kind or backend after the orchestrator has classified the task. Covers the
  current profiling/ contracts: per-kind kernel module, co-located Args schema,
  lazy RunnerRef, BackendSupport, runner contract, tests, and no shared
  profile.db writes unless authorized. Does not cover Torch semantic reference
  design, framework research, or Rust timing/cache wiring.
---

# Impl Register Kernel

You are the implementer for the Python profiling side under `profiling/`. The
orchestrator should already have decided whether this is a new kernel kind or a
new backend for an existing kind, and should provide the intended args fields,
backend name, source/reference implementation, and expected smoke evidence.

This skill follows the current implementation, not stale design text.

## Read First

Read the local implementation before editing:

- `profiling/README.md`
- `profiling/kernels/__init__.py`
- the closest existing `profiling/kernels/<kind>.py`
- the closest runner under `profiling/runners/<family>/`
- `profiling/db/registry.py`, `profiling/db/batch.py`, and
  `profiling/exec/local_worker.py`

## Current Contracts

Concrete args schemas are co-located with the kernel module. `profiling/db/args.py`
only owns `KernelArgs` and `DType`; do not add a new concrete args dataclass
there.

For a new kind, create `profiling/kernels/<kind>.py` with `KIND`, the frozen
`<Kind>Args(KernelArgs)` dataclass, and one or more `register(KernelProfilerSpec(...))`
calls. Add the module to `profiling/kernels/__init__.py` so the registry barrel
loads it.

For a new backend of an existing kind, edit the existing `profiling/kernels/<kind>.py`
and add another `register(...)` row that reuses the existing args schema and
table. Do not create a new kind just to hold a backend.

Keep these invariants:

- `kernel_kind == table_name == KIND`;
- `backend` is routing metadata, not a `KernelArgs` field and not a runner kwarg;
- `KernelArgs` field names and order match the public spec dict, DB key columns,
  and runner kwargs;
- registry rows use lazy `RunnerRef`;
- every production row declares `BackendSupport`, `MetricFamily`, and
  `BatchOutlierPolicy`;
- `BackendSupport` must match evidence from source exploration and runnable
  correctness checks: compute dtype, KV/cache dtype if relevant, and GPU or
  architecture gates. Do not mark unsupported or untested dtype/GPU combinations
  as supported.
- special Python environments go through `KernelProfilerSpec.subprocess_env` and
  `profiling.exec.env`, not runner-local environment mutation;
- `perf_api.py` gets no hand-written `get_<kind>_times` or
  `count_missing_<kind>` wrappers.

## Runner Contract

Runner files allocate tensors, invoke/profile kernels, and return metrics. They
do not write SQL, insert DB rows, choose GPUs, or mutate `CUDA_VISIBLE_DEVICES`.

For ordinary compute kernels, expose a single-spec function:

```python
def profile_<kind>(**schema_kwargs) -> ComputeMetrics: ...
```

`KernelProfilerSpec.load_list_runner()` wraps it with `profiling.runners.batched`.
For backend-specific shared wrappers, expose one registered entry function per
backend, as attention does, because the worker strips `backend` before invoking
the runner.

For list-native multi-GPU comm kernels, expose:

```python
def profile_<kind>_batch(kwargs_list: list[dict]) -> list[RunnerResult]: ...
```

Set `list_native=True` and provide `gpu_count_fn` in the registry row. The batch
function should spawn the rank group once and return one `RunnerResult` per input
spec in order.

Heavy framework imports belong inside the runner function or worker-only path.
Importing `profiling.kernels.<kind>` must not import torch, CUDA libraries, or
the runner module.

## Timing And Metrics

Choose the timing method by mirroring the closest current runner family. Current
compute runners generally use `Timer.cupti` for kernel-only timing and
`Energy.perf(..., per_iter_time_ms=time_ms)` for energy. Use another timer only
when a nearby current runner or an explicit task requirement justifies it.

Return `ComputeMetrics` for `MetricFamily.COMPUTE` and `CommMetrics` for
`MetricFamily.COMM`. Keep metric formulas close to the existing family runner,
and state any approximation in the task report.

## Tests And Smoke

Add focused tests matching existing `tests/test_<kind>.py` style:

- import the per-kernel module directly;
- check args field order and dtype coercion where relevant;
- check `KIND` and registry spec shape;
- check lazy import behavior in a subprocess;
- for new backends, check the backend appears in registry metadata and shares
  the existing args schema/table;
- for list-native comm, check `list_native`, `gpu_count_fn`, and batch function
  routing.

Verify generated metadata without hand-written facades:

```bash
uv run python -m profiling list --json
uv run python -c "from profiling import perf_api; print(hasattr(perf_api, 'get_<kind>_times'))"
```

Then run a real public-entry smoke through the profiling CLI on a representative
spec:

```bash
uv run python -m profiling run <kind> --backend <backend> --db /tmp/<kind>_<backend>_smoke.db --spec '<json spec>' --json
```

This must invoke the registered runner through `profiling.perf_api`, the table
layer, and the execution backend. Do not call the runner directly as the only
smoke.

Do not write the shared `profiling/profile.db` unless the orchestrator or user
explicitly authorized it. Use a temporary DB path for smoke runs and include the
exact command/output in the report. If no compatible GPU or backend environment
is available, report that as a verification blocker; do not claim the backend is
fully verified.

## Report Back

Return the exact files changed, the registry row(s) added, the runner entry
function(s), commands run, smoke/test results, environment assumptions, and any
unsupported dtype/GPU/shape cases. Also include a **Python-to-Rust handoff**
section for the later role that will add the Rust `KernelSpec` / bridge/cache
wiring. This is not Rust implementation work; it is a short list of facts the
Python work established: `KIND` / facade stem, backend strings, `KernelArgs`
fields and order, metric family, dtype/GPU capability axes, a representative
smoke spec, and shape or dtype constraints.
