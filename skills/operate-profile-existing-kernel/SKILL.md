---
name: operate-profile-existing-kernel
description: >-
  Query, fill, refresh, validate, or measure a registered VibeSim L1 kernel
  profile, including DB-row provenance and preservation.
---

# Profile Run Existing Kernel

Use this skill only when the requested profiler already exists in
`profiling.db.registry.REGISTRY`. If `launcher kernel-profile list` does not show
the requested table/backend pair, stop and switch to the add-kernel workflow or
ask before changing code.

## Required Docs

Read these before running or editing:

- `AGENTS.md`, especially L1 rules.
- `doc/detailed_design/L1.md` — the L1 layer overview.
- `profiling/README.md` — the L1 `profiling/` directory map.
- `README.md` Python/uv quick-start.

## Exact Interface

The supported CLI is:

```bash
uv run python -m launcher kernel-profile list [--json]
uv run python -m launcher kernel-profile count-missing <table> --backend <backend> (--spec JSON | --specs PATH) [--gpu-name NAME] [--db PATH] [--json]
uv run python -m launcher kernel-profile query <table> --backend <backend> (--spec JSON | --specs PATH) [--gpu-name NAME] [--db PATH] [--json]
uv run python -m launcher kernel-profile run <table> --backend <backend> (--spec JSON | --specs PATH) [--force] [--output-dir DIR] [--gpu-name NAME] [--db PATH] [--json]
```

`python -m profiling ...` remains a compatibility/developer entry and calls the
same `profiling.cli` implementation. Skills and managed Agent runs use the
launcher form so simulation, timing prediction, alignment, and kernel profiling
share one VibeSim command surface.

Spec input rules:

- `--spec` is one JSON object and may be repeated.
- `--specs` accepts a JSON object, a JSON list, `{"specs": [...]}`, or JSONL.
- Specs contain only `KernelArgs` fields. Pass backend with `--backend`, not in
  each spec.
- One CLI command handles one table/backend pair. Do not mix tables or backends
  in one invocation.

Important semantics:

- `query` is read-only and never profiles.
- `query` exits 0 for a valid request even when rows are absent; missing rows
  appear in JSON as `status: "missing"` and count toward `missing_count`.
- `count-missing` is read-only and never profiles.
- `run` without `--force` enables JIT-fill through `perf_api`: DB hits are read,
  missing specs are profiled, then results are queried.
- `run --force` refreshes every provided spec through `perf_api` even if rows
  already exist.
- `run --output-dir DIR` writes immutable `request.json`, `results.json`,
  `curve.json`, and `job.meta.json`. A managed Agent run requires this option;
  put it below the workspace `logs/` root. Direct development calls may omit it
  when no durable visualization artifact is wanted.
- `--gpu-name` is the DB key/filter used by `perf_api`; it is not a CUDA device
  selector. The configured GPU pool establishes worker device visibility; the
  standard CLI does not expose a `--gpus` hardware-selection flag.
- The CLI is a wrapper over generated `perf_api` functions. Do not call runners
  or `run_profile_batch` directly for this workflow.
- A fifth verb, `measure`, is **not** a cache operation — it is a cache-free
  NVML/CUPTI telemetry diagnostic. See "The `measure` diagnostic" below.

### Row provenance and production equivalence

A row is valid for a consumer only when its profiler contract matches the
production lookup: backend specialization, algorithm, layout and page contract,
numeric and scale contract, exact shape, GPU key, and profiler revision. A
shared operation name does not make ragged, paged, calibration, cache-free, or
different quantization paths interchangeable.

Record these fields with the spec set and verify them before calling a row a DB
hit for the target experiment. If provenance is unavailable, report the row as
unverified rather than inferring equivalence from table/backend alone.

For framework and specialized compute rows, verify that the timed boundary is
the production public callable and that `Timer.cupti` sums one logical
invocation: a stable single kernel may use a name filter, while a compound
callable must include its full launch sequence with `kernel_name=None`. Preserve
the documented `Timer.do_bench` path for simple Torch GEMM. CUDA-event timing is
diagnostic, not DB evidence. Allocation, compile, correctness, and
synchronization must be outside the timed closure.

Inspect input generation before trusting a row for a data-sensitive algorithm.
The synthetic workload must exercise the same planner, histogram/radix path, or
memory-locality regime as production, with deterministic construction outside
timing and an independent correctness oracle. Do not refresh a row merely to
bless a pathological input or a relaxed tolerance.

### Preserve rows written in a worktree

`profiling/profile.db` is tracked, but a worktree may hide its local changes with
`skip-worktree`. Check before writing the shared database:

```bash
git ls-files -v profiling/profile.db
```

An `S` prefix means new profile rows may not appear in `git status`. Record the
worktree and database path that receive the rows. If the rows must be preserved,
explicitly transfer that database to the checkout responsible for keeping it.
Stage or commit it only when the user requests that action. A clean `git status`
does not prove the rows were saved.

### Automatic work submission

For one homogeneous table/backend/GPU-count request, generate the complete
intended spec set and submit it in one public `profiling run --specs ...` call.
Do not manually split that set across CLI calls for concurrency, GPU selection,
JIT amortization, memory heuristics, or failure localization.

`run_profile_batch` and the configured GPU pool own idle-device discovery and
reservation, work distribution, concurrent worker processes, and per-worker
multi-spec execution. The implementation anchors are `profiling/db/batch.py`
and `profiling/exec/local.py`; callers should use the public CLI rather than
reimplementing this scheduling. After the complete submission, retry only specs
reported missing or errored.

Manually shard only when the user explicitly requests it, a retained-allocation
or OOM issue has been diagnosed, or an execution backend has a documented
payload limit. Report the exception and its evidence.

### Do not limit how many GPUs the run may use

**Do not restrict the device set unless there is a concrete reason to.** L1
already does idle-GPU detection and arrangement automatically: `find_idle_gpus`
(`profiling/exec/local.py`) reads `nvidia-smi` and passes only genuinely idle
devices to `LocalGpuPool`, which then chunks the spec set across them and runs
the chunks concurrently. Launching with `CUDA_VISIBLE_DEVICES=<one idle gpu>`,
or forcing `VIBESIM_PROFILE_GPUS`, only *shrinks* what L1 gets to choose from:
it serializes a batch that would have gone parallel, and it hard-fails
(`need N idle GPU(s), found M`) when the pinned card turns out to be busy or
when a multi-GPU spec needs more devices than you left visible. Issue the
command bare.

Narrow the device set only when the user explicitly asks, when the run must stay
off specific cards, or when a diagnosed issue requires it — and then say so in
the report. `VIBESIM_PROFILE_GPUS` in particular *bypasses* the idle guard and
will profile on top of someone else's job, so use it only for GPUs you know are
yours. Note again that `--gpu-name` is a profile.db key/filter, **not** a device
selector — it neither adds nor removes hardware from the pool.

## The `measure` diagnostic (NVML telemetry, out of the cache path)

`launcher kernel-profile` has a fifth verb, `measure`, that the four cache verbs
above do not cover. It is the sustained per-launch CUPTI trend + NVML telemetry
instrument (the ~10 s window): run it to inspect power / SM-clock / mem-clock /
util / temp / throttle drift for one kernel spec, **not** to fill or refresh
`profile.db`.

```bash
uv run python -m launcher kernel-profile measure <table> --backend <backend> --spec JSON \
  [--output-dir DIR] [--duration-s 10] [--telemetry-hz 20] [--no-clear-l2] [--gpu-name NAME] [--db PATH] [--json]
```

How it differs from the cache verbs:

- **Cache-free.** It never reads or writes `profile.db`; no row is stored. The
  Validation and Required-User-Report checklists below are about DB rows and do
  not apply — report the artifact paths and telemetry summary instead.
- **Exactly one spec.** `measure` rejects a multi-spec batch.
- Runs a single CUPTI window of `--duration-s` (default 10 s), records **every**
  per-launch kernel duration, samples NVML telemetry on a background thread at
  `--telemetry-hz` (default 20 Hz), and writes `runtimes.csv`, `telemetry.csv`,
  `summary.json`, and two `.png` plots to `--output-dir` (default
  `./measure_<table>_<backend>`).
- `--no-clear-l2` switches from the default cold per-launch L2 displacement
  (which matches the `profile.db` measurement) to a warm continuous window that
  surfaces sustained power/clock drift.
- Only kernels whose runner times through `Timer.cupti` are accepted; comm
  kernels are rejected.

For the full mechanism — the `Timer.cupti` two-pass timing, the `Energy.perf`
NVML energy window that feeds the recorded `energy_j` column, and the `measure`
output schema — see `profiling/README.md` (the `Timer.cupti` / `Energy.perf`
paragraph and the `measure` paragraph). This skill covers running the
diagnostic, not editing it.

## Workflow Checklist

Copy this checklist before starting. Keep each item as `[]`; change to `[x]`
only after doing the exact action, or write `N/A: reason`.

- [] Read the required docs listed above.
- [] Run `uv run python -m launcher kernel-profile list --json` and confirm the requested
  `<table>` and `--backend` exist.
- [] Record the listed `kernel_kind`, `args`, `metric_family`,
  `subprocess_env`, generated `get_fn`, and generated `count_fn`.
- [] For compute profiling, verify the public callable, CUPTI filter/sum
  boundary, untimed setup/correctness, and representative input distribution.
- [] Build the complete intended spec set. Confirm every spec has exactly the
  listed `args` fields and no routing-only fields such as `backend`.
- [] For a homogeneous multi-spec profiling request, write the complete set to
  one `--specs` input and record its count. Do not pre-shard it into multiple
  `run` calls.
- [] For a single spec, use `--spec`; for a multi-spec set, choose JSON list,
  `{"specs": [...]}`, or JSONL as the one `--specs` input.
- [] Decide DB path. Use a task-scoped DB under `$TMPDIR` for
  validation-only runs; use the
  default/shared DB only when the user explicitly wants to update it.
- [] Decide whether `--gpu-name` is needed. Use it for a known DB key or to
  avoid CUDA name auto-resolution. Do not describe it as selecting a physical
  GPU.
- [] Confirm the command does not narrow the device set (no
  `CUDA_VISIBLE_DEVICES`, no `VIBESIM_PROFILE_GPUS`) — L1 detects and arranges
  idle GPUs itself. If it does, record the concrete reason.
- [] Choose the command mode: `count-missing`, `query`, `run`, or `run --force`.
- [] For `run` / `run --force`, choose a fresh `--output-dir` below `logs/` when
  the result must appear in the managed UI. Never reuse an artifact directory
  containing an existing immutable snapshot.
- [] For a real profiling run, first confirm CUDA availability with an explicit
  environment check such as `uv run python -c "import torch; print(torch.cuda.is_available())"`.
- [] If `uv run` warns that `VIRTUAL_ENV` points at another workspace, note it
  as an environment caveat. It is usually harmless because `uv` still uses this
  repo's project environment, but do not hide it in the report.
- [] Run the CLI with `--json` unless the user asked for human table output.
- [] Capture the command exit code. For `run`, nonzero means rows remained
  missing or the command failed; do not treat that as success.
- [] After the complete `run` submission, retry only specs reported missing or
  errored. If manual sharding was required, record the allowed exception and
  supporting evidence.
- [] For `run` or `run --force`, run `count-missing` again on the same
  table/backend/specs/DB/gpu-name to verify persistence.
- [] For `query` or completed `run`, verify the JSON result count equals the
  spec count and each result has `status: "ok"` unless reporting misses.
- [] Sanity-check metrics by family: compute rows have `time_ms`, `tflops`,
  `memory_bandwidth_gbps`, `energy_j`; comm rows have `time_ms`, `algbw_gbps`,
  `busbw_gbps`, `message_size_bytes`, `energy_j`.
- [] For real profiling output, include the input `KernelArgs` fields in the
  metrics table beside the returned metrics. A reader should not have to
  cross-reference the command to know which shape produced a row.
- [] For real profiling output, report the detected or requested GPU DB key.
  If `--gpu-name` was omitted and the CLI output does not expose the resolved
  name, say that explicitly instead of inventing one.
- [] For compute rows, check `time_ms > 0`, `energy_j >= 0`, and optionally
  report implied watts as `energy_j / (time_ms / 1000)`.
- [] Report to the user using the required report format below.
- [] For a snapshotted run, inspect `curve.json`: varying axes follow
  `KernelArgs` declaration order, constants are in `fixedArgs`, the first two
  axes are x/y, and remaining axes are facets.

## Validation Checklist

Copy this checklist when the task involves more than a read-only query. Mark
items as `[x]` only when complete.

- [] `list --json` showed the intended table/backend pair.
- [] The spec count used by the CLI matches the complete intended spec set.
- [] One complete public `run --specs` submission was used for a homogeneous
  multi-spec request, or an allowed manual-sharding exception is documented
  with evidence.
- [] `count-missing` was run before profiling, or skipped with a stated reason.
- [] `run` or `run --force` completed with exit code 0.
- [] Any retry contained only specs previously reported missing or errored.
- [] Post-run `count-missing` returned 0, or every remaining miss is listed.
- [] `query` or run output returned one result per input spec.
- [] No result has `status: "missing"` unless the final answer explicitly
  reports the miss.
- [] If code or docs were changed while supporting the run, `uv run ruff check`
  and focused pytest were run for those changed files.

## Required User Report

Before interpreting a snapshotted managed result, read
`skills/operate-use-analyzer/SKILL.md`. Use the stable Analyzer profile or
measurement resource ID to read its descriptor, curve/summary, declared plots,
and hardware limits. The job row and CLI summary are lifecycle/provenance only.
Kernel profile and measurement reads return a compact result plus a `citations`
map keyed by the metric, panel, or plot represented in that result. Copy the
matching complete `kprof.*` or `kmeasure.*` token unchanged as Markdown inline
code beside the supported claim so the frontend can resolve it to that typed
resource. Never derive a token from the table, backend, axes, metric, plot, or
resource ID.

Always report:

- Instruction files used: include concrete paths, especially `AGENTS.md` and
  `skills/operate-profile-existing-kernel/SKILL.md`.
- Docs checked: concrete doc paths/sections.
- Command(s): exact CLI command(s), including `--db`, `--gpu-name`, and
  `--force` when used.
- Scope: table, backend, spec count, DB path, and whether DB path was temporary
  or shared.
- Submission: confirm one complete `run --specs` call, or report the allowed
  manual-sharding exception and evidence; list any missing/error-only retry.
- GPU context: requested `--gpu-name`, resolved GPU DB key when available, and
  whether the command relied on CUDA device-name auto-resolution. State that the
  device set was left to L1's idle-GPU detection, or report the narrowing and
  its concrete reason.
- Mode: read-only query, count-missing, JIT-fill, or force refresh.
- Result: missing count before/after when available.
- Metrics: compact table of returned metrics, or JSON status summary for large
  batches. For small batches, include the input `KernelArgs` columns next to
  `status` and metric columns, e.g. `m`, `n`, `k`, `dtype`, `status`,
  `time_ms`, `tflops`, `energy_j` for `single_gemm`.
- Validation: commands run and pass/fail status.
- Artifacts: `--output-dir`, `request.json`, `results.json`, and `curve.json`
  when the run was snapshotted; identify the plotted axes and fixed args.
- Caveats: unresolved missing rows, no GPU available, shared DB not updated, or
  any reason a command was not run.

## Examples

Single-spec run:

```bash
uv run python -m launcher kernel-profile run single_gemm --backend torch --force \
  --db "$TMPDIR/single-gemm/profile.db" \
  --output-dir logs/20260731_0_single_gemm_profile --json \
  --spec '{"m":128,"n":8192,"k":8192,"dtype":"bf16"}'
```

Complete JSON/JSONL spec set:

```bash
uv run python -m launcher kernel-profile run single_gemm --backend torch --gpu-name "H100" --specs specs.json --db "$TMPDIR/single-gemm/profile.db" --output-dir logs/20260731_0_single_gemm_profile --json
uv run python -m launcher kernel-profile count-missing single_gemm --backend torch --gpu-name "H100" --specs specs.json --db "$TMPDIR/single-gemm/profile.db" --json
uv run python -m launcher kernel-profile query single_gemm --backend torch --gpu-name "H100" --specs specs.jsonl --db "$TMPDIR/single-gemm/profile.db" --json
```
