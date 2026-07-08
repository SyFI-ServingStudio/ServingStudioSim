---
name: operate-profile-existing-kernel
description: Use when asked to list, query, count missing rows for, JIT-fill, force-refresh, or validate an existing VibeSim L1 profiler entry through `uv run python -m profiling`. Applies only to registered KernelProfilerSpec table/backend pairs and batched specs.
---

# Profile Run Existing Kernel

Use this skill only when the requested profiler already exists in
`profiling.db.registry.REGISTRY`. If `python -m profiling list` does not show
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
uv run python -m profiling list [--json]
uv run python -m profiling count-missing <table> --backend <backend> (--spec JSON | --specs PATH) [--gpu-name NAME] [--db PATH] [--json]
uv run python -m profiling query <table> --backend <backend> (--spec JSON | --specs PATH) [--gpu-name NAME] [--db PATH] [--json]
uv run python -m profiling run <table> --backend <backend> (--spec JSON | --specs PATH) [--force] [--gpu-name NAME] [--db PATH] [--json]
```

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
- `--gpu-name` is the DB key/filter used by `perf_api`; it is not a CUDA device
  selector. The standard CLI does not expose a `--gpus` hardware-selection flag.
- The CLI is a wrapper over generated `perf_api` functions. Do not call runners
  or `run_profile_batch` directly for this workflow.

## Workflow Checklist

Copy this checklist before starting. Keep each item as `[]`; change to `[x]`
only after doing the exact action, or write `N/A: reason`.

- [] Read the required docs listed above.
- [] Run `uv run python -m profiling list --json` and confirm the requested
  `<table>` and `--backend` exist.
- [] Record the listed `kernel_kind`, `args`, `metric_family`,
  `subprocess_env`, generated `get_fn`, and generated `count_fn`.
- [] Build the spec batch. Confirm every spec has exactly the listed `args`
  fields and no routing-only fields such as `backend`.
- [] Decide spec input form: repeated `--spec`, JSON list, `{"specs": [...]}`,
  or JSONL file. Record the spec count.
- [] Decide DB path. Use `--db /tmp/...` for validation-only runs; use the
  default/shared DB only when the user explicitly wants to update it.
- [] Decide whether `--gpu-name` is needed. Use it for a known DB key or to
  avoid CUDA name auto-resolution. Do not describe it as selecting a physical
  GPU.
- [] Choose the command mode: `count-missing`, `query`, `run`, or `run --force`.
- [] For a real profiling run, first confirm CUDA availability with an explicit
  environment check such as `uv run python -c "import torch; print(torch.cuda.is_available())"`.
- [] If `uv run` warns that `VIRTUAL_ENV` points at another workspace, note it
  as an environment caveat. It is usually harmless because `uv` still uses this
  repo's project environment, but do not hide it in the report.
- [] Run the CLI with `--json` unless the user asked for human table output.
- [] Capture the command exit code. For `run`, nonzero means rows remained
  missing or the command failed; do not treat that as success.
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

## Validation Checklist

Copy this checklist when the task involves more than a read-only query. Mark
items as `[x]` only when complete.

- [] `list --json` showed the intended table/backend pair.
- [] The spec count used by the CLI matches the intended batch size.
- [] `count-missing` was run before profiling, or skipped with a stated reason.
- [] `run` or `run --force` completed with exit code 0.
- [] Post-run `count-missing` returned 0, or every remaining miss is listed.
- [] `query` or run output returned one result per input spec.
- [] No result has `status: "missing"` unless the final answer explicitly
  reports the miss.
- [] If code or docs were changed while supporting the run, `uv run ruff check`
  and focused pytest were run for those changed files.

## Required User Report

Always report:

- Instruction files used: include concrete paths, especially `AGENTS.md` and
  `skills/operate-profile-existing-kernel/SKILL.md`.
- Docs checked: concrete doc paths/sections.
- Command(s): exact CLI command(s), including `--db`, `--gpu-name`, and
  `--force` when used.
- Scope: table, backend, spec count, DB path, and whether DB path was temporary
  or shared.
- GPU context: requested `--gpu-name`, resolved GPU DB key when available, and
  whether the command relied on CUDA device-name auto-resolution.
- Mode: read-only query, count-missing, JIT-fill, or force refresh.
- Result: missing count before/after when available.
- Metrics: compact table of returned metrics, or JSON status summary for large
  batches. For small batches, include the input `KernelArgs` columns next to
  `status` and metric columns, e.g. `m`, `n`, `k`, `dtype`, `status`,
  `time_ms`, `tflops`, `energy_j` for `single_gemm`.
- Validation: commands run and pass/fail status.
- Caveats: unresolved missing rows, no GPU available, shared DB not updated, or
  any reason a command was not run.

## Examples

Repeated inline specs:

```bash
uv run python -m profiling run single_gemm --backend torch --force --db /tmp/profile.db --json \
  --spec '{"m":128,"n":8192,"k":8192,"dtype":"bf16"}' \
  --spec '{"m":256,"n":8192,"k":8192,"dtype":"bf16"}'
```

JSON/JSONL file batch:

```bash
uv run python -m profiling count-missing single_gemm --backend torch --gpu-name "H100" --specs specs.json --db /tmp/profile.db --json
uv run python -m profiling query single_gemm --backend torch --gpu-name "H100" --specs specs.jsonl --db /tmp/profile.db --json
```
