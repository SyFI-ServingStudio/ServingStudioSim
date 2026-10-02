# ROCm rocpd capture — 1-GPU plumbing validation

Goal: confirm, on **one** MI300X with a **small** model, that the AMD alignment
capture path is wired end to end — that a rocprofv3 run records the
`vllm_iteration(N): <phase>` roctx ranges emitted by the iteration shim *and* the
GPU kernel dispatches, and that the offline producer turns that rocpd database
into the normalized Check-1 artifacts. This validates plumbing only; it is not a
timing or accuracy result, and it is deliberately not GLM (GLM-5.3-Flash needs
tensor parallelism `TP >= 2`, which this single-GPU check avoids).

All cluster-specific values are placeholders: `<rocprofv3>` (absolute path to the
rocprofv3 binary), `<FORK_PYTHON>` (absolute path to the ROCm vLLM venv's
`python`), `<MODEL>` (see below), `<OUT_DIR>` (a writable scratch directory).

## Model

Serve **`Qwen/Qwen2.5-0.5B-Instruct`** — a widely available, ungated ~0.5 B dense
decoder that loads on a single MI300X in bf16 with room to spare and needs no
tensor parallelism. (`facebook/opt-125m` is an even smaller ungated fallback if
the Qwen weights are not already cached.) Any single-GPU model works; the point
is the trace shape, not the model.

## Preconditions (do these first, on the GPU host)

1. **The shim plugin must be discoverable in the serving venv.** vLLM auto-loads
   and calls every `vllm.general_plugins` entry point in each worker process.
   This package registers `vibesim_roctx_shim = alignment.profiler.roctx_shim:install_vllm_roctx_shim`.
   Confirm the ROCm vLLM venv sees it:

   ```bash
   <FORK_PYTHON> -c "from importlib.metadata import entry_points; \
     print([e.name for e in entry_points(group='vllm.general_plugins')])"
   ```

   The list must contain `vibesim_roctx_shim`. If it does not, make this package
   installed/importable in that venv (e.g. install it editable into the venv so
   its entry-point metadata is present) and re-run the check. No source edit to
   vLLM is needed — discovery is entirely through this entry point.

2. **A roctx backend must be loadable.** The shim prefers the `libroctx64` C API
   and falls back to ROCm PyTorch's roctx-retargeted `torch.cuda.nvtx`. Either is
   fine; `--marker-trace` records what it pushes.

## Capture command

```bash
VLLM_ROCTX_SCOPES_FOR_PROFILING=1 \
uv run --no-sync python -m alignment rocpd-capture \
  --rocprof <rocprofv3> \
  --output-dir <OUT_DIR> \
  --output-name rank0 \
  --iteration-start 2 \
  --iteration-end 20 \
  --parsed-output <OUT_DIR>/parsed.json \
  --sequences-output <OUT_DIR>/kernel_sequences.json \
  -- \
  <FORK_PYTHON> -m vllm.entrypoints.cli.main serve <MODEL> \
    --enforce-eager \
    --tensor-parallel-size 1 \
    --max-num-batched-tokens 2048 \
    --gpu-memory-utilization 0.85 \
    --port 8000
```

Notes:
- `rocpd-capture` **also** injects `VLLM_ROCTX_SCOPES_FOR_PROFILING=1` into the
  launched server's environment itself (that is the fix this validates), so the
  leading env assignment above is belt-and-suspenders — it makes the intent
  explicit and covers a manually invoked server. Do **not** set `VLLM_PLUGINS`;
  leaving it unset lets vLLM load all general plugins including ours. If your
  environment already sets `VLLM_PLUGINS`, it must include `vibesim_roctx_shim`.
- `--enforce-eager` keeps every kernel a normal launch (no HIP graphs); the
  eager kernel-only path is what Check-1 attributes today.
- `rocpd-capture` wraps the server process and captures until it exits. Drive a
  short, bounded workload against `http://127.0.0.1:8000/v1/...` (a handful of
  completion requests of ~64 output tokens each is plenty — it only needs to
  produce a few dozen decode iterations), then stop the server so rocprofv3
  finalizes the database. `--iteration-start/-end` trims warmup/teardown from the
  parsed window. If an online server is more friction than it is worth for a pure
  plumbing check, wrap an offline `LLM(...).generate([...])` one-liner with
  `<FORK_PYTHON>` instead of `serve`; the shim patches the same
  `GPUModelRunner.execute_model`, so both produce the identical iteration ranges
  and exit on their own.

On success the command prints one JSON line and exits 0, e.g.:

```json
{"rocpd":"<OUT_DIR>/rank0_results.db","validation":{"iteration_ranges":19,"kernel_rows":NNNN,"iteration_span_ms":MM.M,"ok":true}}
```

`ok:true` already means both roctx iteration ranges and kernel-dispatch rows are
present (it runs the real C0 readers). The SQL below is for independent manual
confirmation.

## rocpd verification queries

rocprofv3 writes `<OUT_DIR>/rank0_results.db` for `--output-format rocpd` (older
builds: `rank0.db`). Open it with `sqlite3`. Table names may carry a
`_<session-guid>` suffix (e.g. `rocpd_region_0000..._...`); discover the exact
names first and substitute them into the queries:

```sql
SELECT name FROM sqlite_master WHERE type IN ('table','view')
  AND (name LIKE '%region%' OR name LIKE '%kernel_dispatch%'
       OR name LIKE '%kernel_symbol%' OR name LIKE '%string%');
```

(a) **Iteration roctx ranges exist with the label text intact** — expect one row
per captured forward, each `vllm_iteration(N): forward`:

```sql
SELECT s.string AS region_name, r.start, r.end
FROM rocpd_region r
JOIN rocpd_string s ON s.id = r.name_id
WHERE s.string LIKE 'vllm\_iteration(%): %' ESCAPE '\'
ORDER BY r.start
LIMIT 25;
```

(b) **Kernel-dispatch rows exist** — expect a non-zero count and recognizable
ROCm kernel names (hipBLASLt GEMMs, flash-attention, rmsnorm, SiLU, elementwise):

```sql
SELECT COUNT(*) AS n_dispatches FROM rocpd_kernel_dispatch;

SELECT k.display_name, d.start, d.end
FROM rocpd_kernel_dispatch d
JOIN rocpd_info_kernel_symbol k ON k.id = d.kernel_id
ORDER BY d.start
LIMIT 10;
```

(c) **The actual rocpd filename rocprofv3 wrote** — confirm the on-disk name so a
version bump in the `_results.db` vs `.db` spelling is caught:

```bash
ls -1 <OUT_DIR>/*.db
```

Both (a) and (b) returning rows means dispatches have iteration windows to be
owned by. If (a) is empty, the shim did not fire (entry point not discovered in
the serving venv, env gate off, or `--marker-trace` dropped); if (b) is empty,
`--kernel-trace` recorded no GPU work.

## Parse step and success criterion

The capture command above already chains the parse (omit `--no-parse`). To run it
standalone against an existing database:

```bash
uv run --no-sync python -m alignment rocpd-parse \
  --db <OUT_DIR>/rank0_results.db \
  --iteration-start 2 --iteration-end 20 \
  --output <OUT_DIR>/parsed.json \
  --sequences-output <OUT_DIR>/kernel_sequences.json
```

**Success** means all three artifacts are written and non-empty:
- `<OUT_DIR>/parsed.json` — normalized Check-1 iterations with their owned kernels;
- `<OUT_DIR>/parsed.kernels.parquet` — the per-kernel rows (sibling of
  `parsed.json`), every column non-null;
- `<OUT_DIR>/kernel_sequences.json` — the folded label-ready kernel-sequence
  catalog, with at least one sequence whose kernels are the dispatches that fell
  inside a `vllm_iteration(N): forward` window.

If the queries in (a)/(b) returned rows but the parse produces empty sequences,
the problem is in attribution (timestamp containment / thread matching), not in
capture plumbing — report that distinction rather than re-running the capture.
