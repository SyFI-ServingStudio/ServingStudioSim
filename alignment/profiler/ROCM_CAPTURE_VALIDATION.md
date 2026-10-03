# ROCm rocpd capture — 1-GPU plumbing validation

Goal: confirm, on **one** MI300X with a **small** model, that the AMD alignment
capture path is wired end to end — that a rocprofv3 run records the iteration
boundaries emitted by the iteration shim *and* the GPU kernel dispatches, and
that the offline producer turns that rocpd database into the normalized Check-1
artifacts. This validates plumbing only; it is not a timing or accuracy result,
and it is deliberately not GLM (GLM-5.3-Flash needs tensor parallelism
`TP >= 2`, which this single-GPU check avoids).

## Iteration-boundary mechanism — Option B (sentinel kernels) is now primary

Two mechanisms mark iteration boundaries in the rocpd capture; the reader accepts
either, **roctx-first**:

1. **roctx ranges** (`vllm_iteration(N): <phase>`) via `--marker-trace`. This is
   the NVIDIA-parity path and stays primary *when it records*.
2. **Sentinel kernels** (Option B) via `--kernel-trace`. The shim launches a tiny,
   uniquely-named no-op Triton kernel — `vibesim_sentinel` — as the first GPU work
   of each forward. It lands in rocpd's `rocpd_kernel_dispatch` table, and the
   reader reconstructs iteration ranges from the sentinel dispatches' timestamps
   (sentinel `N`'s launch starts iteration `N`; the next sentinel ends it).

**Why Option B is the mechanism now.** On the pinned ROCm-7.2 / rocprofv3-1.3.2
stack, roctx **marker** recording is confirmed broken: the SDK MARKER service
never registers inside a torch-loaded process (two independent load-order
attempts — the `sitecustomize` `RTLD_GLOBAL` SDK-roctx preload, and forcing the
soname — both yielded **0** `region` rows). Kernel-dispatch capture via
`--kernel-trace` is 100% reliable (23,711 rows every run). So the sentinel kernel
is the dependable boundary signal on this stack. The roctx push is kept too: it is
harmless and lets the same code align on a stack that *does* record roctx (the
reader uses roctx when present, sentinels otherwise).

**Sentinel identity and iteration encoding (how the reader finds and orders
them).** rocpd stores a Triton JIT kernel's Python function name verbatim in the
`rocpd_info_kernel_symbol` table (`display_name` / `kernel_name`), joined to each
dispatch by `kernel_id`. The reader (`alignment/rocpd/evidence.py::is_sentinel_
dispatch`) marks a dispatch as a sentinel iff that name contains `vibesim_sentinel`
— a string no torch/aiter kernel carries, so it cannot collide. The absolute
iteration index rides in the dispatch's **`grid_size_y`** column: the shim launches
the sentinel with grid `(1, iteration + 1, 1)` and `num_warps=1`, so
`grid_size_y == iteration + 1` while every real kernel's `grid_size_y` is ~1. The
reader recovers `iteration = grid_size_y - 1` when that encoding is present and
strictly increasing, and otherwise falls back to the sentinel's ordinal position
in the dispatch stream (always correct for consecutive forwards). `grid_size_y`
(not `x`) carries it because the x-grid is routinely huge on real kernels and the
index stays tiny — well under the HIP 65535 y/z block-count ceiling. **Sentinel
dispatches are markers, not model work: the producer excludes them from the
attributed and compared kernel set in both the roctx and sentinel paths**, so they
never appear in `parsed.kernels.parquet` or `kernel_sequences.json`.

If Triton is somehow unavailable in the serving venv, the shim falls back to a
plain torch GPU op to keep the forward alive and still push roctx, but that
fallback carries neither the collision-free name nor the grid-encoded ordinal (a
torch op's kernel name and launch grid are not ours to set, and HIP caps the y/z
grid dims at 65535 blocks), so it is **not** offline-reconstructable — prefer
Triton, which is present in the vLLM ROCm container.

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

2. **A roctx backend must be loadable, and the SDK roctx lib must win the load
   order.** The shim prefers the rocprofiler-sdk roctx C API
   (`librocprofiler-sdk-roctx.so.1`) and falls back, in order, to legacy
   `libroctx64` then ROCm PyTorch's roctx-retargeted `torch.cuda.nvtx`. On the
   pinned ROCm-7 / rocprofv3-1.3.2 stack `--marker-trace` records roctx ranges
   **only** from `librocprofiler-sdk-roctx`, so that soname is tried first (see
   `roctx_shim.RoctracerBackend`).

   The confirmed GPU failure mode: vLLM's `torch`/`aiter` load legacy
   `libroctx64.so` on import, and that resident legacy library preempts
   rocprofv3's SDK MARKER service, so `--marker-trace` records **zero**
   `vllm_iteration(N)` ranges (kernel capture via `--kernel-trace` is unaffected —
   it recorded 23,711 rows). The fix is a **load-order** one: rocprofv3 registers
   its MARKER service by intercepting the `dlopen` of `librocprofiler-sdk-roctx.so.1`
   *inside the already-wrapped process*, so the SDK lib must be `dlopen`ed — from
   inside that process — **before** torch imports `libroctx64`. The capture driver
   now does exactly that: `build_capture_server_env` prepends a generated
   `sitecustomize.py` dir to the launched server's `PYTHONPATH`, and
   `sitecustomize` (imported automatically at interpreter startup, before any
   user/torch import) calls `alignment.profiler._roctx_preload.preload_sdk_roctx`,
   which does `ctypes.CDLL("librocprofiler-sdk-roctx.so.1", mode=RTLD_GLOBAL)`. That
   runtime load is intercepted by rocprofv3 (MARKER registers) and, being
   `RTLD_GLOBAL` and first, wins `roctx*` symbol resolution over torch's later
   `libroctx64`. The shim's `RoctracerBackend` then reuses that resident handle, so
   its `roctx*` calls resolve to the SDK lib. Do **not** force the roctx lib via
   `LD_PRELOAD` — that loads it before rocprofv3's interception is active, so the
   MARKER never registers (confirmed failure); the preload must happen in-process,
   which is why it is driven from `sitecustomize`, not `LD_PRELOAD`.

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
  launched server's environment itself, so the leading env assignment above is
  belt-and-suspenders — it makes the intent explicit and covers a manually invoked
  server. It further prepends a generated `sitecustomize.py` dir to the launched
  server's `PYTHONPATH` so the SDK roctx library is `RTLD_GLOBAL`-loaded before
  torch (the load-order fix this validates; see Precondition 2). A manually invoked
  server must reproduce **both**: set the env flag *and* put the preload on
  `PYTHONPATH` (e.g. `PYTHONPATH=$(python -c 'import alignment.profiler.rocprof_capture as c; print(c.write_roctx_sitecustomize_dir())'):$PYTHONPATH`).
  Do **not** set `VLLM_PLUGINS`;
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
per captured forward, each `vllm_iteration(N): forward`. The label is JSON in the
joined **event** row's `extdata` (`{"message": "..."}`), NOT in `r.name_id` — on a
real capture `name_id` resolves (via `rocpd_string`) only to the API op name
`roctxThreadRangeA`, identical for every range. This query mirrors what the reader
(`roctx_regions_from_rocpd`) does:

```sql
SELECT json_extract(e.extdata, '$.message') AS region_name, r.start, r.end
FROM rocpd_region r
JOIN rocpd_event e ON e.id = r.event_id
WHERE json_extract(e.extdata, '$.message') LIKE 'vllm\_iteration(%): %' ESCAPE '\'
ORDER BY r.start
LIMIT 25;
```

(Joining `rocpd_string` on `r.name_id` instead returns only `roctxThreadRangeA`;
that is the bug the reader fix corrected, so use the event-extdata query here.)

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

(a2) **Sentinel marker kernels exist (Option B)** — expect one `vibesim_sentinel`
dispatch per captured forward, its `grid_size_y` carrying `iteration + 1` in
dispatch (start) order. This is the boundary signal when query (a) is empty:

```sql
SELECT k.display_name, d.grid_size_y, d.grid_size_y - 1 AS iteration, d.start
FROM rocpd_kernel_dispatch d
JOIN rocpd_info_kernel_symbol k ON k.id = d.kernel_id
WHERE k.display_name LIKE '%vibesim_sentinel%' OR k.kernel_name LIKE '%vibesim_sentinel%'
ORDER BY d.start
LIMIT 25;
```

A non-empty result here with query (a) empty is the expected Option-B state on the
ROCm-7.2 stack: `validate_rocpd` reports `iteration_source:"sentinel"` and
`iteration_ranges > 0`, and `rocpd-parse` attributes the between-sentinel
dispatches to the recovered iterations.

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

## Deferred to the 8-GPU TP4/EP4 run

This 1-GPU check fixes capture plumbing that is confirmable on-host; two facts
remain GPU-only and must be confirmed on the next lease.

1. **Sentinel-kernel iteration boundaries — Option B landed, needs 1-GPU
   confirmation (Gap 2, TP1 is enough).** roctx MARKER recording is confirmed
   broken on this stack (SDK MARKER never registers in a torch-loaded process; two
   load-order attempts both yielded 0 regions). Option B — launching a
   `vibesim_sentinel` kernel per forward and reconstructing iterations from the
   reliable `--kernel-trace` dispatch rows — is now implemented on-host (shim
   emitter + reader + offline test), but sentinel **emission on a real GPU** is
   GPU-only and must be confirmed on the next lease.

   **Exact 1-GPU re-test** — serve `Qwen/Qwen2.5-0.5B-Instruct` on one MI300X with
   the capture command above (unchanged; the shim auto-launches the sentinel under
   the existing `VLLM_ROCTX_SCOPES_FOR_PROFILING=1` gate), drive a short completion
   workload (a few dozen decode iterations), then confirm **all four**:
   - **the sentinel kernels appear in the dispatch table** — query (a2) returns one
     `vibesim_sentinel` row per forward, each with `grid_size_y == iteration + 1` in
     start order (confirms Triton was used and the grid-y ordinal encoding survived
     the real launch — the one grid-units assumption this on-host work could not
     verify);
   - **`validate_rocpd` recovers iterations via sentinels** — the printed JSON line
     shows `iteration_source:"sentinel"`, `iteration_ranges>0`, `sentinel_kernels>0`
     and `ok:true` (roctx query (a) may be empty — that is expected on this stack);
   - **`rocpd-parse` emits the 3 artifacts** non-empty — `parsed.json`,
     `parsed.kernels.parquet`, `kernel_sequences.json` — with the
     `vibesim_sentinel` kernel **absent** from the parquet / sequences (markers are
     excluded from the compared set);
   - the `parsed.json` iteration indices match the sentinels' decoded
     `grid_size_y - 1` (and the warm-up kernels before the first sentinel are
     dropped).

   If query (a2) is empty, the sentinel did not fire: check the plugin is
   discovered in the serving venv (Precondition 1), the env gate is on, and Triton
   is importable in that venv (if Triton is missing the shim falls back to a
   non-reconstructable torch op — install Triton, which ships in the ROCm vLLM
   container). If (a2) has rows but `grid_size_y` does not encode `iteration + 1`
   (e.g. the real launch multiplied the y-grid by a non-unit workgroup-y), the
   reader still recovers correct *consecutive* iterations by sentinel ordinal;
   report the observed `grid_size_y` values so the decode offset can be corrected.

2. **Per-rank capture for TP>1 (Gap 3, needs TP4/EP4).** The driver now templates
   a per-rank rocprofv3 output name for `--tp-size > 1` (`-o <name>_rank%q{RANK}%`,
   `--rank-env` to pick the env var), so each of the N worker processes writes its
   own `<name>_rank<N>_results.db` and each is validated and parsed to a
   `parsed.rank<N>.json`. The TP4/EP4 run must pass:
   - all 4 per-rank databases are written and `locate_rocpd_per_rank` finds them;
   - each carries both `vllm_iteration(N)` roctx ranges (query (a)) and kernel
     dispatch rows (query (b)) — i.e. every worker, not just the launcher, was
     traced;
   - the exact env var the traced worker carries its rank in is confirmed (default
     `RANK`; `LOCAL_RANK` is the per-node alternative) so `%q{...}%` expands to a
     distinct value per worker rather than collapsing all workers onto one file;
   - whether `VLLM_ENABLE_V1_MULTIPROCESSING=0` (defaulted by the driver for the
     in-process TP1 engine) must be *overridden back on* for TP>1 so the workers
     spawn as traceable processes — set it in the capture env if so.
