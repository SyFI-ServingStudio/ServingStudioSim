---
name: validate-kernel-cache
description: Use when asked to validate, measure, or compare the INTERPOLATION / EXTRAPOLATION fidelity of an MLSim L1 kernel's cost cache — i.e. how faithfully the Rust cache reproduces true kernel time at OFF-GRID shapes vs perf_api ground truth. Covers any registered KernelSpec (1D/2D/3D). NOT for profiling existing rows (that is profile-run-existing-kernel) or batch-aggregation fidelity.
---

# Validate Kernel Cache Fidelity

Measure how faithfully a kernel's **cache** (the Rust `Cache2DLinear` /
`Cache1DLinear` interpolation) reproduces true kernel time at **off-grid** shapes
— the points between/beyond profiled sweep-grid rows. This is the cache's
*interpolation/extrapolation* fidelity, distinct from:

- **`profile-run-existing-kernel`** — queries/fills the profiled grid rows
  themselves. Use that to populate `profile.db`; use THIS to check the interp
  between those rows.
- **batch-aggregation fidelity** (`agent-trace/attention_cache_fidelity.md`) — how
  an L2 op collapses a heterogeneous batch into one shape. Separate concern.

## Architecture (why it can't drift)

Two exposed, kernel-agnostic interfaces, fed ONE config from Python:

- **Rust `simulator kernel-query`** (stdin/stdout JSON) = the authoritative
  interp + grid metadata. We never reimplement bilinear in Python.
- **Python `perf_api.get_{kind}_times(...)`** = GPU ground truth.

Python holds the single config and sends it to both, so interp and truth can't
describe different kernels. `ratio = interp / truth`; bar = within ±15%.

## The two `kernel-query` ops

Request on stdin, JSON on stdout (tracing goes to stderr — stdout stays pure):

```jsonc
// grid: fitted axes + resolved config. NO bridge/GPU. Call first.
{"op":"grid","kind":"<KIND>","config":{...}}
//  → {"kind","describe_config","input_fields":[...],"grid_axes":[[...],...]}

// eval: best-of-N interp at query points. Builds the kernel (JIT-profiles
//       missing grid rows → needs GPU + the perf_api bridge).
{"op":"eval","kind":"<KIND>","config":{...},"query_points":[{<input_fields>:v},...]}
//  → {"kind","results":[{"input","time_ms","flops","bytes","energy_j","coverage"},...]}
```

`input_fields` (from `grid`) are the **query-point keys**, in `grid_axes` order.
`coverage` bits: `EXTRAPOLATED=1`, `JIT=2`, `NO_COVERAGE=4`.

## The config → spec contract (how ground truth is built generically)

- A kernel's full `KernelConfig` JSON = `{"backends":[...], "gpu_name":..., <dims>}`.
- **dims = config minus `backends`/`gpu_name`** = exactly the kernel's
  `KernelArgs` fields (e.g. `num_qo_heads`, `head_dim`, `dtype`, `n`, `k`).
- perf_api facade is **`get_{KIND}_times`** by convention. The per-point spec is
  `{**dims, **dict(zip(input_fields, coords))}`; best-of-N = `min` over backends.

This is all `cache_fidelity.py::ground_truth` does — no per-kernel Python.

## The harness — `main/tools/cache-fidelity-analyzer/`

- **`cache_fidelity.py`** — generic, kernel-agnostic core + CLI. Exposes
  `kernel_query`, `query_grid`, `eval_kind`, `ground_truth`, `generic_probes`,
  `stats`, and `run_fidelity(sim_bin, kind, config, *, probes, compare,
  truth_kind, log_dir)`.
- **`flashinfer_attn_fidelity.py`** — the worked example: a thin caller that
  builds the attention config from a HF `model_config` and supplies physical
  probes, then calls `run_fidelity`.

### Recipe A — simple kernel (cache grid == query space)

For kernels whose `grid_axes` ARE the query coordinates (`single_gemm`,
`rms_norm`, `flashinfer_attn_decode`, `flashinfer_attn_rect`): just run the
generic CLI with a config JSON; `generic_probes` places off-grid probes (per-axis
cell midpoints, cross-axis diagonal, beyond-grid extrapolation) automatically,
1D/2D/3D alike.

```bash
cd main
uv run cargo build --release -p simulator        # kernel-query lives in this binary
# cfg.json = full KernelConfig, e.g.
#   {"backends":["cutlass"],"gpu_name":"NVIDIA H200","n":4096,"k":4096,"dtype":"bf16"}
uv run python tools/cache-fidelity-analyzer/cache_fidelity.py \
    --kind single_gemm --config cfg.json \
    --log-dir logs/<YYYYMMDD_N_cache_fidelity>/fidelity
```

### Recipe B — domain / re-axis kernel (grid ≠ query space)

When the cache is built over re-axed coordinates (e.g. `flashinfer_attn_prefill`
caches over `(A=k+q/2, B=q)` but queries are physical `(prefix_len, append_len)`),
`grid_axes` are NOT physical, so generic probe placement would emit wrong shapes.
Write a thin caller (copy `flashinfer_attn_fidelity.py`):

1. Build the `config` dict (`backends`, `gpu_name`, dims).
2. Generate **physical** probes `[((coords...), "region"), ...]` in
   `input_fields` order, on a physical reference grid you declare (add
   domain regions: fresh diagonals, memory-bound corners, beyond-grid extrap).
3. `cf.run_fidelity(sim_bin, kind, config, probes=..., compare=..., log_dir=...)`.

The harness sends your physical `query_points` to `kernel-query` (Rust projects
them) and physical specs to perf_api — the re-axis stays invisible. Use
`compare="<other_kind>"` to A/B two cache variants on identical probes; use
`truth_kind=` if a variant profiles through another kind's table (`profile_kind`).

```bash
cd main
uv run python tools/cache-fidelity-analyzer/flashinfer_attn_fidelity.py \
    --model-config model/config/llama3_8b.json --gpu-name "NVIDIA H200" \
    --backends fa2,fa3 --log-dir logs/<YYYYMMDD_N_cache_fidelity>/fidelity
```

## Reading the report

Per-region lines: `med` (median ratio), `within=k/n` (±15%), `worst`
(by |log ratio|). Partition by region to localize error: extrapolation past the
grid max, the fresh/diagonal curvature, memory-bound small shapes, interior
cells. A CSV of per-probe rows lands in `--log-dir`. `ratio < 1` = the cache
under-estimates time (too optimistic); `> 1` = over-estimates.

## Gotchas (hard-won)

- **`gpu_name` is the exact `profile.db` key** — `"NVIDIA H200"`, not `"H200"`.
  A mismatch reads as a missing row (→ JIT or empty).
- **Run under `uv`** (pins the PyO3 3.12 interp; bare python/cargo links system
  3.9 and crashes). The launcher/harness build nothing extra — build the binary
  once with `uv run cargo build --release -p simulator`.
- **stdout must stay pure JSON.** The simulator routes tracing to stderr; if you
  add logging, keep it off stdout or `kernel-query` parsing breaks.
- **`eval` JIT-profiles missing grid rows on the GPU.** Pre-warm or expect a slow
  first build. Profiling many *fresh* shapes in one process can OOM the GPU
  (tensors not freed between calls) — profile a handful at a time, or pre-warm
  via `profile-run-existing-kernel`.
- **`grid_axes` may be a re-axis space**, not the query space (see Recipe B).
  `input_fields` always labels the *query* keys; if the cache re-axes internally,
  do NOT place probes off `grid_axes` — use a physical reference grid.
- **Dimensionality**: `generic_probes` handles 1D/2D/3D; a hand-written caller
  must emit coords matching `len(input_fields)`.

## Reference

- Worked example + the `(A,B)` re-axis story:
  `agent-trace/flashinfer_attn_prefill_reaxis.md`.
- Kernel-query implementation: `simulator/src/introspect/mod.rs`; the registry +
  `infeasible_mask` / blanket `CacheProbe`: `simulator/src/timing/kernels/engine.rs`.
