---
name: impl-validate-kernel-cache
description: "Use when implementing the cache-fidelity validation leaf for an VibeSim L1 kernel Rust cost cache. Measure or compare INTERPOLATION / EXTRAPOLATION / QUANTIZATION fidelity against perf_api ground truth at off-grid, extrapolated, or bucket-boundary shapes. Covers any registered KernelSpec. NOT for profiling existing rows or batch-aggregation fidelity."
---

# Impl Validate Kernel Cache Fidelity

Measure how faithfully a kernel's **cache** reproduces true kernel time:
`Cache2DLinear` / `Cache1DLinear` interpolation at off-grid shapes,
direct/bucketed caches such as `Cache1DDirect` at bucket boundaries, or fitted
cache variants if the codebase provides one. This is the cache's
*interpolation/extrapolation/quantization* fidelity, distinct from:

- **`operate-profile-existing-kernel`** — queries/fills the profiled grid rows
  themselves. Use that to populate `profile.db`; use THIS to check cache
  behavior between rows, beyond rows, or at bucket boundaries.
- **batch-aggregation fidelity** (`agent-trace/attention_cache_fidelity.md`) — how
  an L2 op collapses a heterogeneous batch into one shape. Separate concern.

## Architecture (why it can't drift)

Two exposed, kernel-agnostic interfaces, fed ONE config from Python:

- **Rust `simulator kernel-query`** (stdin/stdout JSON) = the authoritative
  cache eval + grid metadata. We never reimplement Rust cache math in Python.
- **Python `perf_api.get_{kind}_times(...)`** = GPU ground truth.

Python holds the single config and sends it to both, so cache eval and truth
can't describe different kernels. `ratio = cache_eval / truth`; bar = within
±15%.

## The two `kernel-query` ops

Request on stdin, JSON on stdout (tracing goes to stderr — stdout stays pure):

```jsonc
// grid: cache axes + resolved config. NO bridge/GPU. Call first.
{"op":"grid","kind":"<KIND>","config":{...}}
//  → {"kind","describe_config","input_fields":[...],"grid_axes":[[...],...]}

// eval: best-of-N cache eval at query points. Builds the kernel (JIT-profiles
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

## The harness — `VibeSim/tools/cache-fidelity-analyzer/`

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
cd VibeSim
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
cd VibeSim
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

Treat command success as transport only; validation means the final ratios are
acceptable under the user-specified threshold, or the default ±15% bar. If the
error is too large, classify the failure before changing code:

- interior interpolation error → consider a denser sweep grid, a different
  built-in cache kind, or an existing fitted variant;
- extrapolation error → widen the profiled grid or narrow the supported domain;
- direct-cache quantization error → add bucket-boundary probes, then consider
  denser buckets or a non-direct cache;
- re-axis/domain error → verify probes are physical query points and revise the
  re-axis if the projection is wrong;
- noisy backend measurements → rerun/pre-warm before treating it as a cache
  design problem.

Ask the user before accepting a known inaccurate region, materially increasing
profiling cost with a denser grid, adding a new cache/fitting mechanism, or
changing the public query/input contract. After any remediation, rerun fidelity
and report the new final CSV/log summary.

Grid refinement must preserve the 500-feasible-coordinate design ceiling from
`orchestrator-wire-kernel-to-rust`. If a denser candidate would exceed it, do
not split the profiling work across calls. Remove non-independent axes or
unreachable shapes, improve the physical projection or cache policy, or narrow
the supported domain. Then validate the smaller grid against representative
off-grid physical demand.

Do not add an axis merely because an upstream artifact exposes it. First compare
the absolute timing effect at matched shapes, the fidelity gain, feasible-grid
growth, and DB migration cost. Aggregate coordinates are invalid when they hide
planner, page-lookup, or branch topology; in that case retain exact ragged input
for identity/provenance and define a source-backed cache projection. Keep
runtime allocation capacity distinct from checkpoint capability.

When workload derivation or the physical input contract changes, treat old rows
as belonging to the old contract and re-profile. Cache compatibility is never a
reason to preserve a model-specific legacy transformation that production no
longer uses.

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
  via `operate-profile-existing-kernel`.
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
